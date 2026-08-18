#include "duckdb/optimizer/expression_placement.hpp"

#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_recursive_cte.hpp"

namespace duckdb {

struct ExpressionPlacementCandidate {
	reference<LogicalProjection> consumer;
	reference<unique_ptr<Expression>> slot;
	reference<LogicalComparisonJoin> join;
	idx_t join_depth;
};

struct ExpressionPlacementGroup {
	reference<LogicalComparisonJoin> join;
	idx_t join_depth;
	vector<reference<ExpressionPlacementCandidate>> candidates;
};

struct ExpressionPlacementInput {
	reference<ExpressionPlacementCandidate> candidate;
	column_binding_map_t<idx_t> occurrences;
	vector<ColumnBinding> bindings;
};

struct ExpressionExposureStep {
	reference<LogicalOperator> child;
	reference<LogicalOperator> parent;
	idx_t child_index;
};

class ExpressionPlacementState {
public:
	ExpressionPlacementState(Binder &binder_p, ClientContext &context_p, LogicalOperator &root_p)
	    : binder(binder_p), context(context_p), root(root_p), data_flow(root_p), mutator(data_flow) {
	}

	void Optimize() {
		CollectOperatorDepths(root, 0);
		CollectCandidates(root, false);
		BuildGroups();
		std::sort(groups.begin(), groups.end(),
		          [](const ExpressionPlacementGroup &left, const ExpressionPlacementGroup &right) {
			          return left.join_depth > right.join_depth;
		          });
		for (auto &group : groups) {
			ApplyGroup(group);
		}
		root.ResolveOperatorTypes();
	}

private:
	static bool IsShortCircuit(const Expression &expression) {
		switch (expression.GetExpressionType()) {
		case ExpressionType::CASE_EXPR:
		case ExpressionType::OPERATOR_COALESCE:
		case ExpressionType::CONJUNCTION_AND:
		case ExpressionType::CONJUNCTION_OR:
			return true;
		default:
			return false;
		}
	}

	static bool IsFixedSize(const LogicalType &type, idx_t &width) {
		if (type.IsNested() || type.InternalType() == PhysicalType::VARCHAR ||
		    type.InternalType() == PhysicalType::INVALID) {
			return false;
		}
		width = GetTypeIdSize(type.InternalType());
		return width > 0;
	}

	void CollectOperatorDepths(LogicalOperator &op, idx_t depth) {
		operator_depths.emplace_back(op, depth);
		for (auto &child : op.children) {
			CollectOperatorDepths(*child, depth + 1);
		}
	}

	idx_t GetOperatorDepth(LogicalOperator &op) const {
		for (auto &entry : operator_depths) {
			if (&entry.first.get() == &op) {
				return entry.second;
			}
		}
		throw InternalException("Expression placement operator depth not found");
	}

	void CollectCandidates(LogicalOperator &op, bool inside_recursive_cte) {
		const bool recursive = inside_recursive_cte || op.type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE;
		if (!recursive && op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &projection = op.Cast<LogicalProjection>();
			for (auto &expression : projection.expressions) {
				CollectExpressionCandidates(expression, projection);
			}
		}
		for (auto &child : op.children) {
			CollectCandidates(*child, recursive);
		}
	}

	void CollectExpressionCandidates(unique_ptr<Expression> &slot, LogicalProjection &consumer) {
		auto boundary = FindBoundary(*slot, consumer);
		if (IsShortCircuit(*slot)) {
			if (boundary) {
				AddCandidate(consumer, slot, *boundary);
			}
			return;
		}

		vector<idx_t> candidate_start;
		ExpressionIterator::EnumerateChildren(*slot, [&](unique_ptr<Expression> &child) {
			candidate_start.push_back(candidates.size());
			CollectExpressionCandidates(child, consumer);
		});
		if (!boundary) {
			return;
		}

		for (idx_t child_idx = candidate_start.size(); child_idx > 0; child_idx--) {
			const idx_t begin = candidate_start[child_idx - 1];
			const idx_t end = child_idx == candidate_start.size() ? candidates.size() : candidate_start[child_idx];
			for (idx_t candidate_idx = end; candidate_idx > begin; candidate_idx--) {
				if (&candidates[candidate_idx - 1].join.get() == boundary.get()) {
					candidates.erase(candidates.begin() + NumericCast<int64_t>(candidate_idx - 1));
				}
			}
		}
		AddCandidate(consumer, slot, *boundary);
	}

	optional_ptr<LogicalComparisonJoin> FindBoundary(Expression &expression, LogicalProjection &consumer) {
		if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF || expression.IsAggregate() ||
		    expression.IsWindow() || expression.HasSubquery() || !expression.IsConsistent() || expression.CanThrow()) {
			return nullptr;
		}

		vector<reference<LogicalOperator>> sources;
		bool valid = true;
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    expression, [&](const BoundColumnRefExpression &column_ref) {
			    if (column_ref.Depth() != 0) {
				    valid = false;
				    return;
			    }
			    auto source = data_flow.ResolveSource(column_ref.Binding(), column_ref.Depth(), consumer);
			    if (source.status != LogicalPlanDataFlowStatus::SUCCESS || !source.op) {
				    valid = false;
				    return;
			    }
			    bool seen = false;
			    for (auto existing : sources) {
				    if (&existing.get() == source.op.get()) {
					    seen = true;
					    break;
				    }
			    }
			    if (!seen) {
				    sources.push_back(*source.op);
			    }
		    });
		if (!valid || sources.empty()) {
			return nullptr;
		}

		auto source_lca = optional_ptr<LogicalOperator>(sources[0]);
		for (idx_t source_idx = 1; source_idx < sources.size(); source_idx++) {
			auto lca = data_flow.LowestCommonAncestor(*source_lca, sources[source_idx]);
			if (lca.status != LogicalPlanDataFlowStatus::SUCCESS || !lca.op) {
				return nullptr;
			}
			source_lca = lca.op;
		}

		auto path = data_flow.GetPathSummary(consumer, *source_lca);
		if (path.status != LogicalPlanDataFlowStatus::SUCCESS ||
		    path.summary.Has(LogicalPlanPathProperty::OPAQUE_BOUNDARY) ||
		    path.summary.Has(LogicalPlanPathProperty::CTE_BOUNDARY) ||
		    path.summary.Has(LogicalPlanPathProperty::SIDE_EFFECT_BOUNDARY) ||
		    path.summary.Has(LogicalPlanPathProperty::NULL_EXTENDING)) {
			return nullptr;
		}

		LogicalPlanPathSummary build_boundary;
		build_boundary.Add(LogicalPlanPathProperty::HASH_JOIN_BUILD_BOUNDARY);
		auto edge = data_flow.FindLastPathEdge(consumer, *source_lca, build_boundary);
		if (edge.status != LogicalPlanDataFlowStatus::SUCCESS || !edge.parent || edge.child_index != 1 ||
		    edge.parent->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			return nullptr;
		}
		auto &join = edge.parent->Cast<LogicalComparisonJoin>();
		idx_t range_count = 0;
		join.HasEquality(range_count);
		if (Settings::Get<PreferRangeJoinsSetting>(context) && range_count >= 2) {
			return nullptr;
		}
		return join;
	}

	void AddCandidate(LogicalProjection &consumer, unique_ptr<Expression> &slot, LogicalComparisonJoin &join) {
		candidates.push_back({consumer, slot, join, GetOperatorDepth(join)});
	}

	void BuildGroups() {
		for (auto &candidate : candidates) {
			optional_ptr<ExpressionPlacementGroup> group;
			for (auto &existing : groups) {
				if (&existing.join.get() == &candidate.join.get()) {
					group = existing;
					break;
				}
			}
			if (!group) {
				groups.push_back({candidate.join, candidate.join_depth, {}});
				group = groups.back();
			}
			group->candidates.push_back(candidate);
		}
	}

	ExpressionPlacementInput GetInput(ExpressionPlacementCandidate &candidate) {
		ExpressionPlacementInput result {candidate, {}, {}};
		ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
		    *candidate.slot.get(), [&](const BoundColumnRefExpression &column_ref) {
			    auto entry = result.occurrences.find(column_ref.Binding());
			    if (entry == result.occurrences.end()) {
				    result.occurrences[column_ref.Binding()] = 1;
				    result.bindings.push_back(column_ref.Binding());
			    } else {
				    entry->second++;
			    }
		    });
		return result;
	}

	static bool SharesBinding(const ExpressionPlacementInput &left, const ExpressionPlacementInput &right) {
		for (auto &binding : left.bindings) {
			if (right.occurrences.find(binding) != right.occurrences.end()) {
				return true;
			}
		}
		return false;
	}

	bool BindingIsProjected(const LogicalComparisonJoin &join, idx_t child_column) const {
		if (join.right_projection_map.empty()) {
			return true;
		}
		return std::find(join.right_projection_map.begin(), join.right_projection_map.end(),
		                 ProjectionIndex(child_column)) != join.right_projection_map.end();
	}

	bool IsRemovableUse(const LogicalPlanBindingUse &use, LogicalComparisonJoin &join, idx_t &use_count) {
		if (use.depth != 0) {
			return false;
		}
		if (&use.consumer.get() == &join) {
			return true;
		}
		auto above_join = data_flow.IsFlowAncestor(use.consumer, join);
		if (above_join.status != LogicalPlanDataFlowStatus::SUCCESS) {
			return false;
		}
		if (above_join.value) {
			use_count++;
		}
		return true;
	}

	bool ComponentIsProfitable(LogicalComparisonJoin &join, const vector<ExpressionPlacementInput> &inputs,
	                           const vector<idx_t> &component) {
		auto &build = *join.children[1];
		const auto build_cardinality =
		    build.has_estimated_cardinality ? build.estimated_cardinality : build.EstimateCardinality(context);
		column_binding_map_t<idx_t> selected_occurrences;
		vector<ColumnBinding> operands;
		for (auto input_idx : component) {
			auto &input = inputs[input_idx];
			auto &consumer = input.candidate.get().consumer.get();
			const auto consumer_cardinality = consumer.has_estimated_cardinality
			                                      ? consumer.estimated_cardinality
			                                      : consumer.EstimateCardinality(context);
			if (build_cardinality > consumer_cardinality) {
				return false;
			}
			for (auto &binding : input.bindings) {
				if (selected_occurrences.find(binding) == selected_occurrences.end()) {
					selected_occurrences[binding] = input.occurrences.at(binding);
					operands.push_back(binding);
				} else {
					selected_occurrences[binding] += input.occurrences.at(binding);
				}
			}
		}

		build.ResolveOperatorTypes();
		auto build_bindings = build.GetColumnBindings();
		idx_t removed_width = 0;
		for (auto &binding : operands) {
			auto binding_entry = std::find(build_bindings.begin(), build_bindings.end(), binding);
			if (binding_entry == build_bindings.end()) {
				return false;
			}
			const idx_t child_column = NumericCast<idx_t>(binding_entry - build_bindings.begin());
			if (!BindingIsProjected(join, child_column)) {
				return false;
			}
			idx_t binding_width;
			if (!IsFixedSize(build.types[child_column], binding_width)) {
				return false;
			}
			idx_t uses_above_join = 0;
			for (auto &use : data_flow.GetBindingUses()) {
				if (use.binding == binding && !IsRemovableUse(use, join, uses_above_join)) {
					return false;
				}
			}
			if (uses_above_join != selected_occurrences[binding]) {
				return false;
			}
			removed_width += binding_width;
		}

		idx_t replacement_width = 0;
		vector<reference<Expression>> replacements;
		for (auto input_idx : component) {
			auto &expression = *inputs[input_idx].candidate.get().slot.get();
			bool duplicate = false;
			for (auto existing : replacements) {
				if (Expression::Equals(expression, existing.get())) {
					duplicate = true;
					break;
				}
			}
			if (duplicate) {
				continue;
			}
			idx_t result_width;
			if (!IsFixedSize(expression.GetReturnType(), result_width)) {
				return false;
			}
			replacement_width += result_width;
			replacements.push_back(expression);
		}
		return replacement_width < removed_width;
	}

	vector<reference<ExpressionPlacementCandidate>> SelectProfitableCandidates(ExpressionPlacementGroup &group) {
		vector<ExpressionPlacementInput> inputs;
		inputs.reserve(group.candidates.size());
		for (auto candidate : group.candidates) {
			inputs.push_back(GetInput(candidate));
		}

		vector<bool> assigned(inputs.size(), false);
		vector<reference<ExpressionPlacementCandidate>> result;
		for (idx_t input_idx = 0; input_idx < inputs.size(); input_idx++) {
			if (assigned[input_idx]) {
				continue;
			}
			vector<idx_t> component {input_idx};
			assigned[input_idx] = true;
			for (idx_t component_idx = 0; component_idx < component.size(); component_idx++) {
				for (idx_t other_idx = 0; other_idx < inputs.size(); other_idx++) {
					if (!assigned[other_idx] && SharesBinding(inputs[component[component_idx]], inputs[other_idx])) {
						assigned[other_idx] = true;
						component.push_back(other_idx);
					}
				}
			}
			if (!ComponentIsProfitable(group.join, inputs, component)) {
				continue;
			}
			for (auto selected_idx : component) {
				result.push_back(inputs[selected_idx].candidate);
			}
		}
		return result;
	}

	vector<ExpressionExposureStep> GetExposurePath(LogicalOperator &source, LogicalProjection &consumer) {
		vector<ExpressionExposureStep> result;
		auto current = optional_ptr<LogicalOperator>(source);
		while (current.get() != &consumer) {
			auto parent = data_flow.GetFlowParent(*current);
			if (parent.status != LogicalPlanDataFlowStatus::SUCCESS || !parent.parent) {
				throw InternalException("Cannot expose placed expression binding to its consumer");
			}
			if (parent.parent.get() == &consumer) {
				return result;
			}
			result.push_back({*current, *parent.parent, parent.child_index});
			current = parent.parent;
		}
		return result;
	}

	void ExposeBinding(const vector<ExpressionExposureStep> &path, const ColumnBinding &binding) {
		for (auto &step : path) {
			auto current_bindings = step.child.get().GetColumnBindings();
			auto binding_entry = std::find(current_bindings.begin(), current_bindings.end(), binding);
			if (binding_entry == current_bindings.end()) {
				throw InternalException("Placed expression binding is not available below %s",
				                        EnumUtil::ToString(step.parent.get().type));
			}
			auto projection_map = LogicalOperatorVisitor::GetProjectionMap(step.parent, step.child_index);
			if (projection_map && !projection_map->empty()) {
				ProjectionIndex child_column(NumericCast<idx_t>(binding_entry - current_bindings.begin()));
				if (std::find(projection_map->begin(), projection_map->end(), child_column) == projection_map->end()) {
					projection_map->push_back(child_column);
					mutator.RefreshOperator(step.parent);
				}
			}
		}
	}

	void ApplyGroup(ExpressionPlacementGroup &group) {
		auto selected = SelectProfitableCandidates(group);
		if (selected.empty()) {
			return;
		}
		auto &join = group.join.get();
		auto &build_slot = join.children[1];
		build_slot->ResolveOperatorTypes();
		auto old_bindings = build_slot->GetColumnBindings();
		const auto old_types = build_slot->types;
		const idx_t pass_through_count = old_bindings.size();

		vector<unique_ptr<Expression>> expressions;
		expressions.reserve(pass_through_count + selected.size());
		BindingReplacementGraph replacements;
		const auto projection_index = binder.GenerateTableIndex();
		for (idx_t column_idx = 0; column_idx < pass_through_count; column_idx++) {
			expressions.push_back(make_uniq<BoundColumnRefExpression>(old_types[column_idx], old_bindings[column_idx]));
			replacements.Add(old_bindings[column_idx], ColumnBinding(projection_index, ProjectionIndex(column_idx)));
		}

		vector<reference<Expression>> unique_expressions;
		vector<idx_t> result_indexes;
		result_indexes.reserve(selected.size());
		for (auto candidate : selected) {
			auto &expression = *candidate.get().slot.get();
			idx_t result_index = DConstants::INVALID_INDEX;
			for (idx_t expression_idx = 0; expression_idx < unique_expressions.size(); expression_idx++) {
				if (Expression::Equals(expression, unique_expressions[expression_idx].get())) {
					result_index = expression_idx;
					break;
				}
			}
			if (result_index == DConstants::INVALID_INDEX) {
				result_index = unique_expressions.size();
				unique_expressions.push_back(expression);
				expressions.push_back(expression.Copy());
			}
			result_indexes.push_back(result_index);
		}

		auto projection = make_uniq<LogicalProjection>(projection_index, std::move(expressions));
		projection->SetEstimatedCardinality(build_slot->has_estimated_cardinality
		                                        ? build_slot->estimated_cardinality
		                                        : build_slot->EstimateCardinality(context));
		ColumnBindingRewrite::InsertUnaryAndRewriteBindings(data_flow, mutator, build_slot, std::move(projection),
		                                                    replacements);
		vector<vector<ExpressionExposureStep>> exposure_paths;
		exposure_paths.reserve(selected.size());
		for (auto candidate : selected) {
			exposure_paths.push_back(GetExposurePath(join, candidate.get().consumer));
		}

		vector<reference<LogicalProjection>> changed_consumers;
		{
			auto mutation = mutator.BeginMutation();
			for (idx_t expression_idx = 0; expression_idx < unique_expressions.size(); expression_idx++) {
				join.right_projection_map.emplace_back(pass_through_count + expression_idx);
			}
			mutator.RefreshOperator(join);
			for (idx_t candidate_idx = 0; candidate_idx < selected.size(); candidate_idx++) {
				ExposeBinding(exposure_paths[candidate_idx],
				              ColumnBinding(projection_index,
				                            ProjectionIndex(pass_through_count + result_indexes[candidate_idx])));
			}
			for (idx_t candidate_idx = 0; candidate_idx < selected.size(); candidate_idx++) {
				auto &candidate = selected[candidate_idx].get();
				auto &slot = candidate.slot.get();
				auto replacement = make_uniq<BoundColumnRefExpression>(
				    slot->GetAlias(), slot->GetReturnType(),
				    ColumnBinding(projection_index,
				                  ProjectionIndex(pass_through_count + result_indexes[candidate_idx])));
				replacement->SetQueryLocation(slot->GetQueryLocation());
				slot = std::move(replacement);
				bool seen = false;
				for (auto consumer : changed_consumers) {
					if (&consumer.get() == &candidate.consumer.get()) {
						seen = true;
						break;
					}
				}
				if (!seen) {
					changed_consumers.push_back(candidate.consumer);
				}
			}
			for (auto consumer : changed_consumers) {
				mutator.RefreshOperator(consumer);
			}
		}
	}

private:
	Binder &binder;
	ClientContext &context;
	LogicalOperator &root;
	LogicalPlanDataFlow data_flow;
	LogicalPlanDataFlowMutator mutator;
	vector<pair<reference<LogicalOperator>, idx_t>> operator_depths;
	vector<ExpressionPlacementCandidate> candidates;
	vector<ExpressionPlacementGroup> groups;
};

ExpressionPlacementOptimizer::ExpressionPlacementOptimizer(Binder &binder_p, ClientContext &context_p)
    : binder(binder_p), context(context_p) {
}

void ExpressionPlacementOptimizer::Optimize(LogicalOperator &root) {
	root.ResolveOperatorTypes();
	ExpressionPlacementState state(binder, context, root);
	state.Optimize();
}

} // namespace duckdb
