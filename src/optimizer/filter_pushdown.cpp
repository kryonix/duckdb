#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_subquery_expression.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

static bool ExpressionsBecomeFilters(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		return true;
	}
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return op.Cast<LogicalJoin>().join_type == JoinType::INNER;
	default:
		return false;
	}
}

static bool IsFilterPushdownTransparent(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		return true;
	}
	return op.type == LogicalOperatorType::LOGICAL_DISTINCT && !op.Cast<LogicalDistinct>().order_by;
}

void FilterPushdown::CheckMarkToSemi(LogicalOperator &op, const unordered_set<TableIndex> &table_bindings) {
	auto referenced_bindings = table_bindings;
	if (!ExpressionsBecomeFilters(op)) {
		LogicalOperatorVisitor::EnumerateExpressions(
		    static_cast<const LogicalOperator &>(op), [&](const unique_ptr<Expression> *expression) {
			    ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
			        **expression, [&](const BoundColumnRefExpression &column_ref) {
				        referenced_bindings.insert(column_ref.Binding().table_index);
			        });
		    });
	}

	switch (op.type) {
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type == JoinType::MARK) {
			// Duplicate-eliminated correlated subqueries must keep MARK semantics; converting to SEMI can drop
			// correlation for nested RHS shapes (issue #22267).
			join.convert_mark_to_semi = false;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (join.join_type != JoinType::MARK) {
			break;
		}
		// if an operator above the mark join includes the mark join index,
		// then the mark join cannot be converted to a semi join
		if (referenced_bindings.find(join.mark_index) != referenced_bindings.end()) {
			join.convert_mark_to_semi = false;
		}
		break;
	}
	default:
		break;
	}

	// Recurse into each child with the bindings referenced on this path.
	for (auto &child : op.children) {
		CheckMarkToSemi(*child, referenced_bindings);
	}
}

FilterPushdown::FilterPushdown(Optimizer &optimizer, bool convert_mark_joins, ProjectionMode projection_mode)
    : optimizer(optimizer), combiner(optimizer.context), convert_mark_joins(convert_mark_joins),
      projection_mode(projection_mode) {
}

unique_ptr<LogicalOperator> FilterPushdown::Rewrite(unique_ptr<LogicalOperator> op) {
	RewriteContext context(*op);
	Rewrite(op, context);
	D_ASSERT(context.data_flow.Verify());
	return op;
}

void FilterPushdown::Rewrite(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(!combiner.HasFilters());
#ifdef DEBUG
	VerifyIndexedFilterTargets(*op, context);
#endif
	if (TryRewriteAtIndexedTarget(op, context)) {
		return;
	}
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		return PushdownAggregate(op, context);
	case LogicalOperatorType::LOGICAL_FILTER:
		return PushdownFilter(op, context);
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		return PushdownCrossProduct(op, context);
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		return PushdownJoin(op, context);
	case LogicalOperatorType::LOGICAL_PROJECTION:
		return PushdownProjection(op, context);
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_UNION:
		return PushdownSetOperation(op, context);
	case LogicalOperatorType::LOGICAL_DISTINCT:
		return PushdownDistinct(op, context);
	case LogicalOperatorType::LOGICAL_ORDER_BY:
		// we can just push directly through these operations without any rewriting
		Rewrite(op->children[0], context);
		return;
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: {
		// we can't push filters into the materialized CTE (LHS), but we do want to recurse into it
		FilterPushdown pushdown(optimizer, convert_mark_joins, projection_mode);
		pushdown.Rewrite(op->children[0], context);
		// we can push filters into the rest of the query plan (RHS)
		Rewrite(op->children[1], context);
		return;
	}
	case LogicalOperatorType::LOGICAL_GET:
		return PushdownGet(op, context);
	case LogicalOperatorType::LOGICAL_LIMIT:
		return PushdownLimit(op, context);
	case LogicalOperatorType::LOGICAL_WINDOW:
		return PushdownWindow(op, context);
	case LogicalOperatorType::LOGICAL_UNNEST:
		return PushdownUnnest(op, context);
	default:
		return FinishPushdown(op, context);
	}
}

LogicalPlanDataFlowOperatorResult FilterPushdown::GetFilterConvergence(const Filter &filter, LogicalOperator &consumer,
                                                                       RewriteContext &context) const {
	bool has_subquery = false;
	ExpressionIterator::VisitExpression<BoundSubqueryExpression>(
	    *filter.filter, [&](const BoundSubqueryExpression &) { has_subquery = true; });
	if (has_subquery) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
	}
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::SUCCESS;
	optional_ptr<LogicalOperator> convergence;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
	    *filter.filter, [&](const BoundColumnRefExpression &column_ref) {
		    if (status != LogicalPlanDataFlowStatus::SUCCESS) {
			    return;
		    }
		    auto source = context.data_flow.ResolveSource(column_ref.Binding(), column_ref.Depth(), consumer);
		    if (source.status != LogicalPlanDataFlowStatus::SUCCESS) {
			    status = source.status;
			    return;
		    }
		    if (!source.op) {
			    status = LogicalPlanDataFlowStatus::UNSUPPORTED;
			    return;
		    }
		    if (!convergence) {
			    convergence = source.op;
			    return;
		    }
		    auto lca = context.data_flow.LowestCommonAncestor(*convergence, *source.op);
		    if (lca.status != LogicalPlanDataFlowStatus::SUCCESS) {
			    status = lca.status;
			    return;
		    }
		    if (!lca.op) {
			    status = LogicalPlanDataFlowStatus::UNSUPPORTED;
			    return;
		    }
		    convergence = lca.op;
	    });
	if (status != LogicalPlanDataFlowStatus::SUCCESS) {
		return {status, nullptr};
	}
	if (!convergence) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
	}
	auto is_ancestor = context.data_flow.IsFlowAncestor(consumer, *convergence);
	if (is_ancestor.status != LogicalPlanDataFlowStatus::SUCCESS) {
		return {is_ancestor.status, nullptr};
	}
	if (!is_ancestor.value) {
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, convergence};
}

LogicalPlanDataFlowOperatorResult
FilterPushdown::GetIndexedFilterTarget(const Filter &filter, LogicalOperator &consumer, RewriteContext &context) const {
	auto convergence = GetFilterConvergence(filter, consumer, context);
	if (convergence.status != LogicalPlanDataFlowStatus::SUCCESS || !convergence.op) {
		return convergence;
	}
	LogicalPlanPathSummary boundary;
	boundary.Add(LogicalPlanPathProperty::FILTER_PUSHDOWN_BOUNDARY);
	return context.data_flow.FindFirstPathOperator(consumer, *convergence.op, boundary);
}

bool FilterPushdown::TryRewriteAtIndexedTarget(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	if (filters.empty() || !IsFilterPushdownTransparent(*op) || op->children.size() != 1 || !op->children[0] ||
	    !IsFilterPushdownTransparent(*op->children[0])) {
		return false;
	}
	optional_ptr<LogicalOperator> target;
	for (auto &filter : filters) {
		auto filter_target = GetIndexedFilterTarget(*filter, *op, context);
		if (filter_target.status != LogicalPlanDataFlowStatus::SUCCESS || !filter_target.op) {
			return false;
		}
		if (!target || target == filter_target.op) {
			target = filter_target.op;
			continue;
		}
		auto target_is_ancestor = context.data_flow.IsFlowAncestor(*target, *filter_target.op);
		if (target_is_ancestor.status != LogicalPlanDataFlowStatus::SUCCESS) {
			return false;
		}
		if (target_is_ancestor.value) {
			continue;
		}
		auto filter_target_is_ancestor = context.data_flow.IsFlowAncestor(*filter_target.op, *target);
		if (filter_target_is_ancestor.status != LogicalPlanDataFlowStatus::SUCCESS ||
		    !filter_target_is_ancestor.value) {
			return false;
		}
		target = filter_target.op;
	}
	if (!target || target == op) {
		return false;
	}
	auto owner = context.data_flow.GetOwnershipParent(*target);
	if (owner.status != LogicalPlanDataFlowStatus::SUCCESS || !owner.parent ||
	    owner.child_index == DConstants::INVALID_INDEX || owner.child_index >= owner.parent->children.size() ||
	    target != owner.parent->children[owner.child_index]) {
		return false;
	}
#ifdef DEBUG
	auto skipped = owner.parent;
	while (skipped) {
		D_ASSERT(!IsLegacyFilterPushdownBoundary(*skipped));
		if (skipped == op) {
			break;
		}
		auto skipped_owner = context.data_flow.GetOwnershipParent(*skipped);
		D_ASSERT(skipped_owner.status == LogicalPlanDataFlowStatus::SUCCESS && skipped_owner.parent);
		skipped = skipped_owner.parent;
	}
	D_ASSERT(skipped && skipped == op);
#endif
	Rewrite(owner.parent->children[owner.child_index], context);
	return true;
}

#ifdef DEBUG
bool FilterPushdown::IsLegacyFilterPushdownBoundary(const LogicalOperator &op) {
	return !IsFilterPushdownTransparent(op);
}

LogicalPlanDataFlowOperatorResult FilterPushdown::GetLegacyFilterTarget(LogicalOperator &consumer,
                                                                        LogicalOperator &convergence,
                                                                        RewriteContext &context) const {
	optional_ptr<LogicalOperator> target;
	auto current = optional_ptr<LogicalOperator>(convergence);
	while (current) {
		if (IsLegacyFilterPushdownBoundary(*current)) {
			target = current;
		}
		if (&*current == &consumer) {
			return target ? LogicalPlanDataFlowOperatorResult {LogicalPlanDataFlowStatus::SUCCESS, target}
			              : LogicalPlanDataFlowOperatorResult {LogicalPlanDataFlowStatus::PATH_PROPERTY_NOT_FOUND,
			                                                   nullptr};
		}
		auto parent = context.data_flow.GetFlowParent(*current);
		if (parent.status != LogicalPlanDataFlowStatus::SUCCESS) {
			return {parent.status, nullptr};
		}
		current = parent.parent;
	}
	return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, nullptr};
}

void FilterPushdown::VerifyIndexedFilterTargets(LogicalOperator &consumer, RewriteContext &context) const {
	for (auto &filter : filters) {
		auto indexed = GetIndexedFilterTarget(*filter, consumer, context);
		if (indexed.status != LogicalPlanDataFlowStatus::SUCCESS) {
			continue;
		}
		auto convergence = GetFilterConvergence(*filter, consumer, context);
		D_ASSERT(convergence.status == LogicalPlanDataFlowStatus::SUCCESS && convergence.op);
		auto legacy = GetLegacyFilterTarget(consumer, *convergence.op, context);
		D_ASSERT(legacy.status == LogicalPlanDataFlowStatus::SUCCESS && legacy.op);
		D_ASSERT(indexed.op == legacy.op);
	}
}
#endif

ClientContext &FilterPushdown::GetContext() {
	return optimizer.GetContext();
}

FilterPushdown::JoinBindingState::JoinBindingState(LogicalOperator &join_p, LogicalPlanDataFlow &data_flow_p)
    : join(join_p), data_flow(data_flow_p) {
}

LogicalOperator &FilterPushdown::JoinBindingState::Join() {
	return join.get();
}

void FilterPushdown::JoinBindingState::AddRightBinding(TableIndex table_index) {
	extra_right_bindings.insert(table_index);
#ifdef DEBUG
	if (right_initialized) {
		right_bindings.insert(table_index);
	}
#endif
}

bool FilterPushdown::JoinBindingState::IsExtraRightBinding(TableIndex table_index) const {
	return extra_right_bindings.find(table_index) != extra_right_bindings.end();
}

LogicalPlanDataFlowOperatorResult FilterPushdown::JoinBindingState::ResolveSource(const ColumnBinding &binding,
                                                                                  idx_t depth) {
	if (depth != 0) {
		return data_flow.get().ResolveInputSource(binding, depth, join.get());
	}
	auto entry = indexed_sources.find(binding);
	if (entry != indexed_sources.end()) {
		return entry->second;
	}
	auto result = data_flow.get().ResolveInputSource(binding, depth, join.get());
	if (result.status == LogicalPlanDataFlowStatus::SUCCESS && result.op && &*result.op != &join.get()) {
		indexed_sources.emplace(binding, result);
	}
	return result;
}

#ifdef DEBUG
void FilterPushdown::JoinBindingState::Initialize() {
	InitializeLeft();
	InitializeRight();
}

void FilterPushdown::JoinBindingState::InitializeLeft() {
	if (left_initialized) {
		return;
	}
	LogicalJoin::GetTableReferences(*join.get().children[0], left_bindings);
	left_initialized = true;
}

void FilterPushdown::JoinBindingState::InitializeRight() {
	if (right_initialized) {
		return;
	}
	LogicalJoin::GetTableReferences(*join.get().children[1], right_bindings);
	for (auto table_index : extra_right_bindings) {
		right_bindings.insert(table_index);
	}
	right_initialized = true;
}

JoinSide FilterPushdown::JoinBindingState::GetLegacyJoinSide(const Filter &filter) {
	Initialize();
	return JoinSide::GetJoinSide(filter.bindings, left_bindings, right_bindings);
}

JoinSide FilterPushdown::JoinBindingState::GetLegacyJoinSide(const Expression &expression) {
	Initialize();
	return JoinSide::GetJoinSide(expression, left_bindings, right_bindings);
}
#endif

FilterPushdown::IndexedJoinSourceResult FilterPushdown::ClassifyJoinSources(const Expression &expression,
                                                                            JoinBindingState &binding_state) const {
	IndexedJoinSourceResult result;
	auto &join = binding_state.Join();
	if (join.children.size() != 2) {
		result.status = LogicalPlanDataFlowStatus::UNSUPPORTED;
		return result;
	}
	auto classify_binding = [&](const ColumnBinding &binding) {
		if (binding_state.IsExtraRightBinding(binding.table_index)) {
			result.right = true;
			return;
		}
		auto source = binding_state.ResolveSource(binding, 0);
		if (source.status != LogicalPlanDataFlowStatus::SUCCESS) {
			result.status = source.status;
			return;
		}
		if (!source.op) {
			result.status = LogicalPlanDataFlowStatus::UNSUPPORTED;
			return;
		}
		if (&*source.op == &join) {
			result.join = true;
			return;
		}
		if (source.source_child_index > 1) {
			result.status = LogicalPlanDataFlowStatus::UNSUPPORTED;
			return;
		}
		result.left = result.left || source.source_child_index == 0;
		result.right = result.right || source.source_child_index == 1;
	};
	std::function<void(const Expression &)> classify_expression;
	classify_expression = [&](const Expression &current) {
		if (result.status != LogicalPlanDataFlowStatus::SUCCESS) {
			return;
		}
		if (current.GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
			auto &column_ref = current.Cast<BoundColumnRefExpression>();
			if (column_ref.Depth() == 0) {
				classify_binding(column_ref.Binding());
			}
			return;
		}
		if (current.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY) {
			auto &subquery = current.Cast<BoundSubqueryExpression>();
			for (auto &child : subquery.GetChildren()) {
				classify_expression(*child);
			}
			for (auto &correlated : subquery.GetBinder()->correlated_columns) {
				if (result.status != LogicalPlanDataFlowStatus::SUCCESS) {
					return;
				}
				if (correlated.depth > 1) {
					result.left = true;
					result.right = true;
				} else {
					classify_binding(correlated.binding);
				}
			}
			return;
		}
		ExpressionIterator::EnumerateChildren(current, [&](const Expression &child) { classify_expression(child); });
	};
	classify_expression(expression);
	return result;
}

FilterPushdown::IndexedJoinSideResult FilterPushdown::GetIndexedJoinSide(const IndexedJoinSourceResult &sources,
                                                                         JoinDecisionPolicy policy) {
	if (sources.status != LogicalPlanDataFlowStatus::SUCCESS) {
		return {sources.status, JoinSide::NONE};
	}
	if (sources.join && policy != JoinDecisionPolicy::MARK_JOIN) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, JoinSide::NONE};
	}
	const bool has_left = sources.left;
	const bool has_right = sources.right || sources.join;
	if (has_left && has_right) {
		return {LogicalPlanDataFlowStatus::SUCCESS, JoinSide::BOTH};
	}
	if (has_left) {
		return {LogicalPlanDataFlowStatus::SUCCESS, JoinSide::LEFT};
	}
	if (has_right) {
		return {LogicalPlanDataFlowStatus::SUCCESS, JoinSide::RIGHT};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, JoinSide::NONE};
}

#ifdef DEBUG
FilterPushdown::JoinFilterDecision FilterPushdown::GetJoinFilterDecision(JoinSide side, JoinDecisionPolicy policy) {
	switch (policy) {
	case JoinDecisionPolicy::CROSS_PRODUCT:
		if (side == JoinSide::LEFT) {
			return JoinFilterDecision::PUSH_LEFT;
		}
		if (side == JoinSide::RIGHT) {
			return JoinFilterDecision::PUSH_RIGHT;
		}
		return JoinFilterDecision::CREATE_JOIN_CONDITION;
	case JoinDecisionPolicy::LEFT_JOIN:
		return side == JoinSide::LEFT ? JoinFilterDecision::PUSH_LEFT : JoinFilterDecision::INSPECT_NULL_REJECTION;
	case JoinDecisionPolicy::ASOF_JOIN:
	case JoinDecisionPolicy::SINGLE_JOIN:
		return side == JoinSide::LEFT ? JoinFilterDecision::PUSH_LEFT : JoinFilterDecision::KEEP;
	case JoinDecisionPolicy::MARK_JOIN:
		if (side == JoinSide::LEFT) {
			return JoinFilterDecision::PUSH_LEFT;
		}
		if (side == JoinSide::RIGHT) {
			return JoinFilterDecision::INSPECT_MARK;
		}
		return JoinFilterDecision::KEEP;
	case JoinDecisionPolicy::GENERATED_RIGHT_FILTER:
		return side == JoinSide::RIGHT ? JoinFilterDecision::PUSH_RIGHT : JoinFilterDecision::DISCARD;
	}
	D_ASSERT(false);
	return JoinFilterDecision::KEEP;
}
#endif

JoinSide FilterPushdown::GetJoinSide(const Filter &filter, JoinDecisionPolicy policy,
                                     JoinBindingState &binding_state) const {
	auto indexed_side = GetIndexedJoinSide(ClassifyJoinSources(*filter.filter, binding_state), policy);
	if (indexed_side.status != LogicalPlanDataFlowStatus::SUCCESS) {
		throw InternalException("Indexed join binding classification failed");
	}
#ifdef DEBUG
	auto legacy_side = binding_state.GetLegacyJoinSide(filter);
	D_ASSERT(GetJoinFilterDecision(legacy_side, policy) == GetJoinFilterDecision(indexed_side.side, policy));
#endif
	return indexed_side.side;
}

JoinSide FilterPushdown::GetJoinSide(const Expression &expression, JoinDecisionPolicy policy,
                                     JoinBindingState &binding_state) const {
	auto sources = ClassifyJoinSources(expression, binding_state);
	auto indexed_side = GetIndexedJoinSide(sources, policy);
	if (indexed_side.status != LogicalPlanDataFlowStatus::SUCCESS) {
		throw InternalException("Indexed join binding classification failed");
	}
#ifdef DEBUG
	auto legacy_side = binding_state.GetLegacyJoinSide(expression);
	D_ASSERT(GetJoinFilterDecision(legacy_side, policy) == GetJoinFilterDecision(indexed_side.side, policy));
#endif
	return indexed_side.side;
}

FilterPushdown::IndexedDelimJoinResult
FilterPushdown::GetIndexedDelimJoinDecision(const Expression &expression, LogicalOperator &child,
                                            JoinBindingState &binding_state) const {
	auto &join = binding_state.Join();
	auto sources = ClassifyJoinSources(expression, binding_state);
	if (sources.status != LogicalPlanDataFlowStatus::SUCCESS) {
		return {sources.status, false};
	}
	if (!sources.left && !sources.right && !sources.join) {
		return {LogicalPlanDataFlowStatus::SUCCESS, true};
	} else if (&child == &*join.children[0]) {
		return {LogicalPlanDataFlowStatus::SUCCESS, sources.left && !sources.right && !sources.join};
	} else if (&child == &*join.children[1]) {
		return {LogicalPlanDataFlowStatus::SUCCESS, sources.right && !sources.left && !sources.join};
	}
	return {LogicalPlanDataFlowStatus::UNSUPPORTED, false};
}

#ifdef DEBUG
bool FilterPushdown::GetLegacyDelimJoinDecision(const Filter &filter, LogicalOperator &child) {
	auto child_bindings = child.GetColumnBindings();
	unordered_set<TableIndex> child_bindings_table;
	for (auto &binding : child_bindings) {
		child_bindings_table.insert(binding.table_index);
	}
	for (auto &binding : filter.bindings) {
		if (child_bindings_table.find(binding) == child_bindings_table.end()) {
			return false;
		}
	}
	return true;
}
#endif

void FilterPushdown::PushdownJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_ASOF_JOIN || op->type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
	         op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN);
	auto &join = op->Cast<LogicalJoin>();

	const auto restore_projection_maps = join.HasProjectionMap();
	auto left_projection_map = join.left_projection_map;
	auto right_projection_map = join.right_projection_map;

	JoinBindingState binding_state(*op, context.data_flow);

	switch (join.join_type) {
	case JoinType::OUTER:
		PushdownOuterJoin(op, context);
		break;
	case JoinType::INNER:
		//	AsOf joins can't push anything into the RHS, so treat it as a left join
		if (op->type == LogicalOperatorType::LOGICAL_ASOF_JOIN) {
			PushdownLeftJoin(op, binding_state, context);
			break;
		}
		PushdownInnerJoin(op, context);
		break;
	case JoinType::LEFT:
		PushdownLeftJoin(op, binding_state, context);
		break;
	case JoinType::MARK:
		PushdownMarkJoin(op, binding_state, context);
		break;
	case JoinType::SINGLE:
		PushdownSingleJoin(op, binding_state, context);
		break;
	case JoinType::SEMI:
	case JoinType::ANTI:
		PushdownSemiAntiJoin(op, context);
		break;
	default:
		// unsupported join type: stop pushing down
		return FinishPushdown(op, context);
	}

	if (restore_projection_maps) {
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
			// Pushing down filter through join didn't change operator type (e.g., LogicalEmptyResult), restore maps
			auto &result_join = op->Cast<LogicalJoin>();
			result_join.left_projection_map = std::move(left_projection_map);
			result_join.right_projection_map = std::move(right_projection_map);
			context.mutator.RefreshOperator(result_join);
			break;
		}
		default:
			break;
		}
	}
}
FilterResult FilterPushdown::PushFilters() {
	for (auto &f : filters) {
		auto result = combiner.AddFilter(std::move(f->filter));
		D_ASSERT(result != FilterResult::UNSUPPORTED);
		if (result == FilterResult::UNSATISFIABLE) {
			// one of the filters is unsatisfiable - abort filter pushdown
			return FilterResult::UNSATISFIABLE;
		}
	}
	filters.clear();
	return FilterResult::SUCCESS;
}

FilterResult FilterPushdown::AddFilter(unique_ptr<Expression> expr) {
	if (PushFilters() == FilterResult::UNSATISFIABLE) {
		return FilterResult::UNSATISFIABLE;
	}
	// split up the filters by AND predicate
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(std::move(expr));
	LogicalFilter::SplitPredicates(expressions);
	// push the filters into the combiner
	for (auto &child_expr : expressions) {
		if (combiner.AddFilter(std::move(child_expr)) == FilterResult::UNSATISFIABLE) {
			return FilterResult::UNSATISFIABLE;
		}
	}
	return FilterResult::SUCCESS;
}

void FilterPushdown::GenerateFilters() {
	if (!filters.empty()) {
		D_ASSERT(!combiner.HasFilters());
		return;
	}
	combiner.GenerateFilters([&](unique_ptr<Expression> filter) {
		auto f = make_uniq<Filter>();
		f->filter = std::move(filter);
		f->ExtractBindings();
		filters.push_back(std::move(f));
	});
}

void FilterPushdown::AddLogicalFilter(unique_ptr<LogicalOperator> &op, vector<unique_ptr<Expression>> expressions,
                                      RewriteContext &context) {
	if (expressions.empty()) {
		// No left expressions, so needn't to add an extra filter operator.
		return;
	}
	auto filter = make_uniq<LogicalFilter>();
	if (op->has_estimated_cardinality) {
		// set the filter's estimated cardinality as the child op's.
		// if the filter is created during the filter pushdown optimization, the estimated cardinality will be later
		// overridden during the join order optimization to a more accurate one.
		// if the filter is created during the statistics propagation, the estimated cardinality won't be set unless set
		// here. assuming the filters introduced during the statistics propagation have little effect in reducing the
		// cardinality, we adopt the the cardinality of the child. this could be improved by MinMax info from the
		// statistics propagation
		filter->SetEstimatedCardinality(op->estimated_cardinality);
	}
	filter->expressions = std::move(expressions);
	context.mutator.InsertUnary(op, std::move(filter));
}

void FilterPushdown::PushFinalFilters(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	vector<unique_ptr<Expression>> expressions;
	for (auto &f : filters) {
		expressions.push_back(std::move(f->filter));
	}

	AddLogicalFilter(op, std::move(expressions), context);
}

bool FilterPushdown::PushFiltersIntoDelimJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	for (idx_t i = 0; i < filters.size(); i++) {
		auto &f = *filters[i];
		for (auto &child : op->children) {
			FilterPushdown pushdown(optimizer, convert_mark_joins, projection_mode);
			JoinBindingState binding_state(*op, context.data_flow);
			auto indexed_decision = GetIndexedDelimJoinDecision(*f.filter, *child, binding_state);
			if (indexed_decision.status != LogicalPlanDataFlowStatus::SUCCESS) {
				throw InternalException("Indexed delim join binding classification failed");
			}
			const auto should_push = indexed_decision.should_push;
#ifdef DEBUG
			D_ASSERT(should_push == GetLegacyDelimJoinDecision(f, *child));
#endif

			if (!should_push) {
				continue;
			}

			// copy the filter
			auto filter_copy = f.filter->Copy();
			if (pushdown.AddFilter(std::move(filter_copy)) == FilterResult::UNSATISFIABLE) {
				ReplaceWithEmptyResult(op, context);
				return false;
			}

			// push the filter into the child.
			pushdown.GenerateFilters();
			pushdown.Rewrite(child, context);

			// Don't push same filter again
			filters.erase_at(i);
			i--;
			break;
		}
	}
	return true;
}

void FilterPushdown::FinishPushdown(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	// unhandled type, first perform filter pushdown in its children
	for (auto &child : op->children) {
		FilterPushdown pushdown(optimizer, convert_mark_joins, projection_mode);
		pushdown.Rewrite(child, context);
	}
	// now push any existing filters
	PushFinalFilters(op, context);
}

void FilterPushdown::ReplaceWithEmptyResult(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	op->ResolveOperatorTypes();
	auto empty_result = make_uniq<LogicalEmptyResult>(op->types, op->GetColumnBindings());
	context.mutator.ReplaceSubtree(op, std::move(empty_result));
}

void FilterPushdown::Filter::ExtractBindings() {
	bindings.clear();
	LogicalJoin::GetExpressionBindings(*filter, bindings);
}

} // namespace duckdb
