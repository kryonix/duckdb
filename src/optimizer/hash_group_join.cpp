#include "duckdb/optimizer/hash_group_join.hpp"

#include "duckdb/optimizer/key_properties.hpp"
#include "duckdb/optimizer/optimizer.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/index/art/art.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_group_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

bool HashGroupJoinPlanningEnabled(ClientContext &context) {
	return Settings::Get<DebugGroupJoinStrategySetting>(context) != GroupJoinStrategy::DISABLED &&
	       !Optimizer::OptimizerDisabled(context, OptimizerType::GROUP_JOIN);
}

bool ForceHashGroupJoinPlanning(ClientContext &context) {
	return HashGroupJoinPlanningEnabled(context) &&
	       Settings::Get<DebugGroupJoinStrategySetting>(context) == GroupJoinStrategy::FORCE;
}

static optional_idx GetDirectReferenceIndex(const Expression &expression, LogicalOperator &input) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		return index < input.GetColumnBindings().size() ? optional_idx(index) : optional_idx();
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return optional_idx();
	}
	auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
	auto bindings = input.GetColumnBindings();
	for (idx_t index = 0; index < bindings.size(); index++) {
		if (bindings[index] == binding) {
			return optional_idx(index);
		}
	}
	return optional_idx();
}

static bool ExpressionUsesOnlyJoinChild(const Expression &expression, LogicalComparisonJoin &join, idx_t child_index) {
	bool valid = true;
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
		auto bindings = join.children[child_index]->GetColumnBindings();
		return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		auto left_count = join.children[0]->GetColumnBindings().size();
		if (child_index == 0) {
			return index < left_count;
		}
		return index >= left_count && index < left_count + join.children[1]->GetColumnBindings().size();
	}
	ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) {
		if (valid && !ExpressionUsesOnlyJoinChild(child, join, child_index)) {
			valid = false;
		}
	});
	return valid;
}

static bool AggregateFunctionsSupported(const LogicalAggregate &aggregate, HashGroupJoinCandidateMode mode) {
	if (aggregate.expressions.empty()) {
		return false;
	}
	for (auto &expression : aggregate.expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &aggr = expression->Cast<BoundAggregateExpression>();
		auto &callbacks = aggr.Function().GetCallbacks();
		if ((aggr.GetOrderBys() && mode == HashGroupJoinCandidateMode::STRICT) || aggr.IsVolatile() ||
		    aggr.StateExportMode() != AggregateStateExportMode::NONE || !callbacks.HasStateInitCallback() ||
		    !callbacks.HasStateSizeCallback() || !callbacks.HasStateUpdateCallback() ||
		    !callbacks.HasStateCombineCallback() || !callbacks.HasStateFinalizeCallback()) {
			return false;
		}
		for (auto &child : aggr.GetChildren()) {
			if (child->IsVolatile()) {
				return false;
			}
		}
		if (aggr.GetFilter() && aggr.GetFilter()->IsVolatile()) {
			return false;
		}
		if (aggr.GetOrderBys()) {
			for (auto &order : aggr.GetOrderBys()->orders) {
				if (order.expression->IsVolatile()) {
					return false;
				}
			}
		}
	}
	return true;
}

static bool AggregatesUseProbe(const LogicalAggregate &aggregate, LogicalComparisonJoin &join, idx_t probe_child,
                               HashGroupJoinCandidateMode mode) {
	if (!AggregateFunctionsSupported(aggregate, mode)) {
		return false;
	}
	for (auto &expression : aggregate.expressions) {
		auto &aggr = expression->Cast<BoundAggregateExpression>();
		for (auto &child : aggr.GetChildren()) {
			if (!ExpressionUsesOnlyJoinChild(*child, join, probe_child)) {
				return false;
			}
		}
		if (aggr.GetFilter() && !ExpressionUsesOnlyJoinChild(*aggr.GetFilter(), join, probe_child)) {
			return false;
		}
		if (aggr.GetOrderBys()) {
			for (auto &order : aggr.GetOrderBys()->orders) {
				if (!ExpressionUsesOnlyJoinChild(*order.expression, join, probe_child)) {
					return false;
				}
			}
		}
	}
	return true;
}

static optional_idx GetJoinOutputReference(const Expression &expression, LogicalComparisonJoin &join,
                                           idx_t &child_index) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		auto left_count = join.children[0]->GetColumnBindings().size();
		if (index < left_count) {
			child_index = 0;
			return optional_idx(index);
		}
		index -= left_count;
		if (index < join.children[1]->GetColumnBindings().size()) {
			child_index = 1;
			return optional_idx(index);
		}
		return optional_idx();
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return optional_idx();
	}
	for (idx_t side = 0; side < 2; side++) {
		auto index = GetDirectReferenceIndex(expression, *join.children[side]);
		if (index.IsValid()) {
			child_index = side;
			return index;
		}
	}
	return optional_idx();
}

optional<HashGroupJoinCandidate> TryGetHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                                              ClientContext &context, HashGroupJoinCandidateMode mode) {
	(void)context;
	if ((join.join_type != JoinType::INNER && join.join_type != JoinType::LEFT && join.join_type != JoinType::RIGHT &&
	     join.join_type != JoinType::SEMI) ||
	    join.HasProjectionMap() || join.children.size() != 2 || join.conditions.empty() ||
	    join.HasArbitraryConditions() || !aggregate.grouping_functions.empty() || aggregate.grouping_sets.size() > 1 ||
	    aggregate.groups.empty()) {
		return nullopt;
	}
	if (!aggregate.grouping_sets.empty()) {
		auto &grouping_set = aggregate.grouping_sets[0];
		if (grouping_set.size() != aggregate.groups.size()) {
			return nullopt;
		}
		for (idx_t group_idx = 0; group_idx < aggregate.groups.size(); group_idx++) {
			if (grouping_set.find(ProjectionIndex(group_idx)) == grouping_set.end()) {
				return nullopt;
			}
		}
	}

	vector<idx_t> side_keys[2];
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
		    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType()) {
			return nullopt;
		}
		auto left = GetDirectReferenceIndex(condition.GetLHS(), *join.children[0]);
		auto right = GetDirectReferenceIndex(condition.GetRHS(), *join.children[1]);
		if (!left.IsValid() || !right.IsValid()) {
			return nullopt;
		}
		side_keys[0].push_back(left.GetIndex());
		side_keys[1].push_back(right.GetIndex());
	}

	for (bool require_unique : {true, false}) {
		for (idx_t owner_child : {idx_t(1), idx_t(0)}) {
			HashGroupJoinUnmatchedPolicy unmatched_policy;
			if (join.join_type == JoinType::INNER || (join.join_type == JoinType::SEMI && owner_child == 1)) {
				unmatched_policy = HashGroupJoinUnmatchedPolicy::DISCARD;
			} else if ((join.join_type == JoinType::LEFT && owner_child == 0) ||
			           (join.join_type == JoinType::RIGHT && owner_child == 1)) {
				unmatched_policy = HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW;
			} else {
				continue;
			}
			const auto probe_child = 1 - owner_child;
			if (!AggregatesUseProbe(aggregate, join, probe_child, mode)) {
				continue;
			}
			auto key_property = GetUniqueKeyProperty(*join.children[owner_child], side_keys[owner_child]);
			if (key_property.has_value() != require_unique) {
				continue;
			}
			HashGroupJoinCandidate candidate {owner_child,
			                                  probe_child,
			                                  side_keys[owner_child],
			                                  side_keys[probe_child],
			                                  {},
			                                  {},
			                                  key_property.has_value(),
			                                  unmatched_policy,
			                                  false,
			                                  join.join_type == JoinType::SEMI};

			vector<bool> used_conditions(join.conditions.size(), false);
			bool valid = true;
			for (auto &group : aggregate.groups) {
				if (group->IsVolatile()) {
					valid = false;
					break;
				}
				idx_t group_child;
				auto group_index = GetJoinOutputReference(*group, join, group_child);
				if (!group_index.IsValid()) {
					valid = false;
					break;
				}
				optional_idx condition_index;
				for (idx_t index = 0; index < join.conditions.size(); index++) {
					if (side_keys[group_child][index] == group_index.GetIndex()) {
						if (condition_index.IsValid()) {
							valid = false;
							break;
						}
						condition_index = optional_idx(index);
					}
				}
				if (!valid) {
					break;
				}
				if (condition_index.IsValid()) {
					auto index = condition_index.GetIndex();
					if (candidate.unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD &&
					    group_child != candidate.owner_child) {
						valid = false;
						break;
					}
					if (used_conditions[index]) {
						valid = false;
						break;
					}
					used_conditions[index] = true;
					candidate.output_groups.push_back({HashGroupJoinOutputSource::KEY, index});
					continue;
				}
				if (group_child != candidate.owner_child ||
				    (key_property && !key_property->FunctionallyDetermines(*join.children[candidate.owner_child],
				                                                           group_index.GetIndex()))) {
					valid = false;
					break;
				}
				auto payload_entry = std::find(candidate.owner_payload_indices.begin(),
				                               candidate.owner_payload_indices.end(), group_index.GetIndex());
				if (payload_entry != candidate.owner_payload_indices.end()) {
					valid = false;
					break;
				}
				auto payload_index = candidate.owner_payload_indices.size();
				candidate.owner_payload_indices.push_back(group_index.GetIndex());
				candidate.output_groups.push_back({HashGroupJoinOutputSource::OWNER_PAYLOAD, payload_index});
			}
			if (!valid) {
				continue;
			}
			candidate.routed = !candidate.unique_owner || std::find(used_conditions.begin(), used_conditions.end(),
			                                                        false) != used_conditions.end();
			return candidate;
		}
	}
	return nullopt;
}

static bool AutoHashGroupJoinAggregatesSupported(const LogicalAggregate &aggregate, idx_t &state_size) {
	state_size = sizeof(idx_t);
	for (auto &expression : aggregate.expressions) {
		auto &aggr = expression->Cast<BoundAggregateExpression>();
		auto &callbacks = aggr.Function().GetCallbacks();
		if (aggr.IsDistinct() || aggr.GetFilter() || aggr.GetOrderBys() || callbacks.HasStateDestructorCallback() ||
		    aggr.Function().GetName() == "combine_aggr") {
			return false;
		}
		for (auto &child : aggr.GetChildren()) {
			if (child->GetReturnType().IsAggregateState()) {
				return false;
			}
		}
		state_size += aggr.Function().GetStateSize(aggr.BindInfo().get());
	}
	return state_size <= 128;
}

struct AutoHashGroupJoinCostModel {
	static constexpr idx_t MIN_OWNER_ROWS = 1024;
	static constexpr idx_t MIN_KEY_WIDTH = 16;
	static constexpr double MIN_PROBE_OWNER_RATIO = 4;
	static constexpr double MIN_RETENTION = 0.7;
	static constexpr double MIN_MATCH_DENSITY = 0.7;
	static constexpr double MIN_FANOUT = 2;
	static constexpr double MAX_FANOUT = 32;
	static constexpr double MAX_COST_RATIO = 0.8;
	static constexpr double MAX_INDEX_OWNER_PROBE_RATIO = 0.0001;
	static constexpr double MAX_INDEX_RETENTION = 0.25;
};

static bool HasAutoHashGroupJoinART(LogicalComparisonJoin &join, const HashGroupJoinCandidate &candidate,
                                    ClientContext &context) {
	vector<idx_t> probe_columns = candidate.probe_key_indices;
	reference<LogicalOperator> probe(*join.children[candidate.probe_child]);
	while (probe.get().type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = probe.get().Cast<LogicalFilter>();
		for (auto &expression : filter.expressions) {
			if (expression->IsVolatile()) {
				return false;
			}
		}
		if (!filter.projection_map.empty()) {
			for (auto &column : probe_columns) {
				if (column >= filter.projection_map.size()) {
					return false;
				}
				column = filter.projection_map[column].GetIndex();
			}
		}
		if (filter.children.size() != 1) {
			return false;
		}
		probe = *filter.children[0];
	}
	if (probe.get().type != LogicalOperatorType::LOGICAL_GET) {
		return false;
	}
	auto &get = probe.get().Cast<LogicalGet>();
	if (get.function.name != "seq_scan" || !get.GetTable() || get.GetTable()->type != CatalogType::TABLE_ENTRY) {
		return false;
	}
	auto &table = get.GetTable()->Cast<DuckTableEntry>();
	auto bindings = get.GetColumnBindings();
	vector<idx_t> physical_columns;
	for (auto column : probe_columns) {
		if (column >= bindings.size()) {
			return false;
		}
		auto &column_index = get.GetColumnIndex(bindings[column]);
		if (!column_index.HasPrimaryIndex() || column_index.HasChildren() || column_index.IsVirtualColumn()) {
			return false;
		}
		physical_columns.push_back(table.GetStorageIndex(column_index).GetPrimaryIndex());
	}

	auto &info = table.GetStorage().GetDataTableInfo();
	info->BindIndexes(context, ART::TYPE_NAME);
	for (auto &index : info->GetIndexes().Indexes()) {
		if (!index.IsBound() || index.GetIndexType() != ART::TYPE_NAME) {
			continue;
		}
		auto &art = index.Cast<ART>();
		if (art.unbound_expressions.size() != physical_columns.size()) {
			continue;
		}
		vector<bool> matched_columns(physical_columns.size(), false);
		bool matches = true;
		for (idx_t expression_idx = 0; expression_idx < art.unbound_expressions.size(); expression_idx++) {
			auto &expression = *art.unbound_expressions[expression_idx];
			if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				matches = false;
				break;
			}
			auto &column = expression.Cast<BoundColumnRefExpression>();
			auto index_column = column.Binding().column_index;
			if (column.Depth() != 0 || index_column >= art.GetColumnIds().size()) {
				matches = false;
				break;
			}
			auto entry = std::find(physical_columns.begin(), physical_columns.end(), art.GetColumnIds()[index_column]);
			if (entry == physical_columns.end()) {
				matches = false;
				break;
			}
			auto key_idx = NumericCast<idx_t>(entry - physical_columns.begin());
			if (matched_columns[key_idx]) {
				matches = false;
				break;
			}
			matched_columns[key_idx] = true;
		}
		if (matches) {
			return true;
		}
	}
	return false;
}

static bool PassesAutoIndexGroupJoinCostModel(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                              const HashGroupJoinCandidate &candidate, ClientContext &context) {
	if (join.join_type != JoinType::INNER || candidate.routed || !candidate.unique_owner ||
	    candidate.unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD || !join.has_estimated_cardinality ||
	    !join.children[candidate.owner_child]->has_estimated_cardinality ||
	    !join.children[candidate.probe_child]->has_estimated_cardinality) {
		return false;
	}
	idx_t state_size;
	if (!AutoHashGroupJoinAggregatesSupported(aggregate, state_size)) {
		return false;
	}
	const auto owner_rows = join.children[candidate.owner_child]->estimated_cardinality;
	const auto probe_rows = join.children[candidate.probe_child]->estimated_cardinality;
	const auto match_rows = join.estimated_cardinality;
	if (probe_rows == 0 ||
	    static_cast<double>(owner_rows) >
	        AutoHashGroupJoinCostModel::MAX_INDEX_OWNER_PROBE_RATIO * static_cast<double>(probe_rows) ||
	    static_cast<double>(match_rows) >
	        AutoHashGroupJoinCostModel::MAX_INDEX_RETENTION * static_cast<double>(probe_rows)) {
		return false;
	}
	return HasAutoHashGroupJoinART(join, candidate, context);
}

static bool PassesAutoHashGroupJoinCostModel(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                             const HashGroupJoinCandidate &candidate, ClientContext &context) {
	if (join.join_type != JoinType::INNER || candidate.routed || !candidate.unique_owner ||
	    candidate.unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD ||
	    TaskScheduler::GetScheduler(context).NumberOfThreads() != 1 || !join.has_estimated_cardinality ||
	    !join.children[candidate.owner_child]->has_estimated_cardinality ||
	    !join.children[candidate.probe_child]->has_estimated_cardinality) {
		return false;
	}

	idx_t state_size;
	if (!AutoHashGroupJoinAggregatesSupported(aggregate, state_size)) {
		return false;
	}
	idx_t key_width = 0;
	for (auto &condition : join.conditions) {
		auto &key = candidate.owner_child == 0 ? condition.GetLHS() : condition.GetRHS();
		key_width += GetTypeIdSize(key.GetReturnType().InternalType());
	}

	const auto owner_rows = join.children[candidate.owner_child]->estimated_cardinality;
	const auto probe_rows = join.children[candidate.probe_child]->estimated_cardinality;
	const auto match_rows = join.estimated_cardinality;
	if (key_width < AutoHashGroupJoinCostModel::MIN_KEY_WIDTH ||
	    owner_rows < AutoHashGroupJoinCostModel::MIN_OWNER_ROWS || match_rows == 0 ||
	    static_cast<double>(probe_rows) <
	        AutoHashGroupJoinCostModel::MIN_PROBE_OWNER_RATIO * static_cast<double>(owner_rows)) {
		return false;
	}
	const auto retention = static_cast<double>(match_rows) / static_cast<double>(MaxValue<idx_t>(probe_rows, 1));
	const auto matched_groups = MinValue(owner_rows, match_rows);
	const auto match_density =
	    static_cast<double>(matched_groups) / static_cast<double>(MaxValue<idx_t>(owner_rows, 1));
	const auto fanout = static_cast<double>(match_rows) / static_cast<double>(MaxValue<idx_t>(matched_groups, 1));
	if (retention < AutoHashGroupJoinCostModel::MIN_RETENTION ||
	    match_density < AutoHashGroupJoinCostModel::MIN_MATCH_DENSITY ||
	    fanout < AutoHashGroupJoinCostModel::MIN_FANOUT || fanout > AutoHashGroupJoinCostModel::MAX_FANOUT) {
		return false;
	}

	const auto key_cost = static_cast<double>(key_width) / 8.0;
	const auto state_cost = static_cast<double>(state_size) / 16.0;
	const auto separate_cost = static_cast<double>(owner_rows + probe_rows) * key_cost +
	                           static_cast<double>(match_rows) * (key_cost + state_cost + 1.0);
	const auto eager_cost = static_cast<double>(probe_rows) * (key_cost + state_cost) +
	                        static_cast<double>(matched_groups) * (key_cost + state_cost) +
	                        static_cast<double>(owner_rows) * key_cost;
	const auto group_join_cost = static_cast<double>(owner_rows) * (key_cost + state_cost) +
	                             static_cast<double>(probe_rows) * key_cost +
	                             static_cast<double>(match_rows) * state_cost;
	return group_join_cost <= MinValue(separate_cost, eager_cost) * AutoHashGroupJoinCostModel::MAX_COST_RATIO;
}

optional<HashGroupJoinCandidate> TrySelectHashGroupJoinCandidate(LogicalAggregate &aggregate,
                                                                 LogicalComparisonJoin &join, ClientContext &context,
                                                                 HashGroupJoinCandidateMode mode) {
	if (!HashGroupJoinPlanningEnabled(context)) {
		return nullopt;
	}
	auto candidate = TryGetHashGroupJoinCandidate(aggregate, join, context, mode);
	if (!candidate) {
		return nullopt;
	}
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy == GroupJoinStrategy::FORCE) {
		return candidate;
	}
	if (strategy != GroupJoinStrategy::AUTO) {
		return nullopt;
	}
	auto execution = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	if ((execution == GroupJoinExecutionMode::AUTO || execution == GroupJoinExecutionMode::INDEX) &&
	    PassesAutoIndexGroupJoinCostModel(aggregate, join, *candidate, context)) {
		aggregate.group_join_auto_selected = true;
		aggregate.group_join_auto_index = true;
		return candidate;
	}
	if (!PassesAutoHashGroupJoinCostModel(aggregate, join, *candidate, context)) {
		return nullopt;
	}
	aggregate.group_join_auto_selected = true;
	return candidate;
}

optional<HashGroupJoinCandidate> TryGetPlannedHashGroupJoinCandidate(LogicalAggregate &aggregate,
                                                                     LogicalComparisonJoin &join,
                                                                     ClientContext &context,
                                                                     HashGroupJoinCandidateMode mode) {
	if (!HashGroupJoinPlanningEnabled(context)) {
		return nullopt;
	}
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy == GroupJoinStrategy::DISABLED ||
	    (strategy == GroupJoinStrategy::AUTO && !aggregate.group_join_auto_selected)) {
		return nullopt;
	}
	return TryGetHashGroupJoinCandidate(aggregate, join, context, mode);
}

void PlanHashGroupJoins(unique_ptr<LogicalOperator> &root, ClientContext &context) {
	for (auto &child : root->children) {
		PlanHashGroupJoins(child, context);
	}
	if (root->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || root->children.size() != 1 ||
	    root->children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &aggregate = root->Cast<LogicalAggregate>();
	auto &join = root->children[0]->Cast<LogicalComparisonJoin>();
	auto candidate = TryGetPlannedHashGroupJoinCandidate(aggregate, join, context,
	                                                     HashGroupJoinCandidateMode::ALLOW_AGGREGATE_ORDER);
	if (!candidate) {
		return;
	}

	auto result =
	    make_uniq<LogicalGroupJoin>(aggregate.group_index, aggregate.aggregate_index, std::move(aggregate.expressions));
	result->groupings_index = aggregate.groupings_index;
	result->groups = std::move(aggregate.groups);
	result->grouping_sets = std::move(aggregate.grouping_sets);
	result->grouping_functions = std::move(aggregate.grouping_functions);
	result->group_stats = std::move(aggregate.group_stats);
	result->distinct_validity = aggregate.distinct_validity;
	result->types = std::move(aggregate.types);
	result->estimated_cardinality = aggregate.estimated_cardinality;
	result->has_estimated_cardinality = aggregate.has_estimated_cardinality;

	result->owner_child = candidate->owner_child;
	result->probe_child = candidate->probe_child;
	result->left_column_count = join.children[0]->GetColumnBindings().size();
	result->owner_key_indices = std::move(candidate->owner_key_indices);
	result->probe_key_indices = std::move(candidate->probe_key_indices);
	result->owner_payload_indices = std::move(candidate->owner_payload_indices);
	for (auto &output_group : candidate->output_groups) {
		result->output_group_sources.push_back(output_group.source);
		result->output_group_indices.push_back(output_group.index);
	}
	result->unmatched_policy = candidate->unmatched_policy;
	result->routed = candidate->routed;
	result->unique_owner = candidate->unique_owner;
	result->single_match = candidate->single_match;
	result->use_index = aggregate.group_join_auto_index;
	if (candidate->owner_child == 1) {
		result->filter_pushdown = std::move(join.filter_pushdown);
	}
	result->children = std::move(join.children);
	root = std::move(result);
}

static vector<idx_t> GetProjectedColumns(LogicalOperator &child, const vector<ProjectionIndex> &projection_map) {
	vector<idx_t> result;
	if (projection_map.empty()) {
		for (idx_t index = 0; index < child.GetColumnBindings().size(); index++) {
			result.push_back(index);
		}
	} else {
		for (auto index : projection_map) {
			result.push_back(index.GetIndex());
		}
	}
	return result;
}

optional<StaticHashGroupJoinCandidate> TryGetStaticHashGroupJoinCandidate(LogicalComparisonJoin &join,
                                                                          ClientContext &context,
                                                                          HashGroupJoinCandidateMode mode) {
	(void)context;
	if (join.join_type != JoinType::LEFT || join.children.size() != 2 || join.conditions.empty() ||
	    join.HasArbitraryConditions() ||
	    join.children[1]->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
		return nullopt;
	}
	auto &aggregate = join.children[1]->Cast<LogicalAggregate>();
	if (aggregate.children.size() != 1 || aggregate.groups.empty() || !aggregate.grouping_functions.empty() ||
	    aggregate.grouping_sets.size() > 1 || join.conditions.size() != aggregate.groups.size() ||
	    !AggregateFunctionsSupported(aggregate, mode)) {
		return nullopt;
	}
	if (!aggregate.grouping_sets.empty()) {
		auto &grouping_set = aggregate.grouping_sets[0];
		if (grouping_set.size() != aggregate.groups.size()) {
			return nullopt;
		}
		for (idx_t group_idx = 0; group_idx < aggregate.groups.size(); group_idx++) {
			if (grouping_set.find(ProjectionIndex(group_idx)) == grouping_set.end()) {
				return nullopt;
			}
		}
	}
	for (auto &group : aggregate.groups) {
		if (group->IsVolatile()) {
			return nullopt;
		}
	}

	vector<idx_t> owner_key_indices(aggregate.groups.size(), DConstants::INVALID_INDEX);
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
		    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType()) {
			return nullopt;
		}
		auto owner_key = GetDirectReferenceIndex(condition.GetLHS(), *join.children[0]);
		auto aggregate_output = GetDirectReferenceIndex(condition.GetRHS(), aggregate);
		if (!owner_key.IsValid() || !aggregate_output.IsValid() ||
		    aggregate_output.GetIndex() >= aggregate.groups.size() ||
		    owner_key_indices[aggregate_output.GetIndex()] != DConstants::INVALID_INDEX) {
			return nullopt;
		}
		owner_key_indices[aggregate_output.GetIndex()] = owner_key.GetIndex();
	}
	for (auto owner_key : owner_key_indices) {
		if (owner_key == DConstants::INVALID_INDEX) {
			return nullopt;
		}
	}
	StaticHashGroupJoinCandidate result {&aggregate, owner_key_indices, {}, {}};
	auto left_outputs = GetProjectedColumns(*join.children[0], join.left_projection_map);
	for (auto output_index : left_outputs) {
		auto key_entry = std::find(owner_key_indices.begin(), owner_key_indices.end(), output_index);
		if (key_entry != owner_key_indices.end()) {
			result.output_columns.push_back(
			    {HashGroupJoinOutputSource::KEY, NumericCast<idx_t>(key_entry - owner_key_indices.begin())});
			continue;
		}
		auto payload_entry =
		    std::find(result.owner_payload_indices.begin(), result.owner_payload_indices.end(), output_index);
		idx_t payload_index;
		if (payload_entry == result.owner_payload_indices.end()) {
			payload_index = result.owner_payload_indices.size();
			result.owner_payload_indices.push_back(output_index);
		} else {
			payload_index = NumericCast<idx_t>(payload_entry - result.owner_payload_indices.begin());
		}
		result.output_columns.push_back({HashGroupJoinOutputSource::OWNER_PAYLOAD, payload_index});
	}

	auto right_outputs = GetProjectedColumns(aggregate, join.right_projection_map);
	for (auto output_index : right_outputs) {
		if (output_index < aggregate.groups.size()) {
			result.output_columns.push_back({HashGroupJoinOutputSource::MATCHED_KEY, output_index});
		} else {
			auto aggregate_index = output_index - aggregate.groups.size();
			if (aggregate_index >= aggregate.expressions.size()) {
				return nullopt;
			}
			result.output_columns.push_back({HashGroupJoinOutputSource::AGGREGATE, aggregate_index});
		}
	}
	if (result.output_columns.size() != join.GetColumnBindings().size()) {
		return nullopt;
	}
	return result;
}

bool IsStaticHashGroupJoinAggregate(LogicalOperator &root, LogicalAggregate &aggregate, ClientContext &context,
                                    HashGroupJoinCandidateMode mode) {
	if (root.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
	    root.type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		auto &join = root.Cast<LogicalComparisonJoin>();
		if (join.children.size() == 2 && join.children[1].get() == &aggregate &&
		    TryGetStaticHashGroupJoinCandidate(join, context, mode)) {
			return true;
		}
	}
	for (auto &child : root.children) {
		if (IsStaticHashGroupJoinAggregate(*child, aggregate, context, mode)) {
			return true;
		}
	}
	return false;
}

} // namespace duckdb
