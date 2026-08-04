#include "duckdb/optimizer/hash_group_join.hpp"

#include "duckdb/optimizer/key_properties.hpp"
#include "duckdb/optimizer/join_filter_pushdown_optimizer.hpp"
#include "duckdb/optimizer/optimizer.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/execution/index/art/art.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/function_binder.hpp"
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
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"

#include <cmath>

namespace duckdb {

bool HashGroupJoinPlanningEnabled(ClientContext &context) {
	return Settings::Get<DebugGroupJoinStrategySetting>(context) != GroupJoinStrategy::DISABLED &&
	       !Optimizer::OptimizerDisabled(context, OptimizerType::GROUP_JOIN);
}

bool ForceHashGroupJoinPlanning(ClientContext &context) {
	if (!HashGroupJoinPlanningEnabled(context)) {
		return false;
	}
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	return strategy == GroupJoinStrategy::FORCE || strategy == GroupJoinStrategy::HASH ||
	       strategy == GroupJoinStrategy::PERFECT || strategy == GroupJoinStrategy::EAGER ||
	       strategy == GroupJoinStrategy::INDEX;
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

static bool PhysicalEagerGroupJoinSupported(const LogicalAggregate &aggregate,
                                            const HashGroupJoinCandidate &candidate) {
	for (auto &expression : aggregate.expressions) {
		auto &aggregate_expression = expression->Cast<BoundAggregateExpression>();
		auto &function = aggregate_expression.Function();
		if (candidate.routed && aggregate_expression.IsDistinct()) {
			return false;
		}
		if (aggregate_expression.StateExportMode() != AggregateStateExportMode::NONE ||
		    !function.HasStateCombineCallback() || !function.HasGetStateTypeCallback() ||
		    function.HasExportAggregateStateCallback() != function.HasImportAggregateStateCallback()) {
			return false;
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
	bool supported = true;
	for (auto &expression : aggregate.expressions) {
		auto &aggr = expression->Cast<BoundAggregateExpression>();
		auto &callbacks = aggr.Function().GetCallbacks();
		state_size += aggr.Function().GetStateSize(aggr.BindInfo().get());
		if (aggr.IsDistinct() || aggr.GetFilter() || aggr.GetOrderBys() || callbacks.HasStateDestructorCallback() ||
		    aggr.Function().GetName() == "combine_aggr") {
			supported = false;
		}
		for (auto &child : aggr.GetChildren()) {
			if (child->GetReturnType().IsAggregateState()) {
				supported = false;
			}
		}
	}
	return supported && state_size <= 128;
}

struct AutoHashGroupJoinCostModel {
	static constexpr idx_t MIN_OWNER_ROWS = 100000;
	static constexpr idx_t MIN_KEY_WIDTH = 16;
	static constexpr double MIN_PROBE_OWNER_RATIO = 4;
	static constexpr double MIN_RETENTION = 0.7;
	static constexpr double MIN_MATCH_DENSITY = 0.7;
	static constexpr double MIN_FANOUT = 2;
	static constexpr double MIN_OWNERSHIP_FANOUT = 12;
	static constexpr double MAX_FANOUT = 32;
	static constexpr double MAX_COST_RATIO = 0.9;
	static constexpr double MAX_INDEX_OWNER_PROBE_RATIO = 0.0001;
	static constexpr double MAX_INDEX_RETENTION = 0.25;
	static constexpr double MAX_EAGER_MATERIALIZATION_PENALTY = 0.75;
	static constexpr double EAGER_DISTINCT_KEY_PENALTY = 8;
	static constexpr double EAGER_DISTINCT_MATERIALIZATION_FACTOR = 2.25;
	static constexpr double EAGER_WIDE_KEY_BYTES = 16;
	static constexpr double EAGER_KEY_WIDTH_RANGE = 8;
	static constexpr idx_t OWNERSHIP_WIDE_STATE_BYTES = 96;
	static constexpr idx_t MIN_SERIAL_STATE_BYTES = 96;
	static constexpr double MIN_PHYSICAL_EAGER_FANOUT = 64;
	static constexpr idx_t MIN_PHYSICAL_EAGER_OWNER_ROWS = 20000;
	static constexpr idx_t MAX_PHYSICAL_EAGER_STATE_BYTES = 96;
	static constexpr double PHYSICAL_EAGER_COST_RATIO = 0.85;
	static constexpr idx_t MIN_PERFECT_OWNER_ROWS = 20000;
	static constexpr idx_t MAX_PERFECT_STATE_BYTES = 16;
	static constexpr double MAX_PERFECT_SINGLETON_FANOUT = 1.25;
	static constexpr double MIN_PERFECT_REUSE_FANOUT = 8;
	static constexpr double MAX_PERFECT_FANOUT = 10;
	static constexpr double MIN_PERFECT_DENSITY = 0.7;
	static constexpr double PERFECT_COST_RATIO = 0.75;
};

static bool IsFixedSizeHashGroupJoinKey(const LogicalType &type) {
	switch (type.InternalType()) {
	case PhysicalType::VARCHAR:
	case PhysicalType::LIST:
	case PhysicalType::STRUCT:
	case PhysicalType::ARRAY:
	case PhysicalType::BIT:
		return false;
	default:
		return true;
	}
}

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

HashGroupJoinCostEstimate EstimateHashGroupJoinAlternatives(idx_t owner_rows, idx_t probe_rows, idx_t match_rows,
                                                            idx_t matched_groups, idx_t key_width, idx_t state_width,
                                                            bool routed, bool direct_inner, bool fixed_size_keys,
                                                            bool physical_eager_supported, bool perfect_supported,
                                                            idx_t perfect_range, ClientContext &context) {
	HashGroupJoinCostEstimate result;
	result.owner_rows = owner_rows;
	result.probe_rows = probe_rows;
	result.match_rows = match_rows;
	result.matched_groups = MinValue(matched_groups, owner_rows);
	result.key_width = key_width;
	result.state_width = state_width;
	if (match_rows != 0) {
		auto distinct_keys =
		    static_cast<double>(owner_rows) * static_cast<double>(probe_rows) / static_cast<double>(match_rows);
		result.distinct_probe_keys =
		    MinValue(probe_rows, MaxValue<idx_t>(LossyNumericCast<idx_t>(MaxValue<double>(distinct_keys, 1)), 1));
	}

	const auto key_cost = static_cast<double>(key_width) / 8.0;
	const auto state_cost = static_cast<double>(state_width) / 16.0;
	result.separate_cost = static_cast<double>(owner_rows + probe_rows) * key_cost +
	                       static_cast<double>(match_rows) * (key_cost + state_cost + 1.0);
	result.eager_cost = static_cast<double>(probe_rows) * (key_cost + state_cost) +
	                    static_cast<double>(result.distinct_probe_keys) * (key_cost + state_cost) *
	                        AutoHashGroupJoinCostModel::EAGER_DISTINCT_MATERIALIZATION_FACTOR +
	                    static_cast<double>(owner_rows) * key_cost;
	if (probe_rows != 0 && key_width != 0) {
		const auto collapse_overhead =
		    MinValue<double>(AutoHashGroupJoinCostModel::MAX_EAGER_MATERIALIZATION_PENALTY,
		                     AutoHashGroupJoinCostModel::EAGER_DISTINCT_KEY_PENALTY *
		                         static_cast<double>(result.distinct_probe_keys) / static_cast<double>(probe_rows));
		const auto narrow_key_ratio = MinValue<double>(
		    1.0,
		    MaxValue<double>(0, (AutoHashGroupJoinCostModel::EAGER_WIDE_KEY_BYTES - static_cast<double>(key_width)) /
		                            AutoHashGroupJoinCostModel::EAGER_KEY_WIDTH_RANGE));
		const auto narrow_key_penalty = narrow_key_ratio * narrow_key_ratio;
		result.eager_cost *= 1.0 + collapse_overhead * narrow_key_penalty;
	}
	result.memoizing_cost = static_cast<double>(owner_rows) * (key_cost + state_cost) +
	                        static_cast<double>(probe_rows) * key_cost + static_cast<double>(match_rows) * state_cost;
	result.index_cost = static_cast<double>(owner_rows) * std::log2(static_cast<double>(probe_rows) + 1) +
	                    static_cast<double>(match_rows) * (state_cost + 1.0);

	const auto fanout =
	    static_cast<double>(match_rows) / static_cast<double>(MaxValue<idx_t>(result.matched_groups, 1));
	result.execution_mode = TaskScheduler::GetScheduler(context).NumberOfThreads() == 1 ? GroupJoinExecutionMode::SERIAL
	                        : fanout > AutoHashGroupJoinCostModel::MAX_FANOUT           ? GroupJoinExecutionMode::LOCAL
	                                                                          : GroupJoinExecutionMode::OWNERSHIP;
	if (result.execution_mode == GroupJoinExecutionMode::OWNERSHIP &&
	    state_width > AutoHashGroupJoinCostModel::OWNERSHIP_WIDE_STATE_BYTES) {
		const auto wide_state_penalty =
		    static_cast<double>(state_width) / AutoHashGroupJoinCostModel::OWNERSHIP_WIDE_STATE_BYTES;
		result.eager_cost *= wide_state_penalty;
		result.memoizing_cost *= wide_state_penalty;
	}
	result.physical_eager_cost = result.eager_cost;
	result.perfect_cost = result.separate_cost;
	result.perfect_available = perfect_supported && direct_inner;
	const auto retention = static_cast<double>(match_rows) / static_cast<double>(MaxValue<idx_t>(probe_rows, 1));
	const auto match_density =
	    static_cast<double>(result.matched_groups) / static_cast<double>(MaxValue<idx_t>(owner_rows, 1));
	const auto perfect_density =
	    static_cast<double>(owner_rows) / static_cast<double>(MaxValue<idx_t>(perfect_range + 1, 1));
	const auto estimated_memory =
	    static_cast<double>(owner_rows) * static_cast<double>(key_width + state_width + (routed ? 80 : 48));
	if (estimated_memory > static_cast<double>(BufferManager::GetBufferManager(context).GetMaxMemory()) * 0.25) {
		result.execution_mode = GroupJoinExecutionMode::EXTERNAL;
	}
	if (result.execution_mode != GroupJoinExecutionMode::EXTERNAL && physical_eager_supported && direct_inner &&
	    owner_rows >= AutoHashGroupJoinCostModel::MIN_PHYSICAL_EAGER_OWNER_ROWS &&
	    key_width >= AutoHashGroupJoinCostModel::MIN_KEY_WIDTH &&
	    state_width <= AutoHashGroupJoinCostModel::MAX_PHYSICAL_EAGER_STATE_BYTES &&
	    fanout >= AutoHashGroupJoinCostModel::MIN_PHYSICAL_EAGER_FANOUT &&
	    retention >= AutoHashGroupJoinCostModel::MIN_RETENTION &&
	    match_density >= AutoHashGroupJoinCostModel::MIN_MATCH_DENSITY) {
		result.physical_eager_cost *= AutoHashGroupJoinCostModel::PHYSICAL_EAGER_COST_RATIO;
		result.physical_eager_selected =
		    result.physical_eager_cost <=
		    MinValue(result.separate_cost, result.eager_cost) * AutoHashGroupJoinCostModel::MAX_COST_RATIO;
	}
	if (result.execution_mode != GroupJoinExecutionMode::EXTERNAL && result.perfect_available &&
	    owner_rows >= AutoHashGroupJoinCostModel::MIN_PERFECT_OWNER_ROWS &&
	    state_width <= AutoHashGroupJoinCostModel::MAX_PERFECT_STATE_BYTES &&
	    (fanout <= AutoHashGroupJoinCostModel::MAX_PERFECT_SINGLETON_FANOUT ||
	     fanout >= AutoHashGroupJoinCostModel::MIN_PERFECT_REUSE_FANOUT) &&
	    fanout <= AutoHashGroupJoinCostModel::MAX_PERFECT_FANOUT &&
	    retention >= AutoHashGroupJoinCostModel::MIN_RETENTION &&
	    match_density >= AutoHashGroupJoinCostModel::MIN_MATCH_DENSITY &&
	    perfect_density >= AutoHashGroupJoinCostModel::MIN_PERFECT_DENSITY) {
		result.perfect_cost *= AutoHashGroupJoinCostModel::PERFECT_COST_RATIO;
		result.perfect_selected = result.perfect_cost <= MinValue(result.separate_cost, result.eager_cost) *
		                                                     AutoHashGroupJoinCostModel::MAX_COST_RATIO;
	}

	if (!direct_inner || !fixed_size_keys || key_width < AutoHashGroupJoinCostModel::MIN_KEY_WIDTH ||
	    owner_rows < AutoHashGroupJoinCostModel::MIN_OWNER_ROWS || match_rows == 0 ||
	    (result.execution_mode == GroupJoinExecutionMode::SERIAL &&
	     state_width < AutoHashGroupJoinCostModel::MIN_SERIAL_STATE_BYTES) ||
	    (result.execution_mode == GroupJoinExecutionMode::OWNERSHIP &&
	     fanout < AutoHashGroupJoinCostModel::MIN_OWNERSHIP_FANOUT) ||
	    static_cast<double>(probe_rows) <
	        AutoHashGroupJoinCostModel::MIN_PROBE_OWNER_RATIO * static_cast<double>(owner_rows)) {
		return result;
	}
	if (retention >= AutoHashGroupJoinCostModel::MIN_RETENTION &&
	    match_density >= AutoHashGroupJoinCostModel::MIN_MATCH_DENSITY &&
	    fanout >= AutoHashGroupJoinCostModel::MIN_FANOUT && fanout <= AutoHashGroupJoinCostModel::MAX_FANOUT &&
	    result.memoizing_cost <=
	        MinValue(result.separate_cost, result.eager_cost) * AutoHashGroupJoinCostModel::MAX_COST_RATIO) {
		result.hash_selected = true;
	}
	return result;
}

static bool TryGetPerfectGroupJoinBounds(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                         const HashGroupJoinCandidate &candidate, ClientContext &context,
                                         Value &minimum, Value &maximum, idx_t &range);

HashGroupJoinCostEstimate EstimateHashGroupJoinCost(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                                    const HashGroupJoinCandidate &candidate, ClientContext &context) {
	HashGroupJoinCostEstimate result;
	if (!join.has_estimated_cardinality || !join.children[candidate.owner_child]->has_estimated_cardinality ||
	    !join.children[candidate.probe_child]->has_estimated_cardinality) {
		return result;
	}
	idx_t key_width = 0;
	bool fixed_size_keys = true;
	for (auto &condition : join.conditions) {
		auto &key = candidate.owner_child == 0 ? condition.GetLHS() : condition.GetRHS();
		key_width += GetTypeIdSize(key.GetReturnType().InternalType());
		fixed_size_keys = fixed_size_keys && IsFixedSizeHashGroupJoinKey(key.GetReturnType());
	}
	idx_t state_width;
	const auto auto_aggregates_supported = AutoHashGroupJoinAggregatesSupported(aggregate, state_width);
	const auto physical_eager_supported = PhysicalEagerGroupJoinSupported(aggregate, candidate);
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range = 0;
	const auto perfect_supported =
	    TryGetPerfectGroupJoinBounds(aggregate, join, candidate, context, perfect_min, perfect_max, perfect_range);
	const auto direct_inner = join.join_type == JoinType::INNER && !candidate.routed && candidate.unique_owner &&
	                          candidate.unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD;
	result = EstimateHashGroupJoinAlternatives(
	    join.children[candidate.owner_child]->estimated_cardinality,
	    join.children[candidate.probe_child]->estimated_cardinality, join.estimated_cardinality,
	    aggregate.has_estimated_cardinality
	        ? aggregate.estimated_cardinality
	        : MinValue(join.children[candidate.owner_child]->estimated_cardinality, join.estimated_cardinality),
	    key_width, state_width, candidate.routed, direct_inner, fixed_size_keys, physical_eager_supported,
	    perfect_supported, perfect_range, context);
	if (!auto_aggregates_supported) {
		result.hash_selected = false;
		result.physical_eager_selected = false;
		result.perfect_selected = false;
	}
	result.index_available = direct_inner && HasAutoHashGroupJoinART(join, candidate, context);
	if (auto_aggregates_supported && direct_inner && result.probe_rows != 0 &&
	    static_cast<double>(result.owner_rows) <=
	        AutoHashGroupJoinCostModel::MAX_INDEX_OWNER_PROBE_RATIO * static_cast<double>(result.probe_rows) &&
	    static_cast<double>(result.match_rows) <=
	        AutoHashGroupJoinCostModel::MAX_INDEX_RETENTION * static_cast<double>(result.probe_rows) &&
	    result.index_available &&
	    result.index_cost <=
	        MinValue(result.separate_cost,
	                 MinValue(result.eager_cost,
	                          MinValue(result.physical_eager_cost,
	                                   result.perfect_available ? result.perfect_cost : result.separate_cost))) *
	            AutoHashGroupJoinCostModel::MAX_COST_RATIO) {
		result.index_selected = true;
		result.execution_mode = GroupJoinExecutionMode::INDEX;
	}
	return result;
}

optional<HashGroupJoinOrderContext> GetHashGroupJoinOrderContext(LogicalAggregate &aggregate, ClientContext &context) {
	if (!HashGroupJoinPlanningEnabled(context) || aggregate.children.size() != 1 ||
	    aggregate.children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return nullopt;
	}
	auto &join = aggregate.children[0]->Cast<LogicalComparisonJoin>();
	auto candidate =
	    TryGetHashGroupJoinCandidate(aggregate, join, context, HashGroupJoinCandidateMode::ALLOW_AGGREGATE_ORDER);
	if (!candidate) {
		return nullopt;
	}
	HashGroupJoinOrderContext result;
	result.strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (result.strategy == GroupJoinStrategy::SEPARATE || result.strategy == GroupJoinStrategy::FACTORIZED) {
		return nullopt;
	}
	const auto auto_aggregates_supported = AutoHashGroupJoinAggregatesSupported(aggregate, result.state_width);
	if (result.strategy == GroupJoinStrategy::AUTO && !auto_aggregates_supported) {
		return nullopt;
	}
	for (auto &condition : join.conditions) {
		auto &key = candidate->owner_child == 0 ? condition.GetLHS() : condition.GetRHS();
		result.key_width += GetTypeIdSize(key.GetReturnType().InternalType());
		result.fixed_size_keys = result.fixed_size_keys && IsFixedSizeHashGroupJoinKey(key.GetReturnType());
	}
	LogicalJoin::GetTableReferences(*join.children[candidate->owner_child], result.owner_tables);
	LogicalJoin::GetTableReferences(*join.children[candidate->probe_child], result.probe_tables);
	if (result.owner_tables.empty() || result.probe_tables.empty()) {
		return nullopt;
	}
	result.routed = candidate->routed;
	result.direct_inner = join.join_type == JoinType::INNER && !candidate->routed && candidate->unique_owner &&
	                      candidate->unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD;
	result.physical_eager_supported = PhysicalEagerGroupJoinSupported(aggregate, *candidate);
	Value perfect_min;
	Value perfect_max;
	result.perfect_supported = TryGetPerfectGroupJoinBounds(aggregate, join, *candidate, context, perfect_min,
	                                                        perfect_max, result.perfect_range);
	if ((result.strategy == GroupJoinStrategy::PERFECT && !result.perfect_supported) ||
	    (result.strategy == GroupJoinStrategy::EAGER && !result.physical_eager_supported)) {
		return nullopt;
	}
	return result;
}

static optional_idx GetPerfectGroupJoinKeyGroup(LogicalAggregate &aggregate, const HashGroupJoinCandidate &candidate) {
	if (!candidate.unique_owner || candidate.routed || candidate.owner_key_indices.size() != 1) {
		return optional_idx();
	}
	optional_idx key_group;
	for (idx_t group_idx = 0; group_idx < candidate.output_groups.size(); group_idx++) {
		auto &output = candidate.output_groups[group_idx];
		if (output.source == HashGroupJoinOutputSource::KEY && output.index == 0) {
			key_group = group_idx;
			break;
		}
	}
	if (!key_group.IsValid() || key_group.GetIndex() >= aggregate.groups.size()) {
		return optional_idx();
	}
	auto &key_type = aggregate.groups[key_group.GetIndex()]->GetReturnType();
	switch (key_type.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
		break;
	default:
		return optional_idx();
	}
	return key_group;
}

static optional_ptr<LogicalGet> FindPerfectGroupJoinGet(LogicalOperator &op, TableIndex table_index) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.table_index == table_index) {
			return get;
		}
	}
	for (auto &child : op.children) {
		auto result = FindPerfectGroupJoinGet(*child, table_index);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

static unique_ptr<BaseStatistics> GetPerfectGroupJoinSourceStats(LogicalOperator &owner,
                                                                 const HashGroupJoinCandidate &candidate,
                                                                 ClientContext &context) {
	if (candidate.owner_key_indices.size() != 1) {
		return nullptr;
	}
	auto bindings = owner.GetColumnBindings();
	if (candidate.owner_key_indices[0] >= bindings.size()) {
		return nullptr;
	}
	auto binding = bindings[candidate.owner_key_indices[0]];
	auto get = FindPerfectGroupJoinGet(owner, binding.table_index);
	if (!get || !get->bind_data || (!get->function.statistics && !get->function.statistics_extended) ||
	    binding.column_index >= get->GetColumnIds().size()) {
		return nullptr;
	}
	auto &column_id = get->GetColumnIndex(binding);
	if (get->function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get->bind_data.get(), column_id);
		return get->function.statistics_extended(context, input);
	}
	return get->function.statistics(context, get->bind_data.get(), column_id.GetPrimaryIndex());
}

static bool TryGetPerfectGroupJoinBounds(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                         const HashGroupJoinCandidate &candidate, ClientContext &context,
                                         Value &minimum, Value &maximum, idx_t &range) {
	auto key_group = GetPerfectGroupJoinKeyGroup(aggregate, candidate);
	if (!key_group.IsValid()) {
		return false;
	}
	auto &key_type = aggregate.groups[key_group.GetIndex()]->GetReturnType();
	if (key_group.GetIndex() < aggregate.group_stats.size() && aggregate.group_stats[key_group.GetIndex()] &&
	    NumericStats::HasMinMax(*aggregate.group_stats[key_group.GetIndex()])) {
		minimum = NumericStats::Min(*aggregate.group_stats[key_group.GetIndex()]);
		maximum = NumericStats::Max(*aggregate.group_stats[key_group.GetIndex()]);
	} else {
		auto source_stats = GetPerfectGroupJoinSourceStats(*join.children[candidate.owner_child], candidate, context);
		if (source_stats && source_stats->GetStatsType() == StatisticsType::NUMERIC_STATS &&
		    NumericStats::HasMinMax(*source_stats)) {
			minimum = NumericStats::Min(*source_stats);
			maximum = NumericStats::Max(*source_stats);
		} else {
			switch (key_type.InternalType()) {
			case PhysicalType::INT8:
			case PhysicalType::INT16:
			case PhysicalType::UINT8:
			case PhysicalType::UINT16:
				minimum = Value::MinimumValue(key_type);
				maximum = Value::MaximumValue(key_type);
				break;
			default:
				return false;
			}
		}
	}
	Value minimum_hugeint = minimum;
	Value maximum_hugeint = maximum;
	if (!minimum_hugeint.DefaultTryCastAs(LogicalType::HUGEINT) ||
	    !maximum_hugeint.DefaultTryCastAs(LogicalType::HUGEINT)) {
		return false;
	}
	hugeint_t range_hugeint;
	if (!TrySubtractOperator::Operation(maximum_hugeint.GetValue<hugeint_t>(), minimum_hugeint.GetValue<hugeint_t>(),
	                                    range_hugeint) ||
	    range_hugeint < 0 || range_hugeint > Hugeint::Convert(PERFECT_GROUP_JOIN_MAX_RANGE)) {
		return false;
	}
	range = NumericCast<idx_t>(range_hugeint);
	return PerfectGroupJoinDirectoryFits(range, BufferManager::GetBufferManager(context).GetMaxMemory());
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
	if (strategy == GroupJoinStrategy::FORCE || strategy == GroupJoinStrategy::HASH ||
	    strategy == GroupJoinStrategy::PERFECT) {
		return candidate;
	}
	if (strategy == GroupJoinStrategy::EAGER) {
		return PhysicalEagerGroupJoinSupported(aggregate, *candidate) ? candidate : nullopt;
	}
	if (strategy == GroupJoinStrategy::INDEX) {
		auto cost = EstimateHashGroupJoinCost(aggregate, join, *candidate, context);
		return cost.index_available ? candidate : nullopt;
	}
	if (strategy != GroupJoinStrategy::AUTO) {
		return nullopt;
	}
	auto cost = EstimateHashGroupJoinCost(aggregate, join, *candidate, context);
	auto execution = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	if ((execution == GroupJoinExecutionMode::AUTO || execution == GroupJoinExecutionMode::INDEX) &&
	    cost.index_selected) {
		aggregate.group_join_auto_selected = true;
		aggregate.group_join_auto_index = true;
		return candidate;
	}
	if (cost.perfect_selected) {
		aggregate.group_join_auto_selected = true;
		return candidate;
	}
	if (cost.physical_eager_selected) {
		aggregate.group_join_auto_selected = true;
		return candidate;
	}
	if (!cost.hash_selected) {
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
	if (strategy == GroupJoinStrategy::DISABLED || strategy == GroupJoinStrategy::SEPARATE ||
	    strategy == GroupJoinStrategy::FACTORIZED ||
	    (strategy == GroupJoinStrategy::AUTO && !aggregate.group_join_auto_selected)) {
		return nullopt;
	}
	auto candidate = TryGetHashGroupJoinCandidate(aggregate, join, context, mode);
	if (candidate && strategy == GroupJoinStrategy::INDEX &&
	    !EstimateHashGroupJoinCost(aggregate, join, *candidate, context).index_available) {
		return nullopt;
	}
	if (candidate && strategy == GroupJoinStrategy::EAGER && !PhysicalEagerGroupJoinSupported(aggregate, *candidate)) {
		return nullopt;
	}
	return candidate;
}

struct FactorizedGroupJoinCandidate {
	idx_t nested_child;
	idx_t driver_child;
	vector<idx_t> driver_key_indices;
	vector<idx_t> left_key_indices;
	vector<idx_t> right_key_indices;
	vector<idx_t> group_driver_indices;
	vector<FactorizedAggregateSource> aggregate_sources;
	bool unique_driver;
	bool routed;
	bool preserve_left;
	bool preserve_right;
	bool semi_left;
	bool semi_right;
};

struct FactorizedGroupJoinCostEstimate {
	idx_t driver_rows = 0;
	idx_t left_rows = 0;
	idx_t right_rows = 0;
	idx_t join_rows = 0;
	idx_t matched_drivers = 0;
	idx_t left_scan_rows = 0;
	idx_t right_scan_rows = 0;
	double build_cost = 0;
	double filter_cost = 0;
	double probe_cost = 0;
	double scan_cost = 0;
	double cache_cost = 0;
	double eager_work_cost = 0;
	double routing_cost = 0;
	double spill_cost = 0;
	double factorized_cost = 0;
	double driver_first_cost = 0;
	double factors_first_cost = 0;
	double separate_cost = 0;
	double eager_cost = 0;
	double best_existing_cost = 0;
	bool reliable = false;
	bool selected = false;
	bool driver_first = true;
};

static constexpr double FACTORIZED_MIN_CACHE_AMORTIZATION_RATIO = 64.0;
static constexpr double FACTORIZED_MAX_CACHE_FACTOR_DRIVER_RATIO = 160.0;

struct FactorizedGroupJoinInputs {
	LogicalComparisonJoin &top_join;
	LogicalComparisonJoin &nested_join;
	LogicalOperator &driver;
	LogicalOperator &left_factor;
	LogicalOperator &right_factor;
};

static bool IsFactorizedEqualityJoin(const LogicalComparisonJoin &join) {
	return (join.join_type == JoinType::INNER || join.join_type == JoinType::LEFT ||
	        join.join_type == JoinType::RIGHT || join.join_type == JoinType::SEMI ||
	        join.join_type == JoinType::RIGHT_SEMI) &&
	       join.children.size() == 2 && !join.conditions.empty() && !join.HasArbitraryConditions();
}

static optional<bool> FactorizedEdgePreservesDriver(const LogicalComparisonJoin &join, idx_t driver_child, bool &semi) {
	semi = false;
	if (join.join_type == JoinType::INNER) {
		return false;
	}
	if ((join.join_type == JoinType::SEMI && driver_child == 0) ||
	    (join.join_type == JoinType::RIGHT_SEMI && driver_child == 1)) {
		semi = true;
		return false;
	}
	if ((join.join_type == JoinType::LEFT && driver_child == 0) ||
	    (join.join_type == JoinType::RIGHT && driver_child == 1)) {
		return true;
	}
	return nullopt;
}

static bool ExtractDirectFactorizedEdge(LogicalComparisonJoin &join, idx_t driver_child, vector<idx_t> &driver_keys,
                                        vector<idx_t> &factor_keys, bool &preserve_driver, bool &semi) {
	auto preserve = FactorizedEdgePreservesDriver(join, driver_child, semi);
	if (!preserve) {
		return false;
	}
	preserve_driver = *preserve;
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
		    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType()) {
			return false;
		}
		auto &driver_expression = driver_child == 0 ? condition.GetLHS() : condition.GetRHS();
		auto &factor_expression = driver_child == 0 ? condition.GetRHS() : condition.GetLHS();
		auto driver_index = GetDirectReferenceIndex(driver_expression, *join.children[driver_child]);
		auto factor_index = GetDirectReferenceIndex(factor_expression, *join.children[1 - driver_child]);
		if (!driver_index.IsValid() || !factor_index.IsValid()) {
			return false;
		}
		driver_keys.push_back(driver_index.GetIndex());
		factor_keys.push_back(factor_index.GetIndex());
	}
	return true;
}

static bool ExtractOuterFactorizedEdge(LogicalComparisonJoin &join, idx_t nested_child,
                                       LogicalComparisonJoin &nested_join, idx_t driver_child,
                                       const vector<idx_t> &driver_keys, const vector<idx_t> &left_factor_keys,
                                       vector<idx_t> &right_factor_keys, bool &preserve_driver, bool &semi) {
	auto preserve = FactorizedEdgePreservesDriver(join, nested_child, semi);
	if (!preserve) {
		return false;
	}
	preserve_driver = *preserve;
	vector<optional_idx> mapped_right_keys(driver_keys.size());
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
		    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType()) {
			return false;
		}
		auto &nested_expression = nested_child == 0 ? condition.GetLHS() : condition.GetRHS();
		auto &factor_expression = nested_child == 0 ? condition.GetRHS() : condition.GetLHS();
		auto driver_index = GetDirectReferenceIndex(nested_expression, *nested_join.children[driver_child]);
		auto left_factor_index = GetDirectReferenceIndex(nested_expression, *nested_join.children[1 - driver_child]);
		auto factor_index = GetDirectReferenceIndex(factor_expression, *join.children[1 - nested_child]);
		if (driver_index.IsValid() == left_factor_index.IsValid() || !factor_index.IsValid()) {
			return false;
		}
		auto &nested_keys = driver_index.IsValid() ? driver_keys : left_factor_keys;
		const auto nested_key = driver_index.IsValid() ? driver_index.GetIndex() : left_factor_index.GetIndex();
		optional_idx pair_index;
		for (idx_t key_idx = 0; key_idx < nested_keys.size(); key_idx++) {
			if (nested_keys[key_idx] != nested_key) {
				continue;
			}
			if (pair_index.IsValid()) {
				return false;
			}
			pair_index = optional_idx(key_idx);
		}
		if (!pair_index.IsValid()) {
			return false;
		}
		auto &mapped_right = mapped_right_keys[pair_index.GetIndex()];
		if (mapped_right.IsValid() && mapped_right.GetIndex() != factor_index.GetIndex()) {
			return false;
		}
		mapped_right = factor_index;
	}
	for (auto &right_key : mapped_right_keys) {
		if (!right_key.IsValid()) {
			return false;
		}
		right_factor_keys.push_back(right_key.GetIndex());
	}
	return true;
}

static uint8_t GetFactorizedExpressionSources(const Expression &expression, const vector<ColumnBinding> (&bindings)[3],
                                              bool &valid) {
	if (!valid) {
		return 0;
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
		uint8_t result = 0;
		for (idx_t source_idx = 0; source_idx < 3; source_idx++) {
			if (std::find(bindings[source_idx].begin(), bindings[source_idx].end(), binding) !=
			    bindings[source_idx].end()) {
				result |= NumericCast<uint8_t>(1U << source_idx);
			}
		}
		valid = result != 0 && (result & (result - 1)) == 0;
		return result;
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		valid = false;
		return 0;
	}
	uint8_t result = 0;
	ExpressionIterator::EnumerateChildren(
	    expression, [&](const Expression &child) { result |= GetFactorizedExpressionSources(child, bindings, valid); });
	return result;
}

static optional<FactorizedAggregateSource> GetFactorizedAggregateSource(const BoundAggregateExpression &aggregate,
                                                                        const vector<ColumnBinding> (&bindings)[3]) {
	if (aggregate.IsVolatile() ||
	    (!aggregate.GetOrderBys() &&
	     (aggregate.Function().GetOrderDependent() == AggregateOrderDependent::ORDER_DEPENDENT ||
	      aggregate.Function().GetRepeatCombine() != AggregateRepeatCombine::SUPPORTED))) {
		return nullopt;
	}
	uint8_t sources = 0;
	bool valid = true;
	for (auto &child : aggregate.GetChildren()) {
		if (child->IsVolatile() || child->GetReturnType().IsAggregateState()) {
			return nullopt;
		}
		sources |= GetFactorizedExpressionSources(*child, bindings, valid);
	}
	if (aggregate.GetOrderBys()) {
		for (auto &order : aggregate.GetOrderBys()->orders) {
			if (order.expression->IsVolatile()) {
				return nullopt;
			}
			sources |= GetFactorizedExpressionSources(*order.expression, bindings, valid);
		}
	}
	if (aggregate.GetFilter()) {
		if (aggregate.GetFilter()->IsVolatile()) {
			return nullopt;
		}
		sources |= GetFactorizedExpressionSources(*aggregate.GetFilter(), bindings, valid);
	}
	if (!valid || (sources & (sources - 1)) != 0) {
		return nullopt;
	}
	if (sources == 0 || sources == 1) {
		return FactorizedAggregateSource::DRIVER;
	}
	if (sources == 2) {
		return FactorizedAggregateSource::LEFT_FACTOR;
	}
	if (sources == 4) {
		return FactorizedAggregateSource::RIGHT_FACTOR;
	}
	return nullopt;
}

static optional<FactorizedGroupJoinCandidate> ValidateFactorizedGroupJoinCandidate(
    const vector<unique_ptr<Expression>> &groups, const vector<unique_ptr<Expression>> &expressions,
    LogicalOperator &driver, LogicalOperator &left_factor, LogicalOperator &right_factor, idx_t nested_child,
    idx_t driver_child, vector<idx_t> driver_keys, vector<idx_t> left_keys, vector<idx_t> right_keys,
    bool preserve_left, bool preserve_right, bool semi_left, bool semi_right, bool require_all_driver_keys) {
	const auto unique_driver = GetUniqueKeyProperty(driver, driver_keys).has_value();
	unordered_set<idx_t> group_key_pairs;
	vector<idx_t> group_driver_indices;
	bool routed = false;
	for (auto &group : groups) {
		if (group->IsVolatile()) {
			return nullopt;
		}
		optional_idx pair_index;
		optional_idx direct_driver_index;
		for (idx_t source_idx = 0; source_idx < 3; source_idx++) {
			auto &source = source_idx == 0 ? driver : source_idx == 1 ? left_factor : right_factor;
			auto &source_keys = source_idx == 0 ? driver_keys : source_idx == 1 ? left_keys : right_keys;
			auto source_index = GetDirectReferenceIndex(*group, source);
			if (!source_index.IsValid()) {
				continue;
			}
			if (source_idx == 0) {
				direct_driver_index = source_index;
			}
			if ((source_idx == 1 && (preserve_left || semi_left)) ||
			    (source_idx == 2 && (preserve_right || semi_right))) {
				return nullopt;
			}
			for (idx_t key_idx = 0; key_idx < source_keys.size(); key_idx++) {
				if (source_keys[key_idx] != source_index.GetIndex()) {
					continue;
				}
				if (pair_index.IsValid()) {
					return nullopt;
				}
				pair_index = optional_idx(key_idx);
			}
		}
		if (pair_index.IsValid()) {
			if (!group_key_pairs.insert(pair_index.GetIndex()).second) {
				return nullopt;
			}
			group_driver_indices.push_back(driver_keys[pair_index.GetIndex()]);
		} else {
			if (!direct_driver_index.IsValid()) {
				return nullopt;
			}
			group_driver_indices.push_back(direct_driver_index.GetIndex());
			routed = true;
		}
	}
	if (require_all_driver_keys && group_key_pairs.size() != driver_keys.size()) {
		return nullopt;
	}
	vector<ColumnBinding> source_bindings[3] = {driver.GetColumnBindings(), left_factor.GetColumnBindings(),
	                                            right_factor.GetColumnBindings()};
	vector<FactorizedAggregateSource> aggregate_sources;
	for (auto &expression : expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return nullopt;
		}
		auto source = GetFactorizedAggregateSource(expression->Cast<BoundAggregateExpression>(), source_bindings);
		if (!source) {
			return nullopt;
		}
		if ((*source == FactorizedAggregateSource::LEFT_FACTOR && semi_left) ||
		    (*source == FactorizedAggregateSource::RIGHT_FACTOR && semi_right)) {
			return nullopt;
		}
		aggregate_sources.push_back(*source);
	}
	return FactorizedGroupJoinCandidate {nested_child,
	                                     driver_child,
	                                     std::move(driver_keys),
	                                     std::move(left_keys),
	                                     std::move(right_keys),
	                                     std::move(group_driver_indices),
	                                     std::move(aggregate_sources),
	                                     unique_driver,
	                                     routed,
	                                     preserve_left,
	                                     preserve_right,
	                                     semi_left,
	                                     semi_right};
}

static optional<FactorizedGroupJoinCandidate>
AnalyzeFactorizedGroupJoinCandidate(const vector<unique_ptr<Expression>> &groups,
                                    const vector<unique_ptr<Expression>> &expressions, LogicalComparisonJoin &top_join,
                                    bool require_all_driver_keys = true) {
	if (!IsFactorizedEqualityJoin(top_join)) {
		return nullopt;
	}
	optional<FactorizedGroupJoinCandidate> fallback;
	for (idx_t nested_child = 0; nested_child < 2; nested_child++) {
		if (top_join.children[nested_child]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			continue;
		}
		auto &nested_join = top_join.children[nested_child]->Cast<LogicalComparisonJoin>();
		if (!IsFactorizedEqualityJoin(nested_join)) {
			continue;
		}
		for (idx_t driver_child = 0; driver_child < 2; driver_child++) {
			vector<idx_t> inner_driver_keys;
			vector<idx_t> left_factor_keys;
			vector<idx_t> right_factor_keys;
			bool preserve_left;
			bool preserve_right;
			bool semi_left;
			bool semi_right;
			if (!ExtractDirectFactorizedEdge(nested_join, driver_child, inner_driver_keys, left_factor_keys,
			                                 preserve_left, semi_left) ||
			    !ExtractOuterFactorizedEdge(top_join, nested_child, nested_join, driver_child, inner_driver_keys,
			                                left_factor_keys, right_factor_keys, preserve_right, semi_right)) {
				continue;
			}
			auto candidate = ValidateFactorizedGroupJoinCandidate(
			    groups, expressions, *nested_join.children[driver_child], *nested_join.children[1 - driver_child],
			    *top_join.children[1 - nested_child], nested_child, driver_child, std::move(inner_driver_keys),
			    std::move(left_factor_keys), std::move(right_factor_keys), preserve_left, preserve_right, semi_left,
			    semi_right, require_all_driver_keys);
			if (candidate) {
				if (candidate->unique_driver) {
					return candidate;
				}
				if (!fallback) {
					fallback = std::move(candidate);
				}
			}
		}

		vector<idx_t> nested_keys[2];
		bool nested_preserves_left;
		bool nested_semi_left;
		if (top_join.join_type != JoinType::INNER ||
		    !ExtractDirectFactorizedEdge(nested_join, 0, nested_keys[0], nested_keys[1], nested_preserves_left,
		                                 nested_semi_left) ||
		    nested_join.join_type != JoinType::INNER) {
			continue;
		}
		vector<optional_idx> pair_driver_keys(nested_keys[0].size());
		bool valid = true;
		for (auto &condition : top_join.conditions) {
			auto &nested_expression = nested_child == 0 ? condition.GetLHS() : condition.GetRHS();
			auto &driver_expression = nested_child == 0 ? condition.GetRHS() : condition.GetLHS();
			auto driver_index = GetDirectReferenceIndex(driver_expression, *top_join.children[1 - nested_child]);
			auto nested_left_index = GetDirectReferenceIndex(nested_expression, *nested_join.children[0]);
			auto nested_right_index = GetDirectReferenceIndex(nested_expression, *nested_join.children[1]);
			if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
			    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType() || !driver_index.IsValid() ||
			    nested_left_index.IsValid() == nested_right_index.IsValid()) {
				valid = false;
				break;
			}
			const auto factor_child = nested_left_index.IsValid() ? idx_t(0) : idx_t(1);
			const auto factor_index = factor_child == 0 ? nested_left_index.GetIndex() : nested_right_index.GetIndex();
			optional_idx pair_index;
			for (idx_t key_idx = 0; key_idx < nested_keys[factor_child].size(); key_idx++) {
				if (nested_keys[factor_child][key_idx] != factor_index) {
					continue;
				}
				if (pair_index.IsValid()) {
					valid = false;
					break;
				}
				pair_index = optional_idx(key_idx);
			}
			if (!valid || !pair_index.IsValid()) {
				valid = false;
				break;
			}
			auto &mapped_driver = pair_driver_keys[pair_index.GetIndex()];
			if (mapped_driver.IsValid() && mapped_driver.GetIndex() != driver_index.GetIndex()) {
				valid = false;
				break;
			}
			mapped_driver = driver_index;
		}
		vector<idx_t> driver_keys;
		for (auto &driver_key : pair_driver_keys) {
			if (!driver_key.IsValid()) {
				valid = false;
				break;
			}
			driver_keys.push_back(driver_key.GetIndex());
		}
		if (!valid) {
			continue;
		}
		auto candidate = ValidateFactorizedGroupJoinCandidate(
		    groups, expressions, *top_join.children[1 - nested_child], *nested_join.children[0],
		    *nested_join.children[1], nested_child, DConstants::INVALID_INDEX, std::move(driver_keys),
		    std::move(nested_keys[0]), std::move(nested_keys[1]), false, false, false, false, require_all_driver_keys);
		if (candidate) {
			if (candidate->unique_driver) {
				return candidate;
			}
			if (!fallback) {
				fallback = std::move(candidate);
			}
		}
	}
	return fallback;
}

static void InlineFactorizedProjection(unique_ptr<Expression> &expression, const LogicalProjection &projection) {
	if (expression->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto &reference = expression->Cast<BoundColumnRefExpression>();
		if (reference.Binding().table_index == projection.table_index) {
			auto projection_index = reference.Binding().column_index.GetIndex();
			if (projection_index >= projection.expressions.size()) {
				throw InternalException("Factorized GroupJoin projection reference is out of bounds");
			}
			expression = projection.expressions[projection_index]->Copy();
			return;
		}
	}
	ExpressionIterator::EnumerateChildren(
	    *expression, [&](unique_ptr<Expression> &child) { InlineFactorizedProjection(child, projection); });
}

static optional_ptr<LogicalComparisonJoin> GetFactorizedJoinRoot(LogicalAggregate &aggregate,
                                                                 vector<reference<LogicalProjection>> &projections) {
	if (aggregate.children.size() != 1) {
		return nullptr;
	}
	reference<LogicalOperator> current = *aggregate.children[0];
	while (current.get().type == LogicalOperatorType::LOGICAL_PROJECTION && current.get().children.size() == 1) {
		auto &projection = current.get().Cast<LogicalProjection>();
		for (auto &expression : projection.expressions) {
			if (expression->IsVolatile()) {
				return nullptr;
			}
		}
		projections.push_back(projection);
		current = *projection.children[0];
	}
	if (current.get().type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return nullptr;
	}
	return current.get().Cast<LogicalComparisonJoin>();
}

static void InlineFactorizedProjections(vector<unique_ptr<Expression>> &groups,
                                        vector<unique_ptr<Expression>> &expressions,
                                        const vector<reference<LogicalProjection>> &projections) {
	for (auto &group : groups) {
		for (auto &projection : projections) {
			InlineFactorizedProjection(group, projection);
		}
	}
	for (auto &expression : expressions) {
		for (auto &projection : projections) {
			InlineFactorizedProjection(expression, projection);
		}
	}
}

static vector<unique_ptr<Expression>> CopyFactorizedExpressions(const vector<unique_ptr<Expression>> &expressions) {
	vector<unique_ptr<Expression>> result;
	result.reserve(expressions.size());
	for (auto &expression : expressions) {
		result.push_back(expression->Copy());
	}
	return result;
}

static optional<FactorizedGroupJoinCandidate>
TryAnalyzeFactorizedGroupJoinCandidate(LogicalAggregate &aggregate, ClientContext &context,
                                       bool require_all_driver_keys = true) {
	if (!HashGroupJoinPlanningEnabled(context) || aggregate.groups.empty() || !aggregate.grouping_functions.empty() ||
	    aggregate.grouping_sets.size() > 1 || aggregate.expressions.empty()) {
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

	vector<reference<LogicalProjection>> projections;
	auto top_join = GetFactorizedJoinRoot(aggregate, projections);
	if (!top_join) {
		return nullopt;
	}
	if (projections.empty()) {
		return AnalyzeFactorizedGroupJoinCandidate(aggregate.groups, aggregate.expressions, *top_join,
		                                           require_all_driver_keys);
	}
	auto groups = CopyFactorizedExpressions(aggregate.groups);
	auto expressions = CopyFactorizedExpressions(aggregate.expressions);
	InlineFactorizedProjections(groups, expressions, projections);
	return AnalyzeFactorizedGroupJoinCandidate(groups, expressions, *top_join, require_all_driver_keys);
}

static void ApplyFactorizedProjectionInlining(LogicalAggregate &aggregate) {
	vector<reference<LogicalProjection>> projections;
	auto top_join = GetFactorizedJoinRoot(aggregate, projections);
	D_ASSERT(top_join);
	if (projections.empty()) {
		return;
	}
	InlineFactorizedProjections(aggregate.groups, aggregate.expressions, projections);
	aggregate.children[0] = std::move(projections.back().get().children[0]);
}

static unique_ptr<BaseStatistics>
GetFactorizedPerfectGroupJoinSourceStats(LogicalOperator &driver, const FactorizedGroupJoinCandidate &candidate,
                                         ClientContext &context) {
	if (candidate.driver_key_indices.size() != 1) {
		return nullptr;
	}
	auto bindings = driver.GetColumnBindings();
	if (candidate.driver_key_indices[0] >= bindings.size()) {
		return nullptr;
	}
	auto binding = bindings[candidate.driver_key_indices[0]];
	auto get = FindPerfectGroupJoinGet(driver, binding.table_index);
	if (!get || !get->bind_data || (!get->function.statistics && !get->function.statistics_extended) ||
	    binding.column_index >= get->GetColumnIds().size()) {
		return nullptr;
	}
	auto &column_id = get->GetColumnIndex(binding);
	if (get->function.statistics_extended) {
		TableFunctionGetStatisticsInput input(get->bind_data.get(), column_id);
		return get->function.statistics_extended(context, input);
	}
	return get->function.statistics(context, get->bind_data.get(), column_id.GetPrimaryIndex());
}

static bool TryGetFactorizedPerfectGroupJoinBounds(LogicalAggregate &aggregate, LogicalOperator &driver,
                                                   const FactorizedGroupJoinCandidate &candidate,
                                                   ClientContext &context, Value &minimum, Value &maximum,
                                                   idx_t &range) {
	if (!candidate.unique_driver || candidate.routed || candidate.driver_key_indices.size() != 1) {
		return false;
	}
	auto key_group = std::find(candidate.group_driver_indices.begin(), candidate.group_driver_indices.end(),
	                           candidate.driver_key_indices[0]);
	if (key_group == candidate.group_driver_indices.end()) {
		return false;
	}
	auto group_idx = NumericCast<idx_t>(key_group - candidate.group_driver_indices.begin());
	if (group_idx >= aggregate.groups.size()) {
		return false;
	}
	auto &key_type = aggregate.groups[group_idx]->GetReturnType();
	switch (key_type.InternalType()) {
	case PhysicalType::INT8:
	case PhysicalType::INT16:
	case PhysicalType::INT32:
	case PhysicalType::INT64:
	case PhysicalType::UINT8:
	case PhysicalType::UINT16:
	case PhysicalType::UINT32:
	case PhysicalType::UINT64:
		break;
	default:
		return false;
	}
	if (group_idx < aggregate.group_stats.size() && aggregate.group_stats[group_idx] &&
	    NumericStats::HasMinMax(*aggregate.group_stats[group_idx])) {
		minimum = NumericStats::Min(*aggregate.group_stats[group_idx]);
		maximum = NumericStats::Max(*aggregate.group_stats[group_idx]);
	} else {
		auto source_stats = GetFactorizedPerfectGroupJoinSourceStats(driver, candidate, context);
		if (!source_stats || source_stats->GetStatsType() != StatisticsType::NUMERIC_STATS ||
		    !NumericStats::HasMinMax(*source_stats)) {
			return false;
		}
		minimum = NumericStats::Min(*source_stats);
		maximum = NumericStats::Max(*source_stats);
	}
	Value minimum_hugeint = minimum;
	Value maximum_hugeint = maximum;
	if (!minimum_hugeint.DefaultTryCastAs(LogicalType::HUGEINT) ||
	    !maximum_hugeint.DefaultTryCastAs(LogicalType::HUGEINT)) {
		return false;
	}
	hugeint_t range_hugeint;
	if (!TrySubtractOperator::Operation(maximum_hugeint.GetValue<hugeint_t>(), minimum_hugeint.GetValue<hugeint_t>(),
	                                    range_hugeint) ||
	    range_hugeint < 0 || range_hugeint > Hugeint::Convert(PERFECT_GROUP_JOIN_MAX_RANGE)) {
		return false;
	}
	range = NumericCast<idx_t>(range_hugeint);
	return PerfectGroupJoinDirectoryFits(range, BufferManager::GetBufferManager(context).GetMaxMemory());
}

static FactorizedGroupJoinInputs GetFactorizedGroupJoinInputs(LogicalAggregate &aggregate,
                                                              const FactorizedGroupJoinCandidate &candidate) {
	vector<reference<LogicalProjection>> projections;
	auto top_join_ptr = GetFactorizedJoinRoot(aggregate, projections);
	if (!top_join_ptr ||
	    top_join_ptr->children[candidate.nested_child]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		throw InternalException("Factorized GroupJoin candidate no longer references a two-join star");
	}
	auto &top_join = *top_join_ptr;
	auto &nested_join = top_join.children[candidate.nested_child]->Cast<LogicalComparisonJoin>();
	if (candidate.driver_child == DConstants::INVALID_INDEX) {
		return {top_join, nested_join, *top_join.children[1 - candidate.nested_child], *nested_join.children[0],
		        *nested_join.children[1]};
	}
	return {top_join, nested_join, *nested_join.children[candidate.driver_child],
	        *nested_join.children[1 - candidate.driver_child], *top_join.children[1 - candidate.nested_child]};
}

optional<FactorizedCoarseGroupInfo> GetFactorizedCoarseGroupInfo(LogicalAggregate &aggregate, ClientContext &context) {
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy != GroupJoinStrategy::FACTORIZED && strategy != GroupJoinStrategy::AUTO) {
		return nullopt;
	}
	auto candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context, false);
	if (!candidate || !candidate->unique_driver) {
		return nullopt;
	}
	auto inputs = GetFactorizedGroupJoinInputs(aggregate, *candidate);
	inputs.driver.ResolveOperatorTypes();
	auto driver_bindings = inputs.driver.GetColumnBindings();
	FactorizedCoarseGroupInfo result;
	for (auto driver_key : candidate->driver_key_indices) {
		if (std::find(candidate->group_driver_indices.begin(), candidate->group_driver_indices.end(), driver_key) !=
		    candidate->group_driver_indices.end()) {
			continue;
		}
		if (driver_key >= driver_bindings.size() || driver_key >= inputs.driver.types.size()) {
			return nullopt;
		}
		result.missing_driver_keys.push_back(driver_bindings[driver_key]);
		result.missing_driver_key_types.push_back(inputs.driver.types[driver_key]);
	}
	if (result.missing_driver_keys.empty()) {
		return nullopt;
	}
	result.estimated_driver_rows = inputs.driver.EstimateCardinality(context);
	return result;
}

static idx_t FactorizedCardinalityFromDouble(double value) {
	if (value >= static_cast<double>(NumericLimits<idx_t>::Maximum())) {
		return NumericLimits<idx_t>::Maximum();
	}
	return LossyNumericCast<idx_t>(MaxValue<double>(value, 0));
}

static bool AutoFactorizedGroupJoinAggregatesSupported(const LogicalAggregate &aggregate,
                                                       const FactorizedGroupJoinCandidate &candidate,
                                                       array<idx_t, 3> &state_widths, array<idx_t, 3> &payload_widths) {
	static constexpr idx_t MAX_AUTO_STATE_WIDTH = 256;
	if (aggregate.expressions.size() != candidate.aggregate_sources.size()) {
		return false;
	}
	state_widths.fill(sizeof(idx_t));
	payload_widths.fill(0);
	idx_t total_state_width = 0;
	for (idx_t aggregate_idx = 0; aggregate_idx < aggregate.expressions.size(); aggregate_idx++) {
		auto &aggr = aggregate.expressions[aggregate_idx]->Cast<BoundAggregateExpression>();
		auto &callbacks = aggr.Function().GetCallbacks();
		if (aggr.IsDistinct() || aggr.GetFilter() || aggr.GetOrderBys() || callbacks.HasStateDestructorCallback() ||
		    aggr.Function().GetName() == "combine_aggr") {
			return false;
		}
		auto source_idx = NumericCast<idx_t>(candidate.aggregate_sources[aggregate_idx]);
		if (source_idx >= state_widths.size()) {
			return false;
		}
		state_widths[source_idx] += aggr.Function().GetStateSize(aggr.BindInfo().get());
		for (auto &child : aggr.GetChildren()) {
			if (child->GetReturnType().IsAggregateState()) {
				return false;
			}
			payload_widths[source_idx] += GetTypeIdSize(child->GetReturnType().InternalType());
		}
	}
	for (auto width : state_widths) {
		if (!TryAddOperator::Operation(total_state_width, width, total_state_width)) {
			return false;
		}
	}
	return total_state_width <= MAX_AUTO_STATE_WIDTH;
}

static FactorizedGroupJoinCostEstimate EstimateFactorizedGroupJoinCost(LogicalAggregate &aggregate,
                                                                       const FactorizedGroupJoinCandidate &candidate,
                                                                       ClientContext &context) {
	static constexpr double AUTO_MAX_COST_RATIO = 0.9;
	static constexpr double FILTER_ROW_COST = 0.25;
	static constexpr double DIRECTORY_PROBE_COST = 0.5;
	static constexpr double CACHE_AMORTIZATION_COST = 1.5;
	static constexpr double CACHE_PRESSURE_COST = 2.0;
	static constexpr bool ENABLE_FACTORIZED_AUTO = true;

	FactorizedGroupJoinCostEstimate result;
	auto inputs = GetFactorizedGroupJoinInputs(aggregate, candidate);
	result.driver_rows = inputs.driver.EstimateCardinality(context);
	result.left_rows = inputs.left_factor.EstimateCardinality(context);
	result.right_rows = inputs.right_factor.EstimateCardinality(context);
	inputs.nested_join.EstimateCardinality(context);
	result.join_rows = inputs.top_join.EstimateCardinality(context);
	result.matched_drivers = aggregate.EstimateCardinality(context);
	result.matched_drivers = MinValue(result.matched_drivers, result.driver_rows);

	array<idx_t, 3> state_widths;
	array<idx_t, 3> payload_widths;
	const auto aggregates_supported =
	    AutoFactorizedGroupJoinAggregatesSupported(aggregate, candidate, state_widths, payload_widths);
	idx_t key_width = 0;
	for (auto key_idx : candidate.driver_key_indices) {
		if (key_idx >= inputs.driver.types.size()) {
			return result;
		}
		key_width += GetTypeIdSize(inputs.driver.types[key_idx].InternalType());
	}
	const auto direct_driver_edge = candidate.driver_child != DConstants::INVALID_INDEX;
	const auto inner_edges =
	    inputs.nested_join.join_type == JoinType::INNER && inputs.top_join.join_type == JoinType::INNER;
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range;
	const auto exact_driver_filter = TryGetFactorizedPerfectGroupJoinBounds(
	    aggregate, inputs.driver, candidate, context, perfect_min, perfect_max, perfect_range);
	if (direct_driver_edge) {
		result.left_scan_rows = MinValue(result.left_rows, inputs.nested_join.estimated_cardinality);
		const auto left_fanout = static_cast<double>(result.left_scan_rows) /
		                         static_cast<double>(MaxValue<idx_t>(result.matched_drivers, 1));
		result.right_scan_rows = MinValue(
		    result.right_rows, FactorizedCardinalityFromDouble(std::ceil(static_cast<double>(result.join_rows) /
		                                                                 MaxValue<double>(left_fanout, 1.0))));
	} else {
		// The factor-factor-first orientation does not provide reliable branch-local match estimates.
		result.left_scan_rows = result.left_rows;
		result.right_scan_rows = result.right_rows;
	}
	const auto factor_rows = static_cast<double>(result.left_rows) + static_cast<double>(result.right_rows);
	const auto scan_rows = static_cast<double>(result.left_scan_rows) + static_cast<double>(result.right_scan_rows);
	const auto compact_driver_domain = static_cast<double>(result.driver_rows) <= factor_rows / 10.0;
	const auto sparse_driver_domain =
	    exact_driver_filter &&
	    static_cast<double>(perfect_range) >= static_cast<double>(MaxValue<idx_t>(result.driver_rows, 1)) * 2.0;
	const auto selective_driver = compact_driver_domain && scan_rows <= factor_rows / 4.0;
	result.reliable = direct_driver_edge && inner_edges && (exact_driver_filter || selective_driver) &&
	                  aggregates_supported && inputs.driver.has_estimated_cardinality &&
	                  inputs.left_factor.has_estimated_cardinality && inputs.right_factor.has_estimated_cardinality &&
	                  inputs.nested_join.has_estimated_cardinality && inputs.top_join.has_estimated_cardinality &&
	                  aggregate.has_estimated_cardinality;

	const auto key_cost = MaxValue<double>(static_cast<double>(key_width) / 8.0, 1.0);
	array<double, 3> state_costs;
	array<double, 3> payload_costs;
	for (idx_t source_idx = 0; source_idx < state_costs.size(); source_idx++) {
		state_costs[source_idx] = MaxValue<double>(static_cast<double>(state_widths[source_idx]) / 16.0, 1.0);
		payload_costs[source_idx] = static_cast<double>(payload_widths[source_idx]) / 16.0;
	}
	result.build_cost = static_cast<double>(result.driver_rows) * (key_cost + state_costs[0] + payload_costs[0] + 1.0);
	result.filter_cost = static_cast<double>(result.driver_rows) * 2.0 * key_cost +
	                     static_cast<double>(result.left_rows + result.right_rows) * FILTER_ROW_COST * key_cost;
	result.probe_cost =
	    static_cast<double>(result.left_scan_rows + result.right_scan_rows) * DIRECTORY_PROBE_COST * key_cost;
	result.scan_cost = static_cast<double>(result.left_scan_rows) * (payload_costs[1] + state_costs[1]) +
	                   static_cast<double>(result.right_scan_rows) * (payload_costs[2] + state_costs[2]);
	const auto thread_count = MaxValue<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads(), 1);
	result.cache_cost =
	    static_cast<double>(result.driver_rows) * static_cast<double>(thread_count) * (state_costs[1] + state_costs[2]);
	const auto direct_state_cost = result.cache_cost;
	const auto minimum_amortized_factor_rows =
	    static_cast<double>(result.driver_rows) * FACTORIZED_MIN_CACHE_AMORTIZATION_RATIO;
	const auto cached_factor_rows = static_cast<double>(result.driver_rows) * FACTORIZED_MAX_CACHE_FACTOR_DRIVER_RATIO;
	const auto total_factor_rows = static_cast<double>(result.left_rows + result.right_rows);
	if (total_factor_rows < minimum_amortized_factor_rows) {
		result.cache_cost += (minimum_amortized_factor_rows - total_factor_rows) * CACHE_AMORTIZATION_COST * key_cost;
	}
	if (total_factor_rows > cached_factor_rows) {
		result.cache_cost += (total_factor_rows - cached_factor_rows) * CACHE_PRESSURE_COST * key_cost;
	}
	if (!candidate.unique_driver || candidate.routed) {
		result.eager_work_cost =
		    static_cast<double>(result.left_scan_rows) * (key_cost + payload_costs[1] + state_costs[1]) +
		    static_cast<double>(result.right_scan_rows) * (key_cost + payload_costs[2] + state_costs[2]);
	}
	if (candidate.routed) {
		result.routing_cost = static_cast<double>(result.driver_rows + result.matched_drivers) * (key_cost + 2.0);
	}
	result.driver_first_cost = result.build_cost + result.filter_cost + result.probe_cost + result.scan_cost +
	                           direct_state_cost + result.routing_cost;
	const auto driver_first_orientation_cost = result.driver_first_cost;
	const auto retained_factor_rows =
	    static_cast<double>(result.left_scan_rows) + static_cast<double>(result.right_scan_rows);
	const auto factor_row_count = static_cast<double>(result.left_rows) + static_cast<double>(result.right_rows);
	if (retained_factor_rows > factor_row_count / 4.0) {
		result.driver_first_cost += (retained_factor_rows - factor_row_count / 4.0) * 1.5;
	}
	result.factors_first_cost = result.build_cost + result.filter_cost + result.probe_cost + result.scan_cost +
	                            result.cache_cost + result.eager_work_cost + result.routing_cost;

	const auto intermediate_rows =
	    direct_driver_edge ? inputs.nested_join.estimated_cardinality : result.left_rows + result.right_rows;
	const auto total_payload_cost = payload_costs[0] + payload_costs[1] + payload_costs[2];
	const auto total_state_cost = state_costs[0] + state_costs[1] + state_costs[2];
	const auto separate_factor_rows = direct_driver_edge
	                                      ? static_cast<double>(result.left_scan_rows) + result.right_scan_rows
	                                      : static_cast<double>(result.left_rows) + result.right_rows;
	result.separate_cost = (static_cast<double>(result.driver_rows) + separate_factor_rows) * key_cost +
	                       static_cast<double>(intermediate_rows) * (key_cost + 1.0) +
	                       static_cast<double>(result.join_rows) * (total_payload_cost + total_state_cost + 2.0);
	result.eager_cost = static_cast<double>(result.left_rows) * (key_cost + payload_costs[1] + state_costs[1] + 1.0) +
	                    static_cast<double>(result.right_rows) * (key_cost + payload_costs[2] + state_costs[2] + 1.0) +
	                    result.build_cost + static_cast<double>(result.matched_drivers) * (key_cost + 1.0);
	result.best_existing_cost = MinValue(result.separate_cost, result.eager_cost);
	const auto driver_grain_state_export =
	    candidate.unique_driver && candidate.routed &&
	    std::all_of(aggregate.expressions.begin(), aggregate.expressions.end(),
	                [](const unique_ptr<Expression> &expression) {
		                return expression->Cast<BoundAggregateExpression>().StateExportMode() ==
		                       AggregateStateExportMode::STATE_EXPORT;
	                });
	result.driver_first = driver_grain_state_export ? false
	                      : compact_driver_domain && sparse_driver_domain
	                          ? driver_first_orientation_cost <= result.factors_first_cost
	                          : result.driver_first_cost <= result.factors_first_cost;
	result.factorized_cost = MinValue(result.driver_first_cost, result.factors_first_cost);

	idx_t estimated_memory_rows = 0;
	idx_t estimated_memory = 0;
	const auto row_bytes =
	    key_width + state_widths[0] + state_widths[1] + state_widths[2] + sizeof(hash_t) + sizeof(data_ptr_t) * 2;
	if (!TryMultiplyOperator::Operation(result.driver_rows, thread_count, estimated_memory_rows) ||
	    !TryMultiplyOperator::Operation(estimated_memory_rows, row_bytes, estimated_memory) ||
	    estimated_memory > BufferManager::GetBufferManager(context).GetMaxMemory() / 4) {
		result.spill_cost = result.factorized_cost * 0.75;
		result.factorized_cost += result.spill_cost;
	}
	result.selected = ENABLE_FACTORIZED_AUTO && result.reliable && result.best_existing_cost > 0 &&
	                  result.factorized_cost <= result.best_existing_cost * AUTO_MAX_COST_RATIO;
	return result;
}

bool HasPotentialFactorizedGroupJoinCandidate(LogicalAggregate &aggregate, ClientContext &context) {
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy != GroupJoinStrategy::FACTORIZED && strategy != GroupJoinStrategy::AUTO) {
		return false;
	}
	auto candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context);
	if (!candidate || strategy == GroupJoinStrategy::FACTORIZED) {
		return candidate.has_value();
	}
	auto inputs = GetFactorizedGroupJoinInputs(aggregate, *candidate);
	array<idx_t, 3> state_widths;
	array<idx_t, 3> payload_widths;
	const auto inner_edges =
	    inputs.nested_join.join_type == JoinType::INNER && inputs.top_join.join_type == JoinType::INNER;
	if (candidate->driver_child == DConstants::INVALID_INDEX || !inner_edges ||
	    !AutoFactorizedGroupJoinAggregatesSupported(aggregate, *candidate, state_widths, payload_widths)) {
		return false;
	}
	const auto driver_rows = inputs.driver.EstimateCardinality(context);
	if (driver_rows == 0) {
		return false;
	}
	const auto factor_rows = static_cast<double>(inputs.left_factor.EstimateCardinality(context)) +
	                         static_cast<double>(inputs.right_factor.EstimateCardinality(context));
	const auto factor_driver_ratio = factor_rows / static_cast<double>(driver_rows);
	if (factor_driver_ratio >= 10.0) {
		return true;
	}
	if (!candidate->unique_driver || candidate->routed) {
		return false;
	}
	return factor_driver_ratio >= FACTORIZED_MIN_CACHE_AMORTIZATION_RATIO &&
	       factor_driver_ratio <= FACTORIZED_MAX_CACHE_FACTOR_DRIVER_RATIO;
}

optional<HashGroupJoinOrderContext> GetFactorizedGroupJoinOrderContext(LogicalAggregate &aggregate,
                                                                       ClientContext &context) {
	auto candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context);
	if (candidate) {
		if (!HasPotentialFactorizedGroupJoinCandidate(aggregate, context)) {
			return nullopt;
		}
	} else {
		auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
		if (strategy != GroupJoinStrategy::FACTORIZED || !GetFactorizedCoarseGroupInfo(aggregate, context)) {
			return nullopt;
		}
		candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context, false);
		D_ASSERT(candidate);
	}
	auto inputs = GetFactorizedGroupJoinInputs(aggregate, *candidate);
	HashGroupJoinOrderContext result;
	result.strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	result.factorized = true;
	LogicalJoin::GetTableReferences(inputs.driver, result.factorized_driver_tables);
	LogicalJoin::GetTableReferences(inputs.left_factor, result.factorized_left_tables);
	LogicalJoin::GetTableReferences(inputs.right_factor, result.factorized_right_tables);
	if (result.factorized_driver_tables.empty() || result.factorized_left_tables.empty() ||
	    result.factorized_right_tables.empty()) {
		return nullopt;
	}
	return result;
}

bool HasFactorizedGroupJoinCandidate(LogicalAggregate &aggregate, ClientContext &context) {
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy != GroupJoinStrategy::FACTORIZED && strategy != GroupJoinStrategy::AUTO) {
		return false;
	}
	auto candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context);
	if (!candidate || strategy == GroupJoinStrategy::FACTORIZED) {
		return candidate.has_value();
	}
	auto cost = EstimateFactorizedGroupJoinCost(aggregate, *candidate, context);
	if (cost.selected) {
		return true;
	}
	if (!cost.reliable) {
		return false;
	}
	return false;
}

static optional<FactorizedGroupJoinCandidate> TrySelectFactorizedGroupJoinCandidate(LogicalAggregate &aggregate,
                                                                                    ClientContext &context) {
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy != GroupJoinStrategy::FACTORIZED && strategy != GroupJoinStrategy::AUTO) {
		return nullopt;
	}
	auto candidate = TryAnalyzeFactorizedGroupJoinCandidate(aggregate, context);
	if (!candidate || strategy == GroupJoinStrategy::FACTORIZED) {
		return candidate;
	}
	return EstimateFactorizedGroupJoinCost(aggregate, *candidate, context).selected ? std::move(candidate) : nullopt;
}

static unique_ptr<JoinFilterPushdownInfo> TakeFactorizedRuntimeFilter(LogicalComparisonJoin &join, idx_t factor_child,
                                                                      const vector<idx_t> &factor_keys) {
	// Join filter pushdown always derives filters from the right-hand build child. Other orientations cannot be
	// retained by an independently built factor without changing the filter's source semantics.
	if (factor_child != 1 || !join.filter_pushdown) {
		return nullptr;
	}
	vector<idx_t> remapped_conditions;
	remapped_conditions.reserve(join.filter_pushdown->join_condition.size());
	for (auto condition_idx : join.filter_pushdown->join_condition) {
		if (condition_idx >= join.conditions.size()) {
			return nullptr;
		}
		auto factor_index = GetDirectReferenceIndex(join.conditions[condition_idx].GetRHS(), *join.children[1]);
		if (!factor_index.IsValid()) {
			return nullptr;
		}
		auto entry = std::find(factor_keys.begin(), factor_keys.end(), factor_index.GetIndex());
		if (entry == factor_keys.end()) {
			return nullptr;
		}
		remapped_conditions.push_back(NumericCast<idx_t>(entry - factor_keys.begin()));
	}
	join.filter_pushdown->join_condition = std::move(remapped_conditions);
	return std::move(join.filter_pushdown);
}

static unique_ptr<JoinFilterPushdownInfo> CreateGroupJoinRuntimeFilter(LogicalOperator &probe,
                                                                       const vector<idx_t> &probe_keys,
                                                                       const vector<LogicalType> &key_types,
                                                                       ClientContext &context) {
	if (probe_keys.size() != key_types.size()) {
		throw InternalException("GroupJoin runtime-filter key counts differ");
	}
	auto bindings = probe.GetColumnBindings();
	auto result = make_uniq<JoinFilterPushdownInfo>();
	vector<JoinFilterPushdownColumn> columns;
	columns.reserve(probe_keys.size());
	for (idx_t key_idx = 0; key_idx < probe_keys.size(); key_idx++) {
		if (probe_keys[key_idx] >= bindings.size()) {
			return nullptr;
		}
		BoundColumnRefExpression key_expression(key_types[key_idx], bindings[probe_keys[key_idx]]);
		JoinFilterPushdownColumn column;
		if (!JoinFilterPushdownUtil::PushdownJoinFilterExpression(key_expression, column)) {
			return nullptr;
		}
		column.join_filter_idx = key_idx;
		columns.push_back(std::move(column));
		result->join_condition.push_back(key_idx);
	}

	vector<PushdownFilterTarget> targets;
	JoinFilterPushdownOptimizer::GetPushdownFilterTargets(probe, std::move(columns), targets);
	for (auto &target : targets) {
		auto &get = target.get;
		if (!get.dynamic_filters) {
			get.dynamic_filters = make_shared_ptr<DynamicTableFilterSet>();
		}
		JoinFilterPushdownFilter filter;
		filter.dynamic_filters = get.dynamic_filters;
		filter.columns = std::move(target.columns);
		result->probe_info.push_back(std::move(filter));
	}
	if (result->probe_info.empty()) {
		return nullptr;
	}

	array<AggregateFunction, 2> functions {MinFunction::GetFunction(), MaxFunction::GetFunction()};
	FunctionBinder binder(context);
	for (idx_t key_idx = 0; key_idx < key_types.size(); key_idx++) {
		for (auto &function : functions) {
			vector<unique_ptr<Expression>> children;
			children.push_back(make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx));
			auto aggregate =
			    binder.BindAggregateFunction(function, std::move(children), nullptr, AggregateType::NON_DISTINCT);
			if (aggregate->GetChildren().size() != 1) {
				return nullptr;
			}
			result->min_max_aggregates.push_back(std::move(aggregate));
		}
	}
	result->build_side_has_filter = false;
	return result;
}

static void PlanFactorizedGroupJoin(unique_ptr<LogicalOperator> &root, LogicalAggregate &aggregate,
                                    FactorizedGroupJoinCandidate candidate, ClientContext &context) {
	auto cost = EstimateFactorizedGroupJoinCost(aggregate, candidate, context);
	const auto auto_selected =
	    Settings::Get<DebugGroupJoinStrategySetting>(context) == GroupJoinStrategy::AUTO && cost.selected;
	auto &top_join = aggregate.children[0]->Cast<LogicalComparisonJoin>();
	auto &nested_join = top_join.children[candidate.nested_child]->Cast<LogicalComparisonJoin>();
	optional_ptr<LogicalOperator> driver;
	optional_ptr<LogicalOperator> left_factor;
	optional_ptr<LogicalOperator> right_factor;
	if (candidate.driver_child == DConstants::INVALID_INDEX) {
		driver = top_join.children[1 - candidate.nested_child].get();
		left_factor = nested_join.children[0].get();
		right_factor = nested_join.children[1].get();
	} else {
		driver = nested_join.children[candidate.driver_child].get();
		left_factor = nested_join.children[1 - candidate.driver_child].get();
		right_factor = top_join.children[1 - candidate.nested_child].get();
	}
	auto driver_rows = driver->EstimateCardinality(context);
	auto left_rows = left_factor->EstimateCardinality(context);
	auto right_rows = right_factor->EstimateCardinality(context);
	auto driver_column_count = driver->GetColumnBindings().size();
	auto left_column_count = left_factor->GetColumnBindings().size();
	Value factorized_perfect_min;
	Value factorized_perfect_max;
	idx_t factorized_perfect_range = 0;
	TryGetFactorizedPerfectGroupJoinBounds(aggregate, *driver, candidate, context, factorized_perfect_min,
	                                       factorized_perfect_max, factorized_perfect_range);
	if (candidate.group_driver_indices.size() != aggregate.groups.size()) {
		throw InternalException("Factorized GroupJoin group mapping count does not match");
	}
	auto driver_bindings = driver->GetColumnBindings();
	for (idx_t group_idx = 0; group_idx < aggregate.groups.size(); group_idx++) {
		auto driver_index = candidate.group_driver_indices[group_idx];
		if (driver_index >= driver_bindings.size()) {
			throw InternalException("Factorized GroupJoin driver group index is out of bounds");
		}
		auto alias = aggregate.groups[group_idx]->GetAlias();
		auto type = aggregate.groups[group_idx]->GetReturnType();
		aggregate.groups[group_idx] =
		    make_uniq<BoundColumnRefExpression>(std::move(alias), type, driver_bindings[driver_index]);
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
	result->implementation = GroupJoinImplementation::FACTORIZED_HASH;
	result->unique_owner = candidate.unique_driver;
	result->routed = candidate.routed;
	result->factorized_driver_key_indices = std::move(candidate.driver_key_indices);
	result->factorized_left_key_indices = std::move(candidate.left_key_indices);
	result->factorized_right_key_indices = std::move(candidate.right_key_indices);
	result->factorized_aggregate_sources = std::move(candidate.aggregate_sources);
	result->factorized_driver_column_count = driver_column_count;
	result->factorized_left_column_count = left_column_count;
	result->factorized_preserve_left = candidate.preserve_left;
	result->factorized_preserve_right = candidate.preserve_right;
	result->factorized_semi_left = candidate.semi_left;
	result->factorized_semi_right = candidate.semi_right;
	auto configured_execution = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	if (Settings::Get<DebugForceExternalSetting>(context) || configured_execution == GroupJoinExecutionMode::EXTERNAL) {
		result->execution_mode = GroupJoinExecutionMode::EXTERNAL;
	}
	result->factorized_driver_first = cost.driver_first;
	if (result->factorized_driver_first && result->execution_mode != GroupJoinExecutionMode::EXTERNAL) {
		vector<LogicalType> driver_key_types;
		driver_key_types.reserve(result->factorized_driver_key_indices.size());
		for (auto key_idx : result->factorized_driver_key_indices) {
			if (key_idx >= driver->types.size()) {
				throw InternalException("Factorized GroupJoin driver filter key is out of bounds");
			}
			driver_key_types.push_back(driver->types[key_idx]);
		}
		result->factorized_left_driver_filter_pushdown =
		    CreateGroupJoinRuntimeFilter(*left_factor, result->factorized_left_key_indices, driver_key_types, context);
		result->factorized_right_driver_filter_pushdown = CreateGroupJoinRuntimeFilter(
		    *right_factor, result->factorized_right_key_indices, driver_key_types, context);
	}
	result->estimated_owner_rows = driver_rows;
	result->estimated_left_factor_rows = left_rows;
	result->estimated_right_factor_rows = right_rows;
	result->estimated_factorized_join_rows = cost.join_rows;
	result->estimated_factorized_matched_drivers = cost.matched_drivers;
	result->estimated_left_factor_scan_rows = cost.left_scan_rows;
	result->estimated_right_factor_scan_rows = cost.right_scan_rows;
	result->factorized_build_cost = cost.build_cost;
	result->factorized_filter_cost = cost.filter_cost;
	result->factorized_probe_cost = cost.probe_cost;
	result->factorized_scan_cost = cost.scan_cost;
	result->factorized_cache_cost = cost.cache_cost;
	result->factorized_eager_work_cost = cost.eager_work_cost;
	result->factorized_routing_cost = cost.routing_cost;
	result->factorized_spill_cost = cost.spill_cost;
	result->factorized_cost = cost.factorized_cost;
	result->factorized_best_existing_cost = cost.best_existing_cost;
	result->factorized_driver_first_cost = cost.driver_first_cost;
	result->factorized_factors_first_cost = cost.factors_first_cost;
	result->factorized_cost_reliable = cost.reliable;
	result->factorized_auto_selected = auto_selected;
	if (candidate.driver_child != DConstants::INVALID_INDEX) {
		result->factorized_left_filter_pushdown =
		    TakeFactorizedRuntimeFilter(nested_join, 1 - candidate.driver_child, result->factorized_left_key_indices);
		result->factorized_right_filter_pushdown =
		    TakeFactorizedRuntimeFilter(top_join, 1 - candidate.nested_child, result->factorized_right_key_indices);
	}
	result->perfect_min = std::move(factorized_perfect_min);
	result->perfect_max = std::move(factorized_perfect_max);
	result->perfect_range = factorized_perfect_range;
	if (candidate.driver_child == DConstants::INVALID_INDEX) {
		result->children.push_back(std::move(top_join.children[1 - candidate.nested_child]));
		result->children.push_back(std::move(nested_join.children[0]));
		result->children.push_back(std::move(nested_join.children[1]));
	} else {
		result->children.push_back(std::move(nested_join.children[candidate.driver_child]));
		result->children.push_back(std::move(nested_join.children[1 - candidate.driver_child]));
		result->children.push_back(std::move(top_join.children[1 - candidate.nested_child]));
	}
	root = std::move(result);
}

void PlanHashGroupJoins(unique_ptr<LogicalOperator> &root, ClientContext &context) {
	for (auto &child : root->children) {
		PlanHashGroupJoins(child, context);
	}
	if (root->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY || root->children.size() != 1) {
		return;
	}
	auto &aggregate = root->Cast<LogicalAggregate>();
	if (auto factorized = TrySelectFactorizedGroupJoinCandidate(aggregate, context)) {
		ApplyFactorizedProjectionInlining(aggregate);
		PlanFactorizedGroupJoin(root, aggregate, std::move(*factorized), context);
		return;
	}
	if (root->children[0]->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		return;
	}
	auto &join = root->children[0]->Cast<LogicalComparisonJoin>();
	auto candidate = TryGetPlannedHashGroupJoinCandidate(aggregate, join, context,
	                                                     HashGroupJoinCandidateMode::ALLOW_AGGREGATE_ORDER);
	if (!candidate) {
		return;
	}
	auto cost = EstimateHashGroupJoinCost(aggregate, join, *candidate, context);
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	auto configured_execution = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range = 0;
	const auto can_use_perfect =
	    TryGetPerfectGroupJoinBounds(aggregate, join, *candidate, context, perfect_min, perfect_max, perfect_range);
	if (strategy == GroupJoinStrategy::PERFECT && !can_use_perfect) {
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
	if (can_use_perfect) {
		result->perfect_min = perfect_min;
		result->perfect_max = perfect_max;
		result->perfect_range = perfect_range;
	}
	const auto use_index = strategy == GroupJoinStrategy::INDEX || aggregate.group_join_auto_index ||
	                       (configured_execution == GroupJoinExecutionMode::INDEX && cost.index_available);
	if (strategy == GroupJoinStrategy::EAGER || (strategy == GroupJoinStrategy::AUTO && cost.physical_eager_selected)) {
		result->implementation = GroupJoinImplementation::EAGER_HASH;
	} else if (use_index) {
		result->implementation = GroupJoinImplementation::INDEX;
	} else if (can_use_perfect && (strategy == GroupJoinStrategy::FORCE || strategy == GroupJoinStrategy::PERFECT ||
	                               (strategy == GroupJoinStrategy::AUTO && cost.perfect_selected))) {
		result->implementation = GroupJoinImplementation::PERFECT_HASH;
	}
	result->execution_mode = cost.execution_mode;
	if (configured_execution != GroupJoinExecutionMode::AUTO && configured_execution != GroupJoinExecutionMode::INDEX) {
		result->execution_mode = configured_execution;
	} else if (use_index) {
		result->execution_mode = GroupJoinExecutionMode::INDEX;
	}
	result->estimated_owner_rows = cost.owner_rows;
	result->estimated_probe_rows = cost.probe_rows;
	result->estimated_match_rows = cost.match_rows;
	result->estimated_matched_groups = cost.matched_groups;
	result->estimated_distinct_probe_keys = cost.distinct_probe_keys;
	result->separate_cost = cost.separate_cost;
	result->eager_cost = cost.eager_cost;
	result->physical_eager_cost = cost.physical_eager_cost;
	result->perfect_cost = cost.perfect_cost;
	result->memoizing_cost = cost.memoizing_cost;
	result->index_cost = cost.index_cost;
	const auto streaming_eager =
	    result->implementation == GroupJoinImplementation::EAGER_HASH && result->unique_owner && !result->routed;
	if (streaming_eager) {
		if (result->unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD) {
			vector<LogicalType> probe_key_types;
			probe_key_types.reserve(result->probe_key_indices.size());
			auto &probe_types = join.children[candidate->probe_child]->types;
			for (auto key_idx : result->probe_key_indices) {
				if (key_idx >= probe_types.size()) {
					throw InternalException("GroupJoin runtime-filter probe key is out of bounds");
				}
				probe_key_types.push_back(probe_types[key_idx]);
			}
			result->filter_pushdown = CreateGroupJoinRuntimeFilter(*join.children[candidate->owner_child],
			                                                       result->owner_key_indices, probe_key_types, context);
		}
	} else if (candidate->owner_child == 1 && join.filter_pushdown) {
		result->filter_pushdown = std::move(join.filter_pushdown);
	} else {
		vector<LogicalType> owner_key_types;
		owner_key_types.reserve(result->owner_key_indices.size());
		auto &owner_types = join.children[candidate->owner_child]->types;
		for (auto key_idx : result->owner_key_indices) {
			if (key_idx >= owner_types.size()) {
				throw InternalException("GroupJoin runtime-filter owner key is out of bounds");
			}
			owner_key_types.push_back(owner_types[key_idx]);
		}
		result->filter_pushdown = CreateGroupJoinRuntimeFilter(*join.children[candidate->probe_child],
		                                                       result->probe_key_indices, owner_key_types, context);
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
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy == GroupJoinStrategy::INDEX || strategy == GroupJoinStrategy::PERFECT ||
	    strategy == GroupJoinStrategy::EAGER || strategy == GroupJoinStrategy::FACTORIZED) {
		return nullopt;
	}
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
