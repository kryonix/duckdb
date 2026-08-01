#include "duckdb/optimizer/hash_group_join.hpp"

#include "duckdb/optimizer/key_properties.hpp"
#include "duckdb/optimizer/optimizer.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/operator/subtract.hpp"
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
	if (result.strategy == GroupJoinStrategy::SEPARATE) {
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
	auto strategy = Settings::Get<DebugGroupJoinStrategySetting>(context);
	if (strategy == GroupJoinStrategy::INDEX || strategy == GroupJoinStrategy::PERFECT ||
	    strategy == GroupJoinStrategy::EAGER) {
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
