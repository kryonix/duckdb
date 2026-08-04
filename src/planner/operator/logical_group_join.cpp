#include "duckdb/planner/operator/logical_group_join.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

LogicalGroupJoin::LogicalGroupJoin(TableIndex group_index, TableIndex aggregate_index,
                                   vector<unique_ptr<Expression>> select_list)
    : LogicalAggregate(group_index, aggregate_index, std::move(select_list)) {
	type = LogicalOperatorType::LOGICAL_GROUP_JOIN;
}

InsertionOrderPreservingMap<string> LogicalGroupJoin::ParamsToString() const {
	auto result = LogicalAggregate::ParamsToString();
	if (IsFactorized()) {
		result["Join Type"] = StringUtil::Format("%s × %s STAR",
		                                         factorized_semi_left       ? "SEMI"
		                                         : factorized_preserve_left ? "LEFT"
		                                                                    : "INNER",
		                                         factorized_semi_right       ? "SEMI"
		                                         : factorized_preserve_right ? "LEFT"
		                                                                     : "INNER");
		result["Driver Key"] = unique_owner ? "UNIQUE" : "GENERAL";
		result["Routing"] = routed ? "ROUTED" : "DIRECT";
		result["Implementation"] = EnumUtil::ToString(implementation);
		result["Strategy"] = EnumUtil::ToString(execution_mode);
		result["Orientation"] = factorized_driver_first ? "DRIVER_FIRST" : "FACTORS_FIRST";
		result["Factors"] = StringUtil::Format("left=%llu rows, right=%llu rows", estimated_left_factor_rows,
		                                       estimated_right_factor_rows);
		result["Factorized Cardinalities"] =
		    StringUtil::Format("driver=%llu, left scan=%llu, right scan=%llu, expanded=%llu, matched drivers=%llu",
		                       estimated_owner_rows, estimated_left_factor_scan_rows, estimated_right_factor_scan_rows,
		                       estimated_factorized_join_rows, estimated_factorized_matched_drivers);
		result["Factorized Costs"] = StringUtil::Format(
		    "build=%.2f, filter=%.2f, probe=%.2f, scan=%.2f, cache=%.2f, eager=%.2f, routing=%.2f, spill=%.2f, "
		    "driver-first=%.2f, factors-first=%.2f, total=%.2f, best existing=%.2f",
		    factorized_build_cost, factorized_filter_cost, factorized_probe_cost, factorized_scan_cost,
		    factorized_cache_cost, factorized_eager_work_cost, factorized_routing_cost, factorized_spill_cost,
		    factorized_driver_first_cost, factorized_factors_first_cost, factorized_cost,
		    factorized_best_existing_cost);
		result["Factorized Decision"] = factorized_auto_selected   ? "AUTO SELECTED"
		                                : factorized_cost_reliable ? "FORCED (AUTO COST AVAILABLE)"
		                                                           : "FORCED (UNRELIABLE STATISTICS)";
		return result;
	}
	result["Join Type"] = single_match                                                ? "SEMI"
	                      : unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD ? "INNER"
	                                                                                  : "OWNER OUTER";
	result["Owner Key"] = unique_owner ? "UNIQUE" : "GENERAL";
	result["Routing"] = routed ? "ROUTED" : "DIRECT";
	result["Implementation"] = EnumUtil::ToString(implementation);
	result["Strategy"] = EnumUtil::ToString(execution_mode);
	if (estimated_owner_rows != 0 || estimated_probe_rows != 0 || estimated_match_rows != 0) {
		result["Strategy Cardinalities"] = StringUtil::Format(
		    "owner=%llu, probe=%llu, matches=%llu, groups=%llu, probe keys=%llu", estimated_owner_rows,
		    estimated_probe_rows, estimated_match_rows, estimated_matched_groups, estimated_distinct_probe_keys);
		result["Strategy Costs"] = StringUtil::Format(
		    "separate=%.2f, eager=%.2f, physical eager=%.2f, perfect=%.2f, memoizing=%.2f, index=%.2f", separate_cost,
		    eager_cost, physical_eager_cost, perfect_cost, memoizing_cost, index_cost);
	}
	return result;
}

} // namespace duckdb
