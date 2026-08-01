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
	result["Join Type"] = single_match                                                ? "SEMI"
	                      : unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD ? "INNER"
	                                                                                  : "OWNER OUTER";
	result["Owner Key"] = unique_owner ? "UNIQUE" : "GENERAL";
	result["Routing"] = routed ? "ROUTED" : "DIRECT";
	result["Implementation"] = use_index ? "INDEX" : "HASH";
	result["Strategy"] = EnumUtil::ToString(execution_mode);
	if (estimated_owner_rows != 0 || estimated_probe_rows != 0 || estimated_match_rows != 0) {
		result["Strategy Cardinalities"] = StringUtil::Format(
		    "owner=%llu, probe=%llu, matches=%llu, groups=%llu, probe keys=%llu", estimated_owner_rows,
		    estimated_probe_rows, estimated_match_rows, estimated_matched_groups, estimated_distinct_probe_keys);
		result["Strategy Costs"] = StringUtil::Format("separate=%.2f, eager=%.2f, memoizing=%.2f, index=%.2f",
		                                              separate_cost, eager_cost, memoizing_cost, index_cost);
	}
	return result;
}

} // namespace duckdb
