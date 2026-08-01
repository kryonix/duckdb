#include "duckdb/planner/operator/logical_group_join.hpp"

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
	return result;
}

} // namespace duckdb
