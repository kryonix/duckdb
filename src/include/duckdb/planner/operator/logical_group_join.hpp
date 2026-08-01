//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/group_join_strategy.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

class LogicalGroupJoin : public LogicalAggregate {
public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_GROUP_JOIN;

	LogicalGroupJoin(TableIndex group_index, TableIndex aggregate_index, vector<unique_ptr<Expression>> select_list);

	idx_t owner_child = DConstants::INVALID_INDEX;
	idx_t probe_child = DConstants::INVALID_INDEX;
	idx_t left_column_count = DConstants::INVALID_INDEX;
	vector<idx_t> owner_key_indices;
	vector<idx_t> probe_key_indices;
	vector<idx_t> owner_payload_indices;
	vector<HashGroupJoinOutputSource> output_group_sources;
	vector<idx_t> output_group_indices;
	HashGroupJoinUnmatchedPolicy unmatched_policy = HashGroupJoinUnmatchedPolicy::DISCARD;
	bool routed = false;
	bool unique_owner = false;
	bool single_match = false;
	bool use_index = false;
	//! Runtime filters generated for the probe subtree when the owner is the original join build side
	unique_ptr<JoinFilterPushdownInfo> filter_pushdown;

public:
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
