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
	GroupJoinImplementation implementation = GroupJoinImplementation::MEMOIZING_HASH;
	GroupJoinExecutionMode execution_mode = GroupJoinExecutionMode::AUTO;
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range = 0;
	idx_t estimated_owner_rows = 0;
	idx_t estimated_probe_rows = 0;
	idx_t estimated_match_rows = 0;
	idx_t estimated_matched_groups = 0;
	idx_t estimated_distinct_probe_keys = 0;
	vector<idx_t> factorized_driver_key_indices;
	vector<idx_t> factorized_left_key_indices;
	vector<idx_t> factorized_right_key_indices;
	vector<FactorizedAggregateSource> factorized_aggregate_sources;
	idx_t factorized_driver_column_count = 0;
	idx_t factorized_left_column_count = 0;
	bool factorized_preserve_left = false;
	bool factorized_preserve_right = false;
	bool factorized_semi_left = false;
	bool factorized_semi_right = false;
	idx_t estimated_left_factor_rows = 0;
	idx_t estimated_right_factor_rows = 0;
	idx_t estimated_factorized_join_rows = 0;
	idx_t estimated_factorized_matched_drivers = 0;
	idx_t estimated_left_factor_scan_rows = 0;
	idx_t estimated_right_factor_scan_rows = 0;
	double factorized_build_cost = 0;
	double factorized_filter_cost = 0;
	double factorized_probe_cost = 0;
	double factorized_scan_cost = 0;
	double factorized_cache_cost = 0;
	double factorized_eager_work_cost = 0;
	double factorized_routing_cost = 0;
	double factorized_spill_cost = 0;
	double factorized_cost = 0;
	double factorized_best_existing_cost = 0;
	bool factorized_cost_reliable = false;
	bool factorized_auto_selected = false;
	//! Runtime filters produced independently by the left and right factor builds
	unique_ptr<JoinFilterPushdownInfo> factorized_left_filter_pushdown;
	unique_ptr<JoinFilterPushdownInfo> factorized_right_filter_pushdown;
	//! Runtime filters produced by a driver-first build and consumed by the factor scans
	unique_ptr<JoinFilterPushdownInfo> factorized_left_driver_filter_pushdown;
	unique_ptr<JoinFilterPushdownInfo> factorized_right_driver_filter_pushdown;
	double separate_cost = 0;
	double eager_cost = 0;
	double physical_eager_cost = 0;
	double perfect_cost = 0;
	double memoizing_cost = 0;
	double index_cost = 0;
	//! Runtime filters generated for the probe subtree when the owner is the original join build side
	unique_ptr<JoinFilterPushdownInfo> filter_pushdown;

	bool IsFactorized() const {
		return implementation == GroupJoinImplementation::FACTORIZED_HASH;
	}

public:
	InsertionOrderPreservingMap<string> ParamsToString() const override;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
