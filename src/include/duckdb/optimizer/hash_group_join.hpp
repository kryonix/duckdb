//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/hash_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/execution/group_join_strategy.hpp"

namespace duckdb {

class ClientContext;
class LogicalAggregate;
class LogicalComparisonJoin;
class LogicalOperator;

enum class HashGroupJoinCandidateMode : uint8_t { STRICT, ALLOW_AGGREGATE_ORDER };

struct HashGroupJoinCandidate {
	idx_t owner_child;
	idx_t probe_child;
	vector<idx_t> owner_key_indices;
	vector<idx_t> probe_key_indices;
	vector<idx_t> owner_payload_indices;
	vector<HashGroupJoinOutputColumn> output_groups;
	bool unique_owner;
	HashGroupJoinUnmatchedPolicy unmatched_policy;
	bool routed;
	bool single_match;
};

struct StaticHashGroupJoinCandidate {
	LogicalAggregate *aggregate;
	vector<idx_t> owner_key_indices;
	vector<idx_t> owner_payload_indices;
	vector<HashGroupJoinOutputColumn> output_columns;
};

struct HashGroupJoinCostEstimate {
	idx_t owner_rows = 0;
	idx_t probe_rows = 0;
	idx_t match_rows = 0;
	idx_t matched_groups = 0;
	idx_t distinct_probe_keys = 0;
	idx_t key_width = 0;
	idx_t state_width = 0;
	double separate_cost = 0;
	double eager_cost = 0;
	double memoizing_cost = 0;
	double index_cost = 0;
	GroupJoinExecutionMode execution_mode = GroupJoinExecutionMode::AUTO;
	bool index_available = false;
	bool index_selected = false;
	bool hash_selected = false;
};

//! Returns whether GroupJoin planning is enabled by both the strategy setting and optimizer configuration.
bool HashGroupJoinPlanningEnabled(ClientContext &context);
//! Returns whether the complete forced GroupJoin planning path is enabled.
bool ForceHashGroupJoinPlanning(ClientContext &context);

optional<HashGroupJoinCandidate>
TryGetHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                             HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

HashGroupJoinCostEstimate EstimateHashGroupJoinCost(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                                    const HashGroupJoinCandidate &candidate, ClientContext &context);

//! Selects a forced or cost-qualified automatic candidate and records an automatic selection on the aggregate.
optional<HashGroupJoinCandidate>
TrySelectHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                                HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Returns a forced candidate or a candidate previously selected by the automatic cost gate.
optional<HashGroupJoinCandidate>
TryGetPlannedHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                                    HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Replaces selected aggregate-over-join patterns with first-class logical GroupJoin operators.
void PlanHashGroupJoins(unique_ptr<LogicalOperator> &root, ClientContext &context);

optional<StaticHashGroupJoinCandidate>
TryGetStaticHashGroupJoinCandidate(LogicalComparisonJoin &join, ClientContext &context,
                                   HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Returns whether aggregate is the direct probe aggregate of a valid static GroupJoin in root.
bool IsStaticHashGroupJoinAggregate(LogicalOperator &root, LogicalAggregate &aggregate, ClientContext &context,
                                    HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

} // namespace duckdb
