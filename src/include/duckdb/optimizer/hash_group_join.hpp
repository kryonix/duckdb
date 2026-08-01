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
#include "duckdb/optimizer/key_properties.hpp"

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
	optional<UniqueKeyProof> owner_key_proof;
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

optional<HashGroupJoinCandidate>
TryGetHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                             HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Selects a forced or cost-qualified automatic candidate and records an automatic selection on the aggregate.
optional<HashGroupJoinCandidate>
TrySelectHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                                HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Returns a forced candidate or a candidate previously selected by the automatic cost gate.
optional<HashGroupJoinCandidate>
TryGetPlannedHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                                    HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

optional<StaticHashGroupJoinCandidate>
TryGetStaticHashGroupJoinCandidate(LogicalComparisonJoin &join, ClientContext &context,
                                   HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

//! Returns whether aggregate is the direct probe aggregate of a valid static GroupJoin in root.
bool IsStaticHashGroupJoinAggregate(LogicalOperator &root, LogicalAggregate &aggregate, ClientContext &context,
                                    HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

} // namespace duckdb
