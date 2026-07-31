//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/hash_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

class ClientContext;
class LogicalAggregate;
class LogicalComparisonJoin;

enum class UniqueKeyProof : uint8_t { PRIMARY_KEY, UNIQUE_NOT_NULL };
enum class HashGroupJoinCandidateMode : uint8_t { STRICT, ALLOW_AGGREGATE_ORDER };

struct HashGroupJoinCandidate {
	idx_t owner_child;
	idx_t probe_child;
	vector<idx_t> owner_key_indices;
	vector<idx_t> probe_key_indices;
	UniqueKeyProof owner_key_proof;
};

optional<HashGroupJoinCandidate>
TryGetHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join, ClientContext &context,
                             HashGroupJoinCandidateMode mode = HashGroupJoinCandidateMode::STRICT);

} // namespace duckdb
