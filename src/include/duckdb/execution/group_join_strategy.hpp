//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/group_join_strategy.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

#include "duckdb/common/types.hpp"

namespace duckdb {

enum class GroupJoinStrategy : uint8_t { DISABLED, FORCE, AUTO, SEPARATE, EAGER, HASH, INDEX };

enum class GroupJoinExecutionMode : uint8_t { AUTO, SERIAL, LOCAL, OWNERSHIP, EXTERNAL, INDEX };

enum class HashGroupJoinUnmatchedPolicy : uint8_t { DISCARD, EMPTY_AGGREGATE, NULL_EXTENDED_ROW };

enum class HashGroupJoinOutputSource : uint8_t { KEY, OWNER_PAYLOAD, MATCHED_KEY, AGGREGATE };

struct HashGroupJoinOutputColumn {
	HashGroupJoinOutputSource source;
	idx_t index;
};

} // namespace duckdb
