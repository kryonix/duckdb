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

enum class GroupJoinStrategy : uint8_t { DISABLED, FORCE, AUTO, SEPARATE, EAGER, HASH, PERFECT, INDEX };

enum class GroupJoinImplementation : uint8_t { MEMOIZING_HASH, PERFECT_HASH, EAGER_HASH, INDEX };

enum class GroupJoinExecutionMode : uint8_t { AUTO, SERIAL, LOCAL, OWNERSHIP, EXTERNAL, INDEX };

static constexpr idx_t PERFECT_GROUP_JOIN_MAX_RANGE = 1048575;
static constexpr idx_t PERFECT_GROUP_JOIN_MEMORY_DIVISOR = 10;

inline bool PerfectGroupJoinDirectoryFits(idx_t range, idx_t max_memory) {
	if (range > PERFECT_GROUP_JOIN_MAX_RANGE) {
		return false;
	}
	const auto entry_count = range + 1;
	const auto entry_size = sizeof(data_ptr_t) + sizeof(uint64_t);
	return entry_count <= max_memory / PERFECT_GROUP_JOIN_MEMORY_DIVISOR / entry_size;
}

enum class HashGroupJoinUnmatchedPolicy : uint8_t { DISCARD, EMPTY_AGGREGATE, NULL_EXTENDED_ROW };

enum class HashGroupJoinOutputSource : uint8_t { KEY, OWNER_PAYLOAD, MATCHED_KEY, AGGREGATE };

struct HashGroupJoinOutputColumn {
	HashGroupJoinOutputSource source;
	idx_t index;
};

} // namespace duckdb
