//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/group_join_strategy.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace duckdb {

enum class GroupJoinStrategy : uint8_t { DISABLED, FORCE, AUTO };

} // namespace duckdb
