//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/key_properties.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional.hpp"
#include "duckdb/common/types.hpp"

namespace duckdb {

class LogicalGet;
class LogicalOperator;

enum class UniqueKeyProof : uint8_t { PRIMARY_KEY, UNIQUE_NOT_NULL, AGGREGATE_GROUP };

struct UniqueKeyProperty {
	UniqueKeyProof proof;
	optional_ptr<LogicalGet> base_scan;

	//! Returns whether the proven key functionally determines an output column through direct projections/filters.
	bool FunctionallyDetermines(LogicalOperator &owner, idx_t output_column) const;
};

//! Proves that the output columns are a complete non-NULL unique key of one base table.
optional<UniqueKeyProperty> GetUniqueKeyProperty(LogicalOperator &owner, const vector<idx_t> &output_columns);

//! Proves that foreign_columns are a declared foreign key referencing primary_columns.
bool HasForeignKeyProperty(LogicalOperator &foreign, const vector<idx_t> &foreign_columns, LogicalOperator &primary,
                           const vector<idx_t> &primary_columns);

} // namespace duckdb
