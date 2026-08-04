//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/perfect_group_join_executor.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/selection_vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"

namespace duckdb {

//! Direct lookup directory for a unique, bounded integral GroupJoin owner key.
class PerfectGroupJoinExecutor {
public:
	PerfectGroupJoinExecutor(LogicalType key_type, Value minimum, Value maximum, idx_t range);

	//! Publishes a unique owner chunk. Returns false when a key is outside the planned bounds.
	bool Sink(Vector &keys, Vector &addresses, Vector &group_ids, bool ignore_nulls = false);
	//! Stores matching row addresses at input positions and returns a stable input-order selection of matches.
	idx_t Lookup(Vector &keys, Vector &addresses, Vector &group_ids, SelectionVector &found) const;

	idx_t Size() const {
		return range + 1;
	}

private:
	template <class T>
	bool SinkInternal(Vector &keys, Vector &addresses, Vector &group_ids, bool ignore_nulls);
	template <class T>
	idx_t LookupInternal(Vector &keys, Vector &addresses, Vector &group_ids, SelectionVector &found) const;

private:
	LogicalType key_type;
	Value minimum;
	Value maximum;
	idx_t range;
	unsafe_unique_array<data_ptr_t> directory;
	unsafe_unique_array<uint64_t> directory_group_ids;
};

} // namespace duckdb
