//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/radix_partitioned_hashtable.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/physical_operator_states.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/progress_data.hpp"
#include "duckdb/parser/group_by_node.hpp"

namespace duckdb {
class GlobalSinkState;
class LocalSinkState;

struct AggregatePartition;

//! Task-local scratch state for probing finalized radix aggregate partitions.
struct RadixHTLookupState {
public:
	RadixHTLookupState();

	vector<unique_ptr<AggregateHTLookupState>> partition_states;
	vector<SelectionVector> partition_selections;
	vector<idx_t> partition_counts;
	DataChunk selected_groups;
	Vector selected_hashes;
	SelectionVector selected_found;
};

class RadixPartitionedHashTable {
public:
	RadixPartitionedHashTable(GroupingSet &grouping_set, const GroupedAggregateData &op,
	                          TupleDataValidityType group_validity);
	unique_ptr<GroupedAggregateHashTable> CreateHT(ClientContext &context, const idx_t capacity,
	                                               const idx_t radix_bits) const;

public:
	GroupingSet &grouping_set;
	//! The indices specified in the groups_count that do not appear in the grouping_set
	unsafe_vector<idx_t> null_groups;
	const GroupedAggregateData &op;
	vector<LogicalType> group_types;
	//! The GROUPING values that belong to this hash table
	vector<Value> grouping_values;
	//! Whether there are no NULLs in the groups
	const TupleDataValidityType group_validity;

public:
	//! Sink Interface
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const;
	void ResetGlobalSinkState(ClientContext &context, GlobalSinkState &gstate) const;
	void ResetLocalSinkState(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const;

	void Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input, DataChunk &aggregate_input_chunk,
	          const unsafe_vector<idx_t> &filter) const;
	//! Combines one serialized state column into each aggregate beginning at aggregate_begin.
	void SinkExportedStates(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input,
	                        DataChunk &serialized_states, idx_t aggregate_begin) const;
	void Combine(ExecutionContext &context, GlobalSinkState &gstate, LocalSinkState &lstate) const;
	void Finalize(ClientContext &context, GlobalSinkState &gstate) const;

public:
	//! Source interface
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context) const;
	void ResetGlobalSourceState(ClientContext &context, GlobalSourceState &gstate) const;
	void ResetLocalSourceState(ExecutionContext &context, LocalSourceState &lstate) const;

	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, GlobalSinkState &sink,
	                         OperatorSourceInput &input) const;

	ProgressData GetProgress(ClientContext &context, GlobalSinkState &sink_p, GlobalSourceState &gstate) const;

	shared_ptr<TupleDataLayout> GetLayoutPtr() const;
	const TupleDataLayout &GetLayout() const;
	idx_t MaxThreads(GlobalSinkState &sink) const;
	static void SetMultiScan(GlobalSinkState &sink);

	//! Finalize all radix partitions into immutable lookup tables. Calls can run concurrently.
	void FinalizeLookupPartitions(ClientContext &context, GlobalSinkState &sink) const;
	//! Verify that every lookup partition was finalized.
	void FinishLookup(GlobalSinkState &sink) const;
	//! Probe immutable lookup partitions with precomputed group hashes.
	idx_t LookupGroups(GlobalSinkState &sink, DataChunk &groups, Vector &group_hashes, RadixHTLookupState &state,
	                   Vector &addresses, SelectionVector &found_groups) const;

private:
	void SetGroupingValues();
	void PopulateGroupChunk(DataChunk &group_chunk, DataChunk &input_chunk) const;

	shared_ptr<TupleDataLayout> layout_ptr;
};

} // namespace duckdb
