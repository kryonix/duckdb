//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/join_hashtable.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/storage/storage_info.hpp"

#include <mutex>

namespace duckdb {
class BufferManager;
class BufferHandle;

//! JoinHashTable is a linear probing HT that is used for computing joins
/*!
   The JoinHashTable concatenates incoming chunks inside a linked list of
   data ptrs. The storage looks like this internally.
   [SERIALIZED ROW][NEXT POINTER]
   [SERIALIZED ROW][NEXT POINTER]
   There is a separate hash map of pointers that point into this table.
   This is what is used to resolve the hashes.
   [POINTER]
   [POINTER]
   [POINTER]
   The pointers are either NULL
*/
class JoinHashTable {
public:
	//! Scan structure that can be used to resume scans, as a single probe can
	//! return 1024*N values (where N is the size of the HT). This is
	//! returned by the JoinHashTable::Scan function and can be used to resume a
	//! probe.
	struct ScanStructure {
		Vector pointers;
		Vector build_pointer_vector;
		sel_t sel_vector[STANDARD_VECTOR_SIZE];
		// whether or not the given tuple has found a match, used for LeftJoin
		bool found_match[STANDARD_VECTOR_SIZE];
		JoinHashTable &ht;
		bool finished;

		ScanStructure(JoinHashTable &ht);
		//! Get the next batch of data from the scan structure
		void Next(DataChunk &keys, DataChunk &left, DataChunk &result);

	private:
		//! Next operator for the inner join
		void NextInnerJoin(DataChunk &keys, DataChunk &left, DataChunk &result);
		//! Next operator for the semi join
		void NextSemiJoin(DataChunk &keys, DataChunk &left, DataChunk &result);
		//! Next operator for the anti join
		void NextAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result);
		//! Next operator for the left outer join
		void NextLeftJoin(DataChunk &keys, DataChunk &left, DataChunk &result);
		//! Next operator for the mark join
		void NextMarkJoin(DataChunk &keys, DataChunk &left, DataChunk &result);
		//! Next operator for the single join
		void NextSingleJoin(DataChunk &keys, DataChunk &left, DataChunk &result);

		//! Scan the hashtable for matches of the specified keys, setting the found_match[] array to true or false for
		//! every tuple
		void ScanKeyMatches(DataChunk &keys);
		template <bool MATCH> void NextSemiOrAntiJoin(DataChunk &keys, DataChunk &left, DataChunk &result);

		index_t ScanInnerJoin(DataChunk &keys, DataChunk &left, DataChunk &result);

		void ResolvePredicates(DataChunk &keys, Vector &comparison_result);
		index_t ResolvePredicates(DataChunk &keys, sel_t comparison_result[]);
	};

private:
	//! Nodes store the actual data of the tuples inside the HT as a linked list
	struct HTDataBlock {
		index_t count;
		index_t capacity;
		block_id_t block_id;
	};

	index_t AppendToBlock(HTDataBlock &block, BufferHandle &handle, Vector &key_data, data_ptr_t key_locations[],
	                      data_ptr_t tuple_locations[], data_ptr_t hash_locations[], index_t remaining);

	void Hash(DataChunk &keys, Vector &hashes);

public:
	JoinHashTable(BufferManager &buffer_manager, vector<JoinCondition> &conditions, vector<TypeId> build_types,
	              JoinType type);
	~JoinHashTable();

	//! Add the given data to the HT
	void Build(DataChunk &keys, DataChunk &input);
	//! Finalize the build of the HT, constructing the actual hash table and making the HT ready for probing. Finalize
	//! must be called before any call to Probe, and after Finalize is called Build should no longer be ever called.
	void Finalize();
	//! Probe the HT with the given input chunk, resulting in the given result
	unique_ptr<ScanStructure> Probe(DataChunk &keys);

	//! The stringheap of the JoinHashTable
	StringHeap string_heap;

	index_t size() {
		return count;
	}

	BufferManager &buffer_manager;
	//! The types of the keys used in equality comparison
	vector<TypeId> equality_types;
	//! The types of the keys
	vector<TypeId> condition_types;
	//! The types of all conditions
	vector<TypeId> build_types;
	//! The comparison predicates
	vector<ExpressionType> predicates;
	//! Size of condition keys
	index_t equality_size;
	//! Size of condition keys
	index_t condition_size;
	//! Size of build tuple
	index_t build_size;
	//! The size of an entry as stored in the HashTable
	index_t entry_size;
	//! The total tuple size
	index_t tuple_size;
	//! The join type of the HT
	JoinType join_type;
	//! Whether or not the HT has been finalized
	bool finalized;
	//! Whether or not any of the key elements contain NULL
	bool has_null;
	//! Bitmask for getting relevant bits from the hashes to determine the position
	uint64_t bitmask;
	//! The amount of entries stored per block
	index_t block_capacity;

	struct {
		//! The types of the duplicate eliminated columns, only used in correlated MARK JOIN for flattening ANY()/ALL()
		//! expressions
		vector<TypeId> correlated_types;
		//! The aggregate expression nodes used by the HT
		vector<unique_ptr<Expression>> correlated_aggregates;
		//! The HT that holds the group counts for every correlated column
		unique_ptr<SuperLargeHashTable> correlated_counts;
		//! Group chunk used for aggregating into correlated_counts
		DataChunk group_chunk;
		//! Payload chunk used for aggregating into correlated_counts
		DataChunk payload_chunk;
		//! Result chunk used for aggregating into correlated_counts
		DataChunk result_chunk;
	} correlated_mark_join_info;

private:
	//! Apply a bitmask to the hashes
	void ApplyBitmask(Vector &hashes);
	//! Insert the given set of locations into the HT with the given set of
	//! hashes. Caller should hold lock in parallel HT.
	void InsertHashes(Vector &hashes, data_ptr_t key_locations[]);

	//! The amount of entries stored in the HT currently
	index_t count;
	//! The blocks holding the main data of the hash table
	vector<HTDataBlock> blocks;
	//! Pinned handles, these are pinned during finalization only
	vector<unique_ptr<BufferHandle>> pinned_handles;
	//! The hash map of the HT, created after finalization
	unique_ptr<BufferHandle> hash_map;
	//! Whether or not NULL values are considered equal in each of the comparisons
	vector<bool> null_values_are_equal;

	//! Copying not allowed
	JoinHashTable(const JoinHashTable &) = delete;
};

} // namespace duckdb
