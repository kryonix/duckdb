#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <atomic>
#include <thread>

using namespace duckdb;

static void SetValues(DataChunk &chunk, const vector<vector<Value>> &columns) {
	D_ASSERT(columns.size() == chunk.ColumnCount());
	const auto count = columns.empty() ? 0 : columns[0].size();
	for (idx_t col_idx = 0; col_idx < columns.size(); col_idx++) {
		D_ASSERT(columns[col_idx].size() == count);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			chunk.data[col_idx].SetValue(row_idx, columns[col_idx][row_idx]);
		}
	}
	chunk.SetChildCardinality(count);
}

static void BuildFactorHashTable(JoinHashTable &hash_table, Allocator &allocator,
                                 const vector<vector<Value>> &key_values, const vector<Value> &payload_values,
                                 bool create_factor_definitions = true) {
	vector<LogicalType> layout_types(hash_table.condition_types);
	layout_types.insert(layout_types.end(), hash_table.build_types.begin(), hash_table.build_types.end());
	layout_types.emplace_back(LogicalType::HASH);
	auto layout = make_shared_ptr<TupleDataLayout>();
	layout->Initialize(layout_types, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	hash_table.FinishInitWithLayout(std::move(layout));

	PartitionedTupleDataAppendState append_state;
	hash_table.GetSinkCollection().InitializeAppendState(append_state);
	for (idx_t offset = 0; offset < payload_values.size(); offset += STANDARD_VECTOR_SIZE) {
		const auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, payload_values.size() - offset);
		vector<vector<Value>> key_batch(key_values.size());
		for (idx_t key_idx = 0; key_idx < key_values.size(); key_idx++) {
			key_batch[key_idx].insert(key_batch[key_idx].end(), key_values[key_idx].begin() + offset,
			                          key_values[key_idx].begin() + offset + count);
		}
		vector<Value> payload_batch(payload_values.begin() + offset, payload_values.begin() + offset + count);
		DataChunk keys;
		keys.Initialize(allocator, hash_table.condition_types);
		SetValues(keys, key_batch);
		DataChunk payload;
		payload.Initialize(allocator, hash_table.build_types);
		SetValues(payload, {payload_batch});
		hash_table.Build(append_state, keys, payload);
	}
	hash_table.GetSinkCollection().FlushAppendState(append_state);
	hash_table.Unpartition();

	if (create_factor_definitions) {
		hash_table.EnableFactorDefinitions();
	}
	hash_table.AllocatePointerTable();
	hash_table.InitializePointerTable(0, hash_table.capacity);
	if (create_factor_definitions) {
		hash_table.FinalizeFactorized(0, hash_table.GetDataCollection().ChunkCount(), false);
	} else {
		hash_table.Finalize(0, hash_table.GetDataCollection().ChunkCount(), false);
	}
	hash_table.finalized = true;
	if (create_factor_definitions) {
		hash_table.FinishFactorDefinitions();
	}
}

static idx_t ProbeFactors(JoinHashTable &hash_table, const vector<vector<Value>> &key_values, Allocator &allocator,
                          TupleDataChunkState &key_state, JoinHashTable::FactorProbeState &probe_state,
                          Vector &factor_refs, SelectionVector &match_sel) {
	DataChunk keys;
	keys.Initialize(allocator, hash_table.condition_types);
	SetValues(keys, key_values);
	TupleDataCollection::InitializeChunkState(key_state, hash_table.condition_types);
	return hash_table.ProbeFactorRefs(keys, key_state, probe_state, factor_refs, match_sel);
}

TEST_CASE("Join factor references preserve complete-key chains", "[join_hashtable]") {
	DuckDB database(nullptr);
	Connection connection(database);
	auto &context = *connection.context;
	auto &allocator = Allocator::Get(context);
	PhysicalPlan physical_plan(allocator);
	auto &dummy = physical_plan.Make<PhysicalDummyScan>(vector<LogicalType> {}, 0);

	SECTION("duplicates, misses, NULLs, stable references, and incremental expansion") {
		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(
		    hash_table, allocator,
		    {{Value::INTEGER(10), Value::INTEGER(20), Value::INTEGER(10), Value(), Value::INTEGER(10)}},
		    {Value::INTEGER(100), Value::INTEGER(200), Value::INTEGER(101), Value::INTEGER(999), Value::INTEGER(102)});
		REQUIRE(hash_table.HasFactorDefinitions());
		REQUIRE(hash_table.FactorCount() == 2);
		REQUIRE(hash_table.MaximumFactorLength() == 3);

		TupleDataChunkState key_state;
		JoinHashTable::FactorProbeState probe_state;
		Vector factor_refs(LogicalType::UBIGINT);
		SelectionVector match_sel(STANDARD_VECTOR_SIZE);
		auto match_count =
		    ProbeFactors(hash_table, {{Value::INTEGER(10), Value::INTEGER(99), Value(), Value::INTEGER(20)}}, allocator,
		                 key_state, probe_state, factor_refs, match_sel);
		REQUIRE(match_count == 2);
		REQUIRE(match_sel.get_index(0) == 0);
		REQUIRE(match_sel.get_index(1) == 3);

		auto refs = FlatVector::GetData<uint64_t>(factor_refs);
		const auto first_ref = JoinFactorRef(refs[0]);
		REQUIRE(first_ref.IsValid());
		REQUIRE(hash_table.GetFactorCount(first_ref) == 3);
		REQUIRE(!JoinFactorRef(refs[1]).IsValid());
		REQUIRE(!JoinFactorRef(refs[2]).IsValid());
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[3])) == 1);
		REQUIRE(hash_table.GetFactorId(first_ref) < hash_table.capacity);
		REQUIRE(hash_table.GetFactorId(JoinFactorRef(refs[3])) < hash_table.capacity);
		REQUIRE(hash_table.GetFactorId(first_ref) != hash_table.GetFactorId(JoinFactorRef(refs[3])));

		TupleDataChunkState second_key_state;
		JoinHashTable::FactorProbeState second_probe_state;
		Vector second_refs(LogicalType::UBIGINT);
		SelectionVector second_match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(hash_table, {{Value::INTEGER(20)}}, allocator, second_key_state, second_probe_state,
		                     second_refs, second_match_sel) == 1);
		REQUIRE(hash_table.GetFactorCount(first_ref) == 3);
		REQUIRE(hash_table.GetFactorChainHead(first_ref) != nullptr);

		JoinHashTable::FactorExpansionState expansion_state;
		Vector build_rows(LogicalType::POINTER);
		SelectionVector source_sel(STANDARD_VECTOR_SIZE);
		vector<int32_t> payloads;
		vector<idx_t> sources;
		while (!expansion_state.Finished(match_count)) {
			const auto expanded = hash_table.ExpandFactorRefs(factor_refs, match_sel, match_count, expansion_state,
			                                                  build_rows, source_sel, 2);
			REQUIRE(expanded > 0);
			DataChunk gathered;
			gathered.Initialize(allocator, {LogicalType::INTEGER});
			hash_table.GatherRHS(build_rows, *FlatVector::IncrementalSelectionVector(), expanded, gathered, 0);
			for (idx_t row_idx = 0; row_idx < expanded; row_idx++) {
				payloads.push_back(gathered.GetValue(0, row_idx).GetValue<int32_t>());
				sources.push_back(source_sel.get_index(row_idx));
			}
		}
		std::sort(payloads.begin(), payloads.end());
		REQUIRE(payloads == vector<int32_t> {100, 101, 102, 200});
		REQUIRE(std::count(sources.begin(), sources.end(), idx_t(0)) == 3);
		REQUIRE(std::count(sources.begin(), sources.end(), idx_t(3)) == 1);
	}

	SECTION("constant, dictionary, and precomputed-hash probes preserve factor identity") {
		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(hash_table, allocator,
		                     {{Value::INTEGER(10), Value::INTEGER(20), Value::INTEGER(10), Value::INTEGER(10)}},
		                     {Value::INTEGER(1), Value::INTEGER(2), Value::INTEGER(3), Value::INTEGER(4)});

		DataChunk constant_keys;
		constant_keys.Initialize(allocator, {LogicalType::INTEGER});
		Vector constant_key(Value::INTEGER(10), count_t(3));
		constant_keys.data[0].Reference(constant_key);
		constant_keys.SetChildCardinality(3);
		TupleDataChunkState constant_key_state;
		TupleDataCollection::InitializeChunkState(constant_key_state, hash_table.condition_types);
		JoinHashTable::FactorProbeState constant_probe_state;
		Vector constant_refs(LogicalType::UBIGINT);
		SelectionVector constant_match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(hash_table.ProbeFactorRefs(constant_keys, constant_key_state, constant_probe_state, constant_refs,
		                                   constant_match_sel) == 3);
		auto constant_ref_data = FlatVector::GetData<uint64_t>(constant_refs);
		REQUIRE(constant_ref_data[0] == constant_ref_data[1]);
		REQUIRE(constant_ref_data[1] == constant_ref_data[2]);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(constant_ref_data[0])) == 3);

		DataChunk dictionary_keys;
		dictionary_keys.Initialize(allocator, {LogicalType::INTEGER});
		SetValues(dictionary_keys, {{Value::INTEGER(10), Value::INTEGER(20), Value::INTEGER(99)}});
		SelectionVector dictionary_sel(4);
		dictionary_sel.set_index(0, 0);
		dictionary_sel.set_index(1, 1);
		dictionary_sel.set_index(2, 0);
		dictionary_sel.set_index(3, 2);
		dictionary_keys.data[0].Slice(dictionary_sel, 4);
		dictionary_keys.SetChildCardinality(4);
		Vector precomputed_hashes(LogicalType::HASH);
		VectorOperations::Hash(dictionary_keys.data[0], precomputed_hashes, dictionary_keys.size());
		TupleDataChunkState dictionary_key_state;
		TupleDataCollection::InitializeChunkState(dictionary_key_state, hash_table.condition_types);
		JoinHashTable::FactorProbeState dictionary_probe_state;
		Vector dictionary_refs(LogicalType::UBIGINT);
		SelectionVector dictionary_match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(hash_table.ProbeFactorRefs(dictionary_keys, dictionary_key_state, dictionary_probe_state,
		                                   dictionary_refs, dictionary_match_sel, precomputed_hashes) == 3);
		auto dictionary_ref_data = FlatVector::GetData<uint64_t>(dictionary_refs);
		REQUIRE(dictionary_match_sel.get_index(0) == 0);
		REQUIRE(dictionary_match_sel.get_index(1) == 1);
		REQUIRE(dictionary_match_sel.get_index(2) == 2);
		REQUIRE(dictionary_ref_data[0] == dictionary_ref_data[2]);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(dictionary_ref_data[0])) == 3);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(dictionary_ref_data[1])) == 1);
		REQUIRE(!JoinFactorRef(dictionary_ref_data[3]).IsValid());
	}

	SECTION("ordinary hash tables allocate no factor metadata") {
		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(hash_table, allocator, {{Value::INTEGER(1), Value::INTEGER(1)}},
		                     {Value::INTEGER(10), Value::INTEGER(20)}, false);
		REQUIRE(!hash_table.HasFactorDefinitions());
		REQUIRE(hash_table.FactorCount() == 0);
		REQUIRE(hash_table.FactorDefinitionSizeInBytes() == 0);
	}

	SECTION("factor handles are rejected after the pointer-table generation changes") {
		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(hash_table, allocator, {{Value::INTEGER(1), Value::INTEGER(1)}},
		                     {Value::INTEGER(10), Value::INTEGER(20)});

		TupleDataChunkState first_key_state;
		JoinHashTable::FactorProbeState first_probe_state;
		Vector first_refs(LogicalType::UBIGINT);
		SelectionVector first_match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(hash_table, {{Value::INTEGER(1)}}, allocator, first_key_state, first_probe_state,
		                     first_refs, first_match_sel) == 1);
		auto first_ref = JoinFactorRef(FlatVector::GetData<uint64_t>(first_refs)[0]);
		REQUIRE(hash_table.GetFactorCount(first_ref) == 2);

		vector<JoinCondition> second_conditions;
		second_conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                               make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                               ExpressionType::COMPARE_EQUAL);
		JoinHashTable second_hash_table(context, dummy, second_conditions, {LogicalType::INTEGER}, JoinType::INNER, 0,
		                                output_columns, nullptr);
		BuildFactorHashTable(second_hash_table, allocator, {{Value::INTEGER(1)}}, {Value::INTEGER(30)});
		TupleDataChunkState second_key_state;
		JoinHashTable::FactorProbeState second_probe_state;
		Vector second_refs(LogicalType::UBIGINT);
		SelectionVector second_match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(second_hash_table, {{Value::INTEGER(1)}}, allocator, second_key_state, second_probe_state,
		                     second_refs, second_match_sel) == 1);
		auto second_ref = JoinFactorRef(FlatVector::GetData<uint64_t>(second_refs)[0]);
		REQUIRE(first_ref.Generation() != second_ref.Generation());
		REQUIRE(second_hash_table.GetFactorCount(second_ref) == 1);
		REQUIRE_THROWS(second_hash_table.GetFactorCount(first_ref));

		hash_table.Reset();
		REQUIRE_THROWS(hash_table.GetFactorCount(first_ref));
		REQUIRE_THROWS(hash_table.GetFactorChainHead(first_ref));
	}

	SECTION("composite keys remain exact") {
		vector<JoinCondition> conditions;
		for (idx_t col_idx = 0; col_idx < 2; col_idx++) {
			conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, col_idx),
			                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, col_idx),
			                        ExpressionType::COMPARE_EQUAL);
		}
		vector<idx_t> output_columns {2};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(
		    hash_table, allocator,
		    {{Value::INTEGER(1), Value::INTEGER(1), Value::INTEGER(1), Value::INTEGER(2), Value()},
		     {Value::INTEGER(1), Value::INTEGER(1), Value::INTEGER(2), Value::INTEGER(1), Value::INTEGER(1)}},
		    {Value::INTEGER(11), Value::INTEGER(12), Value::INTEGER(13), Value::INTEGER(21), Value::INTEGER(999)});
		REQUIRE(hash_table.FactorCount() == 3);

		TupleDataChunkState key_state;
		JoinHashTable::FactorProbeState probe_state;
		Vector factor_refs(LogicalType::UBIGINT);
		SelectionVector match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(hash_table,
		                     {{Value::INTEGER(1), Value::INTEGER(1), Value::INTEGER(2), Value::INTEGER(9)},
		                      {Value::INTEGER(1), Value::INTEGER(2), Value::INTEGER(1), Value::INTEGER(9)}},
		                     allocator, key_state, probe_state, factor_refs, match_sel) == 3);
		auto refs = FlatVector::GetData<uint64_t>(factor_refs);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[0])) == 2);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[1])) == 1);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[2])) == 1);
		REQUIRE(!JoinFactorRef(refs[3]).IsValid());
	}

	SECTION("linear-probing hash collisions do not merge factors") {
		unordered_map<idx_t, int32_t> first_per_bucket;
		int32_t first = 0;
		int32_t second = 0;
		for (int32_t candidate = 0; candidate < 100000; candidate++) {
			const auto bucket = Hash(candidate) & (JoinHashTable::MINIMUM_CAPACITY - 1);
			auto entry = first_per_bucket.find(bucket);
			if (entry != first_per_bucket.end()) {
				first = entry->second;
				second = candidate;
				break;
			}
			first_per_bucket.emplace(bucket, candidate);
		}
		REQUIRE(first != second);
		REQUIRE((Hash(first) & (JoinHashTable::MINIMUM_CAPACITY - 1)) ==
		        (Hash(second) & (JoinHashTable::MINIMUM_CAPACITY - 1)));

		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		BuildFactorHashTable(hash_table, allocator,
		                     {{Value::INTEGER(first), Value::INTEGER(second), Value::INTEGER(first)}},
		                     {Value::INTEGER(1), Value::INTEGER(2), Value::INTEGER(3)});
		REQUIRE(hash_table.FactorCount() == 2);

		TupleDataChunkState key_state;
		JoinHashTable::FactorProbeState probe_state;
		Vector factor_refs(LogicalType::UBIGINT);
		SelectionVector match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(hash_table, {{Value::INTEGER(first), Value::INTEGER(second)}}, allocator, key_state,
		                     probe_state, factor_refs, match_sel) == 2);
		auto refs = FlatVector::GetData<uint64_t>(factor_refs);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[0])) == 2);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[1])) == 1);
		REQUIRE(refs[0] != refs[1]);
	}

	SECTION("string keys and factors longer than one vector expand incrementally and concurrently") {
		vector<JoinCondition> conditions;
		conditions.emplace_back(make_uniq<BoundReferenceExpression>(LogicalType::VARCHAR, 0),
		                        make_uniq<BoundReferenceExpression>(LogicalType::VARCHAR, 0),
		                        ExpressionType::COMPARE_EQUAL);
		vector<idx_t> output_columns {1};
		JoinHashTable hash_table(context, dummy, conditions, {LogicalType::INTEGER}, JoinType::INNER, 0, output_columns,
		                         nullptr);
		const idx_t duplicate_count = STANDARD_VECTOR_SIZE * 2 + 17;
		vector<Value> keys;
		vector<Value> payloads;
		keys.reserve(duplicate_count + 1);
		payloads.reserve(duplicate_count + 1);
		for (idx_t row_idx = 0; row_idx < duplicate_count; row_idx++) {
			keys.push_back(Value("hot"));
			payloads.push_back(Value::INTEGER(NumericCast<int32_t>(row_idx)));
		}
		keys.push_back(Value("cold"));
		payloads.push_back(Value::INTEGER(-1));
		BuildFactorHashTable(hash_table, allocator, {keys}, payloads);
		REQUIRE(hash_table.FactorCount() == 2);

		TupleDataChunkState key_state;
		JoinHashTable::FactorProbeState probe_state;
		Vector factor_refs(LogicalType::UBIGINT);
		SelectionVector match_sel(STANDARD_VECTOR_SIZE);
		REQUIRE(ProbeFactors(hash_table, {{Value("hot")}}, allocator, key_state, probe_state, factor_refs, match_sel) ==
		        1);
		auto refs = FlatVector::GetData<uint64_t>(factor_refs);
		REQUIRE(hash_table.GetFactorCount(JoinFactorRef(refs[0])) == duplicate_count);

		JoinHashTable::FactorExpansionState expansion_state;
		Vector build_rows(LogicalType::POINTER);
		SelectionVector source_sel(STANDARD_VECTOR_SIZE);
		idx_t expanded_total = 0;
		idx_t batches = 0;
		while (!expansion_state.Finished(1)) {
			auto expanded =
			    hash_table.ExpandFactorRefs(factor_refs, nullptr, 1, expansion_state, build_rows, source_sel);
			REQUIRE(expanded > 0);
			expanded_total += expanded;
			batches++;
		}
		REQUIRE(expanded_total == duplicate_count);
		REQUIRE(batches == 3);

		std::atomic<idx_t> concurrent_total {0};
		vector<std::thread> workers;
		for (idx_t worker_idx = 0; worker_idx < 4; worker_idx++) {
			workers.emplace_back([&]() {
				Vector local_refs(LogicalType::UBIGINT);
				auto local_ref_data = FlatVector::GetDataMutable<uint64_t>(local_refs);
				local_ref_data[0] = refs[0];
				FlatVector::SetSize(local_refs, 1);
				JoinHashTable::FactorExpansionState local_state;
				Vector local_rows(LogicalType::POINTER);
				SelectionVector local_sources(STANDARD_VECTOR_SIZE);
				idx_t local_total = 0;
				while (!local_state.Finished(1)) {
					local_total +=
					    hash_table.ExpandFactorRefs(local_refs, nullptr, 1, local_state, local_rows, local_sources);
				}
				concurrent_total += local_total;
			});
		}
		for (auto &worker : workers) {
			worker.join();
		}
		REQUIRE(concurrent_total == duplicate_count * 4);
	}
}
