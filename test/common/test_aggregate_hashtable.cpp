#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/operator/aggregate/aggregate_object.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/main/client_context.hpp"

using namespace duckdb;

static void SetIntegerValues(DataChunk &chunk, const vector<Value> &values) {
	for (idx_t row_idx = 0; row_idx < values.size(); row_idx++) {
		chunk.data[0].SetValue(row_idx, values[row_idx]);
	}
	chunk.SetChildCardinality(values.size());
}

static AggregateObject CreateCountAggregate(bool count_star) {
	auto function = BoundAggregateFunction(count_star ? CountStarFun::GetFunction() : CountFunctionBase::GetFunction());
	auto payload_size = AlignValue(function.GetStateSize(nullptr));
	return AggregateObject(std::move(function), nullptr, count_star ? 0 : 1, payload_size, AggregateType::NON_DISTINCT,
	                       PhysicalType::INT64);
}

TEST_CASE("Grouped aggregate address lookup and update", "[aggregate_hashtable]") {
	DuckDB database(nullptr);
	Connection connection(database);
	auto &context = *connection.context;
	auto &allocator = Allocator::Get(context);

	SECTION("linear probing resolves forced hash collisions") {
		GroupedAggregateHashTable hash_table(context, allocator, {LogicalType::INTEGER},
		                                     TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
		DataChunk groups;
		groups.Initialize(allocator, {LogicalType::INTEGER});
		SetIntegerValues(groups, {Value::INTEGER(10), Value::INTEGER(20), Value::INTEGER(30)});

		Vector hashes(LogicalType::HASH, groups.size());
		auto hash_data = FlatVector::GetDataMutable<hash_t>(hashes);
		for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
			hash_data[row_idx] = 42;
		}
		FlatVector::SetSize(hashes, groups.size());

		Vector addresses(LogicalType::POINTER);
		SelectionVector new_groups(STANDARD_VECTOR_SIZE);
		REQUIRE(hash_table.FindOrCreateGroups(groups, hashes, addresses, new_groups) == 3);
		REQUIRE(hash_table.Count() == 3);
	}

	vector<AggregateObject> aggregates;
	aggregates.push_back(CreateCountAggregate(true));
	aggregates.push_back(CreateCountAggregate(false));
	GroupedAggregateHashTable hash_table(context, allocator, {LogicalType::INTEGER}, {LogicalType::INTEGER},
	                                     std::move(aggregates), GroupedAggregateHashTable::InitialCapacity(), 0,
	                                     TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);

	DataChunk groups;
	groups.Initialize(allocator, {LogicalType::INTEGER});
	SetIntegerValues(groups, {Value::INTEGER(10), Value::INTEGER(20), Value::INTEGER(30)});
	Vector build_addresses(LogicalType::POINTER);
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	REQUIRE(hash_table.FindOrCreateGroups(groups, build_addresses, new_groups) == 3);

	DataChunk lookup_keys;
	lookup_keys.Initialize(allocator, {LogicalType::INTEGER});
	SetIntegerValues(lookup_keys, {Value::INTEGER(20), Value::INTEGER(99), Value::INTEGER(10)});
	AggregateHTLookupState lookup_state;
	SelectionVector found(STANDARD_VECTOR_SIZE);
	REQUIRE(hash_table.LookupGroups(lookup_keys, lookup_state, found) == 2);
	REQUIRE(found.get_index(0) == 0);
	REQUIRE(found.get_index(1) == 2);
	REQUIRE(hash_table.Count() == 3);

	DataChunk nullable_keys;
	nullable_keys.Initialize(allocator, {LogicalType::INTEGER});
	SetIntegerValues(nullable_keys, {Value::INTEGER(20), Value(LogicalType::INTEGER), Value::INTEGER(10)});
	SelectionVector non_null(STANDARD_VECTOR_SIZE);
	non_null.set_index(0, 0);
	non_null.set_index(1, 2);
	DataChunk filtered_keys;
	filtered_keys.InitializeEmpty({LogicalType::INTEGER});
	filtered_keys.Slice(nullable_keys, non_null, 2);
	REQUIRE(hash_table.LookupGroups(filtered_keys, lookup_state, found) == 2);

	DataChunk nullable_payload;
	nullable_payload.Initialize(allocator, {LogicalType::INTEGER});
	SetIntegerValues(nullable_payload, {Value::INTEGER(5), Value::INTEGER(999), Value(LogicalType::INTEGER)});
	DataChunk filtered_payload;
	filtered_payload.InitializeEmpty({LogicalType::INTEGER});
	filtered_payload.Slice(nullable_payload, non_null, 2);

	Vector matched_addresses(LogicalType::POINTER);
	matched_addresses.Slice(lookup_state.addresses, found, 2);
	matched_addresses.Flatten();
	DataChunk matched_payload;
	matched_payload.InitializeEmpty({LogicalType::INTEGER});
	matched_payload.Slice(filtered_payload, found, 2);

	AggregateHTUpdateState update_state(hash_table);
	unsafe_vector<idx_t> aggregate_filter {0, 1};
	vector<AggregateObject> other_aggregates;
	other_aggregates.push_back(CreateCountAggregate(true));
	other_aggregates.push_back(CreateCountAggregate(false));
	GroupedAggregateHashTable other_hash_table(
	    context, allocator, {LogicalType::INTEGER}, {LogicalType::INTEGER}, std::move(other_aggregates),
	    GroupedAggregateHashTable::InitialCapacity(), 0, TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
	REQUIRE_THROWS(other_hash_table.UpdateAggregatesAtAddresses(update_state, matched_addresses, matched_payload,
	                                                            aggregate_filter));
	hash_table.UpdateAggregatesAtAddresses(update_state, matched_addresses, matched_payload, aggregate_filter);
	REQUIRE(hash_table.Count() == 3);

	AggregateHTScanState scan_state;
	hash_table.InitializeScan(scan_state);
	DataChunk scanned_groups;
	scanned_groups.Initialize(allocator, {LogicalType::INTEGER});
	DataChunk results;
	results.Initialize(allocator, {LogicalType::BIGINT, LogicalType::BIGINT});
	Vector scan_addresses(LogicalType::POINTER);
	ArenaAllocator finalize_allocator(allocator);
	RowOperationsState finalize_state(finalize_allocator);

	unordered_map<int32_t, pair<int64_t, int64_t>> counts;
	while (hash_table.ScanGroupsAndAddresses(scan_state, scanned_groups, scan_addresses)) {
		if (scanned_groups.size() == 0) {
			continue;
		}
		results.Reset();
		results.SetChildCardinality(scanned_groups.size());
		RowOperations::FinalizeStatesRange(finalize_state, *hash_table.GetLayoutPtr(), scan_addresses, results, 0, 0,
		                                   2);
		for (idx_t row_idx = 0; row_idx < scanned_groups.size(); row_idx++) {
			counts[scanned_groups.GetValue(0, row_idx).GetValue<int32_t>()] = make_pair(
			    results.GetValue(0, row_idx).GetValue<int64_t>(), results.GetValue(1, row_idx).GetValue<int64_t>());
		}
	}

	REQUIRE(counts.size() == 3);
	REQUIRE(counts[10] == make_pair<int64_t, int64_t>(1, 0));
	REQUIRE(counts[20] == make_pair<int64_t, int64_t>(1, 1));
	REQUIRE(counts[30] == make_pair<int64_t, int64_t>(0, 0));
}
