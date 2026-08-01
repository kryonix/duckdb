#include "duckdb/execution/operator/join/physical_index_group_join.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/storage_compatibility.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/table_filter_state.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/table/column_segment.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/duck_transaction_manager.hpp"
#include "duckdb/transaction/local_storage.hpp"

namespace duckdb {

static vector<AggregateObject> CreateIndexGroupJoinAggregates(const PhysicalIndexGroupJoin &op) {
	vector<AggregateObject> result;
	result.reserve(op.owner_payload_data.bindings.size() + op.grouped_aggregate_data.bindings.size());
	for (auto binding : op.owner_payload_data.bindings) {
		result.emplace_back(binding);
	}
	for (auto binding : op.grouped_aggregate_data.bindings) {
		result.emplace_back(binding);
	}
	return result;
}

static idx_t GetIndexGroupJoinAggregateOffset(const PhysicalIndexGroupJoin &op) {
	return op.owner_payload_data.aggregates.size();
}

static vector<LogicalType> GetIndexGroupJoinExpressionTypes(const vector<unique_ptr<Expression>> &expressions) {
	vector<LogicalType> result;
	result.reserve(expressions.size());
	for (auto &expression : expressions) {
		result.push_back(expression->GetReturnType());
	}
	return result;
}

PhysicalIndexGroupJoin::PhysicalIndexGroupJoin(
    PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &owner, DuckTableEntry &probe_table_p,
    Identifier index_name_p, vector<StorageIndex> probe_column_ids_p, vector<LogicalType> probe_scan_types_p,
    vector<idx_t> probe_projection_ids_p, vector<LogicalType> probe_input_types_p,
    unique_ptr<TableFilterSet> probe_filters_p, vector<unique_ptr<Expression>> probe_residual_filters_p,
    vector<unique_ptr<Expression>> probe_expressions_p, vector<idx_t> index_key_map_p,
    vector<LogicalType> index_key_types_p, vector<unique_ptr<Expression>> aggregates,
    vector<unique_ptr<Expression>> owner_payload_aggregates, vector<unique_ptr<Expression>> groups,
    vector<HashGroupJoinOutputColumn> output_groups_p, HashGroupJoinUnmatchedPolicy unmatched_policy_p,
    idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::HASH_GROUP_JOIN, op.types, estimated_cardinality),
      output_groups(std::move(output_groups_p)), unmatched_policy(unmatched_policy_p), probe_table(probe_table_p),
      index_name(std::move(index_name_p)), probe_column_ids(std::move(probe_column_ids_p)),
      probe_scan_types(std::move(probe_scan_types_p)), probe_projection_ids(std::move(probe_projection_ids_p)),
      probe_input_types(std::move(probe_input_types_p)), probe_filters(std::move(probe_filters_p)),
      probe_residual_filters(std::move(probe_residual_filters_p)), probe_expressions(std::move(probe_expressions_p)),
      index_key_map(std::move(index_key_map_p)), index_key_types(std::move(index_key_types_p)) {
	for (auto &group : op.groups) {
		output_group_names.push_back(group->GetName().GetIdentifierName());
	}
	grouped_aggregate_data.InitializeGroupby(std::move(groups), std::move(aggregates), {});
	owner_payload_data.InitializeGroupby({}, std::move(owner_payload_aggregates), {});
	const auto key_count = grouped_aggregate_data.group_types.size();
	for (idx_t expression_idx = key_count; expression_idx < probe_expressions.size(); expression_idx++) {
		unmatched_payload_expressions.push_back(probe_expressions[expression_idx]->Copy());
	}
	idx_t payload_index = 0;
	for (idx_t aggregate_idx = 0; aggregate_idx < grouped_aggregate_data.aggregates.size(); aggregate_idx++) {
		auto &aggregate = grouped_aggregate_data.aggregates[aggregate_idx]->Cast<BoundAggregateExpression>();
		if (aggregate.IsDistinct()) {
			vector<LogicalType> argument_types;
			for (auto &child : aggregate.GetChildren()) {
				argument_types.push_back(child->GetReturnType());
			}
			distinct_aggregates.push_back(
			    {aggregate_idx, payload_index, std::move(argument_types), aggregate.GetFilter() != nullptr});
		} else {
			non_distinct_filter.push_back(aggregate_idx);
		}
		payload_index += aggregate.GetChildren().size();
	}
	children.push_back(owner);
}

class IndexGroupJoinOperatorState : public OperatorState {
public:
	IndexGroupJoinOperatorState(ExecutionContext &context, const PhysicalIndexGroupJoin &op)
	    : probe_executor(context.client, op.probe_expressions), owner_addresses(LogicalType::POINTER),
	      new_groups(STANDARD_VECTOR_SIZE), update_addresses(LogicalType::POINTER), row_ids(LogicalType::ROW_TYPE),
	      row_addresses(LogicalType::POINTER), selected_addresses(LogicalType::POINTER),
	      distinct_addresses(LogicalType::POINTER), distinct_insert_addresses(LogicalType::POINTER),
	      new_distinct_addresses(LogicalType::POINTER), distinct_new_groups(STANDARD_VECTOR_SIZE),
	      selected(STANDARD_VECTOR_SIZE), arena(Allocator::Get(context.client)), row_state(arena) {
		owner_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		owner_payload.InitializeEmpty(op.owner_payload_data.payload_types);
		index_keys.Initialize(Allocator::Get(context.client), op.index_key_types);
		fetched.Initialize(Allocator::Get(context.client), op.probe_scan_types);
		probe_input.Initialize(Allocator::Get(context.client), op.probe_input_types);
		projected_probe.Initialize(Allocator::Get(context.client),
		                           GetIndexGroupJoinExpressionTypes(op.probe_expressions));
		payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		matched_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		groups.Initialize(Allocator::Get(context.client), op.grouped_aggregate_data.group_types);
		group_ids.Initialize(Allocator::Get(context.client), {LogicalType::UBIGINT});
		hidden_count.Initialize(Allocator::Get(context.client), {LogicalType::BIGINT});

		auto aggregate_objects = AggregateObject::CreateAggregateObjects(op.grouped_aggregate_data.bindings);
		distinct_filter_set.Initialize(context.client, aggregate_objects, op.grouped_aggregate_data.payload_types);
		if (op.probe_filters) {
			for (auto &entry : *op.probe_filters) {
				probe_filter_columns.push_back(entry.GetIndex().GetIndex());
				probe_filter_states.push_back(TableFilterState::Initialize(context.client, entry.Filter()));
			}
		}
		for (auto &filter : op.probe_residual_filters) {
			probe_residual_executors.push_back(make_uniq<ExpressionExecutor>(context.client, *filter));
		}
		for (auto &distinct : op.distinct_aggregates) {
			vector<LogicalType> group_types {LogicalType::UBIGINT};
			group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			auto distinct_groups = make_uniq<DataChunk>();
			distinct_groups->InitializeEmpty(group_types);
			distinct_group_chunks.push_back(std::move(distinct_groups));
		}
	}

	ExpressionExecutor probe_executor;
	DataChunk owner_keys;
	DataChunk owner_payload;
	DataChunk index_keys;
	DataChunk fetched;
	DataChunk probe_input;
	DataChunk projected_probe;
	DataChunk payload;
	DataChunk matched_payload;
	DataChunk groups;
	DataChunk group_ids;
	DataChunk hidden_count;
	Vector owner_addresses;
	SelectionVector new_groups;
	Vector update_addresses;
	Vector row_ids;
	Vector row_addresses;
	Vector selected_addresses;
	Vector distinct_addresses;
	Vector distinct_insert_addresses;
	Vector new_distinct_addresses;
	SelectionVector distinct_new_groups;
	SelectionVector selected;
	ColumnFetchState global_fetch_state;
	unique_ptr<GroupedAggregateHashTable> target;
	unique_ptr<AggregateHTUpdateState> update_state;
	vector<data_ptr_t> group_addresses;
	unordered_map<data_ptr_t, idx_t> group_ids_by_address;
	AggregateHTLookupState lookup_state;
	SelectionVector found {STANDARD_VECTOR_SIZE};
	SelectionVector matched_input {STANDARD_VECTOR_SIZE};
	AggregateFilterDataSet distinct_filter_set;
	vector<idx_t> probe_filter_columns;
	vector<unique_ptr<TableFilterState>> probe_filter_states;
	vector<unique_ptr<ExpressionExecutor>> probe_residual_executors;
	SelectionVector probe_residual_selection {STANDARD_VECTOR_SIZE};
	vector<unique_ptr<GroupedAggregateHashTable>> distinct_tables;
	vector<unique_ptr<DataChunk>> distinct_group_chunks;
	ArenaAllocator arena;
	RowOperationsState row_state;
};

unique_ptr<OperatorState> PhysicalIndexGroupJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<IndexGroupJoinOperatorState>(context, *this);
}

static void SinkIndexGroupJoinDistinct(const PhysicalIndexGroupJoin &op, IndexGroupJoinOperatorState &state,
                                       DataChunk &group_ids, Vector &target_addresses, DataChunk &payload) {
	unsafe_vector<idx_t> distinct_filter {0};
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		auto &aggregate =
		    op.grouped_aggregate_data.aggregates[distinct.aggregate_index]->Cast<BoundAggregateExpression>();
		auto &groups = *state.distinct_group_chunks[distinct_idx];
		groups.Reset();
		idx_t count;
		optional_ptr<DataChunk> argument_payload;
		if (distinct.has_filter) {
			auto &filter_data = state.distinct_filter_set.GetFilterData(distinct.aggregate_index);
			count = filter_data.ApplyFilter(payload);
			if (count == 0) {
				continue;
			}
			groups.data[0].Slice(group_ids.data[0], filter_data.true_sel, count);
			state.distinct_addresses.Slice(target_addresses, filter_data.true_sel, count);
			state.distinct_addresses.Flatten();
			argument_payload = filter_data.filtered_payload;
		} else {
			count = payload.size();
			groups.data[0].Reference(group_ids.data[0]);
			state.distinct_addresses.Reference(target_addresses);
			argument_payload = payload;
		}
		for (idx_t child_idx = 0; child_idx < aggregate.GetChildren().size(); child_idx++) {
			groups.data[child_idx + 1].Reference(argument_payload->data[distinct.payload_index + child_idx]);
		}
		groups.SetChildCardinality(count);
		auto new_count = state.distinct_tables[distinct_idx]->FindOrCreateGroups(
		    groups, state.distinct_insert_addresses, state.distinct_new_groups);
		if (new_count == 0) {
			continue;
		}
		state.new_distinct_addresses.Slice(state.distinct_addresses, state.distinct_new_groups, new_count);
		state.new_distinct_addresses.Flatten();
		vector<LogicalType> payload_types = distinct.argument_types;
		if (distinct.has_filter) {
			payload_types.push_back(LogicalType::BOOLEAN);
		}
		DataChunk distinct_payload;
		distinct_payload.InitializeEmpty(payload_types);
		for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
			distinct_payload.data[child_idx].Slice(argument_payload->data[distinct.payload_index + child_idx],
			                                       state.distinct_new_groups, new_count);
		}
		if (distinct.has_filter) {
			distinct_payload.data.back().Reference(Value::BOOLEAN(true), count_t(new_count));
		}
		distinct_payload.SetChildCardinality(new_count);
		state.target->UpdateAggregatesAtAddressesRange(
		    *state.update_state, state.new_distinct_addresses, distinct_payload,
		    GetIndexGroupJoinAggregateOffset(op) + distinct.aggregate_index, 1, distinct_filter);
	}
}

static bool PrepareIndexGroupJoinProbe(const PhysicalIndexGroupJoin &op, IndexGroupJoinOperatorState &state) {
	auto count = state.fetched.size();
	SelectionVector filter_selection;
	for (idx_t filter_idx = 0; filter_idx < state.probe_filter_states.size(); filter_idx++) {
		auto column_idx = state.probe_filter_columns[filter_idx];
		if (column_idx >= state.fetched.ColumnCount()) {
			throw InternalException("INDEX_GROUP_JOIN probe filter column is out of range");
		}
		ColumnSegment::FilterSelection(filter_selection, state.fetched.data[column_idx],
		                               *state.probe_filter_states[filter_idx], state.fetched.size(), count);
		if (count == 0) {
			return false;
		}
	}
	state.probe_input.Reset();
	for (idx_t output_idx = 0; output_idx < op.probe_projection_ids.size(); output_idx++) {
		auto source_idx = op.probe_projection_ids[output_idx];
		if (source_idx >= state.fetched.ColumnCount()) {
			throw InternalException("INDEX_GROUP_JOIN probe projection column is out of range");
		}
		if (count == state.fetched.size()) {
			state.probe_input.data[output_idx].Reference(state.fetched.data[source_idx]);
		} else {
			state.probe_input.data[output_idx].Slice(state.fetched.data[source_idx], filter_selection, count);
		}
	}
	state.probe_input.SetChildCardinality(count);
	for (auto &executor : state.probe_residual_executors) {
		auto filtered_count = executor->SelectExpression(state.probe_input, state.probe_residual_selection);
		if (filtered_count == 0) {
			return false;
		}
		if (filtered_count != state.probe_input.size()) {
			state.probe_input.Slice(state.probe_residual_selection, filtered_count);
		}
	}
	return true;
}

static void UpdateIndexGroupJoinPayload(const PhysicalIndexGroupJoin &op, IndexGroupJoinOperatorState &state,
                                        idx_t group_id, data_ptr_t target_address) {
	if (!PrepareIndexGroupJoinProbe(op, state)) {
		return;
	}
	state.projected_probe.Reset();
	state.probe_executor.Execute(state.probe_input, state.projected_probe);
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	state.payload.Reset();
	for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size(); payload_idx++) {
		state.payload.data[payload_idx].Reference(state.projected_probe.data[key_count + payload_idx]);
	}
	state.payload.SetChildCardinality(state.projected_probe.size());
	state.update_addresses.SetVectorType(VectorType::FLAT_VECTOR);
	auto addresses = FlatVector::GetDataMutable<data_ptr_t>(state.update_addresses);
	for (idx_t row_idx = 0; row_idx < state.payload.size(); row_idx++) {
		addresses[row_idx] = target_address;
	}
	FlatVector::SetSize(state.update_addresses, state.payload.size());
	state.group_ids.Reset();
	state.group_ids.data[0].Reference(Value::UBIGINT(group_id), count_t(state.payload.size()));
	state.group_ids.SetChildCardinality(state.payload.size());
	SinkIndexGroupJoinDistinct(op, state, state.group_ids, state.update_addresses, state.payload);
	state.target->UpdateAggregatesAtAddressesRange(*state.update_state, state.update_addresses, state.payload,
	                                               GetIndexGroupJoinAggregateOffset(op),
	                                               op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
}

static void FetchIndexGroupJoinRows(const PhysicalIndexGroupJoin &op, ClientContext &context,
                                    IndexGroupJoinOperatorState &state, const set<row_t> &matches, idx_t group_id,
                                    data_ptr_t target_address) {
	if (matches.empty()) {
		return;
	}
	auto &storage = op.probe_table.get().GetStorage();
	auto &transaction = DuckTransaction::Get(context, op.probe_table.get().catalog);
	auto entry = matches.begin();
	while (entry != matches.end()) {
		idx_t count = 0;
		state.row_ids.SetVectorType(VectorType::FLAT_VECTOR);
		auto row_ids = FlatVector::GetDataMutable<row_t>(state.row_ids);
		while (entry != matches.end() && count < STANDARD_VECTOR_SIZE) {
			row_ids[count++] = *entry++;
		}
		FlatVector::SetSize(state.row_ids, count);
		state.fetched.Reset();
		storage.Fetch(transaction, state.fetched, op.probe_column_ids, state.row_ids, count, state.global_fetch_state);
		if (state.fetched.size() != 0) {
			UpdateIndexGroupJoinPayload(op, state, group_id, target_address);
		}
	}
}

static void UpdateIndexGroupJoinScannedRows(const PhysicalIndexGroupJoin &op, IndexGroupJoinOperatorState &state) {
	if (!PrepareIndexGroupJoinProbe(op, state)) {
		return;
	}
	state.projected_probe.Reset();
	state.probe_executor.Execute(state.probe_input, state.projected_probe);
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	state.groups.Reset();
	for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
		state.groups.data[key_idx].Reference(state.projected_probe.data[key_idx]);
	}
	state.groups.SetChildCardinality(state.projected_probe.size());
	auto match_count = state.target->LookupGroups(state.groups, state.lookup_state, state.found);
	if (match_count == 0) {
		return;
	}

	state.selected_addresses.SetVectorType(VectorType::FLAT_VECTOR);
	auto target_addresses = FlatVector::GetDataMutable<data_ptr_t>(state.selected_addresses);
	auto lookup_addresses = FlatVector::GetData<data_ptr_t>(state.lookup_state.addresses);
	state.matched_payload.Reset();
	state.group_ids.Reset();
	state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
	auto group_ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
	for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
		auto input_idx = state.found.get_index_unsafe(match_idx);
		auto address = lookup_addresses[input_idx];
		state.matched_input.set_index(match_idx, input_idx);
		target_addresses[match_idx] = address;
		auto group_entry = state.group_ids_by_address.find(address);
		if (group_entry == state.group_ids_by_address.end()) {
			throw InternalException("INDEX_GROUP_JOIN scanned row found an unknown owner group");
		}
		group_ids[match_idx] = group_entry->second;
	}
	FlatVector::SetSize(state.selected_addresses, match_count);
	FlatVector::SetSize(state.group_ids.data[0], match_count);
	state.group_ids.SetChildCardinality(match_count);
	for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size(); payload_idx++) {
		state.matched_payload.data[payload_idx].Slice(state.projected_probe.data[key_count + payload_idx],
		                                              state.matched_input, match_count);
	}
	state.matched_payload.SetChildCardinality(match_count);
	SinkIndexGroupJoinDistinct(op, state, state.group_ids, state.selected_addresses, state.matched_payload);
	state.target->UpdateAggregatesAtAddressesRange(*state.update_state, state.selected_addresses, state.matched_payload,
	                                               GetIndexGroupJoinAggregateOffset(op),
	                                               op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
}

static void ScanIndexGroupJoinLocalRows(const PhysicalIndexGroupJoin &op, ClientContext &context,
                                        IndexGroupJoinOperatorState &state) {
	auto &storage = op.probe_table.get().GetStorage();
	auto &local_storage = LocalStorage::Get(context, op.probe_table.get().catalog);
	auto local_table = local_storage.GetStorage(storage);
	if (!local_table || local_table->GetCollection().GetTotalRows() == 0) {
		return;
	}

	TableScanState scan_state;
	scan_state.Initialize(op.probe_column_ids, context);
	local_storage.InitializeScan(storage, scan_state.local_state, nullptr);
	DataChunk local_rows;
	local_rows.Initialize(Allocator::Get(context), op.probe_scan_types);
	while (true) {
		local_rows.Reset();
		local_storage.Scan(scan_state.local_state, op.probe_column_ids, local_rows);
		if (local_rows.size() == 0) {
			break;
		}
		state.fetched.Reference(local_rows);
		UpdateIndexGroupJoinScannedRows(op, state);
	}
}

static void ScanAllIndexGroupJoinRows(const PhysicalIndexGroupJoin &op, ClientContext &context,
                                      IndexGroupJoinOperatorState &state) {
	auto &storage = op.probe_table.get().GetStorage();
	auto &transaction = DuckTransaction::Get(context, op.probe_table.get().catalog);
	TableScanState scan_state;
	storage.InitializeScan(context, transaction, scan_state, op.probe_column_ids, nullptr);
	DataChunk rows;
	rows.Initialize(Allocator::Get(context), op.probe_scan_types);
	while (true) {
		rows.Reset();
		storage.Scan(transaction, rows, scan_state);
		if (rows.size() == 0) {
			break;
		}
		state.fetched.Reference(rows);
		UpdateIndexGroupJoinScannedRows(op, state);
	}
}

static void UpdateIndexGroupJoinUnmatched(const PhysicalIndexGroupJoin &op, ClientContext &context,
                                          IndexGroupJoinOperatorState &state) {
	if (op.unmatched_policy != HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW) {
		return;
	}
	ExpressionExecutor executor(context);
	for (auto &expression : op.unmatched_payload_expressions) {
		executor.AddExpression(*expression);
	}
	DataChunk null_probe;
	null_probe.Initialize(Allocator::Get(context), op.probe_input_types);
	DataChunk unmatched_payload;
	unmatched_payload.Initialize(Allocator::Get(context), op.grouped_aggregate_data.payload_types);
	Vector unmatched_addresses(LogicalType::POINTER);
	SelectionVector unmatched_sel(STANDARD_VECTOR_SIZE);
	state.hidden_count.Reset();
	state.hidden_count.SetChildCardinality(state.group_addresses.size());
	RowOperations::FinalizeStatesRange(state.row_state, *state.target->GetLayoutPtr(), state.owner_addresses,
	                                   state.hidden_count, 0, GetIndexGroupJoinAggregateOffset(op), 1);
	state.hidden_count.Flatten();
	auto counts = FlatVector::GetData<int64_t>(state.hidden_count.data[0]);
	idx_t unmatched_count = 0;
	for (idx_t row_idx = 0; row_idx < state.group_addresses.size(); row_idx++) {
		if (counts[row_idx] == 0) {
			unmatched_sel.set_index(unmatched_count++, row_idx);
		}
	}
	if (unmatched_count == 0) {
		return;
	}
	unmatched_addresses.Slice(state.owner_addresses, unmatched_sel, unmatched_count);
	unmatched_addresses.Flatten();
	null_probe.Reset();
	for (idx_t column_idx = 0; column_idx < op.probe_input_types.size(); column_idx++) {
		null_probe.data[column_idx].Reference(Value(op.probe_input_types[column_idx]), count_t(unmatched_count));
	}
	null_probe.SetChildCardinality(unmatched_count);
	unmatched_payload.Reset();
	executor.Execute(null_probe, unmatched_payload);
	state.target->UpdateAggregatesAtAddressesRange(*state.update_state, unmatched_addresses, unmatched_payload,
	                                               GetIndexGroupJoinAggregateOffset(op),
	                                               op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
	state.group_ids.Reset();
	state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
	auto group_ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
	for (idx_t unmatched_idx = 0; unmatched_idx < unmatched_count; unmatched_idx++) {
		group_ids[unmatched_idx] = unmatched_sel.get_index_unsafe(unmatched_idx);
	}
	FlatVector::SetSize(state.group_ids.data[0], unmatched_count);
	state.group_ids.SetChildCardinality(unmatched_count);
	SinkIndexGroupJoinDistinct(op, state, state.group_ids, unmatched_addresses, unmatched_payload);
}

OperatorResultType PhysicalIndexGroupJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                   GlobalOperatorState &, OperatorState &state_p) const {
	auto &state = state_p.Cast<IndexGroupJoinOperatorState>();
	state.target = make_uniq<GroupedAggregateHashTable>(
	    context.client, BufferAllocator::Get(context.client), grouped_aggregate_data.group_types,
	    grouped_aggregate_data.payload_types, CreateIndexGroupJoinAggregates(*this),
	    GroupedAggregateHashTable::InitialCapacity(), idx_t(0), TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
	state.update_state = make_uniq<AggregateHTUpdateState>(*state.target);
	state.group_addresses.clear();
	state.group_ids_by_address.clear();
	state.distinct_tables.clear();
	for (auto &distinct : distinct_aggregates) {
		vector<LogicalType> group_types {LogicalType::UBIGINT};
		group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
		state.distinct_tables.push_back(make_uniq<GroupedAggregateHashTable>(
		    context.client, BufferAllocator::Get(context.client), group_types, vector<LogicalType> {},
		    vector<AggregateObject> {}, GroupedAggregateHashTable::InitialCapacity(), idx_t(2),
		    TupleDataValidityType::CAN_HAVE_NULL_VALUES));
	}

	const auto key_count = grouped_aggregate_data.group_types.size();
	state.owner_keys.Reset();
	for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
		state.owner_keys.data[key_idx].Reference(input.data[key_idx]);
	}
	state.owner_keys.SetChildCardinality(input.size());
	auto new_count = state.target->FindOrCreateGroups(state.owner_keys, state.owner_addresses, state.new_groups);
	if (new_count != input.size()) {
		throw InternalException("INDEX_GROUP_JOIN owner key uniqueness proof was violated");
	}
	auto owner_addresses = FlatVector::GetData<data_ptr_t>(state.owner_addresses);
	state.group_addresses.assign(owner_addresses, owner_addresses + input.size());
	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		state.group_ids_by_address.emplace(owner_addresses[row_idx], row_idx);
	}
	if (!owner_payload_data.aggregates.empty()) {
		state.owner_payload.Reset();
		for (idx_t payload_idx = 0; payload_idx < owner_payload_data.payload_types.size(); payload_idx++) {
			state.owner_payload.data[payload_idx].Reference(input.data[key_count + payload_idx]);
		}
		state.owner_payload.SetChildCardinality(input.size());
		state.target->UpdateAggregatesAtAddressesRange(*state.update_state, state.owner_addresses, state.owner_payload,
		                                               0, owner_payload_data.aggregates.size());
	}
	state.index_keys.Reset();
	for (idx_t index_idx = 0; index_idx < index_key_map.size(); index_idx++) {
		auto &source = input.data[index_key_map[index_idx]];
		if (source.GetType() == index_key_types[index_idx]) {
			state.index_keys.data[index_idx].Reference(source);
		} else {
			VectorOperations::Cast(context.client, source, state.index_keys.data[index_idx], input.size(), true);
		}
	}
	state.index_keys.SetChildCardinality(input.size());

	auto &storage = probe_table.get().GetStorage();
	auto &global_indexes = storage.GetDataTableInfo()->GetIndexes();
	unique_ptr<StorageLockKey> vacuum_lock;
	auto &attached = storage.GetAttached();
	const bool indexed_vacuum_may_move_rowids = attached.GetVacuumRebuildIndexThreshold() > 0 ||
	                                            StorageCompatibility::FromDatabase(attached).CanPersistRowIdGaps();
	if (indexed_vacuum_may_move_rowids) {
		vacuum_lock = DuckTransactionManager::Get(attached).SharedVacuumLock();
	}
	auto scan_percentage = Settings::Get<IndexScanPercentageSetting>(context.client);
	auto scan_max_count = Settings::Get<IndexScanMaxCountSetting>(context.client);
	auto percentage_limit = LossyNumericCast<idx_t>(double(storage.GetTotalRows()) * scan_percentage);
	auto row_id_limit = MaxValue<idx_t>(1, MaxValue(scan_max_count, percentage_limit));
	vector<set<row_t>> global_matches(input.size());
	idx_t total_matches = 0;
	bool scan_fallback = false;
	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		if (total_matches >= row_id_limit) {
			scan_fallback = true;
			break;
		}
		auto lookup_result = global_indexes.SearchART(index_name, state.index_keys, row_idx,
		                                              row_id_limit - total_matches, global_matches[row_idx]);
		if (lookup_result == ARTLookupResult::INDEX_NOT_FOUND) {
			throw InternalException("INDEX_GROUP_JOIN lost its bound ART index");
		}
		if (lookup_result == ARTLookupResult::EXCEEDED_LIMIT) {
			scan_fallback = true;
			break;
		}
		total_matches += global_matches[row_idx].size();
	}
	if (scan_fallback) {
		ScanAllIndexGroupJoinRows(*this, context.client, state);
	} else {
		for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
			FetchIndexGroupJoinRows(*this, context.client, state, global_matches[row_idx], row_idx,
			                        owner_addresses[row_idx]);
		}
		ScanIndexGroupJoinLocalRows(*this, context.client, state);
	}
	UpdateIndexGroupJoinUnmatched(*this, context.client, state);

	state.hidden_count.Reset();
	state.hidden_count.SetChildCardinality(input.size());
	RowOperations::FinalizeStatesRange(state.row_state, *state.target->GetLayoutPtr(), state.owner_addresses,
	                                   state.hidden_count, 0, GetIndexGroupJoinAggregateOffset(*this), 1);
	state.hidden_count.Flatten();
	auto counts = FlatVector::GetData<int64_t>(state.hidden_count.data[0]);
	idx_t selected_count = 0;
	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		if (unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD || counts[row_idx] != 0) {
			state.selected.set_index(selected_count++, row_idx);
		}
	}
	if (selected_count != 0) {
		chunk.Reset();
		for (idx_t group_idx = 0; group_idx < output_groups.size(); group_idx++) {
			auto &output_group = output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::KEY) {
				chunk.data[group_idx].Slice(state.owner_keys.data[output_group.index], state.selected, selected_count);
			}
		}
		chunk.SetChildCardinality(selected_count);
		state.selected_addresses.Slice(state.owner_addresses, state.selected, selected_count);
		state.selected_addresses.Flatten();
		for (idx_t group_idx = 0; group_idx < output_groups.size(); group_idx++) {
			auto &output_group = output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::OWNER_PAYLOAD) {
				RowOperations::FinalizeStatesRange(state.row_state, *state.target->GetLayoutPtr(),
				                                   state.selected_addresses, chunk, group_idx, output_group.index, 1);
			}
		}
		RowOperations::FinalizeStatesRange(state.row_state, *state.target->GetLayoutPtr(), state.selected_addresses,
		                                   chunk, output_groups.size(), GetIndexGroupJoinAggregateOffset(*this) + 1,
		                                   grouped_aggregate_data.aggregates.size() - 1);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

string PhysicalIndexGroupJoin::GetName() const {
	return "INDEX_GROUP_JOIN";
}

InsertionOrderPreservingMap<string> PhysicalIndexGroupJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Index"] = index_name.GetIdentifierName();
	result["Join Type"] = unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD ? "INNER" : "OWNER OUTER";
	result["Strategy"] = "INDEX";
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
