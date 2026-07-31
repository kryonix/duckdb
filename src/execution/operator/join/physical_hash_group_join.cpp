#include "duckdb/execution/operator/join/physical_hash_group_join.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

PhysicalHashGroupJoin::PhysicalHashGroupJoin(PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &probe,
                                             PhysicalOperator &owner, vector<unique_ptr<Expression>> aggregates,
                                             vector<unique_ptr<Expression>> groups, idx_t estimated_cardinality)
    : PhysicalJoin(physical_plan, op, PhysicalOperatorType::HASH_GROUP_JOIN, JoinType::INNER, estimated_cardinality) {
	grouped_aggregate_data.InitializeGroupby(std::move(groups), std::move(aggregates), {});
	for (idx_t aggregate_idx = 0; aggregate_idx < grouped_aggregate_data.aggregates.size(); aggregate_idx++) {
		aggregate_filter.push_back(aggregate_idx);
	}
	children.push_back(probe);
	children.push_back(owner);
}

class HashGroupJoinGlobalSinkState : public GlobalSinkState {
public:
	HashGroupJoinGlobalSinkState(const PhysicalHashGroupJoin &op, ClientContext &context)
	    : hash_table(make_uniq<GroupedAggregateHashTable>(
	          context, BufferAllocator::Get(context), op.grouped_aggregate_data.group_types,
	          op.grouped_aggregate_data.payload_types, op.grouped_aggregate_data.bindings,
	          GroupedAggregateHashTable::InitialCapacity(), 0, TupleDataValidityType::CANNOT_HAVE_NULL_VALUES)) {
	}

	unique_ptr<GroupedAggregateHashTable> hash_table;
	bool build_finalized = false;
	idx_t owner_rows = 0;
	idx_t probe_rows = 0;
	idx_t matched_rows = 0;
};

unique_ptr<GlobalSinkState> PhysicalHashGroupJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<HashGroupJoinGlobalSinkState>(*this, context);
}

class HashGroupJoinLocalSinkState : public LocalSinkState {
public:
	HashGroupJoinLocalSinkState() : addresses(LogicalType::POINTER), new_groups(STANDARD_VECTOR_SIZE) {
	}

	Vector addresses;
	SelectionVector new_groups;
};

unique_ptr<LocalSinkState> PhysicalHashGroupJoin::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<HashGroupJoinLocalSinkState>();
}

SinkResultType PhysicalHashGroupJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSinkInput &input) const {
	auto &global_state = input.global_state.Cast<HashGroupJoinGlobalSinkState>();
	auto &local_state = input.local_state.Cast<HashGroupJoinLocalSinkState>();
	if (global_state.build_finalized) {
		throw InternalException("Cannot add owner rows after HASH_GROUP_JOIN build finalization");
	}
	if (PhysicalJoin::HasNullValues(chunk)) {
		throw InternalException("HASH_GROUP_JOIN owner key unexpectedly contained NULL");
	}
	const auto new_group_count =
	    global_state.hash_table->FindOrCreateGroups(chunk, local_state.addresses, local_state.new_groups);
	if (new_group_count != chunk.size()) {
		throw InternalException("HASH_GROUP_JOIN owner key uniqueness proof was violated at execution time");
	}
	global_state.owner_rows += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalHashGroupJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                 OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<HashGroupJoinGlobalSinkState>();
	state.build_finalized = true;
	return state.hash_table->Count() == 0 ? SinkFinalizeType::NO_OUTPUT_POSSIBLE : SinkFinalizeType::READY;
}

static idx_t SelectNonNullGroupJoinKeys(DataChunk &keys, vector<UnifiedVectorFormat> &formats,
                                        SelectionVector &result) {
	for (idx_t key_idx = 0; key_idx < keys.ColumnCount(); key_idx++) {
		keys.data[key_idx].ToUnifiedFormat(formats[key_idx]);
	}
	idx_t result_count = 0;
	for (idx_t row_idx = 0; row_idx < keys.size(); row_idx++) {
		bool valid = true;
		for (auto &format : formats) {
			valid = valid && format.validity.RowIsValid(format.sel->get_index(row_idx));
		}
		if (valid) {
			result.set_index(result_count++, row_idx);
		}
	}
	return result_count;
}

class HashGroupJoinOperatorState : public CachingOperatorState {
public:
	explicit HashGroupJoinOperatorState(const PhysicalHashGroupJoin &op)
	    : non_null_sel(STANDARD_VECTOR_SIZE), found_sel(STANDARD_VECTOR_SIZE), matched_input_sel(STANDARD_VECTOR_SIZE),
	      matched_addresses(LogicalType::POINTER), key_formats(op.grouped_aggregate_data.group_types.size()),
	      update_state(*op.sink_state->Cast<HashGroupJoinGlobalSinkState>().hash_table) {
		probe_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		lookup_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
	}

	DataChunk probe_keys;
	DataChunk lookup_keys;
	DataChunk payload;
	SelectionVector non_null_sel;
	SelectionVector found_sel;
	SelectionVector matched_input_sel;
	Vector matched_addresses;
	AggregateHTLookupState lookup_state;
	vector<UnifiedVectorFormat> key_formats;
	AggregateHTUpdateState update_state;
};

unique_ptr<OperatorState> PhysicalHashGroupJoin::GetOperatorState(ExecutionContext &) const {
	return make_uniq<HashGroupJoinOperatorState>(*this);
}

OperatorResultType PhysicalHashGroupJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                          GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &sink = sink_state->Cast<HashGroupJoinGlobalSinkState>();
	auto &state = state_p.Cast<HashGroupJoinOperatorState>();
	D_ASSERT(sink.build_finalized);
	chunk.Reset();
	sink.probe_rows += input.size();

	const auto group_count = grouped_aggregate_data.group_types.size();
	state.probe_keys.Reset();
	for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
		state.probe_keys.data[group_idx].Reference(input.data[group_idx]);
	}
	state.probe_keys.CheckCardinality(input.size());
	const auto non_null_count = SelectNonNullGroupJoinKeys(state.probe_keys, state.key_formats, state.non_null_sel);
	if (non_null_count == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	state.lookup_keys.Reset();
	if (non_null_count == input.size()) {
		state.lookup_keys.Reference(state.probe_keys);
	} else {
		state.lookup_keys.Slice(state.probe_keys, state.non_null_sel, non_null_count);
	}
	const auto match_count = sink.hash_table->LookupGroups(state.lookup_keys, state.lookup_state, state.found_sel);
	if (match_count == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	const auto all_input_rows_match = non_null_count == input.size() && match_count == input.size();
	state.payload.Reset();
	if (all_input_rows_match) {
		for (idx_t payload_idx = 0; payload_idx < grouped_aggregate_data.payload_types.size(); payload_idx++) {
			state.payload.data[payload_idx].Reference(input.data[group_count + payload_idx]);
		}
		if (state.payload.ColumnCount() == 0) {
			state.payload.SetChildCardinality(match_count);
		} else {
			state.payload.CheckCardinality(match_count);
		}
		sink.hash_table->UpdateAggregatesAtAddresses(state.update_state, state.lookup_state.addresses, state.payload,
		                                             aggregate_filter);
		sink.matched_rows += match_count;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	state.matched_addresses.SetVectorType(VectorType::FLAT_VECTOR);
	auto matched_addresses = FlatVector::GetDataMutable<data_ptr_t>(state.matched_addresses);
	auto lookup_addresses = FlatVector::GetData<data_ptr_t>(state.lookup_state.addresses);
	for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
		const auto lookup_idx = state.found_sel.get_index_unsafe(match_idx);
		const auto input_idx =
		    non_null_count == input.size() ? lookup_idx : state.non_null_sel.get_index_unsafe(lookup_idx);
		state.matched_input_sel.set_index(match_idx, input_idx);
		matched_addresses[match_idx] = lookup_addresses[lookup_idx];
	}
	FlatVector::SetSize(state.matched_addresses, match_count);

	for (idx_t payload_idx = 0; payload_idx < grouped_aggregate_data.payload_types.size(); payload_idx++) {
		state.payload.data[payload_idx].Slice(input.data[group_count + payload_idx], state.matched_input_sel,
		                                      match_count);
	}
	if (state.payload.ColumnCount() == 0) {
		state.payload.SetChildCardinality(match_count);
	} else {
		state.payload.CheckCardinality(match_count);
	}
	sink.hash_table->UpdateAggregatesAtAddresses(state.update_state, state.matched_addresses, state.payload,
	                                             aggregate_filter);
	sink.matched_rows += match_count;
	return OperatorResultType::NEED_MORE_INPUT;
}

class HashGroupJoinGlobalSourceState : public GlobalSourceState {
public:
	explicit HashGroupJoinGlobalSourceState(const PhysicalHashGroupJoin &op) {
		auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
		D_ASSERT(sink.build_finalized);
		sink.hash_table->InitializeScan(scan_state);
	}

	AggregateHTScanState scan_state;
};

class HashGroupJoinLocalSourceState : public LocalSourceState {
public:
	HashGroupJoinLocalSourceState(ExecutionContext &context, const PhysicalHashGroupJoin &op)
	    : row_addresses(LogicalType::POINTER), matched_addresses(LogicalType::POINTER),
	      matched_sel(STANDARD_VECTOR_SIZE), arena(Allocator::Get(context.client)), row_state(arena) {
		groups.Initialize(Allocator::Get(context.client), op.grouped_aggregate_data.group_types);
		hidden_count.Initialize(Allocator::Get(context.client), {LogicalType::BIGINT});
	}

	DataChunk groups;
	DataChunk hidden_count;
	Vector row_addresses;
	Vector matched_addresses;
	SelectionVector matched_sel;
	ArenaAllocator arena;
	RowOperationsState row_state;
};

unique_ptr<GlobalSourceState> PhysicalHashGroupJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<HashGroupJoinGlobalSourceState>(*this);
}

unique_ptr<LocalSourceState> PhysicalHashGroupJoin::GetLocalSourceState(ExecutionContext &context,
                                                                        GlobalSourceState &gstate) const {
	return make_uniq<HashGroupJoinLocalSourceState>(context, *this);
}

SourceResultType PhysicalHashGroupJoin::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSourceInput &input) const {
	auto &sink = sink_state->Cast<HashGroupJoinGlobalSinkState>();
	auto &gstate = input.global_state.Cast<HashGroupJoinGlobalSourceState>();
	auto &state = input.local_state.Cast<HashGroupJoinLocalSourceState>();
	auto layout = sink.hash_table->GetLayoutPtr();
	const auto group_count = grouped_aggregate_data.group_types.size();
	const auto user_aggregate_count = grouped_aggregate_data.aggregates.size() - 1;

	while (sink.hash_table->ScanGroupsAndAddresses(gstate.scan_state, state.groups, state.row_addresses)) {
		if (state.groups.size() == 0) {
			continue;
		}
		state.hidden_count.Reset();
		state.hidden_count.SetChildCardinality(state.groups.size());
		RowOperations::FinalizeStatesRange(state.row_state, *layout, state.row_addresses, state.hidden_count, 0, 0, 1);
		state.hidden_count.Flatten();
		auto counts = FlatVector::GetData<int64_t>(state.hidden_count.data[0]);
		idx_t matched_count = 0;
		for (idx_t row_idx = 0; row_idx < state.groups.size(); row_idx++) {
			if (counts[row_idx] > 0) {
				state.matched_sel.set_index(matched_count++, row_idx);
			}
		}
		if (matched_count == 0) {
			continue;
		}

		chunk.Reset();
		for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
			chunk.data[group_idx].Slice(state.groups.data[group_idx], state.matched_sel, matched_count);
		}
		chunk.SetChildCardinality(matched_count);
		state.matched_addresses.Slice(state.row_addresses, state.matched_sel, matched_count);
		state.matched_addresses.Flatten();
		RowOperations::FinalizeStatesRange(state.row_state, *layout, state.matched_addresses, chunk, group_count, 1,
		                                   user_aggregate_count);
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	return SourceResultType::FINISHED;
}

string PhysicalHashGroupJoin::GetName() const {
	return "HASH_GROUP_JOIN";
}

InsertionOrderPreservingMap<string> PhysicalHashGroupJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = "INNER";
	result["Build Side"] = "OWNER";
	string groups;
	for (idx_t group_idx = 0; group_idx < grouped_aggregate_data.groups.size(); group_idx++) {
		if (group_idx > 0) {
			groups += "\n";
		}
		groups += grouped_aggregate_data.groups[group_idx]->GetName();
	}
	result["Groups"] = groups;
	string aggregates;
	for (idx_t aggregate_idx = 1; aggregate_idx < grouped_aggregate_data.aggregates.size(); aggregate_idx++) {
		if (aggregate_idx > 1) {
			aggregates += "\n";
		}
		aggregates += grouped_aggregate_data.aggregates[aggregate_idx]->GetName();
	}
	result["Aggregates"] = aggregates;
	result["Strategy"] = "SERIAL";
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
