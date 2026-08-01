#include "duckdb/execution/operator/join/physical_hash_group_join.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/radix_partitioning.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/group_join_strategy.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"

namespace duckdb {

static constexpr idx_t GROUP_JOIN_LOCAL_RADIX_BITS = 2;
static constexpr idx_t GROUP_JOIN_EXTERNAL_RADIX_BITS = 4;

struct GroupJoinIdState {
	uint64_t value;
};

struct GroupJoinIdFunction {
	static void Initialize(GroupJoinIdState &state) {
		state.value = 0;
	}

	template <class STATE, class OP>
	static void Operation(STATE &, AggregateInputData &, idx_t) {
	}

	template <class STATE, class OP>
	static void ConstantOperation(STATE &, AggregateInputData &, idx_t) {
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		if (source.value != 0) {
			if (target.value != 0 && target.value != source.value) {
				throw InternalException("Cannot combine different HASH_GROUP_JOIN group identifiers");
			}
			target.value = source.value;
		}
	}

	template <class RESULT_TYPE, class STATE>
	static void Finalize(STATE &state, RESULT_TYPE &target, AggregateFinalizeData &) {
		target = state.value;
	}
};

static AggregateObject CreateGroupJoinIdAggregate() {
	auto function = BoundAggregateFunction(
	    AggregateFunction::NullaryAggregate<GroupJoinIdState, uint64_t, GroupJoinIdFunction>(LogicalType::UBIGINT));
	return AggregateObject(std::move(function), nullptr, 0, AlignValue(sizeof(GroupJoinIdState)),
	                       AggregateType::NON_DISTINCT, PhysicalType::UINT64);
}

static vector<AggregateObject> CreateGlobalGroupJoinAggregates(const PhysicalHashGroupJoin &op) {
	vector<AggregateObject> result;
	result.reserve(op.owner_payload_data.bindings.size() + op.grouped_aggregate_data.bindings.size() + 1);
	result.push_back(CreateGroupJoinIdAggregate());
	for (auto binding : op.owner_payload_data.bindings) {
		result.emplace_back(binding);
	}
	for (auto binding : op.grouped_aggregate_data.bindings) {
		result.emplace_back(binding);
	}
	return result;
}

static vector<AggregateObject> CreateLocalGroupJoinAggregates(const PhysicalHashGroupJoin &op) {
	return AggregateObject::CreateAggregateObjects(op.grouped_aggregate_data.bindings);
}

static vector<AggregateObject> CreateRoutedGroupJoinAggregates(const PhysicalHashGroupJoin &op) {
	vector<AggregateObject> result;
	result.reserve(op.grouped_aggregate_data.bindings.size() + 1);
	result.push_back(CreateGroupJoinIdAggregate());
	for (auto binding : op.grouped_aggregate_data.bindings) {
		result.emplace_back(binding);
	}
	return result;
}

static vector<AggregateObject> CreateRoutingGroupJoinAggregates() {
	vector<AggregateObject> result;
	result.push_back(CreateGroupJoinIdAggregate());
	return result;
}

static vector<LogicalType> CreateGlobalGroupJoinPayloadTypes(const PhysicalHashGroupJoin &op) {
	return op.grouped_aggregate_data.payload_types;
}

static idx_t LoadGroupJoinId(const TupleDataLayout &layout, data_ptr_t row) {
	auto &state = *reinterpret_cast<const GroupJoinIdState *>(row + layout.GetAggrOffset());
	if (state.value == 0) {
		throw InternalException("HASH_GROUP_JOIN encountered an unassigned group identifier");
	}
	return NumericCast<idx_t>(state.value - 1);
}

static void StoreGroupJoinId(const TupleDataLayout &layout, data_ptr_t row, idx_t group_id) {
	auto &state = *reinterpret_cast<GroupJoinIdState *>(row + layout.GetAggrOffset());
	D_ASSERT(state.value == 0);
	state.value = NumericCast<uint64_t>(group_id) + 1;
}

PhysicalHashGroupJoin::PhysicalHashGroupJoin(
    PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &probe, PhysicalOperator &owner,
    vector<unique_ptr<Expression>> aggregates, vector<unique_ptr<Expression>> owner_payload_aggregates,
    vector<unique_ptr<Expression>> groups, vector<HashGroupJoinOutputColumn> output_groups_p,
    HashGroupJoinUnmatchedPolicy unmatched_policy_p, bool routed_p, bool unique_owner_p, bool single_match_p,
    unique_ptr<JoinFilterPushdownInfo> filter_pushdown_p, idx_t estimated_cardinality)
    : PhysicalJoin(physical_plan, op, PhysicalOperatorType::HASH_GROUP_JOIN, JoinType::INNER, estimated_cardinality),
      output_groups(std::move(output_groups_p)), unmatched_policy(unmatched_policy_p), routed(routed_p),
      unique_owner(unique_owner_p), single_match(single_match_p), null_equal(false), static_mode(false),
      filter_pushdown(std::move(filter_pushdown_p)) {
	for (auto &group : op.groups) {
		output_group_types.push_back(group->GetReturnType());
	}
	grouped_aggregate_data.InitializeGroupby(std::move(groups), std::move(aggregates), {});
	owner_payload_data.InitializeGroupby({}, std::move(owner_payload_aggregates), {});
	auto &probe_projection = probe.Cast<PhysicalProjection>();
	D_ASSERT(probe_projection.children.size() == 1);
	unmatched_probe_types = probe_projection.children[0].get().GetTypes();
	for (idx_t payload_idx = grouped_aggregate_data.group_types.size();
	     payload_idx < probe_projection.select_list.size(); payload_idx++) {
		unmatched_payload_expressions.push_back(probe_projection.select_list[payload_idx]->Copy());
	}
	for (auto &group : op.groups) {
		output_group_names.push_back(group->GetName().GetIdentifierName());
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
	D_ASSERT(output_groups.size() == op.groups.size());
	children.push_back(probe);
	children.push_back(owner);
}

PhysicalHashGroupJoin::PhysicalHashGroupJoin(PhysicalPlan &physical_plan, LogicalComparisonJoin &op,
                                             PhysicalOperator &probe, PhysicalOperator &owner,
                                             vector<unique_ptr<Expression>> aggregates,
                                             vector<unique_ptr<Expression>> owner_payload_aggregates,
                                             vector<unique_ptr<Expression>> groups,
                                             vector<HashGroupJoinOutputColumn> output_columns_p,
                                             idx_t estimated_cardinality)
    : PhysicalJoin(physical_plan, op, PhysicalOperatorType::HASH_GROUP_JOIN, JoinType::INNER, estimated_cardinality),
      output_columns(std::move(output_columns_p)), unmatched_policy(HashGroupJoinUnmatchedPolicy::EMPTY_AGGREGATE),
      routed(false), unique_owner(false), single_match(false), null_equal(true), static_mode(true) {
	for (auto &group : groups) {
		output_group_types.push_back(group->GetReturnType());
		output_group_names.push_back(group->GetName().GetIdentifierName());
	}
	grouped_aggregate_data.InitializeGroupby(std::move(groups), std::move(aggregates), {});
	owner_payload_data.InitializeGroupby({}, std::move(owner_payload_aggregates), {});
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
	D_ASSERT(output_columns.size() == op.types.size());
	children.push_back(probe);
	children.push_back(owner);
}

static bool UseExternalHashGroupJoin(const PhysicalHashGroupJoin &op, ClientContext &context) {
	return Settings::Get<DebugForceExternalSetting>(context) ||
	       Settings::Get<DebugGroupJoinExecutionSetting>(context) == GroupJoinExecutionMode::EXTERNAL;
}

class HashGroupJoinGlobalSinkState : public GlobalSinkState {
public:
	HashGroupJoinGlobalSinkState(const PhysicalHashGroupJoin &op, ClientContext &context)
	    : external(UseExternalHashGroupJoin(op, context)) {
		if (op.filter_pushdown) {
			global_filter_state = op.filter_pushdown->GetGlobalState(context, op);
			local_filter_state = op.filter_pushdown->GetLocalState(*global_filter_state);
			const auto owner_rows = op.children[1].get().estimated_cardinality;
			const auto probe_rows = op.children[0].get().estimated_cardinality;
			if (op.filter_pushdown->join_condition.size() == 1 && owner_rows <= probe_rows) {
				filter_keys = make_uniq<ColumnDataCollection>(context, op.grouped_aggregate_data.group_types);
			}
		}
		if (external) {
			auto owner_types = op.children[1].get().GetTypes();
			owner_types.push_back(LogicalType::HASH);
			owner_partitions =
			    make_uniq<RadixPartitionedColumnData>(context, std::move(owner_types), GROUP_JOIN_EXTERNAL_RADIX_BITS,
			                                          op.children[1].get().GetTypes().size());
			owner_local_partitions = owner_partitions->CreateShared();
			owner_local_partitions->InitializeAppendState(owner_partition_append);
			return;
		}
		if (op.static_mode) {
			auto owner_types = op.children[1].get().GetTypes();
			owner_types.push_back(LogicalType::UBIGINT);
			static_owner_rows = make_uniq<ColumnDataCollection>(context, std::move(owner_types));
		}
		if (op.routed) {
			hash_table = make_uniq<GroupedAggregateHashTable>(
			    context, BufferAllocator::Get(context), op.grouped_aggregate_data.group_types, vector<LogicalType> {},
			    CreateRoutingGroupJoinAggregates(), GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
			    op.null_equal || !op.unique_owner ? TupleDataValidityType::CAN_HAVE_NULL_VALUES
			                                      : TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
			final_hash_table = make_uniq<GroupedAggregateHashTable>(
			    context, BufferAllocator::Get(context), op.output_group_types, op.grouped_aggregate_data.payload_types,
			    CreateRoutedGroupJoinAggregates(op), GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
			    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		} else {
			hash_table = make_uniq<GroupedAggregateHashTable>(
			    context, BufferAllocator::Get(context), op.grouped_aggregate_data.group_types,
			    CreateGlobalGroupJoinPayloadTypes(op), CreateGlobalGroupJoinAggregates(op),
			    GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
			    op.null_equal ? TupleDataValidityType::CAN_HAVE_NULL_VALUES
			                  : TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
		}
	}

	unique_ptr<GroupedAggregateHashTable> hash_table;
	unique_ptr<GroupedAggregateHashTable> final_hash_table;
	vector<data_ptr_t> group_addresses;
	vector<vector<idx_t>> route_build_groups;
	vector<idx_t> route_offsets;
	vector<idx_t> route_group_ids;
	unique_ptr<atomic<bool>[]> route_matches;
	unique_ptr<atomic<idx_t>[]> owners;
	unique_ptr<ColumnDataCollection> static_owner_rows;
	unique_ptr<JoinFilterGlobalState> global_filter_state;
	unique_ptr<JoinFilterLocalState> local_filter_state;
	vector<unique_ptr<BloomFilter>> bloom_filters;
	vector<unique_ptr<PrefixRangeFilter>> prefix_range_filters;
	unique_ptr<ColumnDataCollection> filter_keys;
	bool external;
	unique_ptr<RadixPartitionedColumnData> owner_partitions;
	unique_ptr<PartitionedColumnData> owner_local_partitions;
	PartitionedColumnDataAppendState owner_partition_append;
	bool build_finalized = false;
	idx_t owner_rows = 0;
	atomic<idx_t> probe_rows {0};
	atomic<idx_t> matched_rows {0};
};

static GroupedAggregateHashTable &GetHashGroupJoinTarget(HashGroupJoinGlobalSinkState &state) {
	return state.final_hash_table ? *state.final_hash_table : *state.hash_table;
}

static idx_t GetHashGroupJoinAggregateOffset(const PhysicalHashGroupJoin &op) {
	return op.routed ? 1 : op.owner_payload_data.aggregates.size() + 1;
}

unique_ptr<GlobalSinkState> PhysicalHashGroupJoin::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<HashGroupJoinGlobalSinkState>(*this, context);
}

class HashGroupJoinLocalSinkState : public LocalSinkState {
public:
	HashGroupJoinLocalSinkState(ExecutionContext &context, const PhysicalHashGroupJoin &op)
	    : addresses(LogicalType::POINTER), new_groups(STANDARD_VECTOR_SIZE), final_addresses(LogicalType::POINTER),
	      final_new_groups(STANDARD_VECTOR_SIZE), hashes(LogicalType::HASH) {
		auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
		owner_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		owner_payload.InitializeEmpty(op.owner_payload_data.payload_types);
		final_groups.InitializeEmpty(op.output_group_types);
		if (sink.external) {
			auto partition_types = op.children[1].get().GetTypes();
			partition_types.push_back(LogicalType::HASH);
			partition_chunk.InitializeEmpty(partition_types);
		} else {
			update_state = make_uniq<AggregateHTUpdateState>(*sink.hash_table);
		}
		if (op.static_mode && !sink.external) {
			auto owner_types = op.children[1].get().GetTypes();
			owner_types.push_back(LogicalType::UBIGINT);
			static_owner_chunk.Initialize(Allocator::Get(context.client), owner_types);
		}
	}

	DataChunk owner_keys;
	DataChunk owner_payload;
	Vector addresses;
	SelectionVector new_groups;
	DataChunk final_groups;
	DataChunk static_owner_chunk;
	Vector final_addresses;
	SelectionVector final_new_groups;
	unique_ptr<AggregateHTUpdateState> update_state;
	Vector hashes;
	DataChunk partition_chunk;
};

unique_ptr<LocalSinkState> PhysicalHashGroupJoin::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<HashGroupJoinLocalSinkState>(context, *this);
}

SinkResultType PhysicalHashGroupJoin::Sink(ExecutionContext &context, DataChunk &chunk,
                                           OperatorSinkInput &input) const {
	auto &global_state = input.global_state.Cast<HashGroupJoinGlobalSinkState>();
	auto &local_state = input.local_state.Cast<HashGroupJoinLocalSinkState>();
	if (global_state.build_finalized) {
		throw InternalException("Cannot add owner rows after HASH_GROUP_JOIN build finalization");
	}
	const auto key_count = grouped_aggregate_data.group_types.size();
	local_state.owner_keys.Reset();
	for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
		local_state.owner_keys.data[key_idx].Reference(chunk.data[key_idx]);
	}
	local_state.owner_keys.CheckCardinality(chunk.size());
	if (filter_pushdown) {
		D_ASSERT(global_state.local_filter_state);
		filter_pushdown->Sink(local_state.owner_keys, *global_state.local_filter_state);
		if (global_state.filter_keys) {
			global_state.filter_keys->Append(local_state.owner_keys);
		}
	}
	if (unique_owner && !null_equal && PhysicalJoin::HasNullValues(local_state.owner_keys)) {
		throw InternalException("HASH_GROUP_JOIN owner key unexpectedly contained NULL");
	}
	if (global_state.external) {
		local_state.owner_keys.Hash(local_state.hashes);
		local_state.partition_chunk.Reset();
		for (idx_t column_idx = 0; column_idx < chunk.ColumnCount(); column_idx++) {
			local_state.partition_chunk.data[column_idx].Reference(chunk.data[column_idx]);
		}
		local_state.partition_chunk.data.back().Reference(local_state.hashes);
		local_state.partition_chunk.SetChildCardinality(chunk.size());
		global_state.owner_local_partitions->Append(global_state.owner_partition_append, local_state.partition_chunk);
		global_state.owner_rows += chunk.size();
		return SinkResultType::NEED_MORE_INPUT;
	}
	const auto new_group_count = global_state.hash_table->FindOrCreateGroups(
	    local_state.owner_keys, local_state.addresses, local_state.new_groups);
	if (unique_owner && new_group_count != chunk.size()) {
		throw InternalException("HASH_GROUP_JOIN owner key uniqueness proof was violated at execution time");
	}
	auto addresses = FlatVector::GetData<data_ptr_t>(local_state.addresses);
	auto &layout = global_state.hash_table->GetLayout();
	for (idx_t new_idx = 0; new_idx < new_group_count; new_idx++) {
		auto row_idx = local_state.new_groups.get_index_unsafe(new_idx);
		const auto group_id = routed ? global_state.route_build_groups.size() : global_state.group_addresses.size();
		StoreGroupJoinId(layout, addresses[row_idx], group_id);
		if (routed) {
			global_state.route_build_groups.emplace_back();
		} else {
			global_state.group_addresses.push_back(addresses[row_idx]);
		}
	}
	if (static_mode) {
		D_ASSERT(global_state.static_owner_rows);
		local_state.static_owner_chunk.Reset();
		for (idx_t column_idx = 0; column_idx < chunk.ColumnCount(); column_idx++) {
			local_state.static_owner_chunk.data[column_idx].Reference(chunk.data[column_idx]);
		}
		auto &ids = local_state.static_owner_chunk.data.back();
		ids.SetVectorType(VectorType::FLAT_VECTOR);
		auto id_data = FlatVector::GetDataMutable<uint64_t>(ids);
		for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
			id_data[row_idx] = LoadGroupJoinId(layout, addresses[row_idx]);
		}
		FlatVector::SetSize(ids, chunk.size());
		local_state.static_owner_chunk.SetChildCardinality(chunk.size());
		global_state.static_owner_rows->Append(local_state.static_owner_chunk);
	}
	if (routed) {
		local_state.final_groups.Reset();
		for (idx_t group_idx = 0; group_idx < output_groups.size(); group_idx++) {
			auto &output_group = output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::KEY) {
				local_state.final_groups.data[group_idx].Reference(chunk.data[output_group.index]);
			} else {
				local_state.final_groups.data[group_idx].Reference(chunk.data[key_count + output_group.index]);
			}
		}
		local_state.final_groups.CheckCardinality(chunk.size());
		auto &target = *global_state.final_hash_table;
		auto new_final_count = target.FindOrCreateGroups(local_state.final_groups, local_state.final_addresses,
		                                                 local_state.final_new_groups);
		auto final_addresses = FlatVector::GetData<data_ptr_t>(local_state.final_addresses);
		for (idx_t new_idx = 0; new_idx < new_final_count; new_idx++) {
			auto input_idx = local_state.final_new_groups.get_index_unsafe(new_idx);
			const auto final_group_id = global_state.group_addresses.size();
			StoreGroupJoinId(target.GetLayout(), final_addresses[input_idx], final_group_id);
			global_state.group_addresses.push_back(final_addresses[input_idx]);
		}
		for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
			auto route_id = LoadGroupJoinId(layout, addresses[row_idx]);
			D_ASSERT(route_id < global_state.route_build_groups.size());
			auto &routes = global_state.route_build_groups[route_id];
			if (!single_match || routes.empty()) {
				routes.push_back(LoadGroupJoinId(target.GetLayout(), final_addresses[row_idx]));
			}
		}
	}
	if (!routed && !owner_payload_data.aggregates.empty()) {
		local_state.owner_payload.Reset();
		for (idx_t payload_idx = 0; payload_idx < owner_payload_data.payload_types.size(); payload_idx++) {
			local_state.owner_payload.data[payload_idx].Reference(chunk.data[key_count + payload_idx]);
		}
		local_state.owner_payload.CheckCardinality(chunk.size());
		global_state.hash_table->UpdateAggregatesAtAddressesRange(*local_state.update_state, local_state.addresses,
		                                                          local_state.owner_payload, 1,
		                                                          owner_payload_data.aggregates.size());
	}
	global_state.owner_rows += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

static void BuildGroupJoinRuntimeFilters(const PhysicalHashGroupJoin &op, ClientContext &context,
                                         HashGroupJoinGlobalSinkState &state, const DataChunk &final_min_max) {
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	state.bloom_filters.resize(key_count);
	state.prefix_range_filters.resize(key_count);
	if (!state.filter_keys || state.filter_keys->Count() == 0) {
		return;
	}
	D_ASSERT(op.filter_pushdown->join_condition.size() == 1);
	const auto filter_idx = idx_t(0);
	const auto key_idx = op.filter_pushdown->join_condition[filter_idx];
	D_ASSERT(key_idx < key_count);
	const auto probe_rows = op.children[0].get().estimated_cardinality;
	const auto ratio =
	    static_cast<double>(state.owner_rows) / static_cast<double>(MaxValue<idx_t>(probe_rows, idx_t(1)));
	const bool build_filter = state.owner_rows <= probe_rows && (op.filter_pushdown->build_side_has_filter ||
	                                                             ratio <= 0.1 || state.owner_rows <= 4194304);
	if (!build_filter) {
		state.filter_keys.reset();
		return;
	}

	unique_ptr<PrefixRangeFilter::BuildState> prefix_build_state;
	auto &key_type = op.grouped_aggregate_data.group_types[key_idx];
	auto min_value = final_min_max.data[filter_idx * 2].GetValue(0);
	auto max_value = final_min_max.data[filter_idx * 2 + 1].GetValue(0);
	if (min_value.IsNull() || max_value.IsNull() || Value::NotDistinctFrom(min_value, max_value)) {
		state.filter_keys.reset();
		return;
	}
	if (PrefixRangeFilter::SupportedType(key_type)) {
		uhugeint_t span;
		if (PrefixRangeFilter::TryComputeSpan(min_value, max_value, span)) {
			static constexpr idx_t MAX_EXACT_BITS = 1ULL << 26;
			const auto bloom_bits = BloomFilter::GetNumberOfSectors(MaxValue<idx_t>(state.owner_rows, idx_t(1))) * 64;
			idx_t max_bits = 0;
			if (span < MAX_EXACT_BITS) {
				max_bits = MAX_EXACT_BITS;
			} else if (span <= bloom_bits) {
				max_bits = bloom_bits;
			}
			if (max_bits > 0) {
				state.prefix_range_filters[key_idx] = PrefixRangeFilter::CreatePrefixRangeFilter(key_type);
				state.prefix_range_filters[key_idx]->Initialize(context, state.owner_rows, min_value, max_value,
				                                                max_bits);
				prefix_build_state = state.prefix_range_filters[key_idx]->InitializeBuildState(context);
			}
		}
	}
	if (!state.prefix_range_filters[key_idx]) {
		state.bloom_filters[key_idx] = make_uniq<BloomFilter>();
		state.bloom_filters[key_idx]->Initialize(context, state.owner_rows);
	}
	Vector hashes(LogicalType::HASH);
	for (auto &chunk : state.filter_keys->Chunks()) {
		if (state.prefix_range_filters[key_idx]) {
			state.prefix_range_filters[key_idx]->InsertKeys(chunk.data[key_idx], *prefix_build_state);
		} else {
			VectorOperations::Hash(chunk.data[key_idx], hashes, chunk.size());
			hashes.Flatten();
			state.bloom_filters[key_idx]->InsertHashes(hashes);
		}
	}
	if (state.prefix_range_filters[key_idx]) {
		state.prefix_range_filters[key_idx]->MergeBuildState(*prefix_build_state);
	}
	state.filter_keys.reset();
}

SinkFinalizeType PhysicalHashGroupJoin::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                 OperatorSinkFinalizeInput &input) const {
	auto &state = input.global_state.Cast<HashGroupJoinGlobalSinkState>();
	state.build_finalized = true;
	if (filter_pushdown) {
		D_ASSERT(state.global_filter_state && state.local_filter_state);
		filter_pushdown->Combine(*state.global_filter_state, *state.local_filter_state);
		vector<string> key_names;
		for (auto &group : grouped_aggregate_data.groups) {
			key_names.push_back(group->GetName().GetIdentifierName());
		}
		auto final_min_max = filter_pushdown->FinalizeMinMax(*state.global_filter_state);
		BuildGroupJoinRuntimeFilters(*this, context, state, *final_min_max);
		filter_pushdown->FinalizeGroupJoinFilters(context, *this, grouped_aggregate_data.group_types, key_names,
		                                          state.bloom_filters, state.prefix_range_filters,
		                                          std::move(final_min_max));
	}
	if (state.external) {
		state.owner_local_partitions->FlushAppendState(state.owner_partition_append);
		state.owner_partitions->Combine(*state.owner_local_partitions);
		state.owner_local_partitions.reset();
		return state.owner_rows == 0 ? SinkFinalizeType::NO_OUTPUT_POSSIBLE : SinkFinalizeType::READY;
	}
	if (routed) {
		D_ASSERT(state.route_build_groups.size() == state.hash_table->Count());
		state.route_offsets.reserve(state.route_build_groups.size() + 1);
		for (auto &routes : state.route_build_groups) {
			D_ASSERT(!routes.empty());
			state.route_offsets.push_back(state.route_group_ids.size());
			state.route_group_ids.insert(state.route_group_ids.end(), routes.begin(), routes.end());
		}
		state.route_offsets.push_back(state.route_group_ids.size());
		state.route_build_groups.clear();
		D_ASSERT(state.group_addresses.size() == state.final_hash_table->Count());
	} else {
		D_ASSERT(state.group_addresses.size() == state.hash_table->Count());
	}
	state.owners = unique_ptr<atomic<idx_t>[]>(new atomic<idx_t>[state.group_addresses.size()]);
	for (idx_t group_id = 0; group_id < state.group_addresses.size(); group_id++) {
		state.owners[group_id].store(0, std::memory_order_relaxed);
	}
	if (routed && unmatched_policy == HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW) {
		state.route_matches = unique_ptr<atomic<bool>[]>(new atomic<bool>[state.route_group_ids.size()]);
		for (idx_t owner_id = 0; owner_id < state.route_group_ids.size(); owner_id++) {
			state.route_matches[owner_id].store(false, std::memory_order_relaxed);
		}
	}
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

static GroupJoinExecutionMode ResolveGroupJoinExecutionMode(ClientContext &context, const PhysicalHashGroupJoin &op) {
	if (op.sink_state->Cast<HashGroupJoinGlobalSinkState>().external) {
		return GroupJoinExecutionMode::EXTERNAL;
	}
	auto mode = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	return mode == GroupJoinExecutionMode::AUTO || mode == GroupJoinExecutionMode::INDEX
	           ? GroupJoinExecutionMode::OWNERSHIP
	           : mode;
}

class HashGroupJoinGlobalOperatorState : public GlobalOperatorState {
public:
	HashGroupJoinGlobalOperatorState(ClientContext &context, const PhysicalHashGroupJoin &op)
	    : execution_mode(ResolveGroupJoinExecutionMode(context, op)) {
		if (execution_mode == GroupJoinExecutionMode::EXTERNAL) {
			auto probe_types = op.children[0].get().GetTypes();
			probe_types.push_back(LogicalType::HASH);
			probe_partitions =
			    make_uniq<RadixPartitionedColumnData>(context, std::move(probe_types), GROUP_JOIN_EXTERNAL_RADIX_BITS,
			                                          op.children[0].get().GetTypes().size());
		}
	}

	idx_t MaxThreads(idx_t source_max_threads) override {
		return execution_mode == GroupJoinExecutionMode::SERIAL ? 1 : source_max_threads;
	}

	GroupJoinExecutionMode execution_mode;
	atomic<idx_t> next_token {1};
	mutex lock;
	vector<shared_ptr<GroupedAggregateHashTable>> local_tables;
	vector<vector<shared_ptr<GroupedAggregateHashTable>>> local_distinct_tables;
	unique_ptr<RadixPartitionedColumnData> probe_partitions;
	vector<shared_ptr<PartitionedColumnData>> local_probe_partitions;
	vector<shared_ptr<PartitionedColumnDataAppendState>> local_probe_append_states;
	bool merge_complete = false;
};

unique_ptr<GlobalOperatorState> PhysicalHashGroupJoin::GetGlobalOperatorState(ClientContext &context) const {
	return make_uniq<HashGroupJoinGlobalOperatorState>(context, *this);
}

class HashGroupJoinOperatorState : public CachingOperatorState {
public:
	HashGroupJoinOperatorState(ExecutionContext &context, const PhysicalHashGroupJoin &op)
	    : non_null_sel(STANDARD_VECTOR_SIZE), found_sel(STANDARD_VECTOR_SIZE), matched_input_sel(STANDARD_VECTOR_SIZE),
	      route_input_sel(STANDARD_VECTOR_SIZE), owner_sel(STANDARD_VECTOR_SIZE), local_sel(STANDARD_VECTOR_SIZE),
	      matched_addresses(LogicalType::POINTER), routed_addresses(LogicalType::POINTER),
	      owner_addresses(LogicalType::POINTER), key_formats(op.grouped_aggregate_data.group_types.size()) {
		auto &global_state = op.op_state->Cast<HashGroupJoinGlobalOperatorState>();
		auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
		execution_mode = global_state.execution_mode;
		probe_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		if (execution_mode == GroupJoinExecutionMode::EXTERNAL) {
			auto partition_types = op.children[0].get().GetTypes();
			partition_types.push_back(LogicalType::HASH);
			partition_chunk.InitializeEmpty(partition_types);
			auto local_partition = global_state.probe_partitions->CreateShared();
			probe_partition = shared_ptr<PartitionedColumnData>(local_partition.release());
			probe_append_state = make_shared_ptr<PartitionedColumnDataAppendState>();
			probe_partition->InitializeAppendState(*probe_append_state);
			lock_guard<mutex> guard(global_state.lock);
			global_state.local_probe_partitions.push_back(probe_partition);
			global_state.local_probe_append_states.push_back(probe_append_state);
			return;
		}
		auto &target = GetHashGroupJoinTarget(sink);
		lookup_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		routed_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		selected_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		group_ids.Initialize(Allocator::Get(context.client), {LogicalType::UBIGINT});
		selected_group_ids.InitializeEmpty({LogicalType::UBIGINT});
		auto aggregate_objects = CreateLocalGroupJoinAggregates(op);
		distinct_filter_set.Initialize(context.client, aggregate_objects, op.grouped_aggregate_data.payload_types);
		for (auto &distinct : op.distinct_aggregates) {
			vector<LogicalType> group_types {LogicalType::UBIGINT};
			group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			distinct_tables.push_back(make_shared_ptr<GroupedAggregateHashTable>(
			    context.client, BufferAllocator::Get(context.client), group_types, vector<LogicalType> {},
			    vector<AggregateObject> {}, GroupedAggregateHashTable::InitialCapacity(), GROUP_JOIN_LOCAL_RADIX_BITS,
			    TupleDataValidityType::CAN_HAVE_NULL_VALUES));
			auto groups = make_uniq<DataChunk>();
			groups->InitializeEmpty(group_types);
			distinct_groups.push_back(std::move(groups));
		}

		if (execution_mode == GroupJoinExecutionMode::SERIAL) {
			update_state = make_uniq<AggregateHTUpdateState>(target);
		} else {
			local_hash_table = make_shared_ptr<GroupedAggregateHashTable>(
			    context.client, BufferAllocator::Get(context.client), vector<LogicalType> {LogicalType::UBIGINT},
			    op.grouped_aggregate_data.payload_types, std::move(aggregate_objects),
			    GroupedAggregateHashTable::InitialCapacity(), GROUP_JOIN_LOCAL_RADIX_BITS,
			    TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
			if (execution_mode == GroupJoinExecutionMode::OWNERSHIP) {
				token = global_state.next_token.fetch_add(1, std::memory_order_relaxed);
				if (token == 0) {
					throw InternalException("HASH_GROUP_JOIN ownership token overflow");
				}
				update_allocator = make_shared_ptr<ArenaAllocator>(Allocator::Get(context.client));
				update_state = make_uniq<AggregateHTUpdateState>(target, update_allocator);
			}
		}
		lock_guard<mutex> guard(global_state.lock);
		if (local_hash_table) {
			global_state.local_tables.push_back(local_hash_table);
		}
		if (!distinct_tables.empty()) {
			global_state.local_distinct_tables.push_back(distinct_tables);
		}
		if (update_allocator) {
			target.StoreAggregateAllocator(update_allocator);
		}
	}

	DataChunk probe_keys;
	DataChunk lookup_keys;
	DataChunk payload;
	DataChunk routed_payload;
	DataChunk selected_payload;
	DataChunk group_ids;
	DataChunk selected_group_ids;
	SelectionVector non_null_sel;
	SelectionVector found_sel;
	SelectionVector matched_input_sel;
	SelectionVector route_input_sel;
	SelectionVector owner_sel;
	SelectionVector local_sel;
	Vector matched_addresses;
	Vector routed_addresses;
	Vector owner_addresses;
	AggregateHTLookupState lookup_state;
	vector<UnifiedVectorFormat> key_formats;
	GroupJoinExecutionMode execution_mode = GroupJoinExecutionMode::SERIAL;
	idx_t token = 0;
	shared_ptr<ArenaAllocator> update_allocator;
	unique_ptr<AggregateHTUpdateState> update_state;
	shared_ptr<GroupedAggregateHashTable> local_hash_table;
	AggregateFilterDataSet distinct_filter_set;
	vector<shared_ptr<GroupedAggregateHashTable>> distinct_tables;
	vector<unique_ptr<DataChunk>> distinct_groups;
	Vector distinct_addresses {LogicalType::POINTER};
	DataChunk partition_chunk;
	Vector partition_hashes {LogicalType::HASH};
	shared_ptr<PartitionedColumnData> probe_partition;
	shared_ptr<PartitionedColumnDataAppendState> probe_append_state;
};

unique_ptr<OperatorState> PhysicalHashGroupJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<HashGroupJoinOperatorState>(context, *this);
}

static void SinkHashGroupJoinDistinct(const PhysicalHashGroupJoin &op, HashGroupJoinOperatorState &state,
                                      DataChunk &payload, DataChunk &group_ids) {
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		auto &aggregate =
		    op.grouped_aggregate_data.aggregates[distinct.aggregate_index]->Cast<BoundAggregateExpression>();
		auto &groups = *state.distinct_groups[distinct_idx];
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
			argument_payload = filter_data.filtered_payload;
		} else {
			count = payload.size();
			groups.data[0].Reference(group_ids.data[0]);
			argument_payload = payload;
		}
		for (idx_t child_idx = 0; child_idx < aggregate.GetChildren().size(); child_idx++) {
			groups.data[child_idx + 1].Reference(argument_payload->data[distinct.payload_index + child_idx]);
		}
		groups.SetChildCardinality(count);
		state.distinct_tables[distinct_idx]->FindOrCreateGroups(groups, state.distinct_addresses);
	}
}

static void SinkHashGroupJoinMatches(const PhysicalHashGroupJoin &op, HashGroupJoinGlobalSinkState &sink,
                                     HashGroupJoinOperatorState &state, GroupedAggregateHashTable &target,
                                     Vector &matched_addresses, DataChunk &payload, idx_t match_count) {
	state.group_ids.Reset();
	state.group_ids.SetChildCardinality(match_count);
	state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
	auto ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
	auto addresses = FlatVector::GetData<data_ptr_t>(matched_addresses);
	auto &global_layout = target.GetLayout();
	for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
		ids[match_idx] = LoadGroupJoinId(global_layout, addresses[match_idx]);
	}
	FlatVector::SetSize(state.group_ids.data[0], match_count);
	if (!op.distinct_aggregates.empty()) {
		SinkHashGroupJoinDistinct(op, state, payload, state.group_ids);
	}

	if (state.execution_mode == GroupJoinExecutionMode::SERIAL) {
		target.UpdateAggregatesAtAddressesRange(*state.update_state, matched_addresses, payload,
		                                        GetHashGroupJoinAggregateOffset(op),
		                                        op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
		sink.matched_rows.fetch_add(match_count, std::memory_order_relaxed);
		return;
	}

	if (state.execution_mode == GroupJoinExecutionMode::LOCAL) {
		state.local_hash_table->AddChunk(state.group_ids, payload, op.non_distinct_filter);
		sink.matched_rows.fetch_add(match_count, std::memory_order_relaxed);
		return;
	}

	D_ASSERT(state.execution_mode == GroupJoinExecutionMode::OWNERSHIP);
	idx_t owner_count = 0;
	idx_t local_count = 0;
	for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
		auto &owner = sink.owners[NumericCast<idx_t>(ids[match_idx])];
		auto current_owner = owner.load(std::memory_order_relaxed);
		if (current_owner == 0) {
			auto expected = idx_t(0);
			if (owner.compare_exchange_strong(expected, state.token, std::memory_order_relaxed,
			                                  std::memory_order_relaxed)) {
				current_owner = state.token;
			} else {
				current_owner = expected;
			}
		}
		if (current_owner == state.token) {
			state.owner_sel.set_index(owner_count++, match_idx);
		} else {
			state.local_sel.set_index(local_count++, match_idx);
		}
	}

	if (owner_count != 0) {
		state.owner_addresses.Slice(matched_addresses, state.owner_sel, owner_count);
		state.owner_addresses.Flatten();
		state.selected_payload.Reset();
		state.selected_payload.Slice(payload, state.owner_sel, owner_count);
		target.UpdateAggregatesAtAddressesRange(*state.update_state, state.owner_addresses, state.selected_payload,
		                                        GetHashGroupJoinAggregateOffset(op),
		                                        op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
	}
	if (local_count != 0) {
		state.selected_group_ids.Reset();
		state.selected_group_ids.Slice(state.group_ids, state.local_sel, local_count);
		state.selected_payload.Reset();
		state.selected_payload.Slice(payload, state.local_sel, local_count);
		state.local_hash_table->AddChunk(state.selected_group_ids, state.selected_payload, op.non_distinct_filter);
	}
	sink.matched_rows.fetch_add(match_count, std::memory_order_relaxed);
}

OperatorResultType PhysicalHashGroupJoin::ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                          GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &sink = sink_state->Cast<HashGroupJoinGlobalSinkState>();
	auto &state = state_p.Cast<HashGroupJoinOperatorState>();
	D_ASSERT(sink.build_finalized);
	chunk.Reset();
	sink.probe_rows.fetch_add(input.size(), std::memory_order_relaxed);

	const auto group_count = grouped_aggregate_data.group_types.size();
	state.probe_keys.Reset();
	for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
		state.probe_keys.data[group_idx].Reference(input.data[group_idx]);
	}
	state.probe_keys.CheckCardinality(input.size());
	if (state.execution_mode == GroupJoinExecutionMode::EXTERNAL) {
		auto partition_count = input.size();
		if (!null_equal) {
			partition_count = SelectNonNullGroupJoinKeys(state.probe_keys, state.key_formats, state.non_null_sel);
			if (partition_count == 0) {
				return OperatorResultType::NEED_MORE_INPUT;
			}
		}
		state.probe_keys.Hash(state.partition_hashes);
		state.partition_chunk.Reset();
		for (idx_t column_idx = 0; column_idx < input.ColumnCount(); column_idx++) {
			state.partition_chunk.data[column_idx].Reference(input.data[column_idx]);
		}
		state.partition_chunk.data.back().Reference(state.partition_hashes);
		state.partition_chunk.SetChildCardinality(input.size());
		if (partition_count != input.size()) {
			state.partition_chunk.Slice(state.non_null_sel, partition_count);
		}
		state.probe_partition->Append(*state.probe_append_state, state.partition_chunk);
		return OperatorResultType::NEED_MORE_INPUT;
	}
	auto &target = GetHashGroupJoinTarget(sink);
	const auto non_null_count =
	    null_equal ? input.size() : SelectNonNullGroupJoinKeys(state.probe_keys, state.key_formats, state.non_null_sel);
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
	state.matched_addresses.SetVectorType(VectorType::FLAT_VECTOR);
	state.payload.Reset();
	if (all_input_rows_match) {
		state.matched_addresses.Reference(state.lookup_state.addresses);
		for (idx_t payload_idx = 0; payload_idx < grouped_aggregate_data.payload_types.size(); payload_idx++) {
			state.payload.data[payload_idx].Reference(input.data[group_count + payload_idx]);
		}
		if (state.payload.ColumnCount() == 0) {
			state.payload.SetChildCardinality(match_count);
		} else {
			state.payload.CheckCardinality(match_count);
		}
	} else {
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
	}
	if (routed) {
		state.matched_addresses.Flatten();
		auto routing_addresses = FlatVector::GetData<data_ptr_t>(state.matched_addresses);
		auto &routing_layout = sink.hash_table->GetLayout();
		state.routed_addresses.SetVectorType(VectorType::FLAT_VECTOR);
		auto routed_addresses = FlatVector::GetDataMutable<data_ptr_t>(state.routed_addresses);
		idx_t routed_count = 0;
		auto flush_routes = [&]() {
			if (routed_count == 0) {
				return;
			}
			FlatVector::SetSize(state.routed_addresses, routed_count);
			state.routed_payload.Reset();
			state.routed_payload.Slice(state.payload, state.route_input_sel, routed_count);
			if (state.routed_payload.ColumnCount() == 0) {
				state.routed_payload.SetChildCardinality(routed_count);
			}
			SinkHashGroupJoinMatches(*this, sink, state, target, state.routed_addresses, state.routed_payload,
			                         routed_count);
			routed_count = 0;
		};
		for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
			auto route_id = LoadGroupJoinId(routing_layout, routing_addresses[match_idx]);
			if (route_id + 1 >= sink.route_offsets.size()) {
				throw InternalException("HASH_GROUP_JOIN routing identifier exceeds route count");
			}
			for (idx_t route_idx = sink.route_offsets[route_id]; route_idx < sink.route_offsets[route_id + 1];
			     route_idx++) {
				auto group_id = sink.route_group_ids[route_idx];
				if (sink.route_matches) {
					sink.route_matches[route_idx].store(true, std::memory_order_relaxed);
				}
				routed_addresses[routed_count] = sink.group_addresses[group_id];
				state.route_input_sel.set_index(routed_count++, match_idx);
				if (routed_count == STANDARD_VECTOR_SIZE) {
					flush_routes();
				}
			}
		}
		flush_routes();
		return OperatorResultType::NEED_MORE_INPUT;
	}
	SinkHashGroupJoinMatches(*this, sink, state, target, state.matched_addresses, state.payload, match_count);
	return OperatorResultType::NEED_MORE_INPUT;
}

static void MergeHashGroupJoinLocalStates(const PhysicalHashGroupJoin &op, ClientContext &context) {
	auto &global_state = op.op_state->Cast<HashGroupJoinGlobalOperatorState>();
	lock_guard<mutex> guard(global_state.lock);
	if (global_state.merge_complete) {
		return;
	}
	auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
	if (global_state.execution_mode == GroupJoinExecutionMode::EXTERNAL) {
		if (global_state.local_probe_partitions.size() != global_state.local_probe_append_states.size()) {
			throw InternalException("HASH_GROUP_JOIN external probe partition state is inconsistent");
		}
		for (idx_t partition_idx = 0; partition_idx < global_state.local_probe_partitions.size(); partition_idx++) {
			global_state.local_probe_partitions[partition_idx]->FlushAppendState(
			    *global_state.local_probe_append_states[partition_idx]);
			global_state.probe_partitions->Combine(*global_state.local_probe_partitions[partition_idx]);
		}
		global_state.local_probe_partitions.clear();
		global_state.local_probe_append_states.clear();
		global_state.merge_complete = true;
		return;
	}
	auto &target = GetHashGroupJoinTarget(sink);
	bool has_local_groups = false;
	for (auto &local_table : global_state.local_tables) {
		has_local_groups = has_local_groups || local_table->Count() != 0;
	}
	if (global_state.execution_mode == GroupJoinExecutionMode::LOCAL && !has_local_groups &&
	    sink.matched_rows.load(std::memory_order_relaxed) != 0) {
		throw InternalException("HASH_GROUP_JOIN local states were not registered before the merge barrier");
	}
	AggregateHTUpdateState merge_state(target);
	if (has_local_groups) {
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		Vector source_addresses(LogicalType::POINTER);
		Vector target_addresses(LogicalType::POINTER);
		for (auto &local_table : global_state.local_tables) {
			if (local_table->Count() == 0) {
				continue;
			}
			target.InheritAggregateAllocators(*local_table);
			AggregateHTScanState scan_state;
			local_table->InitializeScan(scan_state);
			while (local_table->ScanGroupsAndAddresses(scan_state, group_ids, source_addresses)) {
				if (group_ids.size() == 0) {
					continue;
				}
				group_ids.Flatten();
				auto ids = FlatVector::GetData<uint64_t>(group_ids.data[0]);
				auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
				for (idx_t row_idx = 0; row_idx < group_ids.size(); row_idx++) {
					const auto group_id = NumericCast<idx_t>(ids[row_idx]);
					if (group_id >= sink.group_addresses.size()) {
						throw InternalException(
						    "HASH_GROUP_JOIN local group identifier %llu exceeds global group count %llu", group_id,
						    sink.group_addresses.size());
					}
					targets[row_idx] = sink.group_addresses[group_id];
				}
				FlatVector::SetSize(target_addresses, group_ids.size());
				RowOperations::CombineStatesRange(
				    merge_state.row_state, *local_table->GetLayoutPtr(), source_addresses, 0, *target.GetLayoutPtr(),
				    target_addresses, GetHashGroupJoinAggregateOffset(op), op.grouped_aggregate_data.bindings.size());
			}
		}
	}
	global_state.local_tables.clear();

	unsafe_vector<idx_t> distinct_filter {0};
	vector<unique_ptr<GroupedAggregateHashTable>> global_distinct_tables;
	global_distinct_tables.reserve(op.distinct_aggregates.size());
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		vector<LogicalType> group_types {LogicalType::UBIGINT};
		group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
		auto global_distinct = make_uniq<GroupedAggregateHashTable>(
		    context, BufferAllocator::Get(context), group_types, vector<LogicalType> {}, vector<AggregateObject> {},
		    GroupedAggregateHashTable::InitialCapacity(), GROUP_JOIN_LOCAL_RADIX_BITS,
		    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		DataChunk distinct_rows;
		distinct_rows.Initialize(Allocator::Get(context), group_types);
		Vector insert_addresses(LogicalType::POINTER);
		for (auto &task_tables : global_state.local_distinct_tables) {
			D_ASSERT(task_tables.size() == op.distinct_aggregates.size());
			auto &local_distinct = task_tables[distinct_idx];
			AggregateHTScanState local_scan;
			local_distinct->InitializeScan(local_scan);
			while (local_distinct->ScanGroups(local_scan, distinct_rows)) {
				global_distinct->FindOrCreateGroups(distinct_rows, insert_addresses);
			}
		}

		vector<LogicalType> payload_types = distinct.argument_types;
		if (distinct.has_filter) {
			payload_types.push_back(LogicalType::BOOLEAN);
		}
		DataChunk distinct_payload;
		distinct_payload.InitializeEmpty(payload_types);
		Vector target_addresses(LogicalType::POINTER);
		AggregateHTScanState global_scan;
		global_distinct->InitializeScan(global_scan);
		while (global_distinct->ScanGroups(global_scan, distinct_rows)) {
			distinct_rows.Flatten();
			auto ids = FlatVector::GetData<uint64_t>(distinct_rows.data[0]);
			auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
			for (idx_t row_idx = 0; row_idx < distinct_rows.size(); row_idx++) {
				auto group_id = NumericCast<idx_t>(ids[row_idx]);
				if (group_id >= sink.group_addresses.size()) {
					throw InternalException("HASH_GROUP_JOIN distinct group identifier exceeds global group count");
				}
				targets[row_idx] = sink.group_addresses[group_id];
			}
			FlatVector::SetSize(target_addresses, distinct_rows.size());
			distinct_payload.Reset();
			for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
				distinct_payload.data[child_idx].Reference(distinct_rows.data[child_idx + 1]);
			}
			if (distinct.has_filter) {
				distinct_payload.data.back().Reference(Value::BOOLEAN(true), count_t(distinct_rows.size()));
			}
			distinct_payload.SetChildCardinality(distinct_rows.size());
			target.UpdateAggregatesAtAddressesRange(merge_state, target_addresses, distinct_payload,
			                                        GetHashGroupJoinAggregateOffset(op) + distinct.aggregate_index, 1,
			                                        distinct_filter);
		}
		global_distinct_tables.push_back(std::move(global_distinct));
	}
	global_state.local_distinct_tables.clear();
	if (op.unmatched_policy == HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW) {
		ExpressionExecutor executor(context);
		for (auto &expression : op.unmatched_payload_expressions) {
			executor.AddExpression(*expression);
		}
		D_ASSERT(op.unmatched_payload_expressions.size() == op.grouped_aggregate_data.payload_types.size());
		DataChunk null_probe;
		null_probe.Initialize(Allocator::Get(context), op.unmatched_probe_types);
		DataChunk unmatched_payload;
		unmatched_payload.Initialize(Allocator::Get(context), op.grouped_aggregate_data.payload_types);
		auto aggregate_objects = CreateLocalGroupJoinAggregates(op);
		AggregateFilterDataSet filter_set;
		filter_set.Initialize(context, aggregate_objects, op.grouped_aggregate_data.payload_types);
		auto update_unmatched = [&](Vector &unmatched_addresses, idx_t unmatched_count) {
			if (unmatched_count == 0) {
				return;
			}
			unmatched_addresses.Flatten();
			null_probe.Reset();
			for (idx_t column_idx = 0; column_idx < op.unmatched_probe_types.size(); column_idx++) {
				null_probe.data[column_idx].Reference(Value(op.unmatched_probe_types[column_idx]),
				                                      count_t(unmatched_count));
			}
			null_probe.SetChildCardinality(unmatched_count);
			unmatched_payload.Reset();
			executor.Execute(null_probe, unmatched_payload);
			target.UpdateAggregatesAtAddressesRange(
			    merge_state, unmatched_addresses, unmatched_payload, GetHashGroupJoinAggregateOffset(op),
			    op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);

			for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
				auto &distinct = op.distinct_aggregates[distinct_idx];
				optional_ptr<DataChunk> argument_payload = unmatched_payload;
				Vector distinct_addresses(LogicalType::POINTER);
				idx_t distinct_count = unmatched_count;
				if (distinct.has_filter) {
					auto &filter_data = filter_set.GetFilterData(distinct.aggregate_index);
					distinct_count = filter_data.ApplyFilter(unmatched_payload);
					if (distinct_count == 0) {
						continue;
					}
					distinct_addresses.Slice(unmatched_addresses, filter_data.true_sel, distinct_count);
					distinct_addresses.Flatten();
					argument_payload = filter_data.filtered_payload;
				} else {
					distinct_addresses.Reference(unmatched_addresses);
				}
				distinct_addresses.Flatten();
				vector<LogicalType> payload_types = distinct.argument_types;
				if (distinct.has_filter) {
					payload_types.push_back(LogicalType::BOOLEAN);
				}
				DataChunk distinct_payload;
				distinct_payload.InitializeEmpty(payload_types);
				for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
					distinct_payload.data[child_idx].Reference(
					    argument_payload->data[distinct.payload_index + child_idx]);
				}
				if (distinct.has_filter) {
					distinct_payload.data.back().Reference(Value::BOOLEAN(true), count_t(distinct_count));
				}
				distinct_payload.SetChildCardinality(distinct_count);

				vector<LogicalType> group_types {LogicalType::UBIGINT};
				group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
				DataChunk distinct_groups;
				distinct_groups.Initialize(Allocator::Get(context), group_types);
				distinct_groups.data[0].SetVectorType(VectorType::FLAT_VECTOR);
				auto group_ids = FlatVector::GetDataMutable<uint64_t>(distinct_groups.data[0]);
				auto addresses = FlatVector::GetData<data_ptr_t>(distinct_addresses);
				for (idx_t row_idx = 0; row_idx < distinct_count; row_idx++) {
					group_ids[row_idx] = LoadGroupJoinId(target.GetLayout(), addresses[row_idx]);
				}
				FlatVector::SetSize(distinct_groups.data[0], distinct_count);
				for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
					distinct_groups.data[child_idx + 1].Reference(distinct_payload.data[child_idx]);
				}
				distinct_groups.SetChildCardinality(distinct_count);
				Vector inserted_addresses(LogicalType::POINTER);
				SelectionVector new_groups(STANDARD_VECTOR_SIZE);
				auto new_count = global_distinct_tables[distinct_idx]->FindOrCreateGroups(
				    distinct_groups, inserted_addresses, new_groups);
				if (new_count == 0) {
					continue;
				}
				Vector new_target_addresses(LogicalType::POINTER);
				new_target_addresses.Slice(distinct_addresses, new_groups, new_count);
				new_target_addresses.Flatten();
				DataChunk new_distinct_payload;
				new_distinct_payload.InitializeEmpty(payload_types);
				new_distinct_payload.Slice(distinct_payload, new_groups, new_count);
				target.UpdateAggregatesAtAddressesRange(merge_state, new_target_addresses, new_distinct_payload,
				                                        GetHashGroupJoinAggregateOffset(op) + distinct.aggregate_index,
				                                        1, distinct_filter);
			}
		};

		if (op.routed) {
			D_ASSERT(sink.route_matches);
			Vector unmatched_addresses(LogicalType::POINTER);
			for (idx_t owner_begin = 0; owner_begin < sink.route_group_ids.size();
			     owner_begin += STANDARD_VECTOR_SIZE) {
				auto addresses = FlatVector::GetDataMutable<data_ptr_t>(unmatched_addresses);
				idx_t unmatched_count = 0;
				auto owner_end = MinValue<idx_t>(owner_begin + STANDARD_VECTOR_SIZE, sink.route_group_ids.size());
				for (idx_t owner_id = owner_begin; owner_id < owner_end; owner_id++) {
					if (!sink.route_matches[owner_id].load(std::memory_order_relaxed)) {
						addresses[unmatched_count++] = sink.group_addresses[sink.route_group_ids[owner_id]];
					}
				}
				FlatVector::SetSize(unmatched_addresses, unmatched_count);
				update_unmatched(unmatched_addresses, unmatched_count);
			}
		} else {
			DataChunk groups;
			groups.Initialize(Allocator::Get(context), op.grouped_aggregate_data.group_types);
			DataChunk hidden_count;
			hidden_count.Initialize(Allocator::Get(context), {LogicalType::BIGINT});
			Vector row_addresses(LogicalType::POINTER);
			Vector unmatched_addresses(LogicalType::POINTER);
			SelectionVector unmatched_sel(STANDARD_VECTOR_SIZE);
			ArenaAllocator finalize_arena(Allocator::Get(context));
			RowOperationsState finalize_state(finalize_arena);
			AggregateHTScanState scan_state;
			target.InitializeScan(scan_state);
			while (target.ScanGroupsAndAddresses(scan_state, groups, row_addresses)) {
				hidden_count.Reset();
				hidden_count.SetChildCardinality(groups.size());
				RowOperations::FinalizeStatesRange(finalize_state, *target.GetLayoutPtr(), row_addresses, hidden_count,
				                                   0, GetHashGroupJoinAggregateOffset(op), 1);
				hidden_count.Flatten();
				auto counts = FlatVector::GetData<int64_t>(hidden_count.data[0]);
				idx_t unmatched_count = 0;
				for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
					if (counts[row_idx] == 0) {
						unmatched_sel.set_index(unmatched_count++, row_idx);
					}
				}
				unmatched_addresses.Slice(row_addresses, unmatched_sel, unmatched_count);
				update_unmatched(unmatched_addresses, unmatched_count);
			}
		}
	}
	global_state.merge_complete = true;
	return;
}

OperatorFinalResultType PhysicalHashGroupJoin::OperatorFinalize(Pipeline &, Event &, ClientContext &context,
                                                                OperatorFinalizeInput &) const {
	MergeHashGroupJoinLocalStates(*this, context);
	return OperatorFinalResultType::FINISHED;
}

class ExternalHashGroupJoinDistinctState {
public:
	ExternalHashGroupJoinDistinctState(const PhysicalHashGroupJoin &op, ClientContext &context) {
		auto aggregate_objects = CreateLocalGroupJoinAggregates(op);
		filter_set.Initialize(context, aggregate_objects, op.grouped_aggregate_data.payload_types);
		for (auto &distinct : op.distinct_aggregates) {
			vector<LogicalType> group_types {LogicalType::UBIGINT};
			group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			auto rows = make_uniq<ColumnDataCollection>(context, group_types);
			auto append_state = make_uniq<ColumnDataAppendState>();
			rows->InitializeAppend(*append_state);
			raw_rows.push_back(std::move(rows));
			raw_append_states.push_back(std::move(append_state));
			auto groups = make_uniq<DataChunk>();
			groups->InitializeEmpty(group_types);
			distinct_groups.push_back(std::move(groups));
		}
	}

	void Sink(const PhysicalHashGroupJoin &op, DataChunk &group_ids, DataChunk &payload) {
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			auto &distinct = op.distinct_aggregates[distinct_idx];
			auto &aggregate =
			    op.grouped_aggregate_data.aggregates[distinct.aggregate_index]->Cast<BoundAggregateExpression>();
			auto &groups = *distinct_groups[distinct_idx];
			groups.Reset();
			idx_t count;
			optional_ptr<DataChunk> argument_payload;
			if (distinct.has_filter) {
				auto &filter_data = filter_set.GetFilterData(distinct.aggregate_index);
				count = filter_data.ApplyFilter(payload);
				if (count == 0) {
					continue;
				}
				groups.data[0].Slice(group_ids.data[0], filter_data.true_sel, count);
				argument_payload = filter_data.filtered_payload;
			} else {
				count = payload.size();
				groups.data[0].Reference(group_ids.data[0]);
				argument_payload = payload;
			}
			for (idx_t child_idx = 0; child_idx < aggregate.GetChildren().size(); child_idx++) {
				groups.data[child_idx + 1].Reference(argument_payload->data[distinct.payload_index + child_idx]);
			}
			groups.SetChildCardinality(count);
			raw_rows[distinct_idx]->Append(*raw_append_states[distinct_idx], groups);
		}
	}

	void Finalize(const PhysicalHashGroupJoin &op, GroupedAggregateHashTable &target,
	              const vector<data_ptr_t> &group_addresses, AggregateHTUpdateState &update_state,
	              ClientContext &context) {
		unsafe_vector<idx_t> distinct_filter {0};
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			auto &distinct = op.distinct_aggregates[distinct_idx];
			vector<LogicalType> group_types {LogicalType::UBIGINT};
			group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			auto partition_types = group_types;
			partition_types.push_back(LogicalType::HASH);
			auto partitions = make_uniq<RadixPartitionedColumnData>(
			    context, partition_types, GROUP_JOIN_EXTERNAL_RADIX_BITS, partition_types.size() - 1);
			auto local_partitions = partitions->CreateShared();
			PartitionedColumnDataAppendState partition_append_state;
			local_partitions->InitializeAppendState(partition_append_state);
			DataChunk partition_chunk;
			partition_chunk.InitializeEmpty(partition_types);
			Vector hashes(LogicalType::HASH);
			DataChunk raw_chunk;
			raw_rows[distinct_idx]->InitializeScanChunk(Allocator::Get(context), raw_chunk);
			ColumnDataScanState raw_scan;
			raw_rows[distinct_idx]->InitializeScan(raw_scan);
			while (raw_rows[distinct_idx]->Scan(raw_scan, raw_chunk)) {
				raw_chunk.Hash(hashes);
				partition_chunk.Reset();
				for (idx_t column_idx = 0; column_idx < group_types.size(); column_idx++) {
					partition_chunk.data[column_idx].Reference(raw_chunk.data[column_idx]);
				}
				partition_chunk.data.back().Reference(hashes);
				partition_chunk.SetChildCardinality(raw_chunk.size());
				local_partitions->Append(partition_append_state, partition_chunk);
			}
			local_partitions->FlushAppendState(partition_append_state);
			partitions->Combine(*local_partitions);
			local_partitions.reset();
			raw_rows[distinct_idx].reset();
			raw_append_states[distinct_idx].reset();
			vector<LogicalType> payload_types = distinct.argument_types;
			if (distinct.has_filter) {
				payload_types.push_back(LogicalType::BOOLEAN);
			}
			auto &partition_collections = partitions->GetPartitions();
			for (auto &partition : partition_collections) {
				if (!partition || partition->Count() == 0) {
					continue;
				}
				auto table = make_uniq<GroupedAggregateHashTable>(
				    context, BufferAllocator::Get(context), group_types, vector<LogicalType> {},
				    vector<AggregateObject> {}, GroupedAggregateHashTable::InitialCapacity(),
				    GROUP_JOIN_LOCAL_RADIX_BITS, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
				DataChunk rows;
				partition->InitializeScanChunk(Allocator::Get(context), rows);
				ColumnDataScanState scan;
				partition->InitializeScan(scan);
				DataChunk groups;
				groups.InitializeEmpty(group_types);
				Vector insert_addresses(LogicalType::POINTER);
				SelectionVector new_groups(STANDARD_VECTOR_SIZE);
				Vector target_addresses(LogicalType::POINTER);
				DataChunk distinct_payload;
				distinct_payload.InitializeEmpty(payload_types);
				while (partition->Scan(scan, rows)) {
					groups.Reset();
					for (idx_t column_idx = 0; column_idx < group_types.size(); column_idx++) {
						groups.data[column_idx].Reference(rows.data[column_idx]);
					}
					groups.SetChildCardinality(rows.size());
					auto new_count = table->FindOrCreateGroups(groups, insert_addresses, new_groups);
					if (new_count == 0) {
						continue;
					}
					groups.Flatten();
					auto ids = FlatVector::GetData<uint64_t>(groups.data[0]);
					auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
					for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
						auto row_idx = new_groups.get_index_unsafe(new_idx);
						auto group_id = NumericCast<idx_t>(ids[row_idx]);
						if (group_id >= group_addresses.size()) {
							throw InternalException("HASH_GROUP_JOIN external distinct identifier is out of range");
						}
						targets[new_idx] = group_addresses[group_id];
					}
					FlatVector::SetSize(target_addresses, new_count);
					distinct_payload.Reset();
					for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
						distinct_payload.data[child_idx].Slice(groups.data[child_idx + 1], new_groups, new_count);
					}
					if (distinct.has_filter) {
						distinct_payload.data.back().Reference(Value::BOOLEAN(true), count_t(new_count));
					}
					distinct_payload.SetChildCardinality(new_count);
					target.UpdateAggregatesAtAddressesRange(
					    update_state, target_addresses, distinct_payload,
					    GetHashGroupJoinAggregateOffset(op) + distinct.aggregate_index, 1, distinct_filter);
				}
				partition.reset();
			}
		}
	}

	AggregateFilterDataSet filter_set;
	vector<unique_ptr<ColumnDataCollection>> raw_rows;
	vector<unique_ptr<ColumnDataAppendState>> raw_append_states;
	vector<unique_ptr<DataChunk>> distinct_groups;
};

static void UpdateExternalHashGroupJoinUnmatched(const PhysicalHashGroupJoin &op, ClientContext &context,
                                                 GroupedAggregateHashTable &target,
                                                 ExternalHashGroupJoinDistinctState &distinct_state) {
	if (op.unmatched_policy != HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW) {
		return;
	}
	ExpressionExecutor executor(context);
	for (auto &expression : op.unmatched_payload_expressions) {
		executor.AddExpression(*expression);
	}
	DataChunk null_probe;
	null_probe.Initialize(Allocator::Get(context), op.unmatched_probe_types);
	DataChunk unmatched_payload;
	unmatched_payload.Initialize(Allocator::Get(context), op.grouped_aggregate_data.payload_types);
	DataChunk groups;
	groups.Initialize(Allocator::Get(context), op.grouped_aggregate_data.group_types);
	DataChunk hidden_count;
	hidden_count.Initialize(Allocator::Get(context), {LogicalType::BIGINT});
	DataChunk group_ids;
	group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
	Vector row_addresses(LogicalType::POINTER);
	Vector unmatched_addresses(LogicalType::POINTER);
	SelectionVector unmatched_sel(STANDARD_VECTOR_SIZE);
	ArenaAllocator finalize_arena(Allocator::Get(context));
	RowOperationsState finalize_state(finalize_arena);
	AggregateHTUpdateState update_state(target);
	AggregateHTScanState scan;
	target.InitializeScan(scan);
	while (target.ScanGroupsAndAddresses(scan, groups, row_addresses)) {
		hidden_count.Reset();
		hidden_count.SetChildCardinality(groups.size());
		RowOperations::FinalizeStatesRange(finalize_state, *target.GetLayoutPtr(), row_addresses, hidden_count, 0,
		                                   GetHashGroupJoinAggregateOffset(op), 1);
		hidden_count.Flatten();
		auto counts = FlatVector::GetData<int64_t>(hidden_count.data[0]);
		idx_t unmatched_count = 0;
		for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
			if (counts[row_idx] == 0) {
				unmatched_sel.set_index(unmatched_count++, row_idx);
			}
		}
		if (unmatched_count == 0) {
			continue;
		}
		unmatched_addresses.Slice(row_addresses, unmatched_sel, unmatched_count);
		unmatched_addresses.Flatten();
		null_probe.Reset();
		for (idx_t column_idx = 0; column_idx < op.unmatched_probe_types.size(); column_idx++) {
			null_probe.data[column_idx].Reference(Value(op.unmatched_probe_types[column_idx]),
			                                      count_t(unmatched_count));
		}
		null_probe.SetChildCardinality(unmatched_count);
		unmatched_payload.Reset();
		executor.Execute(null_probe, unmatched_payload);
		target.UpdateAggregatesAtAddressesRange(update_state, unmatched_addresses, unmatched_payload,
		                                        GetHashGroupJoinAggregateOffset(op),
		                                        op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);

		group_ids.Reset();
		group_ids.SetChildCardinality(unmatched_count);
		group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
		auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
		auto addresses = FlatVector::GetData<data_ptr_t>(unmatched_addresses);
		for (idx_t row_idx = 0; row_idx < unmatched_count; row_idx++) {
			ids[row_idx] = LoadGroupJoinId(target.GetLayout(), addresses[row_idx]);
		}
		FlatVector::SetSize(group_ids.data[0], unmatched_count);
		distinct_state.Sink(op, group_ids, unmatched_payload);
	}
}

static void AppendExternalHashGroupJoinTarget(const PhysicalHashGroupJoin &op, ClientContext &context,
                                              GroupedAggregateHashTable &target, ColumnDataCollection &output) {
	DataChunk groups;
	groups.Initialize(Allocator::Get(context), op.grouped_aggregate_data.group_types);
	DataChunk hidden_count;
	hidden_count.Initialize(Allocator::Get(context), {LogicalType::BIGINT});
	DataChunk result;
	result.Initialize(Allocator::Get(context), op.GetTypes());
	Vector row_addresses(LogicalType::POINTER);
	Vector selected_addresses(LogicalType::POINTER);
	SelectionVector selected(STANDARD_VECTOR_SIZE);
	ArenaAllocator arena(Allocator::Get(context));
	RowOperationsState row_state(arena);
	AggregateHTScanState scan;
	target.InitializeScan(scan);
	while (target.ScanGroupsAndAddresses(scan, groups, row_addresses)) {
		hidden_count.Reset();
		hidden_count.SetChildCardinality(groups.size());
		RowOperations::FinalizeStatesRange(row_state, *target.GetLayoutPtr(), row_addresses, hidden_count, 0,
		                                   GetHashGroupJoinAggregateOffset(op), 1);
		hidden_count.Flatten();
		auto counts = FlatVector::GetData<int64_t>(hidden_count.data[0]);
		idx_t selected_count = 0;
		for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
			if (op.unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD || counts[row_idx] > 0) {
				selected.set_index(selected_count++, row_idx);
			}
		}
		if (selected_count == 0) {
			continue;
		}
		result.Reset();
		for (idx_t group_idx = 0; group_idx < op.output_groups.size(); group_idx++) {
			auto &output_group = op.output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::KEY) {
				result.data[group_idx].Slice(groups.data[output_group.index], selected, selected_count);
			}
		}
		result.SetChildCardinality(selected_count);
		selected_addresses.Slice(row_addresses, selected, selected_count);
		selected_addresses.Flatten();
		for (idx_t group_idx = 0; group_idx < op.output_groups.size(); group_idx++) {
			auto &output_group = op.output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::OWNER_PAYLOAD) {
				RowOperations::FinalizeStatesRange(row_state, *target.GetLayoutPtr(), selected_addresses, result,
				                                   group_idx, output_group.index + 1, 1);
			}
		}
		RowOperations::FinalizeStatesRange(row_state, *target.GetLayoutPtr(), selected_addresses, result,
		                                   op.output_groups.size(), GetHashGroupJoinAggregateOffset(op) + 1,
		                                   op.grouped_aggregate_data.aggregates.size() - 1);
		output.Append(result);
	}
}

class StaticHashGroupJoinResults {
public:
	StaticHashGroupJoinResults(const PhysicalHashGroupJoin &op, ClientContext &context,
	                           GroupedAggregateHashTable &target, const vector<data_ptr_t> &group_addresses) {
		auto &result_types = op.grouped_aggregate_data.aggregate_return_types;
		ArenaAllocator arena(Allocator::Get(context));
		RowOperationsState row_state(arena);
		for (idx_t group_begin = 0; group_begin < group_addresses.size(); group_begin += STANDARD_VECTOR_SIZE) {
			auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, group_addresses.size() - group_begin);
			Vector addresses(LogicalType::POINTER);
			auto address_data = FlatVector::GetDataMutable<data_ptr_t>(addresses);
			for (idx_t row_idx = 0; row_idx < count; row_idx++) {
				address_data[row_idx] = group_addresses[group_begin + row_idx];
			}
			FlatVector::SetSize(addresses, count);
			auto result = make_uniq<DataChunk>();
			result->Initialize(Allocator::Get(context), result_types);
			result->SetChildCardinality(count);
			RowOperations::FinalizeStatesRange(row_state, *target.GetLayoutPtr(), addresses, *result, 0,
			                                   GetHashGroupJoinAggregateOffset(op), result_types.size());
			chunks.push_back(std::move(result));
		}
	}

	Value GetValue(idx_t aggregate_idx, idx_t group_id) const {
		auto chunk_idx = group_id / STANDARD_VECTOR_SIZE;
		if (chunk_idx >= chunks.size()) {
			throw InternalException("HASH_GROUP_JOIN static group identifier is out of range");
		}
		auto row_idx = group_id % STANDARD_VECTOR_SIZE;
		if (aggregate_idx >= chunks[chunk_idx]->ColumnCount() || row_idx >= chunks[chunk_idx]->size()) {
			throw InternalException("HASH_GROUP_JOIN static aggregate result is out of range");
		}
		return chunks[chunk_idx]->GetValue(aggregate_idx, row_idx);
	}

	int64_t GetCount(idx_t group_id) const {
		return GetValue(0, group_id).GetValue<int64_t>();
	}

private:
	vector<unique_ptr<DataChunk>> chunks;
};

static void AppendExternalStaticHashGroupJoin(const PhysicalHashGroupJoin &op, ClientContext &context,
                                              GroupedAggregateHashTable &target, ColumnDataCollection &owner_partition,
                                              const vector<data_ptr_t> &group_addresses, ColumnDataCollection &output) {
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	StaticHashGroupJoinResults static_results(op, context, target, group_addresses);
	DataChunk owner_rows;
	owner_partition.InitializeScanChunk(Allocator::Get(context), owner_rows);
	ColumnDataScanState owner_scan;
	owner_partition.InitializeScan(owner_scan);
	DataChunk keys;
	keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
	DataChunk result;
	result.Initialize(Allocator::Get(context), op.GetTypes());
	AggregateHTLookupState lookup_state;
	SelectionVector found(STANDARD_VECTOR_SIZE);
	while (owner_partition.Scan(owner_scan, owner_rows)) {
		keys.Reset();
		for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
			keys.data[key_idx].Reference(owner_rows.data[key_idx]);
		}
		keys.SetChildCardinality(owner_rows.size());
		auto found_count = target.LookupGroups(keys, lookup_state, found);
		if (found_count != owner_rows.size()) {
			throw InternalException("HASH_GROUP_JOIN external static owner row lost its destination");
		}
		auto addresses = FlatVector::GetData<data_ptr_t>(lookup_state.addresses);
		vector<idx_t> group_ids(owner_rows.size());
		for (idx_t row_idx = 0; row_idx < owner_rows.size(); row_idx++) {
			group_ids[row_idx] = LoadGroupJoinId(target.GetLayout(), addresses[row_idx]);
		}
		result.Reset();
		for (idx_t output_idx = 0; output_idx < op.output_columns.size(); output_idx++) {
			auto &column = op.output_columns[output_idx];
			switch (column.source) {
			case HashGroupJoinOutputSource::KEY:
				result.data[output_idx].Reference(owner_rows.data[column.index]);
				break;
			case HashGroupJoinOutputSource::OWNER_PAYLOAD:
				result.data[output_idx].Reference(owner_rows.data[key_count + column.index]);
				break;
			case HashGroupJoinOutputSource::MATCHED_KEY:
				VectorOperations::Copy(owner_rows.data[column.index], result.data[output_idx], owner_rows.size(), 0, 0);
				for (idx_t row_idx = 0; row_idx < owner_rows.size(); row_idx++) {
					if (static_results.GetCount(group_ids[row_idx]) == 0) {
						FlatVector::ValidityMutable(result.data[output_idx]).SetInvalid(row_idx);
					}
				}
				break;
			case HashGroupJoinOutputSource::AGGREGATE:
				for (idx_t row_idx = 0; row_idx < owner_rows.size(); row_idx++) {
					result.data[output_idx].SetValue(row_idx,
					                                 static_results.GetValue(1 + column.index, group_ids[row_idx]));
				}
				break;
			}
		}
		result.SetChildCardinality(owner_rows.size());
		output.Append(result);
	}
}

class ExternalRoutedHashGroupJoinState {
public:
	ExternalRoutedHashGroupJoinState(const PhysicalHashGroupJoin &op, ClientContext &context)
	    : hashes(LogicalType::HASH) {
		auto types = op.output_group_types;
		types.insert(types.end(), op.grouped_aggregate_data.payload_types.begin(),
		             op.grouped_aggregate_data.payload_types.end());
		types.push_back(LogicalType::HASH);
		partitions =
		    make_uniq<RadixPartitionedColumnData>(context, types, GROUP_JOIN_EXTERNAL_RADIX_BITS, types.size() - 1);
		local_partitions = partitions->CreateShared();
		local_partitions->InitializeAppendState(append_state);
		partition_chunk.InitializeEmpty(types);
	}

	void Append(DataChunk &groups, DataChunk &payload) {
		D_ASSERT(groups.size() == payload.size());
		if (groups.size() == 0) {
			return;
		}
		groups.Hash(hashes);
		partition_chunk.Reset();
		idx_t output_idx = 0;
		for (idx_t group_idx = 0; group_idx < groups.ColumnCount(); group_idx++) {
			partition_chunk.data[output_idx++].Reference(groups.data[group_idx]);
		}
		for (idx_t payload_idx = 0; payload_idx < payload.ColumnCount(); payload_idx++) {
			partition_chunk.data[output_idx++].Reference(payload.data[payload_idx]);
		}
		partition_chunk.data[output_idx].Reference(hashes);
		partition_chunk.SetChildCardinality(groups.size());
		local_partitions->Append(append_state, partition_chunk);
	}

	void Finalize() {
		local_partitions->FlushAppendState(append_state);
		partitions->Combine(*local_partitions);
		local_partitions.reset();
	}

	unique_ptr<RadixPartitionedColumnData> partitions;
	unique_ptr<PartitionedColumnData> local_partitions;
	PartitionedColumnDataAppendState append_state;
	DataChunk partition_chunk;
	Vector hashes;
};

static void ProcessExternalRoutedLookupPartition(const PhysicalHashGroupJoin &op, ClientContext &context,
                                                 ColumnDataCollection &owner_partition,
                                                 optional_ptr<ColumnDataCollection> probe_partition,
                                                 ExternalRoutedHashGroupJoinState &routed_state) {
	auto routing_target = make_uniq<GroupedAggregateHashTable>(
	    context, BufferAllocator::Get(context), op.grouped_aggregate_data.group_types, vector<LogicalType> {},
	    CreateRoutingGroupJoinAggregates(), GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
	    op.null_equal || !op.unique_owner ? TupleDataValidityType::CAN_HAVE_NULL_VALUES
	                                      : TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	vector<vector<idx_t>> route_build_rows;
	idx_t owner_row_count = 0;
	DataChunk owner_rows;
	owner_partition.InitializeScanChunk(Allocator::Get(context), owner_rows);
	ColumnDataScanState owner_scan;
	owner_partition.InitializeScan(owner_scan);
	DataChunk owner_keys;
	owner_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
	Vector owner_addresses(LogicalType::POINTER);
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	while (owner_partition.Scan(owner_scan, owner_rows)) {
		owner_keys.Reset();
		for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
			owner_keys.data[key_idx].Reference(owner_rows.data[key_idx]);
		}
		owner_keys.SetChildCardinality(owner_rows.size());
		auto new_count = routing_target->FindOrCreateGroups(owner_keys, owner_addresses, new_groups);
		if (op.unique_owner && new_count != owner_rows.size()) {
			throw InternalException("HASH_GROUP_JOIN external routed owner uniqueness proof was violated");
		}
		auto addresses = FlatVector::GetData<data_ptr_t>(owner_addresses);
		for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
			auto row_idx = new_groups.get_index_unsafe(new_idx);
			auto route_id = route_build_rows.size();
			StoreGroupJoinId(routing_target->GetLayout(), addresses[row_idx], route_id);
			route_build_rows.emplace_back();
		}
		for (idx_t row_idx = 0; row_idx < owner_rows.size(); row_idx++) {
			auto route_id = LoadGroupJoinId(routing_target->GetLayout(), addresses[row_idx]);
			D_ASSERT(route_id < route_build_rows.size());
			auto &routes = route_build_rows[route_id];
			if (!op.single_match || routes.empty()) {
				routes.push_back(owner_row_count + row_idx);
			}
		}
		owner_row_count += owner_rows.size();
	}

	vector<idx_t> route_offsets;
	vector<idx_t> route_owner_rows;
	route_offsets.reserve(route_build_rows.size() + 1);
	for (auto &routes : route_build_rows) {
		D_ASSERT(!routes.empty());
		route_offsets.push_back(route_owner_rows.size());
		route_owner_rows.insert(route_owner_rows.end(), routes.begin(), routes.end());
	}
	route_offsets.push_back(route_owner_rows.size());
	route_build_rows.clear();
	D_ASSERT(owner_partition.Count() == owner_row_count);
	vector<bool> matched(owner_row_count, false);
	DataChunk final_groups;
	final_groups.Initialize(Allocator::Get(context), op.output_group_types);
	DataChunk routed_payload;
	routed_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
	SelectionVector route_input(STANDARD_VECTOR_SIZE);
	DataChunk owner_lookup_rows;
	owner_partition.InitializeScanChunk(Allocator::Get(context), owner_lookup_rows);
	ColumnDataScanState owner_lookup_scan;
	owner_partition.InitializeScan(owner_lookup_scan);
	auto set_owner_groups = [&](idx_t output_row, idx_t owner_row) {
		if (!owner_partition.Seek(owner_row, owner_lookup_scan, owner_lookup_rows)) {
			throw InternalException("HASH_GROUP_JOIN external owner row is out of range");
		}
		auto chunk_row = owner_row - owner_lookup_scan.current_row_index;
		for (idx_t group_idx = 0; group_idx < op.output_groups.size(); group_idx++) {
			auto &output_group = op.output_groups[group_idx];
			auto owner_column = output_group.source == HashGroupJoinOutputSource::KEY ? output_group.index
			                                                                          : key_count + output_group.index;
			VectorOperations::Copy(owner_lookup_rows.data[owner_column], final_groups.data[group_idx], chunk_row + 1,
			                       chunk_row, output_row);
		}
	};
	if (probe_partition) {
		DataChunk probe_rows;
		probe_partition->InitializeScanChunk(Allocator::Get(context), probe_rows);
		ColumnDataScanState probe_scan;
		probe_partition->InitializeScan(probe_scan);
		DataChunk probe_keys;
		probe_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		DataChunk matched_payload;
		matched_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		AggregateHTLookupState lookup_state;
		SelectionVector found(STANDARD_VECTOR_SIZE);
		SelectionVector matched_input(STANDARD_VECTOR_SIZE);
		Vector matched_addresses(LogicalType::POINTER);
		vector<idx_t> route_owner_buffer(STANDARD_VECTOR_SIZE);
		vector<idx_t> route_match_buffer(STANDARD_VECTOR_SIZE);
		vector<idx_t> route_order(STANDARD_VECTOR_SIZE);
		while (probe_partition->Scan(probe_scan, probe_rows)) {
			probe_keys.Reset();
			for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
				probe_keys.data[key_idx].Reference(probe_rows.data[key_idx]);
			}
			probe_keys.SetChildCardinality(probe_rows.size());
			auto match_count = routing_target->LookupGroups(probe_keys, lookup_state, found);
			if (match_count == 0) {
				continue;
			}
			matched_payload.Reset();
			if (match_count == probe_rows.size()) {
				matched_addresses.Reference(lookup_state.addresses);
				for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size();
				     payload_idx++) {
					matched_payload.data[payload_idx].Reference(probe_rows.data[key_count + payload_idx]);
				}
			} else {
				auto targets = FlatVector::GetDataMutable<data_ptr_t>(matched_addresses);
				auto lookup_addresses = FlatVector::GetData<data_ptr_t>(lookup_state.addresses);
				for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
					auto input_idx = found.get_index_unsafe(match_idx);
					matched_input.set_index(match_idx, input_idx);
					targets[match_idx] = lookup_addresses[input_idx];
				}
				FlatVector::SetSize(matched_addresses, match_count);
				for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size();
				     payload_idx++) {
					matched_payload.data[payload_idx].Slice(probe_rows.data[key_count + payload_idx], matched_input,
					                                        match_count);
				}
			}
			matched_payload.SetChildCardinality(match_count);
			auto addresses = FlatVector::GetData<data_ptr_t>(matched_addresses);
			idx_t routed_count = 0;
			auto flush_routes = [&]() {
				if (routed_count == 0) {
					return;
				}
				for (idx_t route_idx = 0; route_idx < routed_count; route_idx++) {
					route_order[route_idx] = route_idx;
				}
				std::sort(
				    route_order.begin(),
				    route_order.begin() + NumericCast<vector<idx_t>::difference_type>(routed_count),
				    [&](idx_t left, idx_t right) { return route_owner_buffer[left] < route_owner_buffer[right]; });
				for (idx_t output_idx = 0; output_idx < routed_count; output_idx++) {
					auto route_idx = route_order[output_idx];
					auto owner_row = route_owner_buffer[route_idx];
					matched[owner_row] = true;
					set_owner_groups(output_idx, owner_row);
					route_input.set_index(output_idx, route_match_buffer[route_idx]);
				}
				final_groups.SetChildCardinality(routed_count);
				routed_payload.Reset();
				routed_payload.Slice(matched_payload, route_input, routed_count);
				if (routed_payload.ColumnCount() == 0) {
					routed_payload.SetChildCardinality(routed_count);
				}
				routed_state.Append(final_groups, routed_payload);
				final_groups.Reset();
				routed_count = 0;
			};
			for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
				auto route_id = LoadGroupJoinId(routing_target->GetLayout(), addresses[match_idx]);
				if (route_id + 1 >= route_offsets.size()) {
					throw InternalException("HASH_GROUP_JOIN external routing identifier exceeds route count");
				}
				for (idx_t route_idx = route_offsets[route_id]; route_idx < route_offsets[route_id + 1]; route_idx++) {
					auto owner_row = route_owner_rows[route_idx];
					route_owner_buffer[routed_count] = owner_row;
					route_match_buffer[routed_count++] = match_idx;
					if (routed_count == STANDARD_VECTOR_SIZE) {
						flush_routes();
					}
				}
			}
			flush_routes();
		}
	}

	if (op.unmatched_policy == HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW) {
		ExpressionExecutor executor(context);
		for (auto &expression : op.unmatched_payload_expressions) {
			executor.AddExpression(*expression);
		}
		DataChunk null_probe;
		null_probe.Initialize(Allocator::Get(context), op.unmatched_probe_types);
		DataChunk unmatched_payload;
		unmatched_payload.Initialize(Allocator::Get(context), op.grouped_aggregate_data.payload_types);
		idx_t unmatched_count = 0;
		auto flush_unmatched = [&]() {
			if (unmatched_count == 0) {
				return;
			}
			final_groups.SetChildCardinality(unmatched_count);
			null_probe.Reset();
			for (idx_t column_idx = 0; column_idx < op.unmatched_probe_types.size(); column_idx++) {
				null_probe.data[column_idx].Reference(Value(op.unmatched_probe_types[column_idx]),
				                                      count_t(unmatched_count));
			}
			null_probe.SetChildCardinality(unmatched_count);
			unmatched_payload.Reset();
			executor.Execute(null_probe, unmatched_payload);
			routed_state.Append(final_groups, unmatched_payload);
			final_groups.Reset();
			unmatched_count = 0;
		};
		for (idx_t owner_row = 0; owner_row < matched.size(); owner_row++) {
			if (matched[owner_row]) {
				continue;
			}
			set_owner_groups(unmatched_count++, owner_row);
			if (unmatched_count == STANDARD_VECTOR_SIZE) {
				flush_unmatched();
			}
		}
		flush_unmatched();
	}
}

static void AppendExternalRoutedHashGroupJoinTarget(const PhysicalHashGroupJoin &op, ClientContext &context,
                                                    GroupedAggregateHashTable &target, ColumnDataCollection &output) {
	DataChunk groups;
	groups.Initialize(Allocator::Get(context), op.output_group_types);
	DataChunk result;
	result.Initialize(Allocator::Get(context), op.GetTypes());
	Vector row_addresses(LogicalType::POINTER);
	ArenaAllocator arena(Allocator::Get(context));
	RowOperationsState row_state(arena);
	AggregateHTScanState scan;
	target.InitializeScan(scan);
	while (target.ScanGroupsAndAddresses(scan, groups, row_addresses)) {
		result.Reset();
		for (idx_t group_idx = 0; group_idx < op.output_group_types.size(); group_idx++) {
			result.data[group_idx].Reference(groups.data[group_idx]);
		}
		result.SetChildCardinality(groups.size());
		RowOperations::FinalizeStatesRange(row_state, *target.GetLayoutPtr(), row_addresses, result,
		                                   op.output_group_types.size(), GetHashGroupJoinAggregateOffset(op) + 1,
		                                   op.grouped_aggregate_data.aggregates.size() - 1);
		output.Append(result);
	}
}

static void ProcessExternalRoutedAggregatePartition(const PhysicalHashGroupJoin &op, ClientContext &context,
                                                    ColumnDataCollection &partition, ColumnDataCollection &output) {
	auto target = make_uniq<GroupedAggregateHashTable>(
	    context, BufferAllocator::Get(context), op.output_group_types, op.grouped_aggregate_data.payload_types,
	    CreateRoutedGroupJoinAggregates(op), GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
	    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	AggregateHTUpdateState update_state(*target);
	vector<data_ptr_t> group_addresses;
	DataChunk rows;
	partition.InitializeScanChunk(Allocator::Get(context), rows);
	ColumnDataScanState scan;
	partition.InitializeScan(scan);
	DataChunk groups;
	groups.InitializeEmpty(op.output_group_types);
	DataChunk payload;
	payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
	Vector addresses(LogicalType::POINTER);
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	ExternalHashGroupJoinDistinctState distinct_state(op, context);
	while (partition.Scan(scan, rows)) {
		groups.Reset();
		for (idx_t group_idx = 0; group_idx < op.output_group_types.size(); group_idx++) {
			groups.data[group_idx].Reference(rows.data[group_idx]);
		}
		groups.SetChildCardinality(rows.size());
		payload.Reset();
		for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size(); payload_idx++) {
			payload.data[payload_idx].Reference(rows.data[op.output_group_types.size() + payload_idx]);
		}
		payload.SetChildCardinality(rows.size());
		auto new_count = target->FindOrCreateGroups(groups, addresses, new_groups);
		auto address_data = FlatVector::GetData<data_ptr_t>(addresses);
		for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
			auto row_idx = new_groups.get_index_unsafe(new_idx);
			auto group_id = group_addresses.size();
			StoreGroupJoinId(target->GetLayout(), address_data[row_idx], group_id);
			group_addresses.push_back(address_data[row_idx]);
		}
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		group_ids.SetChildCardinality(rows.size());
		auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
		for (idx_t row_idx = 0; row_idx < rows.size(); row_idx++) {
			ids[row_idx] = LoadGroupJoinId(target->GetLayout(), address_data[row_idx]);
		}
		FlatVector::SetSize(group_ids.data[0], rows.size());
		distinct_state.Sink(op, group_ids, payload);
		target->UpdateAggregatesAtAddressesRange(update_state, addresses, payload, GetHashGroupJoinAggregateOffset(op),
		                                         op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
	}
	distinct_state.Finalize(op, *target, group_addresses, update_state, context);
	AppendExternalRoutedHashGroupJoinTarget(op, context, *target, output);
}

static void ProcessExternalHashGroupJoinPartition(const PhysicalHashGroupJoin &op, ClientContext &context,
                                                  ColumnDataCollection &owner_partition,
                                                  optional_ptr<ColumnDataCollection> probe_partition,
                                                  ColumnDataCollection &output) {
	auto target = make_uniq<GroupedAggregateHashTable>(
	    context, BufferAllocator::Get(context), op.grouped_aggregate_data.group_types,
	    CreateGlobalGroupJoinPayloadTypes(op), CreateGlobalGroupJoinAggregates(op),
	    GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
	    op.null_equal ? TupleDataValidityType::CAN_HAVE_NULL_VALUES : TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
	AggregateHTUpdateState update_state(*target);
	const auto key_count = op.grouped_aggregate_data.group_types.size();
	vector<data_ptr_t> group_addresses;
	DataChunk owner_rows;
	owner_partition.InitializeScanChunk(Allocator::Get(context), owner_rows);
	ColumnDataScanState owner_scan;
	owner_partition.InitializeScan(owner_scan);
	DataChunk owner_keys;
	owner_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
	DataChunk owner_payload;
	owner_payload.InitializeEmpty(op.owner_payload_data.payload_types);
	Vector owner_addresses(LogicalType::POINTER);
	SelectionVector new_groups(STANDARD_VECTOR_SIZE);
	while (owner_partition.Scan(owner_scan, owner_rows)) {
		owner_keys.Reset();
		for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
			owner_keys.data[key_idx].Reference(owner_rows.data[key_idx]);
		}
		owner_keys.SetChildCardinality(owner_rows.size());
		auto new_count = target->FindOrCreateGroups(owner_keys, owner_addresses, new_groups);
		if (!op.static_mode && new_count != owner_rows.size()) {
			throw InternalException("HASH_GROUP_JOIN external owner uniqueness proof was violated");
		}
		auto addresses = FlatVector::GetData<data_ptr_t>(owner_addresses);
		for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
			auto row_idx = new_groups.get_index_unsafe(new_idx);
			auto group_id = group_addresses.size();
			StoreGroupJoinId(target->GetLayout(), addresses[row_idx], group_id);
			group_addresses.push_back(addresses[row_idx]);
		}
		if (!op.owner_payload_data.aggregates.empty()) {
			owner_payload.Reset();
			for (idx_t payload_idx = 0; payload_idx < op.owner_payload_data.payload_types.size(); payload_idx++) {
				owner_payload.data[payload_idx].Reference(owner_rows.data[key_count + payload_idx]);
			}
			owner_payload.SetChildCardinality(owner_rows.size());
			target->UpdateAggregatesAtAddressesRange(update_state, owner_addresses, owner_payload, 1,
			                                         op.owner_payload_data.aggregates.size());
		}
	}

	ExternalHashGroupJoinDistinctState distinct_state(op, context);
	if (probe_partition) {
		DataChunk probe_rows;
		probe_partition->InitializeScanChunk(Allocator::Get(context), probe_rows);
		ColumnDataScanState probe_scan;
		probe_partition->InitializeScan(probe_scan);
		DataChunk probe_keys;
		probe_keys.InitializeEmpty(op.grouped_aggregate_data.group_types);
		DataChunk payload;
		payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		DataChunk matched_payload;
		matched_payload.InitializeEmpty(op.grouped_aggregate_data.payload_types);
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		AggregateHTLookupState lookup_state;
		SelectionVector found(STANDARD_VECTOR_SIZE);
		SelectionVector matched_input(STANDARD_VECTOR_SIZE);
		Vector matched_addresses(LogicalType::POINTER);
		while (probe_partition->Scan(probe_scan, probe_rows)) {
			probe_keys.Reset();
			for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
				probe_keys.data[key_idx].Reference(probe_rows.data[key_idx]);
			}
			probe_keys.SetChildCardinality(probe_rows.size());
			auto match_count = target->LookupGroups(probe_keys, lookup_state, found);
			if (match_count == 0) {
				continue;
			}
			matched_payload.Reset();
			if (match_count == probe_rows.size()) {
				matched_addresses.Reference(lookup_state.addresses);
				for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size();
				     payload_idx++) {
					matched_payload.data[payload_idx].Reference(probe_rows.data[key_count + payload_idx]);
				}
			} else {
				auto targets = FlatVector::GetDataMutable<data_ptr_t>(matched_addresses);
				auto lookup_addresses = FlatVector::GetData<data_ptr_t>(lookup_state.addresses);
				for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
					auto input_idx = found.get_index_unsafe(match_idx);
					matched_input.set_index(match_idx, input_idx);
					targets[match_idx] = lookup_addresses[input_idx];
				}
				FlatVector::SetSize(matched_addresses, match_count);
				for (idx_t payload_idx = 0; payload_idx < op.grouped_aggregate_data.payload_types.size();
				     payload_idx++) {
					matched_payload.data[payload_idx].Slice(probe_rows.data[key_count + payload_idx], matched_input,
					                                        match_count);
				}
			}
			matched_payload.SetChildCardinality(match_count);
			group_ids.Reset();
			group_ids.SetChildCardinality(match_count);
			group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
			auto addresses = FlatVector::GetData<data_ptr_t>(matched_addresses);
			for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
				ids[match_idx] = LoadGroupJoinId(target->GetLayout(), addresses[match_idx]);
			}
			FlatVector::SetSize(group_ids.data[0], match_count);
			distinct_state.Sink(op, group_ids, matched_payload);
			target->UpdateAggregatesAtAddressesRange(
			    update_state, matched_addresses, matched_payload, GetHashGroupJoinAggregateOffset(op),
			    op.grouped_aggregate_data.aggregates.size(), op.non_distinct_filter);
		}
	}
	UpdateExternalHashGroupJoinUnmatched(op, context, *target, distinct_state);
	distinct_state.Finalize(op, *target, group_addresses, update_state, context);
	if (op.static_mode) {
		AppendExternalStaticHashGroupJoin(op, context, *target, owner_partition, group_addresses, output);
	} else {
		AppendExternalHashGroupJoinTarget(op, context, *target, output);
	}
}

static unique_ptr<ColumnDataCollection> BuildExternalHashGroupJoinOutput(const PhysicalHashGroupJoin &op,
                                                                         ClientContext &context) {
	auto result = make_uniq<ColumnDataCollection>(context, op.GetTypes());
	auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
	auto &operator_state = op.op_state->Cast<HashGroupJoinGlobalOperatorState>();
	auto &owner_partitions = sink.owner_partitions->GetPartitions();
	auto &probe_partitions = operator_state.probe_partitions->GetPartitions();
	if (op.routed) {
		ExternalRoutedHashGroupJoinState routed_state(op, context);
		for (idx_t partition_idx = 0; partition_idx < owner_partitions.size(); partition_idx++) {
			if (!owner_partitions[partition_idx] || owner_partitions[partition_idx]->Count() == 0) {
				continue;
			}
			optional_ptr<ColumnDataCollection> probe_partition;
			if (partition_idx < probe_partitions.size() && probe_partitions[partition_idx] &&
			    probe_partitions[partition_idx]->Count() != 0) {
				probe_partition = probe_partitions[partition_idx].get();
			}
			ProcessExternalRoutedLookupPartition(op, context, *owner_partitions[partition_idx], probe_partition,
			                                     routed_state);
			owner_partitions[partition_idx].reset();
			if (partition_idx < probe_partitions.size()) {
				probe_partitions[partition_idx].reset();
			}
		}
		routed_state.Finalize();
		auto &routed_partitions = routed_state.partitions->GetPartitions();
		for (auto &partition : routed_partitions) {
			if (!partition || partition->Count() == 0) {
				continue;
			}
			ProcessExternalRoutedAggregatePartition(op, context, *partition, *result);
			partition.reset();
		}
		return result;
	}
	for (idx_t partition_idx = 0; partition_idx < owner_partitions.size(); partition_idx++) {
		if (!owner_partitions[partition_idx] || owner_partitions[partition_idx]->Count() == 0) {
			continue;
		}
		optional_ptr<ColumnDataCollection> probe_partition;
		if (partition_idx < probe_partitions.size() && probe_partitions[partition_idx] &&
		    probe_partitions[partition_idx]->Count() != 0) {
			probe_partition = probe_partitions[partition_idx].get();
		}
		ProcessExternalHashGroupJoinPartition(op, context, *owner_partitions[partition_idx], probe_partition, *result);
		owner_partitions[partition_idx].reset();
		if (partition_idx < probe_partitions.size()) {
			probe_partitions[partition_idx].reset();
		}
	}
	return result;
}

class HashGroupJoinGlobalSourceState : public GlobalSourceState {
public:
	HashGroupJoinGlobalSourceState(const PhysicalHashGroupJoin &op, ClientContext &context) {
		MergeHashGroupJoinLocalStates(op, context);
		auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
		D_ASSERT(sink.build_finalized);
		if (sink.external) {
			external_output = BuildExternalHashGroupJoinOutput(op, context);
			external_output->InitializeScan(external_scan);
			return;
		}
		if (op.static_mode) {
			D_ASSERT(sink.static_owner_rows);
			static_results = make_uniq<StaticHashGroupJoinResults>(op, context, *sink.hash_table, sink.group_addresses);
			sink.static_owner_rows->InitializeScan(static_owner_scan);
		} else {
			GetHashGroupJoinTarget(sink).InitializeScan(scan_state);
		}
	}

	AggregateHTScanState scan_state;
	ColumnDataScanState static_owner_scan;
	unique_ptr<StaticHashGroupJoinResults> static_results;
	unique_ptr<ColumnDataCollection> external_output;
	ColumnDataScanState external_scan;
};

class HashGroupJoinLocalSourceState : public LocalSourceState {
public:
	HashGroupJoinLocalSourceState(ExecutionContext &context, const PhysicalHashGroupJoin &op)
	    : row_addresses(LogicalType::POINTER), matched_addresses(LogicalType::POINTER),
	      matched_sel(STANDARD_VECTOR_SIZE), arena(Allocator::Get(context.client)), row_state(arena) {
		groups.Initialize(Allocator::Get(context.client),
		                  op.routed ? op.output_group_types : op.grouped_aggregate_data.group_types);
		hidden_count.Initialize(Allocator::Get(context.client), {LogicalType::BIGINT});
		if (op.static_mode && !op.sink_state->Cast<HashGroupJoinGlobalSinkState>().external) {
			auto &sink = op.sink_state->Cast<HashGroupJoinGlobalSinkState>();
			sink.static_owner_rows->InitializeScanChunk(Allocator::Get(context.client), static_owner_rows);
		}
	}

	DataChunk groups;
	DataChunk hidden_count;
	DataChunk static_owner_rows;
	Vector row_addresses;
	Vector matched_addresses;
	SelectionVector matched_sel;
	ArenaAllocator arena;
	RowOperationsState row_state;
};

unique_ptr<GlobalSourceState> PhysicalHashGroupJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<HashGroupJoinGlobalSourceState>(*this, context);
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
	if (sink.external) {
		return gstate.external_output->Scan(gstate.external_scan, chunk) ? SourceResultType::HAVE_MORE_OUTPUT
		                                                                 : SourceResultType::FINISHED;
	}
	auto &target = GetHashGroupJoinTarget(sink);
	auto layout = target.GetLayoutPtr();
	const auto output_group_count = output_groups.size();
	const auto aggregate_offset = GetHashGroupJoinAggregateOffset(*this);
	const auto user_aggregate_count = grouped_aggregate_data.aggregates.size() - 1;
	if (static_mode) {
		D_ASSERT(sink.static_owner_rows);
		D_ASSERT(gstate.static_results);
		const auto key_count = grouped_aggregate_data.group_types.size();
		while (sink.static_owner_rows->Scan(gstate.static_owner_scan, state.static_owner_rows)) {
			if (state.static_owner_rows.size() == 0) {
				continue;
			}
			state.static_owner_rows.Flatten();
			auto ids = FlatVector::GetData<uint64_t>(state.static_owner_rows.data.back());
			chunk.Reset();
			for (idx_t output_idx = 0; output_idx < output_columns.size(); output_idx++) {
				auto &output = output_columns[output_idx];
				switch (output.source) {
				case HashGroupJoinOutputSource::KEY:
					chunk.data[output_idx].Reference(state.static_owner_rows.data[output.index]);
					break;
				case HashGroupJoinOutputSource::OWNER_PAYLOAD:
					chunk.data[output_idx].Reference(state.static_owner_rows.data[key_count + output.index]);
					break;
				case HashGroupJoinOutputSource::MATCHED_KEY:
					VectorOperations::Copy(state.static_owner_rows.data[output.index], chunk.data[output_idx],
					                       state.static_owner_rows.size(), 0, 0);
					for (idx_t row_idx = 0; row_idx < state.static_owner_rows.size(); row_idx++) {
						auto group_id = NumericCast<idx_t>(ids[row_idx]);
						if (gstate.static_results->GetCount(group_id) == 0) {
							FlatVector::ValidityMutable(chunk.data[output_idx]).SetInvalid(row_idx);
						}
					}
					break;
				case HashGroupJoinOutputSource::AGGREGATE:
					if (output.index >= user_aggregate_count) {
						throw InternalException("HASH_GROUP_JOIN static aggregate output index is out of range");
					}
					for (idx_t row_idx = 0; row_idx < state.static_owner_rows.size(); row_idx++) {
						auto group_id = NumericCast<idx_t>(ids[row_idx]);
						chunk.data[output_idx].SetValue(row_idx,
						                                gstate.static_results->GetValue(1 + output.index, group_id));
					}
					break;
				}
			}
			chunk.SetChildCardinality(state.static_owner_rows.size());
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		return SourceResultType::FINISHED;
	}

	while (target.ScanGroupsAndAddresses(gstate.scan_state, state.groups, state.row_addresses)) {
		if (state.groups.size() == 0) {
			continue;
		}
		state.hidden_count.Reset();
		state.hidden_count.SetChildCardinality(state.groups.size());
		RowOperations::FinalizeStatesRange(state.row_state, *layout, state.row_addresses, state.hidden_count, 0,
		                                   aggregate_offset, 1);
		state.hidden_count.Flatten();
		auto counts = FlatVector::GetData<int64_t>(state.hidden_count.data[0]);
		idx_t matched_count = 0;
		for (idx_t row_idx = 0; row_idx < state.groups.size(); row_idx++) {
			if (unmatched_policy != HashGroupJoinUnmatchedPolicy::DISCARD || counts[row_idx] > 0) {
				state.matched_sel.set_index(matched_count++, row_idx);
			}
		}
		if (matched_count == 0) {
			continue;
		}

		chunk.Reset();
		for (idx_t group_idx = 0; group_idx < output_group_count; group_idx++) {
			if (routed) {
				chunk.data[group_idx].Slice(state.groups.data[group_idx], state.matched_sel, matched_count);
			} else {
				auto &output_group = output_groups[group_idx];
				if (output_group.source == HashGroupJoinOutputSource::KEY) {
					chunk.data[group_idx].Slice(state.groups.data[output_group.index], state.matched_sel,
					                            matched_count);
				}
			}
		}
		chunk.SetChildCardinality(matched_count);
		state.matched_addresses.Slice(state.row_addresses, state.matched_sel, matched_count);
		state.matched_addresses.Flatten();
		for (idx_t group_idx = 0; !routed && group_idx < output_group_count; group_idx++) {
			auto &output_group = output_groups[group_idx];
			if (output_group.source == HashGroupJoinOutputSource::OWNER_PAYLOAD) {
				RowOperations::FinalizeStatesRange(state.row_state, *layout, state.matched_addresses, chunk, group_idx,
				                                   output_group.index + 1, 1);
			}
		}
		RowOperations::FinalizeStatesRange(state.row_state, *layout, state.matched_addresses, chunk, output_group_count,
		                                   aggregate_offset + 1, user_aggregate_count);
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	return SourceResultType::FINISHED;
}

string PhysicalHashGroupJoin::GetName() const {
	return "HASH_GROUP_JOIN";
}

InsertionOrderPreservingMap<string> PhysicalHashGroupJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Join Type"] = single_match                                                ? "SEMI"
	                      : unmatched_policy == HashGroupJoinUnmatchedPolicy::DISCARD ? "INNER"
	                                                                                  : "OWNER OUTER";
	switch (unmatched_policy) {
	case HashGroupJoinUnmatchedPolicy::DISCARD:
		result["Unmatched Policy"] = "DISCARD";
		break;
	case HashGroupJoinUnmatchedPolicy::EMPTY_AGGREGATE:
		result["Unmatched Policy"] = "EMPTY_AGGREGATE";
		break;
	case HashGroupJoinUnmatchedPolicy::NULL_EXTENDED_ROW:
		result["Unmatched Policy"] = "NULL_EXTENDED_ROW";
		break;
	}
	result["Build Side"] = "OWNER";
	string groups;
	for (idx_t group_idx = 0; group_idx < output_group_names.size(); group_idx++) {
		if (group_idx > 0) {
			groups += "\n";
		}
		groups += output_group_names[group_idx];
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
	if (static_mode) {
		result["Variant"] = "STATIC";
	}
	if (op_state) {
		result["Strategy"] = EnumUtil::ToString(op_state->Cast<HashGroupJoinGlobalOperatorState>().execution_mode);
	} else {
		result["Strategy"] = "AUTO";
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
