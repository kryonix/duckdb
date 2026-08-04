#include "duckdb/execution/operator/join/physical_factorized_group_join.hpp"

#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/radix_partitioning.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/column/column_data_collection_segment.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/join_hashtable.hpp"
#include "duckdb/execution/operator/join/perfect_group_join_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/table_filter_functions.hpp"
#include "duckdb/planner/operator/logical_group_join.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/storage/arena_allocator.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/temporary_memory_manager.hpp"

namespace duckdb {

static constexpr idx_t FACTORIZED_GROUP_JOIN_RADIX_BITS = 2;
static constexpr idx_t FACTORIZED_GROUP_JOIN_OUTPUT_RADIX_BITS = 4;
static constexpr idx_t FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS = 4;

static idx_t SaturatingFactorizedAdd(idx_t left, idx_t right) {
	idx_t result;
	return TryAddOperator::Operation(left, right, result) ? result : NumericLimits<idx_t>::Maximum();
}

static idx_t EstimateFactorizedCollectionSize(const ColumnDataCollection &collection) {
	idx_t result = 0;
	for (auto &segment : collection.GetSegments()) {
		for (idx_t chunk_idx = 0; chunk_idx < segment->ChunkCount(); chunk_idx++) {
			result = SaturatingFactorizedAdd(result, segment->GetChunkAllocationSize(chunk_idx));
		}
	}
	return result;
}

static const char *FactorizedBranchModeToString(FactorizedGroupJoinBranchMode mode);
static bool FactorizedBranchHasDistinct(const PhysicalFactorizedGroupJoin &op, idx_t source_idx);

static idx_t FactorizedSourceIndex(FactorizedAggregateSource source) {
	return static_cast<idx_t>(source);
}

static AggregateObject CreateFactorizedCountAggregate() {
	auto function = BoundAggregateFunction(CountStarFun::GetFunction());
	return AggregateObject(std::move(function), nullptr, 0, AlignValue(sizeof(int64_t)), AggregateType::NON_DISTINCT,
	                       PhysicalType::INT64);
}

struct FactorizedGroupJoinIdState {
	uint64_t value;
};

struct FactorizedGroupJoinIdFunction {
	static void Initialize(FactorizedGroupJoinIdState &state) {
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
				throw InternalException("Cannot combine different factorized GroupJoin identifiers");
			}
			target.value = source.value;
		}
	}

	template <class RESULT_TYPE, class STATE>
	static void Finalize(STATE &state, RESULT_TYPE &target, AggregateFinalizeData &) {
		target = state.value;
	}
};

static AggregateObject CreateFactorizedGroupJoinIdAggregate() {
	auto function = BoundAggregateFunction(
	    AggregateFunction::NullaryAggregate<FactorizedGroupJoinIdState, uint64_t, FactorizedGroupJoinIdFunction>(
	        LogicalType::UBIGINT));
	return AggregateObject(std::move(function), nullptr, 0, AlignValue(sizeof(FactorizedGroupJoinIdState)),
	                       AggregateType::NON_DISTINCT, PhysicalType::UINT64);
}

static idx_t LoadFactorizedGroupJoinId(const TupleDataLayout &layout, data_ptr_t row) {
	auto &state = *reinterpret_cast<const FactorizedGroupJoinIdState *>(row + layout.GetAggrOffset());
	if (state.value == 0) {
		throw InternalException("Factorized GroupJoin encountered an unassigned group identifier");
	}
	return NumericCast<idx_t>(state.value - 1);
}

static void StoreFactorizedGroupJoinId(const TupleDataLayout &layout, data_ptr_t row, idx_t group_id) {
	auto &state = *reinterpret_cast<FactorizedGroupJoinIdState *>(row + layout.GetAggrOffset());
	D_ASSERT(state.value == 0);
	state.value = NumericCast<uint64_t>(group_id) + 1;
}

PhysicalFactorizedGroupJoin::PhysicalFactorizedGroupJoin(
    PhysicalPlan &physical_plan, LogicalOperator &op, PhysicalOperator &driver, PhysicalOperator &left_factor,
    PhysicalOperator &right_factor, vector<idx_t> driver_keys, vector<idx_t> left_keys, vector<idx_t> right_keys,
    vector<idx_t> output_group_key_indices_p, vector<unique_ptr<Expression>> aggregates,
    vector<FactorizedAggregateSource> aggregate_sources_p, bool preserve_left_p, bool preserve_right_p,
    bool semi_left_p, bool semi_right_p, bool unique_driver_p, bool routed_p, GroupJoinExecutionMode execution_mode_p,
    Value perfect_min_p, Value perfect_max_p, idx_t perfect_range_p,
    unique_ptr<JoinFilterPushdownInfo> left_filter_pushdown, unique_ptr<JoinFilterPushdownInfo> right_filter_pushdown,
    unique_ptr<JoinFilterPushdownInfo> left_driver_filter_pushdown,
    unique_ptr<JoinFilterPushdownInfo> right_driver_filter_pushdown, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::HASH_GROUP_JOIN, op.types, estimated_cardinality),
      source_input_types(SOURCE_COUNT), key_expressions(SOURCE_COUNT), source_arguments(SOURCE_COUNT),
      source_argument_types(SOURCE_COUNT), aggregate_sources(std::move(aggregate_sources_p)),
      source_ranges(SOURCE_COUNT), output_group_key_indices(std::move(output_group_key_indices_p)),
      preserve_left(preserve_left_p), preserve_right(preserve_right_p), semi_left(semi_left_p),
      semi_right(semi_right_p), unique_driver(unique_driver_p), routed(routed_p),
      planned_execution_mode(execution_mode_p), perfect_min(std::move(perfect_min_p)),
      perfect_max(std::move(perfect_max_p)), perfect_range(perfect_range_p), physical_plan(physical_plan) {
	auto &logical_group_join = op.Cast<LogicalGroupJoin>();
	factor_filter_pushdown[0] = std::move(left_filter_pushdown);
	factor_filter_pushdown[1] = std::move(right_filter_pushdown);
	driver_filter_pushdown[0] = std::move(left_driver_filter_pushdown);
	driver_filter_pushdown[1] = std::move(right_driver_filter_pushdown);
	estimated_driver_rows = logical_group_join.estimated_owner_rows;
	estimated_left_factor_rows = logical_group_join.estimated_left_factor_rows;
	estimated_right_factor_rows = logical_group_join.estimated_right_factor_rows;
	estimated_join_rows = logical_group_join.estimated_factorized_join_rows;
	estimated_matched_drivers = logical_group_join.estimated_factorized_matched_drivers;
	estimated_left_scan_rows = logical_group_join.estimated_left_factor_scan_rows;
	estimated_right_scan_rows = logical_group_join.estimated_right_factor_scan_rows;
	estimated_build_cost = logical_group_join.factorized_build_cost;
	estimated_filter_cost = logical_group_join.factorized_filter_cost;
	estimated_probe_cost = logical_group_join.factorized_probe_cost;
	estimated_scan_cost = logical_group_join.factorized_scan_cost;
	estimated_cache_cost = logical_group_join.factorized_cache_cost;
	estimated_eager_work_cost = logical_group_join.factorized_eager_work_cost;
	estimated_routing_cost = logical_group_join.factorized_routing_cost;
	estimated_spill_cost = logical_group_join.factorized_spill_cost;
	estimated_factorized_cost = logical_group_join.factorized_cost;
	estimated_best_existing_cost = logical_group_join.factorized_best_existing_cost;
	estimated_driver_first_cost = logical_group_join.factorized_driver_first_cost;
	estimated_factors_first_cost = logical_group_join.factorized_factors_first_cost;
	estimated_cost_reliable = logical_group_join.factorized_cost_reliable;
	auto_selected = logical_group_join.factorized_auto_selected;
	driver_first = logical_group_join.factorized_driver_first;
	if (aggregates.size() != aggregate_sources.size()) {
		throw InternalException("Factorized GroupJoin aggregate source count does not match aggregate count");
	}
	if (driver_keys.size() != left_keys.size() || driver_keys.size() != right_keys.size()) {
		throw InternalException("Factorized GroupJoin key counts do not match");
	}

	array<const vector<idx_t> *, SOURCE_COUNT> source_keys {&driver_keys, &left_keys, &right_keys};
	array<const PhysicalOperator *, SOURCE_COUNT> source_operators {&driver, &left_factor, &right_factor};
	for (idx_t source_idx = 0; source_idx < SOURCE_COUNT; source_idx++) {
		source_input_types[source_idx] = source_operators[source_idx]->GetTypes();
		for (auto key_idx : *source_keys[source_idx]) {
			if (key_idx >= source_operators[source_idx]->GetTypes().size()) {
				throw InternalException("Factorized GroupJoin key index is out of bounds");
			}
			auto &type = source_operators[source_idx]->GetTypes()[key_idx];
			if (source_idx == 0) {
				key_types.push_back(type);
			} else if (type != key_types[key_expressions[source_idx].size()]) {
				throw InternalException("Factorized GroupJoin key types do not match");
			}
			key_expressions[source_idx].push_back(make_uniq<BoundReferenceExpression>(type, key_idx));
		}
	}
	if (routed) {
		group_types.reserve(output_group_key_indices.size());
		group_expressions.reserve(output_group_key_indices.size());
		for (idx_t group_idx = 0; group_idx < output_group_key_indices.size(); group_idx++) {
			auto driver_index = output_group_key_indices[group_idx];
			if (driver_index >= driver.GetTypes().size()) {
				throw InternalException("Factorized GroupJoin routed group index is out of bounds");
			}
			auto &type = driver.GetTypes()[driver_index];
			group_types.push_back(type);
			group_expressions.push_back(make_uniq<BoundReferenceExpression>(type, driver_index));
			output_group_key_indices[group_idx] = group_idx;
		}
	} else {
		group_types = key_types;
	}

	array<idx_t, SOURCE_COUNT> next_payload_index {};
	for (idx_t aggregate_idx = 0; aggregate_idx < aggregates.size(); aggregate_idx++) {
		auto &aggregate = aggregates[aggregate_idx]->Cast<BoundAggregateExpression>();
		if (aggregate.GetOrderBys() ||
		    aggregate.Function().GetOrderDependent() == AggregateOrderDependent::ORDER_DEPENDENT ||
		    aggregate.Function().GetRepeatCombine() != AggregateRepeatCombine::SUPPORTED) {
			throw InternalException("Unsupported aggregate reached factorized GroupJoin planning: %s",
			                        aggregate.ToString());
		}
		const auto source_idx = FactorizedSourceIndex(aggregate_sources[aggregate_idx]);
		D_ASSERT(source_idx < SOURCE_COUNT);
		aggregate_names.push_back(aggregate.ToString());
		if (aggregate.IsDistinct()) {
			vector<LogicalType> argument_types;
			for (auto &argument : aggregate.GetChildren()) {
				argument_types.push_back(argument->GetReturnType());
			}
			distinct_aggregates.push_back({aggregate_idx, source_idx, 0, next_payload_index[source_idx],
			                               std::move(argument_types), optional_idx()});
		}
		for (auto &argument : aggregate.GetChildren()) {
			source_arguments[source_idx].push_back(argument->Copy());
			source_argument_types[source_idx].push_back(argument->GetReturnType());
			next_payload_index[source_idx]++;
		}
	}
	for (idx_t aggregate_idx = 0; aggregate_idx < aggregates.size(); aggregate_idx++) {
		auto &aggregate = aggregates[aggregate_idx]->Cast<BoundAggregateExpression>();
		if (!aggregate.GetFilter()) {
			continue;
		}
		const auto source_idx = FactorizedSourceIndex(aggregate_sources[aggregate_idx]);
		const auto filter_idx = source_arguments[source_idx].size();
		auto filter = aggregate.GetFilter()->Copy();
		source_argument_types[source_idx].push_back(filter->GetReturnType());
		source_arguments[source_idx].push_back(std::move(filter));
		aggregate.GetFilterMutable() = make_uniq<BoundReferenceExpression>(LogicalType::BOOLEAN, filter_idx);
		for (auto &distinct : distinct_aggregates) {
			if (distinct.aggregate_index == aggregate_idx) {
				distinct.filter_index = optional_idx(filter_idx);
				break;
			}
		}
	}
	for (auto &aggregate : aggregates) {
		aggregate_expressions.push_back(std::move(aggregate));
	}

	vector<AggregateObject> target_aggregates;
	target_aggregates.reserve(aggregate_expressions.size());
	for (auto &aggregate : aggregate_expressions) {
		target_aggregates.emplace_back(aggregate->Cast<BoundAggregateExpression>());
	}
	target_layout.Initialize(std::move(target_aggregates));

	partial_indexes.resize(aggregate_expressions.size());
	partial_aggregate_objects.push_back(CreateFactorizedGroupJoinIdAggregate());
	for (idx_t source_idx = 0; source_idx < SOURCE_COUNT; source_idx++) {
		auto &range = source_ranges[source_idx];
		range.begin = partial_aggregate_objects.size();
		if (source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) || !unique_driver) {
			range.multiplicity_index = optional_idx(partial_aggregate_objects.size());
			partial_aggregate_objects.push_back(CreateFactorizedCountAggregate());
		}
		for (idx_t aggregate_idx = 0; aggregate_idx < aggregate_expressions.size(); aggregate_idx++) {
			if (FactorizedSourceIndex(aggregate_sources[aggregate_idx]) != source_idx) {
				continue;
			}
			partial_indexes[aggregate_idx] = partial_aggregate_objects.size();
			partial_aggregate_objects.emplace_back(
			    aggregate_expressions[aggregate_idx]->Cast<BoundAggregateExpression>());
		}
		range.count = partial_aggregate_objects.size() - range.begin;
	}
	for (auto &distinct : distinct_aggregates) {
		auto &range = source_ranges[distinct.source_idx];
		D_ASSERT(partial_indexes[distinct.aggregate_index] >= range.begin);
		distinct.range_index = partial_indexes[distinct.aggregate_index] - range.begin;
	}
	driver_input = &driver;
	left_input = &left_factor;
	right_input = &right_factor;
	const auto driver_rows = MaxValue<idx_t>(driver.estimated_cardinality, 1);
	array<idx_t, SOURCE_COUNT - 1> factor_rows {left_factor.estimated_cardinality, right_factor.estimated_cardinality};
	array<idx_t, SOURCE_COUNT - 1> estimated_scan_rows {estimated_left_scan_rows, estimated_right_scan_rows};
	for (idx_t factor_idx = 0; factor_idx < factor_rows.size(); factor_idx++) {
		auto estimated_fanout = factor_rows[factor_idx] / driver_rows;
		const auto matched_driver_count = MaxValue<idx_t>(estimated_matched_drivers, 1);
		const auto matched_fanout =
		    static_cast<double>(estimated_scan_rows[factor_idx]) / static_cast<double>(matched_driver_count);
		const auto scan_coverage = static_cast<double>(estimated_scan_rows[factor_idx]) /
		                           static_cast<double>(MaxValue<idx_t>(factor_rows[factor_idx], 1));
		if (!FactorizedBranchHasDistinct(*this, factor_idx + 1) && matched_fanout >= 16 && scan_coverage >= 0.75) {
			branch_modes[factor_idx] = FactorizedGroupJoinBranchMode::EAGER;
		} else if (routed && estimated_fanout >= 16) {
			branch_modes[factor_idx] = FactorizedGroupJoinBranchMode::EAGER;
		} else if (routed && estimated_fanout >= 4) {
			branch_modes[factor_idx] = FactorizedGroupJoinBranchMode::CACHED;
		} else {
			branch_modes[factor_idx] = FactorizedGroupJoinBranchMode::LAZY;
		}
	}
	streaming_driver =
	    !driver_first && unique_driver && !routed && planned_execution_mode != GroupJoinExecutionMode::EXTERNAL;
}

vector<AggregateObject> PhysicalFactorizedGroupJoin::CreateHashTableAggregates() const {
	return partial_aggregate_objects;
}

vector<AggregateObject> PhysicalFactorizedGroupJoin::CreateSourceAggregates(idx_t source_idx) const {
	D_ASSERT(source_idx < source_ranges.size());
	auto &range = source_ranges[source_idx];
	vector<AggregateObject> result;
	result.reserve(range.count);
	for (idx_t range_idx = 0; range_idx < range.count; range_idx++) {
		result.push_back(partial_aggregate_objects[range.begin + range_idx]);
	}
	return result;
}

static bool UseExternalFactorizedGroupJoin(const PhysicalFactorizedGroupJoin &op, ClientContext &context) {
	if (Settings::Get<DebugForceExternalSetting>(context)) {
		return true;
	}
	auto mode = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	if (mode != GroupJoinExecutionMode::AUTO && mode != GroupJoinExecutionMode::INDEX) {
		return mode == GroupJoinExecutionMode::EXTERNAL;
	}
	return op.planned_execution_mode == GroupJoinExecutionMode::EXTERNAL;
}

static vector<LogicalType> GetExternalFactorizedSourceTypes(const PhysicalFactorizedGroupJoin &op, idx_t source_idx) {
	auto result = op.key_types;
	if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
		result.insert(result.end(), op.group_types.begin(), op.group_types.end());
	}
	result.insert(result.end(), op.source_argument_types[source_idx].begin(),
	              op.source_argument_types[source_idx].end());
	result.push_back(LogicalType::HASH);
	return result;
}

//! In-memory factor directory that hashes each distinct key once and keeps payload rows in dense factor-id ranges.
class FactorizedJoinHashTable {
public:
	struct ProbeState {
		explicit ProbeState(const vector<LogicalType> &key_types) {
			TupleDataCollection::InitializeChunkState(join_keys, key_types);
		}

		AggregateHTLookupState aggregate;
		TupleDataChunkState join_keys;
		JoinHashTable::FactorProbeState join;
	};

	struct ExpansionState {
		idx_t reference_position = 0;
		idx_t row_position = 0;
		JoinHashTable::FactorExpansionState join;

		bool Finished(idx_t reference_count) const {
			return reference_position >= reference_count && join.Finished(reference_count);
		}
	};

	FactorizedJoinHashTable(ClientContext &context_p, const PhysicalFactorizedGroupJoin &op,
	                        vector<LogicalType> key_types_p, vector<LogicalType> payload_types_p, bool preaggregate,
	                        vector<AggregateObject> preaggregate_objects)
	    : context(context_p), key_types(std::move(key_types_p)), payload_types(std::move(payload_types_p)),
	      key_formats(key_types.size()), selected_rows(STANDARD_VECTOR_SIZE) {
		preaggregated = preaggregate;
		if (!preaggregated) {
			vector<LogicalType> layout_types(key_types);
			layout_types.insert(layout_types.end(), payload_types.begin(), payload_types.end());
			layout_types.push_back(LogicalType::HASH);
			auto layout = make_shared_ptr<TupleDataLayout>();
			layout->Initialize(layout_types, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			for (idx_t key_idx = 0; key_idx < key_types.size(); key_idx++) {
				join_conditions.emplace_back(make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx),
				                             make_uniq<BoundReferenceExpression>(key_types[key_idx], key_idx),
				                             ExpressionType::COMPARE_EQUAL);
			}
			for (idx_t payload_idx = 0; payload_idx < payload_types.size(); payload_idx++) {
				join_output_columns.push_back(key_types.size() + payload_idx);
			}
			join_table = make_uniq<JoinHashTable>(context, op, join_conditions, payload_types, JoinType::INNER,
			                                      idx_t(0), join_output_columns, nullptr);
			join_table->FinishInitWithLayout(std::move(layout));
			join_table->EnableFactorDefinitions();
			join_table->GetSinkCollection().InitializeAppendState(join_append_state);
			return;
		}
		vector<AggregateObject> aggregates;
		aggregates.push_back(CreateFactorizedGroupJoinIdAggregate());
		aggregates.push_back(CreateFactorizedCountAggregate());
		if (preaggregated) {
			for (auto &aggregate : preaggregate_objects) {
				aggregates.push_back(aggregate);
			}
		}
		directory = make_uniq<GroupedAggregateHashTable>(
		    context, BufferAllocator::Get(context), key_types, preaggregated ? payload_types : vector<LogicalType> {},
		    std::move(aggregates), GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
		    TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
		selected_keys.InitializeEmpty(key_types);
		selected_payload.InitializeEmpty(payload_types);
		generation = next_generation.fetch_add(1, std::memory_order_relaxed);
		if (generation == 0 || generation > JoinFactorRef::MAX_GENERATION) {
			throw OutOfRangeException("Factorized GroupJoin factor generation overflow");
		}
	}

	void Build(DataChunk &keys, DataChunk &payload) {
		if (!preaggregated) {
			join_table->Build(join_append_state, keys, payload);
			row_count += keys.size();
			return;
		}
		const auto non_null_count = SelectNonNullKeys(keys);
		if (non_null_count == 0) {
			return;
		}
		if (non_null_count == keys.size()) {
			selected_keys.Reference(keys);
			if (payload.ColumnCount() == 0) {
				selected_payload.SetChildCardinality(non_null_count);
			} else {
				selected_payload.Reference(payload);
			}
		} else {
			selected_keys.Slice(keys, selected_rows, non_null_count);
			if (payload.ColumnCount() == 0) {
				selected_payload.SetChildCardinality(non_null_count);
			} else {
				selected_payload.Slice(payload, selected_rows, non_null_count);
			}
		}
		directory->AddChunk(selected_keys, selected_payload, AggregateType::NON_DISTINCT);
		row_count += non_null_count;
	}

	void Merge(FactorizedJoinHashTable &local) {
		local.FinalizeAppend();
		if (local.row_count == 0) {
			return;
		}
		if (!preaggregated) {
			join_table->Merge(*local.join_table);
			row_count += local.row_count;
			return;
		}
		directory->Combine(*local.directory);
		row_count += local.row_count;
	}

	void Finalize() {
		FinalizeAppend();
		if (!preaggregated) {
			join_table->Unpartition();
			row_count = join_table->Count();
			if (row_count != 0) {
				join_table->AllocatePointerTable();
				join_table->InitializePointerTable(0, join_table->capacity);
				join_table->FinalizeFactorized(0, join_table->GetDataCollection().ChunkCount(), false);
			}
			join_table->FinishFactorDefinitions();
			join_table->finalized = true;
			finalized = true;
			return;
		}
		factor_count = 0;
		AggregateHTScanState id_scan;
		directory->InitializeScan(id_scan);
		DataChunk id_groups;
		id_groups.Initialize(Allocator::Get(context), key_types);
		Vector id_addresses(LogicalType::POINTER);
		while (directory->ScanGroupsAndAddresses(id_scan, id_groups, id_addresses)) {
			id_addresses.Flatten();
			auto address_data = FlatVector::GetData<data_ptr_t>(id_addresses);
			for (idx_t row_idx = 0; row_idx < id_groups.size(); row_idx++) {
				StoreFactorizedGroupJoinId(directory->GetLayout(), address_data[row_idx], factor_count++);
			}
		}
		factor_offsets.assign(factor_count + 1, 0);
		AggregateHTScanState scan;
		directory->InitializeScan(scan);
		DataChunk groups;
		groups.Initialize(Allocator::Get(context), key_types);
		Vector directory_addresses(LogicalType::POINTER);
		while (directory->ScanGroupsAndAddresses(scan, groups, directory_addresses)) {
			directory_addresses.Flatten();
			auto address_data = FlatVector::GetData<data_ptr_t>(directory_addresses);
			for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
				auto factor_id = LoadFactorizedGroupJoinId(directory->GetLayout(), address_data[row_idx]);
				factor_offsets[factor_id + 1] = LoadDirectoryCount(address_data[row_idx]);
			}
		}
		for (idx_t factor_idx = 1; factor_idx < factor_offsets.size(); factor_idx++) {
			factor_offsets[factor_idx] += factor_offsets[factor_idx - 1];
		}
		factor_addresses.resize(factor_count, nullptr);
		directory->InitializeScan(scan);
		while (directory->ScanGroupsAndAddresses(scan, groups, directory_addresses)) {
			directory_addresses.Flatten();
			auto address_data = FlatVector::GetData<data_ptr_t>(directory_addresses);
			for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
				auto factor_id = LoadFactorizedGroupJoinId(directory->GetLayout(), address_data[row_idx]);
				factor_addresses[factor_id] = address_data[row_idx];
			}
		}
		finalized = true;
	}

	idx_t ProbeFactorRefs(DataChunk &keys, ProbeState &lookup, Vector &factor_refs, SelectionVector &match_sel) const {
		if (!finalized) {
			throw InternalException("Factorized GroupJoin factor directory was probed before finalization");
		}
		if (!preaggregated) {
			return join_table->ProbeFactorRefs(keys, lookup.join_keys, lookup.join, factor_refs, match_sel);
		}
		factor_refs.SetVectorType(VectorType::FLAT_VECTOR);
		auto refs = FlatVector::GetDataMutable<uint64_t>(factor_refs);
		memset(refs, 0, sizeof(uint64_t) * keys.size());
		FlatVector::SetSize(factor_refs, keys.size());
		if (keys.size() == 0 || factor_count == 0) {
			return 0;
		}
		auto match_count = directory->LookupGroups(keys, lookup.aggregate, match_sel);
		auto address_data = FlatVector::GetData<data_ptr_t>(lookup.aggregate.addresses);
		for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
			auto input_idx = match_sel.get_index_unsafe(match_idx);
			auto factor_id = LoadFactorizedGroupJoinId(directory->GetLayout(), address_data[input_idx]);
			refs[input_idx] = GetFactorRef(factor_id).Value();
		}
		return match_count;
	}

	idx_t ExpandFactorRefs(Vector &factor_refs, optional_ptr<const SelectionVector> ref_sel, idx_t reference_count,
	                       ExpansionState &state, Vector &row_addresses, SelectionVector &source_sel,
	                       idx_t output_capacity = STANDARD_VECTOR_SIZE) const {
		if (!preaggregated) {
			state.reference_position = reference_count;
			return join_table->ExpandFactorRefs(factor_refs, ref_sel, reference_count, state.join, row_addresses,
			                                    source_sel, output_capacity);
		}
		state.join.reference_position = reference_count;
		auto refs = factor_refs.Values<uint64_t>();
		auto output = FlatVector::GetDataMutable<data_ptr_t>(row_addresses);
		idx_t output_count = 0;
		while (output_count < output_capacity && state.reference_position < reference_count) {
			auto source_idx = ref_sel ? ref_sel->get_index(state.reference_position) : state.reference_position;
			auto factor_id = GetFactorId(JoinFactorRef(refs[source_idx].GetValueUnsafe()));
			auto row_begin = factor_offsets[factor_id];
			auto row_end = factor_offsets[factor_id + 1];
			while (output_count < output_capacity && row_begin + state.row_position < row_end) {
				output[output_count] = factor_rows[row_begin + state.row_position++];
				source_sel.set_index(output_count++, source_idx);
			}
			if (row_begin + state.row_position == row_end) {
				state.reference_position++;
				state.row_position = 0;
			}
		}
		FlatVector::SetSize(row_addresses, output_count);
		return output_count;
	}

	void GatherPayload(Vector &row_addresses, idx_t count, DataChunk &result,
	                   vector<unique_ptr<Vector>> &cached_cast_vectors) const {
		if (payload_types.empty()) {
			result.SetChildCardinality(count);
			return;
		}
		D_ASSERT(!preaggregated);
		join_table->GatherRHS(row_addresses, *FlatVector::IncrementalSelectionVector(), count, result, 0);
	}

	idx_t ScanFactorRefs(idx_t &slot, Vector &factor_refs) const {
		if (!preaggregated) {
			return join_table->ScanFactorRefs(slot, factor_refs);
		}
		auto refs = FlatVector::GetDataMutable<uint64_t>(factor_refs);
		idx_t count = 0;
		while (slot < factor_count && count < STANDARD_VECTOR_SIZE) {
			refs[count++] = GetFactorRef(slot++).Value();
		}
		FlatVector::SetSize(factor_refs, count);
		return count;
	}

	idx_t FactorCount() const {
		return preaggregated ? factor_count : join_table->FactorCount();
	}
	idx_t Count() const {
		return preaggregated || !finalized ? row_count : join_table->Count();
	}
	bool IsPreaggregated() const {
		return preaggregated;
	}
	TupleDataLayout &GetLayout() {
		return *directory->GetLayoutPtr();
	}
	void GetPreaggregateAddresses(Vector &factor_refs, idx_t count, Vector &result) const {
		if (!preaggregated) {
			throw InternalException("Factorized GroupJoin requested states from a non-preaggregated factor");
		}
		auto refs = factor_refs.Values<uint64_t>();
		auto addresses = FlatVector::GetDataMutable<data_ptr_t>(result);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			addresses[row_idx] = factor_addresses[GetFactorId(JoinFactorRef(refs[row_idx].GetValueUnsafe()))];
		}
		FlatVector::SetSize(result, count);
	}
	idx_t GetFactorCount(JoinFactorRef ref) const {
		if (!preaggregated) {
			return join_table->GetFactorCount(ref);
		}
		auto factor_id = GetFactorId(ref);
		return factor_offsets[factor_id + 1] - factor_offsets[factor_id];
	}
	idx_t MaximumFactorLength() const {
		if (!preaggregated) {
			return join_table->MaximumFactorLength();
		}
		idx_t result = 0;
		for (idx_t factor_idx = 0; factor_idx < factor_count; factor_idx++) {
			result = MaxValue(result, factor_offsets[factor_idx + 1] - factor_offsets[factor_idx]);
		}
		return result;
	}
	idx_t FactorDefinitionSizeInBytes() const {
		return preaggregated ? factor_offsets.size() * sizeof(idx_t) + factor_rows.size() * sizeof(data_ptr_t)
		                     : join_table->FactorDefinitionSizeInBytes();
	}

private:
	idx_t SelectNonNullKeys(DataChunk &keys) {
		for (idx_t key_idx = 0; key_idx < keys.ColumnCount(); key_idx++) {
			keys.data[key_idx].ToUnifiedFormat(key_formats[key_idx]);
		}
		idx_t count = 0;
		for (idx_t row_idx = 0; row_idx < keys.size(); row_idx++) {
			bool valid = true;
			for (auto &format : key_formats) {
				valid = valid && format.validity.RowIsValid(format.sel->get_index(row_idx));
			}
			if (valid) {
				selected_rows.set_index(count++, row_idx);
			}
		}
		return count;
	}

	void FinalizeAppend() {
		if (!preaggregated && !append_finalized) {
			join_table->GetSinkCollection().FlushAppendState(join_append_state);
			append_finalized = true;
		}
	}

	idx_t LoadDirectoryCount(data_ptr_t row) const {
		auto &layout = directory->GetLayout();
		auto offset_idx = layout.ColumnCount() + 1;
		auto value = Load<int64_t>(row + layout.GetOffsets()[offset_idx]);
		if (value < 0) {
			throw InternalException("Factorized GroupJoin encountered a negative factor count");
		}
		return NumericCast<idx_t>(value);
	}

	JoinFactorRef GetFactorRef(idx_t factor_id) const {
		if (factor_id > JoinFactorRef::SLOT_MASK) {
			throw OutOfRangeException("Factorized GroupJoin factor id exceeds the handle range");
		}
		return JoinFactorRef(generation, factor_id);
	}

	idx_t GetFactorId(JoinFactorRef ref) const {
		if (!ref.IsValid() || ref.Generation() != generation || ref.Slot() >= factor_count) {
			throw InternalException("Stale or invalid factorized GroupJoin factor reference");
		}
		return ref.Slot();
	}

private:
	ClientContext &context;
	vector<LogicalType> key_types;
	vector<LogicalType> payload_types;
	vector<JoinCondition> join_conditions;
	vector<idx_t> join_output_columns;
	unique_ptr<JoinHashTable> join_table;
	PartitionedTupleDataAppendState join_append_state;
	unique_ptr<GroupedAggregateHashTable> directory;
	vector<UnifiedVectorFormat> key_formats;
	DataChunk selected_keys;
	DataChunk selected_payload;
	SelectionVector selected_rows;
	vector<idx_t> factor_offsets;
	vector<data_ptr_t> factor_rows;
	vector<data_ptr_t> factor_addresses;
	idx_t factor_count = 0;
	idx_t row_count = 0;
	idx_t generation;
	bool preaggregated = false;
	bool append_finalized = false;
	bool finalized = false;
	static atomic<idx_t> next_generation;
};

atomic<idx_t> FactorizedJoinHashTable::next_generation {1};

static unique_ptr<FactorizedJoinHashTable> CreateFactorizedJoinHashTable(const PhysicalFactorizedGroupJoin &op,
                                                                         ClientContext &context, idx_t source_idx) {
	D_ASSERT(source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER));
	vector<AggregateObject> preaggregate_objects;
	const auto preaggregate = op.branch_modes[source_idx - 1] == FactorizedGroupJoinBranchMode::EAGER &&
	                          !FactorizedBranchHasDistinct(op, source_idx);
	if (preaggregate) {
		preaggregate_objects = op.CreateSourceAggregates(source_idx);
		D_ASSERT(op.source_ranges[source_idx].multiplicity_index.IsValid() && !preaggregate_objects.empty());
		preaggregate_objects.erase(preaggregate_objects.begin());
	}
	return make_uniq<FactorizedJoinHashTable>(context, op, op.key_types, op.source_argument_types[source_idx],
	                                          preaggregate, std::move(preaggregate_objects));
}

//! Branch-local aggregate states addressed directly by the dense driver group identifier.
class DenseFactorizedAggregateTable {
public:
	class UpdateState {
	public:
		UpdateState(ClientContext &context, TupleDataLayout &layout, const vector<LogicalType> &payload_types)
		    : aggregate_allocator(make_shared_ptr<ArenaAllocator>(Allocator::Get(context))),
		      row_state(*aggregate_allocator), addresses(LogicalType::POINTER), update_addresses(LogicalType::POINTER),
		      initialize_addresses(LogicalType::POINTER) {
			filter_set.Initialize(context, layout.GetAggregates(), payload_types);
			idx_t clustered_count = 0;
			for (auto &aggregate : layout.GetAggregates()) {
				has_filters = has_filters || aggregate.filter;
				clustered_count += !aggregate.filter && aggregate.function.GetStateClusterUpdateCallback() ? 1 : 0;
			}
			clustered_state.n_clustered = clustered_count;
			if (clustered_count > 1) {
				clustered_state.Initialize();
			}
		}

		shared_ptr<ArenaAllocator> aggregate_allocator;
		RowOperationsState row_state;
		AggregateFilterDataSet filter_set;
		ClusteredAggrState clustered_state;
		Vector addresses;
		Vector update_addresses;
		Vector initialize_addresses;
		bool has_filters = false;
	};

	DenseFactorizedAggregateTable(ClientContext &context, idx_t group_count_p, vector<LogicalType> payload_types,
	                              vector<AggregateObject> aggregates)
	    : group_count(group_count_p), aggregate_allocator(Allocator::Get(context)), row_state(aggregate_allocator),
	      addresses(LogicalType::POINTER), payload_types(std::move(payload_types)) {
		layout.Initialize(std::move(aggregates));
		row_width = layout.GetRowWidth();
		idx_t allocation_size;
		if (!TryMultiplyOperator::Operation(group_count, row_width, allocation_size)) {
			throw OutOfMemoryException("Dense factorized aggregate state size overflow");
		}
		data = make_unsafe_uniq_array_uninitialized<data_t>(allocation_size);
		is_set = make_unsafe_uniq_array_uninitialized<bool>(group_count);
		memset(is_set.get(), 0, group_count * sizeof(bool));
	}

	~DenseFactorizedAggregateTable() {
		DestroyStates();
	}

	unique_ptr<UpdateState> CreateUpdateState(ClientContext &context) {
		auto result = make_uniq<UpdateState>(context, layout, payload_types);
		retained_allocators.push_back(result->aggregate_allocator);
		return result;
	}

	void AddChunk(UpdateState &state, Vector &group_ids, DataChunk &payload, const unsafe_vector<idx_t> &filter) {
		if (group_ids.size() != payload.size()) {
			throw InternalException("Dense factorized aggregate group and payload counts differ");
		}
		group_ids.Flatten();
		auto ids = FlatVector::GetData<uint64_t>(group_ids);
		auto address_data = FlatVector::GetDataMutable<data_ptr_t>(state.addresses);
		auto initialize_data = FlatVector::GetDataMutable<data_ptr_t>(state.initialize_addresses);
		idx_t initialize_count = 0;
		for (idx_t row_idx = 0; row_idx < group_ids.size(); row_idx++) {
			if (ids[row_idx] >= group_count) {
				throw InternalException("Dense factorized aggregate identifier exceeds driver group count");
			}
			auto address = data.get() + ids[row_idx] * row_width;
			address_data[row_idx] = address;
			if (!is_set[ids[row_idx]]) {
				is_set[ids[row_idx]] = true;
				initialize_data[initialize_count++] = address;
			}
		}
		FlatVector::SetSize(state.initialize_addresses, initialize_count);
		RowOperations::InitializeStates(layout, state.initialize_addresses, *FlatVector::IncrementalSelectionVector(),
		                                initialize_count);
		FlatVector::SetSize(state.addresses, group_ids.size());
		VectorOperations::Copy(state.addresses, state.update_addresses, state.addresses.size(), 0, 0);
		FlatVector::SetSize(state.update_addresses, state.addresses.size());
		VectorOperations::AddInPlace(state.update_addresses, NumericCast<int64_t>(layout.GetAggrOffset()));

		ClusteredAggr clustered;
		if (!state.has_filters && state.clustered_state.arena &&
		    state.clustered_state.TryBuild(clustered, ids, group_ids.size())) {
			clustered.InitializeStatesFromAddresses(address_data, layout.GetAggrOffset());
			RowOperations::UpdateStatesClusteredRange(state.row_state, layout.GetAggregates(), 0,
			                                          layout.GetAggregates().size(), &state.filter_set, &filter,
			                                          state.update_addresses, payload, clustered, false);
			return;
		}

		idx_t payload_idx = 0;
		idx_t filter_idx = 0;
		for (idx_t aggregate_idx = 0; aggregate_idx < layout.GetAggregates().size(); aggregate_idx++) {
			auto &aggregate = layout.GetAggregates()[aggregate_idx];
			if (filter_idx < filter.size() && filter[filter_idx] == aggregate_idx) {
				if (aggregate.aggr_type != AggregateType::DISTINCT && aggregate.filter) {
					RowOperations::UpdateFilteredStates(state.row_state, state.filter_set.GetFilterData(aggregate_idx),
					                                    aggregate, state.update_addresses, payload, payload_idx);
				} else {
					RowOperations::UpdateStates(state.row_state, aggregate, state.update_addresses, payload,
					                            payload_idx);
				}
				filter_idx++;
			}
			payload_idx += aggregate.child_count;
			VectorOperations::AddInPlace(state.update_addresses, NumericCast<int64_t>(aggregate.payload_size));
		}
	}

	void CombineInto(GroupedAggregateHashTable &target, const vector<data_ptr_t> &target_addresses,
	                 idx_t target_begin) {
		if (target_addresses.size() != group_count) {
			throw InternalException("Dense factorized aggregate target count differs from driver group count");
		}
		AggregateHTUpdateState update_state(target);
		Vector source_addresses(LogicalType::POINTER);
		Vector destination_addresses(LogicalType::POINTER);
		auto sources = FlatVector::GetDataMutable<data_ptr_t>(source_addresses);
		auto destinations = FlatVector::GetDataMutable<data_ptr_t>(destination_addresses);
		idx_t count = 0;
		for (idx_t group_id = 0; group_id < group_count; group_id++) {
			if (!is_set[group_id]) {
				continue;
			}
			sources[count] = data.get() + group_id * row_width;
			destinations[count] = target_addresses[group_id];
			if (++count < STANDARD_VECTOR_SIZE) {
				continue;
			}
			FlatVector::SetSize(source_addresses, count);
			FlatVector::SetSize(destination_addresses, count);
			RowOperations::CombineStatesRange(
			    update_state.row_state, layout, source_addresses, 0, *target.GetLayoutPtr(), destination_addresses,
			    target_begin, layout.GetAggregates().size(), nullptr, AggregateCombineType::PRESERVE_INPUT);
			count = 0;
		}
		FlatVector::SetSize(source_addresses, count);
		FlatVector::SetSize(destination_addresses, count);
		RowOperations::CombineStatesRange(update_state.row_state, layout, source_addresses, 0, *target.GetLayoutPtr(),
		                                  destination_addresses, target_begin, layout.GetAggregates().size(), nullptr,
		                                  AggregateCombineType::PRESERVE_INPUT);
	}

	idx_t SizeInBytes() const {
		return group_count * row_width + group_count * sizeof(bool);
	}

private:
	void DestroyStates() {
		auto address_data = FlatVector::GetDataMutable<data_ptr_t>(addresses);
		idx_t count = 0;
		for (idx_t group_id = 0; group_id < group_count; group_id++) {
			if (!is_set[group_id]) {
				continue;
			}
			address_data[count++] = data.get() + group_id * row_width;
			if (count < STANDARD_VECTOR_SIZE) {
				continue;
			}
			FlatVector::SetSize(addresses, count);
			RowOperations::DestroyStates(row_state, layout, addresses);
			count = 0;
		}
		FlatVector::SetSize(addresses, count);
		RowOperations::DestroyStates(row_state, layout, addresses);
		data.reset();
	}

	TupleDataLayout layout;
	idx_t group_count;
	idx_t row_width;
	ArenaAllocator aggregate_allocator;
	RowOperationsState row_state;
	Vector addresses;
	vector<LogicalType> payload_types;
	vector<shared_ptr<ArenaAllocator>> retained_allocators;
	unsafe_unique_array<data_t> data;
	unsafe_unique_array<bool> is_set;
};

static bool UseDenseFactorizedAggregateTables(const PhysicalFactorizedGroupJoin &op, ClientContext &context,
                                              idx_t group_count) {
	if (group_count == 0) {
		return false;
	}
	idx_t bytes_per_task = 0;
	for (idx_t source_idx = FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR);
	     source_idx < PhysicalFactorizedGroupJoin::SOURCE_COUNT; source_idx++) {
		TupleDataLayout layout;
		layout.Initialize(op.CreateSourceAggregates(source_idx));
		idx_t source_bytes;
		if (!TryMultiplyOperator::Operation(group_count, layout.GetRowWidth() + sizeof(bool), source_bytes) ||
		    !TryAddOperator::Operation(bytes_per_task, source_bytes, bytes_per_task)) {
			return false;
		}
	}
	idx_t total_bytes;
	auto thread_count = MaxValue<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads(), 1);
	if (!TryMultiplyOperator::Operation(bytes_per_task, thread_count, total_bytes)) {
		return false;
	}
	return total_bytes <= BufferManager::GetBufferManager(context).GetMaxMemory() / 8;
}

class FactorizedGroupJoinGlobalSinkState : public GlobalSinkState {
public:
	FactorizedGroupJoinGlobalSinkState(const PhysicalFactorizedGroupJoin &op, ClientContext &context,
	                                   idx_t source_idx_p)
	    : source_idx(source_idx_p), external(UseExternalFactorizedGroupJoin(op, context)),
	      direct_factor_updates(!external && op.driver_first) {
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
			runtime_branch_modes = op.branch_modes;
			branch_mode_selected = {true, true};
		}
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && direct_factor_updates) {
			for (idx_t factor_idx = 0; factor_idx < driver_filter_states.size(); factor_idx++) {
				if (!op.driver_filter_pushdown[factor_idx]) {
					continue;
				}
				driver_filter_states[factor_idx] = op.driver_filter_pushdown[factor_idx]->GetGlobalState(context, op);
			}
			if (driver_filter_states[0] || driver_filter_states[1]) {
				driver_filter_keys = make_uniq<ColumnDataCollection>(context, op.key_types);
			}
		}
		if (source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && !direct_factor_updates &&
		    op.factor_filter_pushdown[source_idx - 1]) {
			global_filter_state = op.factor_filter_pushdown[source_idx - 1]->GetGlobalState(context, op);
		}
		if (external) {
			auto partition_types = GetExternalFactorizedSourceTypes(op, source_idx);
			const auto hash_col_idx = partition_types.size() - 1;
			partitions = make_uniq<RadixPartitionedColumnData>(context, std::move(partition_types),
			                                                   FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS, hash_col_idx);
			return;
		}
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
			parallel_driver =
			    op.unique_driver && !op.routed && TaskScheduler::GetScheduler(context).NumberOfThreads() > 1;
			target = make_uniq<GroupedAggregateHashTable>(
			    context, BufferAllocator::Get(context), op.group_types, vector<LogicalType> {},
			    op.CreateHashTableAggregates(), GroupedAggregateHashTable::InitialCapacity(),
			    parallel_driver ? FACTORIZED_GROUP_JOIN_RADIX_BITS : idx_t(0),
			    op.unique_driver ? TupleDataValidityType::CANNOT_HAVE_NULL_VALUES
			                     : TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			if (op.unique_driver && !parallel_driver) {
				target->SkipLookups();
			}
			if (op.routed) {
				routing_table = make_uniq<GroupedAggregateHashTable>(
				    context, BufferAllocator::Get(context), op.key_types, vector<LogicalType> {},
				    vector<AggregateObject> {CreateFactorizedGroupJoinIdAggregate()},
				    GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
				    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			}
			if (!parallel_driver && !op.routed && op.key_types.size() == 1 && !op.perfect_min.IsNull() &&
			    !op.perfect_max.IsNull() &&
			    PerfectGroupJoinDirectoryFits(op.perfect_range,
			                                  BufferManager::GetBufferManager(context).GetMaxMemory())) {
				perfect_executor = make_uniq<PerfectGroupJoinExecutor>(op.key_types[0], op.perfect_min, op.perfect_max,
				                                                       op.perfect_range);
			}
		} else if (!direct_factor_updates) {
			factor_hash_table = CreateFactorizedJoinHashTable(op, context, source_idx);
		}
	}

	idx_t source_idx;
	bool external;
	bool direct_factor_updates;
	bool parallel_driver = false;
	unique_ptr<GroupedAggregateHashTable> target;
	unique_ptr<GroupedAggregateHashTable> routing_table;
	unique_ptr<PerfectGroupJoinExecutor> perfect_executor;
	unique_ptr<FactorizedJoinHashTable> factor_hash_table;
	vector<data_ptr_t> group_addresses;
	vector<vector<data_ptr_t>> routes;
	vector<idx_t> route_offsets;
	vector<data_ptr_t> route_addresses;
	array<unique_ptr<atomic<idx_t>[]>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> owners;
	vector<unique_ptr<GroupedAggregateHashTable>> local_tables;
	vector<unique_ptr<DenseFactorizedAggregateTable>> local_dense_tables;
	vector<vector<unique_ptr<GroupedAggregateHashTable>>> local_distinct_tables;
	unique_ptr<RadixPartitionedColumnData> partitions;
	unique_ptr<JoinFilterGlobalState> global_filter_state;
	array<unique_ptr<JoinFilterGlobalState>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> driver_filter_states;
	array<vector<unique_ptr<BloomFilter>>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> driver_bloom_filters;
	array<vector<unique_ptr<PrefixRangeFilter>>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1>
	    driver_prefix_range_filters;
	unique_ptr<ColumnDataCollection> driver_filter_keys;
	mutex lock;
	atomic<idx_t> next_token {1};
	atomic<idx_t> next_group_id {0};
	atomic<idx_t> input_rows {0};
	atomic<idx_t> matched_rows {0};
	array<FactorizedGroupJoinBranchMode, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> runtime_branch_modes {
	    FactorizedGroupJoinBranchMode::LAZY, FactorizedGroupJoinBranchMode::LAZY};
	array<idx_t, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> sampled_driver_rows {};
	array<idx_t, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> sampled_matches {};
	array<idx_t, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> sampled_factor_rows {};
	array<idx_t, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> sampled_distinct_factors {};
	array<bool, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> branch_mode_selected {};
	bool finalized = false;
};

class FactorizedGroupJoinGlobalOperatorState : public GlobalOperatorState {
public:
	FactorizedGroupJoinGlobalOperatorState(const PhysicalFactorizedGroupJoin &op, ClientContext &context)
	    : driver_state(op, context, FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
		if (!op.streaming_driver || driver_state.external || driver_state.direct_factor_updates) {
			throw InternalException("Invalid streaming factorized GroupJoin state");
		}
	}

	FactorizedGroupJoinGlobalSinkState driver_state;
};

class PhysicalFactorizedGroupJoinSink : public PhysicalOperator {
public:
	PhysicalFactorizedGroupJoinSink(PhysicalPlan &physical_plan, PhysicalFactorizedGroupJoin &op_p,
	                                PhysicalOperator &input, idx_t source_idx_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::HASH_GROUP_JOIN, {}, input.estimated_cardinality),
	      op(op_p), source_idx(source_idx_p) {
		children.push_back(input);
	}

	PhysicalFactorizedGroupJoin &op;
	idx_t source_idx;

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<FactorizedGroupJoinGlobalSinkState>(op, context, source_idx);
	}
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool ParallelSink() const override {
		return source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) ||
		       (op.unique_driver && !op.routed);
	}
	string GetName() const override {
		return "FACTORIZED_GROUP_JOIN_INPUT";
	}
};

static FactorizedGroupJoinGlobalSinkState &GetFactorizedDriverState(const PhysicalFactorizedGroupJoin &op) {
	if (op.streaming_driver) {
		if (!op.op_state) {
			throw InternalException("Factorized GroupJoin operator state is not initialized");
		}
		return op.op_state->Cast<FactorizedGroupJoinGlobalOperatorState>().driver_state;
	}
	if (!op.driver_sink || !op.driver_sink->sink_state) {
		throw InternalException("Factorized GroupJoin driver sink is not initialized");
	}
	return op.driver_sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
}

static FactorizedGroupJoinGlobalSinkState &GetFactorizedFactorState(const PhysicalFactorizedGroupJoin &op,
                                                                    idx_t source_idx) {
	D_ASSERT(source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER));
	auto sink =
	    source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR) ? op.left_sink : op.right_sink;
	if (!sink || !sink->sink_state) {
		throw InternalException("Factorized GroupJoin factor sink is not initialized");
	}
	return sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
}

static idx_t LoadFactorizedCount(const TupleDataLayout &layout, data_ptr_t row, idx_t aggregate_idx);
static int64_t CheckedFactorizedMultiplicity(idx_t value);

class FactorizedGroupJoinLocalSinkState : public LocalSinkState {
public:
	FactorizedGroupJoinLocalSinkState(ExecutionContext &context, const PhysicalFactorizedGroupJoinSink &sink)
	    : FactorizedGroupJoinLocalSinkState(
	          context, sink.op, sink.source_idx, sink.sink_state->Cast<FactorizedGroupJoinGlobalSinkState>(),
	          sink.source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) ||
	                  (sink.op.driver_first && sink.op.planned_execution_mode != GroupJoinExecutionMode::EXTERNAL)
	              ? &GetFactorizedDriverState(sink.op)
	              : nullptr,
	          false) {
	}

	FactorizedGroupJoinLocalSinkState(ExecutionContext &context, const PhysicalFactorizedGroupJoin &op,
	                                  idx_t source_idx_p, FactorizedGroupJoinGlobalSinkState &factor_state,
	                                  optional_ptr<FactorizedGroupJoinGlobalSinkState> driver_state, bool streaming)
	    : source_idx(source_idx_p), addresses(LogicalType::POINTER), matched_addresses(LogicalType::POINTER),
	      owner_addresses(LogicalType::POINTER), local_addresses(LogicalType::POINTER),
	      new_groups(STANDARD_VECTOR_SIZE), non_null_sel(STANDARD_VECTOR_SIZE), found_groups(STANDARD_VECTOR_SIZE),
	      owner_sel(STANDARD_VECTOR_SIZE), local_sel(STANDARD_VECTOR_SIZE), left_factor_probe_state(op.key_types),
	      right_factor_probe_state(op.key_types), key_formats(op.key_types.size()) {
		key_executor = make_uniq<ExpressionExecutor>(context.client);
		for (auto &expression : op.key_expressions[source_idx]) {
			key_executor->AddExpression(*expression);
		}
		keys.Initialize(Allocator::Get(context.client), op.key_types);
		lookup_keys.InitializeEmpty(op.key_types);
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && op.routed) {
			group_executor = make_uniq<ExpressionExecutor>(context.client);
			for (auto &expression : op.group_expressions) {
				group_executor->AddExpression(*expression);
			}
			groups.Initialize(Allocator::Get(context.client), op.group_types);
		}

		argument_executor = make_uniq<ExpressionExecutor>(context.client);
		for (auto &expression : op.source_arguments[source_idx]) {
			argument_executor->AddExpression(*expression);
		}
		if (!op.source_argument_types[source_idx].empty()) {
			arguments.Initialize(Allocator::Get(context.client), op.source_argument_types[source_idx]);
		}
		lookup_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) &&
		    factor_state.direct_factor_updates) {
			for (idx_t factor_idx = 0; factor_idx < driver_filter_states.size(); factor_idx++) {
				if (!factor_state.driver_filter_states[factor_idx]) {
					continue;
				}
				driver_filter_states[factor_idx] = op.driver_filter_pushdown[factor_idx]->GetLocalState(
				    *factor_state.driver_filter_states[factor_idx]);
			}
			if (factor_state.driver_filter_keys) {
				driver_filter_keys = make_uniq<ColumnDataCollection>(context.client, op.key_types);
			}
		}
		if (source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && factor_state.global_filter_state) {
			D_ASSERT(factor_state.global_filter_state);
			local_filter_state =
			    op.factor_filter_pushdown[source_idx - 1]->GetLocalState(*factor_state.global_filter_state);
		}
		if (factor_state.external) {
			external = true;
			external_chunk.InitializeEmpty(GetExternalFactorizedSourceTypes(op, source_idx));
			external_partitions = factor_state.partitions->CreateShared();
			external_partitions->InitializeAppendState(external_append_state);
			return;
		}
		if (source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) &&
		    !factor_state.direct_factor_updates) {
			factor_lookup_build = true;
			factor_hash_table = CreateFactorizedJoinHashTable(op, context.client, source_idx);
			return;
		}
		auto source_aggregates = op.CreateSourceAggregates(source_idx);
		source_filter_set.Initialize(context.client, source_aggregates, op.source_argument_types[source_idx]);
		for (idx_t range_idx = 0; range_idx < op.source_ranges[source_idx].count; range_idx++) {
			if (source_aggregates[range_idx].aggr_type != AggregateType::DISTINCT) {
				aggregate_filter.push_back(range_idx);
			}
		}
		group_ids.Initialize(Allocator::Get(context.client), {LogicalType::UBIGINT});
		distinct_tables.resize(op.distinct_aggregates.size());
		distinct_groups.resize(op.distinct_aggregates.size());
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			auto &distinct = op.distinct_aggregates[distinct_idx];
			if (distinct.source_idx != source_idx &&
			    source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
				continue;
			}
			vector<LogicalType> distinct_types {LogicalType::UBIGINT};
			distinct_types.insert(distinct_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			distinct_tables[distinct_idx] = make_uniq<GroupedAggregateHashTable>(
			    context.client, BufferAllocator::Get(context.client), distinct_types, vector<LogicalType> {},
			    vector<AggregateObject> {}, GroupedAggregateHashTable::InitialCapacity(),
			    FACTORIZED_GROUP_JOIN_RADIX_BITS, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			distinct_groups[distinct_idx] = make_uniq<DataChunk>();
			distinct_groups[distinct_idx]->Initialize(Allocator::Get(context.client), distinct_types);
		}

		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
			D_ASSERT(driver_state);
			if (driver_state->parallel_driver || streaming) {
				local_driver_target = make_uniq<GroupedAggregateHashTable>(
				    context.client, BufferAllocator::Get(context.client), op.group_types, vector<LogicalType> {},
				    op.CreateHashTableAggregates(), GroupedAggregateHashTable::InitialCapacity(),
				    streaming ? idx_t(0) : FACTORIZED_GROUP_JOIN_RADIX_BITS,
				    TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
				local_driver_target->SkipLookups();
				update_state = make_uniq<AggregateHTUpdateState>(*local_driver_target);
			} else {
				update_state = make_uniq<AggregateHTUpdateState>(*driver_state->target);
			}
			if (!op.source_argument_types[FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)].empty()) {
				left_factor_payload.Initialize(
				    Allocator::Get(context.client),
				    op.source_argument_types[FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)]);
			}
			if (!op.source_argument_types[FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR)].empty()) {
				right_factor_payload.Initialize(
				    Allocator::Get(context.client),
				    op.source_argument_types[FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR)]);
			}
			expanded_group_id_chunk.InitializeEmpty({LogicalType::UBIGINT});
			for (idx_t factor_idx = 0; factor_idx < branch_filter_sets.size(); factor_idx++) {
				auto branch_source_idx = factor_idx + 1;
				auto branch_aggregates = op.CreateSourceAggregates(branch_source_idx);
				branch_filter_sets[factor_idx] = make_uniq<AggregateFilterDataSet>();
				branch_filter_sets[factor_idx]->Initialize(context.client, branch_aggregates,
				                                           op.source_argument_types[branch_source_idx]);
				for (idx_t range_idx = 0; range_idx < branch_aggregates.size(); range_idx++) {
					if (branch_aggregates[range_idx].aggr_type != AggregateType::DISTINCT) {
						branch_aggregate_filters[factor_idx].push_back(range_idx);
					}
				}
			}
			factor_cache_groups.InitializeEmpty({LogicalType::UBIGINT});
			return;
		}
		cached = op.branch_modes[source_idx - 1] != FactorizedGroupJoinBranchMode::LAZY;

		selected_group_ids.InitializeEmpty({LogicalType::UBIGINT});
		routed_group_ids.Initialize(Allocator::Get(context.client), {LogicalType::UBIGINT});
		if (!op.source_argument_types[source_idx].empty()) {
			selected_arguments.Initialize(Allocator::Get(context.client), op.source_argument_types[source_idx]);
		}
		D_ASSERT(driver_state);
		if (!op.routed && UseDenseFactorizedAggregateTables(op, context.client, driver_state->group_addresses.size())) {
			dense_table = make_uniq<DenseFactorizedAggregateTable>(context.client, driver_state->group_addresses.size(),
			                                                       op.source_argument_types[source_idx],
			                                                       std::move(source_aggregates));
			dense_update_state = dense_table->CreateUpdateState(context.client);
			return;
		}
		local_table = make_uniq<GroupedAggregateHashTable>(
		    context.client, BufferAllocator::Get(context.client), vector<LogicalType> {LogicalType::UBIGINT},
		    op.source_argument_types[source_idx], std::move(source_aggregates),
		    GroupedAggregateHashTable::InitialCapacity(), FACTORIZED_GROUP_JOIN_RADIX_BITS,
		    TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
		update_state = make_uniq<AggregateHTUpdateState>(*local_table);
		direct_allocator = make_shared_ptr<ArenaAllocator>(Allocator::Get(context.client));
		direct_update_state = make_uniq<AggregateHTUpdateState>(*driver_state->target, direct_allocator);
		{
			lock_guard<mutex> guard(driver_state->lock);
			driver_state->target->StoreAggregateAllocator(direct_allocator);
		}
		token = factor_state.next_token.fetch_add(1, std::memory_order_relaxed);
		if (token == 0) {
			throw InternalException("Factorized GroupJoin ownership token overflow");
		}
	}

	idx_t source_idx;
	unique_ptr<ExpressionExecutor> key_executor;
	unique_ptr<ExpressionExecutor> group_executor;
	unique_ptr<ExpressionExecutor> argument_executor;
	DataChunk keys;
	DataChunk groups;
	DataChunk lookup_keys;
	DataChunk arguments;
	DataChunk lookup_arguments;
	DataChunk group_ids;
	DataChunk selected_group_ids;
	DataChunk routed_group_ids;
	DataChunk selected_arguments;
	Vector addresses;
	Vector routing_addresses {LogicalType::POINTER};
	Vector matched_addresses;
	Vector lookup_group_ids {LogicalType::UBIGINT};
	Vector owner_addresses;
	Vector local_addresses;
	Vector routed_addresses {LogicalType::POINTER};
	SelectionVector new_groups;
	SelectionVector new_routes {STANDARD_VECTOR_SIZE};
	SelectionVector non_null_sel;
	SelectionVector found_groups;
	SelectionVector owner_sel;
	SelectionVector local_sel;
	SelectionVector routed_input {STANDARD_VECTOR_SIZE};
	AggregateHTLookupState lookup_state;
	unsafe_vector<idx_t> aggregate_filter;
	unique_ptr<GroupedAggregateHashTable> local_table;
	unique_ptr<DenseFactorizedAggregateTable> dense_table;
	unique_ptr<DenseFactorizedAggregateTable::UpdateState> dense_update_state;
	unique_ptr<GroupedAggregateHashTable> local_driver_target;
	unique_ptr<AggregateHTUpdateState> update_state;
	shared_ptr<ArenaAllocator> direct_allocator;
	unique_ptr<AggregateHTUpdateState> direct_update_state;
	AggregateFilterDataSet source_filter_set;
	vector<unique_ptr<GroupedAggregateHashTable>> distinct_tables;
	vector<unique_ptr<DataChunk>> distinct_groups;
	Vector distinct_addresses {LogicalType::POINTER};
	Vector left_factor_refs {LogicalType::UBIGINT};
	Vector right_factor_refs {LogicalType::UBIGINT};
	SelectionVector left_factor_matches {STANDARD_VECTOR_SIZE};
	SelectionVector right_factor_matches {STANDARD_VECTOR_SIZE};
	FactorizedJoinHashTable::ProbeState left_factor_probe_state;
	FactorizedJoinHashTable::ProbeState right_factor_probe_state;
	DataChunk left_factor_payload;
	DataChunk right_factor_payload;
	vector<unique_ptr<Vector>> left_factor_gather_cache;
	vector<unique_ptr<Vector>> right_factor_gather_cache;
	Vector expanded_factor_rows {LogicalType::POINTER};
	Vector expanded_target_addresses {LogicalType::POINTER};
	Vector expanded_group_ids {LogicalType::UBIGINT};
	DataChunk expanded_group_id_chunk;
	SelectionVector factor_update_sel {STANDARD_VECTOR_SIZE};
	SelectionVector factor_source_sel {STANDARD_VECTOR_SIZE};
	array<unique_ptr<AggregateFilterDataSet>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> branch_filter_sets;
	array<unsafe_vector<idx_t>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> branch_aggregate_filters;
	array<unique_ptr<GroupedAggregateHashTable>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> branch_caches;
	array<unique_ptr<AggregateHTUpdateState>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> branch_cache_update_states;
	DataChunk factor_cache_groups;
	Vector compact_factor_refs {LogicalType::UBIGINT};
	Vector factor_cache_addresses {LogicalType::POINTER};
	Vector factor_cache_update_addresses {LogicalType::POINTER};
	Vector compact_target_addresses {LogicalType::POINTER};
	Vector compact_group_ids {LogicalType::UBIGINT};
	SelectionVector factor_cache_new_groups {STANDARD_VECTOR_SIZE};
	bool external = false;
	DataChunk external_chunk;
	Vector external_hashes {LogicalType::HASH};
	unique_ptr<PartitionedColumnData> external_partitions;
	PartitionedColumnDataAppendState external_append_state;
	vector<uint64_t> acquired_groups;
	vector<UnifiedVectorFormat> key_formats;
	idx_t token = 0;
	idx_t input_rows = 0;
	idx_t matched_rows = 0;
	bool cached = false;
	bool factor_lookup_build = false;
	unique_ptr<FactorizedJoinHashTable> factor_hash_table;
	unique_ptr<JoinFilterLocalState> local_filter_state;
	array<unique_ptr<JoinFilterLocalState>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> driver_filter_states;
	unique_ptr<ColumnDataCollection> driver_filter_keys;
};

unique_ptr<LocalSinkState> PhysicalFactorizedGroupJoinSink::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<FactorizedGroupJoinLocalSinkState>(context, *this);
}

static idx_t SelectNonNullFactorizedKeys(DataChunk &keys, vector<UnifiedVectorFormat> &formats,
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

static void SinkFactorizedDistinct(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                   FactorizedGroupJoinLocalSinkState &state, DataChunk &group_ids, DataChunk &payload,
                                   optional_ptr<AggregateFilterDataSet> source_filter_set = nullptr) {
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		if (distinct.source_idx != source_idx) {
			continue;
		}
		auto &groups = *state.distinct_groups[distinct_idx];
		groups.Reset();
		idx_t count;
		optional_ptr<DataChunk> argument_payload;
		if (distinct.filter_index.IsValid()) {
			auto &active_filter_set = source_filter_set ? *source_filter_set : state.source_filter_set;
			auto &filter_data = active_filter_set.GetFilterData(distinct.range_index);
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
		for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
			groups.data[child_idx + 1].Reference(argument_payload->data[distinct.payload_index + child_idx]);
		}
		groups.SetChildCardinality(count);
		state.distinct_tables[distinct_idx]->FindOrCreateGroups(groups, state.distinct_addresses);
	}
}

static void UpdateFactorizedTargetRows(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                       FactorizedGroupJoinLocalSinkState &state,
                                       FactorizedGroupJoinGlobalSinkState &driver_state, Vector &matched_addresses,
                                       DataChunk &group_ids_chunk, DataChunk &selected_arguments, idx_t match_count) {
	auto &range = op.source_ranges[source_idx];
	SinkFactorizedDistinct(op, source_idx, state, group_ids_chunk, selected_arguments);
	if (state.dense_table) {
		state.dense_table->AddChunk(*state.dense_update_state, group_ids_chunk.data[0], selected_arguments,
		                            state.aggregate_filter);
		return;
	}
	if (state.cached) {
		state.local_table->FindOrCreateGroups(group_ids_chunk, state.addresses, state.new_groups);
		state.local_table->UpdateAggregatesAtAddressesRangeWithFilterSet(
		    *state.update_state, state.addresses, group_ids_chunk.data[0], selected_arguments, 0, range.count,
		    state.aggregate_filter, state.source_filter_set);
		return;
	}
	group_ids_chunk.data[0].Flatten();
	auto group_ids = FlatVector::GetData<uint64_t>(group_ids_chunk.data[0]);
	auto &owners = driver_state.owners[source_idx - 1];
	state.acquired_groups.clear();
	idx_t owner_count = 0;
	idx_t local_count = 0;
	uint64_t previous_id = NumericLimits<uint64_t>::Maximum();
	idx_t current_owner = 0;
	for (idx_t match_idx = 0; match_idx < match_count; match_idx++) {
		if (group_ids[match_idx] != previous_id) {
			previous_id = group_ids[match_idx];
			auto &owner = owners[NumericCast<idx_t>(previous_id)];
			current_owner = owner.load(std::memory_order_relaxed);
			if (current_owner == 0) {
				auto expected = idx_t(0);
				if (owner.compare_exchange_strong(expected, state.token, std::memory_order_acquire,
				                                  std::memory_order_relaxed)) {
					current_owner = state.token;
					state.acquired_groups.push_back(previous_id);
				} else {
					current_owner = expected;
				}
			}
		}
		if (current_owner == state.token) {
			state.owner_sel.set_index(owner_count++, match_idx);
		} else {
			state.local_sel.set_index(local_count++, match_idx);
		}
	}
	auto release_groups = [&]() {
		for (auto group_id : state.acquired_groups) {
			owners[NumericCast<idx_t>(group_id)].store(0, std::memory_order_release);
		}
	};
	auto update_direct = [&](Vector &target_addresses, DataChunk &target_group_ids, DataChunk &payload) {
		driver_state.target->UpdateAggregatesAtAddressesRangeWithFilterSet(
		    *state.direct_update_state, target_addresses, target_group_ids.data[0], payload, range.begin, range.count,
		    state.aggregate_filter, state.source_filter_set);
	};
	try {
		if (owner_count == match_count) {
			update_direct(matched_addresses, group_ids_chunk, selected_arguments);
		} else if (owner_count != 0) {
			state.owner_addresses.Slice(matched_addresses, state.owner_sel, owner_count);
			state.owner_addresses.Flatten();
			state.selected_group_ids.Reset();
			state.selected_group_ids.Slice(group_ids_chunk, state.owner_sel, owner_count);
			DataChunk owner_arguments;
			owner_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
			if (selected_arguments.ColumnCount() == 0) {
				owner_arguments.SetChildCardinality(owner_count);
			} else {
				owner_arguments.Slice(selected_arguments, state.owner_sel, owner_count);
			}
			update_direct(state.owner_addresses, state.selected_group_ids, owner_arguments);
		}
		if (local_count != 0) {
			state.selected_group_ids.Reset();
			state.selected_group_ids.Slice(group_ids_chunk, state.local_sel, local_count);
			DataChunk local_arguments;
			local_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
			if (selected_arguments.ColumnCount() == 0) {
				local_arguments.SetChildCardinality(local_count);
			} else {
				local_arguments.Slice(selected_arguments, state.local_sel, local_count);
			}
			state.local_table->FindOrCreateGroups(state.selected_group_ids, state.addresses, state.new_groups);
			state.local_table->UpdateAggregatesAtAddressesRange(*state.update_state, state.addresses,
			                                                    state.selected_group_ids.data[0], local_arguments, 0,
			                                                    range.count, state.aggregate_filter);
		}
	} catch (...) {
		release_groups();
		throw;
	}
	release_groups();
}

static bool FactorizedBranchHasDistinct(const PhysicalFactorizedGroupJoin &op, idx_t source_idx) {
	for (auto &distinct : op.distinct_aggregates) {
		if (distinct.source_idx == source_idx) {
			return true;
		}
	}
	return false;
}

static void SelectFactorizedBranchMode(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                       FactorizedGroupJoinGlobalSinkState &driver_state,
                                       FactorizedJoinHashTable &factor_hash_table, Vector &factor_refs,
                                       idx_t match_count, ClientContext &context) {
	const auto factor_idx = source_idx - 1;
	if (driver_state.branch_mode_selected[factor_idx]) {
		return;
	}
	const auto sample_count = MinValue<idx_t>(factor_refs.size(), 4096);
	auto refs = factor_refs.Values<uint64_t>();
	unordered_set<uint64_t> distinct_refs;
	idx_t sampled_factor_rows = 0;
	for (idx_t row_idx = 0; row_idx < sample_count; row_idx++) {
		auto ref = JoinFactorRef(refs[row_idx].GetValueUnsafe());
		if (!ref.IsValid()) {
			continue;
		}
		if (distinct_refs.insert(ref.Value()).second) {
			sampled_factor_rows = SaturatingFactorizedAdd(sampled_factor_rows, factor_hash_table.GetFactorCount(ref));
		}
	}
	driver_state.sampled_driver_rows[factor_idx] = sample_count;
	driver_state.sampled_matches[factor_idx] = MinValue(match_count, sample_count);
	driver_state.sampled_factor_rows[factor_idx] = sampled_factor_rows;
	driver_state.sampled_distinct_factors[factor_idx] = distinct_refs.size();

	auto mode = FactorizedGroupJoinBranchMode::LAZY;
	if (factor_hash_table.IsPreaggregated()) {
		mode = FactorizedGroupJoinBranchMode::EAGER;
	} else if (!FactorizedBranchHasDistinct(op, source_idx) && !distinct_refs.empty()) {
		const auto average_fanout = sampled_factor_rows / distinct_refs.size();
		const auto sampled_reuse = driver_state.sampled_matches[factor_idx] / distinct_refs.size();
		const auto estimated_driver_rows = MaxValue<idx_t>(op.driver_input->estimated_cardinality, sample_count);
		const auto remaining_driver_rows = estimated_driver_rows - sample_count;
		const auto projected_matches = static_cast<double>(driver_state.sampled_matches[factor_idx]) +
		                               static_cast<double>(remaining_driver_rows) *
		                                   static_cast<double>(driver_state.sampled_matches[factor_idx]) /
		                                   static_cast<double>(sample_count);
		const auto projected_reuse =
		    projected_matches / static_cast<double>(MaxValue<idx_t>(factor_hash_table.FactorCount(), 1));
		idx_t state_width = 0;
		auto &range = op.source_ranges[source_idx];
		for (idx_t range_idx = 0; range_idx < range.count; range_idx++) {
			state_width = SaturatingFactorizedAdd(state_width,
			                                      op.partial_aggregate_objects[range.begin + range_idx].payload_size);
		}
		idx_t cache_bytes;
		if (!TryMultiplyOperator::Operation(MaxValue<idx_t>(factor_hash_table.FactorCount(), 1),
		                                    SaturatingFactorizedAdd(state_width, sizeof(uint64_t) * 2), cache_bytes)) {
			cache_bytes = NumericLimits<idx_t>::Maximum();
		}
		auto reservation = BufferManager::GetBufferManager(context).GetMaxMemory() / 8;
		const auto cache_fits = cache_bytes <= reservation;
		const auto coverage = static_cast<double>(distinct_refs.size()) /
		                      static_cast<double>(MaxValue<idx_t>(factor_hash_table.FactorCount(), 1));
		if (cache_fits && average_fanout >= 16 && (op.routed || (coverage >= 0.75 && projected_reuse >= 2))) {
			mode = FactorizedGroupJoinBranchMode::EAGER;
		} else if (cache_fits && (op.routed || sampled_reuse >= 2 || projected_reuse >= 2)) {
			mode = FactorizedGroupJoinBranchMode::CACHED;
		}
	}
	driver_state.runtime_branch_modes[factor_idx] = mode;
	driver_state.branch_mode_selected[factor_idx] = true;
}

static GroupedAggregateHashTable &GetFactorizedBranchCache(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                                           FactorizedGroupJoinLocalSinkState &state,
                                                           ClientContext &context) {
	const auto factor_idx = source_idx - 1;
	if (!state.branch_caches[factor_idx]) {
		state.branch_caches[factor_idx] = make_uniq<GroupedAggregateHashTable>(
		    context, BufferAllocator::Get(context), vector<LogicalType> {LogicalType::UBIGINT},
		    op.source_argument_types[source_idx], op.CreateSourceAggregates(source_idx),
		    GroupedAggregateHashTable::InitialCapacity(), idx_t(0), TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
		state.branch_cache_update_states[factor_idx] =
		    make_uniq<AggregateHTUpdateState>(*state.branch_caches[factor_idx]);
	}
	return *state.branch_caches[factor_idx];
}

static void PopulateFactorizedBranchCache(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                          FactorizedGroupJoinLocalSinkState &state,
                                          FactorizedJoinHashTable &factor_hash_table, Vector &factor_refs,
                                          idx_t factor_count, DataChunk &factor_payload, ClientContext &context) {
	if (factor_count == 0) {
		return;
	}
	auto factor_idx = source_idx - 1;
	auto &cache = GetFactorizedBranchCache(op, source_idx, state, context);
	state.factor_cache_groups.Reset();
	state.factor_cache_groups.data[0].Reference(factor_refs);
	state.factor_cache_groups.SetChildCardinality(factor_count);
	auto new_count = cache.FindOrCreateGroups(state.factor_cache_groups, state.factor_cache_addresses,
	                                          state.factor_cache_new_groups);
	if (new_count == 0) {
		return;
	}

	FactorizedJoinHashTable::ExpansionState expansion;
	while (!expansion.Finished(new_count)) {
		auto expanded =
		    factor_hash_table.ExpandFactorRefs(factor_refs, state.factor_cache_new_groups, new_count, expansion,
		                                       state.expanded_factor_rows, state.factor_source_sel);
		if (expanded == 0) {
			throw InternalException("Factorized GroupJoin cached expansion made no progress");
		}
		factor_payload.Reset();
		if (factor_payload.ColumnCount() != 0) {
			auto &gather_cache = source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)
			                         ? state.left_factor_gather_cache
			                         : state.right_factor_gather_cache;
			factor_hash_table.GatherPayload(state.expanded_factor_rows, expanded, factor_payload, gather_cache);
		}
		factor_payload.SetChildCardinality(expanded);
		state.factor_cache_update_addresses.Slice(state.factor_cache_addresses, state.factor_source_sel, expanded);
		state.factor_cache_update_addresses.Flatten();
		state.compact_group_ids.Slice(factor_refs, state.factor_source_sel, expanded);
		state.compact_group_ids.Flatten();
		auto &range = op.source_ranges[source_idx];
		cache.UpdateAggregatesAtAddressesRangeWithFilterSet(
		    *state.branch_cache_update_states[factor_idx], state.factor_cache_update_addresses, state.compact_group_ids,
		    factor_payload, 0, range.count, state.branch_aggregate_filters[factor_idx],
		    *state.branch_filter_sets[factor_idx]);
	}
}

static void PopulateAllFactorizedBranchCache(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                             FactorizedGroupJoinLocalSinkState &state,
                                             FactorizedJoinHashTable &factor_hash_table, DataChunk &factor_payload,
                                             ClientContext &context) {
	idx_t slot = 0;
	while (true) {
		auto count = factor_hash_table.ScanFactorRefs(slot, state.compact_factor_refs);
		if (count == 0) {
			break;
		}
		PopulateFactorizedBranchCache(op, source_idx, state, factor_hash_table, state.compact_factor_refs, count,
		                              factor_payload, context);
	}
}

static idx_t UpdateFactorizedBranchFromRefs(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                            FactorizedGroupJoinLocalSinkState &state,
                                            FactorizedGroupJoinGlobalSinkState &driver_state,
                                            GroupedAggregateHashTable &target,
                                            FactorizedJoinHashTable &factor_hash_table, Vector &factor_refs,
                                            DataChunk &factor_payload, ClientContext &context) {
	D_ASSERT(source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER));
	auto &range = op.source_ranges[source_idx];
	auto layout = target.GetLayoutPtr();
	auto refs = factor_refs.Values<uint64_t>();
	state.addresses.Flatten();
	state.group_ids.data[0].Flatten();
	auto target_addresses = FlatVector::GetData<data_ptr_t>(state.addresses);
	auto group_ids = FlatVector::GetData<uint64_t>(state.group_ids.data[0]);
	unordered_set<uint64_t> selected_groups;
	idx_t update_count = 0;
	for (idx_t row_idx = 0; row_idx < factor_refs.size(); row_idx++) {
		if (!JoinFactorRef(refs[row_idx].GetValueUnsafe()).IsValid()) {
			continue;
		}
		if (range.multiplicity_index.IsValid() &&
		    LoadFactorizedCount(*layout, target_addresses[row_idx], range.multiplicity_index.GetIndex()) != 0) {
			continue;
		}
		if (!selected_groups.insert(group_ids[row_idx]).second) {
			continue;
		}
		state.factor_update_sel.set_index(update_count++, row_idx);
	}
	if (update_count == 0) {
		return 0;
	}
	if (factor_hash_table.IsPreaggregated()) {
		state.compact_factor_refs.Slice(factor_refs, state.factor_update_sel, update_count);
		state.compact_factor_refs.Flatten();
		state.compact_target_addresses.Slice(state.addresses, state.factor_update_sel, update_count);
		state.compact_target_addresses.Flatten();
		factor_hash_table.GetPreaggregateAddresses(state.compact_factor_refs, update_count,
		                                           state.factor_cache_addresses);
		RowOperations::CombineStatesRange(state.update_state->row_state, factor_hash_table.GetLayout(),
		                                  state.factor_cache_addresses, 1, *target.GetLayoutPtr(),
		                                  state.compact_target_addresses, range.begin, range.count, nullptr,
		                                  AggregateCombineType::PRESERVE_INPUT);
		idx_t result = 0;
		auto compact_refs = state.compact_factor_refs.Values<uint64_t>();
		for (idx_t ref_idx = 0; ref_idx < update_count; ref_idx++) {
			result = SaturatingFactorizedAdd(
			    result, factor_hash_table.GetFactorCount(JoinFactorRef(compact_refs[ref_idx].GetValueUnsafe())));
		}
		return result;
	}

	auto factor_idx = source_idx - 1;
	auto mode = driver_state.runtime_branch_modes[factor_idx];
	if (mode != FactorizedGroupJoinBranchMode::LAZY) {
		if (mode == FactorizedGroupJoinBranchMode::EAGER && !state.branch_caches[factor_idx]) {
			PopulateAllFactorizedBranchCache(op, source_idx, state, factor_hash_table, factor_payload, context);
		}
		state.compact_factor_refs.Slice(factor_refs, state.factor_update_sel, update_count);
		state.compact_factor_refs.Flatten();
		state.compact_target_addresses.Slice(state.addresses, state.factor_update_sel, update_count);
		state.compact_target_addresses.Flatten();
		PopulateFactorizedBranchCache(op, source_idx, state, factor_hash_table, state.compact_factor_refs, update_count,
		                              factor_payload, context);
		auto &cache = *state.branch_caches[factor_idx];
		state.factor_cache_groups.Reset();
		state.factor_cache_groups.data[0].Reference(state.compact_factor_refs);
		state.factor_cache_groups.SetChildCardinality(update_count);
		AggregateHTLookupState lookup;
		auto found_count = cache.LookupGroups(state.factor_cache_groups, lookup, state.found_groups);
		if (found_count != update_count) {
			throw InternalException("Factorized GroupJoin cache missed a populated factor");
		}
		RowOperations::CombineStatesRange(state.update_state->row_state, *cache.GetLayoutPtr(), lookup.addresses, 0,
		                                  *target.GetLayoutPtr(), state.compact_target_addresses,
		                                  op.source_ranges[source_idx].begin, op.source_ranges[source_idx].count,
		                                  nullptr, AggregateCombineType::PRESERVE_INPUT);
		idx_t result = 0;
		auto refs = state.compact_factor_refs.Values<uint64_t>();
		for (idx_t ref_idx = 0; ref_idx < update_count; ref_idx++) {
			result = SaturatingFactorizedAdd(
			    result, factor_hash_table.GetFactorCount(JoinFactorRef(refs[ref_idx].GetValueUnsafe())));
		}
		return result;
	}

	idx_t factor_rows = 0;
	FactorizedJoinHashTable::ExpansionState expansion;
	while (!expansion.Finished(update_count)) {
		auto expanded =
		    factor_hash_table.ExpandFactorRefs(factor_refs, state.factor_update_sel, update_count, expansion,
		                                       state.expanded_factor_rows, state.factor_source_sel);
		if (expanded == 0) {
			throw InternalException("Factorized GroupJoin factor expansion made no progress");
		}
		factor_payload.Reset();
		if (factor_payload.ColumnCount() != 0) {
			auto &gather_cache = source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)
			                         ? state.left_factor_gather_cache
			                         : state.right_factor_gather_cache;
			factor_hash_table.GatherPayload(state.expanded_factor_rows, expanded, factor_payload, gather_cache);
		}
		factor_payload.SetChildCardinality(expanded);
		state.expanded_target_addresses.Slice(state.addresses, state.factor_source_sel, expanded);
		state.expanded_target_addresses.Flatten();
		state.expanded_group_ids.Slice(state.group_ids.data[0], state.factor_source_sel, expanded);
		state.expanded_group_ids.Flatten();
		state.expanded_group_id_chunk.Reset();
		state.expanded_group_id_chunk.data[0].Reference(state.expanded_group_ids);
		state.expanded_group_id_chunk.SetChildCardinality(expanded);
		SinkFactorizedDistinct(op, source_idx, state, state.expanded_group_id_chunk, factor_payload,
		                       *state.branch_filter_sets[factor_idx]);
		target.UpdateAggregatesAtAddressesRangeWithFilterSet(
		    *state.update_state, state.expanded_target_addresses, state.expanded_group_ids, factor_payload, range.begin,
		    range.count, state.branch_aggregate_filters[factor_idx], *state.branch_filter_sets[factor_idx]);
		factor_rows += expanded;
	}
	return factor_rows;
}

SinkResultType PhysicalFactorizedGroupJoinSink::Sink(ExecutionContext &context, DataChunk &chunk,
                                                     OperatorSinkInput &input) const {
	auto &state = input.local_state.Cast<FactorizedGroupJoinLocalSinkState>();
	state.keys.Reset();
	state.key_executor->Execute(chunk, state.keys);
	if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
		for (idx_t factor_idx = 0; factor_idx < state.driver_filter_states.size(); factor_idx++) {
			if (state.driver_filter_states[factor_idx]) {
				op.driver_filter_pushdown[factor_idx]->Sink(state.keys, *state.driver_filter_states[factor_idx]);
			}
		}
		if (state.driver_filter_keys) {
			state.driver_filter_keys->Append(state.keys);
		}
	}
	if (source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && state.local_filter_state) {
		D_ASSERT(state.local_filter_state);
		op.factor_filter_pushdown[source_idx - 1]->Sink(state.keys, *state.local_filter_state);
	}
	if (state.group_executor) {
		state.groups.Reset();
		state.group_executor->Execute(chunk, state.groups);
	}
	state.arguments.Reset();
	if (state.arguments.ColumnCount() == 0) {
		state.arguments.SetChildCardinality(chunk.size());
	} else {
		state.argument_executor->Execute(chunk, state.arguments);
	}
	state.input_rows += chunk.size();
	if (state.external) {
		state.keys.Hash(state.external_hashes);
		state.external_chunk.Reset();
		idx_t output_idx = 0;
		for (idx_t key_idx = 0; key_idx < state.keys.ColumnCount(); key_idx++) {
			state.external_chunk.data[output_idx++].Reference(state.keys.data[key_idx]);
		}
		if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
			auto &external_groups = op.routed ? state.groups : state.keys;
			for (idx_t group_idx = 0; group_idx < external_groups.ColumnCount(); group_idx++) {
				state.external_chunk.data[output_idx++].Reference(external_groups.data[group_idx]);
			}
		}
		for (idx_t argument_idx = 0; argument_idx < state.arguments.ColumnCount(); argument_idx++) {
			state.external_chunk.data[output_idx++].Reference(state.arguments.data[argument_idx]);
		}
		state.external_chunk.data[output_idx].Reference(state.external_hashes);
		state.external_chunk.SetChildCardinality(chunk.size());
		state.external_partitions->Append(state.external_append_state, state.external_chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}
	if (state.factor_lookup_build) {
		state.factor_hash_table->Build(state.keys, state.arguments);
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &range = op.source_ranges[source_idx];
	if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
		auto &driver_state = input.global_state.Cast<FactorizedGroupJoinGlobalSinkState>();
		auto &target = state.local_driver_target ? *state.local_driver_target : *driver_state.target;
		idx_t left_match_count = 0;
		idx_t right_match_count = 0;
		optional_ptr<FactorizedGroupJoinGlobalSinkState> left_factor_state;
		optional_ptr<FactorizedGroupJoinGlobalSinkState> right_factor_state;
		if (!driver_state.direct_factor_updates) {
			left_factor_state =
			    GetFactorizedFactorState(op, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR));
			right_factor_state =
			    GetFactorizedFactorState(op, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR));
			D_ASSERT(left_factor_state->finalized && right_factor_state->finalized);
			left_match_count = left_factor_state->factor_hash_table->ProbeFactorRefs(
			    state.keys, state.left_factor_probe_state, state.left_factor_refs, state.left_factor_matches);
			right_match_count = right_factor_state->factor_hash_table->ProbeFactorRefs(
			    state.keys, state.right_factor_probe_state, state.right_factor_refs, state.right_factor_matches);
			{
				lock_guard<mutex> guard(driver_state.lock);
				SelectFactorizedBranchMode(op, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR),
				                           driver_state, *left_factor_state->factor_hash_table, state.left_factor_refs,
				                           left_match_count, context.client);
				SelectFactorizedBranchMode(op, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR),
				                           driver_state, *right_factor_state->factor_hash_table,
				                           state.right_factor_refs, right_match_count, context.client);
			}
			left_factor_state->matched_rows.fetch_add(left_match_count, std::memory_order_relaxed);
			right_factor_state->matched_rows.fetch_add(right_match_count, std::memory_order_relaxed);
		}
		auto &target_groups = op.routed ? state.groups : state.keys;
		auto new_count = target.FindOrCreateGroups(target_groups, state.addresses, state.new_groups);
		if (op.unique_driver && new_count != chunk.size()) {
			throw InternalException("Factorized GroupJoin driver key was not unique");
		}
		auto addresses = FlatVector::GetData<data_ptr_t>(state.addresses);
		const auto group_id_begin = driver_state.parallel_driver
		                                ? driver_state.next_group_id.fetch_add(new_count, std::memory_order_relaxed)
		                                : driver_state.group_addresses.size();
		for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
			auto input_idx = state.new_groups.get_index_unsafe(new_idx);
			auto group_id = group_id_begin + new_idx;
			StoreFactorizedGroupJoinId(target.GetLayout(), addresses[input_idx], group_id);
			if (!driver_state.parallel_driver) {
				driver_state.group_addresses.push_back(addresses[input_idx]);
			}
		}
		if (op.routed) {
			D_ASSERT(driver_state.routing_table);
			auto route_count =
			    driver_state.routing_table->FindOrCreateGroups(state.keys, state.routing_addresses, state.new_routes);
			auto route_addresses = FlatVector::GetData<data_ptr_t>(state.routing_addresses);
			for (idx_t new_idx = 0; new_idx < route_count; new_idx++) {
				auto input_idx = state.new_routes.get_index_unsafe(new_idx);
				auto route_id = driver_state.routes.size();
				StoreFactorizedGroupJoinId(driver_state.routing_table->GetLayout(), route_addresses[input_idx],
				                           route_id);
				driver_state.routes.emplace_back();
			}
			for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
				auto input_idx = state.new_groups.get_index_unsafe(new_idx);
				auto route_id =
				    LoadFactorizedGroupJoinId(driver_state.routing_table->GetLayout(), route_addresses[input_idx]);
				if (route_id >= driver_state.routes.size()) {
					throw InternalException("Factorized GroupJoin route identifier exceeds route count");
				}
				driver_state.routes[route_id].push_back(addresses[input_idx]);
			}
		}
		bool has_driver_distinct = false;
		for (auto &distinct : op.distinct_aggregates) {
			has_driver_distinct = has_driver_distinct || distinct.source_idx == source_idx;
		}
		{
			state.group_ids.Reset();
			state.group_ids.SetChildCardinality(chunk.size());
			state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto group_ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
			for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
				group_ids[row_idx] = LoadFactorizedGroupJoinId(target.GetLayout(), addresses[row_idx]);
			}
			FlatVector::SetSize(state.group_ids.data[0], chunk.size());
		}
		if (driver_state.perfect_executor &&
		    !driver_state.perfect_executor->Sink(state.keys.data[0], state.addresses, state.group_ids.data[0])) {
			driver_state.perfect_executor.reset();
		}
		if (has_driver_distinct) {
			SinkFactorizedDistinct(op, source_idx, state, state.group_ids, state.arguments);
		}
		if (range.count > 0) {
			target.UpdateAggregatesAtAddressesRangeWithFilterSet(*state.update_state, state.addresses, state.arguments,
			                                                     range.begin, range.count, state.aggregate_filter,
			                                                     state.source_filter_set);
		}
		if (!driver_state.direct_factor_updates) {
			UpdateFactorizedBranchFromRefs(op, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR), state,
			                               driver_state, target, *left_factor_state->factor_hash_table,
			                               state.left_factor_refs, state.left_factor_payload, context.client);
			UpdateFactorizedBranchFromRefs(op, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR), state,
			                               driver_state, target, *right_factor_state->factor_hash_table,
			                               state.right_factor_refs, state.right_factor_payload, context.client);
			state.matched_rows += MinValue(left_match_count, right_match_count);
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &driver_state = GetFactorizedDriverState(op);
	D_ASSERT(driver_state.finalized);
	auto non_null_count = SelectNonNullFactorizedKeys(state.keys, state.key_formats, state.non_null_sel);
	if (non_null_count == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	state.lookup_keys.Reset();
	state.lookup_arguments.Reset();
	if (non_null_count == chunk.size()) {
		state.lookup_keys.Reference(state.keys);
		if (state.arguments.ColumnCount() == 0) {
			state.lookup_arguments.SetChildCardinality(non_null_count);
		} else {
			state.lookup_arguments.Reference(state.arguments);
		}
	} else {
		state.lookup_keys.Slice(state.keys, state.non_null_sel, non_null_count);
		if (state.arguments.ColumnCount() == 0) {
			state.lookup_arguments.SetChildCardinality(non_null_count);
		} else {
			state.lookup_arguments.Slice(state.arguments, state.non_null_sel, non_null_count);
		}
	}
	idx_t found_count;
	if (driver_state.perfect_executor) {
		found_count = driver_state.perfect_executor->Lookup(state.lookup_keys.data[0], state.lookup_state.addresses,
		                                                    state.lookup_group_ids, state.found_groups);
	} else {
		auto &lookup_table = op.routed ? *driver_state.routing_table : *driver_state.target;
		found_count = lookup_table.LookupGroups(state.lookup_keys, state.lookup_state, state.found_groups);
	}
	if (found_count == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	state.group_ids.Reset();
	state.selected_arguments.Reset();
	if (found_count == non_null_count) {
		state.matched_addresses.Reference(state.lookup_state.addresses);
		if (driver_state.perfect_executor) {
			state.group_ids.data[0].Reference(state.lookup_group_ids);
		} else {
			state.group_ids.SetChildCardinality(found_count);
			state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto group_ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
			auto lookup_addresses = FlatVector::GetData<data_ptr_t>(state.lookup_state.addresses);
			auto &lookup_layout =
			    op.routed ? driver_state.routing_table->GetLayout() : driver_state.target->GetLayout();
			for (idx_t row_idx = 0; row_idx < found_count; row_idx++) {
				group_ids[row_idx] = LoadFactorizedGroupJoinId(lookup_layout, lookup_addresses[row_idx]);
			}
			FlatVector::SetSize(state.group_ids.data[0], found_count);
		}
		if (state.lookup_arguments.ColumnCount() == 0) {
			state.selected_arguments.SetChildCardinality(found_count);
		} else {
			state.selected_arguments.Reference(state.lookup_arguments);
		}
	} else {
		state.matched_addresses.SetVectorType(VectorType::FLAT_VECTOR);
		auto matched_addresses = FlatVector::GetDataMutable<data_ptr_t>(state.matched_addresses);
		auto lookup_addresses = FlatVector::GetData<data_ptr_t>(state.lookup_state.addresses);
		state.group_ids.SetChildCardinality(found_count);
		state.group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
		auto group_ids = FlatVector::GetDataMutable<uint64_t>(state.group_ids.data[0]);
		auto lookup_group_ids =
		    driver_state.perfect_executor ? FlatVector::GetData<uint64_t>(state.lookup_group_ids) : nullptr;
		auto &lookup_layout = op.routed ? driver_state.routing_table->GetLayout() : driver_state.target->GetLayout();
		for (idx_t match_idx = 0; match_idx < found_count; match_idx++) {
			auto input_idx = state.found_groups.get_index_unsafe(match_idx);
			matched_addresses[match_idx] = lookup_addresses[input_idx];
			group_ids[match_idx] = driver_state.perfect_executor
			                           ? lookup_group_ids[input_idx]
			                           : LoadFactorizedGroupJoinId(lookup_layout, lookup_addresses[input_idx]);
		}
		FlatVector::SetSize(state.matched_addresses, found_count);
		FlatVector::SetSize(state.group_ids.data[0], found_count);
		if (state.lookup_arguments.ColumnCount() == 0) {
			state.selected_arguments.SetChildCardinality(found_count);
		} else {
			state.selected_arguments.Slice(state.lookup_arguments, state.found_groups, found_count);
		}
	}
	if (!op.routed) {
		UpdateFactorizedTargetRows(op, source_idx, state, driver_state, state.matched_addresses, state.group_ids,
		                           state.selected_arguments, found_count);
	} else if (state.cached) {
		// Cache by complete-key route. Every output group on the route sees the same factor multiset.
		UpdateFactorizedTargetRows(op, source_idx, state, driver_state, state.matched_addresses, state.group_ids,
		                           state.selected_arguments, found_count);
	} else {
		state.group_ids.data[0].Flatten();
		auto route_ids = FlatVector::GetData<uint64_t>(state.group_ids.data[0]);
		auto routed_addresses = FlatVector::GetDataMutable<data_ptr_t>(state.routed_addresses);
		auto routed_ids = FlatVector::GetDataMutable<uint64_t>(state.routed_group_ids.data[0]);
		idx_t routed_count = 0;
		auto flush_routes = [&]() {
			if (routed_count == 0) {
				return;
			}
			FlatVector::SetSize(state.routed_addresses, routed_count);
			FlatVector::SetSize(state.routed_group_ids.data[0], routed_count);
			state.routed_group_ids.SetChildCardinality(routed_count);
			DataChunk routed_arguments;
			routed_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
			if (state.selected_arguments.ColumnCount() == 0) {
				routed_arguments.SetChildCardinality(routed_count);
			} else {
				routed_arguments.Slice(state.selected_arguments, state.routed_input, routed_count);
			}
			UpdateFactorizedTargetRows(op, source_idx, state, driver_state, state.routed_addresses,
			                           state.routed_group_ids, routed_arguments, routed_count);
			routed_count = 0;
		};
		for (idx_t match_idx = 0; match_idx < found_count; match_idx++) {
			auto route_id = NumericCast<idx_t>(route_ids[match_idx]);
			if (route_id + 1 >= driver_state.route_offsets.size()) {
				throw InternalException("Factorized GroupJoin lookup route exceeds route count");
			}
			for (idx_t route_idx = driver_state.route_offsets[route_id];
			     route_idx < driver_state.route_offsets[route_id + 1]; route_idx++) {
				auto target_address = driver_state.route_addresses[route_idx];
				routed_addresses[routed_count] = target_address;
				routed_ids[routed_count] = LoadFactorizedGroupJoinId(driver_state.target->GetLayout(), target_address);
				state.routed_input.set_index(routed_count, match_idx);
				if (++routed_count == STANDARD_VECTOR_SIZE) {
					flush_routes();
				}
			}
		}
		flush_routes();
	}
	state.matched_rows += found_count;
	return SinkResultType::NEED_MORE_INPUT;
}

static void MergeFactorizedLocalTable(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                      FactorizedGroupJoinGlobalSinkState &driver_state,
                                      GroupedAggregateHashTable &local_table, ClientContext &context) {
	if (local_table.Count() == 0) {
		return;
	}
	auto &target = *driver_state.target;
	auto local_layout = local_table.GetLayoutPtr();
	auto target_layout = target.GetLayoutPtr();
	AggregateHTScanState scan_state;
	local_table.InitializeScan(scan_state);
	DataChunk group_ids;
	group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
	Vector source_addresses(LogicalType::POINTER);
	Vector target_addresses(LogicalType::POINTER);
	Vector routed_source_addresses(LogicalType::POINTER);
	Vector routed_target_addresses(LogicalType::POINTER);
	AggregateHTUpdateState merge_state(target);
	target.InheritAggregateAllocators(local_table);
	auto &range = op.source_ranges[source_idx];
	while (local_table.ScanGroupsAndAddresses(scan_state, group_ids, source_addresses)) {
		if (group_ids.size() == 0) {
			continue;
		}
		group_ids.Flatten();
		auto ids = FlatVector::GetData<uint64_t>(group_ids.data[0]);
		auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
		const auto route_cached = op.routed && op.branch_modes[source_idx - 1] != FactorizedGroupJoinBranchMode::LAZY;
		if (route_cached) {
			source_addresses.Flatten();
			auto sources = FlatVector::GetData<data_ptr_t>(source_addresses);
			auto routed_sources = FlatVector::GetDataMutable<data_ptr_t>(routed_source_addresses);
			auto routed_targets = FlatVector::GetDataMutable<data_ptr_t>(routed_target_addresses);
			idx_t routed_count = 0;
			auto flush = [&]() {
				if (routed_count == 0) {
					return;
				}
				FlatVector::SetSize(routed_source_addresses, routed_count);
				FlatVector::SetSize(routed_target_addresses, routed_count);
				RowOperations::CombineStatesRange(merge_state.row_state, *local_layout, routed_source_addresses, 0,
				                                  *target_layout, routed_target_addresses, range.begin, range.count,
				                                  nullptr, AggregateCombineType::PRESERVE_INPUT);
				routed_count = 0;
			};
			for (idx_t row_idx = 0; row_idx < group_ids.size(); row_idx++) {
				auto route_id = NumericCast<idx_t>(ids[row_idx]);
				if (route_id + 1 >= driver_state.route_offsets.size()) {
					throw InternalException("Factorized GroupJoin cached route exceeds route count");
				}
				for (idx_t route_idx = driver_state.route_offsets[route_id];
				     route_idx < driver_state.route_offsets[route_id + 1]; route_idx++) {
					auto target_address = driver_state.route_addresses[route_idx];
					routed_sources[routed_count] = sources[row_idx];
					routed_targets[routed_count] = target_address;
					if (++routed_count == STANDARD_VECTOR_SIZE) {
						flush();
					}
				}
			}
			flush();
			continue;
		}
		for (idx_t row_idx = 0; row_idx < group_ids.size(); row_idx++) {
			auto group_id = NumericCast<idx_t>(ids[row_idx]);
			if (group_id >= driver_state.group_addresses.size()) {
				throw InternalException("Factorized GroupJoin local group identifier exceeds driver group count");
			}
			targets[row_idx] = driver_state.group_addresses[group_id];
		}
		FlatVector::SetSize(target_addresses, group_ids.size());
		RowOperations::CombineStatesRange(merge_state.row_state, *local_layout, source_addresses, 0, *target_layout,
		                                  target_addresses, range.begin, range.count);
	}
}

static void MergeFactorizedDistinct(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                    FactorizedGroupJoinGlobalSinkState &source_state,
                                    FactorizedGroupJoinGlobalSinkState &driver_state, ClientContext &context) {
	auto &target = *driver_state.target;
	auto allocator = make_shared_ptr<ArenaAllocator>(Allocator::Get(context));
	target.StoreAggregateAllocator(allocator);
	AggregateHTUpdateState update_state(target, allocator);
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		if (distinct.source_idx != source_idx) {
			continue;
		}
		vector<LogicalType> distinct_types {LogicalType::UBIGINT};
		distinct_types.insert(distinct_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
		auto global_distinct = make_uniq<GroupedAggregateHashTable>(
		    context, BufferAllocator::Get(context), distinct_types, vector<LogicalType> {}, vector<AggregateObject> {},
		    GroupedAggregateHashTable::InitialCapacity(), FACTORIZED_GROUP_JOIN_RADIX_BITS,
		    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		DataChunk distinct_rows;
		distinct_rows.Initialize(Allocator::Get(context), distinct_types);
		Vector insert_addresses(LogicalType::POINTER);
		for (auto &task_tables : source_state.local_distinct_tables) {
			if (!task_tables[distinct_idx]) {
				continue;
			}
			AggregateHTScanState local_scan;
			task_tables[distinct_idx]->InitializeScan(local_scan);
			while (task_tables[distinct_idx]->ScanGroups(local_scan, distinct_rows)) {
				global_distinct->FindOrCreateGroups(distinct_rows, insert_addresses);
			}
		}

		DataChunk payload;
		payload.Initialize(Allocator::Get(context), op.source_argument_types[source_idx]);
		Vector target_addresses(LogicalType::POINTER);
		unsafe_vector<idx_t> distinct_filter {distinct.range_index};
		auto &range = op.source_ranges[source_idx];
		AggregateHTScanState global_scan;
		global_distinct->InitializeScan(global_scan);
		while (global_distinct->ScanGroups(global_scan, distinct_rows)) {
			distinct_rows.Flatten();
			auto ids = FlatVector::GetData<uint64_t>(distinct_rows.data[0]);
			auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
			payload.Reset();
			for (idx_t payload_idx = 0; payload_idx < op.source_argument_types[source_idx].size(); payload_idx++) {
				payload.data[payload_idx].Reference(Value(op.source_argument_types[source_idx][payload_idx]),
				                                    count_t(distinct_rows.size()));
			}
			for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
				payload.data[distinct.payload_index + child_idx].Reference(distinct_rows.data[child_idx + 1]);
			}
			if (distinct.filter_index.IsValid()) {
				payload.data[distinct.filter_index.GetIndex()].Reference(Value::BOOLEAN(true),
				                                                         count_t(distinct_rows.size()));
			}
			payload.SetChildCardinality(distinct_rows.size());
			const auto route_cached = op.routed &&
			                          source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) &&
			                          &source_state != &driver_state &&
			                          op.branch_modes[source_idx - 1] != FactorizedGroupJoinBranchMode::LAZY;
			if (route_cached) {
				SelectionVector routed_input(STANDARD_VECTOR_SIZE);
				idx_t routed_count = 0;
				auto flush = [&]() {
					if (routed_count == 0) {
						return;
					}
					FlatVector::SetSize(target_addresses, routed_count);
					DataChunk routed_payload;
					routed_payload.InitializeEmpty(op.source_argument_types[source_idx]);
					if (payload.ColumnCount() == 0) {
						routed_payload.SetChildCardinality(routed_count);
					} else {
						routed_payload.Slice(payload, routed_input, routed_count);
					}
					target.UpdateAggregatesAtAddressesRange(update_state, target_addresses, routed_payload, range.begin,
					                                        range.count, distinct_filter);
					routed_count = 0;
				};
				for (idx_t row_idx = 0; row_idx < distinct_rows.size(); row_idx++) {
					auto route_id = NumericCast<idx_t>(ids[row_idx]);
					if (route_id + 1 >= driver_state.route_offsets.size()) {
						throw InternalException("Factorized GroupJoin distinct route exceeds route count");
					}
					for (idx_t route_idx = driver_state.route_offsets[route_id];
					     route_idx < driver_state.route_offsets[route_id + 1]; route_idx++) {
						auto target_address = driver_state.route_addresses[route_idx];
						targets[routed_count] = target_address;
						routed_input.set_index(routed_count, row_idx);
						if (++routed_count == STANDARD_VECTOR_SIZE) {
							flush();
						}
					}
				}
				flush();
				continue;
			}
			for (idx_t row_idx = 0; row_idx < distinct_rows.size(); row_idx++) {
				auto group_id = NumericCast<idx_t>(ids[row_idx]);
				if (group_id >= driver_state.group_addresses.size()) {
					throw InternalException("Factorized GroupJoin distinct identifier exceeds group count");
				}
				targets[row_idx] = driver_state.group_addresses[group_id];
			}
			FlatVector::SetSize(target_addresses, distinct_rows.size());
			target.UpdateAggregatesAtAddressesRange(update_state, target_addresses, payload, range.begin, range.count,
			                                        distinct_filter);
		}
	}
}

SinkCombineResultType PhysicalFactorizedGroupJoinSink::Combine(ExecutionContext &,
                                                               OperatorSinkCombineInput &input) const {
	auto &global_state = input.global_state.Cast<FactorizedGroupJoinGlobalSinkState>();
	auto &local_state = input.local_state.Cast<FactorizedGroupJoinLocalSinkState>();
	if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) &&
	    (local_state.driver_filter_states[0] || local_state.driver_filter_states[1])) {
		lock_guard<mutex> guard(global_state.lock);
		for (idx_t factor_idx = 0; factor_idx < local_state.driver_filter_states.size(); factor_idx++) {
			if (!local_state.driver_filter_states[factor_idx]) {
				continue;
			}
			op.driver_filter_pushdown[factor_idx]->Combine(*global_state.driver_filter_states[factor_idx],
			                                               *local_state.driver_filter_states[factor_idx]);
		}
		if (local_state.driver_filter_keys) {
			global_state.driver_filter_keys->Combine(*local_state.driver_filter_keys);
			local_state.driver_filter_keys.reset();
		}
	}
	if (source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && global_state.global_filter_state) {
		D_ASSERT(global_state.global_filter_state && local_state.local_filter_state);
		lock_guard<mutex> guard(global_state.lock);
		op.factor_filter_pushdown[source_idx - 1]->Combine(*global_state.global_filter_state,
		                                                   *local_state.local_filter_state);
	}
	if (local_state.external) {
		local_state.external_partitions->FlushAppendState(local_state.external_append_state);
		global_state.partitions->Combine(*local_state.external_partitions);
		local_state.external_partitions.reset();
		global_state.input_rows.fetch_add(local_state.input_rows, std::memory_order_relaxed);
		return SinkCombineResultType::FINISHED;
	}
	if (local_state.factor_lookup_build) {
		lock_guard<mutex> guard(global_state.lock);
		global_state.factor_hash_table->Merge(*local_state.factor_hash_table);
		global_state.input_rows.fetch_add(local_state.input_rows, std::memory_order_relaxed);
		return SinkCombineResultType::FINISHED;
	}
	if (local_state.local_driver_target) {
		lock_guard<mutex> guard(global_state.lock);
		global_state.target->MoveUniqueGroups(*local_state.local_driver_target);
	}
	if (local_state.dense_table) {
		lock_guard<mutex> guard(global_state.lock);
		global_state.local_dense_tables.push_back(std::move(local_state.dense_table));
	} else if (source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) &&
	           local_state.local_table->Count() != 0) {
		lock_guard<mutex> guard(global_state.lock);
		global_state.local_tables.push_back(std::move(local_state.local_table));
	}
	bool has_distinct_rows = false;
	for (auto &table : local_state.distinct_tables) {
		has_distinct_rows = has_distinct_rows || (table && table->Count() != 0);
	}
	if (has_distinct_rows) {
		lock_guard<mutex> guard(global_state.lock);
		global_state.local_distinct_tables.push_back(std::move(local_state.distinct_tables));
	}
	global_state.input_rows.fetch_add(local_state.input_rows, std::memory_order_relaxed);
	global_state.matched_rows.fetch_add(local_state.matched_rows, std::memory_order_relaxed);
	return SinkCombineResultType::FINISHED;
}

static void UpdateFactorizedNullExtendedRows(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                             FactorizedGroupJoinGlobalSinkState &driver_state, ClientContext &context) {
	if ((source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR) && !op.preserve_left) ||
	    (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR) && !op.preserve_right)) {
		return;
	}
	auto &range = op.source_ranges[source_idx];
	D_ASSERT(range.multiplicity_index.IsValid());
	auto &target = *driver_state.target;
	auto layout = target.GetLayoutPtr();
	ExpressionExecutor executor(context);
	for (auto &expression : op.source_arguments[source_idx]) {
		executor.AddExpression(*expression);
	}
	DataChunk null_input;
	null_input.Initialize(Allocator::Get(context), op.source_input_types[source_idx]);
	DataChunk null_payload;
	null_payload.Initialize(Allocator::Get(context), op.source_argument_types[source_idx]);
	Vector unmatched_addresses(LogicalType::POINTER);
	auto source_aggregates = op.CreateSourceAggregates(source_idx);
	unsafe_vector<idx_t> aggregate_filter;
	for (idx_t range_idx = 0; range_idx < range.count; range_idx++) {
		if (source_aggregates[range_idx].aggr_type != AggregateType::DISTINCT) {
			aggregate_filter.push_back(range_idx);
		}
	}
	AggregateFilterDataSet filter_set;
	filter_set.Initialize(context, source_aggregates, op.source_argument_types[source_idx]);
	auto allocator = make_shared_ptr<ArenaAllocator>(Allocator::Get(context));
	target.StoreAggregateAllocator(allocator);
	AggregateHTUpdateState update_state(target, allocator);

	for (idx_t group_begin = 0; group_begin < driver_state.group_addresses.size();
	     group_begin += STANDARD_VECTOR_SIZE) {
		auto group_end = MinValue<idx_t>(group_begin + STANDARD_VECTOR_SIZE, driver_state.group_addresses.size());
		auto addresses = FlatVector::GetDataMutable<data_ptr_t>(unmatched_addresses);
		idx_t unmatched_count = 0;
		for (idx_t group_idx = group_begin; group_idx < group_end; group_idx++) {
			auto address = driver_state.group_addresses[group_idx];
			if (LoadFactorizedCount(*layout, address, range.multiplicity_index.GetIndex()) == 0) {
				addresses[unmatched_count++] = address;
			}
		}
		if (unmatched_count == 0) {
			continue;
		}
		FlatVector::SetSize(unmatched_addresses, unmatched_count);
		null_input.Reset();
		for (idx_t column_idx = 0; column_idx < op.source_input_types[source_idx].size(); column_idx++) {
			null_input.data[column_idx].Reference(Value(op.source_input_types[source_idx][column_idx]),
			                                      count_t(unmatched_count));
		}
		null_input.SetChildCardinality(unmatched_count);
		null_payload.Reset();
		if (op.source_argument_types[source_idx].empty()) {
			null_payload.SetChildCardinality(unmatched_count);
		} else {
			executor.Execute(null_input, null_payload);
		}
		target.UpdateAggregatesAtAddressesRangeWithFilterSet(update_state, unmatched_addresses, null_payload,
		                                                     range.begin, range.count, aggregate_filter, filter_set);
		for (auto &distinct : op.distinct_aggregates) {
			if (distinct.source_idx != source_idx) {
				continue;
			}
			idx_t distinct_count = unmatched_count;
			optional_ptr<DataChunk> distinct_payload = null_payload;
			Vector distinct_addresses(LogicalType::POINTER);
			if (distinct.filter_index.IsValid()) {
				auto &filter_data = filter_set.GetFilterData(distinct.range_index);
				distinct_count = filter_data.ApplyFilter(null_payload);
				if (distinct_count == 0) {
					continue;
				}
				distinct_addresses.Slice(unmatched_addresses, filter_data.true_sel, distinct_count);
				distinct_payload = filter_data.filtered_payload;
			} else {
				distinct_addresses.Reference(unmatched_addresses);
			}
			distinct_addresses.Flatten();
			unsafe_vector<idx_t> distinct_filter {distinct.range_index};
			target.UpdateAggregatesAtAddressesRange(update_state, distinct_addresses, *distinct_payload, range.begin,
			                                        range.count, distinct_filter);
		}
	}
}

static void BuildFactorizedDriverRuntimeFilters(const PhysicalFactorizedGroupJoin &op, ClientContext &context,
                                                FactorizedGroupJoinGlobalSinkState &state, idx_t factor_idx,
                                                const DataChunk &final_min_max) {
	auto &filter = op.driver_filter_pushdown[factor_idx];
	D_ASSERT(filter && state.driver_filter_keys);
	const auto key_count = op.key_types.size();
	auto &bloom_filters = state.driver_bloom_filters[factor_idx];
	auto &prefix_range_filters = state.driver_prefix_range_filters[factor_idx];
	bloom_filters.resize(key_count);
	prefix_range_filters.resize(key_count);
	const auto driver_rows = state.driver_filter_keys->Count();
	const auto factor_rows =
	    factor_idx == 0 ? op.left_input->estimated_cardinality : op.right_input->estimated_cardinality;
	if (driver_rows == 0 || driver_rows > factor_rows ||
	    (driver_rows > 4194304 && !filter->build_side_has_filter && driver_rows > factor_rows / 10)) {
		return;
	}

	for (idx_t filter_idx = 0; filter_idx < filter->join_condition.size(); filter_idx++) {
		auto key_idx = filter->join_condition[filter_idx];
		D_ASSERT(key_idx < key_count);
		auto min_value = final_min_max.data[filter_idx * 2].GetValue(0);
		auto max_value = final_min_max.data[filter_idx * 2 + 1].GetValue(0);
		if (min_value.IsNull() || max_value.IsNull() || Value::NotDistinctFrom(min_value, max_value)) {
			continue;
		}

		unique_ptr<PrefixRangeFilter::BuildState> prefix_build_state;
		if (PrefixRangeFilter::SupportedType(op.key_types[key_idx])) {
			uhugeint_t span;
			if (PrefixRangeFilter::TryComputeSpan(min_value, max_value, span)) {
				static constexpr idx_t MAX_EXACT_BITS = idx_t(1) << 26;
				const auto bloom_bits = BloomFilter::GetNumberOfSectors(MaxValue<idx_t>(driver_rows, idx_t(1))) * 64;
				idx_t max_bits = 0;
				if (span < MAX_EXACT_BITS) {
					max_bits = MAX_EXACT_BITS;
				} else if (span <= bloom_bits) {
					max_bits = bloom_bits;
				}
				if (max_bits != 0) {
					prefix_range_filters[key_idx] = PrefixRangeFilter::CreatePrefixRangeFilter(op.key_types[key_idx]);
					prefix_range_filters[key_idx]->Initialize(context, driver_rows, min_value, max_value, max_bits);
					prefix_build_state = prefix_range_filters[key_idx]->InitializeBuildState(context);
				}
			}
		}
		if (!prefix_range_filters[key_idx]) {
			bloom_filters[key_idx] = make_uniq<BloomFilter>();
			bloom_filters[key_idx]->Initialize(context, driver_rows);
		}

		Vector hashes(LogicalType::HASH);
		for (auto &chunk : state.driver_filter_keys->Chunks()) {
			if (prefix_range_filters[key_idx]) {
				prefix_range_filters[key_idx]->InsertKeys(chunk.data[key_idx], *prefix_build_state);
			} else {
				VectorOperations::Hash(chunk.data[key_idx], hashes, chunk.size());
				hashes.Flatten();
				bloom_filters[key_idx]->InsertHashes(hashes);
			}
		}
		if (prefix_range_filters[key_idx]) {
			prefix_range_filters[key_idx]->MergeBuildState(*prefix_build_state);
		}
	}
}

static void FinalizeFactorizedDriverRuntimeFilters(const PhysicalFactorizedGroupJoin &op, ClientContext &context,
                                                   FactorizedGroupJoinGlobalSinkState &state) {
	if (!state.driver_filter_keys) {
		return;
	}
	vector<string> key_names;
	key_names.reserve(op.key_types.size());
	for (idx_t key_idx = 0; key_idx < op.key_types.size(); key_idx++) {
		key_names.push_back(StringUtil::Format("driver_key_%llu", key_idx));
	}
	for (idx_t factor_idx = 0; factor_idx < state.driver_filter_states.size(); factor_idx++) {
		if (!state.driver_filter_states[factor_idx]) {
			continue;
		}
		auto final_min_max =
		    op.driver_filter_pushdown[factor_idx]->FinalizeMinMax(*state.driver_filter_states[factor_idx]);
		BuildFactorizedDriverRuntimeFilters(op, context, state, factor_idx, *final_min_max);
		op.driver_filter_pushdown[factor_idx]->FinalizeGroupJoinFilters(
		    context, op, op.key_types, key_names, state.driver_bloom_filters[factor_idx],
		    state.driver_prefix_range_filters[factor_idx], std::move(final_min_max));
	}
	state.driver_filter_keys.reset();
}

SinkFinalizeType PhysicalFactorizedGroupJoinSink::Finalize(Pipeline &, Event &, ClientContext &context,
                                                           OperatorSinkFinalizeInput &input) const {
	auto &global_state = input.global_state.Cast<FactorizedGroupJoinGlobalSinkState>();
	if (source_idx > FactorizedSourceIndex(FactorizedAggregateSource::DRIVER) && global_state.global_filter_state) {
		D_ASSERT(global_state.global_filter_state);
		auto final_min_max =
		    op.factor_filter_pushdown[source_idx - 1]->FinalizeMinMax(*global_state.global_filter_state);
		vector<string> key_names;
		key_names.reserve(op.key_types.size());
		for (idx_t key_idx = 0; key_idx < op.key_types.size(); key_idx++) {
			key_names.push_back(StringUtil::Format("factor_key_%llu", key_idx));
		}
		vector<unique_ptr<BloomFilter>> bloom_filters(op.key_types.size());
		vector<unique_ptr<PrefixRangeFilter>> prefix_range_filters(op.key_types.size());
		op.factor_filter_pushdown[source_idx - 1]->FinalizeGroupJoinFilters(
		    context, op, op.key_types, key_names, bloom_filters, prefix_range_filters, std::move(final_min_max));
	}
	if (global_state.external) {
		global_state.finalized = true;
		return SinkFinalizeType::READY;
	}
	if (source_idx != FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
		if (!global_state.direct_factor_updates) {
			global_state.factor_hash_table->Finalize();
			global_state.finalized = true;
			return SinkFinalizeType::READY;
		}
		auto &driver_state = GetFactorizedDriverState(op);
		D_ASSERT(driver_state.finalized);
		lock_guard<mutex> guard(driver_state.lock);
		for (auto &dense_table : global_state.local_dense_tables) {
			dense_table->CombineInto(*driver_state.target, driver_state.group_addresses,
			                         op.source_ranges[source_idx].begin);
		}
		global_state.local_dense_tables.clear();
		for (auto &local_table : global_state.local_tables) {
			MergeFactorizedLocalTable(op, source_idx, driver_state, *local_table, context);
		}
		global_state.local_tables.clear();
		MergeFactorizedDistinct(op, source_idx, global_state, driver_state, context);
		global_state.local_distinct_tables.clear();
		UpdateFactorizedNullExtendedRows(op, source_idx, driver_state, context);
		global_state.finalized = true;
		return SinkFinalizeType::READY;
	}
	{
		if (global_state.parallel_driver) {
			const auto group_count = global_state.next_group_id.load(std::memory_order_relaxed);
			if (group_count != global_state.input_rows.load(std::memory_order_relaxed)) {
				throw InternalException("Factorized GroupJoin parallel driver build lost a unique row");
			}
			global_state.target->PrepareUniqueFinalize(group_count);
			global_state.group_addresses.resize(group_count, nullptr);
			vector<data_ptr_t> finalized_addresses;
			for (idx_t partition_idx = 0; partition_idx < global_state.target->GetPartitionedData().PartitionCount();
			     partition_idx++) {
				global_state.target->FinalizeUniquePartition(partition_idx, finalized_addresses, false);
				for (auto address : finalized_addresses) {
					auto group_id = LoadFactorizedGroupJoinId(global_state.target->GetLayout(), address);
					if (group_id >= global_state.group_addresses.size() || global_state.group_addresses[group_id]) {
						throw InternalException("Factorized GroupJoin parallel driver identifier is invalid");
					}
					global_state.group_addresses[group_id] = address;
				}
				finalized_addresses.clear();
			}
			global_state.target->VerifyUniqueFinalize();
			for (auto address : global_state.group_addresses) {
				if (!address) {
					throw InternalException("Factorized GroupJoin parallel driver did not publish every group");
				}
			}
		} else {
			if (op.unique_driver &&
			    global_state.target->Count() != global_state.input_rows.load(std::memory_order_relaxed)) {
				throw InternalException("Factorized GroupJoin driver key was not unique");
			}
			if (global_state.group_addresses.size() != global_state.target->Count()) {
				throw InternalException("Factorized GroupJoin did not publish every driver group address");
			}
		}
		if (op.routed) {
			global_state.route_offsets.reserve(global_state.routes.size() + 1);
			for (auto &route : global_state.routes) {
				global_state.route_offsets.push_back(global_state.route_addresses.size());
				global_state.route_addresses.insert(global_state.route_addresses.end(), route.begin(), route.end());
			}
			global_state.route_offsets.push_back(global_state.route_addresses.size());
			vector<vector<data_ptr_t>>().swap(global_state.routes);
		}
		MergeFactorizedDistinct(op, source_idx, global_state, global_state, context);
		if (!global_state.direct_factor_updates) {
			MergeFactorizedDistinct(op, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR), global_state,
			                        global_state, context);
			MergeFactorizedDistinct(op, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR), global_state,
			                        global_state, context);
			UpdateFactorizedNullExtendedRows(op, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR),
			                                 global_state, context);
			UpdateFactorizedNullExtendedRows(op, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR),
			                                 global_state, context);
		}
		global_state.local_distinct_tables.clear();
		if (op.unique_driver && !global_state.parallel_driver) {
			global_state.target->PrepareUniqueFinalize(global_state.group_addresses.size());
			vector<data_ptr_t> finalized_addresses;
			for (idx_t partition_idx = 0; partition_idx < global_state.target->GetPartitionedData().PartitionCount();
			     partition_idx++) {
				global_state.target->FinalizeUniquePartition(partition_idx, finalized_addresses, false);
				finalized_addresses.clear();
			}
			global_state.target->VerifyUniqueFinalize();
		}
		for (auto &owners : global_state.owners) {
			owners = unique_ptr<atomic<idx_t>[]>(new atomic<idx_t>[global_state.group_addresses.size()]);
			for (idx_t group_id = 0; group_id < global_state.group_addresses.size(); group_id++) {
				owners[group_id].store(0, std::memory_order_relaxed);
			}
		}
		if (global_state.direct_factor_updates) {
			FinalizeFactorizedDriverRuntimeFilters(op, context, global_state);
		}
		global_state.finalized = true;
	}
	return SinkFinalizeType::READY;
}

void PhysicalFactorizedGroupJoin::InitializeSinks() {
	D_ASSERT(children.empty());
	D_ASSERT(driver_input && left_input && right_input);
	driver_sink = &physical_plan.Make<PhysicalFactorizedGroupJoinSink>(*this, *driver_input, idx_t(0));
	left_sink = &physical_plan.Make<PhysicalFactorizedGroupJoinSink>(*this, *left_input, idx_t(1));
	right_sink = &physical_plan.Make<PhysicalFactorizedGroupJoinSink>(*this, *right_input, idx_t(2));
	if (streaming_driver) {
		children.push_back(*driver_input);
		return;
	}
	children.push_back(*driver_sink);
	children.push_back(*left_sink);
	children.push_back(*right_sink);
}

void PhysicalFactorizedGroupJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	D_ASSERT(driver_sink && left_sink && right_sink);
	op_state.reset();
	driver_sink->sink_state.reset();
	left_sink->sink_state.reset();
	right_sink->sink_state.reset();
	if (streaming_driver) {
		D_ASSERT(children.size() == 1);
		auto &left_meta = meta_pipeline.CreateChildMetaPipeline(current, *left_sink, MetaPipelineType::JOIN_BUILD);
		left_meta.Build(left_sink->children[0]);
		auto left_pipeline = left_meta.GetBasePipeline();

		auto &right_meta = meta_pipeline.CreateChildMetaPipeline(current, *right_sink, MetaPipelineType::JOIN_BUILD);
		right_meta.Build(right_sink->children[0]);
		auto right_pipeline = right_meta.GetBasePipeline();

		current.AddDependency(left_pipeline);
		current.AddDependency(right_pipeline);
		meta_pipeline.GetState().AddPipelineOperator(current, *this);
		children[0].get().BuildPipelines(current, meta_pipeline);
		return;
	}
	D_ASSERT(children.size() == SOURCE_COUNT);
	meta_pipeline.GetState().SetPipelineSource(current, *this);
	const auto direct_factor_updates = driver_first && planned_execution_mode != GroupJoinExecutionMode::EXTERNAL;
	if (direct_factor_updates) {
		auto &driver_meta = meta_pipeline.CreateChildMetaPipeline(current, *driver_sink, MetaPipelineType::REGULAR);
		driver_meta.Build(driver_sink->children[0]);
		auto driver_pipeline = driver_meta.GetBasePipeline();

		auto &left_meta = meta_pipeline.CreateChildMetaPipeline(current, *left_sink, MetaPipelineType::JOIN_BUILD);
		left_meta.GetBasePipeline()->AddDependency(driver_pipeline);
		left_meta.Build(left_sink->children[0]);

		auto &right_meta = meta_pipeline.CreateChildMetaPipeline(current, *right_sink, MetaPipelineType::JOIN_BUILD);
		right_meta.GetBasePipeline()->AddDependency(driver_pipeline);
		right_meta.Build(right_sink->children[0]);
		return;
	}
	auto &left_meta = meta_pipeline.CreateChildMetaPipeline(current, *left_sink, MetaPipelineType::JOIN_BUILD);
	left_meta.Build(left_sink->children[0]);
	auto left_pipeline = left_meta.GetBasePipeline();

	auto &right_meta = meta_pipeline.CreateChildMetaPipeline(current, *right_sink, MetaPipelineType::JOIN_BUILD);
	right_meta.Build(right_sink->children[0]);
	auto right_pipeline = right_meta.GetBasePipeline();

	auto &driver_meta = meta_pipeline.CreateChildMetaPipeline(current, *driver_sink, MetaPipelineType::REGULAR);
	driver_meta.GetBasePipeline()->AddDependency(left_pipeline);
	driver_meta.GetBasePipeline()->AddDependency(right_pipeline);
	driver_meta.Build(driver_sink->children[0]);
}

vector<const_reference<PhysicalOperator>> PhysicalFactorizedGroupJoin::GetSources() const {
	if (streaming_driver) {
		D_ASSERT(children.size() == 1);
		return children[0].get().GetSources();
	}
	return {*this};
}

class FactorizedGroupJoinOutputBuffer {
public:
	FactorizedGroupJoinOutputBuffer(ClientContext &context, const TupleDataLayout &layout_p)
	    : layout(layout_p.Copy()), allocator(Allocator::Get(context)), row_state(allocator),
	      addresses(LogicalType::POINTER) {
		if (layout.AggregateCount() > 0) {
			data = make_unsafe_uniq_array_uninitialized<data_t>(layout.GetRowWidth() * STANDARD_VECTOR_SIZE);
		}
	}

	~FactorizedGroupJoinOutputBuffer() {
		Destroy();
	}

	void Initialize(idx_t count) {
		Destroy();
		if (layout.AggregateCount() == 0) {
			return;
		}
		auto address_data = FlatVector::GetDataMutable<data_ptr_t>(addresses);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			address_data[row_idx] = data.get() + row_idx * layout.GetRowWidth();
		}
		FlatVector::SetSize(addresses, count_t(count));
		RowOperations::InitializeStates(layout, addresses, *FlatVector::IncrementalSelectionVector(), count);
		initialized = true;
	}

	void Destroy() {
		if (!initialized) {
			return;
		}
		RowOperations::DestroyStates(row_state, layout, addresses);
		row_state.local_states.clear();
		allocator.Reset();
		initialized = false;
	}

	TupleDataLayout layout;
	ArenaAllocator allocator;
	RowOperationsState row_state;
	Vector addresses;
	unsafe_unique_array<data_t> data;
	bool initialized = false;
};

class FactorizedGroupJoinOperatorState : public OperatorState {
public:
	FactorizedGroupJoinOperatorState(ExecutionContext &context, const PhysicalFactorizedGroupJoin &op)
	    : driver(context, op, FactorizedSourceIndex(FactorizedAggregateSource::DRIVER),
	             op.op_state->Cast<FactorizedGroupJoinGlobalOperatorState>().driver_state,
	             op.op_state->Cast<FactorizedGroupJoinGlobalOperatorState>().driver_state, true),
	      selected_addresses(LogicalType::POINTER), selected_rows(STANDARD_VECTOR_SIZE),
	      left_multiplicities(LogicalType::BIGINT), right_multiplicities(LogicalType::BIGINT),
	      product_multiplicities(LogicalType::BIGINT),
	      unit_multiplicities(Value::BIGINT(1), count_t(STANDARD_VECTOR_SIZE)),
	      output(context.client, op.target_layout) {
	}

	FactorizedGroupJoinLocalSinkState driver;
	Vector selected_addresses;
	SelectionVector selected_rows;
	Vector left_multiplicities;
	Vector right_multiplicities;
	Vector product_multiplicities;
	Vector unit_multiplicities;
	FactorizedGroupJoinOutputBuffer output;
	InterruptState interrupt_state;
};

static void FinalizeStreamingFactorizedDistinct(const PhysicalFactorizedGroupJoin &op,
                                                FactorizedGroupJoinLocalSinkState &state,
                                                GroupedAggregateHashTable &target, ClientContext &context) {
	if (op.distinct_aggregates.empty()) {
		return;
	}
	state.group_ids.data[0].Flatten();
	state.addresses.Flatten();
	auto group_ids = FlatVector::GetData<uint64_t>(state.group_ids.data[0]);
	auto addresses = FlatVector::GetData<data_ptr_t>(state.addresses);
	unordered_map<uint64_t, data_ptr_t> target_by_group;
	for (idx_t row_idx = 0; row_idx < state.group_ids.size(); row_idx++) {
		target_by_group.emplace(group_ids[row_idx], addresses[row_idx]);
	}

	Vector target_addresses(LogicalType::POINTER);
	for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		auto &table = state.distinct_tables[distinct_idx];
		if (!table || table->Count() == 0) {
			continue;
		}
		vector<LogicalType> distinct_types {LogicalType::UBIGINT};
		distinct_types.insert(distinct_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
		DataChunk distinct_rows;
		distinct_rows.Initialize(Allocator::Get(context), distinct_types);
		DataChunk payload;
		payload.Initialize(Allocator::Get(context), op.source_argument_types[distinct.source_idx]);
		unsafe_vector<idx_t> distinct_filter {distinct.range_index};
		auto &range = op.source_ranges[distinct.source_idx];
		AggregateHTScanState scan;
		table->InitializeScan(scan);
		while (table->ScanGroups(scan, distinct_rows)) {
			distinct_rows.Flatten();
			auto ids = FlatVector::GetData<uint64_t>(distinct_rows.data[0]);
			auto targets = FlatVector::GetDataMutable<data_ptr_t>(target_addresses);
			payload.Reset();
			for (idx_t payload_idx = 0; payload_idx < op.source_argument_types[distinct.source_idx].size();
			     payload_idx++) {
				payload.data[payload_idx].Reference(Value(op.source_argument_types[distinct.source_idx][payload_idx]),
				                                    count_t(distinct_rows.size()));
			}
			for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
				payload.data[distinct.payload_index + child_idx].Reference(distinct_rows.data[child_idx + 1]);
			}
			if (distinct.filter_index.IsValid()) {
				payload.data[distinct.filter_index.GetIndex()].Reference(Value::BOOLEAN(true),
				                                                         count_t(distinct_rows.size()));
			}
			payload.SetChildCardinality(distinct_rows.size());
			for (idx_t row_idx = 0; row_idx < distinct_rows.size(); row_idx++) {
				auto target_entry = target_by_group.find(ids[row_idx]);
				if (target_entry == target_by_group.end()) {
					throw InternalException("Factorized GroupJoin streaming distinct group was not found");
				}
				targets[row_idx] = target_entry->second;
			}
			FlatVector::SetSize(target_addresses, distinct_rows.size());
			target.UpdateAggregatesAtAddressesRange(*state.update_state, target_addresses, payload, range.begin,
			                                        range.count, distinct_filter);
		}
	}
}

static void UpdateStreamingFactorizedNullExtendedRows(const PhysicalFactorizedGroupJoin &op, idx_t source_idx,
                                                      FactorizedGroupJoinLocalSinkState &state,
                                                      GroupedAggregateHashTable &target, ClientContext &context) {
	if ((source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR) && !op.preserve_left) ||
	    (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR) && !op.preserve_right)) {
		return;
	}
	auto &range = op.source_ranges[source_idx];
	D_ASSERT(range.multiplicity_index.IsValid());
	state.addresses.Flatten();
	auto source_addresses = FlatVector::GetData<data_ptr_t>(state.addresses);
	Vector unmatched_addresses(LogicalType::POINTER);
	auto unmatched = FlatVector::GetDataMutable<data_ptr_t>(unmatched_addresses);
	idx_t unmatched_count = 0;
	for (idx_t row_idx = 0; row_idx < state.group_ids.size(); row_idx++) {
		if (LoadFactorizedCount(target.GetLayout(), source_addresses[row_idx], range.multiplicity_index.GetIndex()) ==
		    0) {
			unmatched[unmatched_count++] = source_addresses[row_idx];
		}
	}
	if (unmatched_count == 0) {
		return;
	}
	FlatVector::SetSize(unmatched_addresses, unmatched_count);

	ExpressionExecutor executor(context);
	for (auto &expression : op.source_arguments[source_idx]) {
		executor.AddExpression(*expression);
	}
	DataChunk null_input;
	null_input.Initialize(Allocator::Get(context), op.source_input_types[source_idx]);
	for (idx_t column_idx = 0; column_idx < op.source_input_types[source_idx].size(); column_idx++) {
		null_input.data[column_idx].Reference(Value(op.source_input_types[source_idx][column_idx]),
		                                      count_t(unmatched_count));
	}
	null_input.SetChildCardinality(unmatched_count);
	DataChunk null_payload;
	null_payload.Initialize(Allocator::Get(context), op.source_argument_types[source_idx]);
	if (op.source_argument_types[source_idx].empty()) {
		null_payload.SetChildCardinality(unmatched_count);
	} else {
		executor.Execute(null_input, null_payload);
	}

	auto source_aggregates = op.CreateSourceAggregates(source_idx);
	unsafe_vector<idx_t> aggregate_filter;
	for (idx_t range_idx = 0; range_idx < range.count; range_idx++) {
		if (source_aggregates[range_idx].aggr_type != AggregateType::DISTINCT) {
			aggregate_filter.push_back(range_idx);
		}
	}
	AggregateFilterDataSet filter_set;
	filter_set.Initialize(context, source_aggregates, op.source_argument_types[source_idx]);
	target.UpdateAggregatesAtAddressesRangeWithFilterSet(*state.update_state, unmatched_addresses, null_payload,
	                                                     range.begin, range.count, aggregate_filter, filter_set);
	for (auto &distinct : op.distinct_aggregates) {
		if (distinct.source_idx != source_idx) {
			continue;
		}
		idx_t distinct_count = unmatched_count;
		optional_ptr<DataChunk> distinct_payload = null_payload;
		Vector distinct_addresses(LogicalType::POINTER);
		if (distinct.filter_index.IsValid()) {
			auto &filter_data = filter_set.GetFilterData(distinct.range_index);
			distinct_count = filter_data.ApplyFilter(null_payload);
			if (distinct_count == 0) {
				continue;
			}
			distinct_addresses.Slice(unmatched_addresses, filter_data.true_sel, distinct_count);
			distinct_payload = filter_data.filtered_payload;
		} else {
			distinct_addresses.Reference(unmatched_addresses);
		}
		distinct_addresses.Flatten();
		unsafe_vector<idx_t> distinct_filter {distinct.range_index};
		target.UpdateAggregatesAtAddressesRange(*state.update_state, distinct_addresses, *distinct_payload, range.begin,
		                                        range.count, distinct_filter);
	}
}

unique_ptr<GlobalOperatorState> PhysicalFactorizedGroupJoin::GetGlobalOperatorState(ClientContext &context) const {
	if (!streaming_driver) {
		return PhysicalOperator::GetGlobalOperatorState(context);
	}
	return make_uniq<FactorizedGroupJoinGlobalOperatorState>(*this, context);
}

unique_ptr<OperatorState> PhysicalFactorizedGroupJoin::GetOperatorState(ExecutionContext &context) const {
	if (!streaming_driver) {
		return PhysicalOperator::GetOperatorState(context);
	}
	return make_uniq<FactorizedGroupJoinOperatorState>(context, *this);
}

OperatorResultType PhysicalFactorizedGroupJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                        GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	if (!streaming_driver) {
		throw InternalException("Materialized factorized GroupJoin cannot execute as an operator");
	}
	auto &gstate = gstate_p.Cast<FactorizedGroupJoinGlobalOperatorState>();
	auto &state = state_p.Cast<FactorizedGroupJoinOperatorState>();
	auto &driver = state.driver;
	auto &target = *driver.local_driver_target;
	target.ResetForNewIteration(0);
	target.SkipLookups();
	driver.update_state = make_uniq<AggregateHTUpdateState>(target);
	for (auto &distinct_table : driver.distinct_tables) {
		if (distinct_table) {
			distinct_table->ResetForNewIteration(0);
		}
	}
	chunk.Reset();

	OperatorSinkInput sink_input {gstate.driver_state, driver, state.interrupt_state};
	auto sink_result = driver_sink->Sink(context, input, sink_input);
	D_ASSERT(sink_result == SinkResultType::NEED_MORE_INPUT);
	if (input.size() == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}
	FinalizeStreamingFactorizedDistinct(*this, driver, target, context.client);
	UpdateStreamingFactorizedNullExtendedRows(*this, FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR),
	                                          driver, target, context.client);
	UpdateStreamingFactorizedNullExtendedRows(*this, FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR),
	                                          driver, target, context.client);

	auto layout = target.GetLayoutPtr();
	auto &left_range = source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)];
	auto &right_range = source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR)];
	D_ASSERT(unique_driver && left_range.multiplicity_index.IsValid() && right_range.multiplicity_index.IsValid());

	driver.addresses.Flatten();
	auto addresses = FlatVector::GetData<data_ptr_t>(driver.addresses);
	auto left_values = FlatVector::GetDataMutable<int64_t>(state.left_multiplicities);
	auto right_values = FlatVector::GetDataMutable<int64_t>(state.right_multiplicities);
	auto product_values = FlatVector::GetDataMutable<int64_t>(state.product_multiplicities);
	idx_t result_count = 0;
	for (idx_t row_idx = 0; row_idx < input.size(); row_idx++) {
		auto left_count = LoadFactorizedCount(*layout, addresses[row_idx], left_range.multiplicity_index.GetIndex());
		auto right_count = LoadFactorizedCount(*layout, addresses[row_idx], right_range.multiplicity_index.GetIndex());
		if (semi_left && left_count != 0) {
			left_count = 1;
		}
		if (semi_right && right_count != 0) {
			right_count = 1;
		}
		if ((!preserve_left && left_count == 0) || (!preserve_right && right_count == 0)) {
			continue;
		}
		left_count = left_count == 0 ? 1 : left_count;
		right_count = right_count == 0 ? 1 : right_count;
		idx_t product;
		if (!TryMultiplyOperator::Operation(left_count, right_count, product)) {
			throw OutOfRangeException("Overflow in factorized GroupJoin multiplicity");
		}
		state.selected_rows.set_index(result_count, row_idx);
		left_values[result_count] = CheckedFactorizedMultiplicity(right_count);
		right_values[result_count] = CheckedFactorizedMultiplicity(left_count);
		product_values[result_count] = CheckedFactorizedMultiplicity(product);
		result_count++;
	}
	if (result_count == 0) {
		return OperatorResultType::NEED_MORE_INPUT;
	}

	FlatVector::SetSize(state.left_multiplicities, count_t(result_count));
	FlatVector::SetSize(state.right_multiplicities, count_t(result_count));
	FlatVector::SetSize(state.product_multiplicities, count_t(result_count));
	state.selected_addresses.Slice(driver.addresses, state.selected_rows, result_count);
	state.selected_addresses.Flatten();
	state.output.Initialize(result_count);
	try {
		for (idx_t aggregate_idx = 0; aggregate_idx < aggregate_expressions.size(); aggregate_idx++) {
			auto source_idx = FactorizedSourceIndex(aggregate_sources[aggregate_idx]);
			Vector *multiplicities;
			if (aggregate_expressions[aggregate_idx]->Cast<BoundAggregateExpression>().IsDistinct()) {
				multiplicities = &state.unit_multiplicities;
			} else {
				switch (source_idx) {
				case 0:
					multiplicities = &state.product_multiplicities;
					break;
				case 1:
					multiplicities = &state.left_multiplicities;
					break;
				case 2:
					multiplicities = &state.right_multiplicities;
					break;
				default:
					throw InternalException("Invalid factorized aggregate source");
				}
			}
			RowOperations::CombineStatesRange(state.output.row_state, *layout, state.selected_addresses,
			                                  partial_indexes[aggregate_idx], state.output.layout,
			                                  state.output.addresses, aggregate_idx, 1, *multiplicities,
			                                  AggregateCombineType::PRESERVE_INPUT);
		}
		for (idx_t group_idx = 0; group_idx < output_group_key_indices.size(); group_idx++) {
			chunk.data[group_idx].Slice(driver.keys.data[output_group_key_indices[group_idx]], state.selected_rows,
			                            result_count);
		}
		chunk.SetChildCardinality(result_count);
		RowOperations::FinalizeStates(state.output.row_state, state.output.layout, state.output.addresses, chunk,
		                              output_group_key_indices.size());
	} catch (...) {
		state.output.Destroy();
		throw;
	}
	state.output.Destroy();
	return OperatorResultType::NEED_MORE_INPUT;
}

class ExternalFactorizedDistinctState {
public:
	ExternalFactorizedDistinctState(const PhysicalFactorizedGroupJoin &op, ClientContext &context) {
		raw_rows.resize(op.distinct_aggregates.size());
		raw_append_states.resize(op.distinct_aggregates.size());
		distinct_groups.resize(op.distinct_aggregates.size());
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			auto &distinct = op.distinct_aggregates[distinct_idx];
			vector<LogicalType> distinct_types {LogicalType::UBIGINT};
			distinct_types.insert(distinct_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
			raw_rows[distinct_idx] = make_uniq<ColumnDataCollection>(context, distinct_types);
			raw_append_states[distinct_idx] = make_uniq<ColumnDataAppendState>();
			raw_rows[distinct_idx]->InitializeAppend(*raw_append_states[distinct_idx]);
			distinct_groups[distinct_idx] = make_uniq<DataChunk>();
			distinct_groups[distinct_idx]->InitializeEmpty(distinct_types);
		}
	}

	void Sink(const PhysicalFactorizedGroupJoin &op, idx_t source_idx, AggregateFilterDataSet &filter_set,
	          DataChunk &group_ids, DataChunk &payload) {
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			auto &distinct = op.distinct_aggregates[distinct_idx];
			if (distinct.source_idx != source_idx) {
				continue;
			}
			auto &groups = *distinct_groups[distinct_idx];
			groups.Reset();
			idx_t count;
			optional_ptr<DataChunk> argument_payload;
			if (distinct.filter_index.IsValid()) {
				auto &filter_data = filter_set.GetFilterData(distinct.range_index);
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
			for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
				groups.data[child_idx + 1].Reference(argument_payload->data[distinct.payload_index + child_idx]);
			}
			groups.SetChildCardinality(count);
			raw_rows[distinct_idx]->Append(*raw_append_states[distinct_idx], groups);
		}
	}

	void Finalize(const PhysicalFactorizedGroupJoin &op, GroupedAggregateHashTable &target,
	              const vector<data_ptr_t> &group_addresses, ClientContext &context, idx_t partition_budget) {
		auto allocator = make_shared_ptr<ArenaAllocator>(Allocator::Get(context));
		target.StoreAggregateAllocator(allocator);
		AggregateHTUpdateState update_state(target, allocator);
		for (idx_t distinct_idx = 0; distinct_idx < op.distinct_aggregates.size(); distinct_idx++) {
			if (raw_rows[distinct_idx]->Count() == 0) {
				continue;
			}
			FinalizeAggregate(op, distinct_idx, target, group_addresses, context, update_state, partition_budget);
		}
	}

private:
	struct DistinctPartitionTask {
		unique_ptr<ColumnDataCollection> rows;
		idx_t radix_bits;
	};

	static unique_ptr<RadixPartitionedColumnData> PartitionRows(ClientContext &context, ColumnDataCollection &rows,
	                                                            idx_t radix_bits, bool has_hash) {
		auto row_types = rows.Types();
		auto partition_types = row_types;
		if (!has_hash) {
			partition_types.push_back(LogicalType::HASH);
		}
		auto result =
		    make_uniq<RadixPartitionedColumnData>(context, partition_types, radix_bits, partition_types.size() - 1);
		auto local = result->CreateShared();
		PartitionedColumnDataAppendState append_state;
		local->InitializeAppendState(append_state);
		DataChunk rows_chunk;
		rows.InitializeScanChunk(Allocator::Get(context), rows_chunk);
		ColumnDataScanState scan;
		rows.InitializeScan(scan);
		DataChunk partition_chunk;
		partition_chunk.InitializeEmpty(partition_types);
		Vector hashes(LogicalType::HASH);
		while (rows.Scan(scan, rows_chunk)) {
			context.InterruptCheck();
			partition_chunk.Reset();
			for (idx_t column_idx = 0; column_idx < row_types.size(); column_idx++) {
				partition_chunk.data[column_idx].Reference(rows_chunk.data[column_idx]);
			}
			if (!has_hash) {
				rows_chunk.Hash(hashes);
				partition_chunk.data.back().Reference(hashes);
			}
			partition_chunk.SetChildCardinality(rows_chunk.size());
			local->Append(append_state, partition_chunk);
		}
		local->FlushAppendState(append_state);
		result->Combine(*local);
		return result;
	}

	static idx_t EstimatePartitionSize(ColumnDataCollection &rows) {
		idx_t result = EstimateFactorizedCollectionSize(rows);
		auto pointer_table_size = GroupedAggregateHashTable::GetCapacityForCount(rows.Count()) * sizeof(ht_entry_t);
		if (!TryAddOperator::Operation(result, pointer_table_size, result)) {
			return NumericLimits<idx_t>::Maximum();
		}
		return result;
	}

	void ProcessPartition(const PhysicalFactorizedGroupJoin &op, idx_t distinct_idx, GroupedAggregateHashTable &target,
	                      const vector<data_ptr_t> &group_addresses, ClientContext &context,
	                      AggregateHTUpdateState &update_state, ColumnDataCollection &partition) {
		auto &distinct = op.distinct_aggregates[distinct_idx];
		vector<LogicalType> group_types {LogicalType::UBIGINT};
		group_types.insert(group_types.end(), distinct.argument_types.begin(), distinct.argument_types.end());
		auto table = make_uniq<GroupedAggregateHashTable>(
		    context, BufferAllocator::Get(context), group_types, vector<LogicalType> {}, vector<AggregateObject> {},
		    GroupedAggregateHashTable::InitialCapacity(), FACTORIZED_GROUP_JOIN_RADIX_BITS,
		    TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		DataChunk rows;
		partition.InitializeScanChunk(Allocator::Get(context), rows);
		ColumnDataScanState scan;
		partition.InitializeScan(scan);
		DataChunk groups;
		groups.InitializeEmpty(group_types);
		Vector insert_addresses(LogicalType::POINTER);
		SelectionVector new_groups(STANDARD_VECTOR_SIZE);
		Vector target_addresses(LogicalType::POINTER);
		DataChunk payload;
		payload.Initialize(Allocator::Get(context), op.source_argument_types[distinct.source_idx]);
		unsafe_vector<idx_t> distinct_filter {distinct.range_index};
		auto &range = op.source_ranges[distinct.source_idx];
		while (partition.Scan(scan, rows)) {
			context.InterruptCheck();
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
					throw InternalException("External factorized distinct identifier exceeds group count");
				}
				targets[new_idx] = group_addresses[group_id];
			}
			FlatVector::SetSize(target_addresses, new_count);
			payload.Reset();
			for (idx_t payload_idx = 0; payload_idx < payload.ColumnCount(); payload_idx++) {
				payload.data[payload_idx].Reference(Value(op.source_argument_types[distinct.source_idx][payload_idx]),
				                                    count_t(new_count));
			}
			for (idx_t child_idx = 0; child_idx < distinct.argument_types.size(); child_idx++) {
				payload.data[distinct.payload_index + child_idx].Slice(groups.data[child_idx + 1], new_groups,
				                                                       new_count);
			}
			if (distinct.filter_index.IsValid()) {
				payload.data[distinct.filter_index.GetIndex()].Reference(Value::BOOLEAN(true), count_t(new_count));
			}
			payload.SetChildCardinality(new_count);
			target.UpdateAggregatesAtAddressesRange(update_state, target_addresses, payload, range.begin, range.count,
			                                        distinct_filter);
		}
	}

	void FinalizeAggregate(const PhysicalFactorizedGroupJoin &op, idx_t distinct_idx, GroupedAggregateHashTable &target,
	                       const vector<data_ptr_t> &group_addresses, ClientContext &context,
	                       AggregateHTUpdateState &update_state, idx_t partition_budget) {
		auto partitions =
		    PartitionRows(context, *raw_rows[distinct_idx], FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS, false);
		raw_rows[distinct_idx].reset();
		raw_append_states[distinct_idx].reset();
		vector<DistinctPartitionTask> tasks;
		for (auto &partition : partitions->GetPartitions()) {
			if (partition && partition->Count() != 0) {
				tasks.push_back({std::move(partition), FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS});
			}
		}
		while (!tasks.empty()) {
			context.InterruptCheck();
			auto task = std::move(tasks.back());
			tasks.pop_back();
			if (task.radix_bits < RadixPartitioning::MAX_RADIX_BITS &&
			    EstimatePartitionSize(*task.rows) > partition_budget) {
				auto radix_bits = MinValue<idx_t>(task.radix_bits + 2, RadixPartitioning::MAX_RADIX_BITS);
				auto repartitioned = PartitionRows(context, *task.rows, radix_bits, true);
				for (auto &partition : repartitioned->GetPartitions()) {
					if (partition && partition->Count() != 0) {
						tasks.push_back({std::move(partition), radix_bits});
					}
				}
				continue;
			}
			ProcessPartition(op, distinct_idx, target, group_addresses, context, update_state, *task.rows);
		}
	}

	vector<unique_ptr<ColumnDataCollection>> raw_rows;
	vector<unique_ptr<ColumnDataAppendState>> raw_append_states;
	vector<unique_ptr<DataChunk>> distinct_groups;
};

static unsafe_vector<idx_t> GetExternalFactorizedAggregateFilter(const PhysicalFactorizedGroupJoin &op,
                                                                 idx_t source_idx) {
	auto source_aggregates = op.CreateSourceAggregates(source_idx);
	unsafe_vector<idx_t> result;
	for (idx_t aggregate_idx = 0; aggregate_idx < source_aggregates.size(); aggregate_idx++) {
		if (source_aggregates[aggregate_idx].aggr_type != AggregateType::DISTINCT) {
			result.push_back(aggregate_idx);
		}
	}
	return result;
}

class ExternalFactorizedPartitionProcessor {
public:
	ExternalFactorizedPartitionProcessor(const PhysicalFactorizedGroupJoin &op_p, ClientContext &context_p,
	                                     idx_t partition_budget_p)
	    : op(op_p), context(context_p), partition_budget(partition_budget_p), distinct(op, context) {
		target = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.group_types,
		                                              vector<LogicalType> {}, op.CreateHashTableAggregates(),
		                                              GroupedAggregateHashTable::InitialCapacity(), idx_t(0),
		                                              TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		target->StoreAggregateAllocator(make_shared_ptr<ArenaAllocator>(Allocator::Get(context)));
		update_state = make_uniq<AggregateHTUpdateState>(*target);
		if (op.routed) {
			routing_table = make_uniq<GroupedAggregateHashTable>(
			    context, BufferAllocator::Get(context), op.key_types, vector<LogicalType> {},
			    vector<AggregateObject> {CreateFactorizedGroupJoinIdAggregate()},
			    GroupedAggregateHashTable::InitialCapacity(), idx_t(0), TupleDataValidityType::CAN_HAVE_NULL_VALUES);
		}
		for (idx_t source_idx = 0; source_idx < PhysicalFactorizedGroupJoin::SOURCE_COUNT; source_idx++) {
			auto filter_set = make_uniq<AggregateFilterDataSet>();
			filter_set->Initialize(context, op.CreateSourceAggregates(source_idx),
			                       op.source_argument_types[source_idx]);
			filter_sets.push_back(std::move(filter_set));
			aggregate_filters.push_back(GetExternalFactorizedAggregateFilter(op, source_idx));
		}
	}

	unique_ptr<ColumnDataCollection> Process(unique_ptr<ColumnDataCollection> driver,
	                                         unique_ptr<ColumnDataCollection> left_factor,
	                                         unique_ptr<ColumnDataCollection> right_factor) {
		BuildDriver(*driver);
		driver.reset();
		if (left_factor) {
			ProcessFactor(FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR), *left_factor);
			left_factor.reset();
		}
		if (right_factor) {
			ProcessFactor(FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR), *right_factor);
			right_factor.reset();
		}
		UpdateNullExtended(FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR));
		UpdateNullExtended(FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR));
		distinct.Finalize(op, *target, group_addresses, context, partition_budget);
		return BuildOutput();
	}

	idx_t MatchedRows(idx_t source_idx) const {
		D_ASSERT(source_idx > 0 && source_idx < PhysicalFactorizedGroupJoin::SOURCE_COUNT);
		return matched_rows[source_idx - 1];
	}

private:
	void BuildDriver(ColumnDataCollection &driver) {
		const auto source_idx = FactorizedSourceIndex(FactorizedAggregateSource::DRIVER);
		const auto key_count = op.key_types.size();
		const auto group_count = op.group_types.size();
		DataChunk rows;
		driver.InitializeScanChunk(Allocator::Get(context), rows);
		ColumnDataScanState scan;
		driver.InitializeScan(scan);
		DataChunk keys;
		keys.InitializeEmpty(op.key_types);
		DataChunk groups;
		groups.InitializeEmpty(op.group_types);
		DataChunk arguments;
		arguments.InitializeEmpty(op.source_argument_types[source_idx]);
		Vector addresses(LogicalType::POINTER);
		SelectionVector new_groups(STANDARD_VECTOR_SIZE);
		Vector route_addresses(LogicalType::POINTER);
		SelectionVector new_routes(STANDARD_VECTOR_SIZE);
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		auto &range = op.source_ranges[source_idx];
		while (driver.Scan(scan, rows)) {
			context.InterruptCheck();
			keys.Reset();
			for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
				keys.data[key_idx].Reference(rows.data[key_idx]);
			}
			keys.SetChildCardinality(rows.size());
			groups.Reset();
			for (idx_t group_idx = 0; group_idx < group_count; group_idx++) {
				groups.data[group_idx].Reference(rows.data[key_count + group_idx]);
			}
			groups.SetChildCardinality(rows.size());
			arguments.Reset();
			for (idx_t argument_idx = 0; argument_idx < arguments.ColumnCount(); argument_idx++) {
				arguments.data[argument_idx].Reference(rows.data[key_count + group_count + argument_idx]);
			}
			arguments.SetChildCardinality(rows.size());
			auto new_count = target->FindOrCreateGroups(groups, addresses, new_groups);
			if (op.unique_driver && new_count != rows.size()) {
				throw InternalException("External factorized GroupJoin driver uniqueness proof was violated");
			}
			auto address_data = FlatVector::GetData<data_ptr_t>(addresses);
			for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
				auto row_idx = new_groups.get_index_unsafe(new_idx);
				auto group_id = group_addresses.size();
				StoreFactorizedGroupJoinId(target->GetLayout(), address_data[row_idx], group_id);
				group_addresses.push_back(address_data[row_idx]);
			}
			if (op.routed) {
				auto route_count = routing_table->FindOrCreateGroups(keys, route_addresses, new_routes);
				auto route_data = FlatVector::GetData<data_ptr_t>(route_addresses);
				for (idx_t new_idx = 0; new_idx < route_count; new_idx++) {
					auto row_idx = new_routes.get_index_unsafe(new_idx);
					auto route_id = routes.size();
					StoreFactorizedGroupJoinId(routing_table->GetLayout(), route_data[row_idx], route_id);
					routes.emplace_back();
				}
				for (idx_t new_idx = 0; new_idx < new_count; new_idx++) {
					auto row_idx = new_groups.get_index_unsafe(new_idx);
					auto route_id = LoadFactorizedGroupJoinId(routing_table->GetLayout(), route_data[row_idx]);
					routes[route_id].push_back(address_data[row_idx]);
				}
			}

			group_ids.Reset();
			group_ids.SetChildCardinality(rows.size());
			group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
			for (idx_t row_idx = 0; row_idx < rows.size(); row_idx++) {
				ids[row_idx] = LoadFactorizedGroupJoinId(target->GetLayout(), address_data[row_idx]);
			}
			FlatVector::SetSize(group_ids.data[0], rows.size());
			distinct.Sink(op, source_idx, *filter_sets[source_idx], group_ids, arguments);
			target->UpdateAggregatesAtAddressesRangeWithFilterSet(*update_state, addresses, arguments, range.begin,
			                                                      range.count, aggregate_filters[source_idx],
			                                                      *filter_sets[source_idx]);
		}
	}

	void UpdateFactorBatch(idx_t source_idx, Vector &addresses, DataChunk &group_ids, DataChunk &arguments) {
		auto &range = op.source_ranges[source_idx];
		distinct.Sink(op, source_idx, *filter_sets[source_idx], group_ids, arguments);
		target->UpdateAggregatesAtAddressesRangeWithFilterSet(*update_state, addresses, group_ids.data[0], arguments,
		                                                      range.begin, range.count, aggregate_filters[source_idx],
		                                                      *filter_sets[source_idx]);
	}

	void ProcessFactor(idx_t source_idx, ColumnDataCollection &factor) {
		const auto key_count = op.key_types.size();
		DataChunk rows;
		factor.InitializeScanChunk(Allocator::Get(context), rows);
		ColumnDataScanState scan;
		factor.InitializeScan(scan);
		DataChunk keys;
		keys.InitializeEmpty(op.key_types);
		DataChunk arguments;
		arguments.InitializeEmpty(op.source_argument_types[source_idx]);
		DataChunk selected_keys;
		selected_keys.InitializeEmpty(op.key_types);
		DataChunk selected_arguments;
		selected_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
		SelectionVector non_null(STANDARD_VECTOR_SIZE);
		vector<UnifiedVectorFormat> key_formats(op.key_types.size());
		AggregateHTLookupState lookup_state;
		SelectionVector found(STANDARD_VECTOR_SIZE);
		Vector matched_addresses(LogicalType::POINTER);
		DataChunk matched_arguments;
		matched_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		Vector routed_addresses(LogicalType::POINTER);
		DataChunk routed_group_ids;
		routed_group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		SelectionVector routed_input(STANDARD_VECTOR_SIZE);
		while (factor.Scan(scan, rows)) {
			context.InterruptCheck();
			keys.Reset();
			for (idx_t key_idx = 0; key_idx < key_count; key_idx++) {
				keys.data[key_idx].Reference(rows.data[key_idx]);
			}
			keys.SetChildCardinality(rows.size());
			arguments.Reset();
			for (idx_t argument_idx = 0; argument_idx < arguments.ColumnCount(); argument_idx++) {
				arguments.data[argument_idx].Reference(rows.data[key_count + argument_idx]);
			}
			arguments.SetChildCardinality(rows.size());
			auto non_null_count = SelectNonNullFactorizedKeys(keys, key_formats, non_null);
			if (non_null_count == 0) {
				continue;
			}
			selected_keys.Reset();
			selected_arguments.Reset();
			if (non_null_count == rows.size()) {
				selected_keys.Reference(keys);
				selected_arguments.Reference(arguments);
			} else {
				selected_keys.Slice(keys, non_null, non_null_count);
				if (arguments.ColumnCount() == 0) {
					selected_arguments.SetChildCardinality(non_null_count);
				} else {
					selected_arguments.Slice(arguments, non_null, non_null_count);
				}
			}
			auto &lookup_table = op.routed ? *routing_table : *target;
			auto found_count = lookup_table.LookupGroups(selected_keys, lookup_state, found);
			if (found_count == 0) {
				continue;
			}
			matched_rows[source_idx - 1] += found_count;
			matched_addresses.SetVectorType(VectorType::FLAT_VECTOR);
			auto matched_data = FlatVector::GetDataMutable<data_ptr_t>(matched_addresses);
			auto lookup_data = FlatVector::GetData<data_ptr_t>(lookup_state.addresses);
			group_ids.Reset();
			group_ids.SetChildCardinality(found_count);
			group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
			for (idx_t match_idx = 0; match_idx < found_count; match_idx++) {
				auto input_idx = found.get_index_unsafe(match_idx);
				matched_data[match_idx] = lookup_data[input_idx];
				ids[match_idx] = LoadFactorizedGroupJoinId(lookup_table.GetLayout(), lookup_data[input_idx]);
			}
			FlatVector::SetSize(matched_addresses, found_count);
			FlatVector::SetSize(group_ids.data[0], found_count);
			matched_arguments.Reset();
			if (selected_arguments.ColumnCount() == 0) {
				matched_arguments.SetChildCardinality(found_count);
			} else {
				matched_arguments.Slice(selected_arguments, found, found_count);
			}
			if (!op.routed) {
				UpdateFactorBatch(source_idx, matched_addresses, group_ids, matched_arguments);
				continue;
			}

			auto routed_data = FlatVector::GetDataMutable<data_ptr_t>(routed_addresses);
			auto routed_ids = FlatVector::GetDataMutable<uint64_t>(routed_group_ids.data[0]);
			idx_t routed_count = 0;
			auto flush = [&]() {
				if (routed_count == 0) {
					return;
				}
				FlatVector::SetSize(routed_addresses, routed_count);
				FlatVector::SetSize(routed_group_ids.data[0], routed_count);
				routed_group_ids.SetChildCardinality(routed_count);
				DataChunk routed_arguments;
				routed_arguments.InitializeEmpty(op.source_argument_types[source_idx]);
				if (matched_arguments.ColumnCount() == 0) {
					routed_arguments.SetChildCardinality(routed_count);
				} else {
					routed_arguments.Slice(matched_arguments, routed_input, routed_count);
				}
				UpdateFactorBatch(source_idx, routed_addresses, routed_group_ids, routed_arguments);
				routed_count = 0;
			};
			for (idx_t match_idx = 0; match_idx < found_count; match_idx++) {
				auto route_id = NumericCast<idx_t>(ids[match_idx]);
				if (route_id >= routes.size()) {
					throw InternalException("External factorized route identifier exceeds route count");
				}
				for (auto target_address : routes[route_id]) {
					routed_data[routed_count] = target_address;
					routed_ids[routed_count] = LoadFactorizedGroupJoinId(target->GetLayout(), target_address);
					routed_input.set_index(routed_count, match_idx);
					if (++routed_count == STANDARD_VECTOR_SIZE) {
						flush();
					}
				}
			}
			flush();
		}
	}

	void UpdateNullExtended(idx_t source_idx) {
		if ((source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR) && !op.preserve_left) ||
		    (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR) && !op.preserve_right)) {
			return;
		}
		auto &range = op.source_ranges[source_idx];
		ExpressionExecutor executor(context);
		for (auto &expression : op.source_arguments[source_idx]) {
			executor.AddExpression(*expression);
		}
		DataChunk null_input;
		null_input.Initialize(Allocator::Get(context), op.source_input_types[source_idx]);
		DataChunk null_payload;
		null_payload.Initialize(Allocator::Get(context), op.source_argument_types[source_idx]);
		Vector unmatched_addresses(LogicalType::POINTER);
		DataChunk group_ids;
		group_ids.Initialize(Allocator::Get(context), {LogicalType::UBIGINT});
		auto layout = target->GetLayoutPtr();
		for (idx_t group_begin = 0; group_begin < group_addresses.size(); group_begin += STANDARD_VECTOR_SIZE) {
			context.InterruptCheck();
			auto group_end = MinValue<idx_t>(group_begin + STANDARD_VECTOR_SIZE, group_addresses.size());
			auto addresses = FlatVector::GetDataMutable<data_ptr_t>(unmatched_addresses);
			group_ids.Reset();
			group_ids.data[0].SetVectorType(VectorType::FLAT_VECTOR);
			auto ids = FlatVector::GetDataMutable<uint64_t>(group_ids.data[0]);
			idx_t unmatched_count = 0;
			for (idx_t group_idx = group_begin; group_idx < group_end; group_idx++) {
				auto address = group_addresses[group_idx];
				if (LoadFactorizedCount(*layout, address, range.multiplicity_index.GetIndex()) == 0) {
					addresses[unmatched_count] = address;
					ids[unmatched_count++] = group_idx;
				}
			}
			if (unmatched_count == 0) {
				continue;
			}
			FlatVector::SetSize(unmatched_addresses, unmatched_count);
			FlatVector::SetSize(group_ids.data[0], unmatched_count);
			group_ids.SetChildCardinality(unmatched_count);
			null_input.Reset();
			for (idx_t column_idx = 0; column_idx < null_input.ColumnCount(); column_idx++) {
				null_input.data[column_idx].Reference(Value(op.source_input_types[source_idx][column_idx]),
				                                      count_t(unmatched_count));
			}
			null_input.SetChildCardinality(unmatched_count);
			null_payload.Reset();
			if (null_payload.ColumnCount() == 0) {
				null_payload.SetChildCardinality(unmatched_count);
			} else {
				executor.Execute(null_input, null_payload);
			}
			distinct.Sink(op, source_idx, *filter_sets[source_idx], group_ids, null_payload);
			target->UpdateAggregatesAtAddressesRangeWithFilterSet(
			    *update_state, unmatched_addresses, group_ids.data[0], null_payload, range.begin, range.count,
			    aggregate_filters[source_idx], *filter_sets[source_idx]);
		}
	}

	unique_ptr<ColumnDataCollection> BuildOutput() {
		auto result = make_uniq<ColumnDataCollection>(context, op.GetTypes());
		DataChunk groups;
		groups.Initialize(Allocator::Get(context), op.group_types);
		DataChunk output_chunk;
		output_chunk.Initialize(Allocator::Get(context), op.GetTypes());
		Vector row_addresses(LogicalType::POINTER);
		Vector selected_addresses(LogicalType::POINTER);
		SelectionVector selected_rows(STANDARD_VECTOR_SIZE);
		Vector left_multiplicities(LogicalType::BIGINT);
		Vector right_multiplicities(LogicalType::BIGINT);
		Vector product_multiplicities(LogicalType::BIGINT);
		Vector unit_multiplicities(Value::BIGINT(1), count_t(STANDARD_VECTOR_SIZE));
		FactorizedGroupJoinOutputBuffer output(context, op.target_layout);
		auto layout = target->GetLayoutPtr();
		auto &driver_range = op.source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)];
		auto &left_range = op.source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)];
		auto &right_range = op.source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR)];
		AggregateHTScanState scan;
		target->InitializeScan(scan);
		while (target->ScanGroupsAndAddresses(scan, groups, row_addresses)) {
			context.InterruptCheck();
			row_addresses.Flatten();
			auto addresses = FlatVector::GetData<data_ptr_t>(row_addresses);
			auto left_values = FlatVector::GetDataMutable<int64_t>(left_multiplicities);
			auto right_values = FlatVector::GetDataMutable<int64_t>(right_multiplicities);
			auto product_values = FlatVector::GetDataMutable<int64_t>(product_multiplicities);
			idx_t output_count = 0;
			for (idx_t row_idx = 0; row_idx < groups.size(); row_idx++) {
				auto driver_count = op.unique_driver ? idx_t(1)
				                                     : LoadFactorizedCount(*layout, addresses[row_idx],
				                                                           driver_range.multiplicity_index.GetIndex());
				auto left_count =
				    LoadFactorizedCount(*layout, addresses[row_idx], left_range.multiplicity_index.GetIndex());
				auto right_count =
				    LoadFactorizedCount(*layout, addresses[row_idx], right_range.multiplicity_index.GetIndex());
				if (op.semi_left && left_count != 0) {
					left_count = 1;
				}
				if (op.semi_right && right_count != 0) {
					right_count = 1;
				}
				if ((!op.preserve_left && left_count == 0) || (!op.preserve_right && right_count == 0)) {
					continue;
				}
				left_count = left_count == 0 ? 1 : left_count;
				right_count = right_count == 0 ? 1 : right_count;
				idx_t product;
				idx_t left_multiplicity;
				idx_t right_multiplicity;
				if (!TryMultiplyOperator::Operation(left_count, right_count, product) ||
				    !TryMultiplyOperator::Operation(right_count, driver_count, left_multiplicity) ||
				    !TryMultiplyOperator::Operation(left_count, driver_count, right_multiplicity)) {
					throw OutOfRangeException("Overflow in external factorized GroupJoin multiplicity");
				}
				selected_rows.set_index(output_count, row_idx);
				left_values[output_count] = CheckedFactorizedMultiplicity(left_multiplicity);
				right_values[output_count] = CheckedFactorizedMultiplicity(right_multiplicity);
				product_values[output_count] = CheckedFactorizedMultiplicity(product);
				output_count++;
			}
			if (output_count == 0) {
				continue;
			}
			FlatVector::SetSize(left_multiplicities, output_count);
			FlatVector::SetSize(right_multiplicities, output_count);
			FlatVector::SetSize(product_multiplicities, output_count);
			selected_addresses.Slice(row_addresses, selected_rows, output_count);
			selected_addresses.Flatten();
			output.Initialize(output_count);
			for (idx_t aggregate_idx = 0; aggregate_idx < op.aggregate_expressions.size(); aggregate_idx++) {
				auto source_idx = FactorizedSourceIndex(op.aggregate_sources[aggregate_idx]);
				Vector *multiplicities;
				if (op.aggregate_expressions[aggregate_idx]->Cast<BoundAggregateExpression>().IsDistinct()) {
					multiplicities = &unit_multiplicities;
				} else if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)) {
					multiplicities = &product_multiplicities;
				} else if (source_idx == FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)) {
					multiplicities = &left_multiplicities;
				} else {
					multiplicities = &right_multiplicities;
				}
				RowOperations::CombineStatesRange(
				    output.row_state, *layout, selected_addresses, op.partial_indexes[aggregate_idx], output.layout,
				    output.addresses, aggregate_idx, 1, *multiplicities, AggregateCombineType::PRESERVE_INPUT);
			}
			output_chunk.Reset();
			for (idx_t group_idx = 0; group_idx < op.output_group_key_indices.size(); group_idx++) {
				output_chunk.data[group_idx].Slice(groups.data[op.output_group_key_indices[group_idx]], selected_rows,
				                                   output_count);
			}
			output_chunk.SetChildCardinality(output_count);
			RowOperations::FinalizeStates(output.row_state, output.layout, output.addresses, output_chunk,
			                              op.output_group_key_indices.size());
			result->Append(output_chunk);
			output.Destroy();
		}
		return result;
	}

private:
	const PhysicalFactorizedGroupJoin &op;
	ClientContext &context;
	idx_t partition_budget;
	ExternalFactorizedDistinctState distinct;
	unique_ptr<GroupedAggregateHashTable> target;
	unique_ptr<GroupedAggregateHashTable> routing_table;
	unique_ptr<AggregateHTUpdateState> update_state;
	vector<data_ptr_t> group_addresses;
	vector<vector<data_ptr_t>> routes;
	vector<unique_ptr<AggregateFilterDataSet>> filter_sets;
	vector<unsafe_vector<idx_t>> aggregate_filters;
	array<idx_t, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> matched_rows {};
};

static idx_t EstimateExternalFactorizedWorkingSet(const ColumnDataCollection &driver,
                                                  optional_ptr<const ColumnDataCollection> left,
                                                  optional_ptr<const ColumnDataCollection> right,
                                                  idx_t target_row_width) {
	auto result = EstimateFactorizedCollectionSize(driver);
	auto pointer_table_size = GroupedAggregateHashTable::GetCapacityForCount(driver.Count()) * sizeof(ht_entry_t);
	idx_t target_size;
	if (!TryMultiplyOperator::Operation(driver.Count(), target_row_width, target_size) ||
	    !TryAddOperator::Operation(result, pointer_table_size, result) ||
	    !TryAddOperator::Operation(result, target_size, result)) {
		return NumericLimits<idx_t>::Maximum();
	}
	if (left && !TryAddOperator::Operation(result, EstimateFactorizedCollectionSize(*left), result)) {
		return NumericLimits<idx_t>::Maximum();
	}
	if (right && !TryAddOperator::Operation(result, EstimateFactorizedCollectionSize(*right), result)) {
		return NumericLimits<idx_t>::Maximum();
	}
	return result;
}

class FactorizedGroupJoinGlobalSourceState : public GlobalSourceState {
public:
	FactorizedGroupJoinGlobalSourceState(const PhysicalFactorizedGroupJoin &op, ClientContext &context_p)
	    : context(context_p) {
		auto &driver_state = GetFactorizedDriverState(op);
		D_ASSERT(driver_state.finalized);
		if (driver_state.external) {
			external = true;
			partition_count = driver_state.partitions->GetPartitions().size();
			auto target_types = op.group_types;
			target_types.push_back(LogicalType::HASH);
			TupleDataLayout target_layout;
			target_layout.Initialize(std::move(target_types), op.CreateHashTableAggregates(),
			                         TupleDataValidityType::CAN_HAVE_NULL_VALUES);
			target_row_width = target_layout.GetRowWidth();
			temporary_memory_state = TemporaryMemoryManager::Get(context).Register(context);
			auto &partitions = driver_state.partitions->GetPartitions();
			const auto minimum_working_set = BufferManager::GetBufferManager(context).GetBlockAllocSize() * 32;
			for (idx_t partition_idx = 0; partition_idx < partitions.size(); partition_idx++) {
				auto &partition = partitions[partition_idx];
				if (!partition || partition->Count() == 0) {
					continue;
				}
				// Factor partitions stay buffer-managed while the driver directory and aggregate states are pinned.
				auto partition_working_set =
				    EstimateExternalFactorizedWorkingSet(*partition, nullptr, nullptr, target_row_width);
				max_partition_working_set = MaxValue(max_partition_working_set, partition_working_set);
			}
			max_partition_working_set = MaxValue(max_partition_working_set, minimum_working_set);
			temporary_memory_state->SetMinimumReservation(max_partition_working_set);
			temporary_memory_state->SetZero();
			return;
		}
		lock_guard<mutex> guard(driver_state.lock);
		auto available_partitions = driver_state.target->GetPartitionedData().PartitionCount();
		auto useful_partitions = MinValue<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads(),
		                                         idx_t(1) << FACTORIZED_GROUP_JOIN_OUTPUT_RADIX_BITS);
		if (available_partitions < useful_partitions) {
			driver_state.target->SetRadixBits(FACTORIZED_GROUP_JOIN_OUTPUT_RADIX_BITS);
			driver_state.target->Repartition();
		}
		output_data = driver_state.target->AcquirePartitionedData();
		partition_count = output_data->PartitionCount();
		group_indexes.reserve(op.group_types.size());
		for (idx_t key_idx = 0; key_idx < op.group_types.size(); key_idx++) {
			group_indexes.push_back(NumericCast<column_t>(key_idx));
		}
	}

	idx_t MaxThreads() override {
		if (external) {
			auto max_threads = MinValue<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads(), partition_count);
			idx_t remaining_size;
			if (!TryMultiplyOperator::Operation(max_threads, max_partition_working_set, remaining_size)) {
				remaining_size = NumericLimits<idx_t>::Maximum();
			}
			temporary_memory_state->SetRemainingSizeAndUpdateReservation(context, remaining_size);
			auto partitions_fit =
			    MaxValue<idx_t>(temporary_memory_state->GetReservation() / max_partition_working_set, 1);
			auto result = MinValue(max_threads, partitions_fit);
			partition_budget =
			    MaxValue<idx_t>(temporary_memory_state->GetReservation() / result, max_partition_working_set);
			worker_count = result;
			return result;
		}
		return partition_count;
	}

	unique_ptr<PartitionedTupleData> output_data;
	vector<column_t> group_indexes;
	atomic<idx_t> next_partition {0};
	idx_t partition_count;
	bool external = false;
	ClientContext &context;
	unique_ptr<TemporaryMemoryState> temporary_memory_state;
	idx_t max_partition_working_set = 0;
	idx_t target_row_width = 0;
	idx_t partition_budget = 0;
	idx_t worker_count = 1;
	atomic<idx_t> processed_tasks {0};
	atomic<idx_t> repartitioned_tasks {0};
	atomic<idx_t> maximum_task_bytes {0};
	atomic<idx_t> maximum_radix_bits {FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS};
	array<atomic<idx_t>, PhysicalFactorizedGroupJoin::SOURCE_COUNT - 1> external_matched_rows {};
};

struct ExternalFactorizedPartitionTask {
	unique_ptr<ColumnDataCollection> driver;
	unique_ptr<ColumnDataCollection> left;
	unique_ptr<ColumnDataCollection> right;
	idx_t radix_bits;
};

static unique_ptr<RadixPartitionedColumnData>
RepartitionExternalFactorizedCollection(ClientContext &context, ColumnDataCollection &collection, idx_t radix_bits) {
	auto types = collection.Types();
	auto result = make_uniq<RadixPartitionedColumnData>(context, types, radix_bits, types.size() - 1);
	auto local = result->CreateShared();
	PartitionedColumnDataAppendState append_state;
	local->InitializeAppendState(append_state);
	DataChunk chunk;
	collection.InitializeScanChunk(Allocator::Get(context), chunk);
	ColumnDataScanState scan;
	collection.InitializeScan(scan);
	while (collection.Scan(scan, chunk)) {
		context.InterruptCheck();
		local->Append(append_state, chunk);
	}
	local->FlushAppendState(append_state);
	result->Combine(*local);
	return result;
}

static idx_t EstimateExternalFactorizedTaskSize(const ExternalFactorizedPartitionTask &task, idx_t target_row_width) {
	return EstimateExternalFactorizedWorkingSet(*task.driver, task.left.get(), task.right.get(), target_row_width);
}

static void RepartitionExternalFactorizedTask(ClientContext &context, ExternalFactorizedPartitionTask task,
                                              vector<ExternalFactorizedPartitionTask> &result) {
	const auto radix_bits = MinValue<idx_t>(task.radix_bits + 2, RadixPartitioning::MAX_RADIX_BITS);
	auto driver = RepartitionExternalFactorizedCollection(context, *task.driver, radix_bits);
	unique_ptr<RadixPartitionedColumnData> left;
	unique_ptr<RadixPartitionedColumnData> right;
	if (task.left) {
		left = RepartitionExternalFactorizedCollection(context, *task.left, radix_bits);
	}
	if (task.right) {
		right = RepartitionExternalFactorizedCollection(context, *task.right, radix_bits);
	}
	auto &driver_partitions = driver->GetPartitions();
	for (idx_t partition_idx = 0; partition_idx < driver_partitions.size(); partition_idx++) {
		if (!driver_partitions[partition_idx] || driver_partitions[partition_idx]->Count() == 0) {
			continue;
		}
		ExternalFactorizedPartitionTask child;
		child.driver = std::move(driver_partitions[partition_idx]);
		child.radix_bits = radix_bits;
		if (left) {
			auto &left_partitions = left->GetPartitions();
			if (partition_idx < left_partitions.size()) {
				child.left = std::move(left_partitions[partition_idx]);
			}
		}
		if (right) {
			auto &right_partitions = right->GetPartitions();
			if (partition_idx < right_partitions.size()) {
				child.right = std::move(right_partitions[partition_idx]);
			}
		}
		result.push_back(std::move(child));
	}
}

class FactorizedGroupJoinLocalSourceState : public LocalSourceState {
public:
	FactorizedGroupJoinLocalSourceState(ExecutionContext &context, const PhysicalFactorizedGroupJoin &op)
	    : row_addresses(LogicalType::POINTER), selected_addresses(LogicalType::POINTER),
	      selected_rows(STANDARD_VECTOR_SIZE), left_multiplicities(LogicalType::BIGINT),
	      right_multiplicities(LogicalType::BIGINT), product_multiplicities(LogicalType::BIGINT),
	      unit_multiplicities(Value::BIGINT(1), count_t(STANDARD_VECTOR_SIZE)),
	      source_allocator(Allocator::Get(context.client)), source_row_state(source_allocator),
	      output(context.client, op.target_layout) {
		groups.Initialize(Allocator::Get(context.client), op.group_types);
	}

	DataChunk groups;
	Vector row_addresses;
	Vector selected_addresses;
	SelectionVector selected_rows;
	Vector left_multiplicities;
	Vector right_multiplicities;
	Vector product_multiplicities;
	Vector unit_multiplicities;
	TupleDataScanState scan_state;
	idx_t partition_idx = DConstants::INVALID_INDEX;
	bool partition_assigned = false;
	ArenaAllocator source_allocator;
	RowOperationsState source_row_state;
	FactorizedGroupJoinOutputBuffer output;
	unique_ptr<ColumnDataCollection> external_output;
	ColumnDataScanState external_scan;
	vector<ExternalFactorizedPartitionTask> external_tasks;
};

unique_ptr<GlobalSourceState> PhysicalFactorizedGroupJoin::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<FactorizedGroupJoinGlobalSourceState>(*this, context);
}

unique_ptr<LocalSourceState> PhysicalFactorizedGroupJoin::GetLocalSourceState(ExecutionContext &context,
                                                                              GlobalSourceState &) const {
	return make_uniq<FactorizedGroupJoinLocalSourceState>(context, *this);
}

static int64_t CheckedFactorizedMultiplicity(idx_t value) {
	if (value > NumericCast<idx_t>(NumericLimits<int64_t>::Maximum())) {
		throw OutOfRangeException("Factorized GroupJoin multiplicity exceeds BIGINT");
	}
	return NumericCast<int64_t>(value);
}

static idx_t LoadFactorizedCount(const TupleDataLayout &layout, data_ptr_t row, idx_t aggregate_idx) {
	auto offset_idx = layout.ColumnCount() + aggregate_idx;
	D_ASSERT(offset_idx < layout.GetOffsets().size());
	auto value = *reinterpret_cast<const int64_t *>(row + layout.GetOffsets()[offset_idx]);
	if (value < 0) {
		throw InternalException("Factorized GroupJoin encountered a negative multiplicity");
	}
	return NumericCast<idx_t>(value);
}

static void UpdateFactorizedMaximum(atomic<idx_t> &target, idx_t value) {
	auto current = target.load(std::memory_order_relaxed);
	while (current < value &&
	       !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
	}
}

static void PublishFactorizedGroupJoinMetrics(const PhysicalFactorizedGroupJoin &op, ExecutionContext &context,
                                              FactorizedGroupJoinGlobalSourceState &gstate) {
	auto params = op.ParamsToString();
	auto &driver = GetFactorizedDriverState(op);
	auto &left = op.left_sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
	auto &right = op.right_sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
	params["Driver Input Rows"] = to_string(driver.input_rows.load(std::memory_order_relaxed));
	params["Left Factor Input Rows"] = to_string(left.input_rows.load(std::memory_order_relaxed));
	params["Right Factor Input Rows"] = to_string(right.input_rows.load(std::memory_order_relaxed));
	if (gstate.external) {
		params["Left Matched Factor Rows"] = to_string(gstate.external_matched_rows[0].load(std::memory_order_relaxed));
		params["Right Matched Factor Rows"] =
		    to_string(gstate.external_matched_rows[1].load(std::memory_order_relaxed));
		params["External Workers"] = to_string(gstate.worker_count);
		params["External Partition Budget"] = to_string(gstate.partition_budget);
		params["External Tasks"] = to_string(gstate.processed_tasks.load(std::memory_order_relaxed));
		params["External Repartitions"] = to_string(gstate.repartitioned_tasks.load(std::memory_order_relaxed));
		params["External Maximum Task Bytes"] = to_string(gstate.maximum_task_bytes.load(std::memory_order_relaxed));
		params["External Maximum Radix Bits"] = to_string(gstate.maximum_radix_bits.load(std::memory_order_relaxed));
	} else {
		params["Left Matched Factor Rows"] = to_string(left.matched_rows.load(std::memory_order_relaxed));
		params["Right Matched Factor Rows"] = to_string(right.matched_rows.load(std::memory_order_relaxed));
		params["Left Mode"] = FactorizedBranchModeToString(driver.runtime_branch_modes[0]);
		params["Right Mode"] = FactorizedBranchModeToString(driver.runtime_branch_modes[1]);
		params["Left Sample"] = StringUtil::Format("drivers=%llu, matches=%llu, factor rows=%llu, factors=%llu",
		                                           driver.sampled_driver_rows[0], driver.sampled_matches[0],
		                                           driver.sampled_factor_rows[0], driver.sampled_distinct_factors[0]);
		params["Right Sample"] = StringUtil::Format("drivers=%llu, matches=%llu, factor rows=%llu, factors=%llu",
		                                            driver.sampled_driver_rows[1], driver.sampled_matches[1],
		                                            driver.sampled_factor_rows[1], driver.sampled_distinct_factors[1]);
	}
	context.thread.profiler.GetOperatorMetrics(op).SetExtraInfo(std::move(params));
}

SourceResultType PhysicalFactorizedGroupJoin::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                              OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<FactorizedGroupJoinGlobalSourceState>();
	auto &state = input.local_state.Cast<FactorizedGroupJoinLocalSourceState>();
	if (gstate.external) {
		auto &driver_state = GetFactorizedDriverState(*this);
		auto &left_state = left_sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
		auto &right_state = right_sink->sink_state->Cast<FactorizedGroupJoinGlobalSinkState>();
		while (true) {
			context.client.InterruptCheck();
			if (state.external_output && state.external_output->Scan(state.external_scan, chunk)) {
				PublishFactorizedGroupJoinMetrics(*this, context, gstate);
				return SourceResultType::HAVE_MORE_OUTPUT;
			}
			state.external_output.reset();
			if (state.external_tasks.empty()) {
				auto partition_idx = gstate.next_partition.fetch_add(1, std::memory_order_relaxed);
				if (partition_idx >= gstate.partition_count) {
					PublishFactorizedGroupJoinMetrics(*this, context, gstate);
					return SourceResultType::FINISHED;
				}
				auto &driver_partitions = driver_state.partitions->GetPartitions();
				auto &left_partitions = left_state.partitions->GetPartitions();
				auto &right_partitions = right_state.partitions->GetPartitions();
				if (partition_idx >= driver_partitions.size() || !driver_partitions[partition_idx] ||
				    driver_partitions[partition_idx]->Count() == 0) {
					if (partition_idx < left_partitions.size()) {
						left_partitions[partition_idx].reset();
					}
					if (partition_idx < right_partitions.size()) {
						right_partitions[partition_idx].reset();
					}
					continue;
				}
				ExternalFactorizedPartitionTask task;
				task.driver = std::move(driver_partitions[partition_idx]);
				task.radix_bits = FACTORIZED_GROUP_JOIN_EXTERNAL_RADIX_BITS;
				if (partition_idx < left_partitions.size()) {
					task.left = std::move(left_partitions[partition_idx]);
				}
				if (partition_idx < right_partitions.size()) {
					task.right = std::move(right_partitions[partition_idx]);
				}
				state.external_tasks.push_back(std::move(task));
			}
			auto task = std::move(state.external_tasks.back());
			state.external_tasks.pop_back();
			auto task_size = EstimateExternalFactorizedTaskSize(task, gstate.target_row_width);
			UpdateFactorizedMaximum(gstate.maximum_task_bytes, task_size);
			UpdateFactorizedMaximum(gstate.maximum_radix_bits, task.radix_bits);
			if (task.radix_bits < RadixPartitioning::MAX_RADIX_BITS && task_size > gstate.partition_budget) {
				gstate.repartitioned_tasks.fetch_add(1, std::memory_order_relaxed);
				RepartitionExternalFactorizedTask(context.client, std::move(task), state.external_tasks);
				continue;
			}
			ExternalFactorizedPartitionProcessor processor(*this, context.client, gstate.partition_budget);
			state.external_output =
			    processor.Process(std::move(task.driver), std::move(task.left), std::move(task.right));
			for (idx_t factor_idx = 0; factor_idx < gstate.external_matched_rows.size(); factor_idx++) {
				gstate.external_matched_rows[factor_idx].fetch_add(processor.MatchedRows(factor_idx + 1),
				                                                   std::memory_order_relaxed);
			}
			gstate.processed_tasks.fetch_add(1, std::memory_order_relaxed);
			state.external_output->InitializeScan(state.external_scan);
		}
	}
	auto &driver_state = GetFactorizedDriverState(*this);
	auto &target = *driver_state.target;
	auto layout = target.GetLayoutPtr();
	auto &driver_range = source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::DRIVER)];
	auto &left_range = source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::LEFT_FACTOR)];
	auto &right_range = source_ranges[FactorizedSourceIndex(FactorizedAggregateSource::RIGHT_FACTOR)];
	D_ASSERT((unique_driver || driver_range.multiplicity_index.IsValid()) && left_range.multiplicity_index.IsValid() &&
	         right_range.multiplicity_index.IsValid());

	while (true) {
		if (!state.partition_assigned) {
			state.partition_idx = gstate.next_partition.fetch_add(1, std::memory_order_relaxed);
			if (state.partition_idx >= gstate.partition_count) {
				PublishFactorizedGroupJoinMetrics(*this, context, gstate);
				return SourceResultType::FINISHED;
			}
			gstate.output_data->GetPartitions()[state.partition_idx]->InitializeScan(
			    state.scan_state, gstate.group_indexes, TupleDataPinProperties::DESTROY_AFTER_DONE);
			state.partition_assigned = true;
		}
		D_ASSERT(state.partition_idx < gstate.partition_count);
		auto &partition = *gstate.output_data->GetPartitions()[state.partition_idx];
		state.groups.Reset();
		if (!partition.Scan(state.scan_state, state.groups)) {
			partition.Reset();
			state.partition_assigned = false;
			continue;
		}
		if (state.groups.size() == 0) {
			continue;
		}
		state.row_addresses.Reference(state.scan_state.chunk_state.row_locations);
		FlatVector::SetSize(state.row_addresses, state.groups.size());

		state.row_addresses.Flatten();
		auto addresses = FlatVector::GetData<data_ptr_t>(state.row_addresses);
		auto left_values = FlatVector::GetDataMutable<int64_t>(state.left_multiplicities);
		auto right_values = FlatVector::GetDataMutable<int64_t>(state.right_multiplicities);
		auto product_values = FlatVector::GetDataMutable<int64_t>(state.product_multiplicities);
		idx_t result_count = 0;
		for (idx_t row_idx = 0; row_idx < state.groups.size(); row_idx++) {
			auto driver_count = unique_driver ? idx_t(1)
			                                  : LoadFactorizedCount(*layout, addresses[row_idx],
			                                                        driver_range.multiplicity_index.GetIndex());
			auto left_count =
			    LoadFactorizedCount(*layout, addresses[row_idx], left_range.multiplicity_index.GetIndex());
			auto right_count =
			    LoadFactorizedCount(*layout, addresses[row_idx], right_range.multiplicity_index.GetIndex());
			if (semi_left && left_count != 0) {
				left_count = 1;
			}
			if (semi_right && right_count != 0) {
				right_count = 1;
			}
			if ((!preserve_left && left_count == 0) || (!preserve_right && right_count == 0)) {
				continue;
			}
			left_count = left_count == 0 ? 1 : left_count;
			right_count = right_count == 0 ? 1 : right_count;
			idx_t product;
			if (!TryMultiplyOperator::Operation(left_count, right_count, product)) {
				throw OutOfRangeException("Overflow in factorized GroupJoin multiplicity");
			}
			idx_t left_multiplicity;
			idx_t right_multiplicity;
			if (!TryMultiplyOperator::Operation(right_count, driver_count, left_multiplicity) ||
			    !TryMultiplyOperator::Operation(left_count, driver_count, right_multiplicity)) {
				throw OutOfRangeException("Overflow in factorized GroupJoin multiplicity");
			}
			state.selected_rows.set_index(result_count, row_idx);
			left_values[result_count] = CheckedFactorizedMultiplicity(left_multiplicity);
			right_values[result_count] = CheckedFactorizedMultiplicity(right_multiplicity);
			product_values[result_count] = CheckedFactorizedMultiplicity(product);
			result_count++;
		}
		auto destroy_source_states = [&]() {
			if (layout->HasDestructor()) {
				RowOperations::DestroyStates(state.source_row_state, *layout, state.row_addresses);
			}
		};
		if (result_count == 0) {
			destroy_source_states();
			continue;
		}
		FlatVector::SetSize(state.left_multiplicities, count_t(result_count));
		FlatVector::SetSize(state.right_multiplicities, count_t(result_count));
		FlatVector::SetSize(state.product_multiplicities, count_t(result_count));
		state.selected_addresses.Slice(state.row_addresses, state.selected_rows, result_count);
		state.selected_addresses.Flatten();
		state.output.Initialize(result_count);

		try {
			for (idx_t aggregate_idx = 0; aggregate_idx < aggregate_expressions.size(); aggregate_idx++) {
				auto source_idx = FactorizedSourceIndex(aggregate_sources[aggregate_idx]);
				Vector *multiplicities;
				if (aggregate_expressions[aggregate_idx]->Cast<BoundAggregateExpression>().IsDistinct()) {
					multiplicities = &state.unit_multiplicities;
				} else {
					switch (source_idx) {
					case 0:
						multiplicities = &state.product_multiplicities;
						break;
					case 1:
						multiplicities = &state.left_multiplicities;
						break;
					case 2:
						multiplicities = &state.right_multiplicities;
						break;
					default:
						throw InternalException("Invalid factorized aggregate source");
					}
				}
				RowOperations::CombineStatesRange(state.output.row_state, *layout, state.selected_addresses,
				                                  partial_indexes[aggregate_idx], state.output.layout,
				                                  state.output.addresses, aggregate_idx, 1, *multiplicities,
				                                  AggregateCombineType::PRESERVE_INPUT);
			}

			chunk.Reset();
			for (idx_t group_idx = 0; group_idx < output_group_key_indices.size(); group_idx++) {
				chunk.data[group_idx].Slice(state.groups.data[output_group_key_indices[group_idx]], state.selected_rows,
				                            result_count);
			}
			chunk.SetChildCardinality(result_count);
			RowOperations::FinalizeStates(state.output.row_state, state.output.layout, state.output.addresses, chunk,
			                              output_group_key_indices.size());
		} catch (...) {
			state.output.Destroy();
			destroy_source_states();
			throw;
		}
		state.output.Destroy();
		destroy_source_states();
		PublishFactorizedGroupJoinMetrics(*this, context, gstate);
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
}

string PhysicalFactorizedGroupJoin::GetName() const {
	return "FACTORIZED_GROUP_JOIN";
}

static const char *FactorizedBranchModeToString(FactorizedGroupJoinBranchMode mode) {
	switch (mode) {
	case FactorizedGroupJoinBranchMode::LAZY:
		return "LAZY";
	case FactorizedGroupJoinBranchMode::CACHED:
		return "CACHED";
	case FactorizedGroupJoinBranchMode::EAGER:
		return "EAGER";
	default:
		throw InternalException("Invalid factorized GroupJoin branch mode");
	}
}

InsertionOrderPreservingMap<string> PhysicalFactorizedGroupJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Aggregates"] = StringUtil::Join(aggregate_names, "\n");
	result["Execution"] = planned_execution_mode == GroupJoinExecutionMode::EXTERNAL
	                          ? "common-radix external partitions"
	                      : !driver_first ? "factors-first adaptive linear-chain directories"
	                      : routed        ? "driver-first target + compact routed factor updates"
	                                      : "driver-first target + task-local dense factor states";
	result["Strategy"] = EnumUtil::ToString(planned_execution_mode);
	result["Pipeline"] = streaming_driver ? "STREAMING_DRIVER" : "MATERIALIZED_TARGET";
	result["Driver"] = unique_driver ? "UNIQUE" : "NON_UNIQUE";
	result["Routing"] = routed ? "ROUTED" : "DIRECT";
	result["Lookup"] = planned_execution_mode == GroupJoinExecutionMode::EXTERNAL ? "PARTITIONED HASH"
	                   : !perfect_min.IsNull() && !perfect_max.IsNull()           ? "DIRECTORY"
	                                                                              : "HASH";
	result["Left Edge"] = semi_left ? "SEMI" : preserve_left ? "LEFT" : "INNER";
	result["Right Edge"] = semi_right ? "SEMI" : preserve_right ? "LEFT" : "INNER";
	result["Left Mode"] = FactorizedBranchModeToString(branch_modes[0]);
	result["Right Mode"] = FactorizedBranchModeToString(branch_modes[1]);
	result["Estimated Cardinalities"] = StringUtil::Format(
	    "driver=%llu, left=%llu, right=%llu, left scan=%llu, right scan=%llu, expanded=%llu, matched drivers=%llu",
	    estimated_driver_rows, estimated_left_factor_rows, estimated_right_factor_rows, estimated_left_scan_rows,
	    estimated_right_scan_rows, estimated_join_rows, estimated_matched_drivers);
	result["Estimated Costs"] = StringUtil::Format(
	    "build=%.2f, filter=%.2f, probe=%.2f, scan=%.2f, cache=%.2f, eager=%.2f, routing=%.2f, spill=%.2f, "
	    "driver-first=%.2f, factors-first=%.2f, total=%.2f, best existing=%.2f",
	    estimated_build_cost, estimated_filter_cost, estimated_probe_cost, estimated_scan_cost, estimated_cache_cost,
	    estimated_eager_work_cost, estimated_routing_cost, estimated_spill_cost, estimated_driver_first_cost,
	    estimated_factors_first_cost, estimated_factorized_cost, estimated_best_existing_cost);
	result["Decision"] = auto_selected             ? "AUTO SELECTED"
	                     : estimated_cost_reliable ? "FORCED (AUTO COST AVAILABLE)"
	                                               : "FORCED (UNRELIABLE STATISTICS)";
	result["Runtime Filters"] =
	    StringUtil::Format("driver->left=%s, driver->right=%s, left->driver=%s, right->driver=%s",
	                       driver_filter_pushdown[0] ? "YES" : "NO", driver_filter_pushdown[1] ? "YES" : "NO",
	                       factor_filter_pushdown[0] ? "YES" : "NO", factor_filter_pushdown[1] ? "YES" : "NO");
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
