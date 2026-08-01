#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/common/row_operations/row_operations.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/operator/aggregate/aggregate_object.hpp"

namespace duckdb {

void RowOperations::InitializeStates(TupleDataLayout &layout, Vector &addresses, const SelectionVector &sel,
                                     idx_t count) {
	if (count == 0) {
		return;
	}
	auto pointers = FlatVector::GetData<data_ptr_t>(addresses);
	auto &offsets = layout.GetOffsets();
	auto aggr_idx = layout.ColumnCount();

	for (const auto &aggr : layout.GetAggregates()) {
		AggregateStateInput state_input(aggr.function, aggr.GetFunctionData());
		if (sel.IsSet()) {
			for (idx_t i = 0; i < count; ++i) {
				data_ptr_t state_ptr = pointers[sel.get_index_unsafe(i)] + offsets[aggr_idx];
				aggr.function.GetStateInitCallback()(state_input, &state_ptr, 1);
			}
		} else {
			for (idx_t i = 0; i < count; ++i) {
				data_ptr_t state_ptr = pointers[i] + offsets[aggr_idx];
				aggr.function.GetStateInitCallback()(state_input, &state_ptr, 1);
			}
		}
		++aggr_idx;
	}
}

void RowOperations::DestroyStates(RowOperationsState &state, TupleDataLayout &layout, Vector &addresses) {
	auto count = addresses.size();
	if (count == 0) {
		return;
	}
	//	Move to the first aggregate state
	VectorOperations::AddInPlace(addresses, UnsafeNumericCast<int64_t>(layout.GetAggrOffset()));
	for (const auto &aggr : layout.GetAggregates()) {
		if (aggr.function.HasStateDestructorCallback()) {
			AggregateInputData aggr_input_data(aggr, state.allocator);
			aggr.function.GetStateDestructorCallback()(addresses, aggr_input_data, count);
		}
		// Move to the next aggregate state
		VectorOperations::AddInPlace(addresses, UnsafeNumericCast<int64_t>(aggr.payload_size));
	}
}

void RowOperations::UpdateStates(RowOperationsState &state, AggregateObject &aggr, Vector &addresses,
                                 DataChunk &payload, idx_t arg_idx, optional_ptr<const ClusteredAggr> clustered) {
	auto count = addresses.size();
	AggregateInputData aggr_input_data(aggr, state.allocator);
	auto cluster_update = aggr.function.GetStateClusterUpdateCallback();
	aggr_input_data.clustered = cluster_update ? clustered : nullptr;
	auto inputs = aggr.child_count ? payload.data.data() + arg_idx : nullptr;
	if (clustered && cluster_update) {
		cluster_update(inputs, aggr_input_data, aggr.child_count, *clustered, count);
		return;
	}
	aggr.function.GetStateUpdateCallback()(inputs, aggr_input_data, aggr.child_count, addresses, count);
}

void RowOperations::UpdateFilteredStates(RowOperationsState &state, AggregateFilterData &filter_data,
                                         AggregateObject &aggr, Vector &addresses, DataChunk &payload, idx_t arg_idx) {
	idx_t count = filter_data.ApplyFilter(payload);
	if (count == 0) {
		return;
	}

	Vector filtered_addresses(addresses, filter_data.true_sel, count);
	filtered_addresses.Flatten();

	UpdateStates(state, aggr, filtered_addresses, filter_data.filtered_payload, arg_idx);
}

void RowOperations::UpdateStatesClustered(RowOperationsState &state, vector<AggregateObject> &aggregates,
                                          AggregateFilterDataSet *filter_set, const unsafe_vector<idx_t> *filter,
                                          Vector &addresses, DataChunk &payload, ClusteredAggr &clustered,
                                          bool skip_addresses) {
	UpdateStatesClusteredRange(state, aggregates, 0, aggregates.size(), filter_set, filter, addresses, payload,
	                           clustered, skip_addresses);
}

void RowOperations::UpdateStatesClusteredRange(RowOperationsState &state, vector<AggregateObject> &aggregates,
                                               idx_t aggregate_begin, idx_t aggregate_count,
                                               AggregateFilterDataSet *filter_set, const unsafe_vector<idx_t> *filter,
                                               Vector &addresses, DataChunk &payload, ClusteredAggr &clustered,
                                               bool skip_addresses) {
	idx_t filter_idx = 0;
	idx_t payload_idx = 0;
	for (idx_t range_idx = 0; range_idx < aggregate_count; range_idx++) {
		const auto aggr_idx = aggregate_begin + range_idx;
		auto &aggr = aggregates[aggr_idx];
		if (filter && (filter_idx >= filter->size() || range_idx < (*filter)[filter_idx])) {
			// Skip all the aggregates that are not in the filter
			payload_idx += aggr.child_count;
			if (!skip_addresses) {
				VectorOperations::AddInPlace(addresses, NumericCast<int64_t>(aggr.payload_size));
			}
			clustered.AdvanceStates(aggr.payload_size);
			continue;
		}
		if (filter) {
			D_ASSERT(range_idx == (*filter)[filter_idx]);
		}

		if (aggr.aggr_type != AggregateType::DISTINCT && aggr.filter) {
			D_ASSERT(filter_set);
			RowOperations::UpdateFilteredStates(state, filter_set->GetFilterData(aggr_idx), aggr, addresses, payload,
			                                    payload_idx);
		} else {
			UpdateStates(state, aggr, addresses, payload, payload_idx, clustered);
		}

		// Move to the next aggregate
		payload_idx += aggr.child_count;
		if (!skip_addresses) {
			VectorOperations::AddInPlace(addresses, NumericCast<int64_t>(aggr.payload_size));
		}
		clustered.AdvanceStates(aggr.payload_size);
		if (filter) {
			filter_idx++;
		}
	}
}

void RowOperations::CombineStates(RowOperationsState &state, TupleDataLayout &layout, Vector &sources,
                                  Vector &targets) {
	CombineStatesRange(state, layout, sources, 0, layout, targets, 0, layout.GetAggregates().size());
}

static idx_t GetAggregateStateOffset(TupleDataLayout &layout, idx_t aggregate_idx) {
	idx_t result = layout.GetAggrOffset();
	auto &aggregates = layout.GetAggregates();
	for (idx_t i = 0; i < aggregate_idx; i++) {
		result += aggregates[i].payload_size;
	}
	return result;
}

void RowOperations::CombineStatesRange(RowOperationsState &state, TupleDataLayout &source_layout, Vector &sources,
                                       idx_t source_begin, TupleDataLayout &target_layout, Vector &targets,
                                       idx_t target_begin, idx_t aggregate_count) {
	if (sources.size() != targets.size()) {
		throw InternalException("CombineStatesRange: source count (%llu) does not match target count (%llu)",
		                        sources.size(), targets.size());
	}
	auto &source_aggregates = source_layout.GetAggregates();
	auto &target_aggregates = target_layout.GetAggregates();
	if (source_begin > source_aggregates.size() || aggregate_count > source_aggregates.size() - source_begin) {
		throw InternalException("CombineStatesRange: source range [%llu, %llu) exceeds aggregate count %llu",
		                        source_begin, source_begin + aggregate_count, source_aggregates.size());
	}
	if (target_begin > target_aggregates.size() || aggregate_count > target_aggregates.size() - target_begin) {
		throw InternalException("CombineStatesRange: target range [%llu, %llu) exceeds aggregate count %llu",
		                        target_begin, target_begin + aggregate_count, target_aggregates.size());
	}
	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &source = source_aggregates[source_begin + i];
		auto &target = target_aggregates[target_begin + i];
		if (source.function != target.function ||
		    !FunctionData::Equals(source.GetFunctionData(), target.GetFunctionData()) ||
		    source.payload_size != target.payload_size) {
			throw InternalException("CombineStatesRange: incompatible aggregate states at range offset %llu", i);
		}
		if (!target.function.HasStateCombineCallback()) {
			throw InternalException("CombineStatesRange: aggregate at range offset %llu does not support combine", i);
		}
	}

	auto count = sources.size();
	if (count == 0 || aggregate_count == 0) {
		return;
	}

	auto source_offset = GetAggregateStateOffset(source_layout, source_begin);
	auto target_offset = GetAggregateStateOffset(target_layout, target_begin);
	VectorOperations::AddInPlace(sources, UnsafeNumericCast<int64_t>(source_offset));
	VectorOperations::AddInPlace(targets, UnsafeNumericCast<int64_t>(target_offset));

	for (idx_t i = 0; i < aggregate_count; i++) {
		auto &source = source_aggregates[source_begin + i];
		auto &target = target_aggregates[target_begin + i];
		AggregateInputData aggr_input_data(target, state.allocator, AggregateCombineType::ALLOW_DESTRUCTIVE);
		target.function.GetStateCombineCallback()(sources, targets, aggr_input_data, count);

		VectorOperations::AddInPlace(sources, UnsafeNumericCast<int64_t>(source.payload_size));
		VectorOperations::AddInPlace(targets, UnsafeNumericCast<int64_t>(target.payload_size));
		source_offset += source.payload_size;
		target_offset += target.payload_size;
	}

	VectorOperations::AddInPlace(sources, -UnsafeNumericCast<int64_t>(source_offset));
	VectorOperations::AddInPlace(targets, -UnsafeNumericCast<int64_t>(target_offset));
}

void RowOperations::FinalizeStates(RowOperationsState &state, TupleDataLayout &layout, Vector &addresses,
                                   DataChunk &result, idx_t aggr_idx) {
	FinalizeStatesRange(state, layout, addresses, result, aggr_idx, 0, layout.GetAggregates().size());
}

void RowOperations::FinalizeStatesRange(RowOperationsState &state, TupleDataLayout &layout, Vector &addresses,
                                        DataChunk &result, idx_t result_idx, idx_t aggregate_begin,
                                        idx_t aggregate_count) {
	auto &aggregates = layout.GetAggregates();
	if (aggregate_begin > aggregates.size() || aggregate_count > aggregates.size() - aggregate_begin) {
		throw InternalException("FinalizeStatesRange: aggregate range [%llu, %llu) exceeds aggregate count %llu",
		                        aggregate_begin, aggregate_begin + aggregate_count, aggregates.size());
	}
	if (result_idx > result.ColumnCount() || aggregate_count > result.ColumnCount() - result_idx) {
		throw InternalException("FinalizeStatesRange: result range [%llu, %llu) exceeds column count %llu", result_idx,
		                        result_idx + aggregate_count, result.ColumnCount());
	}

	// Copy the addresses
	if (!state.addresses) {
		state.addresses = make_uniq<Vector>(LogicalType::POINTER);
	}
	auto &addresses_copy = *state.addresses;
	VectorOperations::Copy(addresses, addresses_copy, result.size(), 0, 0);
	FlatVector::SetSize(addresses_copy, count_t(result.size()));

	// initialize the finalize local states once - they are re-used across all finalize calls of this state
	if (state.local_states.size() < aggregates.size()) {
		state.local_states.resize(aggregates.size());
		for (idx_t i = 0; i < aggregates.size(); i++) {
			auto &callbacks = aggregates[i].function.GetCallbacks();
			if (callbacks.HasInitLocalStateFinalizeCallback()) {
				AggregateInputData aggr_input_data(aggregates[i], state.allocator);
				state.local_states[i] =
				    callbacks.GetInitLocalStateFinalizeCallback()(aggr_input_data.function, aggr_input_data.bind_data);
			}
		}
	}

	idx_t state_offset = layout.GetAggrOffset();
	for (idx_t i = 0; i < aggregate_begin; i++) {
		state_offset += aggregates[i].payload_size;
	}
	VectorOperations::AddInPlace(addresses_copy, UnsafeNumericCast<int64_t>(state_offset));

	const auto aggregate_end = aggregate_begin + aggregate_count;
	for (idx_t i = aggregate_begin; i < aggregate_end; i++) {
		auto &target = result.data[result_idx + i - aggregate_begin];
		auto &aggr = aggregates[i];
		AggregateFinalizeInputData finalize_input_data(aggr, state.allocator, state.local_states[i].get());
		aggr.function.GetStateFinalizeCallback()(addresses_copy, finalize_input_data, target, result.size(), 0);
		FlatVector::SetSize(target, count_t(result.size()));

		// Move to the next aggregate state
		VectorOperations::AddInPlace(addresses_copy, UnsafeNumericCast<int64_t>(aggr.payload_size));
	}
}

} // namespace duckdb
