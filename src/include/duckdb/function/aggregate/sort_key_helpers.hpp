//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/aggregate/sort_key_helpers.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/function/create_sort_key.hpp"

namespace duckdb {

struct AggregateSortKeyHelpers {
	template <class STATE, class OP, OrderType ORDER_TYPE = OrderType::ASCENDING, bool IGNORE_NULLS = true>
	static idx_t UnaryUpdateWithChange(Vector inputs[], AggregateInputData &input_data, idx_t input_count,
	                                   Vector &state_vector, idx_t count, SelectionVector &changed) {
		D_ASSERT(input_count == 1);
		auto &input = inputs[0];

		Vector sort_key(LogicalType::BLOB);
		auto modifiers = OrderModifiers(ORDER_TYPE, OrderByNullType::NULLS_LAST);
		CreateSortKeyHelpers::CreateSortKey(input, modifiers, sort_key);

		optional<VectorValidityIterator> input_validity;
		if (IGNORE_NULLS) {
			input_validity = input.Validity();
		}

		UnifiedVectorFormat key_data;
		sort_key.ToUnifiedFormat(key_data);
		UnifiedVectorFormat state_data;
		state_vector.ToUnifiedFormat(state_data);
		auto keys = UnifiedVectorFormat::GetData<string_t>(key_data);
		auto states = UnifiedVectorFormat::GetData<STATE *>(state_data);
		idx_t changed_count = 0;
		for (idx_t i = 0; i < count; i++) {
			if (IGNORE_NULLS && !input_validity.value().IsValid(i)) {
				continue;
			}
			auto &state = *states[state_data.sel->get_index(i)];
			if (OP::template ExecuteWithChange<string_t, STATE, OP>(state, keys[key_data.sel->get_index(i)],
			                                                        input_data)) {
				changed.set_index(changed_count++, i);
			}
		}
		return changed_count;
	}

	template <class STATE, class OP, OrderType ORDER_TYPE = OrderType::ASCENDING, bool IGNORE_NULLS = true>
	static void UnaryUpdate(Vector inputs[], AggregateInputData &input_data, idx_t input_count, Vector &state_vector,
	                        idx_t count) {
		D_ASSERT(input_count == 1);
		auto &input = inputs[0];

		Vector sort_key(LogicalType::BLOB);
		auto modifiers = OrderModifiers(ORDER_TYPE, OrderByNullType::NULLS_LAST);
		CreateSortKeyHelpers::CreateSortKey(input, modifiers, sort_key);

		optional<VectorValidityIterator> input_validity;
		if (IGNORE_NULLS) {
			input_validity = input.Validity();
		}

		UnifiedVectorFormat kdata;
		sort_key.ToUnifiedFormat(kdata);

		UnifiedVectorFormat sdata;
		state_vector.ToUnifiedFormat(sdata);

		auto key_data = UnifiedVectorFormat::GetData<string_t>(kdata);
		auto states = UnifiedVectorFormat::GetData<STATE *>(sdata);
		for (idx_t i = 0; i < count; i++) {
			const auto sidx = sdata.sel->get_index(i);
			if (IGNORE_NULLS) {
				if (!input_validity.value().IsValid(i)) {
					continue;
				}
			}
			const auto key_idx = kdata.sel->get_index(i);
			auto &state = *states[sidx];
			OP::template Execute<string_t, STATE, OP>(state, key_data[key_idx], input_data);
		}
	}
};

} // namespace duckdb
