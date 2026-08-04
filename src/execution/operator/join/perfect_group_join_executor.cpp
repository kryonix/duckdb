#include "duckdb/execution/operator/join/perfect_group_join_executor.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"

#include <type_traits>

namespace duckdb {

template <class T>
static idx_t GetDirectoryIndex(T value, T minimum) {
	using UNSIGNED_TYPE = typename std::make_unsigned<T>::type;
	return NumericCast<idx_t>(static_cast<UNSIGNED_TYPE>(value) - static_cast<UNSIGNED_TYPE>(minimum));
}

PerfectGroupJoinExecutor::PerfectGroupJoinExecutor(LogicalType key_type_p, Value minimum_p, Value maximum_p,
                                                   idx_t range_p)
    : key_type(std::move(key_type_p)), minimum(std::move(minimum_p)), maximum(std::move(maximum_p)), range(range_p),
      directory(make_unsafe_uniq_array<data_ptr_t>(range + 1)),
      directory_group_ids(make_unsafe_uniq_array_uninitialized<uint64_t>(range + 1)) {
}

template <class T>
bool PerfectGroupJoinExecutor::SinkInternal(Vector &keys, Vector &addresses, Vector &group_ids, bool ignore_nulls) {
	UnifiedVectorFormat key_data;
	keys.ToUnifiedFormat(key_data);
	addresses.Flatten();
	group_ids.Flatten();
	const auto address_data = FlatVector::GetData<data_ptr_t>(addresses);
	const auto group_id_data = FlatVector::GetData<uint64_t>(group_ids);
	const auto input_data = UnifiedVectorFormat::GetData<T>(key_data);
	const auto min_value = minimum.GetValueUnsafe<T>();
	const auto max_value = maximum.GetValueUnsafe<T>();
	for (idx_t row_idx = 0; row_idx < keys.size(); row_idx++) {
		const auto input_idx = key_data.sel->get_index(row_idx);
		if (!key_data.validity.RowIsValid(input_idx)) {
			if (ignore_nulls) {
				continue;
			}
			throw InternalException("PERFECT_GROUP_JOIN owner key unexpectedly contained NULL");
		}
		const auto value = input_data[input_idx];
		if (value < min_value || value > max_value) {
			return false;
		}
		const auto directory_idx = GetDirectoryIndex(value, min_value);
		if (directory_idx > range) {
			return false;
		}
		if (directory[directory_idx]) {
			throw InternalException("PERFECT_GROUP_JOIN owner key uniqueness proof was violated at execution time");
		}
		directory[directory_idx] = address_data[row_idx];
		directory_group_ids[directory_idx] = group_id_data[row_idx];
	}
	return true;
}

bool PerfectGroupJoinExecutor::Sink(Vector &keys, Vector &addresses, Vector &group_ids, bool ignore_nulls) {
	switch (key_type.InternalType()) {
	case PhysicalType::INT8:
		return SinkInternal<int8_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::INT16:
		return SinkInternal<int16_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::INT32:
		return SinkInternal<int32_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::INT64:
		return SinkInternal<int64_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::UINT8:
		return SinkInternal<uint8_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::UINT16:
		return SinkInternal<uint16_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::UINT32:
		return SinkInternal<uint32_t>(keys, addresses, group_ids, ignore_nulls);
	case PhysicalType::UINT64:
		return SinkInternal<uint64_t>(keys, addresses, group_ids, ignore_nulls);
	default:
		throw InternalException("Unsupported PERFECT_GROUP_JOIN key type");
	}
}

template <class T>
idx_t PerfectGroupJoinExecutor::LookupInternal(Vector &keys, Vector &addresses, Vector &group_ids,
                                               SelectionVector &found) const {
	UnifiedVectorFormat key_data;
	keys.ToUnifiedFormat(key_data);
	addresses.SetVectorType(VectorType::FLAT_VECTOR);
	const auto address_data = FlatVector::GetDataMutable<data_ptr_t>(addresses);
	group_ids.SetVectorType(VectorType::FLAT_VECTOR);
	const auto group_id_data = FlatVector::GetDataMutable<uint64_t>(group_ids);
	const auto input_data = UnifiedVectorFormat::GetData<T>(key_data);
	const auto min_value = minimum.GetValueUnsafe<T>();
	const auto max_value = maximum.GetValueUnsafe<T>();
	idx_t found_count = 0;
	for (idx_t row_idx = 0; row_idx < keys.size(); row_idx++) {
		const auto input_idx = key_data.sel->get_index(row_idx);
		if (!key_data.validity.RowIsValid(input_idx)) {
			continue;
		}
		const auto value = input_data[input_idx];
		if (value < min_value || value > max_value) {
			continue;
		}
		const auto directory_idx = GetDirectoryIndex(value, min_value);
		if (directory_idx > range) {
			continue;
		}
		const auto address = directory[directory_idx];
		if (address) {
			address_data[row_idx] = address;
			group_id_data[row_idx] = directory_group_ids[directory_idx];
			found.set_index(found_count++, UnsafeNumericCast<sel_t>(row_idx));
		}
	}
	FlatVector::SetSize(addresses, keys.size());
	FlatVector::SetSize(group_ids, keys.size());
	return found_count;
}

idx_t PerfectGroupJoinExecutor::Lookup(Vector &keys, Vector &addresses, Vector &group_ids,
                                       SelectionVector &found) const {
	switch (key_type.InternalType()) {
	case PhysicalType::INT8:
		return LookupInternal<int8_t>(keys, addresses, group_ids, found);
	case PhysicalType::INT16:
		return LookupInternal<int16_t>(keys, addresses, group_ids, found);
	case PhysicalType::INT32:
		return LookupInternal<int32_t>(keys, addresses, group_ids, found);
	case PhysicalType::INT64:
		return LookupInternal<int64_t>(keys, addresses, group_ids, found);
	case PhysicalType::UINT8:
		return LookupInternal<uint8_t>(keys, addresses, group_ids, found);
	case PhysicalType::UINT16:
		return LookupInternal<uint16_t>(keys, addresses, group_ids, found);
	case PhysicalType::UINT32:
		return LookupInternal<uint32_t>(keys, addresses, group_ids, found);
	case PhysicalType::UINT64:
		return LookupInternal<uint64_t>(keys, addresses, group_ids, found);
	default:
		throw InternalException("Unsupported PERFECT_GROUP_JOIN key type");
	}
}

} // namespace duckdb
