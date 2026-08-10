#include "core_functions/aggregate/distributive_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/operator/aggregate_operators.hpp"
#include "duckdb/common/vector_operations/aggregate_executor.hpp"
#include "duckdb/common/types/bit.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/aggregate_state_layout.hpp"

namespace duckdb {

namespace {

template <class T>
struct BitState {
	using value_type = T;
	using STATE_TYPE = OptionalStateType<T>;
	T value;
	bool is_set;
};

template <class STATE, class INPUT, class RESULT, class OP, bool CHANGE_AWARE>
AggregateFunction GetBitfieldTypedAggregate(const LogicalType &type) {
	if constexpr (CHANGE_AWARE) {
		return AggregateFunction::UnaryAggregateWithChange<STATE, INPUT, RESULT, OP>(type, type);
	}
	return AggregateFunction::UnaryAggregate<STATE, INPUT, RESULT, OP>(type, type);
}

template <class OP, bool CHANGE_AWARE = false>
AggregateFunction GetBitfieldUnaryAggregate(LogicalType type) {
	switch (type.id()) {
	case LogicalTypeId::TINYINT:
		return GetBitfieldTypedAggregate<BitState<uint8_t>, int8_t, int8_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::SMALLINT:
		return GetBitfieldTypedAggregate<BitState<uint16_t>, int16_t, int16_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::INTEGER:
		return GetBitfieldTypedAggregate<BitState<uint32_t>, int32_t, int32_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::BIGINT:
		return GetBitfieldTypedAggregate<BitState<uint64_t>, int64_t, int64_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::HUGEINT:
		return GetBitfieldTypedAggregate<BitState<hugeint_t>, hugeint_t, hugeint_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::UTINYINT:
		return GetBitfieldTypedAggregate<BitState<uint8_t>, uint8_t, uint8_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::USMALLINT:
		return GetBitfieldTypedAggregate<BitState<uint16_t>, uint16_t, uint16_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::UINTEGER:
		return GetBitfieldTypedAggregate<BitState<uint32_t>, uint32_t, uint32_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::UBIGINT:
		return GetBitfieldTypedAggregate<BitState<uint64_t>, uint64_t, uint64_t, OP, CHANGE_AWARE>(type);
	case LogicalTypeId::UHUGEINT:
		return GetBitfieldTypedAggregate<BitState<uhugeint_t>, uhugeint_t, uhugeint_t, OP, CHANGE_AWARE>(type);
	default:
		throw InternalException("Unimplemented bitfield type for unary aggregate");
	}
}

struct BitwiseOperation {
	template <class INPUT_TYPE, class STATE, class OP>
	static bool OperationWithChange(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		if (!state.is_set) {
			OP::template Assign<INPUT_TYPE>(state, input);
			state.is_set = true;
			return true;
		}
		return OP::template ExecuteWithChange<INPUT_TYPE>(state, input);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void Operation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &) {
		if (!state.is_set) {
			OP::template Assign<INPUT_TYPE>(state, input);
			state.is_set = true;
		} else {
			OP::template Execute<INPUT_TYPE>(state, input);
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		OP::template Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
	}

	template <class STATE, class OP>
	static void Combine(const STATE &source, STATE &target, AggregateInputData &) {
		if (!source.is_set) {
			// source is NULL, nothing to do.
			return;
		}
		if (!target.is_set) {
			// target is NULL, use source value directly.
			OP::template Assign<typename STATE::value_type>(target, source.value);
			target.is_set = true;
		} else {
			OP::template Execute<typename STATE::value_type>(target, source.value);
		}
	}

	template <class STATE, class OP>
	static bool CombineWithChange(const STATE &source, STATE &target, AggregateInputData &) {
		if (!source.is_set) {
			return false;
		}
		if (!target.is_set) {
			OP::template Assign<typename STATE::value_type>(target, source.value);
			target.is_set = true;
			return true;
		}
		return OP::template ExecuteWithChange<typename STATE::value_type>(target, source.value);
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE &state, INPUT_TYPE input) {
		state.value = typename STATE::value_type(input);
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.is_set) {
			finalize_data.ReturnNull();
		} else {
			target = T(state.value);
		}
	}

	static bool IgnoreNull() {
		return true;
	}
};

template <class OP>
struct NumericBitwiseOperation : public BitwiseOperation, public ClusteredStateCopy {
	template <class INPUT_TYPE, class STATE>
	static void UpdateClusteredLocal(STATE &local, const INPUT_TYPE &input) {
		if (!local.is_set) {
			Assign(local, input);
			local.is_set = true;
		} else {
			OP::template Execute<INPUT_TYPE>(local, input);
		}
	}

	template <class INPUT_TYPE, class STATE>
	static void UpdateClusteredLocal(STATE &local, const INPUT_TYPE &input, idx_t count) {
		if (count != 0) {
			UpdateClusteredLocal(local, input);
		}
	}
};

template <class OP>
struct SimpleBitwiseOperation : public NumericBitwiseOperation<SimpleBitwiseOperation<OP>> {
	template <class INPUT_TYPE, class STATE>
	static bool ExecuteWithChange(STATE &state, INPUT_TYPE input) {
		const auto previous = state.value;
		Execute(state, input);
		return previous != state.value;
	}

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE &state, INPUT_TYPE input) {
		state.value =
		    OP::template Operation<typename STATE::value_type>(state.value, typename STATE::value_type(input));
	}
};

using BitAndOperation = SimpleBitwiseOperation<BitAnd>;
using BitOrOperation = SimpleBitwiseOperation<BitOr>;

struct BitXorOperation : public NumericBitwiseOperation<BitXorOperation> {
	using NumericBitwiseOperation<BitXorOperation>::UpdateClusteredLocal;

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE &state, INPUT_TYPE input) {
		state.value =
		    BitXor::template Operation<typename STATE::value_type>(state.value, typename STATE::value_type(input));
	}

	template <class INPUT_TYPE, class STATE>
	static void UpdateClusteredLocal(STATE &local, const INPUT_TYPE &input, idx_t count) {
		if ((count & 1) != 0) {
			NumericBitwiseOperation<BitXorOperation>::template UpdateClusteredLocal<INPUT_TYPE>(local, input);
		} else if (count != 0 && !local.is_set) {
			local.value = typename STATE::value_type(0);
			local.is_set = true;
		}
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}
};

using BitStringState = BitState<string_t>;

struct BitStringBitwiseOperation : public BitwiseOperation {
	template <class STATE>
	static void Destroy(STATE &state, AggregateInputData &aggr_input_data) {
		if (state.is_set && !state.value.IsInlined()) {
			delete[] state.value.GetData();
		}
	}

	template <class INPUT_TYPE, class STATE>
	static void Assign(STATE &state, INPUT_TYPE input) {
		D_ASSERT(state.is_set == false);
		if (input.IsInlined()) {
			state.value = input;
		} else { // non-inlined string, need to allocate space for it
			auto len = input.GetSize();
			auto ptr = new char[len];
			memcpy(ptr, input.GetData(), len);

			state.value = string_t(ptr, UnsafeNumericCast<uint32_t>(len));
		}
	}

	template <class T, class STATE>
	static void Finalize(STATE &state, T &target, AggregateFinalizeData &finalize_data) {
		if (!state.is_set) {
			finalize_data.ReturnNull();
		} else {
			target = finalize_data.ReturnString(state.value);
		}
	}
};

struct BitStringAndOperation : public BitStringBitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static bool ExecuteWithChange(STATE &state, INPUT_TYPE input) {
		if (Bit::BitLength(input) != Bit::BitLength(state.value)) {
			Bit::BitwiseAnd(input, state.value, state.value);
			return true;
		}
		const auto state_data = reinterpret_cast<const uint8_t *>(state.value.GetData());
		const auto input_data = reinterpret_cast<const uint8_t *>(input.GetData());
		bool changed = false;
		for (idx_t i = 1; i < state.value.GetSize(); i++) {
			changed = changed || (state_data[i] & input_data[i]) != state_data[i];
		}
		Bit::BitwiseAnd(input, state.value, state.value);
		return changed;
	}

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE &state, INPUT_TYPE input) {
		Bit::BitwiseAnd(input, state.value, state.value);
	}
};

struct BitStringOrOperation : public BitStringBitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static bool ExecuteWithChange(STATE &state, INPUT_TYPE input) {
		if (Bit::BitLength(input) != Bit::BitLength(state.value)) {
			Bit::BitwiseOr(input, state.value, state.value);
			return true;
		}
		const auto state_data = reinterpret_cast<const uint8_t *>(state.value.GetData());
		const auto input_data = reinterpret_cast<const uint8_t *>(input.GetData());
		bool changed = false;
		for (idx_t i = 1; i < state.value.GetSize(); i++) {
			changed = changed || (state_data[i] | input_data[i]) != state_data[i];
		}
		Bit::BitwiseOr(input, state.value, state.value);
		return changed;
	}

	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE &state, INPUT_TYPE input) {
		Bit::BitwiseOr(input, state.value, state.value);
	}
};

struct BitStringXorOperation : public BitStringBitwiseOperation {
	template <class INPUT_TYPE, class STATE>
	static void Execute(STATE &state, INPUT_TYPE input) {
		Bit::BitwiseXor(input, state.value, state.value);
	}

	template <class INPUT_TYPE, class STATE, class OP>
	static void ConstantOperation(STATE &state, const INPUT_TYPE &input, AggregateUnaryInput &unary_input,
	                              idx_t count) {
		for (idx_t i = 0; i < count; i++) {
			Operation<INPUT_TYPE, STATE, OP>(state, input, unary_input);
		}
	}
};

} // namespace

AggregateFunctionSet BitAndFun::GetFunctions() {
	AggregateFunctionSet bit_and;
	for (auto &type : LogicalType::Integral()) {
		bit_and.AddFunction(GetBitfieldUnaryAggregate<BitAndOperation, true>(type));
	}
	bit_and.AddFunction(
	    AggregateFunction::UnaryAggregateWithChange<BitStringState, string_t, string_t, BitStringAndOperation>(
	        LogicalType::BIT, LogicalType::BIT));
	return bit_and;
}

AggregateFunctionSet BitOrFun::GetFunctions() {
	AggregateFunctionSet bit_or;
	for (auto &type : LogicalType::Integral()) {
		bit_or.AddFunction(GetBitfieldUnaryAggregate<BitOrOperation, true>(type));
	}
	bit_or.AddFunction(
	    AggregateFunction::UnaryAggregateWithChange<BitStringState, string_t, string_t, BitStringOrOperation>(
	        LogicalType::BIT, LogicalType::BIT));
	return bit_or;
}

AggregateFunctionSet BitXorFun::GetFunctions() {
	AggregateFunctionSet bit_xor;
	for (auto &type : LogicalType::Integral()) {
		bit_xor.AddFunction(GetBitfieldUnaryAggregate<BitXorOperation>(type));
	}
	bit_xor.AddFunction(AggregateFunction::UnaryAggregate<BitStringState, string_t, string_t, BitStringXorOperation>(
	    LogicalType::BIT, LogicalType::BIT));
	return bit_xor;
}

} // namespace duckdb
