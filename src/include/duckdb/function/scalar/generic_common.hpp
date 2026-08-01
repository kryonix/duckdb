//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/generic_common.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {
class BoundFunctionExpression;
class ArenaAllocator;
class Vector;
struct AggregateObject;

struct ConstantOrNull {
	static unique_ptr<FunctionData> Bind(Value value);
	static bool IsConstantOrNull(BoundFunctionExpression &expr, const Value &val);
};

struct ExportAggregateFunctionBindData : public FunctionData {
	unique_ptr<BoundAggregateExpression> aggregate;
	explicit ExportAggregateFunctionBindData(unique_ptr<Expression> aggregate_p);
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
};

struct ExportAggregateFunction {
	static unique_ptr<BoundAggregateExpression> Bind(unique_ptr<BoundAggregateExpression> child_aggregate);
	static void SetStateExport(BoundAggregateExpression &aggregate, LogicalType state_layout);
	//! Imports serialized states and combines them into initialized target state addresses.
	static void CombineStates(const AggregateObject &aggregate, Vector &serialized_states, Vector &target_states,
	                          ArenaAllocator &allocator);
};

} // namespace duckdb
