//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar_function.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/function.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/common/vector_operations/unary_loops.hpp"
#include "duckdb/common/vector_operations/binary_loops.hpp"
#include "duckdb/execution/expression_executor_state.hpp"

namespace duckdb {
class BoundFunctionExpression;
class ScalarFunctionCatalogEntry;

//! The type used for scalar functions
typedef void (*scalar_function_t)(DataChunk &input, ExpressionState &state, Vector &result);
//! Binds the scalar function and creates the function data
typedef unique_ptr<FunctionData> (*bind_scalar_function_t)(BoundFunctionExpression &expr, ClientContext &context);
//! Adds the dependencies of this BoundFunctionExpression to the set of dependencies
typedef void (*dependency_function_t)(BoundFunctionExpression &expr, unordered_set<CatalogEntry *> &dependencies);

class ScalarFunction : public SimpleFunction {
public:
	ScalarFunction(string name, vector<SQLType> arguments, SQLType return_type, scalar_function_t function,
	               bool has_side_effects = false, bind_scalar_function_t bind = nullptr,
	               dependency_function_t dependency = nullptr)
	    : SimpleFunction(name, arguments, return_type, has_side_effects), function(function), bind(bind),
	      dependency(dependency) {
	}

	ScalarFunction(vector<SQLType> arguments, SQLType return_type, scalar_function_t function,
	               bool has_side_effects = false, bind_scalar_function_t bind = nullptr,
	               dependency_function_t dependency = nullptr)
	    : ScalarFunction(string(), arguments, return_type, function, has_side_effects, bind, dependency) {
	}

	//! The main scalar function to execute
	scalar_function_t function;
	//! The bind function (if any)
	bind_scalar_function_t bind;
	// The dependency function (if any)
	dependency_function_t dependency;

	static unique_ptr<BoundFunctionExpression> BindScalarFunction(ClientContext &context, string schema, string name,
	                                                              vector<SQLType> &arguments,
	                                                              vector<unique_ptr<Expression>> children,
	                                                              bool is_operator = false);
	static unique_ptr<BoundFunctionExpression>
	BindScalarFunction(ClientContext &context, ScalarFunctionCatalogEntry &function, vector<SQLType> &arguments,
	                   vector<unique_ptr<Expression>> children, bool is_operator = false);

	bool operator==(const ScalarFunction &rhs) const {
		return function == rhs.function && bind == rhs.bind && dependency == rhs.dependency;
	}
	bool operator!=(const ScalarFunction &rhs) const {
		return !(*this == rhs);
	}

public:
	static void NopFunction(DataChunk &input, ExpressionState &state, Vector &result) {
		assert(input.column_count >= 1);
		result.Reference(input.data[0]);
	};

	template <class TA, class TR, class OP, bool SKIP_NULLS = false>
	static void UnaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
		assert(input.column_count >= 1);
		templated_unary_loop<TA, TR, OP, SKIP_NULLS>(input.data[0], result);
	};

	template <class TA, class TB, class TR, class OP, bool IGNORE_NULL = false>
	static void BinaryFunction(DataChunk &input, ExpressionState &state, Vector &result) {
		assert(input.column_count == 2);
		templated_binary_loop<TA, TB, TR, OP, IGNORE_NULL>(input.data[0], input.data[1], result);
	};

public:
	template <class OP> static scalar_function_t GetScalarUnaryFunction(SQLType type) {
		switch (type.id) {
		case SQLTypeId::TINYINT:
			return ScalarFunction::UnaryFunction<int8_t, int8_t, OP>;
		case SQLTypeId::SMALLINT:
			return ScalarFunction::UnaryFunction<int16_t, int16_t, OP>;
		case SQLTypeId::INTEGER:
			return ScalarFunction::UnaryFunction<int32_t, int32_t, OP>;
		case SQLTypeId::BIGINT:
			return ScalarFunction::UnaryFunction<int64_t, int64_t, OP>;
		case SQLTypeId::FLOAT:
			return ScalarFunction::UnaryFunction<float, float, OP>;
		case SQLTypeId::DOUBLE:
			return ScalarFunction::UnaryFunction<double, double, OP>;
		case SQLTypeId::DECIMAL:
			return ScalarFunction::UnaryFunction<double, double, OP>;
		default:
			throw NotImplementedException("Unimplemented type for GetScalarUnaryFunction");
		}
	}

	template <class TR, class OP> static scalar_function_t GetScalarUnaryFunctionFixedReturn(SQLType type) {
		switch (type.id) {
		case SQLTypeId::TINYINT:
			return ScalarFunction::UnaryFunction<int8_t, TR, OP>;
		case SQLTypeId::SMALLINT:
			return ScalarFunction::UnaryFunction<int16_t, TR, OP>;
		case SQLTypeId::INTEGER:
			return ScalarFunction::UnaryFunction<int32_t, TR, OP>;
		case SQLTypeId::BIGINT:
			return ScalarFunction::UnaryFunction<int64_t, TR, OP>;
		case SQLTypeId::FLOAT:
			return ScalarFunction::UnaryFunction<float, TR, OP>;
		case SQLTypeId::DOUBLE:
			return ScalarFunction::UnaryFunction<double, TR, OP>;
		case SQLTypeId::DECIMAL:
			return ScalarFunction::UnaryFunction<double, TR, OP>;
		default:
			throw NotImplementedException("Unimplemented type for GetScalarUnaryFunctionFixedReturn");
		}
	}

	template <class OP> static scalar_function_t GetScalarIntegerBinaryFunction(SQLType type) {
		switch (type.id) {
		case SQLTypeId::TINYINT:
			return ScalarFunction::BinaryFunction<int8_t, int8_t, int8_t, OP>;
		case SQLTypeId::SMALLINT:
			return ScalarFunction::BinaryFunction<int16_t, int16_t, int16_t, OP>;
		case SQLTypeId::INTEGER:
			return ScalarFunction::BinaryFunction<int32_t, int32_t, int32_t, OP>;
		case SQLTypeId::BIGINT:
			return ScalarFunction::BinaryFunction<int64_t, int64_t, int64_t, OP>;
		default:
			throw NotImplementedException("Unimplemented type for GetScalarIntegerBinaryFunction");
		}
	}
	template <class TB, class OP> static scalar_function_t GetScalarBinaryFunctionFixedArgument(SQLType type) {
		switch (type.id) {
		case SQLTypeId::TINYINT:
			return ScalarFunction::BinaryFunction<int8_t, TB, int8_t, OP>;
		case SQLTypeId::SMALLINT:
			return ScalarFunction::BinaryFunction<int16_t, TB, int16_t, OP>;
		case SQLTypeId::INTEGER:
			return ScalarFunction::BinaryFunction<int32_t, TB, int32_t, OP>;
		case SQLTypeId::BIGINT:
			return ScalarFunction::BinaryFunction<int64_t, TB, int64_t, OP>;
		case SQLTypeId::FLOAT:
			return ScalarFunction::BinaryFunction<float, TB, float, OP>;
		case SQLTypeId::DOUBLE:
			return ScalarFunction::BinaryFunction<double, TB, double, OP>;
		case SQLTypeId::DECIMAL:
			return ScalarFunction::BinaryFunction<double, TB, double, OP>;
		default:
			throw NotImplementedException("Unimplemented type for GetScalarUnaryFunctionFixedReturn");
		}
	}
};

} // namespace duckdb
