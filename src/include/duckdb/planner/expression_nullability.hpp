//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/expression_nullability.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/table_index.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

class ClientContext;
class Expression;
class LogicalOperator;
class LogicalPlanDataFlow;

//! Conservatively proves that an expression cannot be NULL at a logical operator's output.
class NotNullExpressionAnalyzer {
public:
	NotNullExpressionAnalyzer(ClientContext &context, LogicalPlanDataFlow &data_flow);

	bool IsNotNull(LogicalOperator &op, const Expression &expr);

private:
	bool IsNotNull(LogicalOperator &op, const Expression &expr, vector<TableIndex> &seen_ctes);

private:
	ClientContext &context;
	LogicalPlanDataFlow &data_flow;
};

} // namespace duckdb
