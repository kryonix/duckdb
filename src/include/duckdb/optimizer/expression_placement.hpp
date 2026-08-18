//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/expression_placement.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"

namespace duckdb {

class Binder;
class ClientContext;
class LogicalOperator;

//! Places scalar expressions before hash-join build boundaries when this reduces the materialized payload.
class ExpressionPlacementOptimizer {
public:
	ExpressionPlacementOptimizer(Binder &binder, ClientContext &context);

	void Optimize(LogicalOperator &root);

private:
	Binder &binder;
	ClientContext &context;
};

} // namespace duckdb
