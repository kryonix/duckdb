#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"

namespace duckdb {

void FilterPushdown::PushdownLimit(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	auto &limit = op->Cast<LogicalLimit>();

	if (limit.limit_val.Type() == LimitNodeType::CONSTANT_VALUE && limit.limit_val.GetConstantValue() == 0) {
		ReplaceWithEmptyResult(op, context);
		return;
	}

	FinishPushdown(op, context);
}

} // namespace duckdb
