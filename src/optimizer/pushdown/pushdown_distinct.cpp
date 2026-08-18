#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"

namespace duckdb {

void FilterPushdown::PushdownDistinct(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_DISTINCT);
	auto &distinct = op->Cast<LogicalDistinct>();
	if (!distinct.order_by) {
		// regular DISTINCT - can just push down
		Rewrite(op->children[0], context);
		return;
	}
	// no pushdown through DISTINCT ON (yet?)
	FinishPushdown(op, context);
}

} // namespace duckdb
