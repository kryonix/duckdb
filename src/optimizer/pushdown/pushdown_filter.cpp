#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

void FilterPushdown::PushdownFilter(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_FILTER);
	auto &filter = op->Cast<LogicalFilter>();
	if (filter.HasProjectionMap()) {
		return FinishPushdown(op, context);
	}
	// filter: gather the filters and remove the filter from the set of operations
	for (auto &expression : filter.expressions) {
		if (AddFilter(std::move(expression)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			ReplaceWithEmptyResult(op, context);
			return;
		}
	}
	GenerateFilters();
	context.mutator.RemoveUnary(op);
	Rewrite(op, context);
}

} // namespace duckdb
