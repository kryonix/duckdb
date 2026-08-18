#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

void FilterPushdown::PushdownInnerJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	auto &join = op->Cast<LogicalJoin>();
	D_ASSERT(join.join_type == JoinType::INNER);
	if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN) {
		if (PushFiltersIntoDelimJoin(op, context)) {
			FinishPushdown(op, context);
		}
		return;
	}
	// inner join: gather all the conditions of the inner join and add to the filter list
	if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &any_join = join.Cast<LogicalAnyJoin>();
		// any join: only one filter to add
		if (AddFilter(std::move(any_join.condition)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			ReplaceWithEmptyResult(op, context);
			return;
		}
	} else {
		// comparison join
		D_ASSERT(op->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
		auto &comp_join = join.Cast<LogicalComparisonJoin>();
		// turn the conditions into filters
		for (auto &i : comp_join.conditions) {
			auto condition = JoinCondition::CreateExpression(std::move(i));
			if (AddFilter(std::move(condition)) == FilterResult::UNSATISFIABLE) {
				// filter statically evaluates to false, strip tree
				ReplaceWithEmptyResult(op, context);
				return;
			}
		}
	}
	GenerateFilters();

	// turn the inner join into a cross product
	auto cross_product = make_uniq<LogicalCrossProduct>();

	// preserve the estimated cardinality of the operator
	if (op->has_estimated_cardinality) {
		cross_product->SetEstimatedCardinality(op->estimated_cardinality);
	}
	context.mutator.ReplaceOperator(op, std::move(cross_product));

	// then push down cross product
	PushdownCrossProduct(op, context);
}

} // namespace duckdb
