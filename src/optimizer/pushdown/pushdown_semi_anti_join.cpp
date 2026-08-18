#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

void FilterPushdown::PushdownSemiAntiJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	auto &join = op->Cast<LogicalJoin>();

	// push all current filters down the left side
	Rewrite(op->children[0], context);
	FilterPushdown right_pushdown(optimizer, convert_mark_joins, projection_mode);
	right_pushdown.Rewrite(op->children[1], context);

	bool left_empty = op->children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	bool right_empty = op->children[1]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	if (left_empty && right_empty) {
		// both empty: return empty result
		ReplaceWithEmptyResult(op, context);
		return;
	}
	// TODO: if semi/anti join is created from a intersect/except statement, then we can
	//  push filters down into both children.
	// filter pushdown happens before join order optimization, so right_anti and right_semi are not possible yet here
	if (left_empty) {
		// left child is empty result
		switch (join.join_type) {
		case JoinType::ANTI:
		case JoinType::SEMI:
			ReplaceWithEmptyResult(op, context);
			return;
		default:
			break;
		}
	} else if (right_empty) {
		// right child is empty result
		switch (join.join_type) {
		case JoinType::ANTI:
			// just return the left child.
			context.mutator.PromoteChild(op, 0);
			return;
		case JoinType::SEMI:
			ReplaceWithEmptyResult(op, context);
			return;
		default:
			break;
		}
	}
}

} // namespace duckdb
