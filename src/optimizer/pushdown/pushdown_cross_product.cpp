#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

void FilterPushdown::PushdownCrossProduct(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(op->children.size() > 1);
	FilterPushdown left_pushdown(optimizer, convert_mark_joins, projection_mode);
	FilterPushdown right_pushdown(optimizer, convert_mark_joins, projection_mode);
	vector<unique_ptr<Expression>> join_expressions;
	auto join_ref_type = JoinRefType::REGULAR;
	switch (op->type) {
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		break;
	default:
		throw InternalException("Unsupported join type for cross product push down");
	}
	JoinBindingState binding_state(*op, context.data_flow);
	if (!filters.empty()) {
		// now check the set of filters
		for (auto &f : filters) {
			auto side = GetJoinSide(*f, JoinDecisionPolicy::CROSS_PRODUCT, binding_state);
			if (side == JoinSide::LEFT) {
				// bindings match left side: push into left
				left_pushdown.filters.push_back(std::move(f));
			} else if (side == JoinSide::RIGHT) {
				right_pushdown.filters.push_back(std::move(f));
			} else {
				D_ASSERT(side == JoinSide::BOTH || side == JoinSide::NONE);
				// bindings match both: turn into join condition
				join_expressions.push_back(std::move(f->filter));
			}
		}
	}
	vector<JoinCondition> conditions;
	const bool create_join = !join_expressions.empty();
	if (create_join) {
		// join conditions found: turn into inner join
		// extract join conditions
		const auto join_type = JoinType::INNER;
		LogicalComparisonJoin::ExtractJoinConditionsWithoutPushdown(
		    GetContext(), join_type, join_ref_type,
		    [&](const Expression &expression) {
			    return GetJoinSide(expression, JoinDecisionPolicy::CROSS_PRODUCT, binding_state);
		    },
		    join_expressions, conditions);
	}

	left_pushdown.Rewrite(op->children[0], context);
	right_pushdown.Rewrite(op->children[1], context);

	if (create_join) {
		const auto join_type = JoinType::INNER;
		// create the join from the join conditions
		auto new_op = LogicalComparisonJoin::CreateJoin(join_type, join_ref_type, std::move(conditions));

		// possible cases are: AnyJoin, ComparisonJoin, or Filter + ComparisonJoin
		if (op->has_estimated_cardinality) {
			// set the estimated cardinality of the new operator
			new_op->SetEstimatedCardinality(op->estimated_cardinality);
		}
		context.mutator.ReplaceOperator(op, std::move(new_op));
	} else {
		// no join conditions found: keep as cross product
		D_ASSERT(op->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
		return;
	}
}

} // namespace duckdb
