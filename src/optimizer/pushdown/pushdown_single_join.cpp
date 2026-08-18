#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

void FilterPushdown::PushdownSingleJoin(unique_ptr<LogicalOperator> &op, JoinBindingState &binding_state,
                                        RewriteContext &context) {
	D_ASSERT(op->Cast<LogicalJoin>().join_type == JoinType::SINGLE);
	FilterPushdown left_pushdown(optimizer, convert_mark_joins, projection_mode);
	FilterPushdown right_pushdown(optimizer, convert_mark_joins, projection_mode);
	// now check the set of filters
	for (idx_t i = 0; i < filters.size(); i++) {
		auto side = GetJoinSide(*filters[i], JoinDecisionPolicy::SINGLE_JOIN, binding_state);
		if (side == JoinSide::LEFT) {
			// bindings match left side: push into left
			left_pushdown.filters.push_back(std::move(filters[i]));
			// erase the filter from the list of filters
			filters.erase_at(i);
			i--;
		}
	}
	left_pushdown.Rewrite(op->children[0], context);
	right_pushdown.Rewrite(op->children[1], context);
	PushFinalFilters(op, context);
}

} // namespace duckdb
