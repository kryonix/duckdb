#include "duckdb/optimizer/cte_filter_pusher.hpp"

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

namespace duckdb {

CTEFilterPusher::CTEFilterPusher(Optimizer &optimizer_p) : optimizer(optimizer_p) {
}

unique_ptr<LogicalOperator> CTEFilterPusher::Optimize(unique_ptr<LogicalOperator> op) {
	if (!HasMaterializedCTE(*op)) {
		return op;
	}

	RewriteContext context(*op);
	auto materialized_ctes = context.data_flow.GetMaterializedCTEs();
	for (auto it = materialized_ctes.rbegin(); it != materialized_ctes.rend(); it++) {
		auto indexed = GetIndexedCandidate(*it, context);
		if (indexed.all_cte_refs_are_filtered) {
			PushFilterIntoCTE(*it, indexed.filters, context);
		}
	}
	D_ASSERT(context.data_flow.Verify());
	return op;
}

bool CTEFilterPusher::HasMaterializedCTE(LogicalOperator &op) {
	vector<reference<LogicalOperator>> pending {op};
	while (!pending.empty()) {
		auto current = pending.back();
		pending.pop_back();
		if (current.get().type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			return true;
		}
		for (auto &child : current.get().children) {
			pending.push_back(*child);
		}
	}
	return false;
}

CTEFilterPusher::IndexedMaterializedCTEInfo CTEFilterPusher::GetIndexedCandidate(LogicalOperator &materialized_cte_op,
                                                                                 RewriteContext &context) {
	if (materialized_cte_op.type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE ||
	    materialized_cte_op.children.size() != 2) {
		throw InternalException("Indexed materialized CTE candidate has an invalid wrapper");
	}
	auto &materialized_cte = materialized_cte_op.Cast<LogicalMaterializedCTE>();
	auto cte_index = materialized_cte.table_index;
	auto producer = context.data_flow.GetCTEProducer(cte_index);
	if (producer.status != LogicalPlanDataFlowStatus::SUCCESS ||
	    producer.op.get() != materialized_cte.children[0].get()) {
		throw InternalException("Indexed materialized CTE candidate has inconsistent producer lineage");
	}

	IndexedMaterializedCTEInfo result;
	auto readers = context.data_flow.GetCTEReaders(cte_index);
	if (readers.status != LogicalPlanDataFlowStatus::SUCCESS) {
		throw InternalException("Indexed materialized CTE candidate has inconsistent reader lineage");
	}
	for (auto &reader : readers.readers) {
		if (reader.get().type != LogicalOperatorType::LOGICAL_CTE_REF) {
			throw InternalException("Indexed materialized CTE lineage contains a non-reader operator");
		}
		auto &cte_ref = reader.get().Cast<LogicalCTERef>();
		if (cte_ref.cte_index != cte_index || cte_ref.is_recurring) {
			throw InternalException("Indexed materialized CTE lineage contains an invalid reader");
		}
		auto reader_parent = context.data_flow.GetOwnershipParent(reader.get());
		if (reader_parent.status != LogicalPlanDataFlowStatus::SUCCESS) {
			throw InternalException("Indexed materialized CTE reader has inconsistent ownership");
		}
		if (!reader_parent.parent || reader_parent.child_index != 0 ||
		    reader_parent.parent->type != LogicalOperatorType::LOGICAL_FILTER ||
		    reader_parent.parent->children[0].get() != &reader.get()) {
			result.all_cte_refs_are_filtered = false;
			continue;
		}
		result.filters.push_back(*reader_parent.parent);
	}
	return result;
}

void CTEFilterPusher::PushFilterIntoCTE(LogicalOperator &materialized_cte,
                                        const vector<reference<LogicalOperator>> &filters, RewriteContext &context) {
	D_ASSERT(materialized_cte.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE);
	if (filters.empty()) {
		return;
	}

	// Create an OR expression with all the filters on all references of the CTE
	unique_ptr<Expression> outer_expr;
	for (auto &filter : filters) {
		D_ASSERT(filter.get().type == LogicalOperatorType::LOGICAL_FILTER);

		auto old_bindings = filter.get().children[0]->GetColumnBindings();
		auto new_bindings = materialized_cte.children[0]->GetColumnBindings();
		D_ASSERT(old_bindings.size() == new_bindings.size());

		ColumnBindingReplacer replacer;
		replacer.AddReplacements(old_bindings, new_bindings);

		// We copy the filters and replace the CTE reference bindings with the bindings in the CTE definition
		unique_ptr<Expression> inner_expr;
		for (auto &filter_expr : filter.get().expressions) {
			auto filter_expr_copy = filter_expr->Copy();
			replacer.VisitExpression(&filter_expr_copy);
			if (inner_expr) {
				inner_expr = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND,
				                                                   std::move(inner_expr), std::move(filter_expr_copy));
			} else {
				inner_expr = std::move(filter_expr_copy);
			}
		}

		if (outer_expr) {
			outer_expr = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_OR, std::move(outer_expr),
			                                                   std::move(inner_expr));
		} else {
			outer_expr = std::move(inner_expr);
		}
	}

	// Add the filter on top of the CTE definition and split the predicates
	auto new_cte = make_uniq_base<LogicalOperator, LogicalFilter>(std::move(outer_expr));
	LogicalFilter::SplitPredicates(new_cte->Cast<LogicalFilter>().expressions);

	// Rewrite the operator expressions before adding the child op (children should be rewritten already)
	optimizer.rewriter.VisitOperator(*new_cte);
	context.mutator.InsertUnary(materialized_cte.children[0], std::move(new_cte));

	// Push down the filter
	FilterPushdown pushdown(optimizer, true, FilterPushdown::ProjectionMode::PRESERVE_COMPUTED_EXPRESSIONS);
	FilterPushdown::RewriteContext pushdown_context(context.data_flow, context.mutator);
	pushdown.Rewrite(materialized_cte.children[0], pushdown_context);
	D_ASSERT(context.data_flow.Verify());
}

} // namespace duckdb
