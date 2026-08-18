#include "duckdb/planner/expression_nullability.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"

#include <algorithm>

namespace duckdb {

static bool GetColumnRefBinding(const Expression &expr, ColumnBinding &binding) {
	if (expr.GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &colref = expr.Cast<BoundColumnRefExpression>();
	if (colref.Depth() != 0) {
		return false;
	}
	binding = colref.Binding();
	return true;
}

static bool FilterRejectsNull(const Expression &filter, const Expression &expr) {
	if (filter.GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
		auto &conjunction = filter.Cast<BoundConjunctionExpression>();
		for (auto &child : conjunction.GetChildren()) {
			if (FilterRejectsNull(*child, expr)) {
				return true;
			}
		}
		return false;
	}
	if (filter.GetExpressionType() == ExpressionType::CONJUNCTION_OR) {
		auto &conjunction = filter.Cast<BoundConjunctionExpression>();
		if (conjunction.GetChildren().empty()) {
			return false;
		}
		for (auto &child : conjunction.GetChildren()) {
			if (!FilterRejectsNull(*child, expr)) {
				return false;
			}
		}
		return true;
	}
	if (filter.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
		auto &op = filter.Cast<BoundOperatorExpression>();
		return !op.GetChildren().empty() && Expression::Equals(*op.GetChildren()[0], expr);
	}
	if (!BoundComparisonExpression::IsComparison(filter) ||
	    filter.GetExpressionType() == ExpressionType::COMPARE_DISTINCT_FROM ||
	    filter.GetExpressionType() == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
		return false;
	}
	auto &comparison = filter.Cast<BoundFunctionExpression>();
	return Expression::Equals(BoundComparisonExpression::Left(comparison), expr) ||
	       Expression::Equals(BoundComparisonExpression::Right(comparison), expr);
}

NotNullExpressionAnalyzer::NotNullExpressionAnalyzer(ClientContext &context_p, LogicalPlanDataFlow &data_flow_p)
    : context(context_p), data_flow(data_flow_p) {
}

bool NotNullExpressionAnalyzer::IsNotNull(LogicalOperator &op, const Expression &expr, vector<TableIndex> &seen_ctes) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto &projection = op.Cast<LogicalProjection>();
		ColumnBinding binding;
		if (projection.children.size() != 1 || !GetColumnRefBinding(expr, binding) ||
		    binding.table_index != projection.table_index) {
			return false;
		}
		auto column_index = binding.column_index.GetIndex();
		if (column_index >= projection.expressions.size()) {
			return false;
		}
		return IsNotNull(*projection.children[0], *projection.expressions[column_index], seen_ctes);
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		auto &filter = op.Cast<LogicalFilter>();
		for (auto &filter_expr : filter.expressions) {
			if (FilterRejectsNull(*filter_expr, expr)) {
				return true;
			}
		}
		return filter.children.size() == 1 && IsNotNull(*filter.children[0], expr, seen_ctes);
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		ColumnBinding binding;
		if (!GetColumnRefBinding(expr, binding)) {
			return false;
		}
		auto source = data_flow.ResolveSource(binding, 0, op);
		if (source.status != LogicalPlanDataFlowStatus::SUCCESS) {
			return false;
		}
		LogicalPlanPathSummary boundary_property;
		boundary_property.Add(LogicalPlanPathProperty::NULLABILITY_BOUNDARY);
		auto boundary = data_flow.FindFirstPathOperator(op, *source.op, boundary_property);
		if (boundary.status == LogicalPlanDataFlowStatus::PATH_PROPERTY_NOT_FOUND) {
			return false;
		}
		if (boundary.status != LogicalPlanDataFlowStatus::SUCCESS || boundary.op.get() == &op) {
			return false;
		}
		auto path = data_flow.GetPathSummary(op, *boundary.op);
		if (path.status != LogicalPlanDataFlowStatus::SUCCESS ||
		    path.summary.Has(LogicalPlanPathProperty::NULL_EXTENDING)) {
			return false;
		}
		return IsNotNull(*boundary.op, expr, seen_ctes);
	}
	case LogicalOperatorType::LOGICAL_GET: {
		ColumnBinding binding;
		if (!GetColumnRefBinding(expr, binding)) {
			return false;
		}
		auto &get = op.Cast<LogicalGet>();
		if (binding.table_index != get.table_index) {
			return false;
		}
		if (get.table_filters.HasFilter(binding.column_index)) {
			auto column_expr = make_uniq<BoundColumnRefExpression>(expr.GetReturnType(), binding);
			auto filter_expr =
			    get.table_filters.GetFilterByColumnIndex(binding.column_index).ToExpression(*column_expr);
			if (FilterRejectsNull(*filter_expr, expr)) {
				return true;
			}
		}
		auto table = get.GetTable();
		if (!table) {
			return false;
		}
		auto &column_index = get.GetColumnIndex(binding);
		if (!column_index.HasPrimaryIndex() || column_index.HasChildren() ||
		    column_index.GetPrimaryIndex() == DConstants::INVALID_INDEX) {
			return false;
		}
		auto stats = table->GetStatistics(context, column_index.GetPrimaryIndex());
		return stats && !stats->CanHaveNull();
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		ColumnBinding binding;
		if (!GetColumnRefBinding(expr, binding)) {
			return false;
		}
		auto &cte_ref = op.Cast<LogicalCTERef>();
		if (binding.table_index != cte_ref.table_index ||
		    std::find(seen_ctes.begin(), seen_ctes.end(), cte_ref.cte_index) != seen_ctes.end()) {
			return false;
		}
		auto producer = data_flow.GetCTEProducer(cte_ref.cte_index);
		if (producer.status != LogicalPlanDataFlowStatus::SUCCESS ||
		    producer.op->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
			return false;
		}
		auto column_index = binding.column_index.GetIndex();
		auto source_bindings = producer.op->GetColumnBindings();
		if (column_index >= source_bindings.size() || column_index >= producer.op->types.size()) {
			return false;
		}
		seen_ctes.push_back(cte_ref.cte_index);
		auto source_expr = BoundColumnRefExpression(producer.op->types[column_index], source_bindings[column_index]);
		auto result = IsNotNull(*producer.op, source_expr, seen_ctes);
		seen_ctes.pop_back();
		return result;
	}
	default:
		return false;
	}
}

bool NotNullExpressionAnalyzer::IsNotNull(LogicalOperator &op, const Expression &expr) {
	vector<TableIndex> seen_ctes;
	return IsNotNull(op, expr, seen_ctes);
}

} // namespace duckdb
