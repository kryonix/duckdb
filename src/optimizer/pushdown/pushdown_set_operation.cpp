#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"

namespace duckdb {

using Filter = FilterPushdown::Filter;

static void ReplaceSetOpBindings(vector<ColumnBinding> &bindings, Filter &filter, unique_ptr<Expression> &root_expr,
                                 LogicalSetOperation &setop) {
	ExpressionIterator::VisitExpressionMutable<BoundColumnRefExpression>(
	    root_expr, [&](BoundColumnRefExpression &colref, unique_ptr<Expression> &expr) {
		    D_ASSERT(colref.Binding().table_index == setop.table_index);
		    D_ASSERT(colref.Depth() == 0);

		    // rewrite the binding by looking into the bound_tables list of the subquery
		    colref.BindingMutable() = bindings[colref.Binding().column_index];
		    filter.bindings.insert(colref.Binding().table_index);
	    });
}

void FilterPushdown::PushdownSetOperation(unique_ptr<LogicalOperator> &op, RewriteContext &context) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_UNION || op->type == LogicalOperatorType::LOGICAL_EXCEPT ||
	         op->type == LogicalOperatorType::LOGICAL_INTERSECT);
	auto &setop = op->Cast<LogicalSetOperation>();

	for (auto &child : op->children) {
		auto child_bindings = child->GetColumnBindings();

		FilterPushdown child_pushdown(optimizer, convert_mark_joins, projection_mode);
		for (auto &original_filter : filters) {
			// first create a copy of the filter
			auto filter = make_uniq<Filter>();
			filter->filter = original_filter->filter->Copy();

			//  rewrite references to the result of the union into references to the child index
			ReplaceSetOpBindings(child_bindings, *filter, filter->filter, setop);

			// extract bindings again
			filter->ExtractBindings();

			// move the filters into the child pushdown nodes
			child_pushdown.filters.push_back(std::move(filter));
		}

		// pushdown into the child
		child_pushdown.Rewrite(child, context);
	}
	bool all_empty = true;
	for (auto &child : op->children) {
		if (child->type != LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
			all_empty = false;
		}
	}
	if (all_empty) {
		// all sides are empty: the result must be empty
		ReplaceWithEmptyResult(op, context);
		return;
	}
	if (op->type == LogicalOperatorType::LOGICAL_UNION) {
		// for UNION (ALL) - delete all empty children and return
		for (idx_t i = op->children.size(); i > 0; i--) {
			auto child_index = i - 1;
			if (op->children[child_index]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
				context.mutator.EraseChild(*op, child_index);
			}
		}
		return;
	}
	bool left_empty = op->children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	bool right_empty = op->children[1]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT;
	if (left_empty && setop.setop_all) {
		// left child is empty result
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_EXCEPT:
			// except: if left child is empty, return empty result
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: if any child is empty, return empty result itself
			ReplaceWithEmptyResult(op, context);
			return;
		default:
			throw InternalException("Unsupported set operation");
		}
	} else if (right_empty && setop.setop_all) {
		// right child is empty result
		switch (op->type) {
		case LogicalOperatorType::LOGICAL_EXCEPT:
			if (op->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				// union or except with empty right child: return left child
				auto &projection = op->children[0]->Cast<LogicalProjection>();
				projection.table_index = setop.table_index;
				context.mutator.PromoteChild(op, 0);
				return;
			}
			break;
		case LogicalOperatorType::LOGICAL_INTERSECT:
			// intersect: if any child is empty, return empty result itself
			ReplaceWithEmptyResult(op, context);
			return;
		default:
			throw InternalException("Unsupported set operation");
		}
	}
}

} // namespace duckdb
