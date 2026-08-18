#include "duckdb/optimizer/projection_pullup.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

class ProjectionPullupDataFlowContext {
public:
	void Ensure(LogicalOperator &root) {
		if (data_flow) {
			return;
		}
		data_flow = make_uniq<LogicalPlanDataFlow>(root);
		mutator = make_uniq<LogicalPlanDataFlowMutator>(*data_flow);
	}

	LogicalPlanDataFlow &GetDataFlow(LogicalOperator &root) {
		Ensure(root);
		return *data_flow;
	}

	LogicalPlanDataFlowMutator &GetMutator(LogicalOperator &root) {
		Ensure(root);
		return *mutator;
	}

private:
	unique_ptr<LogicalPlanDataFlow> data_flow;
	unique_ptr<LogicalPlanDataFlowMutator> mutator;
};

ProjectionPullup::ProjectionPullup(Optimizer &optimizer_p, unique_ptr<LogicalOperator> &root_p)
    : optimizer(optimizer_p), root(root_p) {
}

ProjectionPullup::ProjectionPullup(Optimizer &optimizer_p, unique_ptr<LogicalOperator> &root_p,
                                   ProjectionPullupDataFlowContext &context_p)
    : context(context_p), optimizer(optimizer_p), root(root_p) {
}

ProjectionPullup::~ProjectionPullup() {
}

ProjectionPullupDataFlowContext &ProjectionPullup::GetDataFlowContext() {
	if (!context) {
		owned_context = make_uniq<ProjectionPullupDataFlowContext>();
		context = *owned_context;
	}
	return *context;
}

void ProjectionPullup::PopParents(const LogicalOperator &op) {
	// pop back elements until the last operator in the stack is THIS operator
	while (!parents.empty() && &parents.back().get() != &op) {
		parents.pop_back();
	}
	// then pop THIS operator back, and stop
	if (!parents.empty()) {
		parents.pop_back();
	}
}

void ProjectionPullup::InsertProjectionBelowOp(unique_ptr<LogicalOperator> &child) {
	if (child->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		child->ResolveOperatorTypes();
		auto proj_index = optimizer.binder.GenerateTableIndex();
		auto child_bindings = child->GetColumnBindings();
		const auto child_types = child->types;
		const auto column_count = child_bindings.size();

		vector<unique_ptr<Expression>> expressions;
		expressions.reserve(column_count);
		for (idx_t i = 0; i < column_count; i++) {
			expressions.push_back(make_uniq<BoundColumnRefExpression>(child_types[i], child_bindings[i]));
		}

		BindingReplacementGraph replacements;
		for (idx_t col_idx = 0; col_idx < column_count; col_idx++) {
			const auto &old_binding = child_bindings[col_idx];
			replacements.Add(old_binding, ColumnBinding(proj_index, ProjectionIndex(col_idx)));
		}

		auto new_projection = make_uniq<LogicalProjection>(proj_index, std::move(expressions));
		if (child->has_estimated_cardinality) {
			new_projection->SetEstimatedCardinality(child->estimated_cardinality);
		}

		auto &rewrite_context = GetDataFlowContext();
		auto &data_flow = rewrite_context.GetDataFlow(*root);
		auto &mutator = rewrite_context.GetMutator(*root);
		ColumnBindingRewrite::InsertUnaryAndRewriteBindings(data_flow, mutator, child, std::move(new_projection),
		                                                    replacements);
	}
	ProjectionPullup next(optimizer, root, GetDataFlowContext());
	next.Optimize(child->children[0]);
}

void ProjectionPullup::PullUpColrefProjection(unique_ptr<LogicalOperator> &op, LogicalProjection &proj,
                                              vector<ColumnBinding> &proj_bindings) {
	// LOGICAL_DISTINCT sets `everything_referenced = true` in RemoveUnusedColumns
	// for its subtree. The projection above it acts as a binding barrier; removing
	// it lets upstream references point past DISTINCT and breaks column pruning
	// down to READ_PARQUET. Repro: TPC-DS Q54 regresses 4-5x without this guard.
	if (proj.children[0]->type == LogicalOperatorType::LOGICAL_DISTINCT) {
		ProjectionPullup next(optimizer, root, GetDataFlowContext());
		next.Optimize(proj.children[0]);
		return;
	}
	BindingReplacementGraph replacements;
	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		auto &colref = proj.expressions[i]->Cast<BoundColumnRefExpression>();
		replacements.Add(proj_bindings[i], colref.Binding());
	}

	// Re-run optimization after removing this projection.
	// Binding rewrites can make parent projections redundant, and without
	// another pass they would not be eliminated.
	auto &rewrite_context = GetDataFlowContext();
	auto &data_flow = rewrite_context.GetDataFlow(*root);
	auto &mutator = rewrite_context.GetMutator(*root);
	ColumnBindingRewrite::PromoteChildAndRewriteBindings(data_flow, mutator, op, 0, replacements);
	Optimize(op);

	return;
}
void ProjectionPullup::PullUpNonColrefProjection(unique_ptr<LogicalOperator> &op, LogicalProjection &proj,
                                                 vector<ColumnBinding> &proj_bindings, idx_t pull_up_to_here) {
	// Not all expressions are colrefs. We can pull up instead of removing
	for (idx_t i = 0; i < parents.size(); i++) {
		LogicalOperator &parent_op = parents[i].get();

		// Do not pull non-colref expressions through outer joins.
		// non-colref expressions on the nullable side of a LEFT/RIGHT/OUTER JOIN must not be pulled above the
		// join. If pulled up, expressions (e.g COALESCE) evaluate after the join and return non-null for
		// unmatched rows instead of null.
		if (parent_op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    parent_op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &join = parent_op.Cast<LogicalComparisonJoin>();
			if (join.join_type == JoinType::LEFT || join.join_type == JoinType::RIGHT ||
			    join.join_type == JoinType::OUTER) {
				// Recurse into child without pulling up
				ProjectionPullup next(optimizer, root, GetDataFlowContext());
				next.Optimize(proj.children[0]);
				return;
			}
		}
	}
	LogicalOperator &insert_at_node = parents[parents.size() - pull_up_to_here].get();
	auto &rewrite_context = GetDataFlowContext();
	auto &data_flow = rewrite_context.GetDataFlow(*root);
	auto &mutator = rewrite_context.GetMutator(*root);
	auto insert_owner = data_flow.GetOwnershipParent(insert_at_node);
	if (insert_owner.status != LogicalPlanDataFlowStatus::SUCCESS) {
		throw InternalException("Cannot locate the indexed projection pullup target");
	}
	reference<unique_ptr<LogicalOperator>> insert_slot(root);
	if (insert_owner.parent) {
		if (insert_owner.child_index >= insert_owner.parent->children.size() ||
		    insert_owner.parent->children[insert_owner.child_index].get() != &insert_at_node) {
			throw InternalException("Indexed projection pullup target has an invalid ownership slot");
		}
		insert_slot = insert_owner.parent->children[insert_owner.child_index];
	}

	BindingReplacementGraph inner_replacements;
	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		if (proj.expressions[i]->GetExpressionType() == ExpressionType::BOUND_COLUMN_REF) {
			auto &colref = proj.expressions[i]->Cast<BoundColumnRefExpression>();
			inner_replacements.Add(proj_bindings[i], colref.Binding());
		}
	}

	insert_at_node.ResolveOperatorTypes();
	auto insert_bindings = insert_at_node.GetColumnBindings();
	const auto insert_types = insert_at_node.types;
	column_binding_set_t existing_bindings(proj_bindings.begin(), proj_bindings.end());
	BindingReplacementGraph outer_replacements;
	vector<unique_ptr<Expression>> pass_through_expressions;
	idx_t next_col = proj.expressions.size();
	for (idx_t i = 0; i < insert_bindings.size(); i++) {
		if (existing_bindings.find(insert_bindings[i]) == existing_bindings.end()) {
			pass_through_expressions.push_back(
			    make_uniq<BoundColumnRefExpression>(insert_types[i], insert_bindings[i]));
			outer_replacements.Add(insert_bindings[i], ColumnBinding(proj.table_index, ProjectionIndex(next_col)));
			next_col++;
		}
	}
	proj.expressions.reserve(proj.expressions.size() + pass_through_expressions.size());

	vector<reference<LogicalOperator>> outer_ancestors;
	auto current = insert_owner;
	while (current.parent) {
		outer_ancestors.push_back(*current.parent);
		current = data_flow.GetOwnershipParent(*current.parent);
		if (current.status != LogicalPlanDataFlowStatus::SUCCESS) {
			throw InternalException("Cannot locate an indexed projection pullup ancestor");
		}
	}

	auto mutation = mutator.BeginMutation();
	for (idx_t i = 0; i < pull_up_to_here; i++) {
		ColumnBindingRewrite::ApplyToOperatorBindings(parents[i], inner_replacements);
		mutator.RefreshOperator(parents[i]);
	}
	auto projection_to_move = mutator.RemoveUnary(op);
	auto &moved_projection = projection_to_move->Cast<LogicalProjection>();
	for (auto &expression : pass_through_expressions) {
		moved_projection.expressions.push_back(std::move(expression));
	}
	mutator.InsertUnary(insert_slot, std::move(projection_to_move));
	for (auto &ancestor : outer_ancestors) {
		ColumnBindingRewrite::ApplyToOperatorBindings(ancestor, outer_replacements);
		mutator.RefreshOperator(ancestor);
	}
}

void ProjectionPullup::CanPullThrough(column_binding_map_t<unique_ptr<Expression>> &projection_map,
                                      bool &can_pull_through) {
	// if expressions in the projections are colrefs, we can always pull it up
	// if it's not a colref, we can pull it up only if it does not appear in the operator enumerate expressions
	for (idx_t i = parents.size(); i > 0; i--) {
		idx_t parent_idx = i - 1;
		LogicalOperator &parent_op = parents[parent_idx].get();
		can_pull_through = true;

		LogicalOperatorVisitor::EnumerateExpressions(parent_op, [&](unique_ptr<Expression> *expr) {
			ExpressionIterator::EnumerateExpression(*expr, [&](unique_ptr<Expression> &child_expr) {
				if (child_expr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
					return;
				}

				auto &colref = child_expr->Cast<BoundColumnRefExpression>();
				auto entry = projection_map.find(colref.Binding());

				if (entry == projection_map.end()) {
					return;
				}

				// This parent references a projection output
				// If that output is NOT a simple column ref, we cannot pull through
				if (entry->second->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
					can_pull_through = false;
				}
			});
		});

		if (!can_pull_through) {
			break;
		}
	}
}

void ProjectionPullup::Optimize(unique_ptr<LogicalOperator> &op) {
	GetDataFlowContext();
	VisitOperator(op);
}

void ProjectionPullup::VisitOperator(unique_ptr<LogicalOperator> &op) {
	switch (op->type) {
	// These operators depend on column order.
	// If their immediate child is a projection, keep it and recurse into the projection’s child.
	// If no projection is present, insert one, then recurse into the newly added projection’s child.
	case LogicalOperatorType::LOGICAL_INTERSECT:
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_UNION: {
		for (auto &child : op->children) {
			InsertProjectionBelowOp(child);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
	case LogicalOperatorType::LOGICAL_CTE_REF:
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_PIVOT: {
		for (auto &child : op->children) {
			InsertProjectionBelowOp(child);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &comp_join = op->Cast<LogicalComparisonJoin>();
		if (comp_join.join_type == JoinType::MARK || comp_join.join_type == JoinType::SINGLE) {
			break; // bail
		}

		// We can pull through this operator, add it to the stack
		parents.push_back(*op);
		if (comp_join.join_type == JoinType::SEMI || comp_join.join_type == JoinType::ANTI) {
			// LHS: can pull through
			VisitOperator(comp_join.children[0]);

			// RHS: Cannot pull through. Add a projection "barrier"
			InsertProjectionBelowOp(comp_join.children[1]);
		} else {
			// All other joins: recurse normally on both sides
			VisitOperator(comp_join.children[0]);
			VisitOperator(comp_join.children[1]);
		}

		PopParents(*op);
		return;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		// We can pull through this operator, add it to the stack
		parents.push_back(*op);

		// Recurse
		VisitOperator(op->children[0]);

		PopParents(*op);
		return;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto &proj = op->Cast<LogicalProjection>();
		auto proj_bindings = proj.GetColumnBindings();

		bool all_column_refs = true;
		for (auto &expression : proj.expressions) {
			if (expression->GetExpressionType() != ExpressionType::BOUND_COLUMN_REF) {
				all_column_refs = false;
			}
			if (expression->IsVolatile()) {
				ProjectionPullup next(optimizer, root, GetDataFlowContext());
				next.Optimize(proj.children[0]);
				return; // bail
			}
		}

		bool can_pull_through = true;
		idx_t pull_up_to_here = parents.size();
		if (!all_column_refs) {
			column_binding_map_t<unique_ptr<Expression>> projection_map;
			for (idx_t i = 0; i < proj.expressions.size(); i++) {
				projection_map[proj_bindings[i]] = proj.expressions[i]->Copy();
			}
			CanPullThrough(projection_map, can_pull_through);
		}

		// Partial pullup is intentionally not implemented.
		// Pulling a projection only partially up could leave it in an intermediate state between operators. This would
		// reduce the opportunities for join reordering without providing any benefit.
		if (!can_pull_through) {
			// Recurse into child;
			ProjectionPullup next(optimizer, root, GetDataFlowContext());
			next.Optimize(proj.children[0]);
			return;
		}

		// after the loop we figured out how far we can pull it up
		// If we can pull up, replace bindings along parents and remove this projection
		if (pull_up_to_here > 0) {
			// Do not remove projections above UNNEST. The projection above the unnest extracts just the required
			// fields. Removing it forces all other operators to carry the full struct, eventually causing the
			// memory blowup.
			if (op->children[0]->type == LogicalOperatorType::LOGICAL_UNNEST) {
				parents.push_back(*op);
				Optimize(op->children[0]);
				PopParents(*op);
				return;
			}
			if (all_column_refs) {
				PullUpColrefProjection(op, proj, proj_bindings);
				return;
			}
			PullUpNonColrefProjection(op, proj, proj_bindings, pull_up_to_here);
			return;
		}

		// Recurse on child
		Optimize(op->children[0]);
		PopParents(*op);
		return;
	}
	default: {
		break;
	}
	}

	// Start a fresh traversal for each child.
	ProjectionPullup next(optimizer, root, GetDataFlowContext());
	for (auto &child : op->children) {
		next.VisitOperator(child);
	}
}
} // namespace duckdb
