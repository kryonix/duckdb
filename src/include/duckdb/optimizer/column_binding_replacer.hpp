//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/column_binding_replacer.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class BoundColumnRefExpression;
class BoundSubqueryExpression;
class ColumnBindingReplacer;
class LogicalPlanDataFlow;
class LogicalPlanDataFlowMutator;

struct ReplacementBinding {
public:
	ReplacementBinding(ColumnBinding old_binding, ColumnBinding new_binding);
	ReplacementBinding(ColumnBinding old_binding, ColumnBinding new_binding, LogicalType new_type);

public:
	ColumnBinding old_binding;
	ColumnBinding new_binding;

	bool replace_type;
	LogicalType new_type;
};

//! A validated graph of direct binding replacement edges.
//! Resolution follows these edges transitively and is independent of insertion order.
class BindingReplacementGraph {
public:
	ColumnBinding Resolve(ColumnBinding binding) const;
	bool TryAdd(const ReplacementBinding &replacement);
	void Add(ColumnBinding old_binding, ColumnBinding new_binding);
	void Add(const ReplacementBinding &replacement);
	void Merge(const BindingReplacementGraph &replacements);
	void AddTo(ColumnBindingReplacer &replacer) const;

	bool Empty() const {
		return replacement_bindings.empty();
	}
	vector<ReplacementBinding>::const_iterator begin() const {
		return replacement_bindings.begin();
	}
	vector<ReplacementBinding>::const_iterator end() const {
		return replacement_bindings.end();
	}

private:
	ReplacementBinding ResolveReplacement(ColumnBinding binding) const;
	vector<ReplacementBinding> replacement_bindings;
};

//! The ColumnBindingReplacer updates column bindings (e.g., after changing the operator plan), utility for optimizers
class ColumnBindingReplacer : public LogicalOperatorVisitor {
public:
	ColumnBindingReplacer();

public:
	//! Update each operator of the plan
	void VisitOperator(LogicalOperator &op) override;
	//! Update bindings owned by this operator without visiting its children
	virtual void VisitOperatorBindings(LogicalOperator &op);
	//! Add binding replacements by position
	void AddReplacements(const vector<ColumnBinding> &old_bindings, const vector<ColumnBinding> &new_bindings);

protected:
	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override;

public:
	//! Do not recurse further than this operator (optional)
	optional_ptr<LogicalOperator> stop_operator;

	//! Contains all bindings that need to be updated
	vector<ReplacementBinding> replacement_bindings;
};

//! Like ColumnBindingReplacer, but also updates correlated-column metadata and nested subquery plans.
class CorrelatedColumnBindingReplacer : public ColumnBindingReplacer {
public:
	void VisitOperatorBindings(LogicalOperator &op) override;

protected:
	using ColumnBindingReplacer::VisitReplace;
	unique_ptr<Expression> VisitReplace(BoundSubqueryExpression &expr, unique_ptr<Expression> *expr_ptr) override;
};

//! Applies binding replacements together with their projection-layout invariants.
class ColumnBindingRewrite {
public:
	//! Preserve the projected binding identities after one child output is rewritten.
	static void RemapProjectionMap(LogicalOperator &op, idx_t child_index,
	                               const vector<ColumnBinding> &old_child_bindings,
	                               const BindingReplacementGraph &replacements);
	//! Preserve reachable identities while allowing a pruning rewrite to remove dead selections.
	static void RemapPrunedProjectionMap(LogicalOperator &op, idx_t child_index,
	                                     const vector<ColumnBinding> &old_child_bindings,
	                                     const BindingReplacementGraph &replacements);
	//! Apply the output-boundary view of a complete replacement graph to one parent-child edge.
	static void ApplyToChild(unique_ptr<LogicalOperator> &op, idx_t child_index,
	                         vector<ColumnBinding> old_child_bindings, const BindingReplacementGraph &replacements);
	static void ApplyToChild(LogicalOperator &op, idx_t child_index, vector<ColumnBinding> old_child_bindings,
	                         const BindingReplacementGraph &replacements);
	static void ApplyToOperatorBindings(LogicalOperator &op, const BindingReplacementGraph &replacements);
	//! Verify that every previous public output binding is still reachable at the rewritten output boundary.
	static void ValidateOutput(const vector<ColumnBinding> &old_output, const vector<ColumnBinding> &new_output,
	                           const BindingReplacementGraph &replacements);
	//! Replace one indexed subtree and preserve binding identities through every ownership ancestor.
	static unique_ptr<LogicalOperator> ReplaceSubtreeAndRewriteBindings(LogicalPlanDataFlow &data_flow,
	                                                                    LogicalPlanDataFlowMutator &mutator,
	                                                                    unique_ptr<LogicalOperator> &slot,
	                                                                    unique_ptr<LogicalOperator> replacement,
	                                                                    const BindingReplacementGraph &replacements);
	static unique_ptr<LogicalOperator> PromoteChildAndRewriteBindings(LogicalPlanDataFlow &data_flow,
	                                                                  LogicalPlanDataFlowMutator &mutator,
	                                                                  unique_ptr<LogicalOperator> &slot,
	                                                                  idx_t child_index,
	                                                                  const BindingReplacementGraph &replacements);
	static void InsertUnaryAndRewriteBindings(LogicalPlanDataFlow &data_flow, LogicalPlanDataFlowMutator &mutator,
	                                          unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> wrapper,
	                                          const BindingReplacementGraph &replacements);

private:
	static void RemapProjectionMapStrict(vector<ProjectionIndex> &projection_map,
	                                     const vector<ColumnBinding> &child_bindings_before,
	                                     const vector<ColumnBinding> &child_bindings_after);
};

} // namespace duckdb
