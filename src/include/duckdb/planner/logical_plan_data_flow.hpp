//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/logical_plan_data_flow.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_ptr.hpp"
#include "duckdb/common/table_index.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/column_binding_map.hpp"

namespace duckdb {

class LogicalOperator;
class LogicalDependentJoin;
class LogicalPlanDataFlowState;
class LogicalPlanDataFlowMutator;
class LogicalPlanDataFlowMutationScope;

enum class LogicalPlanDataFlowStatus {
	SUCCESS,
	OPERATOR_NOT_INDEXED,
	BINDING_NOT_FOUND,
	BINDING_NOT_AVAILABLE,
	CORRELATED_REFERENCE,
	OPAQUE_BOUNDARY,
	CTE_BOUNDARY,
	DISCONNECTED,
	NOT_ANCESTOR,
	PATH_PROPERTY_NOT_FOUND,
	UNSUPPORTED
};

enum class LogicalPlanPathProperty {
	OPAQUE_BOUNDARY,
	CTE_BOUNDARY,
	PROJECTION_BOUNDARY,
	AGGREGATE_BOUNDARY,
	SET_OPERATION_BOUNDARY,
	WINDOW_BOUNDARY,
	UNNEST_BOUNDARY,
	LIMIT_BOUNDARY,
	SIDE_EFFECT_BOUNDARY,
	FILTER_PUSHDOWN_BOUNDARY,
	NULLABILITY_BOUNDARY,
	NULL_EXTENDING,
	BINDING_AVAILABILITY_BOUNDARY
};

struct LogicalPlanPathSummary {
	uint64_t properties = 0;

	void Add(LogicalPlanPathProperty property);
	void Merge(const LogicalPlanPathSummary &other);
	bool Has(LogicalPlanPathProperty property) const;

	bool operator==(const LogicalPlanPathSummary &other) const;
	bool operator!=(const LogicalPlanPathSummary &other) const;
};

struct LogicalPlanDataFlowOperatorResult {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	optional_ptr<LogicalOperator> op;
	//! The consumer's direct flow child containing op, when the result came from ResolveSource.
	idx_t source_child_index = DConstants::INVALID_INDEX;
};

struct LogicalPlanDataFlowParentResult {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	optional_ptr<LogicalOperator> parent;
	idx_t child_index = DConstants::INVALID_INDEX;
};

struct LogicalPlanDataFlowBooleanResult {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	bool value = false;
};

struct LogicalPlanDataFlowPathResult {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	LogicalPlanPathSummary summary;
};

struct LogicalPlanDataFlowReadersResult {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	vector<reference<LogicalOperator>> readers;
};

struct LogicalPlanBindingUse {
	ColumnBinding binding;
	idx_t depth;
	reference<LogicalOperator> consumer;
};

struct LogicalPlanCorrelatedUse {
	LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::UNSUPPORTED;
	ColumnBinding binding;
	idx_t depth;
	reference<LogicalOperator> consumer;
	optional_ptr<LogicalOperator> source;
	optional_ptr<LogicalDependentJoin> owning_join;
};

//! A pass-scoped, non-owning index of ownership and data flow in one logical plan.
class LogicalPlanDataFlow {
public:
	explicit LogicalPlanDataFlow(LogicalOperator &root);
	~LogicalPlanDataFlow();

	LogicalPlanDataFlow(const LogicalPlanDataFlow &) = delete;
	LogicalPlanDataFlow &operator=(const LogicalPlanDataFlow &) = delete;

public:
	LogicalPlanDataFlowOperatorResult ResolveSource(const ColumnBinding &binding, idx_t depth,
	                                                LogicalOperator &consumer) const;
	//! Resolve the source and direct ownership child that exposes a binding to a consumer.
	LogicalPlanDataFlowOperatorResult ResolveInputSource(const ColumnBinding &binding, idx_t depth,
	                                                     LogicalOperator &consumer) const;
	LogicalPlanDataFlowParentResult GetOwnershipParent(LogicalOperator &op) const;
	LogicalPlanDataFlowParentResult GetFlowParent(LogicalOperator &op) const;
	LogicalPlanDataFlowBooleanResult SameFlowTree(LogicalOperator &left, LogicalOperator &right) const;
	LogicalPlanDataFlowBooleanResult IsFlowAncestor(LogicalOperator &ancestor, LogicalOperator &descendant) const;
	LogicalPlanDataFlowOperatorResult LowestCommonAncestor(LogicalOperator &left, LogicalOperator &right) const;
	LogicalPlanDataFlowPathResult GetPathSummary(LogicalOperator &ancestor, LogicalOperator &descendant) const;
	LogicalPlanDataFlowOperatorResult FindFirstPathOperator(LogicalOperator &ancestor, LogicalOperator &descendant,
	                                                        const LogicalPlanPathSummary &properties) const;

	LogicalPlanDataFlowOperatorResult GetCTEProducer(TableIndex cte_index) const;
	LogicalPlanDataFlowReadersResult GetCTEReaders(TableIndex cte_index) const;
	vector<reference<LogicalOperator>> GetMaterializedCTEs() const;
	const vector<LogicalPlanBindingUse> &GetBindingUses() const;
	vector<LogicalPlanCorrelatedUse> GetCorrelatedUses() const;
	idx_t OperatorCount() const;

	//! Performs an expensive verification against the ownership tree and parent-walk semantics.
	bool Verify() const;

private:
	friend class LogicalPlanDataFlowMutator;
	friend class LogicalPlanDataFlowDetachedSubtree;
	friend class LogicalPlanDataFlowMutationScope;

	shared_ptr<LogicalPlanDataFlowState> state;
};

//! Owns an indexed subtree while it is detached from the main logical plan.
class LogicalPlanDataFlowDetachedSubtree {
public:
	~LogicalPlanDataFlowDetachedSubtree();

	LogicalPlanDataFlowDetachedSubtree(LogicalPlanDataFlowDetachedSubtree &&other) noexcept;
	LogicalPlanDataFlowDetachedSubtree &operator=(LogicalPlanDataFlowDetachedSubtree &&other) noexcept;

	LogicalPlanDataFlowDetachedSubtree(const LogicalPlanDataFlowDetachedSubtree &) = delete;
	LogicalPlanDataFlowDetachedSubtree &operator=(const LogicalPlanDataFlowDetachedSubtree &) = delete;

public:
	explicit operator bool() const;
	LogicalOperator &Get();

private:
	friend class LogicalPlanDataFlowMutator;

	LogicalPlanDataFlowDetachedSubtree(const shared_ptr<LogicalPlanDataFlowState> &state,
	                                   unique_ptr<LogicalOperator> subtree);
	void Reset();

private:
	weak_ptr<LogicalPlanDataFlowState> state;
	unique_ptr<LogicalOperator> subtree;
};

//! Defers full verification while coordinated indexed mutations leave metadata temporarily stale.
class LogicalPlanDataFlowMutationScope {
public:
	~LogicalPlanDataFlowMutationScope();

	LogicalPlanDataFlowMutationScope(LogicalPlanDataFlowMutationScope &&other) noexcept;
	LogicalPlanDataFlowMutationScope &operator=(LogicalPlanDataFlowMutationScope &&other) noexcept;

	LogicalPlanDataFlowMutationScope(const LogicalPlanDataFlowMutationScope &) = delete;
	LogicalPlanDataFlowMutationScope &operator=(const LogicalPlanDataFlowMutationScope &) = delete;

private:
	friend class LogicalPlanDataFlowMutator;

	explicit LogicalPlanDataFlowMutationScope(LogicalPlanDataFlowMutator &mutator);
	void Reset();

private:
	optional_ptr<LogicalPlanDataFlowMutator> mutator;
};

//! Maintains a LogicalPlanDataFlow while its logical plan is rewritten.
class LogicalPlanDataFlowMutator {
public:
	explicit LogicalPlanDataFlowMutator(LogicalPlanDataFlow &data_flow);

public:
	LogicalPlanDataFlowMutationScope BeginMutation();
	LogicalPlanDataFlowDetachedSubtree RegisterSubtree(unique_ptr<LogicalOperator> subtree);
	unique_ptr<LogicalOperator> UnregisterSubtree(LogicalPlanDataFlowDetachedSubtree subtree);
	LogicalPlanDataFlowDetachedSubtree DetachChild(LogicalOperator &parent, idx_t child_index);
	void AttachChild(LogicalOperator &parent, idx_t child_index, LogicalPlanDataFlowDetachedSubtree subtree);
	void AttachChild(LogicalOperator &parent, idx_t child_index, unique_ptr<LogicalOperator> subtree);
	unique_ptr<LogicalOperator> EraseChild(LogicalOperator &parent, idx_t child_index);
	unique_ptr<LogicalOperator> ReplaceSubtree(unique_ptr<LogicalOperator> &slot,
	                                           unique_ptr<LogicalOperator> replacement);
	unique_ptr<LogicalOperator> ReplaceOperator(unique_ptr<LogicalOperator> &slot,
	                                            unique_ptr<LogicalOperator> replacement);
	unique_ptr<LogicalOperator> PromoteChild(unique_ptr<LogicalOperator> &slot, idx_t child_index);
	void InsertUnary(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> wrapper);
	void InsertUnary(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> wrapper,
	                 LogicalOperator &changed_parent);
	void InsertParent(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> parent, idx_t child_index);
	void RotateParentWithChild(unique_ptr<LogicalOperator> &slot, idx_t child_index, idx_t grandchild_index);
	unique_ptr<LogicalOperator> RemoveUnary(unique_ptr<LogicalOperator> &slot);
	void SwapChildren(LogicalOperator &parent, idx_t left_index, idx_t right_index);
	void RefreshOperator(LogicalOperator &op);
	LogicalPlanDataFlowBooleanResult HasCorrelatedBinding(LogicalOperator &subtree,
	                                                      const column_binding_set_t &bindings) const;

private:
	friend class LogicalPlanDataFlowMutationScope;

	void EndMutation();
	void VerifyAfterMutation() const;

private:
	LogicalPlanDataFlow &data_flow;
};

} // namespace duckdb
