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

namespace duckdb {

class LogicalOperator;
class LogicalPlanDataFlowState;

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
	SIDE_EFFECT_BOUNDARY
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
	LogicalPlanDataFlowParentResult GetOwnershipParent(LogicalOperator &op) const;
	LogicalPlanDataFlowParentResult GetFlowParent(LogicalOperator &op) const;
	LogicalPlanDataFlowBooleanResult SameFlowTree(LogicalOperator &left, LogicalOperator &right) const;
	LogicalPlanDataFlowBooleanResult IsFlowAncestor(LogicalOperator &ancestor, LogicalOperator &descendant) const;
	LogicalPlanDataFlowOperatorResult LowestCommonAncestor(LogicalOperator &left, LogicalOperator &right) const;
	LogicalPlanDataFlowPathResult GetPathSummary(LogicalOperator &ancestor, LogicalOperator &descendant) const;

	LogicalPlanDataFlowOperatorResult GetCTEProducer(TableIndex cte_index) const;
	LogicalPlanDataFlowReadersResult GetCTEReaders(TableIndex cte_index) const;
	const vector<LogicalPlanBindingUse> &GetBindingUses() const;
	idx_t OperatorCount() const;

	//! Performs an expensive verification against the ownership tree and parent-walk semantics.
	bool Verify() const;

private:
	unique_ptr<LogicalPlanDataFlowState> state;
};

} // namespace duckdb
