#pragma once

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/planner/column_binding_map.hpp"

namespace duckdb {

class Optimizer;
class LogicalOperator;
class ProjectionPullupDataFlowContext;

class ProjectionPullup : public LogicalOperatorVisitor {
public:
	explicit ProjectionPullup(Optimizer &optimizer, unique_ptr<LogicalOperator> &root);
	~ProjectionPullup();

	void Optimize(unique_ptr<LogicalOperator> &op);
	void VisitOperator(unique_ptr<LogicalOperator> &op) override;

private:
	ProjectionPullup(Optimizer &optimizer, unique_ptr<LogicalOperator> &root, ProjectionPullupDataFlowContext &context);
	ProjectionPullupDataFlowContext &GetDataFlowContext();

	unique_ptr<ProjectionPullupDataFlowContext> owned_context;
	optional_ptr<ProjectionPullupDataFlowContext> context;
	Optimizer &optimizer;
	unique_ptr<LogicalOperator> &root;
	vector<reference<LogicalOperator>> parents;
	void PopParents(const LogicalOperator &op);
	void InsertProjectionBelowOp(unique_ptr<LogicalOperator> &child);
	void CanPullThrough(column_binding_map_t<unique_ptr<Expression>> &projection_map, bool &can_pull_through);
	void PullUpColrefProjection(unique_ptr<LogicalOperator> &op, LogicalProjection &proj,
	                            vector<ColumnBinding> &proj_bindings);
	void PullUpNonColrefProjection(unique_ptr<LogicalOperator> &op, LogicalProjection &proj,
	                               vector<ColumnBinding> &proj_bindings, idx_t pull_up_to_here);
};

} // namespace duckdb
