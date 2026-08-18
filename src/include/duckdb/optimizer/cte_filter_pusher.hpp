//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/cte_filter_pusher.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_plan_data_flow.hpp"

namespace duckdb {

class LogicalOperator;
class Optimizer;

class CTEFilterPusher {
public:
	explicit CTEFilterPusher(Optimizer &optimizer);
	//! Finds all materialized CTEs and pushes OR filters into them (if applicable)
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	struct RewriteContext {
		explicit RewriteContext(LogicalOperator &root) : data_flow(root), mutator(data_flow) {
		}

		LogicalPlanDataFlow data_flow;
		LogicalPlanDataFlowMutator mutator;
	};

	struct IndexedMaterializedCTEInfo {
		vector<reference<LogicalOperator>> filters;
		bool all_cte_refs_are_filtered = true;
	};

private:
	//! Check whether the plan contains work for this optimizer
	static bool HasMaterializedCTE(LogicalOperator &op);
	//! Find a materialized CTE and its directly filtered refs through the data-flow index
	static IndexedMaterializedCTEInfo GetIndexedCandidate(LogicalOperator &materialized_cte, RewriteContext &context);
	//! Creates an OR filter and pushes it into a materialized CTE
	void PushFilterIntoCTE(LogicalOperator &materialized_cte, const vector<reference<LogicalOperator>> &filters,
	                       RewriteContext &context);

private:
	//! The optimizer
	Optimizer &optimizer;
};

} // namespace duckdb
