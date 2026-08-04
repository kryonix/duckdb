#include "duckdb/execution/physical_plan_generator.hpp"

#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/pipeline_broadcast_exchange.hpp"

namespace duckdb {

static double EstimatePhysicalRowWidth(const vector<LogicalType> &types) {
	if (types.empty()) {
		return 1;
	}
	TupleDataLayout tuple_layout;
	tuple_layout.Initialize(types, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	double row_width = static_cast<double>(tuple_layout.GetRowWidth());
	for (const auto &type : types) {
		TypeVisitor::VisitReplace(type, [&](const LogicalType &visited_type) {
			switch (visited_type.InternalType()) {
			case PhysicalType::VARCHAR:
				row_width += 8;
				break;
			case PhysicalType::LIST:
			case PhysicalType::ARRAY:
				row_width += 32;
				break;
			default:
				break;
			}
			row_width += 2;
			return visited_type;
		});
	}
	return row_width;
}

static double EstimatePhysicalPlanCost(PhysicalOperator &op, TableIndex target_cte_index,
                                       PhysicalPlanCostEstimate &estimate, reference_set_t<PhysicalOperator> &visited) {
	if (!visited.insert(op).second) {
		return 0;
	}
	double cost = 0;
	for (auto &child : op.children) {
		cost += EstimatePhysicalPlanCost(child.get(), target_cte_index, estimate, visited);
	}

	if (op.type != PhysicalOperatorType::CTE && op.type != PhysicalOperatorType::CTE_SCAN) {
		cost += static_cast<double>(op.estimated_cardinality) * EstimatePhysicalRowWidth(op.GetTypes());
	}
	if (op.type != PhysicalOperatorType::CTE) {
		return cost;
	}

	auto &cte = op.Cast<PhysicalCTE>();
	const auto output_bytes = static_cast<double>(op.children[0].get().estimated_cardinality) *
	                          EstimatePhysicalRowWidth(op.children[0].get().GetTypes());
	if (!cte.exchange) {
		if (cte.table_index == target_cte_index) {
			estimate.target_cte_found = true;
			estimate.materialized_cte_consumers = cte.cte_scans.size();
		}
		return cost + output_bytes * static_cast<double>(cte.cte_scans.size() + 1);
	}
	auto summary = cte.exchange->GetConsumerSummary();
	D_ASSERT(summary.unresolved == 0);
	if (cte.table_index == target_cte_index) {
		estimate.target_cte_found = true;
		estimate.direct_cte_consumers = summary.direct;
		estimate.buffered_cte_consumers = summary.buffered;
		estimate.materialized_cte_consumers = summary.materialized;
	}
	if (summary.materialized > 0) {
		cost += output_bytes * static_cast<double>(summary.materialized + 1);
	}
	// Unknown early-stop demand is deliberately priced at the full output.
	cost += output_bytes * static_cast<double>(summary.buffered * 2);
	return cost;
}

optional<PhysicalPlanCostEstimate> PhysicalPlanGenerator::EstimateCandidateCost(ClientContext &context,
                                                                                unique_ptr<LogicalOperator> logical,
                                                                                TableIndex target_cte_index) {
	try {
		PhysicalPlanGenerator physical_planner(context);
		PreparedStatementData statement_data(StatementType::SELECT_STATEMENT);
		statement_data.physical_plan = physical_planner.Plan(std::move(logical));
		statement_data.types = statement_data.physical_plan->Root().GetTypes();
		statement_data.names.resize(statement_data.types.size(), Identifier("candidate"));
		statement_data.output_type = QueryResultOutputType::FORCE_MATERIALIZED;
		statement_data.memory_type = QueryResultMemoryType::IN_MEMORY;
		auto collector = PhysicalResultCollector::GetResultCollector(context, statement_data);
		Executor topology_resolver(context);
		topology_resolver.ResolvePipelineTopology(*collector);
		reference_set_t<PhysicalOperator> visited;
		PhysicalPlanCostEstimate estimate;
		estimate.cost =
		    EstimatePhysicalPlanCost(statement_data.physical_plan->Root(), target_cte_index, estimate, visited);
		return estimate;
	} catch (NotImplementedException &) {
		return optional<PhysicalPlanCostEstimate>();
	} catch (InternalException &) {
		return optional<PhysicalPlanCostEstimate>();
	}
}

} // namespace duckdb
