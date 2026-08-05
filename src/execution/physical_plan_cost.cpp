#include "duckdb/execution/physical_plan_generator.hpp"

#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/executor.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/set/physical_cte.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parallel/pipeline_broadcast_exchange.hpp"
#include "duckdb/storage/buffer_manager.hpp"

#include <cmath>

namespace duckdb {

static constexpr double MATERIALIZED_EXCHANGE_COST = 2;

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

static double EstimateOutputBytes(const PhysicalOperator &op) {
	return static_cast<double>(op.estimated_cardinality) * EstimatePhysicalRowWidth(op.GetTypes());
}

static double EstimateInputBytes(const PhysicalOperator &op) {
	double input_bytes = 0;
	for (auto &child : op.children) {
		input_bytes += EstimateOutputBytes(child.get());
	}
	return input_bytes;
}

static double EstimateOrderingPasses(idx_t count) {
	return MaxValue<double>(1, std::ceil(std::log2(static_cast<double>(MaxValue<idx_t>(count, 2)))));
}

static double EstimateOperatorWork(ClientContext &context, const PhysicalOperator &op, double &state_bytes) {
	const auto output_bytes = EstimateOutputBytes(op);
	const auto input_bytes = EstimateInputBytes(op);
	switch (op.type) {
	case PhysicalOperatorType::TABLE_SCAN: {
		auto &scan = op.Cast<PhysicalTableScan>();
		idx_t scanned_cardinality = op.estimated_cardinality;
		if (scan.function.cardinality) {
			auto statistics = scan.function.cardinality(context, scan.bind_data.get());
			if (statistics && statistics->has_estimated_cardinality) {
				scanned_cardinality = MaxValue(scanned_cardinality, statistics->estimated_cardinality);
			}
		}
		return static_cast<double>(scanned_cardinality) * EstimatePhysicalRowWidth(op.GetTypes());
	}
	case PhysicalOperatorType::FILTER:
	case PhysicalOperatorType::PROJECTION:
		return input_bytes + output_bytes;
	case PhysicalOperatorType::UNGROUPED_AGGREGATE:
	case PhysicalOperatorType::HASH_GROUP_BY:
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY:
	case PhysicalOperatorType::PARTITIONED_AGGREGATE:
	case PhysicalOperatorType::LIMITED_DISTINCT:
		state_bytes = output_bytes * 2;
		return input_bytes * 2 + output_bytes;
	case PhysicalOperatorType::ORDER_BY:
	case PhysicalOperatorType::WINDOW:
		state_bytes = input_bytes;
		return input_bytes * (1 + EstimateOrderingPasses(op.estimated_cardinality)) + output_bytes;
	case PhysicalOperatorType::TOP_N:
		state_bytes = output_bytes;
		return input_bytes * (1 + EstimateOrderingPasses(op.estimated_cardinality)) + output_bytes;
	case PhysicalOperatorType::HASH_JOIN: {
		if (op.children.size() != 2) {
			return input_bytes + output_bytes;
		}
		const auto probe_bytes = EstimateOutputBytes(op.children[0].get());
		const auto build_bytes = EstimateOutputBytes(op.children[1].get());
		state_bytes = build_bytes * 2;
		return probe_bytes + build_bytes * 2 + output_bytes;
	}
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
	case PhysicalOperatorType::IE_JOIN:
	case PhysicalOperatorType::ASOF_JOIN:
		state_bytes = input_bytes;
		return input_bytes * (1 + EstimateOrderingPasses(op.estimated_cardinality)) + output_bytes;
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN:
	case PhysicalOperatorType::NESTED_LOOP_JOIN:
	case PhysicalOperatorType::CROSS_PRODUCT:
		if (op.children.size() == 2) {
			return static_cast<double>(op.children[0].get().estimated_cardinality) *
			           static_cast<double>(op.children[1].get().estimated_cardinality) +
			       output_bytes;
		}
		return input_bytes + output_bytes;
	default:
		return MaxValue(input_bytes, output_bytes);
	}
}

static idx_t EstimateSourceCardinality(ClientContext &context, const PhysicalOperator &op) {
	if (op.type == PhysicalOperatorType::TABLE_SCAN) {
		auto &scan = op.Cast<PhysicalTableScan>();
		if (scan.function.cardinality) {
			auto statistics = scan.function.cardinality(context, scan.bind_data.get());
			if (statistics && statistics->has_estimated_cardinality) {
				return statistics->estimated_cardinality;
			}
		}
	}
	return op.estimated_cardinality;
}

static double EstimatePhysicalPlanCost(ClientContext &context, PhysicalOperator &op, TableIndex target_cte_index,
                                       PhysicalPlanCostEstimate &estimate, reference_set_t<PhysicalOperator> &visited,
                                       double operator_memory_limit) {
	if (!visited.insert(op).second) {
		return 0;
	}
	estimate.maximum_cardinality = MaxValue(estimate.maximum_cardinality, op.estimated_cardinality);
	if (op.children.empty() && op.type != PhysicalOperatorType::CTE_SCAN) {
		estimate.source_cardinality += EstimateSourceCardinality(context, op);
	}
	double cost = 0;
	double first_child_cost = 0;
	for (idx_t child_idx = 0; child_idx < op.children.size(); child_idx++) {
		const auto child_cost = EstimatePhysicalPlanCost(context, op.children[child_idx].get(), target_cte_index,
		                                                 estimate, visited, operator_memory_limit);
		cost += child_cost;
		if (child_idx == 0) {
			first_child_cost = child_cost;
		}
	}

	if (op.type != PhysicalOperatorType::CTE && op.type != PhysicalOperatorType::CTE_SCAN) {
		double state_bytes = 0;
		const auto operator_work = EstimateOperatorWork(context, op, state_bytes);
		estimate.operator_work += operator_work;
		estimate.peak_memory = MaxValue(estimate.peak_memory, state_bytes);
		if (state_bytes > operator_memory_limit) {
			const auto spill_work = (state_bytes - operator_memory_limit) * 4;
			estimate.spill_work += spill_work;
			cost += spill_work;
		}
		cost += operator_work;
	}
	if (op.type != PhysicalOperatorType::CTE) {
		return cost;
	}

	auto &cte = op.Cast<PhysicalCTE>();
	const auto output_bytes = EstimateOutputBytes(op.children[0].get());
	if (cte.table_index == target_cte_index) {
		estimate.target_cte_producer_work = first_child_cost;
	}
	if (!cte.exchange) {
		if (cte.table_index == target_cte_index) {
			estimate.target_cte_found = true;
			estimate.materialized_cte_consumers = cte.cte_scans.size();
		}
		const auto exchange_work =
		    output_bytes * static_cast<double>(cte.cte_scans.size() + 1) * MATERIALIZED_EXCHANGE_COST;
		estimate.exchange_work += exchange_work;
		if (cte.table_index == target_cte_index) {
			estimate.target_cte_exchange_work += exchange_work;
		}
		return cost + exchange_work;
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
		const auto exchange_work =
		    output_bytes * static_cast<double>(summary.materialized + 1) * MATERIALIZED_EXCHANGE_COST;
		estimate.exchange_work += exchange_work;
		if (cte.table_index == target_cte_index) {
			estimate.target_cte_exchange_work += exchange_work;
		}
		cost += exchange_work;
	}
	// Unknown early-stop demand is deliberately priced at the full output.
	const auto buffered_work = output_bytes * static_cast<double>(summary.buffered * 2) * MATERIALIZED_EXCHANGE_COST;
	const auto direct_chunks = std::ceil(static_cast<double>(op.children[0].get().estimated_cardinality) /
	                                     static_cast<double>(STANDARD_VECTOR_SIZE));
	const auto direct_work =
	    direct_chunks * static_cast<double>(summary.direct) * EstimatePhysicalRowWidth(op.children[0].get().GetTypes());
	estimate.exchange_work += buffered_work + direct_work;
	if (cte.table_index == target_cte_index) {
		estimate.target_cte_exchange_work += buffered_work + direct_work;
	}
	cost += buffered_work + direct_work;
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
		const auto operator_memory_limit =
		    static_cast<double>(BufferManager::GetBufferManager(context).GetOperatorMemoryLimit());
		estimate.cost = EstimatePhysicalPlanCost(context, statement_data.physical_plan->Root(), target_cte_index,
		                                         estimate, visited, operator_memory_limit);
		if (estimate.source_cardinality > 0 && static_cast<double>(estimate.maximum_cardinality) >
		                                           static_cast<double>(estimate.source_cardinality) * 1000) {
			estimate.reliable = false;
		}
		return estimate;
	} catch (NotImplementedException &) {
		return optional<PhysicalPlanCostEstimate>();
	} catch (InternalException &) {
		return optional<PhysicalPlanCostEstimate>();
	}
}

} // namespace duckdb
