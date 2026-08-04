#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"

#include "duckdb/common/enums/join_type.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/optimizer/join_order/cardinality_estimator.hpp"
#include "duckdb/optimizer/join_order/cost_model.hpp"
#include "duckdb/optimizer/join_order/plan_enumerator.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {

JoinOrderOptimizer::JoinOrderOptimizer(ClientContext &context)
    : context(context), query_graph_manager(context), depth(1) {
}

JoinOrderOptimizer JoinOrderOptimizer::CreateChildOptimizer() {
	JoinOrderOptimizer child_optimizer(context);
	child_optimizer.materialized_cte_stats = materialized_cte_stats;
	child_optimizer.delim_scan_stats = delim_scan_stats;
	child_optimizer.depth = depth + 1;
	child_optimizer.recursive_cte_indexes = recursive_cte_indexes;
	return child_optimizer;
}

void JoinOrderOptimizer::SetGroupJoinContext(HashGroupJoinOrderContext context) {
	group_join_context = std::move(context);
}

unique_ptr<LogicalOperator> JoinOrderOptimizer::Optimize(unique_ptr<LogicalOperator> plan,
                                                         optional_ptr<RelationStats> stats) {
	auto max_expression_depth = Settings::Get<MaxExpressionDepthSetting>(query_graph_manager.context);
	if (depth > max_expression_depth) {
		// Very deep plans will eventually consume quite some stack space
		// Returning the current plan is always a valid choice
		return plan;
	}

	// make sure query graph manager has not extracted a relation graph already
	LogicalOperator *op = plan.get();

	// extract the relations that go into the hyper graph.
	// We optimize the children of any non-reorderable operations we come across.
	bool reorderable = query_graph_manager.Build(*this, *op);

	// get relation_stats here since the reconstruction process will move all relations.
	auto relation_stats = query_graph_manager.relation_manager.GetRelationStats();
	unique_ptr<LogicalOperator> new_logical_plan = nullptr;

	if (reorderable) {
		// query graph now has filters and relations
		auto cardinality_estimator =
		    CardinalityEstimator(query_graph_manager.set_manager, query_graph_manager.GetPredicateModel());
		optional<GroupJoinOrderCostContext> group_join_cost_context;
		if (group_join_context) {
			GroupJoinOrderCostContext mapped {{},
			                                  {},
			                                  {},
			                                  {},
			                                  {},
			                                  group_join_context->key_width,
			                                  group_join_context->state_width,
			                                  group_join_context->routed,
			                                  group_join_context->direct_inner,
			                                  group_join_context->fixed_size_keys,
			                                  group_join_context->physical_eager_supported,
			                                  group_join_context->perfect_supported,
			                                  group_join_context->perfect_range,
			                                  group_join_context->mixed_distinct_memoizing,
			                                  group_join_context->factorized_cost,
			                                  group_join_context->strategy,
			                                  group_join_context->factorized};
			bool valid = true;
			auto map_relations = [&](const unordered_set<TableIndex> &tables, unordered_set<RelationIndex> &relations) {
				for (auto table : tables) {
					auto entry = query_graph_manager.relation_manager.relation_mapping.find(table);
					if (entry == query_graph_manager.relation_manager.relation_mapping.end()) {
						valid = false;
						return;
					}
					relations.insert(entry->second);
				}
			};
			map_relations(group_join_context->owner_tables, mapped.owner_relations);
			map_relations(group_join_context->probe_tables, mapped.probe_relations);
			map_relations(group_join_context->factorized_driver_tables, mapped.factorized_driver_relations);
			map_relations(group_join_context->factorized_left_tables, mapped.factorized_left_relations);
			map_relations(group_join_context->factorized_right_tables, mapped.factorized_right_relations);
			for (auto relation : mapped.owner_relations) {
				if (mapped.probe_relations.find(relation) != mapped.probe_relations.end()) {
					valid = false;
					break;
				}
			}
			if (mapped.factorized) {
				for (auto relation : mapped.factorized_driver_relations) {
					if (mapped.factorized_left_relations.find(relation) != mapped.factorized_left_relations.end() ||
					    mapped.factorized_right_relations.find(relation) != mapped.factorized_right_relations.end()) {
						valid = false;
						break;
					}
				}
				for (auto relation : mapped.factorized_left_relations) {
					if (mapped.factorized_right_relations.find(relation) != mapped.factorized_right_relations.end()) {
						valid = false;
						break;
					}
				}
			}
			const auto valid_factorized = mapped.factorized && !mapped.factorized_driver_relations.empty() &&
			                              !mapped.factorized_left_relations.empty() &&
			                              !mapped.factorized_right_relations.empty();
			const auto valid_binary =
			    !mapped.factorized && !mapped.owner_relations.empty() && !mapped.probe_relations.empty();
			if (valid && (valid_factorized || valid_binary)) {
				group_join_cost_context = std::move(mapped);
			}
		}
		auto cost_model = CostModel(query_graph_manager, cardinality_estimator,
		                            group_join_cost_context ? &*group_join_cost_context : nullptr);

		// Initialize a plan enumerator.
		auto plan_enumerator =
		    PlanEnumerator(query_graph_manager, cost_model, query_graph_manager.GetQueryGraphEdges());

		// Initialize the leaf/single node plans
		plan_enumerator.InitLeafPlans();
		if (plan_enumerator.SolveJoinOrder()) {
			// now reconstruct a logical plan from the query graph plan
			query_graph_manager.plans = plan_enumerator.GetPlans();
			new_logical_plan = query_graph_manager.Reconstruct(std::move(plan));
		} else {
			// Approximate enumeration can conservatively reject every remaining partition. The original tree is still
			// intact and is always a valid fallback.
			new_logical_plan = std::move(plan);
		}
	} else {
		new_logical_plan = std::move(plan);
		if (relation_stats.size() == 1) {
			new_logical_plan->estimated_cardinality = relation_stats.at(0).cardinality;
			new_logical_plan->has_estimated_cardinality = true;
		}
	}

	// Propagate up a stats object from the top of the new_logical_plan if stats exist.
	if (stats) {
		auto cardinality = new_logical_plan->EstimateCardinality(context);
		auto bindings = new_logical_plan->GetColumnBindings();
		auto new_stats = RelationStatisticsHelper::CombineStatsOfReorderableOperator(bindings, relation_stats);
		new_stats.cardinality = cardinality;
		RelationStatisticsHelper::CopyRelationStats(*stats, new_stats);
	} else {
		// starts recursively setting cardinality
		new_logical_plan->EstimateCardinality(context);
	}

	if (new_logical_plan->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
		new_logical_plan->SetEstimatedCardinality(3);
	}

	return new_logical_plan;
}

void JoinOrderOptimizer::AddMaterializedCTEStats(TableIndex index, RelationStats &&stats) {
	materialized_cte_stats.emplace(index, std::move(stats));
}

RelationStats JoinOrderOptimizer::GetMaterializedCTEStats(TableIndex index) {
	auto it = materialized_cte_stats.find(index);
	if (it == materialized_cte_stats.end()) {
		throw InternalException("Unable to find materialized CTE stats with index %llu", index.index);
	}
	return it->second;
}

void JoinOrderOptimizer::AddDelimScanStats(RelationStats &stats) {
	delim_scan_stats = &stats;
}

RelationStats JoinOrderOptimizer::GetDelimScanStats() {
	if (!delim_scan_stats) {
		throw InternalException("Unable to find delim scan stats!");
	}
	return *delim_scan_stats;
}

} // namespace duckdb
