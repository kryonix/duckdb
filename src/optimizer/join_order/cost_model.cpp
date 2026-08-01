#include "duckdb/optimizer/join_order/join_node.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/optimizer/join_order/cost_model.hpp"

#include "duckdb/optimizer/hash_group_join.hpp"
#include "duckdb/optimizer/join_order/query_graph_manager.hpp"

namespace duckdb {

CostModel::CostModel(QueryGraphManager &query_graph_manager, CardinalityEstimator &cardinality_estimator,
                     optional_ptr<GroupJoinOrderCostContext> group_join_context_p)
    : query_graph_manager(query_graph_manager), cardinality_estimator(cardinality_estimator),
      group_join_context(group_join_context_p) {
}

static bool ContainsRelations(const JoinRelationSet &set, const unordered_set<RelationIndex> &relations) {
	for (auto relation : relations) {
		bool found = false;
		for (idx_t relation_idx = 0; relation_idx < set.count; relation_idx++) {
			if (set.relations[relation_idx] == relation) {
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}

CardinalityEstimator &CostModel::GetCardinalityEstimator() {
	return cardinality_estimator;
}

static double GetLeftJoinInputCost(CardinalityEstimator &cardinality_estimator,
                                   const vector<reference<NeighborInfo>> &possible_connections) {
	double cost = 0;
	reference_set_t<JoinRelationSet> seen_right_sides;
	for (auto &connection : possible_connections) {
		for (auto predicate_ref : connection.get().predicates) {
			auto &predicate = predicate_ref.get();
			if (predicate.GetJoinType() != JoinType::LEFT) {
				continue;
			}
			D_ASSERT(predicate.GetRightSetOptional());
			if (!seen_right_sides.insert(predicate.GetRightSet()).second) {
				continue;
			}
			cost += cardinality_estimator.EstimateCardinalityWithSet<double>(predicate.GetRightSet());
		}
	}
	return cost;
}

// Currently cost of a join mostly factors in the cardinalities.
// LEFT joins need an explicit RHS input component because their output cardinality preserves the LHS,
// which otherwise makes early LEFT joins over large RHS inputs look almost free.
double CostModel::ComputeCost(DPJoinNode &left, DPJoinNode &right, JoinRelationSet &combination,
                              const vector<reference<NeighborInfo>> &possible_connections) {
	auto join_card = cardinality_estimator.EstimateCardinalityWithSet<double>(combination);
	auto join_cost = join_card;
	if (group_join_context && combination.count == query_graph_manager.relation_manager.NumRelations()) {
		auto left_is_owner = ContainsRelations(left.set, group_join_context->owner_relations) &&
		                     ContainsRelations(right.set, group_join_context->probe_relations);
		auto right_is_owner = ContainsRelations(right.set, group_join_context->owner_relations) &&
		                      ContainsRelations(left.set, group_join_context->probe_relations);
		auto &owner = right_is_owner ? right : left;
		auto &probe = right_is_owner ? left : right;
		auto owner_rows = cardinality_estimator.EstimateCardinalityWithSet<idx_t>(owner.set);
		auto probe_rows = cardinality_estimator.EstimateCardinalityWithSet<idx_t>(probe.set);
		auto match_rows = cardinality_estimator.EstimateCardinalityWithSet<idx_t>(combination);
		auto costs = EstimateHashGroupJoinAlternatives(
		    owner_rows, probe_rows, match_rows, MinValue(owner_rows, match_rows), group_join_context->key_width,
		    group_join_context->state_width, group_join_context->routed, group_join_context->direct_inner,
		    group_join_context->fixed_size_keys, group_join_context->physical_eager_supported,
		    group_join_context->perfect_supported, group_join_context->perfect_range, query_graph_manager.context);
		if (left_is_owner || right_is_owner) {
			if (group_join_context->strategy == GroupJoinStrategy::FORCE ||
			    group_join_context->strategy == GroupJoinStrategy::HASH ||
			    group_join_context->strategy == GroupJoinStrategy::PERFECT ||
			    group_join_context->strategy == GroupJoinStrategy::EAGER ||
			    group_join_context->strategy == GroupJoinStrategy::INDEX) {
				join_cost = -left.cost - right.cost;
			} else if (costs.perfect_selected && costs.separate_cost != 0) {
				join_cost = join_card * costs.perfect_cost / costs.separate_cost;
			} else if (costs.physical_eager_selected && costs.separate_cost != 0) {
				join_cost = join_card * costs.physical_eager_cost / costs.separate_cost;
			} else if (costs.hash_selected && costs.separate_cost != 0) {
				join_cost = join_card * costs.memoizing_cost / costs.separate_cost;
			}
		}
	}
	if (query_graph_manager.GetPredicateModel().HasLeftJoinPredicates()) {
		join_cost += GetLeftJoinInputCost(cardinality_estimator, possible_connections);
	}
	return join_cost + left.cost + right.cost;
}

} // namespace duckdb
