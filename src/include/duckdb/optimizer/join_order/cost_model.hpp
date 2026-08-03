//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/cost_model.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/optimizer/join_order/join_node.hpp"
#include "duckdb/optimizer/join_order/cardinality_estimator.hpp"
#include "duckdb/execution/group_join_strategy.hpp"

namespace duckdb {

class QueryGraphManager;

struct GroupJoinOrderCostContext {
	unordered_set<RelationIndex> owner_relations;
	unordered_set<RelationIndex> probe_relations;
	unordered_set<RelationIndex> factorized_driver_relations;
	unordered_set<RelationIndex> factorized_left_relations;
	unordered_set<RelationIndex> factorized_right_relations;
	idx_t key_width;
	idx_t state_width;
	bool routed;
	bool direct_inner;
	bool fixed_size_keys;
	bool physical_eager_supported;
	bool perfect_supported;
	idx_t perfect_range;
	GroupJoinStrategy strategy;
	bool factorized;
};

class CostModel {
public:
	explicit CostModel(QueryGraphManager &query_graph_manager, CardinalityEstimator &cardinality_estimator,
	                   optional_ptr<GroupJoinOrderCostContext> group_join_context = nullptr);

public:
	//! Compute cost of a join relation set
	double ComputeCost(DPJoinNode &left, DPJoinNode &right, JoinRelationSet &combination,
	                   const vector<reference<NeighborInfo>> &possible_connections);
	CardinalityEstimator &GetCardinalityEstimator();

private:
	//! query graph storing relation manager information
	QueryGraphManager &query_graph_manager;
	CardinalityEstimator &cardinality_estimator;
	optional_ptr<GroupJoinOrderCostContext> group_join_context;
};

} // namespace duckdb
