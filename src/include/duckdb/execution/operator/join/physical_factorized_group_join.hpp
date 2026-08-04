//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_factorized_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/execution/group_join_strategy.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/joinside.hpp"

namespace duckdb {

enum class FactorizedGroupJoinBranchMode : uint8_t { LAZY, CACHED, EAGER };

struct FactorizedGroupJoinAggregateRange {
	idx_t begin = 0;
	idx_t count = 0;
	optional_idx multiplicity_index;
};

struct FactorizedGroupJoinDistinctAggregate {
	idx_t aggregate_index;
	idx_t source_idx;
	idx_t range_index;
	idx_t payload_index;
	vector<LogicalType> argument_types;
	optional_idx filter_index;
};

//! Aggregates two independent many-side inputs into a driver-keyed hash table.
class PhysicalFactorizedGroupJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_GROUP_JOIN;
	static constexpr idx_t SOURCE_COUNT = 3;

	PhysicalFactorizedGroupJoin(PhysicalPlan &physical_plan, LogicalOperator &op, PhysicalOperator &driver,
	                            PhysicalOperator &left_factor, PhysicalOperator &right_factor,
	                            vector<idx_t> driver_keys, vector<idx_t> left_keys, vector<idx_t> right_keys,
	                            vector<idx_t> output_group_key_indices, vector<unique_ptr<Expression>> aggregates,
	                            vector<FactorizedAggregateSource> aggregate_sources, bool preserve_left,
	                            bool preserve_right, bool semi_left, bool semi_right, bool unique_driver, bool routed,
	                            GroupJoinExecutionMode execution_mode, Value perfect_min, Value perfect_max,
	                            idx_t perfect_range, unique_ptr<JoinFilterPushdownInfo> left_filter_pushdown,
	                            unique_ptr<JoinFilterPushdownInfo> right_filter_pushdown,
	                            unique_ptr<JoinFilterPushdownInfo> left_driver_filter_pushdown,
	                            unique_ptr<JoinFilterPushdownInfo> right_driver_filter_pushdown,
	                            idx_t estimated_cardinality);

	void InitializeSinks();
	vector<AggregateObject> CreateHashTableAggregates() const;
	vector<AggregateObject> CreateSourceAggregates(idx_t source_idx) const;

	vector<LogicalType> key_types;
	vector<LogicalType> group_types;
	vector<vector<LogicalType>> source_input_types;
	vector<vector<unique_ptr<Expression>>> key_expressions;
	vector<unique_ptr<Expression>> group_expressions;
	vector<vector<unique_ptr<Expression>>> source_arguments;
	vector<vector<LogicalType>> source_argument_types;
	vector<unique_ptr<Expression>> aggregate_expressions;
	vector<AggregateObject> partial_aggregate_objects;
	vector<FactorizedAggregateSource> aggregate_sources;
	vector<FactorizedGroupJoinAggregateRange> source_ranges;
	vector<FactorizedGroupJoinDistinctAggregate> distinct_aggregates;
	vector<idx_t> partial_indexes;
	vector<idx_t> output_group_key_indices;
	TupleDataLayout target_layout;
	vector<string> aggregate_names;
	bool preserve_left;
	bool preserve_right;
	bool semi_left;
	bool semi_right;
	bool unique_driver;
	bool routed;
	GroupJoinExecutionMode planned_execution_mode;
	array<FactorizedGroupJoinBranchMode, SOURCE_COUNT - 1> branch_modes;
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range;
	array<unique_ptr<JoinFilterPushdownInfo>, SOURCE_COUNT - 1> factor_filter_pushdown;
	array<unique_ptr<JoinFilterPushdownInfo>, SOURCE_COUNT - 1> driver_filter_pushdown;
	idx_t estimated_driver_rows = 0;
	idx_t estimated_left_factor_rows = 0;
	idx_t estimated_right_factor_rows = 0;
	idx_t estimated_join_rows = 0;
	idx_t estimated_matched_drivers = 0;
	idx_t estimated_left_scan_rows = 0;
	idx_t estimated_right_scan_rows = 0;
	double estimated_build_cost = 0;
	double estimated_filter_cost = 0;
	double estimated_probe_cost = 0;
	double estimated_scan_cost = 0;
	double estimated_cache_cost = 0;
	double estimated_eager_work_cost = 0;
	double estimated_routing_cost = 0;
	double estimated_spill_cost = 0;
	double estimated_factorized_cost = 0;
	double estimated_best_existing_cost = 0;
	double estimated_driver_first_cost = 0;
	double estimated_factors_first_cost = 0;
	bool estimated_cost_reliable = false;
	bool auto_selected = false;
	bool driver_first = true;
	bool streaming_driver = false;
	PhysicalPlan &physical_plan;
	optional_ptr<PhysicalOperator> driver_input;
	optional_ptr<PhysicalOperator> left_input;
	optional_ptr<PhysicalOperator> right_input;
	optional_ptr<PhysicalOperator> driver_sink;
	optional_ptr<PhysicalOperator> left_sink;
	optional_ptr<PhysicalOperator> right_sink;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSource() const override {
		return !streaming_driver;
	}
	bool ParallelOperator() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}

	void BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) override;
	vector<const_reference<PhysicalOperator>> GetSources() const override;

	string GetName() const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
