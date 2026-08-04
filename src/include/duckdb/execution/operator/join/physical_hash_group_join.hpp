//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_hash_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/group_join_strategy.hpp"
#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"

namespace duckdb {

class LogicalAggregate;
class LogicalComparisonJoin;

struct HashGroupJoinDistinctAggregate {
	idx_t aggregate_index;
	idx_t payload_index;
	vector<LogicalType> argument_types;
	bool has_filter;
};

class PhysicalHashGroupJoin : public PhysicalJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_GROUP_JOIN;

	PhysicalHashGroupJoin(PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &probe,
	                      PhysicalOperator &owner, vector<unique_ptr<Expression>> aggregates,
	                      vector<unique_ptr<Expression>> owner_payload_aggregates,
	                      vector<unique_ptr<Expression>> groups, vector<HashGroupJoinOutputColumn> output_groups,
	                      HashGroupJoinUnmatchedPolicy unmatched_policy, bool routed, bool unique_owner,
	                      bool single_match, unique_ptr<JoinFilterPushdownInfo> filter_pushdown,
	                      GroupJoinImplementation implementation, GroupJoinExecutionMode execution_mode,
	                      Value perfect_min, Value perfect_max, idx_t perfect_range,
	                      vector<LogicalType> eager_payload_types, vector<LogicalType> unmatched_probe_types,
	                      vector<unique_ptr<Expression>> unmatched_payload_expressions, bool streaming_eager_raw,
	                      idx_t estimated_cardinality);
	PhysicalHashGroupJoin(PhysicalPlan &physical_plan, LogicalComparisonJoin &op, PhysicalOperator &probe,
	                      PhysicalOperator &owner, vector<unique_ptr<Expression>> aggregates,
	                      vector<unique_ptr<Expression>> owner_payload_aggregates,
	                      vector<unique_ptr<Expression>> groups, vector<HashGroupJoinOutputColumn> output_columns,
	                      idx_t estimated_cardinality);

	GroupedAggregateData grouped_aggregate_data;
	GroupedAggregateData owner_payload_data;
	vector<HashGroupJoinOutputColumn> output_groups;
	vector<HashGroupJoinOutputColumn> output_columns;
	vector<string> output_group_names;
	unsafe_vector<idx_t> non_distinct_filter;
	vector<HashGroupJoinDistinctAggregate> distinct_aggregates;
	HashGroupJoinUnmatchedPolicy unmatched_policy;
	vector<LogicalType> unmatched_probe_types;
	vector<unique_ptr<Expression>> unmatched_payload_expressions;
	vector<LogicalType> eager_payload_types;
	bool routed;
	bool unique_owner;
	bool single_match;
	bool null_equal;
	bool static_mode;
	bool streaming_eager;
	bool streaming_eager_raw;
	bool parallel_owner_build;
	GroupJoinImplementation implementation;
	GroupJoinExecutionMode planned_execution_mode;
	Value perfect_min;
	Value perfect_max;
	idx_t perfect_range;
	vector<LogicalType> output_group_types;
	unique_ptr<JoinFilterPushdownInfo> filter_pushdown;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	unique_ptr<GlobalOperatorState> GetGlobalOperatorState(ClientContext &context) const override;
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override;
	OperatorFinalResultType OperatorFinalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                                         OperatorFinalizeInput &input) const override;

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
	SinkCombineResultType Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const override;
	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                          OperatorSinkFinalizeInput &input) const override;

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override;
	unique_ptr<LocalSourceState> GetLocalSourceState(ExecutionContext &context,
	                                                 GlobalSourceState &gstate) const override;
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override;

	bool IsSink() const override {
		return true;
	}
	bool IsSource() const override {
		return !streaming_eager;
	}
	bool ParallelSink() const override {
		return parallel_owner_build;
	}
	bool ParallelOperator() const override {
		return true;
	}
	bool ParallelSource() const override {
		return true;
	}
	bool RequiresOperatorFinalize() const override {
		return !streaming_eager;
	}

	string GetName() const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
