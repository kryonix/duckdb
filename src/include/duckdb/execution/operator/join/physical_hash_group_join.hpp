//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_hash_group_join.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"

namespace duckdb {

class LogicalAggregate;

class PhysicalHashGroupJoin : public PhysicalJoin {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_GROUP_JOIN;

	PhysicalHashGroupJoin(PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &probe,
	                      PhysicalOperator &owner, vector<unique_ptr<Expression>> aggregates,
	                      vector<unique_ptr<Expression>> groups, idx_t estimated_cardinality);

	GroupedAggregateData grouped_aggregate_data;
	unsafe_vector<idx_t> aggregate_filter;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType ExecuteInternal(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                                   GlobalOperatorState &gstate, OperatorState &state) const override;

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override;
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
		return true;
	}
	bool ParallelSink() const override {
		return false;
	}
	bool ParallelOperator() const override {
		return false;
	}
	bool ParallelSource() const override {
		return false;
	}

	string GetName() const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
