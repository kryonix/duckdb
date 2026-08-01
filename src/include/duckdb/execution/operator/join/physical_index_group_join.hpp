//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/operator/join/physical_index_group_join.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/execution/operator/aggregate/grouped_aggregate_data.hpp"
#include "duckdb/execution/operator/join/physical_hash_group_join.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/storage/storage_index.hpp"

namespace duckdb {

class DuckTableEntry;
class LogicalAggregate;

class PhysicalIndexGroupJoin : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::HASH_GROUP_JOIN;

	PhysicalIndexGroupJoin(PhysicalPlan &physical_plan, LogicalAggregate &op, PhysicalOperator &owner,
	                       DuckTableEntry &probe_table, Identifier index_name, vector<StorageIndex> probe_column_ids,
	                       vector<LogicalType> probe_scan_types, vector<idx_t> probe_projection_ids,
	                       vector<LogicalType> probe_input_types, unique_ptr<TableFilterSet> probe_filters,
	                       vector<unique_ptr<Expression>> probe_residual_filters,
	                       vector<unique_ptr<Expression>> probe_expressions, vector<idx_t> index_key_map,
	                       vector<LogicalType> index_key_types, vector<unique_ptr<Expression>> aggregates,
	                       vector<unique_ptr<Expression>> owner_payload_aggregates,
	                       vector<unique_ptr<Expression>> groups, vector<HashGroupJoinOutputColumn> output_groups,
	                       HashGroupJoinUnmatchedPolicy unmatched_policy, idx_t estimated_cardinality);

	GroupedAggregateData grouped_aggregate_data;
	GroupedAggregateData owner_payload_data;
	vector<HashGroupJoinOutputColumn> output_groups;
	vector<string> output_group_names;
	unsafe_vector<idx_t> non_distinct_filter;
	vector<HashGroupJoinDistinctAggregate> distinct_aggregates;
	HashGroupJoinUnmatchedPolicy unmatched_policy;
	reference<DuckTableEntry> probe_table;
	Identifier index_name;
	vector<StorageIndex> probe_column_ids;
	vector<LogicalType> probe_scan_types;
	vector<idx_t> probe_projection_ids;
	vector<LogicalType> probe_input_types;
	unique_ptr<TableFilterSet> probe_filters;
	vector<unique_ptr<Expression>> probe_residual_filters;
	vector<unique_ptr<Expression>> probe_expressions;
	vector<idx_t> index_key_map;
	vector<LogicalType> index_key_types;
	vector<unique_ptr<Expression>> unmatched_payload_expressions;

public:
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;

	bool ParallelOperator() const override {
		return true;
	}
	PipelineExternalInputSupport GetExternalInputSupport() const override {
		return PipelineExternalInputSupport::SUPPORTED;
	}

	string GetName() const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;
};

} // namespace duckdb
