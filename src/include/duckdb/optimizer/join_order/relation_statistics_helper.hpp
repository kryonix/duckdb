//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/join_order/statistics_extractor.hpp
//
//
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class CardinalityEstimator;

enum class DistinctCountSource : uint8_t { CARDINALITY, MIN_MAX, HLL, EXACT };

struct DistinctCount {
	DistinctCount(idx_t distinct_count, DistinctCountSource source);

	idx_t distinct_count;
	DistinctCountSource source;
};

struct ExpressionBinding {
public:
	bool FoundExpression() const;
	bool FoundColumnRef() const;

public:
	optional_ptr<Expression> expression;
	ColumnBinding child_binding;
	bool expression_is_constant = false;
};

enum class NumericDistributionSource : uint8_t { BASE_COLUMN, SUM, COUNT, AVERAGE, DERIVED };

//! Approximate distribution of a numeric relation column used only for cardinality estimation.
struct NumericDistributionStats {
	NumericDistributionStats(idx_t count, double mean, double variance, NumericDistributionSource source);

	idx_t count;
	double mean;
	double variance;
	NumericDistributionSource source;
	bool is_aggregate = false;
	//! Identifies SUM/COUNT results derived from the same aggregate input.
	hash_t lineage = 0;
	double input_mean = 0;
	double input_variance = 0;
	double mean_group_size = 0;
};

struct RelationStats {
public:
	RelationStats();

public:
	//! column_id -> estimated distinct count for column
	vector<DistinctCount> column_distinct_count;
	//! column_id -> estimated numeric distribution, or nullptr if unavailable
	vector<shared_ptr<NumericDistributionStats>> column_numeric_distribution;
	idx_t cardinality;
	double filter_strength = 1;
	bool stats_initialized = false;

	//! for debug, column names and tables
	vector<Identifier> column_names;
	Identifier table_name;
};

class RelationStatisticsHelper {
public:
	static constexpr double DEFAULT_SELECTIVITY = 0.2;

public:
	static idx_t InspectTableFilter(idx_t cardinality, const TableFilter &filter, BaseStatistics &base_stats);
	//! Extract Statistics from a LogicalGet.
	static RelationStats ExtractGetStats(LogicalGet &get, ClientContext &context);
	static RelationStats ExtractDelimGetStats(LogicalDelimGet &delim_get, ClientContext &context);
	//! Create the statistics for a projection using the statistics of the operator that sits underneath the
	//! projection. Then also create statistics for any extra columns the projection creates.
	static RelationStats ExtractDummyScanStats(LogicalDummyScan &dummy_scan, ClientContext &context);
	static RelationStats ExtractExpressionGetStats(LogicalExpressionGet &expression_get, ClientContext &context);
	//! All relation extractors for blocking relations
	static RelationStats ExtractProjectionStats(LogicalProjection &proj, RelationStats &child_stats);
	static RelationStats ExtractAggregationStats(LogicalAggregate &aggr, RelationStats &child_stats);
	static RelationStats ExtractWindowStats(LogicalWindow &window, RelationStats &child_stats);
	static RelationStats ExtractEmptyResultStats(LogicalEmptyResult &empty);
	//! Estimate a filter over a numeric computed column. Returns false when no distribution is available.
	static bool EstimateFilterCardinality(const Expression &filter, const RelationStats &stats, idx_t &cardinality);
	//! Called after reordering a query plan with potentially 2+ relations.
	static RelationStats CombineStatsOfReorderableOperator(vector<ColumnBinding> &bindings,
	                                                       vector<RelationStats> relation_stats);
	//! Called after reordering a query plan with potentially 2+ relations.
	static RelationStats CombineStatsOfNonReorderableOperator(LogicalOperator &op,
	                                                          const vector<RelationStats> &child_stats);
	static void CopyRelationStats(RelationStats &to, const RelationStats &from);

private:
	static unique_ptr<BaseStatistics> GetColumnStatistics(LogicalGet &get, ClientContext &context,
	                                                      const ColumnIndex &column_id);
};

} // namespace duckdb
