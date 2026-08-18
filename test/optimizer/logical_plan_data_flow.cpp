#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

using namespace duckdb;

static unique_ptr<LogicalOperator> CreateProjection(TableIndex input_index, TableIndex output_index) {
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(input_index, ProjectionIndex(0))));
	auto projection = make_uniq<LogicalProjection>(output_index, std::move(expressions));
	projection->children.push_back(make_uniq<LogicalDummyScan>(input_index));
	return std::move(projection);
}

static unique_ptr<LogicalOperator> CreateTwoColumnProjection(TableIndex input_index, TableIndex output_index) {
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(input_index, ProjectionIndex(0))));
	expressions.push_back(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(input_index, ProjectionIndex(0))));
	auto projection = make_uniq<LogicalProjection>(output_index, std::move(expressions));
	projection->children.push_back(make_uniq<LogicalDummyScan>(input_index));
	return std::move(projection);
}

class TestLogicalExtensionOperator : public LogicalExtensionOperator {
public:
	TestLogicalExtensionOperator(TableIndex table_index_p, bool pass_through_p)
	    : table_index(table_index_p), pass_through(pass_through_p) {
	}

	TableIndex table_index;
	bool pass_through;

public:
	vector<TableIndex> GetTableIndex() const override {
		return table_index.IsValid() ? vector<TableIndex> {table_index} : vector<TableIndex> {};
	}

	vector<ColumnBinding> GetColumnBindings() override {
		if (table_index.IsValid()) {
			return GenerateColumnBindings(table_index, 1);
		}
		if (pass_through && children.size() == 1) {
			return children[0]->GetColumnBindings();
		}
		return {};
	}

	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &) override {
		throw InternalException("TestLogicalExtensionOperator cannot create a physical plan");
	}

protected:
	void ResolveTypes() override {
		if (table_index.IsValid()) {
			types = {LogicalType::INTEGER};
		} else if (pass_through && children.size() == 1) {
			types = children[0]->types;
		}
	}
};

TEST_CASE("Logical plan data flow resolves renamed bindings", "[optimizer][logical_plan_data_flow]") {
	auto projection = CreateProjection(TableIndex(10), TableIndex(20));
	auto &scan = *projection->children[0];
	auto &projection_ref = *projection;
	auto filter = make_uniq<LogicalFilter>(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(20), ProjectionIndex(0))));
	filter->children.push_back(std::move(projection));

	LogicalPlanDataFlow data_flow(*filter);
	REQUIRE(data_flow.OperatorCount() == 3);
	REQUIRE(data_flow.Verify());

	auto input_source = data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, projection_ref);
	REQUIRE(input_source.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(input_source.op.get() == &scan);

	auto output_source = data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter);
	REQUIRE(output_source.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(output_source.op.get() == &projection_ref);

	auto hidden_source = data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *filter);
	REQUIRE(hidden_source.status == LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);

	auto path = data_flow.GetPathSummary(*filter, scan);
	REQUIRE(path.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(path.summary.Has(LogicalPlanPathProperty::PROJECTION_BOUNDARY));
}

TEST_CASE("Logical plan data flow validates source binding layouts", "[optimizer][logical_plan_data_flow]") {
	auto scan = make_uniq<LogicalDummyScan>(TableIndex(10));
	LogicalPlanDataFlow data_flow(*scan);

	auto missing_column = data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(1)), 0, *scan);
	REQUIRE(missing_column.status == LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);
}

TEST_CASE("Logical plan data flow finds join convergence", "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &left = *join->children[0];
	auto &right = *join->children[1];

	LogicalPlanDataFlow data_flow(*join);
	REQUIRE(data_flow.Verify());
	auto lca = data_flow.LowestCommonAncestor(left, right);
	REQUIRE(lca.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(lca.op.get() == join.get());
	REQUIRE(data_flow.IsFlowAncestor(*join, left).value);
	REQUIRE_FALSE(data_flow.IsFlowAncestor(left, *join).value);
}

TEST_CASE("Logical plan data flow respects join output maps", "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(join));

	LogicalPlanDataFlow data_flow(*filter);
	auto left = data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *filter);
	REQUIRE(left.status == LogicalPlanDataFlowStatus::SUCCESS);
	auto right = data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter);
	REQUIRE(right.status == LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);
}

TEST_CASE("Logical plan data flow models every comparison join output", "[optimizer][logical_plan_data_flow]") {
	const vector<JoinType> join_types {JoinType::LEFT,       JoinType::RIGHT,     JoinType::INNER, JoinType::OUTER,
	                                   JoinType::SEMI,       JoinType::ANTI,      JoinType::MARK,  JoinType::SINGLE,
	                                   JoinType::RIGHT_SEMI, JoinType::RIGHT_ANTI};
	for (auto join_type : join_types) {
		CAPTURE(join_type);
		auto join = make_uniq<LogicalComparisonJoin>(join_type);
		join->mark_index = TableIndex(30);
		join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
		join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
		auto &join_ref = *join;
		auto &left_ref = *join->children[0];
		auto &right_ref = *join->children[1];
		auto filter = make_uniq<LogicalFilter>();
		filter->children.push_back(std::move(join));

		LogicalPlanDataFlow data_flow(*filter);
		REQUIRE(data_flow.Verify());
		REQUIRE(data_flow.LowestCommonAncestor(left_ref, right_ref).op.get() == &join_ref);

		auto left = data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *filter);
		auto right = data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter);
		const bool projects_left = join_type != JoinType::RIGHT_SEMI && join_type != JoinType::RIGHT_ANTI;
		const bool projects_right =
		    join_type != JoinType::SEMI && join_type != JoinType::ANTI && join_type != JoinType::MARK;
		REQUIRE((left.status == LogicalPlanDataFlowStatus::SUCCESS) == projects_left);
		REQUIRE((right.status == LogicalPlanDataFlowStatus::SUCCESS) == projects_right);
		if (join_type == JoinType::MARK) {
			auto mark = data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, *filter);
			REQUIRE(mark.status == LogicalPlanDataFlowStatus::SUCCESS);
			REQUIRE(mark.op.get() == &join_ref);
		}
	}

	auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	join->left_projection_map = {ProjectionIndex(1)};
	join->children.push_back(CreateTwoColumnProjection(TableIndex(1), TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(join));
	LogicalPlanDataFlow data_flow(*filter);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(1)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
}

TEST_CASE("Logical plan data flow keeps materialized CTE components separate", "[optimizer][logical_plan_data_flow]") {
	auto producer = CreateProjection(TableIndex(10), TableIndex(20));
	auto &producer_ref = *producer;
	auto cte_ref = make_uniq<LogicalCTERef>(TableIndex(40), TableIndex(30), vector<LogicalType> {LogicalType::INTEGER},
	                                        vector<Identifier> {Identifier("i")});
	auto &reader_ref = *cte_ref;
	auto cte = make_uniq<LogicalMaterializedCTE>(Identifier("values"), TableIndex(30), 1, std::move(producer),
	                                             std::move(cte_ref), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	auto &cte_ref_op = *cte;
	auto filter = make_uniq<LogicalFilter>(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(40), ProjectionIndex(0))));
	filter->children.push_back(std::move(cte));

	LogicalPlanDataFlow data_flow(*filter);
	REQUIRE(data_flow.Verify());
	auto continuation_source = data_flow.ResolveSource(ColumnBinding(TableIndex(40), ProjectionIndex(0)), 0, *filter);
	REQUIRE(continuation_source.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(continuation_source.op.get() == &reader_ref);
	auto producer_result = data_flow.GetCTEProducer(TableIndex(30));
	REQUIRE(producer_result.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(producer_result.op.get() == &producer_ref);
	auto readers = data_flow.GetCTEReaders(TableIndex(30));
	REQUIRE(readers.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(readers.readers.size() == 1);
	REQUIRE(&readers.readers[0].get() == &reader_ref);

	auto connected = data_flow.SameFlowTree(producer_ref, cte_ref_op);
	REQUIRE(connected.status == LogicalPlanDataFlowStatus::CTE_BOUNDARY);
	REQUIRE_FALSE(connected.value);
	auto lca = data_flow.LowestCommonAncestor(producer_ref, reader_ref);
	REQUIRE(lca.status == LogicalPlanDataFlowStatus::CTE_BOUNDARY);
}

TEST_CASE("Logical plan data flow treats extension operators as opaque", "[optimizer][logical_plan_data_flow]") {
	for (idx_t child_count : {idx_t(0), idx_t(1), idx_t(2)}) {
		auto extension = make_uniq<TestLogicalExtensionOperator>(TableIndex(), true);
		for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
			extension->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10 + child_idx)));
		}
		vector<unique_ptr<Expression>> expressions;
		if (child_count == 1) {
			expressions.push_back(make_uniq<BoundColumnRefExpression>(
			    LogicalType::INTEGER, ColumnBinding(TableIndex(10), ProjectionIndex(0))));
		}
		auto projection = make_uniq<LogicalProjection>(TableIndex(30), std::move(expressions));
		projection->children.push_back(std::move(extension));
		LogicalPlanDataFlow data_flow(*projection);
		REQUIRE(data_flow.OperatorCount() == child_count + 2);
		REQUIRE(data_flow.Verify());
		for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
			auto source =
			    data_flow.ResolveSource(ColumnBinding(TableIndex(10 + child_idx), ProjectionIndex(0)), 0, *projection);
			REQUIRE(source.status == LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY);
		}
	}

	auto extension = make_uniq<TestLogicalExtensionOperator>(TableIndex(50), false);
	auto &extension_ref = *extension;
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(50), ProjectionIndex(0))));
	auto projection = make_uniq<LogicalProjection>(TableIndex(60), std::move(expressions));
	projection->children.push_back(std::move(extension));
	LogicalPlanDataFlow data_flow(*projection);
	REQUIRE(data_flow.Verify());
	auto fresh_source = data_flow.ResolveSource(ColumnBinding(TableIndex(50), ProjectionIndex(0)), 0, *projection);
	REQUIRE(fresh_source.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(fresh_source.op.get() == &extension_ref);
}

TEST_CASE("Logical plan data flow records correlated uses", "[optimizer][logical_plan_data_flow]") {
	auto scan = make_uniq<LogicalDummyScan>(TableIndex(10));
	auto filter = make_uniq<LogicalFilter>(make_uniq<BoundColumnRefExpression>(
	    Identifier("correlated"), LogicalType::INTEGER, ColumnBinding(TableIndex(20), ProjectionIndex(0)), 1));
	filter->children.push_back(std::move(scan));
	LogicalPlanDataFlow data_flow(*filter);
	REQUIRE(data_flow.GetBindingUses().size() == 1);
	REQUIRE(data_flow.GetBindingUses()[0].depth == 1);
	auto source = data_flow.ResolveSource(data_flow.GetBindingUses()[0].binding, 1, *filter);
	REQUIRE(source.status == LogicalPlanDataFlowStatus::CORRELATED_REFERENCE);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Logical plan data flow handles deep ownership trees iteratively", "[optimizer][logical_plan_data_flow]") {
	constexpr idx_t OPERATOR_COUNT = 10000;
	unique_ptr<LogicalOperator> plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	for (idx_t op_idx = 1; op_idx < OPERATOR_COUNT; op_idx++) {
		auto filter = make_uniq<LogicalFilter>();
		filter->children.push_back(std::move(plan));
		plan = std::move(filter);
	}
	LogicalPlanDataFlow data_flow(*plan);
	REQUIRE(data_flow.OperatorCount() == OPERATOR_COUNT);
	REQUIRE(data_flow.Verify());

	// Avoid relying on recursive unique_ptr destruction for the adversarial plan shape.
	while (!plan->children.empty()) {
		auto child = std::move(plan->children[0]);
		plan->children.clear();
		plan = std::move(child);
	}
}

TEST_CASE("Logical plan data flow verifies representative SQL plans", "[optimizer][logical_plan_data_flow]") {
	DuckDB db;
	Connection connection(db);
	const vector<string> queries {
	    "SELECT i FROM range(10) t(i) WHERE i > 3 ORDER BY i LIMIT 2",
	    "SELECT i, count(*) FROM range(10) t(i) GROUP BY GROUPING SETS ((i), ())",
	    "SELECT i, row_number() OVER (ORDER BY i) FROM range(10) t(i)",
	    "SELECT i, u FROM range(3) t(i), UNNEST([i, i + 1]) x(u)",
	    "SELECT * FROM range(3) l(i) JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) LEFT JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) SEMI JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) ANTI JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) ASOF JOIN range(3) r(j) ON i >= j",
	    "SELECT i FROM range(3) t(i) UNION SELECT i FROM range(3) u(i)",
	    "SELECT i FROM range(3) t(i) INTERSECT SELECT i FROM range(3) u(i)",
	    "SELECT i FROM range(3) t(i) EXCEPT SELECT i FROM range(3) u(i)",
	    "WITH values AS MATERIALIZED (SELECT i FROM range(3) t(i)) SELECT * FROM values",
	    "WITH RECURSIVE values(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM values WHERE i < 3) SELECT * FROM values",
	};
	for (auto &query : queries) {
		auto plan = connection.ExtractPlan(query);
		INFO(query);
		LogicalPlanDataFlow data_flow(*plan);
		REQUIRE(data_flow.Verify());
	}
}
