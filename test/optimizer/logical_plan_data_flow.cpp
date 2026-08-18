#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_dependent_join.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_unconditional_join.hpp"

#include <random>

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

static vector<reference<LogicalOperator>> CollectOperators(LogicalOperator &root) {
	vector<reference<LogicalOperator>> result;
	vector<reference<LogicalOperator>> pending {root};
	while (!pending.empty()) {
		auto op = pending.back();
		pending.pop_back();
		result.push_back(op);
		for (auto &child : op.get().children) {
			pending.push_back(*child);
		}
	}
	return result;
}

static void RequireEquivalentDataFlow(LogicalOperator &root, LogicalPlanDataFlow &live) {
	LogicalPlanDataFlow rebuilt(root);
	REQUIRE(live.Verify());
	REQUIRE(rebuilt.Verify());
	REQUIRE(live.OperatorCount() == rebuilt.OperatorCount());
	auto operators = CollectOperators(root);
	for (auto op : operators) {
		auto live_owner = live.GetOwnershipParent(op);
		auto rebuilt_owner = rebuilt.GetOwnershipParent(op);
		REQUIRE(live_owner.status == rebuilt_owner.status);
		REQUIRE(live_owner.parent == rebuilt_owner.parent);
		REQUIRE(live_owner.child_index == rebuilt_owner.child_index);
		auto live_flow = live.GetFlowParent(op);
		auto rebuilt_flow = rebuilt.GetFlowParent(op);
		REQUIRE(live_flow.status == rebuilt_flow.status);
		REQUIRE(live_flow.parent == rebuilt_flow.parent);
		REQUIRE(live_flow.child_index == rebuilt_flow.child_index);
		for (auto binding : op.get().GetColumnBindings()) {
			auto live_source = live.ResolveSource(binding, 0, root);
			auto rebuilt_source = rebuilt.ResolveSource(binding, 0, root);
			REQUIRE(live_source.status == rebuilt_source.status);
			REQUIRE(live_source.op == rebuilt_source.op);
		}
	}
	for (auto left : operators) {
		for (auto right : operators) {
			auto live_same = live.SameFlowTree(left, right);
			auto rebuilt_same = rebuilt.SameFlowTree(left, right);
			REQUIRE(live_same.status == rebuilt_same.status);
			REQUIRE(live_same.value == rebuilt_same.value);
			auto live_lca = live.LowestCommonAncestor(left, right);
			auto rebuilt_lca = rebuilt.LowestCommonAncestor(left, right);
			REQUIRE(live_lca.status == rebuilt_lca.status);
			REQUIRE(live_lca.op == rebuilt_lca.op);
			auto live_path = live.GetPathSummary(left, right);
			auto rebuilt_path = rebuilt.GetPathSummary(left, right);
			REQUIRE(live_path.status == rebuilt_path.status);
			REQUIRE(live_path.summary == rebuilt_path.summary);
		}
	}
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

TEST_CASE("Logical plan data flow connects specialized join inputs", "[optimizer][logical_plan_data_flow]") {
	vector<unique_ptr<LogicalOperator>> joins;
	joins.push_back(make_uniq<LogicalComparisonJoin>(JoinType::INNER, LogicalOperatorType::LOGICAL_DELIM_JOIN));
	joins.push_back(make_uniq<LogicalComparisonJoin>(JoinType::INNER, LogicalOperatorType::LOGICAL_ASOF_JOIN));
	auto any_join = make_uniq<LogicalAnyJoin>(JoinType::INNER);
	any_join->condition = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
	joins.push_back(std::move(any_join));
	auto dependent_join = make_uniq<LogicalDependentJoin>(JoinType::INNER);
	dependent_join->condition = make_uniq<BoundConstantExpression>(Value::BOOLEAN(true));
	joins.push_back(std::move(dependent_join));
	joins.push_back(make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT));
	joins.push_back(make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_POSITIONAL_JOIN));
	for (auto &join : joins) {
		CAPTURE(join->type);
		join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
		join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
		auto &left = *join->children[0];
		auto &right = *join->children[1];
		LogicalPlanDataFlow data_flow(*join);
		REQUIRE(data_flow.Verify());
		REQUIRE(data_flow.LowestCommonAncestor(left, right).op.get() == join.get());
	}
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

TEST_CASE("Logical plan data flow registers and unregisters detached subtrees", "[optimizer][logical_plan_data_flow]") {
	auto plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	{
		auto detached = mutator.RegisterSubtree(CreateProjection(TableIndex(20), TableIndex(30)));
		REQUIRE(data_flow.OperatorCount() == 3);
		REQUIRE(data_flow.Verify());
		auto &projection = detached.Get();
		auto source = data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, projection);
		REQUIRE(source.status == LogicalPlanDataFlowStatus::SUCCESS);
		REQUIRE(source.op.get() == projection.children[0].get());
	}
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.Verify());

	auto detached = mutator.RegisterSubtree(CreateProjection(TableIndex(40), TableIndex(50)));
	auto unregistered = mutator.UnregisterSubtree(std::move(detached));
	REQUIRE(unregistered->type == LogicalOperatorType::LOGICAL_PROJECTION);
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.GetOwnershipParent(*unregistered).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.Verify());

	REQUIRE_THROWS_AS(mutator.RegisterSubtree(make_uniq<LogicalDummyScan>(TableIndex(10))), InternalException);
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Logical plan data flow refreshes operator-local metadata", "[optimizer][logical_plan_data_flow]") {
	auto projection = CreateProjection(TableIndex(10), TableIndex(20));
	auto &projection_ref = projection->Cast<LogicalProjection>();
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(projection));
	LogicalPlanDataFlow data_flow(*filter);
	LogicalPlanDataFlowMutator mutator(data_flow);

	projection_ref.table_index = TableIndex(30);
	REQUIRE_FALSE(data_flow.Verify());
	mutator.RefreshOperator(projection_ref);
	filter->expressions.clear();
	filter->expressions.push_back(make_uniq<BoundColumnRefExpression>(
	    Identifier("correlated"), LogicalType::INTEGER, ColumnBinding(TableIndex(40), ProjectionIndex(0)), 1));
	REQUIRE_FALSE(data_flow.Verify());
	mutator.RefreshOperator(*filter);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_FOUND);
	auto renamed = data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, *filter);
	REQUIRE(renamed.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(renamed.op.get() == &projection_ref);
	REQUIRE(data_flow.GetBindingUses().size() == 2);
	REQUIRE((data_flow.GetBindingUses()[0].depth == 1 || data_flow.GetBindingUses()[1].depth == 1));
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Logical plan data flow refreshes CTE lineage", "[optimizer][logical_plan_data_flow]") {
	auto cte_ref = make_uniq<LogicalCTERef>(TableIndex(40), TableIndex(30), vector<LogicalType> {LogicalType::INTEGER},
	                                        vector<Identifier> {Identifier("i")});
	auto &cte_ref_op = *cte_ref;
	LogicalPlanDataFlow data_flow(*cte_ref);
	LogicalPlanDataFlowMutator mutator(data_flow);

	cte_ref->cte_index = TableIndex(31);
	REQUIRE_FALSE(data_flow.Verify());
	mutator.RefreshOperator(*cte_ref);
	REQUIRE(data_flow.GetCTEReaders(TableIndex(30)).status == LogicalPlanDataFlowStatus::BINDING_NOT_FOUND);
	auto readers = data_flow.GetCTEReaders(TableIndex(31));
	REQUIRE(readers.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(readers.readers.size() == 1);
	REQUIRE(&readers.readers[0].get() == &cte_ref_op);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations detach attach erase and swap children",
          "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &left = *join->children[0];
	auto &right = *join->children[1];
	LogicalPlanDataFlow data_flow(*join);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto detached = mutator.DetachChild(*join, 1);
	REQUIRE(data_flow.GetOwnershipParent(right).parent == nullptr);
	REQUIRE(data_flow.SameFlowTree(left, right).status == LogicalPlanDataFlowStatus::DISCONNECTED);
	mutator.AttachChild(*join, 0, std::move(detached));
	REQUIRE(data_flow.GetOwnershipParent(right).child_index == 0);
	REQUIRE(data_flow.GetOwnershipParent(left).child_index == 1);
	REQUIRE(data_flow.LowestCommonAncestor(left, right).op.get() == join.get());

	mutator.SwapChildren(*join, 0, 1);
	REQUIRE(join->children[0].get() == &left);
	REQUIRE(data_flow.GetOwnershipParent(left).child_index == 0);
	REQUIRE(data_flow.GetFlowParent(right).child_index == 1);

	mutator.AttachChild(*join, 2, make_uniq<LogicalDummyScan>(TableIndex(30)));
	auto &extra = *join->children[2];
	REQUIRE(data_flow.OperatorCount() == 4);
	auto erased = mutator.EraseChild(*join, 2);
	REQUIRE(erased.get() == &extra);
	REQUIRE(data_flow.GetOwnershipParent(*erased).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.OperatorCount() == 3);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations reject invalid attachments before ownership transfer",
          "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	LogicalPlanDataFlow data_flow(*join);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto detached = mutator.DetachChild(*join, 1);
	REQUIRE_THROWS_AS(mutator.AttachChild(*join, 2, std::move(detached)), InternalException);
	REQUIRE(join->children.size() == 1);
	REQUIRE(data_flow.OperatorCount() == 2);
	REQUIRE(data_flow.Verify());

	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(30)));
	mutator.AttachChild(*join, 1, std::move(filter));
	auto cyclic_subtree = mutator.DetachChild(*join, 1);
	auto &descendant = *cyclic_subtree.Get().children[0];
	REQUIRE_THROWS_AS(mutator.AttachChild(descendant, 0, std::move(cyclic_subtree)), InternalException);
	REQUIRE(join->children.size() == 1);
	REQUIRE(data_flow.OperatorCount() == 2);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations can move referenced join children", "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	join->conditions.emplace_back(
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(10), ProjectionIndex(0))),
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(20), ProjectionIndex(0))),
	    ExpressionType::COMPARE_EQUAL);
	auto &right = *join->children[1];
	LogicalPlanDataFlow data_flow(*join);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto detached = mutator.DetachChild(*join, 1);
	REQUIRE(data_flow.SameFlowTree(*join, right).status == LogicalPlanDataFlowStatus::DISCONNECTED);
	mutator.AttachChild(*join, 1, std::move(detached));
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *join).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Detached logical plan subtrees can outlive their data flow", "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	unique_ptr<LogicalPlanDataFlowDetachedSubtree> detached;
	{
		LogicalPlanDataFlow data_flow(*join);
		LogicalPlanDataFlowMutator mutator(data_flow);
		detached = make_uniq<LogicalPlanDataFlowDetachedSubtree>(mutator.DetachChild(*join, 1));
		REQUIRE(data_flow.OperatorCount() == 3);
		REQUIRE(data_flow.Verify());
	}
	REQUIRE(detached->Get().type == LogicalOperatorType::LOGICAL_DUMMY_SCAN);
	detached.reset();
}

TEST_CASE("Indexed logical plan mutations replace root and child subtrees", "[optimizer][logical_plan_data_flow]") {
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(CreateProjection(TableIndex(10), TableIndex(20)));
	LogicalPlanDataFlow data_flow(*filter);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto old_child = mutator.ReplaceSubtree(filter->children[0], CreateProjection(TableIndex(30), TableIndex(20)));
	REQUIRE(old_child->type == LogicalOperatorType::LOGICAL_PROJECTION);
	REQUIRE(data_flow.GetOwnershipParent(*old_child).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	auto new_source =
	    data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, *filter->children[0]);
	REQUIRE(new_source.status == LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(new_source.op.get() == filter->children[0]->children[0].get());
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter).op.get() ==
	        filter->children[0].get());
	REQUIRE(data_flow.Verify());

	unique_ptr<LogicalOperator> root = std::move(filter);
	auto old_root = mutator.ReplaceSubtree(root, make_uniq<LogicalDummyScan>(TableIndex(50)));
	REQUIRE(root->type == LogicalOperatorType::LOGICAL_DUMMY_SCAN);
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.GetOwnershipParent(*old_root).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations reject invalid replacements before mutation",
          "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &old_left = *join->children[0];
	LogicalPlanDataFlow data_flow(*join);
	LogicalPlanDataFlowMutator mutator(data_flow);

	REQUIRE_THROWS_AS(mutator.ReplaceSubtree(join->children[0], make_uniq<LogicalDummyScan>(TableIndex(20))),
	                  InternalException);
	REQUIRE(join->children[0].get() == &old_left);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *join).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations replace operator shells", "[optimizer][logical_plan_data_flow]") {
	auto cross_product = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &left = *cross_product->children[0];
	auto &right = *cross_product->children[1];
	unique_ptr<LogicalOperator> plan = std::move(cross_product);
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto comparison_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	auto old_operator = mutator.ReplaceOperator(plan, std::move(comparison_join));
	REQUIRE(plan->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN);
	REQUIRE(plan->children[0].get() == &left);
	REQUIRE(plan->children[1].get() == &right);
	REQUIRE(old_operator->children.empty());
	REQUIRE(data_flow.GetOwnershipParent(left).parent.get() == plan.get());
	REQUIRE(data_flow.GetOwnershipParent(right).parent.get() == plan.get());
	REQUIRE(data_flow.GetOwnershipParent(*old_operator).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Indexed logical plan mutations promote a selected child", "[optimizer][logical_plan_data_flow]") {
	auto cross_product = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &right = *cross_product->children[1];
	unique_ptr<LogicalOperator> plan = std::move(cross_product);
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto old_operator = mutator.PromoteChild(plan, 1);
	REQUIRE(plan.get() == &right);
	REQUIRE(old_operator->children.empty());
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.GetOwnershipParent(*old_operator).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *plan).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_FOUND);
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Indexed logical plan operator changes reject invalid replacements before mutation",
          "[optimizer][logical_plan_data_flow]") {
	auto cross_product = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	cross_product->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &left = *cross_product->children[0];
	unique_ptr<LogicalOperator> plan = std::move(cross_product);
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto replacement = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	replacement->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(30)));
	REQUIRE_THROWS_AS(mutator.ReplaceOperator(plan, std::move(replacement)), InternalException);
	REQUIRE(plan->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	REQUIRE(plan->children[0].get() == &left);
	REQUIRE_THROWS_AS(mutator.PromoteChild(plan, 2), InternalException);
	REQUIRE(plan->children[0].get() == &left);
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Indexed logical plan mutations insert and remove unary operators", "[optimizer][logical_plan_data_flow]") {
	unique_ptr<LogicalOperator> plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	auto &scan = *plan;
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	mutator.InsertUnary(plan, make_uniq<LogicalFilter>());
	REQUIRE(plan->type == LogicalOperatorType::LOGICAL_FILTER);
	REQUIRE(plan->children[0].get() == &scan);
	REQUIRE(data_flow.GetOwnershipParent(scan).parent.get() == plan.get());
	REQUIRE(data_flow.OperatorCount() == 2);
	REQUIRE(data_flow.Verify());

	auto removed = mutator.RemoveUnary(plan);
	REQUIRE(plan.get() == &scan);
	REQUIRE(removed->children.empty());
	REQUIRE(data_flow.GetOwnershipParent(*removed).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.Verify());

	auto join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(30)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(40)));
	auto &child_scan = *join->children[0];
	LogicalPlanDataFlow join_data_flow(*join);
	LogicalPlanDataFlowMutator join_mutator(join_data_flow);
	join_mutator.InsertUnary(join->children[0], make_uniq<LogicalFilter>());
	REQUIRE(join->children[0]->children[0].get() == &child_scan);
	auto child_wrapper = join_mutator.RemoveUnary(join->children[0]);
	REQUIRE(join->children[0].get() == &child_scan);
	REQUIRE(child_wrapper->children.empty());
	REQUIRE(join_data_flow.Verify());

	REQUIRE_THROWS_AS(join_mutator.InsertUnary(join->children[0], make_uniq<LogicalComparisonJoin>(JoinType::INNER)),
	                  InternalException);
	REQUIRE(join->children[0].get() == &child_scan);
	REQUIRE(join_data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations insert binary parents", "[optimizer][logical_plan_data_flow]") {
	unique_ptr<LogicalOperator> plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	auto &left = *plan;
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto root_join = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	root_join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &right = *root_join->children[0];
	mutator.InsertParent(plan, std::move(root_join), 0);
	REQUIRE(plan->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	REQUIRE(plan->children[0].get() == &left);
	REQUIRE(plan->children[1].get() == &right);
	REQUIRE(data_flow.LowestCommonAncestor(left, right).op.get() == plan.get());
	RequireEquivalentDataFlow(*plan, data_flow);

	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(plan));
	unique_ptr<LogicalOperator> nested_plan = std::move(filter);
	LogicalPlanDataFlow nested_data_flow(*nested_plan);
	LogicalPlanDataFlowMutator nested_mutator(nested_data_flow);
	auto nested_join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	nested_join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(30)));
	auto &nested_sibling = *nested_join->children[0];
	auto &old_child = *nested_plan->children[0];
	nested_mutator.InsertParent(nested_plan->children[0], std::move(nested_join), 1);
	REQUIRE(nested_plan->children[0]->children[0].get() == &nested_sibling);
	REQUIRE(nested_plan->children[0]->children[1].get() == &old_child);
	REQUIRE(nested_data_flow.GetOwnershipParent(*nested_plan->children[0]).parent.get() == nested_plan.get());
	RequireEquivalentDataFlow(*nested_plan, nested_data_flow);
}

TEST_CASE("Indexed logical plan mutations insert materialized CTE parents", "[optimizer][logical_plan_data_flow]") {
	unique_ptr<LogicalOperator> plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	reference<LogicalOperator> continuation = *plan;
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto cte = make_uniq<LogicalMaterializedCTE>(Identifier("inserted"), TableIndex(30), 1,
	                                             make_uniq<LogicalDummyScan>(TableIndex(20)),
	                                             CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	reference<LogicalOperator> producer = *cte->children[0];
	mutator.InsertParent(plan, std::move(cte), 1);

	REQUIRE(plan->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE);
	REQUIRE(plan->children[0].get() == &producer.get());
	REQUIRE(plan->children[1].get() == &continuation.get());
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Indexed logical plan mutations reject invalid binary parents", "[optimizer][logical_plan_data_flow]") {
	unique_ptr<LogicalOperator> plan = make_uniq<LogicalDummyScan>(TableIndex(10));
	auto &scan = *plan;
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	REQUIRE_THROWS_AS(
	    mutator.InsertParent(plan, make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT), 0),
	    InternalException);
	auto unary = make_uniq<LogicalFilter>();
	unary->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	REQUIRE_THROWS_AS(mutator.InsertParent(plan, std::move(unary), 0), InternalException);
	auto invalid_index = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	invalid_index->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	REQUIRE_THROWS_AS(mutator.InsertParent(plan, std::move(invalid_index), 2), InternalException);
	auto duplicate_source = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	duplicate_source->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	REQUIRE_THROWS_AS(mutator.InsertParent(plan, std::move(duplicate_source), 0), InternalException);
	REQUIRE(plan.get() == &scan);
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Indexed logical plan mutations rotate a parent with its child", "[optimizer][logical_plan_data_flow]") {
	auto producer = CreateProjection(TableIndex(10), TableIndex(20));
	auto &producer_ref = *producer;
	auto continuation =
	    make_uniq<LogicalCTERef>(TableIndex(40), TableIndex(30), vector<LogicalType> {LogicalType::INTEGER},
	                             vector<Identifier> {Identifier("i")});
	auto &continuation_ref = *continuation;
	auto cte = make_uniq<LogicalMaterializedCTE>(Identifier("values"), TableIndex(30), 1, std::move(producer),
	                                             std::move(continuation), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	auto &cte_ref = *cte;
	auto dependent_join = make_uniq<LogicalDependentJoin>(JoinType::INNER);
	dependent_join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(1)));
	dependent_join->children.push_back(std::move(cte));
	auto &dependent_join_ref = *dependent_join;
	unique_ptr<LogicalOperator> plan = std::move(dependent_join);
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	mutator.RotateParentWithChild(plan, 1, 1);
	REQUIRE(plan.get() == &cte_ref);
	REQUIRE(plan->children[0].get() == &producer_ref);
	REQUIRE(plan->children[1].get() == &dependent_join_ref);
	REQUIRE(plan->children[1]->children[1].get() == &continuation_ref);
	REQUIRE(data_flow.GetCTEProducer(TableIndex(30)).op.get() == &producer_ref);
	REQUIRE(&data_flow.GetCTEReaders(TableIndex(30)).readers[0].get() == &continuation_ref);
	RequireEquivalentDataFlow(*plan, data_flow);

	auto nested_continuation =
	    make_uniq<LogicalCTERef>(TableIndex(80), TableIndex(70), vector<LogicalType> {LogicalType::INTEGER},
	                             vector<Identifier> {Identifier("i")});
	auto nested_cte = make_uniq<LogicalMaterializedCTE>(
	    Identifier("nested"), TableIndex(70), 1, CreateProjection(TableIndex(50), TableIndex(60)),
	    std::move(nested_continuation), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	auto &nested_cte_ref = *nested_cte;
	auto nested_dependent_join = make_uniq<LogicalDependentJoin>(JoinType::INNER);
	nested_dependent_join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(41)));
	nested_dependent_join->children.push_back(std::move(nested_cte));
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(nested_dependent_join));
	unique_ptr<LogicalOperator> nested_plan = std::move(filter);
	LogicalPlanDataFlow nested_data_flow(*nested_plan);
	LogicalPlanDataFlowMutator nested_mutator(nested_data_flow);
	nested_mutator.RotateParentWithChild(nested_plan->children[0], 1, 1);
	REQUIRE(nested_plan->children[0].get() == &nested_cte_ref);
	REQUIRE(nested_data_flow.GetOwnershipParent(nested_cte_ref).parent.get() == nested_plan.get());
	RequireEquivalentDataFlow(*nested_plan, nested_data_flow);
}

TEST_CASE("Indexed logical plan mutation scopes coordinate metadata rewrites", "[optimizer][logical_plan_data_flow]") {
	unique_ptr<LogicalOperator> plan = CreateProjection(TableIndex(10), TableIndex(20));
	auto &upper_projection = plan->Cast<LogicalProjection>();
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);

	{
		auto mutation = mutator.BeginMutation();
		vector<unique_ptr<Expression>> lower_expressions;
		lower_expressions.push_back(make_uniq<BoundColumnRefExpression>(
		    LogicalType::INTEGER, ColumnBinding(TableIndex(10), ProjectionIndex(0))));
		mutator.InsertUnary(plan->children[0],
		                    make_uniq<LogicalProjection>(TableIndex(30), std::move(lower_expressions)));
		REQUIRE_THROWS_AS(data_flow.Verify(), InternalException);
		upper_projection.expressions[0] = make_uniq<BoundColumnRefExpression>(
		    LogicalType::INTEGER, ColumnBinding(TableIndex(30), ProjectionIndex(0)));
		mutator.RefreshOperator(upper_projection);
	}
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, upper_projection).op.get() ==
	        plan->children[0].get());
	RequireEquivalentDataFlow(*plan, data_flow);
}

TEST_CASE("Logical plan data flow refreshes join output semantics", "[optimizer][logical_plan_data_flow]") {
	auto join = make_uniq<LogicalComparisonJoin>(JoinType::INNER);
	join->children.push_back(CreateTwoColumnProjection(TableIndex(1), TableIndex(10)));
	join->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	auto &join_ref = *join;
	auto filter = make_uniq<LogicalFilter>();
	filter->children.push_back(std::move(join));
	LogicalPlanDataFlow data_flow(*filter);
	LogicalPlanDataFlowMutator mutator(data_flow);

	join_ref.join_type = JoinType::MARK;
	join_ref.mark_index = TableIndex(30);
	mutator.RefreshOperator(join_ref);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(20), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);

	join_ref.join_type = JoinType::INNER;
	join_ref.left_projection_map = {ProjectionIndex(1)};
	mutator.RefreshOperator(join_ref);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(30), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_FOUND);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(0)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(10), ProjectionIndex(1)), 0, *filter).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations maintain materialized CTE producers", "[optimizer][logical_plan_data_flow]") {
	auto producer = CreateProjection(TableIndex(10), TableIndex(20));
	auto continuation =
	    make_uniq<LogicalCTERef>(TableIndex(40), TableIndex(30), vector<LogicalType> {LogicalType::INTEGER},
	                             vector<Identifier> {Identifier("i")});
	auto cte = make_uniq<LogicalMaterializedCTE>(Identifier("values"), TableIndex(30), 1, std::move(producer),
	                                             std::move(continuation), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
	LogicalPlanDataFlow data_flow(*cte);
	LogicalPlanDataFlowMutator mutator(data_flow);

	auto old_producer = mutator.ReplaceSubtree(cte->children[0], CreateProjection(TableIndex(11), TableIndex(21)));
	REQUIRE(data_flow.GetCTEProducer(TableIndex(30)).op.get() == cte->children[0].get());
	REQUIRE(data_flow.GetOwnershipParent(*old_producer).status == LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED);
	REQUIRE(data_flow.ResolveSource(ColumnBinding(TableIndex(11), ProjectionIndex(0)), 0, *cte->children[0]).status ==
	        LogicalPlanDataFlowStatus::SUCCESS);
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations reject opaque extension child edits", "[optimizer][logical_plan_data_flow]") {
	auto extension = make_uniq<TestLogicalExtensionOperator>(TableIndex(), true);
	extension->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	auto &child = *extension->children[0];
	LogicalPlanDataFlow data_flow(*extension);
	LogicalPlanDataFlowMutator mutator(data_flow);

	REQUIRE_THROWS_AS(mutator.DetachChild(*extension, 0), InternalException);
	REQUIRE(extension->children[0].get() == &child);
	REQUIRE(data_flow.GetOwnershipParent(child).parent.get() == extension.get());
	REQUIRE(data_flow.Verify());
}

TEST_CASE("Indexed logical plan mutations agree with rebuilding after randomized edits",
          "[optimizer][logical_plan_data_flow]") {
	auto plan = make_uniq<LogicalUnconditionalJoin>(LogicalOperatorType::LOGICAL_CROSS_PRODUCT);
	plan->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(10)));
	plan->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(20)));
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);
	std::mt19937_64 random(0xD0CDBULL);
	idx_t next_table_index = 100;

	for (idx_t iteration = 0; iteration < 50; iteration++) {
		auto operation = random() % 5;
		auto child_index = NumericCast<idx_t>(random() % 2);
		switch (operation) {
		case 0:
			mutator.SwapChildren(*plan, 0, 1);
			break;
		case 1:
			if (plan->children[child_index]->type == LogicalOperatorType::LOGICAL_FILTER) {
				mutator.RemoveUnary(plan->children[child_index]);
			} else {
				mutator.InsertUnary(plan->children[child_index], make_uniq<LogicalFilter>());
			}
			break;
		case 2:
			mutator.ReplaceSubtree(plan->children[child_index],
			                       make_uniq<LogicalDummyScan>(TableIndex(next_table_index++)));
			break;
		case 3: {
			auto detached = mutator.DetachChild(*plan, child_index);
			mutator.AttachChild(*plan, 1 - child_index, std::move(detached));
			break;
		}
		case 4:
			mutator.AttachChild(*plan, 2, make_uniq<LogicalDummyScan>(TableIndex(next_table_index++)));
			mutator.EraseChild(*plan, 2);
			break;
		default:
			FAIL("Unexpected randomized mutation");
		}
		RequireEquivalentDataFlow(*plan, data_flow);
	}
}

TEST_CASE("Indexed logical plan mutation registers deep subtrees iteratively", "[optimizer][logical_plan_data_flow]") {
	constexpr idx_t OPERATOR_COUNT = 10000;
	auto plan = make_uniq<LogicalDummyScan>(TableIndex(1));
	LogicalPlanDataFlow data_flow(*plan);
	LogicalPlanDataFlowMutator mutator(data_flow);
	unique_ptr<LogicalOperator> subtree = make_uniq<LogicalDummyScan>(TableIndex(10));
	for (idx_t op_idx = 1; op_idx < OPERATOR_COUNT; op_idx++) {
		auto filter = make_uniq<LogicalFilter>();
		filter->children.push_back(std::move(subtree));
		subtree = std::move(filter);
	}

	auto detached = mutator.RegisterSubtree(std::move(subtree));
	REQUIRE(data_flow.OperatorCount() == OPERATOR_COUNT + 1);
	REQUIRE(data_flow.Verify());
	auto unregistered = mutator.UnregisterSubtree(std::move(detached));
	REQUIRE(data_flow.OperatorCount() == 1);
	REQUIRE(data_flow.Verify());
	while (!unregistered->children.empty()) {
		auto child = std::move(unregistered->children[0]);
		unregistered->children.clear();
		unregistered = std::move(child);
	}
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
	    "SELECT * FROM range(3) l(i), range(3) r(j)",
	    "SELECT * FROM range(3) l(i) LEFT JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) RIGHT JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) FULL JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) SEMI JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) ANTI JOIN range(3) r(j) ON i = j",
	    "SELECT * FROM range(3) l(i) ASOF JOIN range(3) r(j) ON i >= j",
	    "SELECT i, i IN (SELECT j FROM range(3) r(j)) FROM range(3) l(i)",
	    "SELECT i, (SELECT j FROM range(1) r(j) WHERE j = i) FROM range(3) l(i)",
	    "SELECT i FROM range(3) l(i) WHERE i = ANY (SELECT j FROM range(3) r(j))",
	    "SELECT i FROM range(3) l(i) WHERE EXISTS (SELECT 1 FROM range(3) r(j) WHERE i = j)",
	    "SELECT i FROM range(3) t(i) UNION SELECT i FROM range(3) u(i)",
	    "SELECT i FROM range(3) t(i) INTERSECT SELECT i FROM range(3) u(i)",
	    "SELECT i FROM range(3) t(i) EXCEPT SELECT i FROM range(3) u(i)",
	    "WITH values AS MATERIALIZED (SELECT i FROM range(3) t(i)) SELECT * FROM values",
	    "WITH values AS MATERIALIZED (SELECT i FROM range(3) t(i)) SELECT * FROM values a, values b",
	    "WITH outer_values AS MATERIALIZED (WITH inner_values AS MATERIALIZED (SELECT i FROM range(3) t(i)) "
	    "SELECT * FROM inner_values) SELECT * FROM outer_values",
	    "WITH RECURSIVE values(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM values WHERE i < 3) SELECT * FROM values",
	};
	for (auto &query : queries) {
		auto plan = connection.ExtractPlan(query);
		INFO(query);
		LogicalPlanDataFlow data_flow(*plan);
		REQUIRE(data_flow.Verify());
	}

	REQUIRE_NO_FAIL(connection.Query("CREATE TABLE dml_values(i INTEGER)"));
	const vector<string> statement_queries {
	    "INSERT INTO dml_values VALUES (1)",     "UPDATE dml_values SET i = i + 1",
	    "DELETE FROM dml_values WHERE i > 0",    "EXPLAIN SELECT * FROM dml_values",
	    "CREATE TABLE another_table(i INTEGER)",
	};
	for (auto &query : statement_queries) {
		auto plan = connection.ExtractPlan(query);
		INFO(query);
		LogicalPlanDataFlow data_flow(*plan);
		REQUIRE(data_flow.Verify());
	}
}
