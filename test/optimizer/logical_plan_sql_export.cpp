#include "catch.hpp"
#include "test_helpers.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_default_expression.hpp"
#include "duckdb/planner/logical_plan_sql_exporter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_dummy_scan.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator_extension.hpp"
#include "duckdb/planner/planner.hpp"

#include <new>
#include <stdexcept>
#include <type_traits>

using namespace duckdb;

namespace {

using PlanExportResult = LogicalPlanVerificationResult<LogicalPlanSQLExportRelation>;

static_assert(!std::is_copy_constructible<LogicalPlanSQLExportRelation>::value,
              "Logical plan SQL export relations must remain move-only");

static unique_ptr<LogicalOperator> OptimizeLogicalPlanExportQuery(Connection &connection, const string &query) {
	Parser parser(connection.context->GetParserOptions());
	parser.ParseQuery(query);
	REQUIRE(parser.statements.size() == 1);
	Planner planner(*connection.context);
	planner.CreatePlan(std::move(parser.statements[0]));
	Optimizer optimizer(*planner.binder, *connection.context);
	return optimizer.Optimize(std::move(planner.plan));
}

static optional_ptr<LogicalOperator> FindLogicalPlanExportOperator(LogicalOperator &op, LogicalOperatorType type) {
	if (op.type == type) {
		return op;
	}
	for (auto &child : op.children) {
		if (!child) {
			continue;
		}
		auto result = FindLogicalPlanExportOperator(*child, type);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

static const Value &GetLogicalPlanExportFact(const LogicalPlanVerificationIssue &issue, const string &name) {
	for (auto &fact : issue.facts) {
		if (fact.first == name) {
			return fact.second;
		}
	}
	throw InternalException("Missing logical plan SQL export issue fact");
}

static void RequireLogicalPlanExportIssue(const PlanExportResult &result, LogicalPlanVerificationIssueCode code,
                                          LogicalPlanVerificationPhase phase, const LogicalPlanVerificationPath &path) {
	REQUIRE(result.IsValid());
	REQUIRE(result.HasError());
	REQUIRE_FALSE(result.IsSuccess());
	REQUIRE(result.GetIssues().size() == 1);
	INFO(result.GetIssues()[0].message);
	REQUIRE(result.GetIssues()[0].code == code);
	REQUIRE(result.GetIssues()[0].phase == phase);
	REQUIRE(result.GetIssues()[0].path == optional<LogicalPlanVerificationPath>(path));
}

static void RequirePlanExportIssue(const PlanExportResult &result, LogicalPlanVerificationIssueCode code,
                                   const LogicalPlanVerificationPath &path = {}) {
	RequireLogicalPlanExportIssue(result, code, LogicalPlanVerificationPhase::PLAN_EXPORT, path);
}

static void RequireInvariant(const PlanExportResult &result, const string &invariant,
                             const LogicalPlanVerificationPath &path = {}) {
	RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::INTERNAL_INVARIANT, path);
	REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value(invariant));
}

static unique_ptr<Expression> PlanIntegerConstant(int32_t value) {
	return make_uniq<BoundConstantExpression>(Value::INTEGER(value));
}

static unique_ptr<LogicalExpressionGet>
IntegerValues(TableIndex table_index, std::initializer_list<std::initializer_list<int32_t>> input_rows) {
	vector<vector<unique_ptr<Expression>>> rows;
	idx_t column_count = 0;
	for (auto &input_row : input_rows) {
		vector<unique_ptr<Expression>> row;
		for (auto value : input_row) {
			row.push_back(PlanIntegerConstant(value));
		}
		column_count = row.size();
		rows.push_back(std::move(row));
	}
	vector<LogicalType> types(column_count, LogicalType::INTEGER);
	auto result = make_uniq<LogicalExpressionGet>(table_index, std::move(types), std::move(rows));
	result->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(table_index.index + 1000)));
	return result;
}

static unique_ptr<LogicalProjection> PlanProjection(TableIndex table_index, unique_ptr<LogicalOperator> child,
                                                    vector<unique_ptr<Expression>> expressions) {
	auto result = make_uniq<LogicalProjection>(table_index, std::move(expressions));
	result->children.push_back(std::move(child));
	return result;
}

static unique_ptr<ParsedExpression> PlanParsedInteger(int32_t value, const Identifier &alias) {
	auto result = make_uniq<ConstantExpression>(Value::INTEGER(value));
	result->SetAlias(alias);
	return std::move(result);
}

static LogicalPlanSQLExportRelation PlanConstantRelation(const LogicalPlanSQLExportExtensionInput &input,
                                                         int32_t value) {
	auto select = make_uniq<SelectNode>();
	for (auto &field : input.output_fields) {
		select->select_list.push_back(PlanParsedInteger(value, field.identifier));
	}
	return LogicalPlanSQLExportRelation {std::move(select), input.output_fields};
}

class SQLExportExtensionOperator : public LogicalExtensionOperator {
public:
	SQLExportExtensionOperator(string name_p, vector<ColumnBinding> bindings_p, vector<LogicalType> types_p,
	                           vector<TableIndex> table_indexes_p = {},
	                           vector<unique_ptr<Expression>> expressions_p = {}, string verification_name_p = {})
	    : LogicalExtensionOperator(std::move(expressions_p)), name(std::move(name_p)),
	      verification_name(verification_name_p.empty() ? name : std::move(verification_name_p)),
	      bindings(std::move(bindings_p)), resolved_types(std::move(types_p)),
	      table_indexes(std::move(table_indexes_p)) {
	}

	vector<ColumnBinding> GetColumnBindings() override {
		return bindings;
	}

	vector<TableIndex> GetTableIndex() const override {
		return table_indexes;
	}

	optional_ptr<const string> GetTypeBindingVerificationIdentifier() const noexcept override {
		return verification_name;
	}

	string GetExtensionName() const override {
		return name;
	}

	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &) override {
		throw NotImplementedException("Synthetic SQL export operator cannot create a physical plan");
	}

protected:
	void ResolveTypes() override {
		types = resolved_types;
	}

private:
	string name;
	string verification_name;
	vector<ColumnBinding> bindings;
	vector<LogicalType> resolved_types;
	vector<TableIndex> table_indexes;
};

class SQLExportOperatorExtension : public OperatorExtension {
public:
	explicit SQLExportOperatorExtension(string name_p) : name(std::move(name_p)) {
		Bind = nullptr;
	}

	std::string GetName() override {
		return name;
	}

	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &) override {
		return nullptr;
	}

private:
	string name;
};

class LegacyOperatorExtension : public OperatorExtension {
public:
	LegacyOperatorExtension() {
		Bind = nullptr;
	}

	std::string GetName() override {
		return "legacy_sql_export_test";
	}

	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &) override {
		return nullptr;
	}
};

class SyntheticLogicalOperator : public LogicalOperator {
public:
	explicit SyntheticLogicalOperator(LogicalOperatorType type_p) : LogicalOperator(type_p) {
	}

	vector<ColumnBinding> GetColumnBindings() override {
		return {};
	}

protected:
	void ResolveTypes() override {
	}
};

static unique_ptr<SQLExportExtensionOperator> LeafExtension(const string &name, TableIndex table_index) {
	auto binding = ColumnBinding(table_index, ProjectionIndex(0));
	return make_uniq<SQLExportExtensionOperator>(name, vector<ColumnBinding> {binding},
	                                             vector<LogicalType> {LogicalType::INTEGER},
	                                             vector<TableIndex> {table_index});
}

static unique_ptr<SQLExportExtensionOperator> ChildExtension(const string &name, TableIndex child_index) {
	auto binding = ColumnBinding(child_index, ProjectionIndex(0));
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(
	    make_uniq<BoundColumnRefExpression>(Identifier("display_name"), LogicalType::INTEGER, binding));
	auto result = make_uniq<SQLExportExtensionOperator>(name, vector<ColumnBinding> {binding},
	                                                    vector<LogicalType> {LogicalType::INTEGER},
	                                                    vector<TableIndex> {}, std::move(expressions));
	result->children.push_back(IntegerValues(child_index, {{42}}));
	return result;
}

static LogicalPlanSQLExportRelation PlanChildRelation(const LogicalPlanSQLExportExtensionInput &input) {
	if (input.children.size() != 1 || input.expression_count != 1) {
		throw InternalException("Unexpected synthetic extension export input");
	}
	auto expression = input.export_expression(0);
	if (expression.HasError()) {
		throw InternalException("Synthetic extension expression export failed");
	}
	expression.GetValue()->SetAlias(input.output_fields[0].identifier);
	auto statement = make_uniq<SelectStatement>();
	statement->node = input.children[0].relation.query->Copy();
	auto select = make_uniq<SelectNode>();
	select->select_list.push_back(std::move(expression.GetValue()));
	select->from_table = make_uniq<SubqueryRef>(std::move(statement), input.children[0].relation_alias);
	return LogicalPlanSQLExportRelation {std::move(select), input.output_fields};
}

static LogicalPlanSQLExportOptions PlanResolverOptions(const string &identifier, logical_plan_sql_export_t callback) {
	LogicalPlanSQLExportOptions options;
	options.extension_resolvers.push_back({identifier, std::move(callback)});
	return options;
}

static unique_ptr<SQLExportExtensionOperator> MalformedRelationPlan() {
	return LeafExtension("malformed_relation", TableIndex(700));
}

static PlanExportResult ExportMalformedRelation(Connection &connection,
                                                std::function<void(LogicalPlanSQLExportExtensionResult &)> mutate) {
	auto plan = MalformedRelationPlan();
	auto options =
	    PlanResolverOptions("malformed_relation_resolver", [&](const LogicalPlanSQLExportExtensionInput &input) {
		    auto result = LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 42));
		    mutate(result);
		    return result;
	    });
	return LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
}

} // namespace

TEST_CASE("Logical plan SQL export executes the frozen vertical slice", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	connection.BeginTransaction();
	const string query = "SELECT sum(i) AS total FROM (VALUES (1), (2), (2), (4)) AS t(i) WHERE i > 1";
	auto plan = OptimizeLogicalPlanExportQuery(connection, query);

	auto first = LogicalPlanSQLExporter::Export(*connection.context, *plan);
	REQUIRE(first.IsSuccess());
	REQUIRE(first.GetValue().fields.size() == 1);
	REQUIRE(first.GetValue().fields[0].type == LogicalType::HUGEINT);
	REQUIRE(first.GetValue().fields[0].identifier == Identifier("c0"));
	REQUIRE(first.GetValue().fields[0].ordinal == 0);
	REQUIRE(first.GetValue().fields[0].source_binding == plan->GetColumnBindings()[0]);

	auto second = LogicalPlanSQLExporter::Export(*connection.context, *plan);
	REQUIRE(second.IsSuccess());
	REQUIRE(first.GetValue().query->Equals(second.GetValue().query.get()));
	REQUIRE(first.GetValue().query->ToString() == second.GetValue().query->ToString());
	REQUIRE(first.GetValue().fields[0].source_binding == second.GetValue().fields[0].source_binding);
	REQUIRE(first.GetValue().fields[0].type == second.GetValue().fields[0].type);
	REQUIRE(first.GetValue().fields[0].identifier == second.GetValue().fields[0].identifier);

	Parser parser(connection.context->GetParserOptions());
	parser.ParseQuery(first.GetValue().query->ToString());
	REQUIRE(parser.statements.size() == 1);
	Planner planner(*connection.context);
	planner.CreatePlan(std::move(parser.statements[0]));

	auto direct = connection.Query(query);
	auto exported = connection.Query(first.GetValue().query->ToString());
	REQUIRE_FALSE(direct->HasError());
	REQUIRE_FALSE(exported->HasError());
	REQUIRE(direct->GetTypes() == vector<LogicalType> {LogicalType::HUGEINT});
	REQUIRE(exported->GetTypes() == direct->GetTypes());
	auto direct_chunk = direct->Fetch();
	auto exported_chunk = exported->Fetch();
	REQUIRE(direct_chunk);
	REQUIRE(exported_chunk);
	REQUIRE(direct_chunk->size() == 1);
	REQUIRE(exported_chunk->size() == 1);
	REQUIRE(direct_chunk->GetValue(0, 0) == Value::HUGEINT(8));
	REQUIRE(exported_chunk->GetValue(0, 0) == direct_chunk->GetValue(0, 0));
	connection.Rollback();
}

TEST_CASE("Logical plan SQL export preserves values and canonical fields", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto plan = IntegerValues(TableIndex(10), {{1, 2}, {3, 4}});

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
	REQUIRE(result.IsSuccess());
	REQUIRE(result.GetValue().fields.size() == 2);
	for (idx_t field_index = 0; field_index < 2; field_index++) {
		REQUIRE(result.GetValue().fields[field_index].source_binding ==
		        ColumnBinding(TableIndex(10), ProjectionIndex(field_index)));
		REQUIRE(result.GetValue().fields[field_index].type == LogicalType::INTEGER);
		REQUIRE(result.GetValue().fields[field_index].identifier == Identifier("c" + to_string(field_index)));
		REQUIRE(result.GetValue().fields[field_index].ordinal == field_index);
	}
	auto query_result = connection.Query(result.GetValue().query->ToString());
	REQUIRE_FALSE(query_result->HasError());
	REQUIRE(query_result->GetTypes() == vector<LogicalType> {LogicalType::INTEGER, LogicalType::INTEGER});
	auto chunk = query_result->Fetch();
	REQUIRE(chunk);
	REQUIRE(chunk->size() == 2);
	REQUIRE(chunk->GetValue(0, 0) == Value::INTEGER(1));
	REQUIRE(chunk->GetValue(1, 0) == Value::INTEGER(2));
	REQUIRE(chunk->GetValue(0, 1) == Value::INTEGER(3));
	REQUIRE(chunk->GetValue(1, 1) == Value::INTEGER(4));
}

TEST_CASE("Logical plan SQL export resolves bindings independently of display aliases", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto child = IntegerValues(TableIndex(20), {{10, 20}});
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(make_uniq<BoundColumnRefExpression>(Identifier("same"), LogicalType::INTEGER,
	                                                          ColumnBinding(TableIndex(20), ProjectionIndex(1))));
	expressions.push_back(make_uniq<BoundColumnRefExpression>(Identifier("same"), LogicalType::INTEGER,
	                                                          ColumnBinding(TableIndex(20), ProjectionIndex(0))));
	expressions.push_back(make_uniq<BoundColumnRefExpression>(Identifier("same"), LogicalType::INTEGER,
	                                                          ColumnBinding(TableIndex(20), ProjectionIndex(1))));
	auto plan = PlanProjection(TableIndex(21), std::move(child), std::move(expressions));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
	REQUIRE(result.IsSuccess());
	auto query_result = connection.Query(result.GetValue().query->ToString());
	REQUIRE_FALSE(query_result->HasError());
	auto chunk = query_result->Fetch();
	REQUIRE(chunk);
	REQUIRE(chunk->size() == 1);
	REQUIRE(chunk->GetValue(0, 0) == Value::INTEGER(20));
	REQUIRE(chunk->GetValue(1, 0) == Value::INTEGER(10));
	REQUIRE(chunk->GetValue(2, 0) == Value::INTEGER(20));
}

TEST_CASE("Logical plan SQL export applies filter predicates and projection maps", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	auto child = IntegerValues(TableIndex(30), {{1, 10, 100}, {2, 20, 200}, {3, 30, 300}});
	auto filter = make_uniq<LogicalFilter>();
	filter->expressions.push_back(BoundComparisonExpression::Create(
	    ExpressionType::COMPARE_GREATERTHAN,
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(30), ProjectionIndex(0))),
	    PlanIntegerConstant(1)));
	filter->expressions.push_back(BoundComparisonExpression::Create(
	    ExpressionType::COMPARE_LESSTHAN,
	    make_uniq<BoundColumnRefExpression>(LogicalType::INTEGER, ColumnBinding(TableIndex(30), ProjectionIndex(1))),
	    PlanIntegerConstant(30)));
	filter->projection_map = {ProjectionIndex(2), ProjectionIndex(1)};
	filter->children.push_back(std::move(child));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *filter);
	REQUIRE(result.IsSuccess());
	REQUIRE(result.GetValue().fields.size() == 2);
	REQUIRE(result.GetValue().fields[0].source_binding == ColumnBinding(TableIndex(30), ProjectionIndex(2)));
	REQUIRE(result.GetValue().fields[1].source_binding == ColumnBinding(TableIndex(30), ProjectionIndex(1)));
	auto query_result = connection.Query(result.GetValue().query->ToString());
	REQUIRE_FALSE(query_result->HasError());
	auto chunk = query_result->Fetch();
	REQUIRE(chunk);
	REQUIRE(chunk->size() == 1);
	REQUIRE(chunk->GetValue(0, 0) == Value::INTEGER(200));
	REQUIRE(chunk->GetValue(1, 0) == Value::INTEGER(20));
}

TEST_CASE("Logical plan SQL export accepts both ungrouped aggregate encodings", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	connection.BeginTransaction();

	SECTION("empty grouping set list") {
		auto plan = OptimizeLogicalPlanExportQuery(connection, "SELECT sum(i) FROM (VALUES (1), (2)) t(i)");
		auto aggregate = FindLogicalPlanExportOperator(*plan, LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
		REQUIRE(aggregate);
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *aggregate);
		REQUIRE(result.IsSuccess());
		REQUIRE(connection.Query(result.GetValue().query->ToString())->GetValue(0, 0) == Value::HUGEINT(3));
	}

	SECTION("one empty grouping set") {
		auto plan = OptimizeLogicalPlanExportQuery(connection, "SELECT sum(i) FROM (VALUES (1), (2)) t(i)");
		auto aggregate = FindLogicalPlanExportOperator(*plan, LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
		REQUIRE(aggregate);
		auto &logical_aggregate = aggregate->Cast<LogicalAggregate>();
		logical_aggregate.grouping_sets.push_back(GroupingSet());
		auto result = LogicalPlanSQLExporter::Export(*connection.context, logical_aggregate);
		REQUIRE(result.IsSuccess());
		REQUIRE(connection.Query(result.GetValue().query->ToString())->GetValue(0, 0) == Value::HUGEINT(3));
	}
	connection.Rollback();
}

TEST_CASE("Logical plan SQL export rejects malformed supported operators atomically", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);

	SECTION("empty values") {
		vector<vector<unique_ptr<Expression>>> rows;
		auto plan = make_uniq<LogicalExpressionGet>(TableIndex(40), vector<LogicalType> {}, std::move(rows));
		plan->children.push_back(make_uniq<LogicalDummyScan>(TableIndex(1040)));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "empty_values");
	}

	SECTION("ragged values") {
		auto plan = IntegerValues(TableIndex(41), {{1, 2}, {3, 4}});
		plan->expressions[1].pop_back();
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "values_row_arity");
	}

	SECTION("values type arity") {
		auto plan = IntegerValues(TableIndex(42), {{1, 2}});
		plan->expr_types.pop_back();
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "values_type_arity");
	}

	SECTION("values cell type") {
		auto plan = IntegerValues(TableIndex(43), {{1}});
		plan->expressions[0][0] = make_uniq<BoundConstantExpression>(Value("wrong"));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "values_cell_type");
	}

	SECTION("non-representable values type") {
		auto plan = IntegerValues(TableIndex(44), {{1}});
		plan->expr_types[0] = LogicalType::ANY;
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "invalid_output_type");
	}

	SECTION("empty filter") {
		auto plan = make_uniq<LogicalFilter>();
		plan->children.push_back(IntegerValues(TableIndex(45), {{1}}));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "empty_filter");
	}

	SECTION("non-Boolean filter") {
		auto plan = make_uniq<LogicalFilter>();
		plan->expressions.push_back(PlanIntegerConstant(1));
		plan->children.push_back(IntegerValues(TableIndex(46), {{1}}));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "filter_predicate_type");
	}

	SECTION("duplicate filter projection binding") {
		auto plan = make_uniq<LogicalFilter>(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		plan->projection_map = {ProjectionIndex(0), ProjectionIndex(0)};
		plan->children.push_back(IntegerValues(TableIndex(47), {{1, 2}}));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "duplicate_output_binding");
	}

	SECTION("empty projection") {
		vector<unique_ptr<Expression>> expressions;
		auto plan = PlanProjection(TableIndex(48), IntegerValues(TableIndex(49), {{1}}), std::move(expressions));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "empty_projection");
	}

	SECTION("empty aggregate") {
		vector<unique_ptr<Expression>> expressions;
		auto plan = make_uniq<LogicalAggregate>(TableIndex(50), TableIndex(51), std::move(expressions));
		plan->children.push_back(IntegerValues(TableIndex(52), {{1}}));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "empty_aggregate");
	}

	SECTION("non-aggregate expression") {
		vector<unique_ptr<Expression>> expressions;
		expressions.push_back(PlanIntegerConstant(1));
		auto plan = make_uniq<LogicalAggregate>(TableIndex(53), TableIndex(54), std::move(expressions));
		plan->children.push_back(IntegerValues(TableIndex(55), {{1}}));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "aggregate_expression_class");
	}
}

TEST_CASE("Logical plan SQL export closes unsupported shapes and operator enums", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	connection.BeginTransaction();

	SECTION("grouped aggregate") {
		auto plan =
		    OptimizeLogicalPlanExportQuery(connection, "SELECT i, sum(i) FROM (VALUES (1), (2)) t(i) GROUP BY i");
		auto aggregate = FindLogicalPlanExportOperator(*plan, LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
		REQUIRE(aggregate);
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *aggregate);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_EXPORT_FEATURE);
		REQUIRE(result.GetIssues()[0].construct->identifier == "grouped_aggregate");
	}

	SECTION("standalone dummy scan") {
		LogicalDummyScan plan(TableIndex(60));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, plan);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_OPERATOR);
	}

	SECTION("opaque source") {
		auto plan = OptimizeLogicalPlanExportQuery(connection, "SELECT * FROM range(1)");
		auto get = FindLogicalPlanExportOperator(*plan, LogicalOperatorType::LOGICAL_GET);
		REQUIRE(get);
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *get);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_SOURCE);
		REQUIRE(result.GetIssues()[0].construct->type == LogicalPlanVerificationConstructType::SOURCE_FUNCTION);
	}

	SECTION("invalid enum") {
		SyntheticLogicalOperator plan(LogicalOperatorType::LOGICAL_INVALID);
		auto result = LogicalPlanSQLExporter::Export(*connection.context, plan);
		RequireInvariant(result, "invalid_operator_type");
	}

	SECTION("unknown future enum") {
		SyntheticLogicalOperator plan(static_cast<LogicalOperatorType>(254));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, plan);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_OPERATOR);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "unknown_operator_type") == Value::BOOLEAN(true));
	}
	connection.Rollback();
}

TEST_CASE("Logical plan SQL export returns verifier failures before invoking resolvers", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	idx_t calls = 0;
	auto options = PlanResolverOptions("must_not_run", [&](const LogicalPlanSQLExportExtensionInput &input) {
		calls++;
		return LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 42));
	});
	vector<unique_ptr<Expression>> expressions;
	expressions.push_back(nullptr);
	auto binding = ColumnBinding(TableIndex(70), ProjectionIndex(0));
	auto plan = make_uniq<SQLExportExtensionOperator>("bypassed_extension", vector<ColumnBinding> {binding},
	                                                  vector<LogicalType> {LogicalType::INTEGER},
	                                                  vector<TableIndex> {TableIndex(70)}, std::move(expressions));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
	RequireLogicalPlanExportIssue(
	    result, LogicalPlanVerificationIssueCode::INTERNAL_INVARIANT, LogicalPlanVerificationPhase::VERIFY,
	    LogicalPlanVerificationPath {LogicalPlanVerificationPathRoot::LOGICAL_PLAN,
	                                 {{LogicalPlanVerificationPathComponentType::OPERATOR_EXPRESSION, 0}}});
	REQUIRE(calls == 0);
}

TEST_CASE("Logical plan SQL export supports ordered invocation-local extension resolvers",
          "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);

	SECTION("ordered decline then child-consuming export") {
		vector<string> calls;
		LogicalPlanSQLExportOptions options;
		options.extension_resolvers.push_back({"first", [&](const LogicalPlanSQLExportExtensionInput &) {
			                                       calls.push_back("first");
			                                       return LogicalPlanSQLExportExtensionResult::NotHandled();
		                                       }});
		options.extension_resolvers.push_back({"second", [&](const LogicalPlanSQLExportExtensionInput &input) {
			                                       calls.push_back("second");
			                                       return LogicalPlanSQLExportExtensionResult::Exported(
			                                           PlanChildRelation(input));
		                                       }});
		auto plan = ChildExtension("child_extension", TableIndex(80));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		REQUIRE(result.IsSuccess());
		REQUIRE(calls == vector<string> {"first", "second"});
		REQUIRE(connection.Query(result.GetValue().query->ToString())->GetValue(0, 0) == Value::INTEGER(42));
	}

	SECTION("terminal unsupported") {
		idx_t later_calls = 0;
		LogicalPlanSQLExportOptions options;
		options.extension_resolvers.push_back({"rejector", [&](const LogicalPlanSQLExportExtensionInput &) {
			                                       return LogicalPlanSQLExportExtensionResult::Unsupported(
			                                           "synthetic capability is unavailable");
		                                       }});
		options.extension_resolvers.push_back({"later", [&](const LogicalPlanSQLExportExtensionInput &) {
			                                       later_calls++;
			                                       return LogicalPlanSQLExportExtensionResult::NotHandled();
		                                       }});
		auto plan = LeafExtension("terminal_extension", TableIndex(81));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_EXTENSION);
		REQUIRE(result.GetIssues()[0].message == "synthetic capability is unavailable");
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "handler_identifier") == Value("rejector"));
		REQUIRE(later_calls == 0);
	}

	SECTION("no handler") {
		auto plan = LeafExtension("unhandled_extension", TableIndex(82));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_EXTENSION);
	}

	SECTION("child failure bypasses resolver") {
		idx_t calls = 0;
		auto options = PlanResolverOptions("must_not_run", [&](const LogicalPlanSQLExportExtensionInput &) {
			calls++;
			return LogicalPlanSQLExportExtensionResult::NotHandled();
		});
		auto plan = ChildExtension("child_failure", TableIndex(83));
		plan->children[0]->Cast<LogicalExpressionGet>().expressions[0][0] =
		    make_uniq<BoundDefaultExpression>(LogicalType::INTEGER);
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequireLogicalPlanExportIssue(
		    result, LogicalPlanVerificationIssueCode::UNSUPPORTED_EXPRESSION,
		    LogicalPlanVerificationPhase::EXPRESSION_EXPORT,
		    LogicalPlanVerificationPath {LogicalPlanVerificationPathRoot::LOGICAL_PLAN,
		                                 {{LogicalPlanVerificationPathComponentType::OPERATOR_CHILD, 0},
		                                  {LogicalPlanVerificationPathComponentType::OPERATOR_EXPRESSION, 0}}});
		REQUIRE(calls == 0);
	}
}

TEST_CASE("Logical plan SQL export gives invocation resolvers precedence over registration",
          "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	auto registered = make_shared_ptr<SQLExportOperatorExtension>("precedence_extension");
	idx_t registered_calls = 0;
	registered->SetSQLExportCallback([&](const LogicalPlanSQLExportExtensionInput &input) {
		registered_calls++;
		return LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 99));
	});
	OperatorExtension::Register(DBConfig::GetConfig(*db.instance), registered);
	Connection connection(db);
	auto options = PlanResolverOptions("local", [&](const LogicalPlanSQLExportExtensionInput &input) {
		return LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 42));
	});
	auto plan = LeafExtension("precedence_extension", TableIndex(90));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
	REQUIRE(result.IsSuccess());
	REQUIRE(registered_calls == 0);
	REQUIRE(connection.Query(result.GetValue().query->ToString())->GetValue(0, 0) == Value::INTEGER(42));
}

TEST_CASE("Logical plan SQL export uses the matching registered extension callback", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	auto registered = make_shared_ptr<SQLExportOperatorExtension>("registered_extension");
	registered->SetSQLExportCallback([&](const LogicalPlanSQLExportExtensionInput &input) {
		return LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 84));
	});
	OperatorExtension::Register(DBConfig::GetConfig(*db.instance), registered);
	Connection connection(db);
	auto plan = LeafExtension("registered_extension", TableIndex(91));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
	REQUIRE(result.IsSuccess());
	REQUIRE(connection.Query(result.GetValue().query->ToString())->GetValue(0, 0) == Value::INTEGER(84));

	LegacyOperatorExtension legacy;
	REQUIRE_FALSE(legacy.HasSQLExportCallback());
	OperatorExtension::Register(DBConfig::GetConfig(*db.instance), make_shared_ptr<LegacyOperatorExtension>());
	auto legacy_plan = LeafExtension("legacy_sql_export_test", TableIndex(92));
	auto legacy_result = LogicalPlanSQLExporter::Export(*connection.context, *legacy_plan);
	RequirePlanExportIssue(legacy_result, LogicalPlanVerificationIssueCode::UNSUPPORTED_EXTENSION);
}

TEST_CASE("Logical plan SQL export rejects malformed resolver and extension result states",
          "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);

	SECTION("invalid resolver name") {
		auto options = PlanResolverOptions("", [](const LogicalPlanSQLExportExtensionInput &) {
			return LogicalPlanSQLExportExtensionResult::NotHandled();
		});
		auto plan = LeafExtension("malformed_options", TableIndex(100));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
	}

	SECTION("duplicate resolver name") {
		auto options = PlanResolverOptions("duplicate", [](const LogicalPlanSQLExportExtensionInput &) {
			return LogicalPlanSQLExportExtensionResult::NotHandled();
		});
		options.extension_resolvers.push_back({"duplicate", [](const LogicalPlanSQLExportExtensionInput &) {
			                                       return LogicalPlanSQLExportExtensionResult::NotHandled();
		                                       }});
		auto plan = LeafExtension("malformed_options", TableIndex(101));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
	}

	SECTION("null resolver callback") {
		LogicalPlanSQLExportOptions options;
		options.extension_resolvers.push_back({"null_callback", {}});
		auto plan = LeafExtension("malformed_options", TableIndex(106));
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
	}

	SECTION("invalid extension name") {
		auto binding = ColumnBinding(TableIndex(107), ProjectionIndex(0));
		auto plan = make_uniq<SQLExportExtensionOperator>(
		    string("invalid\0name", 12), vector<ColumnBinding> {binding}, vector<LogicalType> {LogicalType::INTEGER},
		    vector<TableIndex> {TableIndex(107)}, vector<unique_ptr<Expression>> {}, "valid_verification_name");
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan);
		RequireInvariant(result, "extension_identifier");
	}

	SECTION("NOT_HANDLED payload") {
		auto plan = LeafExtension("malformed_state", TableIndex(102));
		auto options = PlanResolverOptions("malformed", [](const LogicalPlanSQLExportExtensionInput &) {
			auto result = LogicalPlanSQLExportExtensionResult::NotHandled();
			result.reason = "unexpected";
			return result;
		});
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("not_handled_payload"));
	}

	SECTION("EXPORTED without relation") {
		auto plan = LeafExtension("malformed_state", TableIndex(103));
		auto options = PlanResolverOptions("malformed", [](const LogicalPlanSQLExportExtensionInput &) {
			LogicalPlanSQLExportExtensionResult result;
			result.type = LogicalPlanSQLExportExtensionResultType::EXPORTED;
			return result;
		});
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("exported_payload"));
	}

	SECTION("UNSUPPORTED without reason") {
		auto plan = LeafExtension("malformed_state", TableIndex(104));
		auto options = PlanResolverOptions("malformed", [](const LogicalPlanSQLExportExtensionInput &) {
			return LogicalPlanSQLExportExtensionResult::Unsupported("");
		});
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("unsupported_payload"));
	}

	SECTION("unknown result type") {
		auto plan = LeafExtension("malformed_state", TableIndex(105));
		auto options = PlanResolverOptions("malformed", [](const LogicalPlanSQLExportExtensionInput &) {
			LogicalPlanSQLExportExtensionResult result;
			result.type = static_cast<LogicalPlanSQLExportExtensionResultType>(42);
			return result;
		});
		auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("result_type"));
	}
}

TEST_CASE("Logical plan SQL export validates extension result schemas centrally", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);

	SECTION("null query") {
		auto result = ExportMalformedRelation(
		    connection, [](LogicalPlanSQLExportExtensionResult &result) { result.relation->query.reset(); });
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("null_query"));
	}

	SECTION("field count") {
		auto result = ExportMalformedRelation(
		    connection, [](LogicalPlanSQLExportExtensionResult &result) { result.relation->fields.clear(); });
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("field_count"));
	}

	SECTION("field ordinal") {
		auto result = ExportMalformedRelation(
		    connection, [](LogicalPlanSQLExportExtensionResult &result) { result.relation->fields[0].ordinal = 1; });
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("field_ordinal"));
	}

	SECTION("field binding") {
		auto result = ExportMalformedRelation(connection, [](LogicalPlanSQLExportExtensionResult &result) {
			result.relation->fields[0].source_binding = ColumnBinding(TableIndex(999), ProjectionIndex(0));
		});
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("field_binding"));
	}

	SECTION("field type") {
		auto result = ExportMalformedRelation(connection, [](LogicalPlanSQLExportExtensionResult &result) {
			result.relation->fields[0].type = LogicalType::BIGINT;
		});
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("field_type"));
	}

	SECTION("field identifier") {
		auto result = ExportMalformedRelation(connection, [](LogicalPlanSQLExportExtensionResult &result) {
			result.relation->fields[0].identifier = Identifier("not_c0");
		});
		RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT);
		REQUIRE(GetLogicalPlanExportFact(result.GetIssues()[0], "invariant") == Value("field_identifier"));
	}
}

TEST_CASE("Logical plan SQL export does not descend through unsupported parents", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);
	idx_t calls = 0;
	auto options = PlanResolverOptions("must_not_run", [&](const LogicalPlanSQLExportExtensionInput &input) {
		calls++;
		return LogicalPlanSQLExportExtensionResult::Exported(PlanConstantRelation(input, 1));
	});
	auto plan = make_uniq<LogicalCrossProduct>(LeafExtension("left_extension", TableIndex(110)),
	                                           LeafExtension("right_extension", TableIndex(111)));

	auto result = LogicalPlanSQLExporter::Export(*connection.context, *plan, options);
	RequirePlanExportIssue(result, LogicalPlanVerificationIssueCode::UNSUPPORTED_OPERATOR);
	REQUIRE(calls == 0);
}

TEST_CASE("Logical plan SQL export propagates callback exceptions unchanged", "[logical_plan_sql_export]") {
	DuckDB db(nullptr);
	Connection connection(db);

	SECTION("runtime exception") {
		auto plan = LeafExtension("throwing_extension", TableIndex(120));
		auto options = PlanResolverOptions(
		    "thrower", [](const LogicalPlanSQLExportExtensionInput &) -> LogicalPlanSQLExportExtensionResult {
			    throw std::runtime_error("synthetic callback failure");
		    });
		REQUIRE_THROWS_AS(LogicalPlanSQLExporter::Export(*connection.context, *plan, options), std::runtime_error);
	}

	SECTION("allocation exception") {
		auto plan = LeafExtension("throwing_extension", TableIndex(121));
		auto options = PlanResolverOptions(
		    "thrower", [](const LogicalPlanSQLExportExtensionInput &) -> LogicalPlanSQLExportExtensionResult {
			    throw std::bad_alloc();
		    });
		REQUIRE_THROWS_AS(LogicalPlanSQLExporter::Export(*connection.context, *plan, options), std::bad_alloc);
	}
}

TEST_CASE("Logical plan SQL export results outlive plans, contexts, and callbacks", "[logical_plan_sql_export]") {
	optional<LogicalPlanSQLExportRelation> owned_relation;
	string exported_sql;
	{
		DuckDB source_db(nullptr);
		Connection source_connection(source_db);
		auto plan = ChildExtension("owned_extension", TableIndex(130));
		auto options = PlanResolverOptions("owner", [](const LogicalPlanSQLExportExtensionInput &input) {
			return LogicalPlanSQLExportExtensionResult::Exported(PlanChildRelation(input));
		});
		auto result = LogicalPlanSQLExporter::Export(*source_connection.context, *plan, options);
		REQUIRE(result.IsSuccess());
		owned_relation = std::move(result.GetValue());
		exported_sql = owned_relation->query->ToString();
	}

	REQUIRE(owned_relation);
	REQUIRE(owned_relation->query->ToString() == exported_sql);
	DuckDB target_db(nullptr);
	Connection target_connection(target_db);
	auto result = target_connection.Query(exported_sql);
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->GetValue(0, 0) == Value::INTEGER(42));
}
