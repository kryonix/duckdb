#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/procedure_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/function/language_macro_function.hpp"
#include "duckdb/function/language_procedure_function.hpp"
#include "duckdb/function/language_table_macro_function.hpp"
#include "duckdb/main/language_function_binder.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/expression/cast_expression.hpp"
#include "duckdb/common/extension_type_info.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/parser/parsed_data/extra_drop_info.hpp"
#include "duckdb/parser/expression/type_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/expression/type_expression.hpp"
#include "duckdb/parser/parsed_data/create_language_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "test_helpers.hpp"

using namespace duckdb;

// These tests exercise the C++ extension contract around ValidateCreate and internal entries. SQL
// cannot drive the create view handed to a language binder, register a language binder, or
// create an internal macro entry, so the setup must be C++. The SQL-observable behavior of valid
// definitions stays in test/sql/function/language_function.

namespace {

//! A binder whose ValidateCreate performs a configurable action through the narrow create view
class ViewBinder : public LanguageFunctionBinder {
public:
	explicit ViewBinder(std::function<void(LanguageFunctionCreateInfo &)> action_p) : action(std::move(action_p)) {
	}

	void ValidateCreate(ClientContext &context, LanguageFunctionCreateInfo &info) override {
		if (action) {
			action(info);
		}
	}
	BindResult BindScalarCall(ExpressionBinder &binder, ScalarMacroCatalogEntry &entry, LanguageMacroFunction &member,
	                          vector<unique_ptr<ParsedExpression>> arguments, idx_t depth) override {
		// the compiled form is the constant the source denotes, like the testlang demo
		return BindResult(make_uniq<BoundConstantExpression>(Value(member.source)));
	}
	void ExpandCall(ClientContext &context, ProcedureCatalogEntry &entry, LanguageProcedureFunction &member,
	                vector<unique_ptr<ParsedExpression>> arguments,
	                vector<unique_ptr<SQLStatement>> &statements) override {
		// the expansion is the member source parsed verbatim, like the testlang demo
		Parser parser(context);
		parser.ParseQuery(member.source);
		for (auto &statement : parser.statements) {
			statements.push_back(std::move(statement));
		}
	}

private:
	std::function<void(LanguageFunctionCreateInfo &)> action;
};

//! Register a language the way ExtensionLoader::RegisterLanguage does, without an extension
static void RegisterTestLanguage(DatabaseInstance &db, const string &language,
                                 shared_ptr<LanguageFunctionBinder> binder) {
	CreateLanguageInfo info((Identifier(language)));
	info.binder = std::move(binder);
	info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	auto &schema = system_catalog.GetSchema(data, Identifier::DefaultSchema()).Cast<DuckSchemaEntry>();
	schema.CreateLanguage(data, info);
}

} // namespace

TEST_CASE("ValidateCreate rewrites a member source through the create view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "rewritelang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.SetMemberSource(0, "rewritten");
	}));

	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION rewritten_fun() LANGUAGE rewritelang AS 'original'"));
	// the rewritten source is what got stored and what the call compiles to
	auto result = con.Query("SELECT rewritten_fun()");
	REQUIRE(CHECK_COLUMN(result, 0, {"rewritten"}));
	result = con.Query("SELECT macro_definition FROM duckdb_functions() WHERE function_name = 'rewritten_fun'");
	REQUIRE(CHECK_COLUMN(result, 0, {"rewritten"}));
}

TEST_CASE("ValidateCreate reads members and rejects definitions through the create view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "readlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		REQUIRE(info.MemberCount() == 1);
		if (StringUtil::Contains(info.MemberSource(0), "reject")) {
			throw BinderException("readlang: rejected %s", info.GetFunctionName().GetIdentifierName());
		}
	}));

	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION accepted_fun(x INTEGER) LANGUAGE readlang AS 'fine'"));
	auto result = con.Query("CREATE FUNCTION rejected_fun() LANGUAGE readlang AS 'reject me'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "readlang: rejected rejected_fun"));
}

TEST_CASE("ValidateCreate publishes documentation through the create view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "doclang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		for (idx_t member_idx = 0; member_idx < info.MemberCount(); member_idx++) {
			FunctionDescription description;
			for (idx_t i = 0; i < info.MemberParameterCount(member_idx); i++) {
				description.parameter_names.push_back(info.MemberParameterName(member_idx, i));
				const auto &type = info.MemberParameterType(member_idx, i);
				description.parameter_types.push_back(type.id() == LogicalTypeId::UNKNOWN ? LogicalType::ANY : type);
			}
			description.description = "doclang doc";
			info.AddDescription(std::move(description));
		}
	}));

	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION documented_fun(x INTEGER) LANGUAGE doclang AS 'd'"));
	auto result = con.Query("SELECT description FROM duckdb_functions() WHERE function_name = 'documented_fun'");
	REQUIRE(CHECK_COLUMN(result, 0, {"doclang doc"}));
}

TEST_CASE("ValidateCreate publishes a schema and language version through the create view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "publang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		if (info.MemberIsTableFunction(0) && info.MemberReturnColumnCount(0) == 0) {
			info.PublishMemberSchema(0, {Identifier("published")}, {LogicalType::INTEGER});
		}
		if (info.MemberLanguageVersion(0).empty()) {
			info.PublishMemberLanguageVersion(0, "publang 7");
		}
	}));

	// the bare RETURNS TABLE form stores an unresolved schema the binder may publish once
	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION published_tf() RETURNS TABLE LANGUAGE publang AS 'x'"));
	auto result =
	    con.Query("SELECT return_type, language_version FROM duckdb_functions() WHERE function_name = 'published_tf'");
	REQUIRE(CHECK_COLUMN(result, 0, {"TABLE(published INTEGER)"}));
	REQUIRE(CHECK_COLUMN(result, 1, {"publang 7"}));

	// a declared schema reads back through the view
	RegisterTestLanguage(*db.instance, "readcols", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		REQUIRE(info.MemberIsTableFunction(0));
		REQUIRE(info.MemberReturnColumnCount(0) == 2);
		REQUIRE(info.MemberReturnColumnName(0, 1) == Identifier("b"));
		REQUIRE(info.MemberReturnColumnType(0, 1) == LogicalType::VARCHAR);
	}));
	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION declared_tf() RETURNS TABLE(a INTEGER, b VARCHAR) "
	                          "LANGUAGE readcols AS 'x'"));
}

TEST_CASE("A prepared CALL re-expands at every execution", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "preplang", make_shared_ptr<ViewBinder>(nullptr));

	REQUIRE_NO_FAIL(con.Query("CREATE PROCEDURE prep_p() RETURNS TABLE(n BIGINT) LANGUAGE preplang AS "
	                          "'SELECT 1::BIGINT AS n'"));
	// the client Prepare API keeps the written CALL and re-expands it at each execution, so a
	// member replacement between executions is visible (SQL-level PREPARE rejects CALL at parse)
	auto prepared = con.Prepare("CALL prep_p()");
	REQUIRE_FALSE(prepared->HasError());
	auto result = prepared->Execute();
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
	REQUIRE_NO_FAIL(con.Query("CREATE OR REPLACE PROCEDURE prep_p() RETURNS TABLE(n BIGINT) LANGUAGE preplang AS "
	                          "'SELECT 2::BIGINT AS n'"));
	result = prepared->Execute();
	REQUIRE(CHECK_COLUMN(result, 0, {2}));

	// a parameterized prepared CALL is rejected: parameters cannot thread through an expansion
	auto parameterized = con.Prepare("CALL prep_p($1)");
	REQUIRE(parameterized->HasError());
	REQUIRE(StringUtil::Contains(parameterized->GetError(), "parameters"));
}

// the publish-once and identity checks raise InternalException, which aborts the process when
// DUCKDB_CRASH_ON_ASSERT is set
#ifndef DUCKDB_CRASH_ON_ASSERT

TEST_CASE("Publishing over a declared or published schema is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "republ", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::INTEGER});
	}));

	// publish-once only covers an EMPTY schema: a declared RETURNS TABLE shape is out of reach
	auto result = con.Query("CREATE FUNCTION declared_repub() RETURNS TABLE(a INTEGER) LANGUAGE republ AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "tried to change the result schema"));

	// ... and a published schema is set exactly once
	RegisterTestLanguage(*db.instance, "twicelang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::INTEGER});
		info.PublishMemberSchema(0, {Identifier("y")}, {LogicalType::VARCHAR});
	}));
	result = con.Query("CREATE FUNCTION published_twice() RETURNS TABLE LANGUAGE twicelang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "tried to change the result schema"));
}

TEST_CASE("An invalid published schema is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// a published schema must be as valid as a declared one: unique names, resolved types
	RegisterTestLanguage(*db.instance, "duplang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x"), Identifier("x")}, {LogicalType::INTEGER, LogicalType::INTEGER});
	}));
	auto result = con.Query("CREATE FUNCTION dup_published() RETURNS TABLE LANGUAGE duplang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "duplicate column name"));

	RegisterTestLanguage(*db.instance, "unknownlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::UNKNOWN});
	}));
	result = con.Query("CREATE FUNCTION unknown_published() RETURNS TABLE LANGUAGE unknownlang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "unresolved column type"));

	// ... also when the unresolved type hides below a concrete container
	RegisterTestLanguage(*db.instance, "nestedunk", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::STRUCT({{"m", LogicalType::UNKNOWN}})});
	}));
	result = con.Query("CREATE FUNCTION nested_published() RETURNS TABLE LANGUAGE nestedunk AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "unresolved column type"));

	// ... and incomplete types (ANY, nested ANY, containers without parameters) are no result shape
	RegisterTestLanguage(*db.instance, "anylang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::ANY});
	}));
	result = con.Query("CREATE FUNCTION any_published() RETURNS TABLE LANGUAGE anylang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "incomplete column type"));

	RegisterTestLanguage(*db.instance, "nestedany", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::LIST(LogicalType::ANY)});
	}));
	result = con.Query("CREATE FUNCTION nested_any_published() RETURNS TABLE LANGUAGE nestedany AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "incomplete column type"));

	RegisterTestLanguage(*db.instance, "barelist", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType(LogicalTypeId::LIST)});
	}));
	result = con.Query("CREATE FUNCTION bare_list_published() RETURNS TABLE LANGUAGE barelist AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "incomplete column type"));

	// ... and name and type counts must line up
	RegisterTestLanguage(*db.instance, "countlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::INTEGER, LogicalType::INTEGER});
	}));
	result = con.Query("CREATE FUNCTION count_published() RETURNS TABLE LANGUAGE countlang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "column names and"));

	// a schema cannot be published onto a scalar member
	RegisterTestLanguage(*db.instance, "scalarpub", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberSchema(0, {Identifier("x")}, {LogicalType::INTEGER});
	}));
	result = con.Query("CREATE FUNCTION scalar_published() LANGUAGE scalarpub AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "not a table function or procedure member"));
}

TEST_CASE("A language version change is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "verlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberLanguageVersion(0, "verlang 2");
	}));

	// publish-once: a definition arriving with a version must keep it (only constructible via the API)
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("versioned_fun"));
	auto member = make_uniq<LanguageMacroFunction>("verlang", "x");
	member->language_version = "verlang 1";
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "tried to change the published language version"));
}

TEST_CASE("An in-place mutation of a result column type is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// LogicalType copies share their type info: a binder that built a column type can keep a path
	// to its extension metadata and mutate the stored schema in place, without any view call. The
	// serialized-identity freeze is what makes the detached-copy convention enforceable.
	RegisterTestLanguage(*db.instance, "mutlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto ext_info = make_uniq<ExtensionTypeInfo>();
		auto &ext_ref = *ext_info;
		auto column_type = LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info));
		info.PublishMemberSchema(0, {Identifier("x")}, {std::move(column_type)});
		// mutate the published schema through the retained metadata reference
		ext_ref.modifiers.emplace_back(Value::INTEGER(10));
	}));

	auto result = con.Query("CREATE FUNCTION mutated_published() RETURNS TABLE LANGUAGE mutlang AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a declared or published result schema"));

	// the same freeze covers a DECLARED schema, captured before the callback runs
	optional_ptr<ExtensionTypeInfo> declared_ext;
	RegisterTestLanguage(*db.instance, "mutdeclared",
	                     make_shared_ptr<ViewBinder>([&declared_ext](LanguageFunctionCreateInfo &info) {
		                     declared_ext->modifiers.emplace_back(Value::INTEGER(10));
	                     }));
	auto ext_info = make_uniq<ExtensionTypeInfo>();
	declared_ext = ext_info.get();
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->SetFunctionName(Identifier("mutated_declared"));
	auto member = make_uniq<LanguageTableMacroFunction>("mutdeclared", "x");
	member->return_names.emplace_back("c");
	member->return_types.push_back(LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info)));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a declared or published result schema"));
}

TEST_CASE("Retained references into published values are rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the publish setters adopt the caller's allocation: a retained reference into a published
	// vector or string still names the stored value after the moves, without any view call
	RegisterTestLanguage(*db.instance, "namemut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		vector<Identifier> names {Identifier("a"), Identifier("b")};
		auto &second = names[1];
		info.PublishMemberSchema(0, std::move(names), {LogicalType::INTEGER, LogicalType::INTEGER});
		second = Identifier("a");
	}));
	auto result = con.Query("CREATE FUNCTION name_mutated() RETURNS TABLE LANGUAGE namemut AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a declared or published result schema"));

	RegisterTestLanguage(*db.instance, "markermut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		vector<idx_t> markers {0};
		auto &marker = markers[0];
		info.PublishMemberRelationParameters(0, std::move(markers));
		marker = info.MemberParameterCount(0);
	}));
	result = con.Query("CREATE FUNCTION marker_mutated(r) RETURNS TABLE LANGUAGE markermut AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed the relation parameters"));

	// a case-only rename changes the stored spelling even though SQL lookup treats it as equal
	RegisterTestLanguage(*db.instance, "casemut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		vector<Identifier> names {Identifier("Result")};
		auto &name = names[0];
		info.PublishMemberSchema(0, std::move(names), {LogicalType::INTEGER});
		name = Identifier("RESULT");
	}));
	result = con.Query("CREATE FUNCTION case_mutated() RETURNS TABLE LANGUAGE casemut AS 'x'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a declared or published result schema"));

	// whether a string move transfers its heap buffer is implementation-defined, so assert the
	// invariant instead: the mutation is either rejected or it never reached the stored version
	RegisterTestLanguage(*db.instance, "versionmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		string version(64, '1');
		auto &first = version[0];
		info.PublishMemberLanguageVersion(0, std::move(version));
		first = '2';
	}));
	result = con.Query("CREATE FUNCTION version_mutated() LANGUAGE versionmut AS 'x'");
	if (result->HasError()) {
		REQUIRE(StringUtil::Contains(result->GetError(), "changed the language version"));
	} else {
		auto shown =
		    con.Query("SELECT language_version FROM duckdb_functions() WHERE function_name = 'version_mutated'");
		REQUIRE(CHECK_COLUMN(shown, 0, {Value(string(64, '1'))}));
	}
}

TEST_CASE("A declared incomplete column type is rejected at CREATE", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the SQL grammar cannot spell ANY in type position, so the state is API-only
	RegisterTestLanguage(*db.instance, "declany", make_shared_ptr<ViewBinder>(nullptr));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->SetFunctionName(Identifier("declared_any"));
	auto member = make_uniq<LanguageTableMacroFunction>("declany", "x");
	member->return_names.emplace_back("c");
	member->return_types.emplace_back(LogicalType::ANY);
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "incomplete RETURNS TABLE column type"));
}

TEST_CASE("A resolved table member signature mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// a returned parameter type shares its interned metadata, and extension_info stays a mutable
	// pointee behind the const LogicalTypeInfo: without the signature freeze this collapses two
	// distinct overload signatures after the batch conflict check already accepted them
	RegisterTestLanguage(*db.instance, "sigmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 0);
		parameter_type.GetTypeInfo().extension_info->modifiers.emplace_back(Value::INTEGER(9));
	}));

	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value::INTEGER(1));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->SetFunctionName(Identifier("sig_mutated"));
	auto member = make_uniq<LanguageTableMacroFunction>("sigmut", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
	member->types.push_back(LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info)));
	member->return_names.emplace_back("i");
	member->return_types.emplace_back(LogicalType::INTEGER);
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

#endif

TEST_CASE("ValidateCreate reads the destination, statement text, and defaults through the view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	auto observed = make_shared_ptr<vector<string>>();
	RegisterTestLanguage(*db.instance, "ctxlang",
	                     make_shared_ptr<ViewBinder>([observed](LanguageFunctionCreateInfo &info) {
		                     // the destination is resolved before validation, so the language sees the real target
		                     auto &destination = info.GetQualifiedFunctionName();
		                     observed->push_back(destination.Catalog().GetIdentifierName() + "." +
		                                         destination.Schema().GetIdentifierName() + "." +
		                                         destination.Name().GetIdentifierName());
		                     observed->push_back(info.GetStatementText());
		                     auto default_value = info.MemberParameterDefault(0, 1);
		                     observed->push_back(default_value ? default_value->ToString() : "<none>");
		                     REQUIRE(!info.MemberParameterDefault(0, 0));
	                     }));

	const string create_text = "CREATE FUNCTION ctx_fun(x INTEGER, y INTEGER := 42) LANGUAGE ctxlang AS 'src'";
	REQUIRE_NO_FAIL(con.Query(create_text));
	REQUIRE(observed->size() == 3);
	REQUIRE((*observed)[0] == "memory.main.ctx_fun");
	REQUIRE((*observed)[1] == create_text);
	REQUIRE((*observed)[2] == "42");
}

TEST_CASE("ValidateCreate publishes relation parameters through the view", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "rellang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		if (info.MemberRelationParameters(0).empty()) {
			info.PublishMemberRelationParameters(0, {0});
		}
		REQUIRE(info.MemberRelationParameters(0) == vector<idx_t> {0});
	}));

	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION rel_tf(r) RETURNS TABLE(i INTEGER) LANGUAGE rellang AS 'rows:1'"));
	// the published markers are stored on the member
	con.BeginTransaction();
	auto &entry = Catalog::GetEntry<TableMacroCatalogEntry>(*con.context, QualifiedName("memory", "main", "rel_tf"));
	REQUIRE(entry.macros[0]->relation_parameters == vector<idx_t> {0});
	con.Rollback();

	// a language that registers no table support rejects table calls through the default seam
	auto result = con.Query("SELECT * FROM rel_tf(1)");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "does not implement table function binding"));
}

TEST_CASE("Published relation markers persist through a reopen", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the markers are not SQL-visible, so the serialized field 105 round trip needs C++ inspection
	RegisterTestLanguage(*db.instance, "relpersist", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		if (info.MemberRelationParameters(0).empty()) {
			info.PublishMemberRelationParameters(0, {0});
		}
	}));
	auto path = TestCreatePath("relation_marker_persistence.db");
	DeleteDatabase(path);
	REQUIRE_NO_FAIL(con.Query("ATTACH '" + path + "' AS reldb (STORAGE_VERSION 'v2.1.0')"));
	REQUIRE_NO_FAIL(con.Query(
	    "CREATE FUNCTION reldb.main.rel_persist_tf(r) RETURNS TABLE(i INTEGER) LANGUAGE relpersist AS 'rows:1'"));
	REQUIRE_NO_FAIL(con.Query("CHECKPOINT reldb"));
	REQUIRE_NO_FAIL(con.Query("DETACH reldb"));
	REQUIRE_NO_FAIL(con.Query("ATTACH '" + path + "' AS reldb"));
	con.BeginTransaction();
	auto &entry =
	    Catalog::GetEntry<TableMacroCatalogEntry>(*con.context, QualifiedName("reldb", "main", "rel_persist_tf"));
	REQUIRE(entry.macros[0]->relation_parameters == vector<idx_t> {0});
	con.Rollback();
}

TEST_CASE("Direct catalog overload adds reject kind mismatches", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "kindlang", make_shared_ptr<ViewBinder>(nullptr));

	// reach the schema entry directly, the way an extension registering functions does
	auto &system_catalog = Catalog::GetSystemCatalog(*db.instance);
	auto data = CatalogTransaction::GetSystemTransaction(*db.instance);
	auto &schema = system_catalog.GetSchema(data, Identifier::DefaultSchema());

	CreateMacroInfo info(CatalogType::TABLE_MACRO_ENTRY);
	info.SetFunctionName(Identifier("kind_guard"));
	info.internal = true;
	auto table_member = make_uniq<LanguageTableMacroFunction>("kindlang", "x");
	table_member->return_names.emplace_back("i");
	table_member->return_types.emplace_back(LogicalType::INTEGER);
	info.macros.push_back(std::move(table_member));
	schema.CreateFunction(data, info);

	// a scalar member aimed at the table catalog set must not reach the overload-add path
	CreateMacroInfo mismatched(CatalogType::TABLE_MACRO_ENTRY);
	mismatched.SetFunctionName(Identifier("kind_guard"));
	mismatched.internal = true;
	mismatched.macros.push_back(make_uniq<LanguageMacroFunction>("kindlang", "y"));
	REQUIRE_THROWS_WITH(schema.CreateFunction(data, mismatched),
	                    Catch::Contains("cannot combine language function members of different kinds"));
}

#ifndef DUCKDB_CRASH_ON_ASSERT

TEST_CASE("Invalid relation-parameter publications are rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// out-of-range and non-increasing indexes are rejected
	RegisterTestLanguage(*db.instance, "relbad", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberRelationParameters(0, {1});
	}));
	auto result = con.Query("CREATE FUNCTION rel_bad(r) RETURNS TABLE(i INTEGER) LANGUAGE relbad AS 'rows:1'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "not strictly increasing declared parameter indexes"));

	// publish-once: markers are set exactly once
	RegisterTestLanguage(*db.instance, "reltwice", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		info.PublishMemberRelationParameters(0, {0});
		info.PublishMemberRelationParameters(0, {0});
	}));
	result = con.Query("CREATE FUNCTION rel_twice(r) RETURNS TABLE(i INTEGER) LANGUAGE reltwice AS 'rows:1'");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "tried to change the relation parameters"));
}

#endif

TEST_CASE("Member mutations respect internal entry protection", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "intlang", make_shared_ptr<ViewBinder>(nullptr));

	// register an internal two-member language entry in the system catalog, the way an extension
	// registering functions does
	CreateMacroInfo info(CatalogType::MACRO_ENTRY);
	info.SetFunctionName(Identifier("internal_fun"));
	info.internal = true;
	auto int_member = make_uniq<LanguageMacroFunction>("intlang", "int member");
	int_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	int_member->types.push_back(LogicalType::INTEGER);
	info.macros.push_back(std::move(int_member));
	auto varchar_member = make_uniq<LanguageMacroFunction>("intlang", "varchar member");
	varchar_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	varchar_member->types.push_back(LogicalType::VARCHAR);
	info.macros.push_back(std::move(varchar_member));
	auto &system_catalog = Catalog::GetSystemCatalog(*db.instance);
	auto data = CatalogTransaction::GetSystemTransaction(*db.instance);
	auto &schema = system_catalog.GetSchema(data, Identifier::DefaultSchema());
	schema.CreateFunction(data, info);

	// member-granular drop from SQL is blocked by the generic internal-entry drop protection
	auto result = con.Query("DROP FUNCTION internal_fun(INTEGER)");
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "internal"));

	// a CREATE from SQL resolves into the user catalog and creates a shadowing entry there, it
	// never merges into the system entry
	REQUIRE_NO_FAIL(con.Query("CREATE FUNCTION internal_fun(x DATE) LANGUAGE intlang AS 'date member'"));
	result = con.Query("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'internal_fun' AND "
	                   "database_name = 'memory'");
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
	REQUIRE_NO_FAIL(con.Query("DROP FUNCTION internal_fun"));

	// reaching the schema entry directly with a non-internal overload add is rejected: the member
	// rewrite is a drop+create and the internal protection applies to it
	CreateMacroInfo add_info(CatalogType::MACRO_ENTRY);
	add_info.SetFunctionName(Identifier("internal_fun"));
	auto date_member = make_uniq<LanguageMacroFunction>("intlang", "date member");
	date_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	date_member->types.push_back(LogicalType::DATE);
	add_info.macros.push_back(std::move(date_member));
	auto add_data = CatalogTransaction::GetSystemTransaction(*db.instance);
	REQUIRE_THROWS(schema.CreateFunction(add_data, add_info));

	// the entry is untouched and still callable
	result = con.Query("SELECT internal_fun(1), internal_fun('a'::VARCHAR)");
	REQUIRE(CHECK_COLUMN(result, 0, {"int member"}));
	REQUIRE(CHECK_COLUMN(result, 1, {"varchar member"}));

	// the drop guard directly: a member drop without authority is rejected ...
	DropInfo drop_info;
	drop_info.type = CatalogType::MACRO_ENTRY;
	drop_info.SetQualifiedName(QualifiedName(Identifier(), Identifier::DefaultSchema(), Identifier("internal_fun")));
	auto extra_info = make_uniq<ExtraDropFunctionSignatureInfo>();
	extra_info->signature.push_back(LogicalType::INTEGER);
	drop_info.extra_drop_info = std::move(extra_info);
	con.BeginTransaction();
	bool threw = false;
	try {
		schema.DropEntry(*con.context, drop_info);
	} catch (std::exception &ex) {
		threw = StringUtil::Contains(ex.what(), "internal system entry");
	}
	REQUIRE(threw);
	con.Rollback();

	// an extension registering internal members one at a time still merges: the internal info
	// carries the authority the guard checks
	CreateMacroInfo readd_info(CatalogType::MACRO_ENTRY);
	readd_info.SetFunctionName(Identifier("internal_fun"));
	readd_info.internal = true;
	auto readd_member = make_uniq<LanguageMacroFunction>("intlang", "date member");
	readd_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	readd_member->types.push_back(LogicalType::DATE);
	readd_info.macros.push_back(std::move(readd_member));
	auto readd_data = CatalogTransaction::GetSystemTransaction(*db.instance);
	schema.CreateFunction(readd_data, readd_info);
	result = con.Query("SELECT internal_fun(DATE '2020-01-01'), internal_fun('a'::VARCHAR)");
	REQUIRE(CHECK_COLUMN(result, 0, {"date member"}));
	REQUIRE(CHECK_COLUMN(result, 1, {"varchar member"}));
}

static LogicalType UnresolvedUserType(const string &name) {
	vector<unique_ptr<ParsedExpression>> no_children;
	return LogicalType::UNBOUND(make_uniq<TypeExpression>(Identifier(name), std::move(no_children)));
}

TEST_CASE("Nested unbound member types resolve at CREATE", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "nestlang", make_shared_ptr<ViewBinder>(nullptr));
	REQUIRE_NO_FAIL(con.Query("CREATE TYPE mood AS ENUM ('happy', 'sad')"));

	// a dialect transformer may emit a concrete container around an unresolved user type, unlike
	// the stock grammar, which wraps the whole unresolved type at the top level
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("nested_fun"));
	auto list_member = make_uniq<LanguageMacroFunction>("nestlang", "arg:0");
	list_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	list_member->types.push_back(LogicalType::LIST(UnresolvedUserType("mood")));
	list_member->return_type = LogicalType::LIST(UnresolvedUserType("mood"));
	info->macros.push_back(std::move(list_member));
	auto struct_member = make_uniq<LanguageMacroFunction>("nestlang", "struct member");
	struct_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	struct_member->types.push_back(LogicalType::STRUCT({{"m", UnresolvedUserType("mood")}}));
	info->macros.push_back(std::move(struct_member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	unique_ptr<SQLStatement> sql_statement = std::move(statement);
	auto result = con.Query(std::move(sql_statement));
	REQUIRE_NO_FAIL(*result);

	// the stored signatures are fully resolved: overload matching selects by the nested user type,
	// the RETURNS cast applies, and introspection shows resolved types
	auto call = con.Query("SELECT typeof(nested_fun(['happy'::mood])), nested_fun({'m': 'happy'::mood})");
	REQUIRE(CHECK_COLUMN(call, 0, {"ENUM('happy', 'sad')[]"}));
	REQUIRE(CHECK_COLUMN(call, 1, {"struct member"}));
	auto types = con.Query("SELECT bool_and(parameter_types[1] NOT LIKE '%mood%') "
	                       "FROM duckdb_functions() WHERE function_name = 'nested_fun'");
	REQUIRE(CHECK_COLUMN(types, 0, {true}));

	// bound sibling subtrees keep their aliases: two members whose aliased payloads differ must
	// stay distinct signatures when the unresolved sibling is bound
	auto alias_info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	alias_info->SetFunctionName(Identifier("aliased_sibling"));
	for (auto payload_alias : {"payload_a", "payload_b"}) {
		auto member = make_uniq<LanguageMacroFunction>("nestlang", payload_alias);
		member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
		child_list_t<LogicalType> fields;
		fields.emplace_back("payload", LogicalType::LIST(LogicalType::INTEGER).WithAlias(payload_alias));
		fields.emplace_back("m", UnresolvedUserType("mood"));
		member->types.push_back(LogicalType::STRUCT(std::move(fields)));
		alias_info->macros.push_back(std::move(member));
	}
	auto alias_statement = make_uniq<CreateStatement>();
	alias_statement->info = std::move(alias_info);
	unique_ptr<SQLStatement> alias_sql = std::move(alias_statement);
	auto alias_result = con.Query(std::move(alias_sql));
	REQUIRE_NO_FAIL(*alias_result);
	auto alias_types = con.Query("SELECT count(*), bool_and(parameter_types[1] LIKE '%payload_a%' OR "
	                             "parameter_types[1] LIKE '%payload_b%') "
	                             "FROM duckdb_functions() WHERE function_name = 'aliased_sibling'");
	REQUIRE(CHECK_COLUMN(alias_types, 0, {2}));
	REQUIRE(CHECK_COLUMN(alias_types, 1, {true}));
}

TEST_CASE("Batch member creation follows the overload identity", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "batchlang", make_shared_ptr<ViewBinder>(nullptr));

	// a dialect may submit several members in one CREATE: the set must satisfy the same conflict
	// identity as incremental creation, or EXPORT emits a set IMPORT cannot re-create
	auto make_member = [](const string &source, bool with_default) {
		auto member = make_uniq<LanguageMacroFunction>("batchlang", source);
		member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("a")));
		member->types.push_back(LogicalType::INTEGER);
		if (with_default) {
			auto default_expr = make_uniq<ConstantExpression>(Literal(LiteralKind::INTEGER, "5"));
			default_expr->SetAlias(Identifier("c"));
			member->default_parameters[Identifier("c")] = std::move(default_expr);
			member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("c")));
			member->types.push_back(LogicalType::INTEGER);
		}
		return member;
	};

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("batch_fun"));
	info->macros.push_back(make_member("fixed", false));
	info->macros.push_back(make_member("ranged", true));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	unique_ptr<SQLStatement> sql_statement = std::move(statement);
	auto result = con.Query(std::move(sql_statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "conflicting overload members"));

	// the identity is compared on the resolved signatures: an unresolved spelling of the same
	// type must not slip past the pairwise check
	auto spelled_info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	spelled_info->SetFunctionName(Identifier("batch_fun"));
	auto spelled_member = make_uniq<LanguageMacroFunction>("batchlang", "fixed");
	spelled_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("a")));
	spelled_member->types.push_back(UnresolvedUserType("integer"));
	spelled_info->macros.push_back(std::move(spelled_member));
	spelled_info->macros.push_back(make_member("ranged", true));
	auto spelled_statement = make_uniq<CreateStatement>();
	spelled_statement->info = std::move(spelled_info);
	unique_ptr<SQLStatement> spelled_sql = std::move(spelled_statement);
	auto spelled_result = con.Query(std::move(spelled_sql));
	REQUIRE_FAIL(spelled_result);
	REQUIRE(StringUtil::Contains(spelled_result->GetError(), "conflicting overload members"));

	// a pairwise valid batch is accepted
	auto valid_info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	valid_info->SetFunctionName(Identifier("batch_fun"));
	valid_info->macros.push_back(make_member("fixed", false));
	auto varchar_member = make_uniq<LanguageMacroFunction>("batchlang", "varchar ranged");
	varchar_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("a")));
	varchar_member->types.push_back(LogicalType::VARCHAR);
	auto default_expr = make_uniq<ConstantExpression>(Literal(LiteralKind::INTEGER, "5"));
	default_expr->SetAlias(Identifier("c"));
	varchar_member->default_parameters[Identifier("c")] = std::move(default_expr);
	varchar_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("c")));
	varchar_member->types.push_back(LogicalType::INTEGER);
	valid_info->macros.push_back(std::move(varchar_member));
	auto valid_statement = make_uniq<CreateStatement>();
	valid_statement->info = std::move(valid_info);
	unique_ptr<SQLStatement> valid_sql = std::move(valid_statement);
	auto valid_result = con.Query(std::move(valid_sql));
	REQUIRE_NO_FAIL(*valid_result);
	auto count = con.Query("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'batch_fun'");
	REQUIRE(CHECK_COLUMN(count, 0, {2}));
}

//! A zero-parameter language member whose undeclared RETURNS type carries extension metadata
static unique_ptr<LanguageMacroFunction> UndeclaredReturnsMetadataMember(const string &language) {
	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value::INTEGER(1));
	auto member = make_uniq<LanguageMacroFunction>(language, "x");
	member->return_type = LogicalType(LogicalTypeId::UNKNOWN).WithExtensionInfo(std::move(ext_info));
	return member;
}

TEST_CASE("Metadata on an undeclared RETURNS type is rejected at the catalog boundary", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// an undeclared RETURNS type serializes as its bare default, so attached metadata would
	// silently vanish on reopen: the rejection sits at the catalog boundary every producer funnels
	// through, not only in the CREATE binder
	RegisterTestLanguage(*db.instance, "unkretlang", make_shared_ptr<ViewBinder>(nullptr));

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("unk_ret_fun"));
	info->macros.push_back(UndeclaredReturnsMetadataMember("unkretlang"));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "metadata to an undeclared RETURNS type"));

	// the direct registration path an extension uses bypasses the CREATE binder and is rejected too
	CreateMacroInfo direct_info(CatalogType::MACRO_ENTRY);
	direct_info.SetFunctionName(Identifier("unk_ret_direct"));
	direct_info.macros.push_back(UndeclaredReturnsMetadataMember("unkretlang"));
	auto &system_catalog = Catalog::GetSystemCatalog(*db.instance);
	auto data = CatalogTransaction::GetSystemTransaction(*db.instance);
	auto &schema = system_catalog.GetSchema(data, Identifier::DefaultSchema());
	REQUIRE_THROWS(schema.CreateFunction(data, direct_info));
}

// the guard raises InternalException, which aborts the process when DUCKDB_CRASH_ON_ASSERT is set
#ifndef DUCKDB_CRASH_ON_ASSERT

//! Build a type the way only the raw C++ constructor can: a parameterless (INVALID) info carrying
//! extension metadata, which LogicalType normalizes to the serializable GENERIC representation
static LogicalType RawConstructedType(LogicalTypeId id, int32_t modifier) {
	auto type_info = make_uniq<LogicalTypeInfo>(LogicalTypeInfoType::INVALID_TYPE_INFO);
	type_info->extension_info = make_uniq<ExtensionTypeInfo>();
	type_info->extension_info->modifiers.emplace_back(Value::INTEGER(modifier));
	return LogicalType(id, std::move(type_info));
}

TEST_CASE("A raw-constructed type representation stays visible to the identity freeze", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the raw type constructor used to accept a parameterless info carrying extension metadata,
	// a representation serialization drops entirely: the freeze fingerprints would have matched
	// before and after this mutation. The constructor now normalizes it to GENERIC info
	RegisterTestLanguage(*db.instance, "rawlang", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 0);
		parameter_type.GetTypeInfo().extension_info->modifiers[0].value = Value::INTEGER(9);
	}));

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("raw_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("rawlang", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
	member->types.push_back(RawConstructedType(LogicalTypeId::INTEGER, 1));
	REQUIRE(member->types[0].GetTypeInfo().type == LogicalTypeInfoType::GENERIC_TYPE_INFO);
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("An ANY target type mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// AnyTypeInfo's target is serialized but not visited by the general-purpose TypeVisitor: the
	// fingerprint must observe a mutation of shared metadata below it
	RegisterTestLanguage(*db.instance, "anymut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 0);
		auto target = AnyType::GetTargetType(parameter_type);
		target.GetTypeInfo().extension_info->modifiers[0].value = Value::INTEGER(9);
	}));

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("any_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("anymut", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
	member->types.push_back(LogicalType::ANY_PARAMS(RawConstructedType(LogicalTypeId::INTEGER, 1)));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("A modifier value type mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// a modifier Value carries its own type, whose shared metadata is serialized with the Value:
	// the fingerprint must observe a mutation of metadata nested inside the extension info itself
	RegisterTestLanguage(*db.instance, "valmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 0);
		auto &value_type = parameter_type.GetTypeInfo().extension_info->modifiers[0].value.type();
		value_type.GetTypeInfo().extension_info->modifiers[0].value = Value::INTEGER(9);
	}));

	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value(RawConstructedType(LogicalTypeId::INTEGER, 1)));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("val_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("valmut", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
	member->types.push_back(LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info)));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("A resolved signature mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// a returned parameter type shares its interned metadata, and extension_info stays a mutable
	// pointee behind the const LogicalTypeInfo: without the signature freeze this collapses two
	// distinct overload signatures after the batch conflict check already accepted them
	RegisterTestLanguage(*db.instance, "sigmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 0);
		parameter_type.GetTypeInfo().extension_info->modifiers.emplace_back(Value::INTEGER(9));
	}));

	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value::INTEGER(1));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("sig_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("sigmut", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
	member->types.push_back(LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info)));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("A RETURNS type mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the declared RETURNS type is a cast boundary core enforces on every call: a retained-metadata
	// mutation after validation would change the cast the batch check already accepted
	RegisterTestLanguage(*db.instance, "retmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto return_type = info.MemberReturnType(0);
		return_type.GetTypeInfo().extension_info->modifiers.emplace_back(Value::INTEGER(9));
	}));

	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value::INTEGER(1));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("ret_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("retmut", "x");
	member->return_type = LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("A batch overload collapse through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the batch conflict check accepted two members distinct only in a modifier value: mutating the
	// second member's modifier to equal the first collapses the overload identities after the check
	RegisterTestLanguage(*db.instance, "collapselang",
	                     make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		                     auto parameter_type = info.MemberParameterType(1, 0);
		                     parameter_type.GetTypeInfo().extension_info->modifiers[0].value = Value::INTEGER(1);
	                     }));

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("collapsed_fun"));
	for (int32_t modifier = 1; modifier <= 2; modifier++) {
		auto ext_info = make_uniq<ExtensionTypeInfo>();
		ext_info->modifiers.emplace_back(Value::INTEGER(modifier));
		auto member = make_uniq<LanguageMacroFunction>("collapselang", "x");
		member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("p")));
		member->types.push_back(LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info)));
		info->macros.push_back(std::move(member));
	}
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

TEST_CASE("A nested type metadata mutation through a returned type copy is rejected", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// a returned container type shares its children's interned metadata too: the serialized
	// identity must observe a mutation below the top level and past parameter index zero
	RegisterTestLanguage(*db.instance, "nestmut", make_shared_ptr<ViewBinder>([](LanguageFunctionCreateInfo &info) {
		auto parameter_type = info.MemberParameterType(0, 1);
		auto &child = StructType::GetChildTypes(parameter_type)[0].second;
		child.GetTypeInfo().extension_info->modifiers.emplace_back(Value::INTEGER(9));
	}));

	auto ext_info = make_uniq<ExtensionTypeInfo>();
	ext_info->modifiers.emplace_back(Value::INTEGER(1));
	auto inner = LogicalType(LogicalTypeId::INTEGER).WithExtensionInfo(std::move(ext_info));
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	info->SetFunctionName(Identifier("nest_mutated"));
	auto member = make_uniq<LanguageMacroFunction>("nestmut", "x");
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("a")));
	member->types.push_back(LogicalType::INTEGER);
	member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("b")));
	member->types.push_back(LogicalType::STRUCT({{"i", inner}}));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "changed a resolved signature"));
}

#endif

TEST_CASE("Batch table member creation follows the overload identity", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	RegisterTestLanguage(*db.instance, "batchtablelang", make_shared_ptr<ViewBinder>(nullptr));

	auto make_table_member = [](const string &source, LogicalType param_type, bool with_default = false) {
		auto member = make_uniq<LanguageTableMacroFunction>("batchtablelang", source);
		member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("a")));
		member->types.push_back(std::move(param_type));
		if (with_default) {
			auto default_expr = make_uniq<ConstantExpression>(Literal(LiteralKind::INTEGER, "5"));
			default_expr->SetAlias(Identifier("c"));
			member->default_parameters[Identifier("c")] = std::move(default_expr);
			member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("c")));
			member->types.push_back(LogicalType::INTEGER);
		}
		member->return_names.emplace_back("i");
		member->return_types.emplace_back(LogicalType::INTEGER);
		return member;
	};
	auto query_batch = [&](unique_ptr<MacroFunction> first, unique_ptr<MacroFunction> second) {
		auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
		info->SetFunctionName(Identifier("batch_tf"));
		info->macros.push_back(std::move(first));
		info->macros.push_back(std::move(second));
		auto statement = make_uniq<CreateStatement>();
		statement->info = std::move(info);
		unique_ptr<SQLStatement> sql_statement = std::move(statement);
		return con.Query(std::move(sql_statement));
	};

	// a batch of table members follows the same conflict identity as incremental creation,
	// including the default arity range
	auto result = query_batch(make_table_member("fixed", LogicalType::INTEGER),
	                          make_table_member("ranged", LogicalType::INTEGER, true));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "conflicting overload members"));

	// an unresolved spelling of the same nested type must not slip past the pairwise check
	result = query_batch(make_table_member("resolved ranged", LogicalType::STRUCT({{"m", LogicalType::INTEGER}}), true),
	                     make_table_member("spelled", LogicalType::STRUCT({{"m", UnresolvedUserType("integer")}})));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "conflicting overload members"));

	// distinct unbound STRUCT spellings resolve to distinct signatures and are not wrongly rejected
	result =
	    query_batch(make_table_member("int struct", LogicalType::STRUCT({{"m", UnresolvedUserType("integer")}})),
	                make_table_member("varchar struct", LogicalType::STRUCT({{"m", UnresolvedUserType("varchar")}})));
	REQUIRE_NO_FAIL(*result);
	auto count = con.Query("SELECT count(*) FROM duckdb_functions() WHERE function_name = 'batch_tf'");
	REQUIRE(CHECK_COLUMN(count, 0, {2}));
	auto types = con.Query("SELECT unnest(parameter_types) FROM duckdb_functions() WHERE function_name = 'batch_tf' "
	                       "ORDER BY 1");
	REQUIRE(CHECK_COLUMN(types, 0, {"STRUCT(m INTEGER)", "STRUCT(m VARCHAR)"}));
}

TEST_CASE("Unresolved column-type spellings resolve at CREATE", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// the columns must already be resolved when ValidateCreate observes them (and gets frozen so)
	auto saw_resolved = make_shared_ptr<bool>(false);
	RegisterTestLanguage(*db.instance, "coltypelang",
	                     make_shared_ptr<ViewBinder>([saw_resolved](LanguageFunctionCreateInfo &info) {
		                     auto nested = info.MemberReturnColumnType(0, 0);
		                     *saw_resolved = nested.id() == LogicalTypeId::STRUCT &&
		                                     StructType::GetChildTypes(nested)[0].second.id() == LogicalTypeId::ENUM;
	                     }));
	REQUIRE_NO_FAIL(con.Query("CREATE TYPE col_mood AS ENUM ('happy')"));

	// a dialect may nest an unresolved user type inside a concrete container in a RETURN column;
	// the collated sibling pins that untouched columns keep their identity through resolution
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->SetFunctionName(Identifier("col_tf"));
	auto member = make_uniq<LanguageTableMacroFunction>("coltypelang", "column types");
	member->return_names.emplace_back("c");
	member->return_types.push_back(LogicalType::STRUCT({{"m", UnresolvedUserType("col_mood")}}));
	member->return_names.emplace_back("s");
	member->return_types.push_back(LogicalType::VARCHAR_COLLATION("nocase"));
	info->macros.push_back(std::move(member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	unique_ptr<SQLStatement> sql_statement = std::move(statement);
	auto result = con.Query(std::move(sql_statement));
	REQUIRE_NO_FAIL(*result);
	REQUIRE(*saw_resolved);

	// the stored column types are fully resolved and the sibling kept its collation
	auto shown = con.Query("SELECT return_type FROM duckdb_functions() WHERE function_name = 'col_tf'");
	REQUIRE(CHECK_COLUMN(shown, 0, {"TABLE(c STRUCT(m ENUM('happy')), s VARCHAR('nocase'))"}));
}

TEST_CASE("Mixed scalar and table language members are rejected at CREATE", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	// only constructible via the API: the SQL grammar produces homogeneous member kinds
	auto info = make_uniq<CreateMacroInfo>(CatalogType::TABLE_MACRO_ENTRY);
	info->SetFunctionName(Identifier("mixed_kinds"));
	auto table_member = make_uniq<LanguageTableMacroFunction>("missinglang", "table source");
	table_member->return_names.emplace_back("i");
	table_member->return_types.emplace_back(LogicalType::INTEGER);
	info->macros.push_back(std::move(table_member));
	auto scalar_member = make_uniq<LanguageMacroFunction>("missinglang", "scalar source");
	scalar_member->parameters.push_back(make_uniq<ColumnRefExpression>(Identifier("x")));
	scalar_member->types.push_back(LogicalType::INTEGER);
	info->macros.push_back(std::move(scalar_member));
	auto statement = make_uniq<CreateStatement>();
	statement->info = std::move(info);
	auto result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "cannot combine language function members of different kinds"));

	// ... and in the other direction: a table member in a scalar entry
	auto scalar_info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	scalar_info->SetFunctionName(Identifier("mixed_kinds2"));
	auto lone_table_member = make_uniq<LanguageTableMacroFunction>("missinglang", "table source");
	lone_table_member->return_names.emplace_back("i");
	lone_table_member->return_types.emplace_back(LogicalType::INTEGER);
	scalar_info->macros.push_back(std::move(lone_table_member));
	statement = make_uniq<CreateStatement>();
	statement->info = std::move(scalar_info);
	result = con.Query(std::move(statement));
	REQUIRE_FAIL(result);
	REQUIRE(StringUtil::Contains(result->GetError(), "cannot combine language function members of different kinds"));
}
