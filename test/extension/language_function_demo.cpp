#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/scalar_macro_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/common/operator/cast_operators.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/language_macro_function.hpp"
#include "duckdb/function/language_table_macro_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/language_function_binder.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"

using namespace duckdb;

//! A deliberately trivial language: a function "compiles" to the constant its source text denotes.
//! It exists to exercise the macro-family language function layer from SQL tests.
class TestLangBinder : public LanguageFunctionBinder {
public:
	void ValidateCreate(ClientContext &context, LanguageFunctionCreateInfo &info) override {
		for (idx_t member_idx = 0; member_idx < info.MemberCount(); member_idx++) {
			if (StringUtil::Contains(info.MemberSource(member_idx), "invalid")) {
				throw BinderException("testlang: the source of %s may not contain the word \"invalid\"",
				                      info.GetFunctionName());
			}
			if (info.MemberIsTableFunction(member_idx)) {
				if (!StringUtil::StartsWith(info.MemberSource(member_idx), "rows:")) {
					// a testlang table function emits the rows its 'rows:<n>' source denotes
					throw BinderException("testlang: the source of table function %s must have the form 'rows:<n>'",
					                      info.GetFunctionName());
				}
				if (info.MemberReturnColumnCount(member_idx) == 0) {
					// publish-once: resolve an undeclared schema at CREATE, like an inferring language
					info.PublishMemberSchema(member_idx, {Identifier("i")}, {LogicalType::BIGINT});
				}
			}
			if (info.MemberLanguageVersion(member_idx).empty()) {
				// publish-once: record the language version the definition was validated with
				info.PublishMemberLanguageVersion(member_idx, "testlang 1.0");
			}
		}
		// publish per-member documentation: function metadata must survive member mutations
		for (idx_t member_idx = 0; member_idx < info.MemberCount(); member_idx++) {
			FunctionDescription description;
			for (idx_t i = 0; i < info.MemberParameterCount(member_idx); i++) {
				description.parameter_names.push_back(info.MemberParameterName(member_idx, i));
				// an untyped parameter is documented as a wildcard, like built-in function docs
				const auto &type = info.MemberParameterType(member_idx, i);
				description.parameter_types.push_back(type.id() == LogicalTypeId::UNKNOWN ? LogicalType::ANY : type);
			}
			// a per-source description, so tests distinguish stale metadata from replaced metadata
			description.description = "testlang: " + info.MemberSource(member_idx);
			info.AddDescription(std::move(description));
		}
	}

	BindResult BindScalarCall(ExpressionBinder &binder, ScalarMacroCatalogEntry &entry, LanguageMacroFunction &member,
	                          vector<unique_ptr<ParsedExpression>> arguments, idx_t depth) override {
		// a testlang constant ignores its arguments, but argument resolution errors must still surface
		ErrorData error;
		vector<unique_ptr<Expression>> bound_arguments;
		for (auto &argument : arguments) {
			auto bound_argument = binder.BindChild(argument, depth, error);
			if (error.HasError()) {
				return BindResult(std::move(error));
			}
			bound_arguments.push_back(std::move(bound_argument));
		}
		// the source 'arg:<n>' compiles to the n-th argument, so tests observe the declaration
		// order, default insertion, and parameter casts core hands to the language binder
		if (StringUtil::StartsWith(member.source, "arg:")) {
			idx_t argument_idx;
			const auto index_text = member.source.substr(4);
			if (index_text.empty() || !TryCast::Operation<string_t, idx_t>(string_t(index_text), argument_idx)) {
				return BindResult(BinderException("testlang: malformed argument index \"%s\"", index_text));
			}
			if (argument_idx >= bound_arguments.size()) {
				return BindResult(BinderException("testlang: argument index %llu is out of range", argument_idx));
			}
			return BindResult(std::move(bound_arguments[argument_idx]));
		}
		// otherwise the compiled form is the constant the source denotes, core applies the
		// declared RETURNS cast
		return BindResult(make_uniq<BoundConstantExpression>(Value(member.source)));
	}

	unique_ptr<SelectStatement> BindTableCall(Binder &binder, TableMacroCatalogEntry &entry,
	                                          LanguageTableMacroFunction &member,
	                                          vector<unique_ptr<ParsedExpression>> arguments) override {
		// the source 'rows:<n>' compiles to a query emitting n rows; 'rows:arg:<i>' takes the row
		// count from the i-th UNEVALUATED argument expression, so correlated (lateral) arguments
		// exercise the seam placement before argument evaluation
		if (!StringUtil::StartsWith(member.source, "rows:")) {
			throw BinderException("testlang: the source of table function \"%s\" is not a 'rows:<n>' spec",
			                      entry.name.GetIdentifierName());
		}
		auto spec = member.source.substr(5);
		if (spec == "self") {
			// a self-call, so tests observe the recursion guard on the expansion
			auto select = make_uniq<SelectNode>();
			select->select_list.push_back(make_uniq<StarExpression>());
			auto self_ref = make_uniq<TableFunctionRef>();
			self_ref->function = make_uniq<FunctionExpression>(entry.name, vector<unique_ptr<ParsedExpression>>());
			select->from_table = std::move(self_ref);
			auto statement = make_uniq<SelectStatement>();
			statement->node = std::move(select);
			return statement;
		}
		// a ':wide' suffix emits one extra column, so tests observe the declared-schema shape check
		bool wide = StringUtil::EndsWith(spec, ":wide");
		if (wide) {
			spec = spec.substr(0, spec.size() - 5);
		}
		unique_ptr<ParsedExpression> count_expr;
		if (StringUtil::StartsWith(spec, "arg:")) {
			idx_t argument_idx;
			const auto index_text = spec.substr(4);
			if (index_text.empty() || !TryCast::Operation<string_t, idx_t>(string_t(index_text), argument_idx)) {
				throw BinderException("testlang: malformed argument index \"%s\"", index_text);
			}
			if (argument_idx >= arguments.size()) {
				throw BinderException("testlang: argument index %llu is out of range", argument_idx);
			}
			count_expr = std::move(arguments[argument_idx]);
		} else {
			idx_t row_count;
			if (spec.empty() || !TryCast::Operation<string_t, idx_t>(string_t(spec), row_count)) {
				throw BinderException("testlang: malformed row count \"%s\"", spec);
			}
			count_expr = ConstantExpression::FromValue(Value::BIGINT(NumericCast<int64_t>(row_count)));
		}
		// compile to SELECT range [, range ...] FROM range(count): core binds the statement in a
		// correlated child binder and renames/casts the columns to the declared schema
		auto select = make_uniq<SelectNode>();
		const idx_t column_count = (member.return_names.empty() ? 1 : member.return_names.size()) + (wide ? 1 : 0);
		for (idx_t i = 0; i < column_count; i++) {
			select->select_list.push_back(make_uniq<ColumnRefExpression>("range"));
		}
		vector<unique_ptr<ParsedExpression>> range_arguments;
		range_arguments.push_back(std::move(count_expr));
		auto range_ref = make_uniq<TableFunctionRef>();
		range_ref->function = make_uniq<FunctionExpression>("range", std::move(range_arguments));
		select->from_table = std::move(range_ref);
		auto statement = make_uniq<SelectStatement>();
		statement->node = std::move(select);
		return statement;
	}
};

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(language_function_demo, loader) {
	loader.RegisterLanguage("testlang", make_shared_ptr<TestLangBinder>());
	// an aliased collated type, so tests pin that signature rendering preserves type identity
	loader.RegisterType("ci_text", LogicalType::VARCHAR_COLLATION("nocase").WithAlias("ci_text"));
	// an alias that collides with a keyword, so tests pin that rendered signatures re-parse
	loader.RegisterType("table", LogicalType(LogicalTypeId::SMALLINT).WithAlias("table"));
}
}
