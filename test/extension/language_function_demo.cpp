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
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
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
	void ValidateCreate(ClientContext &context, CreateMacroInfo &info) override {
		// deliberately reallocate the member vector: core must not hold member references across this callback
		info.macros.reserve(info.macros.capacity() + 1);
		for (auto &member : info.macros) {
			auto &language_member = LanguageMacroFunction::Get(*member);
			if (StringUtil::Contains(language_member.source, "invalid")) {
				throw BinderException("testlang: the source of \"%s\" may not contain the word \"invalid\"",
				                      info.GetFunctionName().GetIdentifierName());
			}
			if (member->type == MacroType::LANGUAGE_TABLE_MACRO &&
			    !StringUtil::StartsWith(language_member.source, "rows:")) {
				// a testlang table function emits the rows its 'rows:<n>' source denotes
				throw BinderException("testlang: the source of table function \"%s\" must have the form 'rows:<n>'",
				                      info.GetFunctionName().GetIdentifierName());
			}
		}
		// publish per-member documentation: function metadata must survive member mutations
		for (auto &member : info.macros) {
			auto &language_member = LanguageMacroFunction::Get(*member);
			FunctionDescription description;
			for (idx_t i = 0; i < member->parameters.size(); i++) {
				auto &param_name = member->parameters[i]->Cast<ColumnRefExpression>().GetColumnName();
				description.parameter_names.push_back(param_name.GetIdentifierName());
				// an untyped parameter is documented as a wildcard, like built-in function docs
				const auto &type = member->types[i];
				description.parameter_types.push_back(type.id() == LogicalTypeId::UNKNOWN ? LogicalType::ANY : type);
			}
			// a per-source description, so tests distinguish stale metadata from replaced metadata
			description.description = "testlang: " + language_member.source;
			info.descriptions.push_back(std::move(description));
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
		const auto spec = member.source.substr(5);
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
		const idx_t column_count = member.return_names.empty() ? 1 : member.return_names.size();
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
}
}
