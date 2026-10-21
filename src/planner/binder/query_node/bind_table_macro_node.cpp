#include "duckdb/catalog/catalog_entry/scalar_macro_catalog_entry.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/subquery_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/comparison_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/tableref/joinref.hpp"
#include "duckdb/catalog/catalog_entry/language_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/main/language_function_binder.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/function/language_table_macro_function.hpp"
#include "duckdb/function/table_macro_function.hpp"

namespace duckdb {

unique_ptr<QueryNode> Binder::BindTableMacro(FunctionExpression &function, TableMacroCatalogEntry &macro_func,
                                             idx_t depth) {
	// expanding nested macros can take a long time, allow the query to be interrupted
	context.InterruptCheck();
	// validate the arguments and separate positional and default arguments
	vector<unique_ptr<ParsedExpression>> positional_arguments;
	InsertionOrderPreservingMap<unique_ptr<ParsedExpression>, Identifier, identifier_map_t<idx_t>> named_arguments;

	auto bind_result = MacroFunction::BindMacroFunction(*this, macro_func.macros, macro_func.name, function,
	                                                    positional_arguments, named_arguments, depth);
	if (!bind_result.error.empty()) {
		throw BinderException(function, bind_result.error);
	}
	auto &macro_def = *macro_func.macros[bind_result.function_idx.GetIndex()];

	auto new_macro_binding =
	    MacroFunction::CreateDummyBinding(macro_def, macro_func.name, positional_arguments, named_arguments);
	new_macro_binding->arguments = &positional_arguments;

	// We need an ExpressionBinder so that we can call ExpressionBinder::ReplaceMacroParametersRecursive()
	auto eb = ExpressionBinder(*this, this->context);

	eb.macro_binding = new_macro_binding.get();
	vector<identifier_set_t> lambda_params;

	auto node = macro_def.Cast<TableMacroFunction>().query_node->Copy();
	ParsedExpressionIterator::EnumerateQueryNodeChildren(
	    *node, [&](unique_ptr<ParsedExpression> &child) { eb.ReplaceMacroParameters(child, lambda_params); });

	return node;
}

BoundStatement Binder::BindLanguageTableMacro(TableFunctionRef &ref, FunctionExpression &function,
                                              TableMacroCatalogEntry &macro_func) {
	context.InterruptCheck();
	// select the member: core resolves the overload, named arguments, defaults, and parameter casts
	vector<unique_ptr<ParsedExpression>> positional_arguments;
	InsertionOrderPreservingMap<unique_ptr<ParsedExpression>, Identifier, identifier_map_t<idx_t>> named_arguments;
	auto bind_result = MacroFunction::BindMacroFunction(*this, macro_func.macros, macro_func.name, function,
	                                                    positional_arguments, named_arguments, 0);
	if (!bind_result.error.empty()) {
		throw BinderException(function, bind_result.error);
	}
	auto &member = macro_func.macros[bind_result.function_idx.GetIndex()]->Cast<LanguageTableMacroFunction>();

	// flatten the arguments into parameter declaration order, defaults were already filled in.
	// the expressions stay unevaluated, so correlated (lateral) arguments pass through
	auto arguments = std::move(positional_arguments);
	for (idx_t i = arguments.size(); i < member.parameters.size(); i++) {
		auto &param_name = member.parameters[i]->Cast<ColumnRefExpression>().GetColumnName();
		auto named_it = named_arguments.find(param_name);
		D_ASSERT(named_it != named_arguments.end());
		arguments.push_back(std::move(named_it->second));
	}
	for (auto &argument : arguments) {
		// drop named-argument and default bookkeeping aliases: a spliced argument must not turn
		// into a named argument of whatever call the language binder builds around it
		argument->ClearAlias();
	}

	// raises a MissingExtensionException naming the function and language when none is registered
	auto language_binder =
	    LanguageCatalogEntry::GetBinder(context, member.language, macro_func.name.GetIdentifierName());
	auto statement = language_binder->BindTableCall(*this, macro_func, member, std::move(arguments));
	if (!statement || !statement->node) {
		throw InternalException("The language binder for \"%s\" returned no statement for table function %s",
		                        member.language, macro_func.name);
	}

	// bind the returned statement in a correlated child binder, like a table macro expansion
	auto binder = Binder::CreateBinder(context, this);
	binder->SetCanContainNulls(true);
	binder->AddBoundTableMacro(macro_func);
	binder->alias = ref.alias.empty() ? "unnamed_query" : ref.alias;
	BoundStatement query;
	try {
		query = binder->BindNode(*statement->node);
	} catch (std::exception &ex) {
		ErrorData error(ex);
		error.AddQueryLocation(ref);
		error.Throw();
	}

	// a declared or published schema is enforced column-wise, like the scalar RETURNS cast
	if (!member.return_names.empty()) {
		if (query.types.size() != member.return_types.size()) {
			throw BinderException(function,
			                      "Table function %s declares %llu result columns, but its %s binder returned a "
			                      "statement with %llu columns",
			                      macro_func.name, member.return_types.size(), member.language, query.types.size());
		}
		query.plan = CastLogicalOperatorToTypes(query.types, member.return_types, std::move(query.plan));
		query.types = member.return_types;
		query.names = member.return_names;
	}

	auto bind_index = query.plan->GetRootIndex();
	string alias = (ref.alias.empty() ? "unnamed_query" + to_string(bind_index.index) : ref.alias.GetIdentifierName());
	bind_context.AddSubquery(bind_index, Identifier(alias), ref, query);
	MoveCorrelatedExpressions(*binder);
	return query;
}

} // namespace duckdb
