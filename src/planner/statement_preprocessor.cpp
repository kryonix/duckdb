#include "duckdb/planner/statement_preprocessor.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/parser/parser.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/pragma_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/language_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/procedure_catalog_entry.hpp"
#include "duckdb/function/language_procedure_function.hpp"
#include "duckdb/main/language_function_binder.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/statement/call_statement.hpp"
#include "duckdb/parser/statement/multi_statement.hpp"
#include "duckdb/parser/statement/transaction_statement.hpp"
#include "duckdb/parser/parsed_data/bound_pragma_info.hpp"
#include "duckdb/function/function.hpp"

#include "duckdb/main/client_context.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/function_binder.hpp"

namespace duckdb {

StatementPreprocessor::StatementPreprocessor(ClientContext &context) : context(context) {
}

vector<unique_ptr<SQLStatement>> StatementPreprocessor::TryReparsePragma(unique_ptr<SQLStatement> statement) const {
	// Try reparsing
	const auto info = statement->Cast<PragmaStatement>().info->Copy();
	QueryErrorContext error_context(statement->stmt_location);
	const auto binder = Binder::CreateBinder(context);
	const auto bound_info = binder->BindPragma(*info, error_context);
	if (bound_info->function.query) {
		// Needs reparsing
		FunctionParameters parameters {bound_info->parameters, bound_info->named_parameters, &bound_info->function};
		const auto query_to_reparse = bound_info->function.query(context, parameters);
		Parser parser(context);
		parser.ParseQuery(query_to_reparse);
		return std::move(parser.statements);
	}
	vector<unique_ptr<SQLStatement>> res;
	res.push_back(std::move(statement));
	return res;
}

vector<unique_ptr<SQLStatement>> StatementPreprocessor::TryExpandCall(unique_ptr<SQLStatement> statement) const {
	auto &call = statement->Cast<CallStatement>();
	vector<unique_ptr<SQLStatement>> result;
	if (!call.function || call.function->GetExpressionType() != ExpressionType::FUNCTION) {
		result.push_back(std::move(statement));
		return result;
	}
	auto &fexpr = call.function->Cast<FunctionExpression>();
	// resolve the CALL's qualified name as a procedure; a miss is the ordinary CALL path untouched
	const auto binder = Binder::CreateBinder(context);
	QueryErrorContext error_context(statement->stmt_location);
	auto bound_name = binder->BindTableName(fexpr.GetQualifiedName());
	EntryLookupInfo procedure_lookup(CatalogType::PROCEDURE_ENTRY, bound_name, error_context);
	auto entry = binder->GetCatalogEntry(procedure_lookup, OnEntryNotFound::RETURN_NULL);
	if (!entry || entry->type != CatalogType::PROCEDURE_ENTRY) {
		result.push_back(std::move(statement));
		return result;
	}
	auto &procedure = entry->Cast<ProcedureCatalogEntry>();

	// structural member selection: arity, named arguments, defaults filled, declaration order
	vector<unique_ptr<ParsedExpression>> positional_arguments;
	InsertionOrderPreservingMap<unique_ptr<ParsedExpression>, Identifier, identifier_map_t<idx_t>> named_arguments;
	auto bind_result = MacroFunction::BindMacroFunction(*binder, procedure.macros, procedure.name, fexpr,
	                                                    positional_arguments, named_arguments, 0);
	if (!bind_result.error.empty()) {
		throw BinderException(fexpr, bind_result.error);
	}
	auto &member = procedure.macros[bind_result.function_idx.GetIndex()]->Cast<LanguageProcedureFunction>();

	// flatten into parameter declaration order; the expressions stay unevaluated
	auto arguments = std::move(positional_arguments);
	for (idx_t i = arguments.size(); i < member.parameters.size(); i++) {
		auto &param_name = member.parameters[i]->Cast<ColumnRefExpression>().GetColumnName();
		auto named_it = named_arguments.find(param_name);
		D_ASSERT(named_it != named_arguments.end());
		arguments.push_back(std::move(named_it->second));
	}
	for (auto &argument : arguments) {
		// drop named-argument and default bookkeeping aliases, like the scalar and table seams
		argument->ClearAlias();
	}

	// raises a MissingExtensionException naming the procedure and language when none is registered
	auto language_binder = LanguageCatalogEntry::GetBinder(context, member.language, procedure.name);
	language_binder->ExpandCall(context, procedure, member, std::move(arguments), result);
	return result;
}

PreprocessedStatement StatementPreprocessor::ExpandStatement(unique_ptr<SQLStatement> statement) const {
	PreprocessedStatement result;
	switch (statement->type) {
	case StatementType::MULTI_STATEMENT: {
		auto &multi_statement = statement->Cast<MultiStatement>();
		result.statements = std::move(multi_statement.statements);
		break;
	}
	case StatementType::PRAGMA_STATEMENT:
		result.statements = TryReparsePragma(std::move(statement));
		break;
	case StatementType::CALL_STATEMENT: {
		result.statements = TryExpandCall(std::move(statement));
		// a CALL expansion of any size runs inside the call's transaction, so even a single
		// statement must not be a transaction statement
		for (auto &expanded : result.statements) {
			if (expanded->type == StatementType::TRANSACTION_STATEMENT) {
				throw InvalidInputException("A CALL expansion cannot contain a transaction statement: %s",
				                            expanded->query);
			}
		}
		break;
	}
	default:
		result.statements.push_back(std::move(statement));
		return result;
	}
	if (result.statements.size() > 1) {
		for (auto &expanded : result.statements) {
			// The statements run as one query in one transaction, which a transaction statement would end or take over
			if (expanded->type == StatementType::TRANSACTION_STATEMENT) {
				throw InvalidInputException(
				    "A statement that expands into several cannot contain a transaction statement: %s",
				    expanded->query);
			}
		}
	}
	return result;
}

PreprocessedStatement StatementPreprocessor::PreprocessStatement(ClientContextLock &lock,
                                                                 unique_ptr<SQLStatement> statement) {
	if (statement->type != StatementType::PRAGMA_STATEMENT && statement->type != StatementType::MULTI_STATEMENT &&
	    statement->type != StatementType::CALL_STATEMENT) {
		return ExpandStatement(std::move(statement));
	}
	// Only PRAGMA, CALL, and multi-statements touch the catalog and need a transaction
	PreprocessedStatement result;
	context.RunFunctionInTransactionInternal(lock, [&] { result = ExpandStatement(std::move(statement)); });
	return result;
}

} // namespace duckdb
