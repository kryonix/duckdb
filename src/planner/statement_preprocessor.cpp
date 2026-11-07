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
#include "duckdb/parser/statement/explain_statement.hpp"
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

void StatementPreprocessor::TryExpandCall(unique_ptr<SQLStatement> statement, PreprocessedStatement &result) const {
	auto &call = statement->Cast<CallStatement>();
	if (!call.function || call.function->GetExpressionType() != ExpressionType::FUNCTION) {
		result.statements.push_back(std::move(statement));
		return;
	}
	auto &fexpr = call.function->Cast<FunctionExpression>();
	// resolve the CALL's qualified name as a procedure; a miss is the ordinary CALL path untouched
	const auto binder = Binder::CreateBinder(context);
	QueryErrorContext error_context(statement->stmt_location);
	auto bound_name = binder->BindTableName(fexpr.GetQualifiedName());
	EntryLookupInfo procedure_lookup(CatalogType::PROCEDURE_ENTRY, bound_name, error_context);
	auto entry = binder->GetCatalogEntry(procedure_lookup, OnEntryNotFound::RETURN_NULL);
	if (!entry || entry->type != CatalogType::PROCEDURE_ENTRY) {
		result.statements.push_back(std::move(statement));
		return;
	}
	auto &procedure = entry->Cast<ProcedureCatalogEntry>();
	if (!statement->named_param_map.empty()) {
		// parameters cannot thread through a re-expanded statement sequence
		throw InvalidInputException(ClientContext::EXPANDING_PARAMETERS_ERROR);
	}

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
	vector<unique_ptr<SQLStatement>> expansion;
	language_binder->ExpandCall(context, procedure, member, std::move(arguments), expansion);

	// normalize the expansion like top-level statements: unpack multi-statements and reparse pragmas
	for (auto &expanded : expansion) {
		if (expanded->type == StatementType::MULTI_STATEMENT) {
			for (auto &child : expanded->Cast<MultiStatement>().statements) {
				if (child->type == StatementType::PRAGMA_STATEMENT) {
					auto reparsed = TryReparsePragma(std::move(child));
					for (auto &reparsed_statement : reparsed) {
						result.statements.push_back(std::move(reparsed_statement));
					}
					continue;
				}
				result.statements.push_back(std::move(child));
			}
			continue;
		}
		if (expanded->type == StatementType::PRAGMA_STATEMENT) {
			auto reparsed = TryReparsePragma(std::move(expanded));
			for (auto &reparsed_statement : reparsed) {
				result.statements.push_back(std::move(reparsed_statement));
			}
			continue;
		}
		result.statements.push_back(std::move(expanded));
	}

	result.call_procedure = procedure.name.GetIdentifierName();
	result.call_result_names = member.return_names;
	result.call_result_types = member.return_types;
	if (result.statements.empty() && !result.call_result_types.empty()) {
		throw BinderException("CALL of procedure %s: the procedure declares TABLE(%s) but its expansion contains no "
		                      "row-returning statement",
		                      result.call_procedure, member.ColumnsToSQL());
	}
}

static bool ContainsTransactionStatement(const SQLStatement &statement) {
	if (statement.type == StatementType::TRANSACTION_STATEMENT) {
		return true;
	}
	// EXPLAIN ANALYZE executes its child, so a wrapped transaction statement would run
	if (statement.type == StatementType::EXPLAIN_STATEMENT) {
		auto &child = statement.Cast<ExplainStatement>().stmt;
		return child && ContainsTransactionStatement(*child);
	}
	return false;
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
		TryExpandCall(std::move(statement), result);
		// a CALL expansion of any size runs inside the call's transaction, so even a single
		// statement must not be a transaction statement
		for (auto &expanded : result.statements) {
			if (ContainsTransactionStatement(*expanded)) {
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
			if (ContainsTransactionStatement(*expanded)) {
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
