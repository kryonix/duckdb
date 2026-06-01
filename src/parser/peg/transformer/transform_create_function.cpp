#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/parsed_data/create_duckpl_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/statement/duckpl_script_statement.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"

namespace duckdb {

static void ValidateRoutineParameters(const vector<MacroParameter> &parameters, const char *object_type) {
	bool default_value_found = false;
	case_insensitive_string_set_t parameter_names;
	for (auto &parameter : parameters) {
		D_ASSERT(!parameter.name.empty());
		if (parameter_names.find(parameter.name) != parameter_names.end()) {
			throw ParserException("Duplicate parameter '%s' in %s definition", parameter.name, object_type);
		}
		parameter_names.insert(parameter.name);
		if (!parameter.is_default && default_value_found) {
			throw ParserException("Parameter without a default follows parameter with a default");
		}
		default_value_found = default_value_found || parameter.is_default;
	}
}

static void ApplyMacroParameters(MacroFunction &macro_function, vector<MacroParameter> parameters) {
	for (auto &parameter : parameters) {
		if (parameter.is_default) {
			auto default_expr = std::move(parameter.expression);
			default_expr->SetAlias(parameter.name);
			macro_function.default_parameters[parameter.name] = std::move(default_expr);
			macro_function.parameters.push_back(make_uniq<ColumnRefExpression>(parameter.name));
		} else {
			macro_function.parameters.push_back(std::move(parameter.expression));
		}
		macro_function.types.push_back(parameter.type);
	}
}

static void SetCreateSchemaCatalog(CreateFunctionInfo &info, const QualifiedName &qualified_name) {
	if (qualified_name.schema.empty()) {
		info.schema = qualified_name.catalog;
	} else {
		info.catalog = qualified_name.catalog;
		info.schema = qualified_name.schema;
	}
}

static void ValidateDuckPLLanguageName(const string &language) {
	if (!StringUtil::CIEquals(language, "duckpl") && !StringUtil::CIEquals(language, "plpgsql")) {
		throw ParserException("Unsupported DuckPL language specifier '%s'", language);
	}
}

static unique_ptr<DuckPLBlockStatement> PrependDuckPLStatement(unique_ptr<DuckPLParseStatement> statement,
                                                               unique_ptr<DuckPLBlockStatement> remaining_statements) {
	vector<unique_ptr<DuckPLParseStatement>> result;
	result.push_back(std::move(statement));
	if (remaining_statements) {
		for (auto &entry : remaining_statements->statements) {
			result.push_back(std::move(entry));
		}
	}
	return make_uniq<DuckPLBlockStatement>(std::move(result));
}

unique_ptr<CreateStatement>
PEGTransformerFactory::TransformCreateFunctionStmt(PEGTransformer &transformer, const bool &if_not_exists,
                                                   const QualifiedName &qualified_name,
                                                   vector<CreateRoutineDefinition> function_create_definition) {
	D_ASSERT(!function_create_definition.empty());

	auto result = make_uniq<CreateStatement>();
	auto on_conflict = if_not_exists ? OnCreateConflict::IGNORE_ON_CONFLICT : OnCreateConflict::ERROR_ON_CONFLICT;
	if (function_create_definition[0].IsDuckPL()) {
		if (function_create_definition.size() != 1) {
			throw ParserException("Expected a single DuckPL function definition");
		}
		auto definition = std::move(function_create_definition[0]);
		D_ASSERT(definition.duckpl);

		auto info = make_uniq<CreateDuckPLFunctionInfo>();
		SetCreateSchemaCatalog(*info, qualified_name);
		info->name = qualified_name.name;
		info->on_conflict = on_conflict;
		info->definition = make_shared_ptr<DuckPLFunctionDefinition>(
		    qualified_name.name, std::move(definition.parameters), false,
		    definition.has_return_type ? definition.return_type : LogicalType::UNKNOWN,
		    std::move(definition.duckpl->body), definition.duckpl->location);
		result->info = std::move(info);
		return result;
	}

	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	SetCreateSchemaCatalog(*info, qualified_name);
	info->name = qualified_name.name;
	info->on_conflict = on_conflict;

	for (auto &routine_definition : function_create_definition) {
		if (routine_definition.IsDuckPL() || !routine_definition.macro) {
			throw ParserException("Expected macro definition");
		}
		info->macros.push_back(std::move(routine_definition.macro));
	}

	D_ASSERT(!info->macros.empty());
	auto macro_type = info->macros[0]->type;
	for (idx_t i = 1; i < info->macros.size(); ++i) {
		if (info->macros[i]->type != macro_type) {
			throw ParserException("Cannot mix table and scalar macro function definitions");
		}
	}
	info->type = macro_type == MacroType::TABLE_MACRO ? CatalogType::TABLE_MACRO_ENTRY : CatalogType::MACRO_ENTRY;
	result->info = std::move(info);
	transformer.PivotEntryCheck("macro");
	return result;
}

vector<CreateRoutineDefinition>
PEGTransformerFactory::TransformDuckPLFunctionDefinitions(PEGTransformer &transformer,
                                                          CreateRoutineDefinition duck_pl_function_definition) {
	vector<CreateRoutineDefinition> result;
	result.push_back(std::move(duck_pl_function_definition));
	return std::move(result);
}

CreateRoutineDefinition PEGTransformerFactory::TransformDuckPLFunctionDefinition(
    PEGTransformer &transformer, vector<MacroParameter> duck_pl_function_parameters, const LogicalType &duck_pl_returns,
    const shared_ptr<DuckPLScriptDefinition> &duck_pl_wrapped_definition) {
	ValidateRoutineParameters(duck_pl_function_parameters, "function");

	CreateRoutineDefinition result;
	result.parameters = std::move(duck_pl_function_parameters);
	result.has_return_type = duck_pl_returns != LogicalType::INVALID;
	result.return_type = result.has_return_type ? duck_pl_returns : LogicalType::UNKNOWN;
	result.duckpl = duck_pl_wrapped_definition;
	return result;
}

vector<CreateRoutineDefinition> PEGTransformerFactory::TransformLegacyMacroFunctionDefinitions(
    PEGTransformer &transformer, vector<CreateRoutineDefinition> legacy_macro_function_definition) {
	return std::move(legacy_macro_function_definition);
}

CreateRoutineDefinition
PEGTransformerFactory::TransformLegacyTableMacroFunctionDefinition(PEGTransformer &transformer,
                                                                   vector<MacroParameter> duck_pl_function_parameters,
                                                                   unique_ptr<MacroFunction> table_macro_definition) {
	ValidateRoutineParameters(duck_pl_function_parameters, "macro");

	CreateRoutineDefinition result;
	ApplyMacroParameters(*table_macro_definition, std::move(duck_pl_function_parameters));
	result.parameters = std::move(duck_pl_function_parameters);
	result.macro = std::move(table_macro_definition);
	return result;
}

CreateRoutineDefinition
PEGTransformerFactory::TransformLegacyScalarMacroFunctionDefinition(PEGTransformer &transformer,
                                                                    vector<MacroParameter> duck_pl_function_parameters,
                                                                    unique_ptr<MacroFunction> scalar_macro_definition) {
	ValidateRoutineParameters(duck_pl_function_parameters, "macro");

	CreateRoutineDefinition result;
	ApplyMacroParameters(*scalar_macro_definition, std::move(duck_pl_function_parameters));
	result.parameters = std::move(duck_pl_function_parameters);
	result.macro = std::move(scalar_macro_definition);
	return result;
}

vector<MacroParameter>
PEGTransformerFactory::TransformDuckPLNonEmptyFunctionParameters(PEGTransformer &transformer,
                                                                 vector<MacroParameter> macro_parameters) {
	return std::move(macro_parameters);
}

vector<MacroParameter> PEGTransformerFactory::TransformEmptyDuckPLFunctionParameters(PEGTransformer &transformer) {
	return {};
}

shared_ptr<DuckPLScriptDefinition>
PEGTransformerFactory::TransformDuckPLWrappedDefinition(PEGTransformer &transformer,
                                                        unique_ptr<DuckPLBlockStatement> duck_pl_wrapped_body,
                                                        const string &duck_pl_language_spec) {
	ValidateDuckPLLanguageName(duck_pl_language_spec);
	return make_shared_ptr<DuckPLScriptDefinition>(std::move(duck_pl_wrapped_body));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLDeclaration(PEGTransformer &transformer, const string &identifier,
                                                  const LogicalType &type,
                                                  unique_ptr<ParsedExpression> duck_pl_declaration_default) {
	return make_uniq<DuckPLDeclarationStatement>(identifier, type, std::move(duck_pl_declaration_default), false,
	                                             false);
}

unique_ptr<ParsedExpression>
PEGTransformerFactory::TransformDuckPLDeclarationDefault(PEGTransformer &transformer,
                                                         unique_ptr<ParsedExpression> expression) {
	return expression;
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLCompoundRecursive(PEGTransformer &transformer,
                                                        unique_ptr<DuckPLParseStatement> duck_pl_compound_statement,
                                                        unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return PrependDuckPLStatement(std::move(duck_pl_compound_statement), std::move(duck_pl_statements));
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLSimpleRecursive(PEGTransformer &transformer,
                                                      unique_ptr<DuckPLParseStatement> duck_pl_simple_statement,
                                                      unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return PrependDuckPLStatement(std::move(duck_pl_simple_statement), std::move(duck_pl_statements));
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLCompoundTerminal(PEGTransformer &transformer,
                                                       unique_ptr<DuckPLParseStatement> duck_pl_compound_statement) {
	return PrependDuckPLStatement(std::move(duck_pl_compound_statement), nullptr);
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLSimpleTerminal(PEGTransformer &transformer,
                                                     unique_ptr<DuckPLParseStatement> duck_pl_simple_statement) {
	return PrependDuckPLStatement(std::move(duck_pl_simple_statement), nullptr);
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLAssignment(PEGTransformer &transformer, const string &identifier,
                                                 unique_ptr<ParsedExpression> expression) {
	return make_uniq<DuckPLAssignmentStatement>(identifier, std::move(expression));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLReturnNext(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression) {
	return make_uniq<DuckPLReturnNextStatement>(std::move(expression));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLReturnQuery(PEGTransformer &transformer,
                                                  unique_ptr<SelectStatement> select_statement_internal) {
	return make_uniq<DuckPLReturnQueryStatement>(std::move(select_statement_internal));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLReturnExpr(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression) {
	return make_uniq<DuckPLReturnStatement>(std::move(expression));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLIfClassic(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression,
                                                unique_ptr<DuckPLBlockStatement> duck_pl_statements,
                                                vector<DuckPLIfBranch> duck_pl_classic_else_if_clause,
                                                unique_ptr<DuckPLBlockStatement> duck_pl_classic_else_clause) {
	return make_uniq<DuckPLIfStatement>(std::move(expression), std::move(duck_pl_statements),
	                                    std::move(duck_pl_classic_else_if_clause),
	                                    std::move(duck_pl_classic_else_clause));
}

DuckPLIfBranch
PEGTransformerFactory::TransformDuckPLClassicElseIfClause(PEGTransformer &transformer,
                                                          unique_ptr<ParsedExpression> expression,
                                                          unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return DuckPLIfBranch(std::move(expression), std::move(duck_pl_statements));
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLClassicElseClause(PEGTransformer &transformer,
                                                        unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return duck_pl_statements;
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLIfBrace(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression,
                                              unique_ptr<DuckPLBlockStatement> duck_pl_brace_block,
                                              vector<DuckPLIfBranch> duck_pl_brace_else_if_clause,
                                              unique_ptr<DuckPLBlockStatement> duck_pl_brace_else_clause) {
	return make_uniq<DuckPLIfStatement>(std::move(expression), std::move(duck_pl_brace_block),
	                                    std::move(duck_pl_brace_else_if_clause), std::move(duck_pl_brace_else_clause));
}

DuckPLIfBranch
PEGTransformerFactory::TransformDuckPLBraceElseIfClause(PEGTransformer &transformer,
                                                        unique_ptr<ParsedExpression> expression,
                                                        unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return DuckPLIfBranch(std::move(expression), std::move(duck_pl_brace_block));
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLBraceElseClause(PEGTransformer &transformer,
                                                      unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return duck_pl_brace_block;
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLLoopClassic(PEGTransformer &transformer,
                                                  unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return make_uniq<DuckPLLoopStatement>(std::move(duck_pl_statements));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLLoopBrace(PEGTransformer &transformer,
                                                unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return make_uniq<DuckPLLoopStatement>(std::move(duck_pl_brace_block));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLWhileClassic(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression,
                                                   unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return make_uniq<DuckPLWhileStatement>(std::move(expression), std::move(duck_pl_statements));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLWhileBrace(PEGTransformer &transformer, unique_ptr<ParsedExpression> expression,
                                                 unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return make_uniq<DuckPLWhileStatement>(std::move(expression), std::move(duck_pl_brace_block));
}

unique_ptr<DuckPLParseStatement> PEGTransformerFactory::TransformDuckPLForClassic(
    PEGTransformer &transformer, const string &identifier, const bool &duck_pl_reverse,
    unique_ptr<ParsedExpression> expression, unique_ptr<ParsedExpression> expression_1,
    unique_ptr<ParsedExpression> duck_pl_for_step_clause, unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return make_uniq<DuckPLForStatement>(identifier, std::move(expression), std::move(expression_1),
	                                     std::move(duck_pl_statements), duck_pl_reverse,
	                                     std::move(duck_pl_for_step_clause));
}

unique_ptr<DuckPLParseStatement> PEGTransformerFactory::TransformDuckPLForBrace(
    PEGTransformer &transformer, const string &identifier, const bool &duck_pl_reverse,
    unique_ptr<ParsedExpression> expression, unique_ptr<ParsedExpression> expression_1,
    unique_ptr<ParsedExpression> duck_pl_for_step_clause, unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return make_uniq<DuckPLForStatement>(identifier, std::move(expression), std::move(expression_1),
	                                     std::move(duck_pl_brace_block), duck_pl_reverse,
	                                     std::move(duck_pl_for_step_clause));
}

bool PEGTransformerFactory::TransformDuckPLReverse(PEGTransformer &transformer) {
	return true;
}

unique_ptr<ParsedExpression>
PEGTransformerFactory::TransformDuckPLForStepClause(PEGTransformer &transformer,
                                                    unique_ptr<ParsedExpression> expression) {
	return expression;
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLForeachClassic(PEGTransformer &transformer, const string &identifier,
                                                     unique_ptr<ParsedExpression> expression,
                                                     unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return make_uniq<DuckPLForeachStatement>(identifier, std::move(expression), std::move(duck_pl_statements));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLForeachBrace(PEGTransformer &transformer, const string &identifier,
                                                   unique_ptr<ParsedExpression> expression,
                                                   unique_ptr<DuckPLBlockStatement> duck_pl_brace_block) {
	return make_uniq<DuckPLForeachStatement>(identifier, std::move(expression), std::move(duck_pl_brace_block));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLContinue(PEGTransformer &transformer, const string &identifier,
                                               unique_ptr<ParsedExpression> duck_pl_when_clause) {
	return make_uniq<DuckPLContinueStatement>(identifier, std::move(duck_pl_when_clause));
}

unique_ptr<DuckPLParseStatement>
PEGTransformerFactory::TransformDuckPLExit(PEGTransformer &transformer, const string &identifier,
                                           unique_ptr<ParsedExpression> duck_pl_when_clause) {
	return make_uniq<DuckPLExitStatement>(identifier, std::move(duck_pl_when_clause));
}

unique_ptr<ParsedExpression> PEGTransformerFactory::TransformDuckPLWhenClause(PEGTransformer &transformer,
                                                                              unique_ptr<ParsedExpression> expression) {
	return expression;
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLBeginEndBlock(PEGTransformer &transformer,
                                                    unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return duck_pl_statements;
}

unique_ptr<DuckPLBlockStatement>
PEGTransformerFactory::TransformDuckPLBraceBlock(PEGTransformer &transformer,
                                                 unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return duck_pl_statements;
}

unique_ptr<SQLStatement>
PEGTransformerFactory::TransformDuckPLScriptStatement(PEGTransformer &transformer,
                                                      unique_ptr<DuckPLBlockStatement> duck_pl_statements) {
	return make_uniq<DuckPLScriptStatement>(make_shared_ptr<DuckPLScriptDefinition>(std::move(duck_pl_statements)));
}

string PEGTransformerFactory::TransformDuckPLLanguageSpec(PEGTransformer &transformer, const string &duck_pl_language) {
	ValidateDuckPLLanguageName(duck_pl_language);
	return duck_pl_language;
}

string PEGTransformerFactory::TransformDuckPLDuckLanguage(PEGTransformer &transformer) {
	return "DUCKPL";
}

string PEGTransformerFactory::TransformDuckPLPlpgsqlLanguage(PEGTransformer &transformer) {
	return "PLPGSQL";
}

LogicalType PEGTransformerFactory::TransformDuckPLReturns(PEGTransformer &transformer, const LogicalType &type) {
	return type;
}

} // namespace duckdb
