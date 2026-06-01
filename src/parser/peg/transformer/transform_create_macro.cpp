#include "duckdb/parser/peg/ast/macro_parameter.hpp"
#include "duckdb/function/table_macro_function.hpp"
#include "duckdb/parser/peg/ast/create_routine_definition.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/parser/peg/transformer/peg_transformer.hpp"
#include "duckdb/function/scalar_macro_function.hpp"

namespace duckdb {

static ParseResult &UnwrapChoice(ParseResult &parse_result) {
	if (parse_result.type == ParseResultType::CHOICE) {
		return parse_result.Cast<ChoiceParseResult>().GetResult();
	}
	return parse_result;
}

static ParseResult *FindNamedMacroParseResult(ParseResult &parse_result) {
	if (!parse_result.name.empty() && parse_result.name != "MacroDefinition") {
		return &parse_result;
	}
	if (parse_result.type == ParseResultType::CHOICE) {
		auto &choice = parse_result.Cast<ChoiceParseResult>();
		return FindNamedMacroParseResult(choice.GetResult());
	}
	if (parse_result.type == ParseResultType::LIST) {
		auto &list = parse_result.Cast<ListParseResult>();
		for (auto child_ref : list.GetChildren()) {
			auto &child = child_ref.get();
			if (child.type == ParseResultType::KEYWORD) {
				continue;
			}
			auto result = FindNamedMacroParseResult(child);
			if (result) {
				return result;
			}
		}
	}
	return nullptr;
}

static ParseResult &FindMacroDefinitionParseResult(ParseResult &parse_result) {
	auto result = FindNamedMacroParseResult(parse_result);
	if (!result) {
		throw InternalException("Failed to locate macro definition parse result");
	}
	return *result;
}

unique_ptr<CreateStatement> PEGTransformerFactory::TransformCreateMacroStmt(PEGTransformer &transformer,
                                                                            ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto result = make_uniq<CreateStatement>();
	auto if_not_exists = list_pr.Child<OptionalParseResult>(1).HasResult();
	auto qualified_name = transformer.Transform<QualifiedName>(list_pr.Child<ListParseResult>(2));
	auto macro_definition_list = ExtractParseResultsFromList(list_pr.Child<ListParseResult>(3));

	vector<CreateRoutineDefinition> routine_definitions;
	for (auto macro_definition : macro_definition_list) {
		routine_definitions.push_back(transformer.Transform<CreateRoutineDefinition>(macro_definition));
	}

	D_ASSERT(!routine_definitions.empty());
	auto info = make_uniq<CreateMacroInfo>(CatalogType::MACRO_ENTRY);
	if (qualified_name.schema.empty()) {
		info->schema = qualified_name.catalog;
	} else {
		info->catalog = qualified_name.catalog;
		info->schema = qualified_name.schema;
	}
	info->name = qualified_name.name;
	info->on_conflict = if_not_exists ? OnCreateConflict::IGNORE_ON_CONFLICT : OnCreateConflict::ERROR_ON_CONFLICT;

	for (auto &routine_definition : routine_definitions) {
		if (!routine_definition.macro) {
			throw ParserException("Expected macro definition");
		}
		info->macros.push_back(std::move(routine_definition.macro));
	}

	auto macro_type = info->macros[0]->type;
	if (info->macros.size() > 1) {
		for (idx_t i = 1; i < info->macros.size(); ++i) {
			if (info->macros[i]->type != macro_type) {
				throw ParserException("Cannot mix table and scalar macro function definitions");
			}
		}
	}
	info->type = macro_type == MacroType::TABLE_MACRO ? CatalogType::TABLE_MACRO_ENTRY : CatalogType::MACRO_ENTRY;
	result->info = std::move(info);
	transformer.PivotEntryCheck("macro");
	return result;
}

CreateRoutineDefinition PEGTransformerFactory::TransformMacroDefinition(PEGTransformer &transformer,
                                                                        ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	CreateRoutineDefinition result;
	auto &parameters_pr = ExtractResultFromParens(list_pr.Child<ListParseResult>(0));
	bool default_value_found = false;
	vector<MacroParameter> parameters;
	if (parameters_pr.type == ParseResultType::OPTIONAL) {
		auto &optional_parameters = parameters_pr.Cast<OptionalParseResult>();
		if (optional_parameters.HasResult()) {
			parameters = transformer.Transform<vector<MacroParameter>>(optional_parameters.GetResult());
		}
	} else {
		parameters = transformer.Transform<vector<MacroParameter>>(parameters_pr);
	}
	case_insensitive_string_set_t parameter_names;
	for (auto &parameter : parameters) {
		D_ASSERT(!parameter.name.empty());
		if (parameter_names.find(parameter.name) != parameter_names.end()) {
			throw ParserException("Duplicate parameter '%s' in macro definition", parameter.name);
		}
		parameter_names.insert(parameter.name);
		if (!parameter.is_default && default_value_found) {
			throw ParserException("Parameter without a default follows parameter with a default");
		}
		default_value_found = default_value_found || parameter.is_default;
		result.parameters.push_back(std::move(parameter));
	}

	auto &definition_pr = FindMacroDefinitionParseResult(list_pr.GetChild(2));
	auto macro_function = transformer.Transform<unique_ptr<MacroFunction>>(definition_pr);
	for (auto &parameter : result.parameters) {
		if (parameter.is_default) {
			auto default_expr = std::move(parameter.expression);
			default_expr->SetAlias(parameter.name);
			macro_function->default_parameters[parameter.name] = std::move(default_expr);
			macro_function->parameters.push_back(make_uniq<ColumnRefExpression>(parameter.name));
		} else {
			macro_function->parameters.push_back(std::move(parameter.expression));
		}
		macro_function->types.push_back(parameter.type);
	}
	result.macro = std::move(macro_function);
	return result;
}

unique_ptr<MacroFunction> PEGTransformerFactory::TransformTableMacroDefinition(PEGTransformer &transformer,
                                                                               ParseResult &parse_result) {
	auto &definition_pr = UnwrapChoice(parse_result);
	auto &list_pr = definition_pr.Cast<ListParseResult>();
	auto result = make_uniq<TableMacroFunction>();
	auto select_statement = transformer.Transform<unique_ptr<SelectStatement>>(list_pr.Child<ListParseResult>(1));
	result->query_node = std::move(select_statement->node);
	return std::move(result);
}

unique_ptr<MacroFunction> PEGTransformerFactory::TransformScalarMacroDefinition(PEGTransformer &transformer,
                                                                                ParseResult &parse_result) {
	auto &definition_pr = UnwrapChoice(parse_result);
	auto &list_pr = definition_pr.Cast<ListParseResult>();
	auto result = make_uniq<ScalarMacroFunction>();
	result->expression = transformer.Transform<unique_ptr<ParsedExpression>>(list_pr.Child<ListParseResult>(0));
	return std::move(result);
}

vector<MacroParameter> PEGTransformerFactory::TransformMacroParameters(PEGTransformer &transformer,
                                                                       ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto parameter_list = ExtractParseResultsFromList(list_pr.Child<ListParseResult>(0));
	vector<MacroParameter> parameters;
	for (auto parameter : parameter_list) {
		parameters.push_back(transformer.Transform<MacroParameter>(parameter));
	}
	return parameters;
}

MacroParameter PEGTransformerFactory::TransformMacroParameter(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto &choice_pr = UnwrapChoice(list_pr.GetChild(0));
	return transformer.Transform<MacroParameter>(choice_pr);
}

MacroParameter PEGTransformerFactory::TransformSimpleParameter(PEGTransformer &transformer, ParseResult &parse_result) {
	auto &list_pr = parse_result.Cast<ListParseResult>();
	auto parameter = transformer.Transform<string>(list_pr.Child<ListParseResult>(0));
	MacroParameter result;
	result.name = parameter;
	result.expression = make_uniq<ColumnRefExpression>(parameter);
	transformer.TransformOptional<LogicalType>(list_pr, 1, result.type);
	result.is_default = false;
	return result;
}

} // namespace duckdb
