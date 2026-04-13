//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/duckpl/parse_definition.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/parser/function_parameter.hpp"
#include "duckdb/parser/duckpl/parse_statement.hpp"

namespace duckdb {

class DuckPLScriptDefinition : public DuckPLParseDefinition {
public:
	static constexpr DuckPLParseDefinitionType TYPE = DuckPLParseDefinitionType::SCRIPT;

public:
	explicit DuckPLScriptDefinition(unique_ptr<DuckPLBlockStatement> body_p, optional_idx location_p = optional_idx())
	    : DuckPLParseDefinition(TYPE, location_p), body(std::move(body_p)) {
	}

public:
	unique_ptr<DuckPLBlockStatement> body;
};

class DuckPLFunctionDefinition : public DuckPLParseDefinition {
public:
	static constexpr DuckPLParseDefinitionType TYPE = DuckPLParseDefinitionType::FUNCTION;

public:
	explicit DuckPLFunctionDefinition(string name_p, vector<FunctionParameter> parameters_p, bool returns_table_p,
	                                  LogicalType return_type_p, unique_ptr<DuckPLBlockStatement> body_p,
	                                  optional_idx location_p = optional_idx())
	    : DuckPLParseDefinition(TYPE, location_p), name(std::move(name_p)), parameters(std::move(parameters_p)),
	      returns_table(returns_table_p), return_type(std::move(return_type_p)), body(std::move(body_p)) {
	}

public:
	string name;
	vector<FunctionParameter> parameters;
	bool returns_table;
	LogicalType return_type;
	unique_ptr<DuckPLBlockStatement> body;
};

} // namespace duckdb
