//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/language_procedure_function.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/language_macro_function.hpp"

namespace duckdb {

//! The procedure variant of a language member: a multi-statement routine the language expands at
//! CALL. Next to the language and verbatim source it carries the declared RETURNS TABLE columns;
//! an empty schema means the procedure yields nothing.
class LanguageProcedureFunction : public LanguageMacroFunction {
public:
	static constexpr const MacroType TYPE = MacroType::LANGUAGE_PROCEDURE_MACRO;

public:
	LanguageProcedureFunction(string language, string source);
	LanguageProcedureFunction();

	//! The declared or published result schema, empty when the procedure yields nothing
	vector<Identifier> return_names;
	//! The result column types, parallel to return_names
	vector<LogicalType> return_types;

public:
	unique_ptr<MacroFunction> Copy() const override;

	string ToSQL() const override;
	//! The declared columns as "name TYPE, ...", the RETURNS TABLE list without the parentheses
	string ColumnsToSQL() const;

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<MacroFunction> Deserialize(Deserializer &deserializer);
};

} // namespace duckdb
