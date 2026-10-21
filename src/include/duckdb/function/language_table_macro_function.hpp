//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/language_table_macro_function.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/language_macro_function.hpp"

namespace duckdb {

//! The table-valued variant of a language member: next to the language and verbatim source it
//! carries the declared RETURNS TABLE columns, the fixed result shape of every call.
class LanguageTableMacroFunction : public LanguageMacroFunction {
public:
	static constexpr const MacroType TYPE = MacroType::LANGUAGE_TABLE_MACRO;

public:
	LanguageTableMacroFunction(string language, string source);
	LanguageTableMacroFunction();

	//! The declared or published result schema, empty while unresolved: an unresolved member binds
	//! its schema at each call from the statement the language binder returns
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
