#include "duckdb/function/language_table_macro_function.hpp"

#include "duckdb/parser/keyword_helper.hpp"

namespace duckdb {

LanguageTableMacroFunction::LanguageTableMacroFunction(string language_p, string source_p)
    : LanguageMacroFunction(MacroType::LANGUAGE_TABLE_MACRO) {
	language = std::move(language_p);
	source = std::move(source_p);
}

LanguageTableMacroFunction::LanguageTableMacroFunction() : LanguageMacroFunction(MacroType::LANGUAGE_TABLE_MACRO) {
}

unique_ptr<MacroFunction> LanguageTableMacroFunction::Copy() const {
	auto result = make_uniq<LanguageTableMacroFunction>(language, source);
	CopyProperties(*result);
	result->column_names = column_names;
	result->column_types = column_types;
	return std::move(result);
}

string LanguageTableMacroFunction::ColumnsToSQL() const {
	D_ASSERT(column_names.size() == column_types.size());
	vector<string> columns;
	for (idx_t col_idx = 0; col_idx < column_names.size(); col_idx++) {
		columns.push_back(SQLIdentifier::ToString(column_names[col_idx]) + " " + TypeToSQL(column_types[col_idx]));
	}
	return StringUtil::Join(columns, ", ");
}

string LanguageTableMacroFunction::ToSQL() const {
	return ParametersToSQL() + " RETURNS TABLE(" + ColumnsToSQL() + ") LANGUAGE " +
	       KeywordHelper::WriteOptionallyQuoted(language) + " AS " + KeywordHelper::WriteQuoted(source);
}

} // namespace duckdb
