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
	result->language_version = language_version;
	CopyProperties(*result);
	result->return_names = return_names;
	result->return_types = return_types;
	return std::move(result);
}

string LanguageTableMacroFunction::ColumnsToSQL() const {
	D_ASSERT(return_names.size() == return_types.size());
	vector<string> columns;
	for (idx_t col_idx = 0; col_idx < return_names.size(); col_idx++) {
		columns.push_back(SQLIdentifier::ToString(return_names[col_idx]) + " " + TypeToSQL(return_types[col_idx]));
	}
	return StringUtil::Join(columns, ", ");
}

string LanguageTableMacroFunction::ToSQL() const {
	// an unresolved schema renders as the bare RETURNS TABLE form
	string returns_table = return_names.empty() ? " RETURNS TABLE" : " RETURNS TABLE(" + ColumnsToSQL() + ")";
	return ParametersToSQL() + returns_table + " LANGUAGE " + KeywordHelper::WriteOptionallyQuoted(language) + " AS " +
	       KeywordHelper::WriteQuoted(source);
}

} // namespace duckdb
