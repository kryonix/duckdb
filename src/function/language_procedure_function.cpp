#include "duckdb/function/language_procedure_function.hpp"

#include "duckdb/common/sql_identifier.hpp"

namespace duckdb {

LanguageProcedureFunction::LanguageProcedureFunction(string language_p, string source_p)
    : LanguageMacroFunction(MacroType::LANGUAGE_PROCEDURE_MACRO) {
	language = std::move(language_p);
	source = std::move(source_p);
}

LanguageProcedureFunction::LanguageProcedureFunction() : LanguageMacroFunction(MacroType::LANGUAGE_PROCEDURE_MACRO) {
}

unique_ptr<MacroFunction> LanguageProcedureFunction::Copy() const {
	auto result = make_uniq<LanguageProcedureFunction>(language, source);
	result->language_version = language_version;
	CopyProperties(*result);
	result->return_names = return_names;
	result->return_types = return_types;
	return std::move(result);
}

string LanguageProcedureFunction::ColumnsToSQL() const {
	D_ASSERT(return_names.size() == return_types.size());
	vector<string> columns;
	for (idx_t col_idx = 0; col_idx < return_names.size(); col_idx++) {
		columns.push_back(SQLIdentifier::ToString(return_names[col_idx]) + " " + TypeToSQL(return_types[col_idx]));
	}
	return StringUtil::Join(columns, ", ");
}

string LanguageProcedureFunction::ToSQL() const {
	// an empty schema is the no-result procedure, rendered without a RETURNS clause
	string returns_table = return_names.empty() ? "" : " RETURNS TABLE(" + ColumnsToSQL() + ")";
	return ParametersToSQL() + returns_table + " LANGUAGE " + SQLIdentifier::ToString(language) + " AS " +
	       SQLString::ToString(source);
}

} // namespace duckdb
