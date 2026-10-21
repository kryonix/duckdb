#include "duckdb/main/language_function_binder.hpp"

#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/common/exception/binder_exception.hpp"
#include "duckdb/function/language_table_macro_function.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {

unique_ptr<SelectStatement> LanguageFunctionBinder::BindTableCall(Binder &binder, TableMacroCatalogEntry &entry,
                                                                  LanguageTableMacroFunction &member,
                                                                  vector<unique_ptr<ParsedExpression>> arguments) {
	throw BinderException("Table function %s is a %s language function, but the registered %s binder does not "
	                      "implement table function binding",
	                      entry.name, member.language, member.language);
}

} // namespace duckdb
