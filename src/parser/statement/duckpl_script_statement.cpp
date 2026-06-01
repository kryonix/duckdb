#include "duckdb/parser/statement/duckpl_script_statement.hpp"

namespace duckdb {

DuckPLScriptStatement::DuckPLScriptStatement() : SQLStatement(StatementType::DUCKPL_STATEMENT) {
}

DuckPLScriptStatement::DuckPLScriptStatement(shared_ptr<DuckPLScriptDefinition> script_p)
    : SQLStatement(StatementType::DUCKPL_STATEMENT), script(std::move(script_p)) {
}

DuckPLScriptStatement::DuckPLScriptStatement(const DuckPLScriptStatement &other)
    : SQLStatement(other), script(other.script) {
}

unique_ptr<SQLStatement> DuckPLScriptStatement::Copy() const {
	return unique_ptr<SQLStatement>(new DuckPLScriptStatement(*this));
}

string DuckPLScriptStatement::ToString() const {
	return query.empty() ? "DUCKPL" : query;
}

} // namespace duckdb
