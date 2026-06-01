//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/statement/duckpl_script_statement.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/parser/duckpl/parse_definition.hpp"
#include "duckdb/parser/sql_statement.hpp"

namespace duckdb {

class DuckPLScriptStatement : public SQLStatement {
public:
	static constexpr const StatementType TYPE = StatementType::DUCKPL_STATEMENT;

public:
	DuckPLScriptStatement();
	explicit DuckPLScriptStatement(shared_ptr<DuckPLScriptDefinition> script_p);

public:
	shared_ptr<DuckPLScriptDefinition> script;

protected:
	DuckPLScriptStatement(const DuckPLScriptStatement &other);

public:
	unique_ptr<SQLStatement> Copy() const override;
	string ToString() const override;
};

} // namespace duckdb
