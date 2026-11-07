//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/statement_preprocessor.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/identifier.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/statement/pragma_statement.hpp"

namespace duckdb {
class ClientContext;
class ClientContextLock;
class SQLStatement;
struct PragmaInfo;

//! The statements one user statement expands into
struct PreprocessedStatement {
	vector<unique_ptr<SQLStatement>> statements;
	//! The procedure a top-level CALL expanded through; empty = no procedure expansion
	string call_procedure;
	//! The procedure's declared result schema; empty = the CALL yields no rows
	vector<Identifier> call_result_names;
	//! The result column types, parallel to call_result_names
	vector<LogicalType> call_result_types;

	bool IsProcedureExpansion() const {
		return !call_procedure.empty();
	}
};

//! Expands a parsed user statement: reparses pragmas and unpacks multi-statements
class StatementPreprocessor {
public:
	explicit StatementPreprocessor(ClientContext &context);

	//! Expands one user statement, in a transaction of its own unless one is active
	PreprocessedStatement PreprocessStatement(ClientContextLock &lock, unique_ptr<SQLStatement> statement);
	//! Expands one user statement, for a caller that already runs inside a transaction
	PreprocessedStatement ExpandStatement(unique_ptr<SQLStatement> statement) const;

private:
	ClientContext &context;

private:
	//! Handles a pragma statement, determines whether the statement needs reparsing, if it does, it returns the
	//! statement(s) to replace the current one. Otherwise, it just returns back the original statement in a vector.
	vector<unique_ptr<SQLStatement>> TryReparsePragma(unique_ptr<SQLStatement> statement) const;
	//! Expands a CALL of a procedure into its statement sequence; a non-procedure CALL passes through
	void TryExpandCall(unique_ptr<SQLStatement> statement, PreprocessedStatement &result) const;
};

} // namespace duckdb
