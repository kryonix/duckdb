#pragma once

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/function/macro_function.hpp"
#include "duckdb/parser/duckpl/parse_definition.hpp"
#include "duckdb/parser/peg/ast/macro_parameter.hpp"

namespace duckdb {

struct CreateRoutineDefinition {
	vector<MacroParameter> parameters;
	bool has_return_type = false;
	LogicalType return_type = LogicalType::UNKNOWN;
	unique_ptr<MacroFunction> macro;
	shared_ptr<DuckPLScriptDefinition> duckpl;

	bool IsDuckPL() const {
		return duckpl != nullptr;
	}
};

} // namespace duckdb
