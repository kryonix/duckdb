//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/duckpl/parse_node.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {

enum class DuckPLParseStatementType : uint8_t {
	INVALID = 0,
	BLOCK,
	DECLARE,
	ASSIGNMENT,
	RETURN,
	RETURN_NEXT,
	RETURN_QUERY,
	IF,
	LOOP,
	WHILE,
	FOR,
	FOREACH,
	CONTINUE,
	EXIT
};

enum class DuckPLParseDefinitionType : uint8_t { SCRIPT = 0, FUNCTION };

class DuckPLParseNode {
public:
	explicit DuckPLParseNode(optional_idx location_p = optional_idx()) : location(location_p) {
	}
	virtual ~DuckPLParseNode() = default;

public:
	optional_idx location;
};

class DuckPLParseStatement : public DuckPLParseNode {
public:
	explicit DuckPLParseStatement(DuckPLParseStatementType type_p, optional_idx location_p = optional_idx())
	    : DuckPLParseNode(location_p), type(type_p) {
	}
	virtual ~DuckPLParseStatement() = default;

public:
	DuckPLParseStatementType type;

public:
	template <class TARGET>
	TARGET &Cast() {
		if (TARGET::TYPE != DuckPLParseStatementType::INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast DuckPL parse statement - type mismatch");
		}
		return reinterpret_cast<TARGET &>(*this);
	}

	template <class TARGET>
	const TARGET &Cast() const {
		if (TARGET::TYPE != DuckPLParseStatementType::INVALID && type != TARGET::TYPE) {
			throw InternalException("Failed to cast DuckPL parse statement - type mismatch");
		}
		return reinterpret_cast<const TARGET &>(*this);
	}
};

class DuckPLParseDefinition : public DuckPLParseNode {
public:
	explicit DuckPLParseDefinition(DuckPLParseDefinitionType type_p, optional_idx location_p = optional_idx())
	    : DuckPLParseNode(location_p), type(type_p) {
	}
	virtual ~DuckPLParseDefinition() = default;

public:
	DuckPLParseDefinitionType type;

public:
	template <class TARGET>
	TARGET &Cast() {
		if (TARGET::TYPE != type) {
			throw InternalException("Failed to cast DuckPL parse definition - type mismatch");
		}
		return reinterpret_cast<TARGET &>(*this);
	}

	template <class TARGET>
	const TARGET &Cast() const {
		if (TARGET::TYPE != type) {
			throw InternalException("Failed to cast DuckPL parse definition - type mismatch");
		}
		return reinterpret_cast<const TARGET &>(*this);
	}
};

} // namespace duckdb
