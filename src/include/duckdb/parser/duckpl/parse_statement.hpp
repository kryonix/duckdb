//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/duckpl/parse_statement.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/parser/duckpl/parse_node.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/statement/select_statement.hpp"

namespace duckdb {

class DuckPLBlockStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::BLOCK;

public:
	explicit DuckPLBlockStatement(optional_idx location_p = optional_idx()) : DuckPLParseStatement(TYPE, location_p) {
	}

	explicit DuckPLBlockStatement(vector<unique_ptr<DuckPLParseStatement>> statements_p,
	                              optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), statements(std::move(statements_p)) {
	}

public:
	vector<unique_ptr<DuckPLParseStatement>> statements;
};

class DuckPLDeclarationStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::DECLARE;

public:
	explicit DuckPLDeclarationStatement(string name_p, LogicalType type_p = LogicalType::UNKNOWN,
	                                    unique_ptr<ParsedExpression> default_value_p = nullptr,
	                                    bool is_constant_p = false, bool not_null_p = false,
	                                    optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), name(std::move(name_p)), type(std::move(type_p)),
	      default_value(std::move(default_value_p)), is_constant(is_constant_p), not_null(not_null_p) {
	}

public:
	string name;
	LogicalType type;
	unique_ptr<ParsedExpression> default_value;
	bool is_constant;
	bool not_null;
};

struct DuckPLIfBranch : public DuckPLParseNode {
public:
	explicit DuckPLIfBranch(unique_ptr<ParsedExpression> condition_p, unique_ptr<DuckPLBlockStatement> block_p,
	                        optional_idx location_p = optional_idx())
	    : DuckPLParseNode(location_p), condition(std::move(condition_p)), block(std::move(block_p)) {
	}

public:
	unique_ptr<ParsedExpression> condition;
	unique_ptr<DuckPLBlockStatement> block;
};

class DuckPLAssignmentStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::ASSIGNMENT;

public:
	explicit DuckPLAssignmentStatement(string target_name_p, unique_ptr<ParsedExpression> value_p,
	                                   optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), target_name(std::move(target_name_p)), value(std::move(value_p)) {
	}

public:
	string target_name;
	unique_ptr<ParsedExpression> value;
};

class DuckPLReturnStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::RETURN;

public:
	explicit DuckPLReturnStatement(unique_ptr<ParsedExpression> expression_p, optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), expression(std::move(expression_p)) {
	}

public:
	unique_ptr<ParsedExpression> expression;
};

class DuckPLReturnNextStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::RETURN_NEXT;

public:
	explicit DuckPLReturnNextStatement(unique_ptr<ParsedExpression> expression_p,
	                                   optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), expression(std::move(expression_p)) {
	}

public:
	unique_ptr<ParsedExpression> expression;
};

class DuckPLReturnQueryStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::RETURN_QUERY;

public:
	explicit DuckPLReturnQueryStatement(unique_ptr<SelectStatement> query_p, optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), query(std::move(query_p)) {
	}

public:
	unique_ptr<SelectStatement> query;
};

class DuckPLIfStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::IF;

public:
	explicit DuckPLIfStatement(unique_ptr<ParsedExpression> condition_p, unique_ptr<DuckPLBlockStatement> then_branch_p,
	                           vector<DuckPLIfBranch> elseif_branches_p = {},
	                           unique_ptr<DuckPLBlockStatement> else_branch_p = nullptr,
	                           optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), condition(std::move(condition_p)),
	      then_branch(std::move(then_branch_p)), elseif_branches(std::move(elseif_branches_p)),
	      else_branch(std::move(else_branch_p)) {
	}

public:
	unique_ptr<ParsedExpression> condition;
	unique_ptr<DuckPLBlockStatement> then_branch;
	vector<DuckPLIfBranch> elseif_branches;
	unique_ptr<DuckPLBlockStatement> else_branch;
};

class DuckPLLoopStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::LOOP;

public:
	explicit DuckPLLoopStatement(unique_ptr<DuckPLBlockStatement> body_p, string label_p = string(),
	                             optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), body(std::move(body_p)), label(std::move(label_p)) {
	}

public:
	unique_ptr<DuckPLBlockStatement> body;
	string label;
};

class DuckPLWhileStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::WHILE;

public:
	explicit DuckPLWhileStatement(unique_ptr<ParsedExpression> condition_p, unique_ptr<DuckPLBlockStatement> body_p,
	                              string label_p = string(), optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), condition(std::move(condition_p)), body(std::move(body_p)),
	      label(std::move(label_p)) {
	}

public:
	unique_ptr<ParsedExpression> condition;
	unique_ptr<DuckPLBlockStatement> body;
	string label;
};

class DuckPLForStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::FOR;

public:
	explicit DuckPLForStatement(string target_name_p, unique_ptr<ParsedExpression> start_p,
	                            unique_ptr<ParsedExpression> end_p, unique_ptr<DuckPLBlockStatement> body_p,
	                            bool reverse_p = false, unique_ptr<ParsedExpression> step_p = nullptr,
	                            string label_p = string(), optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), target_name(std::move(target_name_p)), start(std::move(start_p)),
	      end(std::move(end_p)), step(std::move(step_p)), body(std::move(body_p)), reverse(reverse_p),
	      label(std::move(label_p)) {
	}

public:
	string target_name;
	unique_ptr<ParsedExpression> start;
	unique_ptr<ParsedExpression> end;
	unique_ptr<ParsedExpression> step;
	unique_ptr<DuckPLBlockStatement> body;
	bool reverse;
	string label;
};

class DuckPLForeachStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::FOREACH;

public:
	explicit DuckPLForeachStatement(string target_name_p, unique_ptr<ParsedExpression> array_expression_p,
	                                unique_ptr<DuckPLBlockStatement> body_p, string label_p = string(),
	                                optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), target_name(std::move(target_name_p)),
	      array_expression(std::move(array_expression_p)), body(std::move(body_p)), label(std::move(label_p)) {
	}

public:
	string target_name;
	unique_ptr<ParsedExpression> array_expression;
	unique_ptr<DuckPLBlockStatement> body;
	string label;
};

class DuckPLContinueStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::CONTINUE;

public:
	explicit DuckPLContinueStatement(string label_p = string(), unique_ptr<ParsedExpression> when_condition_p = nullptr,
	                                 optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), label(std::move(label_p)),
	      when_condition(std::move(when_condition_p)) {
	}

public:
	string label;
	unique_ptr<ParsedExpression> when_condition;
};

class DuckPLExitStatement : public DuckPLParseStatement {
public:
	static constexpr DuckPLParseStatementType TYPE = DuckPLParseStatementType::EXIT;

public:
	explicit DuckPLExitStatement(string label_p = string(), unique_ptr<ParsedExpression> when_condition_p = nullptr,
	                             optional_idx location_p = optional_idx())
	    : DuckPLParseStatement(TYPE, location_p), label(std::move(label_p)),
	      when_condition(std::move(when_condition_p)) {
	}

public:
	string label;
	unique_ptr<ParsedExpression> when_condition;
};

} // namespace duckdb
