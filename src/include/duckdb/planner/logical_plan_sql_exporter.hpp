//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/logical_plan_sql_exporter.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/identifier.hpp"
#include "duckdb/common/optional.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/logical_plan_verification_result.hpp"

#include <functional>

namespace duckdb {

class ClientContext;
class LogicalOperator;
struct LogicalExtensionOperator;

struct LogicalPlanSQLExportField {
	ColumnBinding source_binding;
	LogicalType type;
	Identifier identifier;
	idx_t ordinal;
};

struct LogicalPlanSQLExportRelation {
	unique_ptr<QueryNode> query;
	vector<LogicalPlanSQLExportField> fields;
};

enum class LogicalPlanSQLExportExtensionResultType { NOT_HANDLED, EXPORTED, UNSUPPORTED };

struct LogicalPlanSQLExportExtensionResult {
	LogicalPlanSQLExportExtensionResultType type = LogicalPlanSQLExportExtensionResultType::NOT_HANDLED;
	optional<LogicalPlanSQLExportRelation> relation;
	string reason;

	DUCKDB_API static LogicalPlanSQLExportExtensionResult NotHandled();
	DUCKDB_API static LogicalPlanSQLExportExtensionResult Exported(LogicalPlanSQLExportRelation relation);
	DUCKDB_API static LogicalPlanSQLExportExtensionResult Unsupported(string reason);
};

struct LogicalPlanSQLExportChild {
	LogicalPlanSQLExportChild(const LogicalPlanSQLExportRelation &relation_p, Identifier relation_alias_p)
	    : relation(relation_p), relation_alias(std::move(relation_alias_p)) {
	}

	const LogicalPlanSQLExportRelation &relation;
	Identifier relation_alias;
};

using LogicalPlanSQLExpressionExporter =
    std::function<LogicalPlanVerificationResult<unique_ptr<ParsedExpression>>(idx_t expression_ordinal)>;

struct LogicalPlanSQLExportExtensionInput {
	LogicalPlanSQLExportExtensionInput(const LogicalExtensionOperator &op_p,
	                                   const vector<LogicalPlanSQLExportChild> &children_p,
	                                   const vector<LogicalPlanSQLExportField> &output_fields_p,
	                                   idx_t expression_count_p, LogicalPlanSQLExpressionExporter export_expression_p)
	    : op(op_p), children(children_p), output_fields(output_fields_p), expression_count(expression_count_p),
	      export_expression(std::move(export_expression_p)) {
	}

	const LogicalExtensionOperator &op;
	const vector<LogicalPlanSQLExportChild> &children;
	const vector<LogicalPlanSQLExportField> &output_fields;
	const idx_t expression_count;
	const LogicalPlanSQLExpressionExporter export_expression;
};

using logical_plan_sql_export_t =
    std::function<LogicalPlanSQLExportExtensionResult(const LogicalPlanSQLExportExtensionInput &input)>;

struct LogicalPlanSQLExportResolver {
	string identifier;
	logical_plan_sql_export_t callback;
};

struct LogicalPlanSQLExportOptions {
	vector<LogicalPlanSQLExportResolver> extension_resolvers;
};

class LogicalPlanSQLExporter {
public:
	DUCKDB_API static LogicalPlanVerificationResult<LogicalPlanSQLExportRelation>
	Export(ClientContext &context, LogicalOperator &root, const LogicalPlanSQLExportOptions &options = {});
};

} // namespace duckdb
