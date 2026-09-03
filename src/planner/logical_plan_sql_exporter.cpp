#include "duckdb/planner/logical_plan_sql_exporter.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/conjunction_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/expressionlistref.hpp"
#include "duckdb/parser/tableref/subqueryref.hpp"
#include "duckdb/planner/bound_expression_sql_exporter.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/logical_plan_verifier.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator_extension.hpp"

namespace duckdb {

LogicalPlanSQLExportExtensionResult LogicalPlanSQLExportExtensionResult::NotHandled() {
	return LogicalPlanSQLExportExtensionResult();
}

LogicalPlanSQLExportExtensionResult
LogicalPlanSQLExportExtensionResult::Exported(LogicalPlanSQLExportRelation relation_p) {
	LogicalPlanSQLExportExtensionResult result;
	result.type = LogicalPlanSQLExportExtensionResultType::EXPORTED;
	result.relation = std::move(relation_p);
	return result;
}

LogicalPlanSQLExportExtensionResult LogicalPlanSQLExportExtensionResult::Unsupported(string reason_p) {
	LogicalPlanSQLExportExtensionResult result;
	result.type = LogicalPlanSQLExportExtensionResultType::UNSUPPORTED;
	result.reason = std::move(reason_p);
	return result;
}

namespace logical_plan_sql_export {

using LogicalPlanSQLExportResult = LogicalPlanVerificationResult<LogicalPlanSQLExportRelation>;
using LogicalPlanSQLFieldResult = LogicalPlanVerificationResult<vector<LogicalPlanSQLExportField>>;

static LogicalPlanVerificationPath PlanChildPath(const LogicalPlanVerificationPath &path, idx_t child_index) {
	auto child_path = path;
	child_path.components.push_back({LogicalPlanVerificationPathComponentType::OPERATOR_CHILD, child_index});
	return child_path;
}

static LogicalPlanVerificationPath PlanExpressionPath(const LogicalPlanVerificationPath &path, idx_t expression_index) {
	auto expression_path = path;
	expression_path.components.push_back(
	    {LogicalPlanVerificationPathComponentType::OPERATOR_EXPRESSION, expression_index});
	return expression_path;
}

static LogicalPlanSQLExportResult PlanFailure(LogicalPlanVerificationIssue issue) {
	vector<LogicalPlanVerificationIssue> issues;
	issues.push_back(std::move(issue));
	return LogicalPlanSQLExportResult::Failure(std::move(issues));
}

static LogicalPlanSQLFieldResult FieldFailure(LogicalPlanVerificationIssue issue) {
	vector<LogicalPlanVerificationIssue> issues;
	issues.push_back(std::move(issue));
	return LogicalPlanSQLFieldResult::Failure(std::move(issues));
}

static bool IsValidText(const string &value) {
	return !value.empty() && value.find('\0') == string::npos && Value::StringIsValid(value);
}

static bool IsValidPlanIdentifier(const Identifier &identifier) {
	return IsValidText(identifier.GetIdentifierName());
}

static bool IsPlanSQLRepresentableType(const LogicalType &type) {
	if (!type.IsComplete()) {
		return false;
	}
	return !TypeVisitor::Contains(type, [](const LogicalType &entry) {
		switch (entry.id()) {
		case LogicalTypeId::INVALID:
		case LogicalTypeId::UNKNOWN:
		case LogicalTypeId::ANY:
		case LogicalTypeId::UNBOUND:
		case LogicalTypeId::TEMPLATE:
		case LogicalTypeId::TYPE:
		case LogicalTypeId::STRING_LITERAL:
		case LogicalTypeId::INTEGER_LITERAL:
		case LogicalTypeId::POINTER:
		case LogicalTypeId::VALIDITY:
		case LogicalTypeId::TABLE:
		case LogicalTypeId::LEGACY_AGGREGATE_STATE:
		case LogicalTypeId::LAMBDA:
		case LogicalTypeId::TUPLE:
			return true;
		default:
			return false;
		}
	});
}

static LogicalPlanVerificationIssue InternalInvariant(const LogicalPlanVerificationPath &path,
                                                      const LogicalOperator &op, string invariant, string message) {
	LogicalPlanVerificationIssue issue;
	issue.code = LogicalPlanVerificationIssueCode::INTERNAL_INVARIANT;
	issue.phase = LogicalPlanVerificationPhase::PLAN_EXPORT;
	issue.path = path;
	if (op.type != LogicalOperatorType::LOGICAL_INVALID) {
		issue.construct = LogicalPlanVerificationConstructIdentity::LogicalOperator(op.type);
	}
	issue.facts.emplace_back("invariant", Value(std::move(invariant)));
	issue.message = std::move(message);
	return issue;
}

static LogicalPlanVerificationIssue ExtensionIssue(LogicalPlanVerificationIssueCode code,
                                                   const LogicalPlanVerificationPath &path,
                                                   const string &extension_identifier, string message) {
	LogicalPlanVerificationIssue issue;
	issue.code = code;
	issue.phase = LogicalPlanVerificationPhase::PLAN_EXPORT;
	issue.path = path;
	issue.construct = LogicalPlanVerificationConstructIdentity::Extension(extension_identifier);
	issue.message = std::move(message);
	return issue;
}

static LogicalPlanVerificationIssue PlanUnsupportedFeature(const LogicalPlanVerificationPath &path, string feature,
                                                           string message) {
	LogicalPlanVerificationIssue issue;
	issue.code = LogicalPlanVerificationIssueCode::UNSUPPORTED_EXPORT_FEATURE;
	issue.phase = LogicalPlanVerificationPhase::PLAN_EXPORT;
	issue.path = path;
	issue.construct = LogicalPlanVerificationConstructIdentity::ExportFeature(std::move(feature));
	issue.message = std::move(message);
	return issue;
}

static LogicalPlanVerificationIssue UnsupportedOperator(const LogicalPlanVerificationPath &path,
                                                        LogicalOperatorType type) {
	LogicalPlanVerificationIssue issue;
	issue.code = LogicalPlanVerificationIssueCode::UNSUPPORTED_OPERATOR;
	issue.phase = LogicalPlanVerificationPhase::PLAN_EXPORT;
	issue.path = path;
	issue.construct = LogicalPlanVerificationConstructIdentity::LogicalOperator(type);
	issue.facts.emplace_back("logical_operator_type", Value::UBIGINT(static_cast<uint64_t>(type)));
	issue.message = "The logical operator does not have a SQL AST representation in this exporter";
	return issue;
}

static LogicalPlanVerificationFunctionIdentity SourceIdentity(const LogicalOperator &op) {
	LogicalPlanVerificationFunctionIdentity identity;
	identity.return_type = LogicalType::TABLE;
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		identity.catalog = get.function.GetCatalogName().GetIdentifierName();
		identity.schema = get.function.GetSchemaName().GetIdentifierName();
		identity.name = get.function.GetName().GetIdentifierName();
		bool complete_arguments = true;
		for (auto &argument : get.function.GetArguments()) {
			if (!argument.IsComplete()) {
				complete_arguments = false;
				break;
			}
		}
		if (complete_arguments) {
			identity.arguments = get.function.GetArguments();
		}
	}
	if (identity.name.empty()) {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_GET:
			identity.name = "logical_get";
			break;
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
			identity.name = "logical_column_data_get";
			break;
		case LogicalOperatorType::LOGICAL_DELIM_GET:
			identity.name = "logical_delim_get";
			break;
		default:
			identity.name = "logical_source";
			break;
		}
	}
	return identity;
}

static LogicalPlanVerificationIssue UnsupportedSource(const LogicalPlanVerificationPath &path,
                                                      const LogicalOperator &op) {
	LogicalPlanVerificationIssue issue;
	issue.code = LogicalPlanVerificationIssueCode::UNSUPPORTED_SOURCE;
	issue.phase = LogicalPlanVerificationPhase::PLAN_EXPORT;
	issue.path = path;
	issue.construct = LogicalPlanVerificationConstructIdentity::SourceFunction(SourceIdentity(op));
	issue.facts.emplace_back("logical_operator_type", Value::UBIGINT(static_cast<uint64_t>(op.type)));
	issue.message = "The logical source does not expose structural SQL export semantics";
	return issue;
}

static LogicalPlanVerificationIssue SchemaIssue(const LogicalPlanVerificationPath &path, const LogicalOperator &op,
                                                const optional<string> &extension_identifier, string invariant,
                                                string message) {
	if (extension_identifier) {
		auto issue = ExtensionIssue(LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT, path,
		                            *extension_identifier, std::move(message));
		issue.facts.emplace_back("invariant", Value(std::move(invariant)));
		return issue;
	}
	return InternalInvariant(path, op, std::move(invariant), std::move(message));
}

static Identifier FieldIdentifier(idx_t ordinal) {
	return Identifier("c" + to_string(ordinal));
}

static LogicalPlanSQLFieldResult CreateFields(LogicalOperator &op, const LogicalPlanVerificationPath &path,
                                              optional<string> extension_identifier = {}) {
	auto bindings = op.GetColumnBindings();
	if (bindings.size() != op.types.size()) {
		auto issue = SchemaIssue(path, op, extension_identifier, "output_schema_arity",
		                         "Logical operator output binding and type counts differ");
		issue.facts.emplace_back("binding_count", Value::UBIGINT(bindings.size()));
		issue.facts.emplace_back("type_count", Value::UBIGINT(op.types.size()));
		return FieldFailure(std::move(issue));
	}
	vector<LogicalPlanSQLExportField> fields;
	fields.reserve(bindings.size());
	for (idx_t field_index = 0; field_index < bindings.size(); field_index++) {
		auto &binding = bindings[field_index];
		auto &type = op.types[field_index];
		if (!binding.table_index.IsValid() || !binding.column_index.IsValid()) {
			auto issue = SchemaIssue(path, op, extension_identifier, "invalid_output_binding",
			                         "Logical operator output contains an invalid binding");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			issue.facts.emplace_back("table_index", Value::UBIGINT(binding.table_index.index));
			issue.facts.emplace_back("column_index", Value::UBIGINT(binding.column_index.GetIndexUnsafe()));
			return FieldFailure(std::move(issue));
		}
		if (!IsPlanSQLRepresentableType(type)) {
			auto issue = SchemaIssue(path, op, extension_identifier, "invalid_output_type",
			                         "Logical operator output contains a type that cannot be represented in SQL");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			return FieldFailure(std::move(issue));
		}
		for (idx_t previous_index = 0; previous_index < field_index; previous_index++) {
			if (bindings[previous_index] == binding) {
				auto issue = SchemaIssue(path, op, extension_identifier, "duplicate_output_binding",
				                         "Logical operator output contains a duplicate binding");
				issue.facts.emplace_back("first_field_index", Value::UBIGINT(previous_index));
				issue.facts.emplace_back("duplicate_field_index", Value::UBIGINT(field_index));
				return FieldFailure(std::move(issue));
			}
		}
		fields.push_back(LogicalPlanSQLExportField {binding, type, FieldIdentifier(field_index), field_index});
	}
	return LogicalPlanSQLFieldResult::Success(std::move(fields));
}

static optional<LogicalPlanVerificationIssue> ValidateRelation(const LogicalPlanSQLExportRelation &relation,
                                                               const vector<LogicalPlanSQLExportField> &expected_fields,
                                                               const LogicalPlanVerificationPath &path,
                                                               const LogicalOperator &op,
                                                               optional<string> extension_identifier = {}) {
	if (!relation.query) {
		return SchemaIssue(path, op, extension_identifier, "null_query", "SQL export returned a null query AST");
	}
	if (relation.fields.size() != expected_fields.size()) {
		auto issue = SchemaIssue(path, op, extension_identifier, "field_count",
		                         "SQL export returned an unexpected number of fields");
		issue.facts.emplace_back("expected_field_count", Value::UBIGINT(expected_fields.size()));
		issue.facts.emplace_back("actual_field_count", Value::UBIGINT(relation.fields.size()));
		return issue;
	}
	for (idx_t field_index = 0; field_index < expected_fields.size(); field_index++) {
		auto &field = relation.fields[field_index];
		auto &expected = expected_fields[field_index];
		if (field.ordinal != field_index || field.ordinal != expected.ordinal) {
			auto issue = SchemaIssue(path, op, extension_identifier, "field_ordinal",
			                         "SQL export returned an unexpected field ordinal");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			issue.facts.emplace_back("field_ordinal", Value::UBIGINT(field.ordinal));
			return issue;
		}
		if (!field.source_binding.table_index.IsValid() || !field.source_binding.column_index.IsValid() ||
		    field.source_binding != expected.source_binding) {
			auto issue = SchemaIssue(path, op, extension_identifier, "field_binding",
			                         "SQL export returned an unexpected field binding");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			return issue;
		}
		if (!IsPlanSQLRepresentableType(field.type) || field.type != expected.type) {
			auto issue = SchemaIssue(path, op, extension_identifier, "field_type",
			                         "SQL export returned an unexpected field type");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			return issue;
		}
		if (!IsValidPlanIdentifier(field.identifier) ||
		    field.identifier.GetIdentifierName() != expected.identifier.GetIdentifierName()) {
			auto issue = SchemaIssue(path, op, extension_identifier, "field_identifier",
			                         "SQL export returned an unexpected field identifier");
			issue.facts.emplace_back("field_index", Value::UBIGINT(field_index));
			return issue;
		}
	}
	return {};
}

struct LogicalPlanSQLBindingEntry {
	ColumnBinding binding;
	LogicalType type;
	vector<Identifier> names;
};

struct LogicalPlanSQLExportedChild {
	LogicalPlanSQLExportRelation relation;
	Identifier relation_alias;
};

static BoundExpressionSQLExportContext
CreateBindingContext(const vector<reference<const LogicalPlanSQLExportedChild>> &children) {
	vector<LogicalPlanSQLBindingEntry> entries;
	for (auto &child_reference : children) {
		auto &child = child_reference.get();
		for (auto &field : child.relation.fields) {
			entries.push_back(LogicalPlanSQLBindingEntry {
			    field.source_binding, field.type, {child.relation_alias, field.identifier}});
		}
	}
	BoundExpressionSQLExportContext result;
	result.resolve_binding =
	    [entries = std::move(entries)](const ColumnBinding &binding) -> optional<ResolvedSQLColumnReference> {
		for (auto &entry : entries) {
			if (entry.binding == binding) {
				return ResolvedSQLColumnReference {entry.names, entry.type};
			}
		}
		return {};
	};
	return result;
}

static unique_ptr<TableRef> CreateSubquery(LogicalPlanSQLExportedChild child) {
	auto statement = make_uniq<SelectStatement>();
	statement->node = std::move(child.relation.query);
	return make_uniq<SubqueryRef>(std::move(statement), std::move(child.relation_alias));
}

static unique_ptr<ParsedExpression> ChildColumn(const LogicalPlanSQLExportedChild &child, idx_t field_index) {
	auto &field = child.relation.fields[field_index];
	return make_uniq<ColumnRefExpression>(field.identifier, child.relation_alias);
}

class LogicalPlanSQLExportState {
public:
	LogicalPlanSQLExportState(ClientContext &context_p, const LogicalPlanSQLExportOptions &options_p)
	    : context(context_p), options(options_p) {
	}

	LogicalPlanSQLExportResult Export(LogicalOperator &op, const LogicalPlanVerificationPath &path) {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
			return ExportExpressionGet(op.Cast<LogicalExpressionGet>(), path);
		case LogicalOperatorType::LOGICAL_FILTER:
			return ExportFilter(op.Cast<LogicalFilter>(), path);
		case LogicalOperatorType::LOGICAL_PROJECTION:
			return ExportProjection(op.Cast<LogicalProjection>(), path);
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
			return ExportAggregate(op.Cast<LogicalAggregate>(), path);
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
			return ExportExtension(op.Cast<LogicalExtensionOperator>(), path);
		case LogicalOperatorType::LOGICAL_GET:
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
		case LogicalOperatorType::LOGICAL_DELIM_GET:
			return PlanFailure(UnsupportedSource(path, op));
		case LogicalOperatorType::LOGICAL_INVALID:
			return PlanFailure(InternalInvariant(path, op, "invalid_operator_type",
			                                     "Logical plan SQL export received an invalid operator type"));
		case LogicalOperatorType::LOGICAL_WINDOW:
		case LogicalOperatorType::LOGICAL_UNNEST:
		case LogicalOperatorType::LOGICAL_LIMIT:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
		case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		case LogicalOperatorType::LOGICAL_DISTINCT:
		case LogicalOperatorType::LOGICAL_SAMPLE:
		case LogicalOperatorType::LOGICAL_PIVOT:
		case LogicalOperatorType::LOGICAL_COPY_DATABASE:
		case LogicalOperatorType::LOGICAL_SECURE_VIEW:
		case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		case LogicalOperatorType::LOGICAL_CTE_REF:
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		case LogicalOperatorType::LOGICAL_INSERT:
		case LogicalOperatorType::LOGICAL_DELETE:
		case LogicalOperatorType::LOGICAL_UPDATE:
		case LogicalOperatorType::LOGICAL_MERGE_INTO:
		case LogicalOperatorType::LOGICAL_TRIGGER:
		case LogicalOperatorType::LOGICAL_ALTER:
		case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		case LogicalOperatorType::LOGICAL_CREATE_INDEX:
		case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
		case LogicalOperatorType::LOGICAL_CREATE_VIEW:
		case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
		case LogicalOperatorType::LOGICAL_CREATE_MACRO:
		case LogicalOperatorType::LOGICAL_DROP:
		case LogicalOperatorType::LOGICAL_PRAGMA:
		case LogicalOperatorType::LOGICAL_TRANSACTION:
		case LogicalOperatorType::LOGICAL_CREATE_TYPE:
		case LogicalOperatorType::LOGICAL_ATTACH:
		case LogicalOperatorType::LOGICAL_DETACH:
		case LogicalOperatorType::LOGICAL_CREATE_TRIGGER:
		case LogicalOperatorType::LOGICAL_EXPLAIN:
		case LogicalOperatorType::LOGICAL_PREPARE:
		case LogicalOperatorType::LOGICAL_EXECUTE:
		case LogicalOperatorType::LOGICAL_EXPORT:
		case LogicalOperatorType::LOGICAL_VACUUM:
		case LogicalOperatorType::LOGICAL_SET:
		case LogicalOperatorType::LOGICAL_LOAD:
		case LogicalOperatorType::LOGICAL_RESET:
		case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		case LogicalOperatorType::LOGICAL_CONNECT:
		case LogicalOperatorType::LOGICAL_DISCONNECT:
		case LogicalOperatorType::LOGICAL_EXTERNAL_RESOURCE:
		case LogicalOperatorType::LOGICAL_CREATE_SECRET:
			return PlanFailure(UnsupportedOperator(path, op.type));
		}
		auto issue = UnsupportedOperator(path, op.type);
		issue.facts.emplace_back("unknown_operator_type", Value::BOOLEAN(true));
		issue.message = "Logical plan SQL export received an unknown operator type";
		return PlanFailure(std::move(issue));
	}

private:
	Identifier NextRelationAlias() {
		return Identifier("r" + to_string(next_relation_ordinal++));
	}

	LogicalPlanVerificationResult<LogicalPlanSQLExportedChild> ExportChild(LogicalOperator &child,
	                                                                       const LogicalPlanVerificationPath &path) {
		auto exported = Export(child, path);
		if (exported.HasError()) {
			return LogicalPlanVerificationResult<LogicalPlanSQLExportedChild>::Failure(exported.GetIssues());
		}
		LogicalPlanSQLExportedChild result {std::move(exported.GetValue()), NextRelationAlias()};
		return LogicalPlanVerificationResult<LogicalPlanSQLExportedChild>::Success(std::move(result));
	}

	LogicalPlanVerificationResult<unique_ptr<ParsedExpression>>
	ExportExpression(const LogicalOperator &op, idx_t expression_ordinal,
	                 const BoundExpressionSQLExportContext &expression_context,
	                 const LogicalPlanVerificationPath &path) {
		vector<reference<const Expression>> expressions;
		LogicalOperatorVisitor::EnumerateExpressions(op, [&](const unique_ptr<Expression> *expression) {
			D_ASSERT(expression && *expression);
			expressions.push_back(reference<const Expression>(**expression));
		});
		if (expression_ordinal >= expressions.size()) {
			auto issue = InternalInvariant(path, op, "expression_ordinal",
			                               "Logical plan SQL export requested an invalid expression ordinal");
			issue.facts.emplace_back("expression_ordinal", Value::UBIGINT(expression_ordinal));
			issue.facts.emplace_back("expression_count", Value::UBIGINT(expressions.size()));
			return LogicalPlanVerificationResult<unique_ptr<ParsedExpression>>::Failure({std::move(issue)});
		}
		return BoundExpressionSQLExporter::ExportAtPath(expressions[expression_ordinal].get(), expression_context,
		                                                PlanExpressionPath(path, expression_ordinal));
	}

	LogicalPlanSQLExportResult ExportExpressionGet(LogicalExpressionGet &get, const LogicalPlanVerificationPath &path) {
		if (get.children.size() != 1 || !get.children[0] ||
		    get.children[0]->type != LogicalOperatorType::LOGICAL_DUMMY_SCAN || !get.children[0]->children.empty() ||
		    !get.children[0]->expressions.empty()) {
			return PlanFailure(InternalInvariant(path, get, "expression_get_dummy_child",
			                                     "Logical expression get does not have its exact dummy scan scaffold"));
		}
		if (get.expressions.empty() || get.expressions[0].empty()) {
			return PlanFailure(InternalInvariant(path, get, "empty_values", "Logical expression get has no values"));
		}
		auto column_count = get.expressions[0].size();
		if (get.expr_types.size() != column_count) {
			auto issue = InternalInvariant(path, get, "values_type_arity",
			                               "Logical expression get column and type counts differ");
			issue.facts.emplace_back("column_count", Value::UBIGINT(column_count));
			issue.facts.emplace_back("type_count", Value::UBIGINT(get.expr_types.size()));
			return PlanFailure(std::move(issue));
		}
		auto fields = CreateFields(get, path);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		for (idx_t row_index = 0; row_index < get.expressions.size(); row_index++) {
			if (get.expressions[row_index].size() != column_count) {
				auto issue = InternalInvariant(path, get, "values_row_arity",
				                               "Logical expression get contains a non-rectangular row");
				issue.facts.emplace_back("row_index", Value::UBIGINT(row_index));
				issue.facts.emplace_back("expected_column_count", Value::UBIGINT(column_count));
				issue.facts.emplace_back("actual_column_count", Value::UBIGINT(get.expressions[row_index].size()));
				return PlanFailure(std::move(issue));
			}
			for (idx_t column_index = 0; column_index < column_count; column_index++) {
				if (get.expressions[row_index][column_index]->GetReturnType() != get.expr_types[column_index]) {
					auto issue = InternalInvariant(path, get, "values_cell_type",
					                               "Logical expression get cell and column types differ");
					issue.facts.emplace_back("row_index", Value::UBIGINT(row_index));
					issue.facts.emplace_back("column_index", Value::UBIGINT(column_index));
					return PlanFailure(std::move(issue));
				}
			}
		}

		auto values = make_uniq<ExpressionListRef>();
		values->alias = NextRelationAlias();
		values->expected_types = get.expr_types;
		for (auto &field : fields.GetValue()) {
			values->expected_names.push_back(field.identifier);
		}
		BoundExpressionSQLExportContext expression_context;
		idx_t expression_ordinal = 0;
		for (auto &row : get.expressions) {
			vector<unique_ptr<ParsedExpression>> exported_row;
			for (idx_t column_index = 0; column_index < row.size(); column_index++) {
				auto expression = ExportExpression(get, expression_ordinal++, expression_context, path);
				if (expression.HasError()) {
					return LogicalPlanSQLExportResult::Failure(expression.GetIssues());
				}
				exported_row.push_back(std::move(expression.GetValue()));
			}
			values->values.push_back(std::move(exported_row));
		}
		auto values_alias = values->alias;
		auto select = make_uniq<SelectNode>();
		for (auto &field : fields.GetValue()) {
			auto expression = make_uniq<ColumnRefExpression>(field.identifier, values_alias);
			expression->SetAlias(field.identifier);
			select->select_list.push_back(std::move(expression));
		}
		select->from_table = std::move(values);
		LogicalPlanSQLExportRelation relation {std::move(select), std::move(fields.GetValue())};
		return ValidateBuiltInRelation(get, path, std::move(relation));
	}

	LogicalPlanSQLExportResult ExportFilter(LogicalFilter &filter, const LogicalPlanVerificationPath &path) {
		if (filter.expressions.empty()) {
			return PlanFailure(InternalInvariant(path, filter, "empty_filter", "Logical filter has no predicates"));
		}
		for (idx_t expression_index = 0; expression_index < filter.expressions.size(); expression_index++) {
			if (filter.expressions[expression_index]->GetReturnType() != LogicalType::BOOLEAN) {
				auto issue =
				    InternalInvariant(path, filter, "filter_predicate_type", "Logical filter predicate is not Boolean");
				issue.facts.emplace_back("expression_index", Value::UBIGINT(expression_index));
				return PlanFailure(std::move(issue));
			}
		}
		auto fields = CreateFields(filter, path);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		auto child = ExportChild(*filter.children[0], PlanChildPath(path, 0));
		if (child.HasError()) {
			return LogicalPlanSQLExportResult::Failure(child.GetIssues());
		}
		vector<reference<const LogicalPlanSQLExportedChild>> child_references {child.GetValue()};
		auto expression_context = CreateBindingContext(child_references);
		vector<unique_ptr<ParsedExpression>> predicates;
		for (idx_t expression_index = 0; expression_index < filter.expressions.size(); expression_index++) {
			auto predicate = ExportExpression(filter, expression_index, expression_context, path);
			if (predicate.HasError()) {
				return LogicalPlanSQLExportResult::Failure(predicate.GetIssues());
			}
			predicates.push_back(std::move(predicate.GetValue()));
		}
		auto select = make_uniq<SelectNode>();
		for (idx_t field_index = 0; field_index < fields.GetValue().size(); field_index++) {
			auto child_field_index =
			    filter.projection_map.empty() ? field_index : filter.projection_map[field_index].GetIndexUnsafe();
			auto expression = ChildColumn(child.GetValue(), child_field_index);
			expression->SetAlias(fields.GetValue()[field_index].identifier);
			select->select_list.push_back(std::move(expression));
		}
		if (predicates.size() == 1) {
			select->where_clause = std::move(predicates[0]);
		} else {
			select->where_clause =
			    make_uniq<ConjunctionExpression>(ExpressionType::CONJUNCTION_AND, std::move(predicates));
		}
		select->from_table = CreateSubquery(std::move(child.GetValue()));
		LogicalPlanSQLExportRelation relation {std::move(select), std::move(fields.GetValue())};
		return ValidateBuiltInRelation(filter, path, std::move(relation));
	}

	LogicalPlanSQLExportResult ExportProjection(LogicalProjection &projection,
	                                            const LogicalPlanVerificationPath &path) {
		if (projection.expressions.empty()) {
			return PlanFailure(
			    InternalInvariant(path, projection, "empty_projection", "Logical projection has no expressions"));
		}
		auto fields = CreateFields(projection, path);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		if (projection.expressions.size() != fields.GetValue().size()) {
			return PlanFailure(InternalInvariant(path, projection, "projection_arity",
			                                     "Logical projection expression and output counts differ"));
		}
		auto child = ExportChild(*projection.children[0], PlanChildPath(path, 0));
		if (child.HasError()) {
			return LogicalPlanSQLExportResult::Failure(child.GetIssues());
		}
		vector<reference<const LogicalPlanSQLExportedChild>> child_references {child.GetValue()};
		auto expression_context = CreateBindingContext(child_references);
		auto select = make_uniq<SelectNode>();
		for (idx_t expression_index = 0; expression_index < projection.expressions.size(); expression_index++) {
			auto expression = ExportExpression(projection, expression_index, expression_context, path);
			if (expression.HasError()) {
				return LogicalPlanSQLExportResult::Failure(expression.GetIssues());
			}
			expression.GetValue()->SetAlias(fields.GetValue()[expression_index].identifier);
			select->select_list.push_back(std::move(expression.GetValue()));
		}
		select->from_table = CreateSubquery(std::move(child.GetValue()));
		LogicalPlanSQLExportRelation relation {std::move(select), std::move(fields.GetValue())};
		return ValidateBuiltInRelation(projection, path, std::move(relation));
	}

	LogicalPlanSQLExportResult ExportAggregate(LogicalAggregate &aggregate, const LogicalPlanVerificationPath &path) {
		if (!aggregate.groups.empty() || !aggregate.grouping_functions.empty() || aggregate.grouping_sets.size() > 1 ||
		    (aggregate.grouping_sets.size() == 1 && !aggregate.grouping_sets[0].empty())) {
			return PlanFailure(
			    PlanUnsupportedFeature(path, "grouped_aggregate", "Only ungrouped logical aggregates can be exported"));
		}
		if (aggregate.expressions.empty()) {
			return PlanFailure(InternalInvariant(path, aggregate, "empty_aggregate",
			                                     "Logical aggregate has no aggregate expressions"));
		}
		for (idx_t expression_index = 0; expression_index < aggregate.expressions.size(); expression_index++) {
			if (aggregate.expressions[expression_index]->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
				auto issue = InternalInvariant(path, aggregate, "aggregate_expression_class",
				                               "Logical aggregate contains a non-aggregate expression");
				issue.facts.emplace_back("expression_index", Value::UBIGINT(expression_index));
				return PlanFailure(std::move(issue));
			}
		}
		auto fields = CreateFields(aggregate, path);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		if (aggregate.expressions.size() != fields.GetValue().size()) {
			return PlanFailure(InternalInvariant(path, aggregate, "aggregate_arity",
			                                     "Logical aggregate expression and output counts differ"));
		}
		auto child = ExportChild(*aggregate.children[0], PlanChildPath(path, 0));
		if (child.HasError()) {
			return LogicalPlanSQLExportResult::Failure(child.GetIssues());
		}
		vector<reference<const LogicalPlanSQLExportedChild>> child_references {child.GetValue()};
		auto expression_context = CreateBindingContext(child_references);
		auto select = make_uniq<SelectNode>();
		for (idx_t expression_index = 0; expression_index < aggregate.expressions.size(); expression_index++) {
			auto expression = ExportExpression(aggregate, expression_index, expression_context, path);
			if (expression.HasError()) {
				return LogicalPlanSQLExportResult::Failure(expression.GetIssues());
			}
			expression.GetValue()->SetAlias(fields.GetValue()[expression_index].identifier);
			select->select_list.push_back(std::move(expression.GetValue()));
		}
		select->from_table = CreateSubquery(std::move(child.GetValue()));
		LogicalPlanSQLExportRelation relation {std::move(select), std::move(fields.GetValue())};
		return ValidateBuiltInRelation(aggregate, path, std::move(relation));
	}

	LogicalPlanSQLExportResult ExportExtension(LogicalExtensionOperator &extension,
	                                           const LogicalPlanVerificationPath &path) {
		auto extension_identifier = extension.GetExtensionName();
		if (!IsValidText(extension_identifier)) {
			auto issue = InternalInvariant(path, extension, "extension_identifier",
			                               "Logical extension operator has an invalid stable identifier");
			return PlanFailure(std::move(issue));
		}
		auto fields = CreateFields(extension, path, extension_identifier);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		unordered_set<string> resolver_identifiers;
		for (idx_t resolver_index = 0; resolver_index < options.extension_resolvers.size(); resolver_index++) {
			auto &resolver = options.extension_resolvers[resolver_index];
			if (!IsValidText(resolver.identifier) || !resolver.callback ||
			    !resolver_identifiers.insert(resolver.identifier).second) {
				auto issue = ExtensionIssue(LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT, path,
				                            extension_identifier, "Logical plan SQL export resolver is malformed");
				issue.facts.emplace_back("resolver_index", Value::UBIGINT(resolver_index));
				return PlanFailure(std::move(issue));
			}
		}

		vector<LogicalPlanSQLExportedChild> exported_children;
		exported_children.reserve(extension.children.size());
		for (idx_t child_index = 0; child_index < extension.children.size(); child_index++) {
			auto child = ExportChild(*extension.children[child_index], PlanChildPath(path, child_index));
			if (child.HasError()) {
				return LogicalPlanSQLExportResult::Failure(child.GetIssues());
			}
			exported_children.push_back(std::move(child.GetValue()));
		}
		vector<reference<const LogicalPlanSQLExportedChild>> child_references;
		vector<LogicalPlanSQLExportChild> child_views;
		for (auto &child : exported_children) {
			child_references.push_back(child);
			child_views.emplace_back(child.relation, child.relation_alias);
		}
		auto expression_context = CreateBindingContext(child_references);
		idx_t expression_count = 0;
		LogicalOperatorVisitor::EnumerateExpressions(extension,
		                                             [&](const unique_ptr<Expression> *) { expression_count++; });
		auto expression_exporter = [&](idx_t expression_ordinal) {
			return ExportExpression(extension, expression_ordinal, expression_context, path);
		};
		LogicalPlanSQLExportExtensionInput input(extension, child_views, fields.GetValue(), expression_count,
		                                         std::move(expression_exporter));

		for (auto &resolver : options.extension_resolvers) {
			auto result = resolver.callback(input);
			auto handled = HandleExtensionResult(extension, path, extension_identifier, resolver.identifier,
			                                     std::move(result), fields.GetValue());
			if (handled) {
				return std::move(*handled);
			}
		}
		for (auto &registered_extension : OperatorExtension::Iterate(context)) {
			if (registered_extension->GetName() != extension_identifier) {
				continue;
			}
			if (!registered_extension->HasSQLExportCallback()) {
				break;
			}
			auto result = registered_extension->GetSQLExportCallback()(input);
			auto handled = HandleExtensionResult(extension, path, extension_identifier, extension_identifier,
			                                     std::move(result), fields.GetValue());
			if (handled) {
				return std::move(*handled);
			}
			break;
		}
		return PlanFailure(ExtensionIssue(LogicalPlanVerificationIssueCode::UNSUPPORTED_EXTENSION, path,
		                                  extension_identifier,
		                                  "No SQL export handler accepted the extension operator"));
	}

	optional<LogicalPlanSQLExportResult>
	HandleExtensionResult(LogicalExtensionOperator &extension, const LogicalPlanVerificationPath &path,
	                      const string &extension_identifier, const string &handler_identifier,
	                      LogicalPlanSQLExportExtensionResult result,
	                      const vector<LogicalPlanSQLExportField> &expected_fields) {
		auto malformed = [&](string invariant, string message) {
			auto issue = ExtensionIssue(LogicalPlanVerificationIssueCode::MALFORMED_EXTENSION_RESULT, path,
			                            extension_identifier, std::move(message));
			issue.facts.emplace_back("handler_identifier", Value(handler_identifier));
			issue.facts.emplace_back("invariant", Value(std::move(invariant)));
			return LogicalPlanSQLExportResult::Failure(vector<LogicalPlanVerificationIssue> {std::move(issue)});
		};

		switch (result.type) {
		case LogicalPlanSQLExportExtensionResultType::NOT_HANDLED:
			if (result.relation || !result.reason.empty()) {
				return malformed("not_handled_payload", "NOT_HANDLED extension result contains a payload");
			}
			return {};
		case LogicalPlanSQLExportExtensionResultType::EXPORTED:
			if (!result.relation || !result.reason.empty()) {
				return malformed("exported_payload", "EXPORTED extension result has a malformed payload");
			}
			if (auto issue =
			        ValidateRelation(*result.relation, expected_fields, path, extension, extension_identifier)) {
				issue->facts.emplace_back("handler_identifier", Value(handler_identifier));
				return LogicalPlanSQLExportResult::Failure({std::move(*issue)});
			}
			return LogicalPlanSQLExportResult::Success(std::move(*result.relation));
		case LogicalPlanSQLExportExtensionResultType::UNSUPPORTED:
			if (result.relation || !IsValidText(result.reason)) {
				return malformed("unsupported_payload", "UNSUPPORTED extension result has a malformed payload");
			}
			{
				auto issue = ExtensionIssue(LogicalPlanVerificationIssueCode::UNSUPPORTED_EXTENSION, path,
				                            extension_identifier, std::move(result.reason));
				issue.facts.emplace_back("handler_identifier", Value(handler_identifier));
				return LogicalPlanSQLExportResult::Failure({std::move(issue)});
			}
		}
		return malformed("result_type", "Extension SQL export returned an unknown result type");
	}

	LogicalPlanSQLExportResult ValidateBuiltInRelation(LogicalOperator &op, const LogicalPlanVerificationPath &path,
	                                                   LogicalPlanSQLExportRelation relation) {
		auto fields = CreateFields(op, path);
		if (fields.HasError()) {
			return LogicalPlanSQLExportResult::Failure(fields.GetIssues());
		}
		if (auto issue = ValidateRelation(relation, fields.GetValue(), path, op)) {
			return PlanFailure(std::move(*issue));
		}
		return LogicalPlanSQLExportResult::Success(std::move(relation));
	}

private:
	ClientContext &context;
	const LogicalPlanSQLExportOptions &options;
	idx_t next_relation_ordinal = 0;
};

} // namespace logical_plan_sql_export

LogicalPlanVerificationResult<LogicalPlanSQLExportRelation>
LogicalPlanSQLExporter::Export(ClientContext &context, LogicalOperator &root,
                               const LogicalPlanSQLExportOptions &options) {
	auto verification = LogicalPlanVerifier::VerifyAlways(root);
	if (verification.HasError()) {
		return LogicalPlanVerificationResult<LogicalPlanSQLExportRelation>::Failure(verification.GetIssues());
	}
	logical_plan_sql_export::LogicalPlanSQLExportState state(context, options);
	return state.Export(root, LogicalPlanVerificationPath());
}

} // namespace duckdb
