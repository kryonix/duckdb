#include "duckdb/optimizer/key_properties.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/constraints/foreign_key_constraint.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

static optional_idx GetKeyPropertyDirectReferenceIndex(const Expression &expression, LogicalOperator &input) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		return index < input.GetColumnBindings().size() ? optional_idx(index) : optional_idx();
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return optional_idx();
	}
	auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
	auto bindings = input.GetColumnBindings();
	for (idx_t index = 0; index < bindings.size(); index++) {
		if (bindings[index] == binding) {
			return optional_idx(index);
		}
	}
	return optional_idx();
}

static bool TraceBaseColumns(LogicalOperator &op, vector<idx_t> &column_indices, optional_ptr<LogicalGet> &base_scan) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_FILTER: {
		if (op.children.size() != 1) {
			return false;
		}
		auto &filter = op.Cast<LogicalFilter>();
		if (!filter.projection_map.empty()) {
			for (auto &index : column_indices) {
				if (index >= filter.projection_map.size()) {
					return false;
				}
				index = filter.projection_map[index].GetIndex();
			}
		}
		return TraceBaseColumns(*op.children[0], column_indices, base_scan);
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		if (op.children.size() != 1) {
			return false;
		}
		auto &projection = op.Cast<LogicalProjection>();
		auto &child = *op.children[0];
		for (auto &index : column_indices) {
			if (index >= projection.expressions.size()) {
				return false;
			}
			auto child_index = GetKeyPropertyDirectReferenceIndex(*projection.expressions[index], child);
			if (!child_index.IsValid()) {
				return false;
			}
			index = child_index.GetIndex();
		}
		return TraceBaseColumns(child, column_indices, base_scan);
	}
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.name != "seq_scan" || !get.GetTable()) {
			return false;
		}
		auto bindings = get.GetColumnBindings();
		for (auto &index : column_indices) {
			if (index >= bindings.size()) {
				return false;
			}
			auto &column_index = get.GetColumnIndex(bindings[index]);
			if (!column_index.HasPrimaryIndex() || column_index.HasChildren() || column_index.IsVirtualColumn()) {
				return false;
			}
			index = column_index.GetPrimaryIndex();
		}
		base_scan = get;
		return true;
	}
	default:
		return false;
	}
}

optional<UniqueKeyProperty> GetUniqueKeyProperty(LogicalOperator &owner, const vector<idx_t> &output_columns) {
	if (output_columns.empty()) {
		return nullopt;
	}
	auto logical_columns = output_columns;
	optional_ptr<LogicalGet> base_scan;
	if (!TraceBaseColumns(owner, logical_columns, base_scan) || !base_scan) {
		return nullopt;
	}

	unordered_set<idx_t> key_set;
	for (auto column : logical_columns) {
		if (!key_set.insert(column).second) {
			return nullopt;
		}
	}

	auto &table = *base_scan->GetTable();
	unordered_set<idx_t> not_null_columns;
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type == ConstraintType::NOT_NULL) {
			not_null_columns.insert(constraint->Cast<NotNullConstraint>().index.index);
		}
	}
	for (auto &constraint : table.GetConstraints()) {
		if (constraint->type != ConstraintType::UNIQUE) {
			continue;
		}
		auto &unique = constraint->Cast<UniqueConstraint>();
		auto indexes = unique.GetLogicalIndexes(table.GetColumns());
		if (indexes.size() != key_set.size()) {
			continue;
		}
		bool matches = true;
		for (auto index : indexes) {
			if (key_set.find(index.index) == key_set.end() ||
			    (!unique.IsPrimaryKey() && not_null_columns.find(index.index) == not_null_columns.end())) {
				matches = false;
				break;
			}
		}
		if (matches) {
			return UniqueKeyProperty {
			    unique.IsPrimaryKey() ? UniqueKeyProof::PRIMARY_KEY : UniqueKeyProof::UNIQUE_NOT_NULL, base_scan};
		}
	}
	return nullopt;
}

bool HasForeignKeyProperty(LogicalOperator &foreign, const vector<idx_t> &foreign_columns, LogicalOperator &primary,
                           const vector<idx_t> &primary_columns) {
	if (foreign_columns.empty() || foreign_columns.size() != primary_columns.size()) {
		return false;
	}
	auto foreign_base_columns = foreign_columns;
	auto primary_base_columns = primary_columns;
	optional_ptr<LogicalGet> foreign_scan;
	optional_ptr<LogicalGet> primary_scan;
	if (!TraceBaseColumns(foreign, foreign_base_columns, foreign_scan) || !foreign_scan ||
	    !TraceBaseColumns(primary, primary_base_columns, primary_scan) || !primary_scan) {
		return false;
	}
	auto &foreign_table = *foreign_scan->GetTable();
	auto &primary_table = *primary_scan->GetTable();
	if (&foreign_table.ParentCatalog() != &primary_table.ParentCatalog()) {
		return false;
	}
	unordered_set<idx_t> not_null_columns;
	for (auto &constraint : foreign_table.GetConstraints()) {
		if (constraint->type == ConstraintType::NOT_NULL) {
			not_null_columns.insert(constraint->Cast<NotNullConstraint>().index.index);
		}
	}
	for (auto column : foreign_base_columns) {
		if (not_null_columns.find(column) == not_null_columns.end()) {
			return false;
		}
	}
	for (auto &constraint : foreign_table.GetConstraints()) {
		if (constraint->type != ConstraintType::FOREIGN_KEY) {
			continue;
		}
		auto &foreign_key = constraint->Cast<ForeignKeyConstraint>();
		if (foreign_key.info.type != ForeignKeyType::FK_TYPE_FOREIGN_KEY_TABLE ||
		    foreign_key.info.table != primary_table.name ||
		    foreign_key.info.schema != primary_table.ParentSchema().name ||
		    foreign_key.info.fk_keys.size() != foreign_base_columns.size() ||
		    foreign_key.info.pk_keys.size() != primary_base_columns.size()) {
			continue;
		}
		bool matches = true;
		for (idx_t key_idx = 0; key_idx < foreign_base_columns.size(); key_idx++) {
			if (foreign_key.info.fk_keys[key_idx].index != foreign_base_columns[key_idx] ||
			    foreign_key.info.pk_keys[key_idx].index != primary_base_columns[key_idx]) {
				matches = false;
				break;
			}
		}
		if (matches) {
			return true;
		}
	}
	return false;
}

bool UniqueKeyProperty::FunctionallyDetermines(LogicalOperator &owner, idx_t output_column) const {
	vector<idx_t> columns {output_column};
	optional_ptr<LogicalGet> dependent_scan;
	return TraceBaseColumns(owner, columns, dependent_scan) && dependent_scan && dependent_scan == base_scan;
}

} // namespace duckdb
