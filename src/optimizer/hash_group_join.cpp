#include "duckdb/optimizer/hash_group_join.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

static optional_idx GetDirectReferenceIndex(const Expression &expression, LogicalOperator &input) {
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
			auto child_index = GetDirectReferenceIndex(*projection.expressions[index], child);
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

static optional<UniqueKeyProof> ProveUniqueKey(LogicalOperator &owner, const vector<idx_t> &owner_keys) {
	if (owner_keys.empty()) {
		return nullopt;
	}
	auto logical_columns = owner_keys;
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
			if (key_set.find(index.index) == key_set.end()) {
				matches = false;
				break;
			}
			if (!unique.IsPrimaryKey() && not_null_columns.find(index.index) == not_null_columns.end()) {
				matches = false;
				break;
			}
		}
		if (matches) {
			return unique.IsPrimaryKey() ? UniqueKeyProof::PRIMARY_KEY : UniqueKeyProof::UNIQUE_NOT_NULL;
		}
	}
	return nullopt;
}

static bool ExpressionUsesOnlyJoinChild(const Expression &expression, LogicalComparisonJoin &join, idx_t child_index) {
	bool valid = true;
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		auto binding = expression.Cast<BoundColumnRefExpression>().Binding();
		auto bindings = join.children[child_index]->GetColumnBindings();
		return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
	}
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		auto left_count = join.children[0]->GetColumnBindings().size();
		if (child_index == 0) {
			return index < left_count;
		}
		return index >= left_count && index < left_count + join.children[1]->GetColumnBindings().size();
	}
	ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) {
		if (valid && !ExpressionUsesOnlyJoinChild(child, join, child_index)) {
			valid = false;
		}
	});
	return valid;
}

static bool AggregatesUseProbe(const LogicalAggregate &aggregate, LogicalComparisonJoin &join, idx_t probe_child,
                               HashGroupJoinCandidateMode mode) {
	if (aggregate.expressions.empty()) {
		return false;
	}
	for (auto &expression : aggregate.expressions) {
		if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
			return false;
		}
		auto &aggr = expression->Cast<BoundAggregateExpression>();
		auto &callbacks = aggr.Function().GetCallbacks();
		if (aggr.IsDistinct() || aggr.GetFilter() ||
		    (aggr.GetOrderBys() && mode == HashGroupJoinCandidateMode::STRICT) || aggr.IsVolatile() ||
		    aggr.StateExportMode() != AggregateStateExportMode::NONE || !callbacks.HasStateInitCallback() ||
		    !callbacks.HasStateSizeCallback() || !callbacks.HasStateUpdateCallback() ||
		    !callbacks.HasStateCombineCallback() || !callbacks.HasStateFinalizeCallback()) {
			return false;
		}
		for (auto &child : aggr.GetChildren()) {
			if (child->IsVolatile() || !ExpressionUsesOnlyJoinChild(*child, join, probe_child)) {
				return false;
			}
		}
		if (aggr.GetOrderBys()) {
			for (auto &order : aggr.GetOrderBys()->orders) {
				if (order.expression->IsVolatile() ||
				    !ExpressionUsesOnlyJoinChild(*order.expression, join, probe_child)) {
					return false;
				}
			}
		}
	}
	return true;
}

static optional_idx GetJoinOutputReference(const Expression &expression, LogicalComparisonJoin &join,
                                           idx_t &child_index) {
	if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto index = expression.Cast<BoundReferenceExpression>().Index();
		auto left_count = join.children[0]->GetColumnBindings().size();
		if (index < left_count) {
			child_index = 0;
			return optional_idx(index);
		}
		index -= left_count;
		if (index < join.children[1]->GetColumnBindings().size()) {
			child_index = 1;
			return optional_idx(index);
		}
		return optional_idx();
	}
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
		return optional_idx();
	}
	for (idx_t side = 0; side < 2; side++) {
		auto index = GetDirectReferenceIndex(expression, *join.children[side]);
		if (index.IsValid()) {
			child_index = side;
			return index;
		}
	}
	return optional_idx();
}

optional<HashGroupJoinCandidate> TryGetHashGroupJoinCandidate(LogicalAggregate &aggregate, LogicalComparisonJoin &join,
                                                              ClientContext &context, HashGroupJoinCandidateMode mode) {
	(void)context;
	if (join.join_type != JoinType::INNER || join.HasProjectionMap() || join.children.size() != 2 ||
	    join.conditions.empty() || join.HasArbitraryConditions() || !aggregate.grouping_functions.empty() ||
	    aggregate.grouping_sets.size() > 1 || aggregate.groups.empty()) {
		return nullopt;
	}
	if (!aggregate.grouping_sets.empty()) {
		auto &grouping_set = aggregate.grouping_sets[0];
		if (grouping_set.size() != aggregate.groups.size()) {
			return nullopt;
		}
		for (idx_t group_idx = 0; group_idx < aggregate.groups.size(); group_idx++) {
			if (grouping_set.find(ProjectionIndex(group_idx)) == grouping_set.end()) {
				return nullopt;
			}
		}
	}

	vector<idx_t> side_keys[2];
	for (auto &condition : join.conditions) {
		if (!condition.IsComparison() || condition.GetComparisonType() != ExpressionType::COMPARE_EQUAL ||
		    condition.GetLHS().GetReturnType() != condition.GetRHS().GetReturnType()) {
			return nullopt;
		}
		auto left = GetDirectReferenceIndex(condition.GetLHS(), *join.children[0]);
		auto right = GetDirectReferenceIndex(condition.GetRHS(), *join.children[1]);
		if (!left.IsValid() || !right.IsValid()) {
			return nullopt;
		}
		side_keys[0].push_back(left.GetIndex());
		side_keys[1].push_back(right.GetIndex());
	}

	optional<HashGroupJoinCandidate> candidate;
	for (idx_t owner_child : {idx_t(1), idx_t(0)}) {
		auto proof = ProveUniqueKey(*join.children[owner_child], side_keys[owner_child]);
		const auto probe_child = 1 - owner_child;
		if (proof && AggregatesUseProbe(aggregate, join, probe_child, mode)) {
			candidate = HashGroupJoinCandidate {owner_child, probe_child, {}, {}, *proof};
			break;
		}
	}
	if (!candidate) {
		return nullopt;
	}

	vector<bool> used_conditions(join.conditions.size(), false);
	for (auto &group : aggregate.groups) {
		if (group->IsVolatile()) {
			return nullopt;
		}
		idx_t group_child;
		auto group_index = GetJoinOutputReference(*group, join, group_child);
		if (!group_index.IsValid()) {
			return nullopt;
		}
		optional_idx condition_index;
		for (idx_t index = 0; index < join.conditions.size(); index++) {
			if (side_keys[group_child][index] == group_index.GetIndex()) {
				if (condition_index.IsValid()) {
					return nullopt;
				}
				condition_index = optional_idx(index);
			}
		}
		if (!condition_index.IsValid() || used_conditions[condition_index.GetIndex()]) {
			return nullopt;
		}
		auto index = condition_index.GetIndex();
		used_conditions[index] = true;
		candidate->owner_key_indices.push_back(side_keys[candidate->owner_child][index]);
		candidate->probe_key_indices.push_back(side_keys[candidate->probe_child][index]);
	}
	if (candidate->owner_key_indices.size() != join.conditions.size()) {
		return nullopt;
	}
	return candidate;
}

} // namespace duckdb
