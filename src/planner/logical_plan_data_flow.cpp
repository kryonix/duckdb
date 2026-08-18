#include "duckdb/planner/logical_plan_data_flow.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/detail/rooted_dynamic_forest.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_empty_result.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_recursive_cte.hpp"

namespace duckdb {

static RootedDynamicForestPathValue ToForestValue(const LogicalPlanPathSummary &summary) {
	return {summary.properties};
}

static LogicalPlanPathSummary FromForestValue(const RootedDynamicForestPathValue &value) {
	return {value.flags};
}

static uint64_t PathPropertyMask(LogicalPlanPathProperty property) {
	return uint64_t(1) << NumericCast<uint64_t>(property);
}

void LogicalPlanPathSummary::Add(LogicalPlanPathProperty property) {
	properties |= PathPropertyMask(property);
}

void LogicalPlanPathSummary::Merge(const LogicalPlanPathSummary &other) {
	properties |= other.properties;
}

bool LogicalPlanPathSummary::Has(LogicalPlanPathProperty property) const {
	return (properties & PathPropertyMask(property)) != 0;
}

bool LogicalPlanPathSummary::operator==(const LogicalPlanPathSummary &other) const {
	return properties == other.properties;
}

bool LogicalPlanPathSummary::operator!=(const LogicalPlanPathSummary &other) const {
	return !(*this == other);
}

struct LogicalPlanDataFlowEntry {
	explicit LogicalPlanDataFlowEntry(LogicalOperator &op_p) : op(op_p) {
	}

	reference<LogicalOperator> op;
	optional_ptr<LogicalOperator> owner_parent;
	idx_t owner_child_index = DConstants::INVALID_INDEX;
	optional_ptr<LogicalOperator> flow_parent;
	idx_t flow_child_index = DConstants::INVALID_INDEX;
	RootedDynamicForestNode forest_node;
	LogicalPlanPathSummary node_value;
	LogicalPlanPathSummary edge_to_parent;
	bool opaque = false;
};

struct LogicalPlanCTELineage {
	optional_ptr<LogicalOperator> producer;
	vector<reference<LogicalOperator>> readers;
};

struct LogicalPlanBuildTask {
	reference<LogicalOperator> op;
	optional_ptr<LogicalOperator> owner_parent;
	idx_t owner_child_index;
};

class LogicalPlanDataFlowState {
public:
	explicit LogicalPlanDataFlowState(LogicalOperator &root_p) : root(root_p) {
		BuildOwnership();
		BuildFlow();
		BuildSourcesAndLineage();
		BuildUses();
	}

	reference<LogicalOperator> root;
	vector<unique_ptr<LogicalPlanDataFlowEntry>> entries;
	reference_map_t<LogicalOperator, idx_t> entry_map;
	RootedDynamicForest forest;
	reference_map_t<RootedDynamicForestNode, LogicalOperator *> forest_operators;
	unordered_map<TableIndex, LogicalOperator *> binding_sources;
	unordered_map<TableIndex, LogicalPlanCTELineage> cte_lineage;
	vector<LogicalPlanBindingUse> binding_uses;
	bool valid = true;

public:
	optional_ptr<LogicalPlanDataFlowEntry> GetEntry(LogicalOperator &op) const {
		auto entry = entry_map.find(op);
		if (entry == entry_map.end()) {
			return nullptr;
		}
		return *entries[entry->second];
	}

	bool IsFlowChild(const LogicalOperator &parent, idx_t child_index) const {
		if (child_index >= parent.children.size()) {
			return false;
		}
		switch (parent.type) {
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_FILTER:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_WINDOW:
		case LogicalOperatorType::LOGICAL_UNNEST:
		case LogicalOperatorType::LOGICAL_LIMIT:
		case LogicalOperatorType::LOGICAL_ORDER_BY:
		case LogicalOperatorType::LOGICAL_TOP_N:
		case LogicalOperatorType::LOGICAL_DISTINCT:
		case LogicalOperatorType::LOGICAL_SAMPLE:
		case LogicalOperatorType::LOGICAL_PIVOT:
			return child_index == 0;
		case LogicalOperatorType::LOGICAL_GET:
			// Table-in/table-out functions may expose selected input columns.
			return child_index == 0 && parent.children.size() == 1;
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
			return child_index < 2;
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
			return true;
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
			return child_index == 1;
		case LogicalOperatorType::LOGICAL_INVALID:
		case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		case LogicalOperatorType::LOGICAL_COPY_DATABASE:
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
		case LogicalOperatorType::LOGICAL_DELIM_GET:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		case LogicalOperatorType::LOGICAL_EMPTY_RESULT:
		case LogicalOperatorType::LOGICAL_CTE_REF:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
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
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
			return false;
		}
		return false;
	}

	bool IsOpaque(const LogicalOperator &op) const {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
		case LogicalOperatorType::LOGICAL_COPY_DATABASE:
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
			return true;
		default:
			return false;
		}
	}

	vector<TableIndex> GetProducedTableIndexes(const LogicalOperator &op) const {
		if (op.type == LogicalOperatorType::LOGICAL_EMPTY_RESULT) {
			unordered_set<TableIndex> indexes;
			for (auto &binding : op.Cast<LogicalEmptyResult>().bindings) {
				indexes.insert(binding.table_index);
			}
			return vector<TableIndex>(indexes.begin(), indexes.end());
		}
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_WINDOW:
		case LogicalOperatorType::LOGICAL_UNNEST:
		case LogicalOperatorType::LOGICAL_PIVOT:
		case LogicalOperatorType::LOGICAL_GET:
		case LogicalOperatorType::LOGICAL_CHUNK_GET:
		case LogicalOperatorType::LOGICAL_DELIM_GET:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		case LogicalOperatorType::LOGICAL_DUMMY_SCAN:
		case LogicalOperatorType::LOGICAL_CTE_REF:
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
			return op.GetTableIndex();
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
			// An extension can explicitly identify a fresh namespace through the existing virtual method.
			return op.GetTableIndex();
		default:
			return {};
		}
	}

	LogicalPlanPathSummary GetNodeValue(const LogicalOperator &op) const {
		LogicalPlanPathSummary result;
		if (IsOpaque(op)) {
			result.Add(LogicalPlanPathProperty::OPAQUE_BOUNDARY);
		}
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		case LogicalOperatorType::LOGICAL_CTE_REF:
			result.Add(LogicalPlanPathProperty::CTE_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_PROJECTION:
			result.Add(LogicalPlanPathProperty::PROJECTION_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_PIVOT:
			result.Add(LogicalPlanPathProperty::AGGREGATE_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
			result.Add(LogicalPlanPathProperty::SET_OPERATION_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_WINDOW:
			result.Add(LogicalPlanPathProperty::WINDOW_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_UNNEST:
			result.Add(LogicalPlanPathProperty::UNNEST_BOUNDARY);
			break;
		case LogicalOperatorType::LOGICAL_LIMIT:
		case LogicalOperatorType::LOGICAL_TOP_N:
			result.Add(LogicalPlanPathProperty::LIMIT_BOUNDARY);
			break;
		default:
			break;
		}
		if (op.type == LogicalOperatorType::LOGICAL_INSERT || op.type == LogicalOperatorType::LOGICAL_UPDATE ||
		    op.type == LogicalOperatorType::LOGICAL_DELETE || op.type == LogicalOperatorType::LOGICAL_MERGE_INTO ||
		    op.type == LogicalOperatorType::LOGICAL_COPY_TO_FILE) {
			result.Add(LogicalPlanPathProperty::SIDE_EFFECT_BOUNDARY);
		}
		return result;
	}

	void BuildOwnership() {
		vector<LogicalPlanBuildTask> pending;
		pending.push_back({root, nullptr, DConstants::INVALID_INDEX});
		while (!pending.empty()) {
			auto task = pending.back();
			pending.pop_back();
			if (entry_map.find(task.op) != entry_map.end()) {
				valid = false;
				continue;
			}
			auto entry = make_uniq<LogicalPlanDataFlowEntry>(task.op);
			entry->owner_parent = task.owner_parent;
			entry->owner_child_index = task.owner_child_index;
			entry->opaque = IsOpaque(task.op.get());
			entry->node_value = GetNodeValue(task.op.get());
			forest.SetNodeValue(entry->forest_node, ToForestValue(entry->node_value));
			entry_map.emplace(task.op, entries.size());
			forest_operators.emplace(entry->forest_node, &task.op.get());
			entries.push_back(std::move(entry));

			for (idx_t child_idx = task.op.get().children.size(); child_idx > 0; child_idx--) {
				auto actual_idx = child_idx - 1;
				if (!task.op.get().children[actual_idx]) {
					valid = false;
					continue;
				}
				pending.push_back({reference<LogicalOperator>(*task.op.get().children[actual_idx]),
				                   optional_ptr<LogicalOperator>(task.op.get()), actual_idx});
			}
		}
	}

	void BuildFlow() {
		for (auto &entry : entries) {
			auto &parent = entry->op.get();
			for (idx_t child_idx = 0; child_idx < parent.children.size(); child_idx++) {
				if (!IsFlowChild(parent, child_idx)) {
					continue;
				}
				auto child_entry = GetEntry(*parent.children[child_idx]);
				if (!child_entry || child_entry->flow_parent) {
					valid = false;
					continue;
				}
				child_entry->flow_parent = parent;
				child_entry->flow_child_index = child_idx;
				auto parent_entry = GetEntry(parent);
				if (!parent_entry || !forest.Link(child_entry->forest_node, parent_entry->forest_node,
				                                  ToForestValue(child_entry->edge_to_parent))) {
					valid = false;
				}
			}
		}
	}

	optional_ptr<LogicalOperator> GetForestOperator(RootedDynamicForestNode &node) const {
		auto entry = forest_operators.find(node);
		if (entry == forest_operators.end()) {
			return nullptr;
		}
		return entry->second;
	}

	void RegisterSource(TableIndex table_index, LogicalOperator &op) {
		if (!table_index.IsValid()) {
			valid = false;
			return;
		}
		auto source = binding_sources.find(table_index);
		if (source != binding_sources.end() && source->second != &op) {
			valid = false;
			return;
		}
		binding_sources[table_index] = &op;
	}

	void RegisterProducer(TableIndex cte_index, LogicalOperator &producer) {
		auto &lineage = cte_lineage[cte_index];
		if (lineage.producer && lineage.producer.get() != &producer) {
			valid = false;
			return;
		}
		lineage.producer = producer;
	}

	void BuildSourcesAndLineage() {
		for (auto &entry : entries) {
			auto &op = entry->op.get();
			for (auto table_index : GetProducedTableIndexes(op)) {
				RegisterSource(table_index, op);
			}
			switch (op.type) {
			case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: {
				auto &cte = op.Cast<LogicalMaterializedCTE>();
				if (cte.children.size() != 2) {
					valid = false;
					break;
				}
				RegisterProducer(cte.table_index, *cte.children[0]);
				break;
			}
			case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
				auto &cte = op.Cast<LogicalRecursiveCTE>();
				RegisterProducer(cte.table_index, cte);
				break;
			}
			case LogicalOperatorType::LOGICAL_CTE_REF: {
				auto &cte_ref = op.Cast<LogicalCTERef>();
				cte_lineage[cte_ref.cte_index].readers.push_back(cte_ref);
				break;
			}
			default:
				break;
			}
		}
	}

	void BuildUses() {
		for (auto &entry : entries) {
			auto &op = entry->op.get();
			LogicalOperatorVisitor::EnumerateExpressions(
			    static_cast<const LogicalOperator &>(op), [&](const unique_ptr<Expression> *expression) {
				    ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
				        **expression, [&](const BoundColumnRefExpression &column_ref) {
					        binding_uses.push_back({column_ref.Binding(), column_ref.Depth(), op});
				        });
			    });
		}
	}

	optional_ptr<LogicalPlanDataFlowEntry> FlowRoot(LogicalPlanDataFlowEntry &entry) const {
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(entry);
		while (current->flow_parent) {
			current = GetEntry(*current->flow_parent);
			if (!current) {
				return nullptr;
			}
		}
		return current;
	}

	bool ParentIsAncestor(LogicalPlanDataFlowEntry &ancestor, LogicalPlanDataFlowEntry &descendant) const {
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(descendant);
		while (current) {
			if (&current->op.get() == &ancestor.op.get()) {
				return true;
			}
			if (!current->flow_parent) {
				return false;
			}
			current = GetEntry(*current->flow_parent);
		}
		return false;
	}

	optional_ptr<LogicalPlanDataFlowEntry> ParentLCA(LogicalPlanDataFlowEntry &left,
	                                                 LogicalPlanDataFlowEntry &right) const {
		if (FlowRoot(left) != FlowRoot(right)) {
			return nullptr;
		}
		unordered_set<LogicalOperator *> left_path;
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(left);
		while (current) {
			left_path.insert(&current->op.get());
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		current = right;
		while (current) {
			if (left_path.find(&current->op.get()) != left_path.end()) {
				return current;
			}
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		return nullptr;
	}

	bool ParentPathSummary(LogicalPlanDataFlowEntry &ancestor, LogicalPlanDataFlowEntry &descendant,
	                       LogicalPlanPathSummary &result) const {
		if (!ParentIsAncestor(ancestor, descendant)) {
			return false;
		}
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(descendant);
		while (current) {
			result.Merge(current->node_value);
			if (&current->op.get() == &ancestor.op.get()) {
				return true;
			}
			result.Merge(current->edge_to_parent);
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		return false;
	}

	LogicalPlanDataFlowStatus BoundaryBetween(LogicalPlanDataFlowEntry &left, LogicalPlanDataFlowEntry &right) const {
		unordered_set<LogicalOperator *> left_path;
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(left);
		while (current) {
			left_path.insert(&current->op.get());
			current = current->owner_parent ? GetEntry(*current->owner_parent) : nullptr;
		}
		auto right_cursor = optional_ptr<LogicalPlanDataFlowEntry>(right);
		while (right_cursor && left_path.find(&right_cursor->op.get()) == left_path.end()) {
			right_cursor = right_cursor->owner_parent ? GetEntry(*right_cursor->owner_parent) : nullptr;
		}
		auto ownership_lca = right_cursor;
		bool opaque = false;
		bool cte = false;
		for (auto cursor = optional_ptr<LogicalPlanDataFlowEntry>(left); cursor && cursor != ownership_lca;
		     cursor = cursor->owner_parent ? GetEntry(*cursor->owner_parent) : nullptr) {
			opaque = opaque || cursor->node_value.Has(LogicalPlanPathProperty::OPAQUE_BOUNDARY);
			cte = cte || cursor->node_value.Has(LogicalPlanPathProperty::CTE_BOUNDARY);
		}
		for (auto cursor = optional_ptr<LogicalPlanDataFlowEntry>(right); cursor && cursor != ownership_lca;
		     cursor = cursor->owner_parent ? GetEntry(*cursor->owner_parent) : nullptr) {
			opaque = opaque || cursor->node_value.Has(LogicalPlanPathProperty::OPAQUE_BOUNDARY);
			cte = cte || cursor->node_value.Has(LogicalPlanPathProperty::CTE_BOUNDARY);
		}
		if (ownership_lca) {
			opaque = opaque || ownership_lca->node_value.Has(LogicalPlanPathProperty::OPAQUE_BOUNDARY);
			cte = cte || ownership_lca->node_value.Has(LogicalPlanPathProperty::CTE_BOUNDARY);
		}
		if (opaque) {
			return LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY;
		}
		if (cte) {
			return LogicalPlanDataFlowStatus::CTE_BOUNDARY;
		}
		return LogicalPlanDataFlowStatus::DISCONNECTED;
	}

	bool ContainsBinding(const vector<ColumnBinding> &bindings, const ColumnBinding &binding) const {
		return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
	}

	bool ProjectionMapContains(LogicalOperator &child, const vector<ProjectionIndex> &projection_map,
	                           const ColumnBinding &binding) const {
		if (projection_map.empty()) {
			return true;
		}
		auto child_bindings = child.GetColumnBindings();
		for (auto projection_index : projection_map) {
			if (projection_index.GetIndex() < child_bindings.size() &&
			    child_bindings[projection_index.GetIndex()] == binding) {
				return true;
			}
		}
		return false;
	}

	idx_t ChildOnPath(LogicalPlanDataFlowEntry &source, LogicalPlanDataFlowEntry &parent) const {
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(source);
		while (current && current->flow_parent && current->flow_parent.get() != &parent.op.get()) {
			current = GetEntry(*current->flow_parent);
		}
		if (!current || !current->flow_parent) {
			return DConstants::INVALID_INDEX;
		}
		return current->flow_child_index;
	}

	bool JoinOutputContains(LogicalPlanDataFlowEntry &join_entry, LogicalPlanDataFlowEntry &source,
	                        const ColumnBinding &binding) const {
		auto &join = join_entry.op.get().Cast<LogicalJoin>();
		if (join.join_type == JoinType::MARK && binding.table_index == join.mark_index) {
			return true;
		}
		auto child_index = ChildOnPath(source, join_entry);
		if (child_index > 1) {
			return false;
		}
		switch (join.join_type) {
		case JoinType::SEMI:
		case JoinType::ANTI:
		case JoinType::MARK:
			if (child_index != 0) {
				return false;
			}
			break;
		case JoinType::RIGHT_SEMI:
		case JoinType::RIGHT_ANTI:
			if (child_index != 1) {
				return false;
			}
			break;
		default:
			break;
		}
		auto &projection_map = child_index == 0 ? join.left_projection_map : join.right_projection_map;
		return ProjectionMapContains(*join.children[child_index], projection_map, binding);
	}

	bool OutputContainsBinding(LogicalPlanDataFlowEntry &entry, LogicalPlanDataFlowEntry &source,
	                           const ColumnBinding &binding) const {
		if (&entry.op.get() == &source.op.get()) {
			return ContainsBinding(entry.op.get().GetColumnBindings(), binding);
		}
		auto &op = entry.op.get();
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_PIVOT:
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
			return false;
		case LogicalOperatorType::LOGICAL_FILTER: {
			auto &filter = op.Cast<LogicalFilter>();
			return ProjectionMapContains(*filter.children[0], filter.projection_map, binding);
		}
		case LogicalOperatorType::LOGICAL_ORDER_BY: {
			auto &order = op.Cast<LogicalOrder>();
			return ProjectionMapContains(*order.children[0], order.projection_map, binding);
		}
		case LogicalOperatorType::LOGICAL_GET: {
			auto &get = op.Cast<LogicalGet>();
			if (get.children.empty()) {
				return false;
			}
			return ContainsBinding(get.GetColumnBindings(), binding);
		}
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
			return JoinOutputContains(entry, source, binding);
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
			return false;
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
			return ContainsBinding(op.GetColumnBindings(), binding);
		default:
			return true;
		}
	}
};

LogicalPlanDataFlow::LogicalPlanDataFlow(LogicalOperator &root) : state(make_uniq<LogicalPlanDataFlowState>(root)) {
}

LogicalPlanDataFlow::~LogicalPlanDataFlow() {
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::ResolveSource(const ColumnBinding &binding, idx_t depth,
                                                                     LogicalOperator &consumer) const {
	if (depth != 0) {
		return {LogicalPlanDataFlowStatus::CORRELATED_REFERENCE, nullptr};
	}
	auto consumer_entry = state->GetEntry(consumer);
	if (!consumer_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr};
	}
	auto source = state->binding_sources.find(binding.table_index);
	if (source == state->binding_sources.end()) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_FOUND, nullptr};
	}
	auto source_entry = state->GetEntry(*source->second);
	if (!source_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr};
	}
	if (!state->forest.Connected(source_entry->forest_node, consumer_entry->forest_node)) {
		return {state->BoundaryBetween(*source_entry, *consumer_entry), nullptr};
	}
	if (!state->forest.IsAncestor(consumer_entry->forest_node, source_entry->forest_node)) {
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, nullptr};
	}
	if (!state->OutputContainsBinding(*source_entry, *source_entry, binding)) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE, nullptr};
	}
	auto current = source_entry;
	while (&current->op.get() != &consumer) {
		current = current->flow_parent ? state->GetEntry(*current->flow_parent) : nullptr;
		if (!current) {
			return {LogicalPlanDataFlowStatus::DISCONNECTED, nullptr};
		}
		if (&current->op.get() != &consumer && !state->OutputContainsBinding(*current, *source_entry, binding)) {
			return {LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE, nullptr};
		}
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, source->second};
}

LogicalPlanDataFlowParentResult LogicalPlanDataFlow::GetOwnershipParent(LogicalOperator &op) const {
	auto entry = state->GetEntry(op);
	if (!entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr, DConstants::INVALID_INDEX};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, entry->owner_parent, entry->owner_child_index};
}

LogicalPlanDataFlowParentResult LogicalPlanDataFlow::GetFlowParent(LogicalOperator &op) const {
	auto entry = state->GetEntry(op);
	if (!entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr, DConstants::INVALID_INDEX};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, entry->flow_parent, entry->flow_child_index};
}

LogicalPlanDataFlowBooleanResult LogicalPlanDataFlow::SameFlowTree(LogicalOperator &left,
                                                                   LogicalOperator &right) const {
	auto left_entry = state->GetEntry(left);
	auto right_entry = state->GetEntry(right);
	if (!left_entry || !right_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, false};
	}
	if (!state->forest.Connected(left_entry->forest_node, right_entry->forest_node)) {
		return {state->BoundaryBetween(*left_entry, *right_entry), false};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, true};
}

LogicalPlanDataFlowBooleanResult LogicalPlanDataFlow::IsFlowAncestor(LogicalOperator &ancestor,
                                                                     LogicalOperator &descendant) const {
	auto ancestor_entry = state->GetEntry(ancestor);
	auto descendant_entry = state->GetEntry(descendant);
	if (!ancestor_entry || !descendant_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, false};
	}
	if (!state->forest.Connected(ancestor_entry->forest_node, descendant_entry->forest_node)) {
		return {state->BoundaryBetween(*ancestor_entry, *descendant_entry), false};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS,
	        state->forest.IsAncestor(ancestor_entry->forest_node, descendant_entry->forest_node)};
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::LowestCommonAncestor(LogicalOperator &left,
                                                                            LogicalOperator &right) const {
	auto left_entry = state->GetEntry(left);
	auto right_entry = state->GetEntry(right);
	if (!left_entry || !right_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr};
	}
	if (!state->forest.Connected(left_entry->forest_node, right_entry->forest_node)) {
		return {state->BoundaryBetween(*left_entry, *right_entry), nullptr};
	}
	auto lca = state->forest.LowestCommonAncestor(left_entry->forest_node, right_entry->forest_node);
	if (!lca) {
		return {LogicalPlanDataFlowStatus::DISCONNECTED, nullptr};
	}
	auto lca_operator = state->GetForestOperator(*lca);
	if (!lca_operator) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, lca_operator};
}

LogicalPlanDataFlowPathResult LogicalPlanDataFlow::GetPathSummary(LogicalOperator &ancestor,
                                                                  LogicalOperator &descendant) const {
	auto ancestor_entry = state->GetEntry(ancestor);
	auto descendant_entry = state->GetEntry(descendant);
	if (!ancestor_entry || !descendant_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, {}};
	}
	if (!state->forest.Connected(ancestor_entry->forest_node, descendant_entry->forest_node)) {
		return {state->BoundaryBetween(*ancestor_entry, *descendant_entry), {}};
	}
	if (!state->forest.IsAncestor(ancestor_entry->forest_node, descendant_entry->forest_node)) {
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, {}};
	}
	RootedDynamicForestPathValue result;
	if (!state->forest.GetPathValue(ancestor_entry->forest_node, descendant_entry->forest_node, result)) {
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, {}};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, FromForestValue(result)};
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::GetCTEProducer(TableIndex cte_index) const {
	auto lineage = state->cte_lineage.find(cte_index);
	if (lineage == state->cte_lineage.end() || !lineage->second.producer) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_FOUND, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, lineage->second.producer};
}

LogicalPlanDataFlowReadersResult LogicalPlanDataFlow::GetCTEReaders(TableIndex cte_index) const {
	auto lineage = state->cte_lineage.find(cte_index);
	if (lineage == state->cte_lineage.end()) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_FOUND, {}};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, lineage->second.readers};
}

const vector<LogicalPlanBindingUse> &LogicalPlanDataFlow::GetBindingUses() const {
	return state->binding_uses;
}

idx_t LogicalPlanDataFlow::OperatorCount() const {
	return state->entries.size();
}

bool LogicalPlanDataFlow::Verify() const {
	if (!state->valid || state->entries.empty() || &state->entries[0]->op.get() != &state->root.get()) {
		return false;
	}
	idx_t ownership_count = 0;
	vector<reference<LogicalOperator>> pending {state->root};
	while (!pending.empty()) {
		auto op = pending.back();
		pending.pop_back();
		auto entry = state->GetEntry(op);
		if (!entry) {
			return false;
		}
		ownership_count++;
		for (idx_t child_idx = 0; child_idx < op.get().children.size(); child_idx++) {
			auto child_entry = state->GetEntry(*op.get().children[child_idx]);
			if (!child_entry || child_entry->owner_parent.get() != &op.get() ||
			    child_entry->owner_child_index != child_idx) {
				return false;
			}
			if (state->IsFlowChild(op, child_idx)) {
				if (child_entry->flow_parent.get() != &op.get() || child_entry->flow_child_index != child_idx) {
					return false;
				}
			} else if (child_entry->flow_parent.get() == &op.get()) {
				return false;
			}
			pending.push_back(*op.get().children[child_idx]);
		}
	}
	if (ownership_count != state->entries.size()) {
		return false;
	}
	for (auto &entry : state->entries) {
		auto represented_parent = state->forest.GetRepresentedParent(entry->forest_node);
		auto forest_parent = represented_parent ? state->GetForestOperator(*represented_parent) : nullptr;
		if (forest_parent != entry->flow_parent) {
			return false;
		}
		auto expected_root = state->FlowRoot(*entry);
		auto forest_root = state->GetForestOperator(state->forest.FindRoot(entry->forest_node));
		if (!expected_root || forest_root.get() != &expected_root->op.get()) {
			return false;
		}
		LogicalPlanPathSummary expected_path;
		if (!state->ParentPathSummary(*expected_root, *entry, expected_path)) {
			return false;
		}
		if (FromForestValue(state->forest.GetRootPathValue(entry->forest_node)) != expected_path) {
			return false;
		}
	}
	constexpr idx_t ALL_PAIRS_LIMIT = 128;
	if (state->entries.size() <= ALL_PAIRS_LIMIT) {
		for (auto &left : state->entries) {
			for (auto &right : state->entries) {
				const bool expected_connected = state->FlowRoot(*left) == state->FlowRoot(*right);
				if (state->forest.Connected(left->forest_node, right->forest_node) != expected_connected) {
					return false;
				}
				auto expected_lca = state->ParentLCA(*left, *right);
				auto forest_lca = state->forest.LowestCommonAncestor(left->forest_node, right->forest_node);
				auto actual_lca = forest_lca ? state->GetForestOperator(*forest_lca) : nullptr;
				if ((expected_lca ? &expected_lca->op.get() : nullptr) != actual_lca.get()) {
					return false;
				}
				const bool expected_ancestor = state->ParentIsAncestor(*left, *right);
				if (state->forest.IsAncestor(left->forest_node, right->forest_node) != expected_ancestor) {
					return false;
				}
				RootedDynamicForestPathValue forest_path;
				const bool has_path = state->forest.GetPathValue(left->forest_node, right->forest_node, forest_path);
				if (has_path != expected_ancestor) {
					return false;
				}
				if (has_path) {
					LogicalPlanPathSummary expected_path;
					if (!state->ParentPathSummary(*left, *right, expected_path) ||
					    FromForestValue(forest_path) != expected_path) {
						return false;
					}
				}
			}
		}
	}
	for (auto &use : state->binding_uses) {
		auto result = ResolveSource(use.binding, use.depth, use.consumer);
		if (use.depth != 0) {
			if (result.status != LogicalPlanDataFlowStatus::CORRELATED_REFERENCE) {
				return false;
			}
			continue;
		}
		if (result.status == LogicalPlanDataFlowStatus::SUCCESS ||
		    result.status == LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY ||
		    result.status == LogicalPlanDataFlowStatus::CTE_BOUNDARY) {
			continue;
		}
		auto consumer_entry = state->GetEntry(use.consumer);
		if (!consumer_entry || !consumer_entry->opaque) {
			return false;
		}
	}
	return true;
}

} // namespace duckdb
