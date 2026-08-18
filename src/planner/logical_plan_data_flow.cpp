#include "duckdb/planner/logical_plan_data_flow.hpp"

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/detail/rooted_dynamic_forest.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_dependent_join.hpp"
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
	return {summary.properties, summary.minimum_cardinality};
}

static LogicalPlanPathSummary FromForestValue(const RootedDynamicForestPathValue &value) {
	return {value.flags, value.minimum_cardinality};
}

static uint64_t PathPropertyMask(LogicalPlanPathProperty property) {
	return uint64_t(1) << NumericCast<uint64_t>(property);
}

void LogicalPlanPathSummary::Add(LogicalPlanPathProperty property) {
	properties |= PathPropertyMask(property);
}

void LogicalPlanPathSummary::Merge(const LogicalPlanPathSummary &other) {
	properties |= other.properties;
	if (other.minimum_cardinality.IsValid() &&
	    (!minimum_cardinality.IsValid() || other.minimum_cardinality.GetIndex() < minimum_cardinality.GetIndex())) {
		minimum_cardinality = other.minimum_cardinality;
	}
}

bool LogicalPlanPathSummary::Has(LogicalPlanPathProperty property) const {
	return (properties & PathPropertyMask(property)) != 0;
}

bool LogicalPlanPathSummary::operator==(const LogicalPlanPathSummary &other) const {
	return properties == other.properties && minimum_cardinality == other.minimum_cardinality;
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
	vector<TableIndex> produced_table_indexes;
	TableIndex cte_producer_index;
	optional_ptr<LogicalOperator> cte_producer;
	TableIndex cte_reader_index;
	vector<LogicalPlanBindingUse> binding_uses;
	vector<ColumnBinding> correlated_bindings;
};

struct LogicalPlanCTELineage {
	optional_ptr<LogicalOperator> producer_owner;
	optional_ptr<LogicalOperator> producer;
	vector<reference<LogicalOperator>> readers;
};

struct LogicalPlanBuildTask {
	reference<LogicalOperator> op;
	optional_ptr<LogicalOperator> owner_parent;
	idx_t owner_child_index;
};

struct LogicalPlanDataFlowMetadata {
	vector<TableIndex> produced_table_indexes;
	TableIndex cte_producer_index;
	optional_ptr<LogicalOperator> cte_producer;
	TableIndex cte_reader_index;
	vector<LogicalPlanBindingUse> binding_uses;
	vector<ColumnBinding> correlated_bindings;
};

class LogicalPlanDataFlowState {
public:
	explicit LogicalPlanDataFlowState(LogicalOperator &root_p) : root(root_p) {
		BuildOwnership();
		BuildFlow();
		BuildSourcesAndLineage();
		BuildUses();
	}

	optional_ptr<LogicalOperator> root;
	reference_map_t<LogicalOperator, unique_ptr<LogicalPlanDataFlowEntry>> entries;
	RootedDynamicForest forest;
	reference_map_t<RootedDynamicForestNode, reference<LogicalOperator>> forest_operators;
	unordered_map<TableIndex, reference<LogicalOperator>> binding_sources;
	unordered_map<TableIndex, LogicalPlanCTELineage> cte_lineage;
	column_binding_map_t<reference_set_t<LogicalOperator>> correlated_operators;
	mutable vector<LogicalPlanBindingUse> binding_uses;
	mutable bool binding_uses_dirty = false;
	idx_t active_mutation_scopes = 0;
	bool valid = true;

public:
	void EnsureQueryable() const {
		if (active_mutation_scopes != 0) {
			throw InternalException("Cannot query logical plan data flow during a coordinated mutation");
		}
	}

	optional_ptr<LogicalPlanDataFlowEntry> GetEntry(LogicalOperator &op) const {
		auto entry = entries.find(op);
		if (entry == entries.end()) {
			return nullptr;
		}
		return *entry->second;
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
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
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

	bool SupportsUnaryInsertion(const LogicalOperator &op) const {
		switch (op.type) {
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
		case LogicalOperatorType::LOGICAL_GET:
			return true;
		default:
			return false;
		}
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

	bool ChangesBindingAvailability(const LogicalOperator &op) const {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_PIVOT:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
		case LogicalOperatorType::LOGICAL_UNION:
		case LogicalOperatorType::LOGICAL_EXCEPT:
		case LogicalOperatorType::LOGICAL_INTERSECT:
		case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
			return true;
		case LogicalOperatorType::LOGICAL_FILTER:
			return !op.Cast<LogicalFilter>().projection_map.empty();
		case LogicalOperatorType::LOGICAL_ORDER_BY:
			return !op.Cast<LogicalOrder>().projection_map.empty();
		case LogicalOperatorType::LOGICAL_GET:
			return !op.children.empty();
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN: {
			auto &join = op.Cast<LogicalJoin>();
			if (join.HasProjectionMap()) {
				return true;
			}
			switch (join.join_type) {
			case JoinType::SEMI:
			case JoinType::ANTI:
			case JoinType::MARK:
			case JoinType::RIGHT_SEMI:
			case JoinType::RIGHT_ANTI:
				return true;
			default:
				return false;
			}
		}
		default:
			return false;
		}
	}

	LogicalPlanPathSummary GetNodeValue(const LogicalOperator &op) const {
		LogicalPlanPathSummary result;
		if (op.has_estimated_cardinality) {
			result.minimum_cardinality = op.estimated_cardinality;
		}
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
		if (op.type != LogicalOperatorType::LOGICAL_ORDER_BY &&
		    (op.type != LogicalOperatorType::LOGICAL_DISTINCT || op.Cast<LogicalDistinct>().order_by)) {
			result.Add(LogicalPlanPathProperty::FILTER_PUSHDOWN_BOUNDARY);
		}
		if (ChangesBindingAvailability(op)) {
			result.Add(LogicalPlanPathProperty::BINDING_AVAILABILITY_BOUNDARY);
		}
		if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
		    op.Cast<LogicalComparisonJoin>().join_type == JoinType::INVALID) {
			result.Add(LogicalPlanPathProperty::NULLABILITY_BOUNDARY);
		}
		return result;
	}

	LogicalPlanPathSummary GetEdgeValue(const LogicalOperator &parent, idx_t child_idx) const {
		LogicalPlanPathSummary result;
		if (parent.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || child_idx > 1) {
			return result;
		}
		auto &join = parent.Cast<LogicalComparisonJoin>();
		switch (join.join_type) {
		case JoinType::LEFT:
		case JoinType::SINGLE:
			if (child_idx == 1) {
				result.Add(LogicalPlanPathProperty::NULL_EXTENDING);
			}
			break;
		case JoinType::RIGHT:
			if (child_idx == 0) {
				result.Add(LogicalPlanPathProperty::NULL_EXTENDING);
			}
			break;
		case JoinType::OUTER:
			result.Add(LogicalPlanPathProperty::NULL_EXTENDING);
			break;
		case JoinType::INNER:
		case JoinType::SEMI:
		case JoinType::ANTI:
		case JoinType::MARK:
		case JoinType::RIGHT_SEMI:
		case JoinType::RIGHT_ANTI:
		case JoinType::INVALID:
			break;
		}
		return result;
	}

	void BuildOwnership() {
		vector<LogicalPlanBuildTask> pending;
		pending.push_back({*root, nullptr, DConstants::INVALID_INDEX});
		while (!pending.empty()) {
			auto task = pending.back();
			pending.pop_back();
			if (entries.find(task.op) != entries.end()) {
				valid = false;
				continue;
			}
			auto entry = make_uniq<LogicalPlanDataFlowEntry>(task.op);
			entry->owner_parent = task.owner_parent;
			entry->owner_child_index = task.owner_child_index;
			entry->opaque = IsOpaque(task.op.get());
			entry->node_value = GetNodeValue(task.op.get());
			forest.SetNodeValue(entry->forest_node, ToForestValue(entry->node_value));
			forest_operators.emplace(entry->forest_node, task.op);
			entries.emplace(task.op, std::move(entry));

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
		for (auto &entry_pair : entries) {
			auto &parent = entry_pair.second->op.get();
			for (idx_t child_idx = 0; child_idx < parent.children.size(); child_idx++) {
				if (!IsFlowChild(parent, child_idx)) {
					continue;
				}
				auto child_entry = GetEntry(*parent.children[child_idx]);
				if (!child_entry || child_entry->flow_parent) {
					valid = false;
					continue;
				}
				child_entry->edge_to_parent = GetEdgeValue(parent, child_idx);
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
		return entry->second.get();
	}

	bool RegisterSource(TableIndex table_index, LogicalOperator &op) {
		if (!table_index.IsValid()) {
			return false;
		}
		auto source = binding_sources.find(table_index);
		if (source != binding_sources.end() && &source->second.get() != &op) {
			return false;
		}
		if (source == binding_sources.end()) {
			binding_sources.emplace(table_index, op);
		}
		return true;
	}

	bool RegisterProducer(TableIndex cte_index, LogicalOperator &owner, LogicalOperator &producer) {
		auto &lineage = cte_lineage[cte_index];
		if (lineage.producer_owner && lineage.producer_owner.get() != &owner) {
			return false;
		}
		lineage.producer_owner = owner;
		lineage.producer = producer;
		return true;
	}

	bool GetMetadata(LogicalOperator &op, LogicalPlanDataFlowMetadata &result) const {
		unordered_set<TableIndex> produced_table_indexes;
		column_binding_set_t correlated_bindings;
		auto add_correlated_binding = [&](const ColumnBinding &binding) {
			if (correlated_bindings.insert(binding).second) {
				result.correlated_bindings.push_back(binding);
			}
		};
		for (auto table_index : GetProducedTableIndexes(op)) {
			if (!table_index.IsValid()) {
				return false;
			}
			if (produced_table_indexes.insert(table_index).second) {
				result.produced_table_indexes.push_back(table_index);
			}
		}
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: {
			auto &cte = op.Cast<LogicalMaterializedCTE>();
			if (cte.children.size() != 2 || !cte.children[0] || !cte.table_index.IsValid()) {
				return false;
			}
			result.cte_producer_index = cte.table_index;
			result.cte_producer = *cte.children[0];
			break;
		}
		case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
			auto &cte = op.Cast<LogicalRecursiveCTE>();
			if (!cte.table_index.IsValid()) {
				return false;
			}
			result.cte_producer_index = cte.table_index;
			result.cte_producer = cte;
			break;
		}
		case LogicalOperatorType::LOGICAL_CTE_REF: {
			auto &cte_ref = op.Cast<LogicalCTERef>();
			if (!cte_ref.cte_index.IsValid()) {
				return false;
			}
			result.cte_reader_index = cte_ref.cte_index;
			for (auto &binding : cte_ref.correlated_bindings) {
				add_correlated_binding(binding);
			}
			break;
		}
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN: {
			for (auto &column : op.Cast<LogicalDependentJoin>().correlated_columns) {
				add_correlated_binding(column.binding);
			}
			break;
		}
		default:
			break;
		}
		LogicalOperatorVisitor::EnumerateExpressions(
		    static_cast<const LogicalOperator &>(op), [&](optional_ptr<const unique_ptr<Expression>> expression) {
			    ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
			        **expression, [&](const BoundColumnRefExpression &column_ref) {
				        result.binding_uses.push_back({column_ref.Binding(), column_ref.Depth(), op});
				        if (column_ref.Depth() > 0) {
					        add_correlated_binding(column_ref.Binding());
				        }
			        });
		    });
		return true;
	}

	bool AddContributions(LogicalPlanDataFlowEntry &entry, LogicalPlanDataFlowMetadata metadata) {
		auto &op = entry.op.get();
		for (auto table_index : metadata.produced_table_indexes) {
			if (!RegisterSource(table_index, op)) {
				return false;
			}
		}
		if (metadata.cte_producer_index.IsValid() &&
		    !RegisterProducer(metadata.cte_producer_index, op, *metadata.cte_producer)) {
			return false;
		}
		if (metadata.cte_reader_index.IsValid()) {
			cte_lineage[metadata.cte_reader_index].readers.push_back(op);
		}
		for (auto &binding : metadata.correlated_bindings) {
			correlated_operators[binding].insert(op);
		}
		entry.produced_table_indexes = std::move(metadata.produced_table_indexes);
		entry.cte_producer_index = metadata.cte_producer_index;
		entry.cte_producer = metadata.cte_producer;
		entry.cte_reader_index = metadata.cte_reader_index;
		entry.binding_uses = std::move(metadata.binding_uses);
		entry.correlated_bindings = std::move(metadata.correlated_bindings);
		binding_uses_dirty = true;
		return true;
	}

	void RemoveContributions(LogicalPlanDataFlowEntry &entry) {
		auto &op = entry.op.get();
		for (auto table_index : entry.produced_table_indexes) {
			auto source = binding_sources.find(table_index);
			if (source != binding_sources.end() && &source->second.get() == &op) {
				binding_sources.erase(source);
			}
		}
		if (entry.cte_producer_index.IsValid()) {
			auto lineage = cte_lineage.find(entry.cte_producer_index);
			if (lineage != cte_lineage.end() && lineage->second.producer_owner.get() == &op) {
				lineage->second.producer_owner = nullptr;
				lineage->second.producer = nullptr;
				if (lineage->second.readers.empty()) {
					cte_lineage.erase(lineage);
				}
			}
		}
		if (entry.cte_reader_index.IsValid()) {
			auto lineage = cte_lineage.find(entry.cte_reader_index);
			if (lineage != cte_lineage.end()) {
				auto &readers = lineage->second.readers;
				readers.erase(
				    std::remove_if(readers.begin(), readers.end(),
				                   [&](const reference<LogicalOperator> &reader) { return &reader.get() == &op; }),
				    readers.end());
				if (!lineage->second.producer && readers.empty()) {
					cte_lineage.erase(lineage);
				}
			}
		}
		for (auto &binding : entry.correlated_bindings) {
			auto correlated = correlated_operators.find(binding);
			if (correlated == correlated_operators.end()) {
				continue;
			}
			correlated->second.erase(op);
			if (correlated->second.empty()) {
				correlated_operators.erase(correlated);
			}
		}
		entry.produced_table_indexes.clear();
		entry.cte_producer_index = TableIndex();
		entry.cte_producer = nullptr;
		entry.cte_reader_index = TableIndex();
		entry.binding_uses.clear();
		entry.correlated_bindings.clear();
		binding_uses_dirty = true;
	}

	void BuildSourcesAndLineage() {
		vector<reference<LogicalOperator>> pending {*root};
		while (!pending.empty()) {
			auto op = pending.back();
			pending.pop_back();
			auto entry = GetEntry(op);
			LogicalPlanDataFlowMetadata metadata;
			if (!entry || !GetMetadata(op, metadata) || !AddContributions(*entry, std::move(metadata))) {
				valid = false;
			}
			for (idx_t child_idx = op.get().children.size(); child_idx > 0; child_idx--) {
				pending.push_back(*op.get().children[child_idx - 1]);
			}
		}
	}

	void BuildUses() {
		RebuildBindingUses();
	}

	void RebuildBindingUses() const {
		binding_uses.clear();
		for (auto &entry_pair : entries) {
			for (auto &use : entry_pair.second->binding_uses) {
				binding_uses.push_back(use);
			}
		}
		binding_uses_dirty = false;
	}

	void
	ValidateNewSubtree(LogicalOperator &subtree,
	                   optional_ptr<const reference_set_t<const LogicalOperator>> ignored_operators = nullptr) const {
		vector<reference<LogicalOperator>> pending {subtree};
		reference_set_t<LogicalOperator> seen;
		unordered_map<TableIndex, reference<LogicalOperator>> new_sources;
		unordered_map<TableIndex, reference<LogicalOperator>> new_producers;
		while (!pending.empty()) {
			auto op = pending.back();
			pending.pop_back();
			if (!seen.insert(op).second) {
				throw InternalException("Cannot register a logical plan subtree with repeated operators");
			}
			if (GetEntry(op)) {
				throw InternalException("Cannot register an operator that is already indexed");
			}
			LogicalPlanDataFlowMetadata metadata;
			if (!GetMetadata(op, metadata)) {
				throw InternalException("Cannot register a logical plan subtree with invalid operator metadata");
			}
			for (auto table_index : metadata.produced_table_indexes) {
				auto existing = binding_sources.find(table_index);
				if (existing != binding_sources.end() && &existing->second.get() != &op.get() &&
				    (!ignored_operators || ignored_operators->find(reference<const LogicalOperator>(
				                               existing->second)) == ignored_operators->end())) {
					throw InternalException("Cannot register duplicate logical plan table index %llu",
					                        table_index.index);
				}
				auto inserted = new_sources.emplace(table_index, op);
				if (!inserted.second && &inserted.first->second.get() != &op.get()) {
					throw InternalException("Cannot register duplicate logical plan table index %llu",
					                        table_index.index);
				}
			}
			if (metadata.cte_producer_index.IsValid()) {
				auto existing = cte_lineage.find(metadata.cte_producer_index);
				if (existing != cte_lineage.end() && existing->second.producer_owner &&
				    (!ignored_operators || ignored_operators->find(reference<const LogicalOperator>(
				                               *existing->second.producer_owner)) == ignored_operators->end())) {
					throw InternalException("Cannot register duplicate logical plan CTE producer %llu",
					                        metadata.cte_producer_index.index);
				}
				if (!new_producers.emplace(metadata.cte_producer_index, op).second) {
					throw InternalException("Cannot register duplicate logical plan CTE producer %llu",
					                        metadata.cte_producer_index.index);
				}
			}
			for (auto &child : op.get().children) {
				if (!child) {
					throw InternalException("Cannot register a logical plan subtree with a null child");
				}
				pending.push_back(*child);
			}
		}
	}

	void RegisterSubtree(LogicalOperator &subtree) {
		ValidateNewSubtree(subtree);
		vector<LogicalPlanBuildTask> pending {{subtree, nullptr, DConstants::INVALID_INDEX}};
		vector<reference<LogicalOperator>> registered;
		while (!pending.empty()) {
			auto task = pending.back();
			pending.pop_back();
			auto entry = make_uniq<LogicalPlanDataFlowEntry>(task.op);
			entry->owner_parent = task.owner_parent;
			entry->owner_child_index = task.owner_child_index;
			entry->opaque = IsOpaque(task.op);
			entry->node_value = GetNodeValue(task.op);
			forest.SetNodeValue(entry->forest_node, ToForestValue(entry->node_value));
			forest_operators.emplace(entry->forest_node, task.op);
			entries.emplace(task.op, std::move(entry));
			registered.push_back(task.op);
			for (idx_t child_idx = task.op.get().children.size(); child_idx > 0; child_idx--) {
				auto actual_idx = child_idx - 1;
				pending.push_back(
				    {*task.op.get().children[actual_idx], optional_ptr<LogicalOperator>(task.op.get()), actual_idx});
			}
		}
		for (auto op : registered) {
			LogicalPlanDataFlowMetadata metadata;
			const bool has_metadata = GetMetadata(op, metadata);
			D_ASSERT(has_metadata);
			const bool added = AddContributions(*GetEntry(op), std::move(metadata));
			D_ASSERT(added);
		}
		for (auto op : registered) {
			RefreshFlowChildren(op);
		}
	}

	void UnregisterSubtree(LogicalOperator &subtree) {
		auto subtree_entry = GetEntry(subtree);
		if (!subtree_entry || subtree_entry->owner_parent) {
			throw InternalException("Can only unregister the root of a detached indexed subtree");
		}
		if (root.get() == &subtree) {
			throw InternalException("Cannot unregister the main logical plan root");
		}
		vector<reference<LogicalOperator>> pending {subtree};
		vector<reference<LogicalOperator>> registered;
		while (!pending.empty()) {
			auto op = pending.back();
			pending.pop_back();
			auto entry = GetEntry(op);
			if (!entry) {
				throw InternalException("Detached logical plan subtree is not fully indexed");
			}
			registered.push_back(op);
			for (idx_t child_idx = 0; child_idx < op.get().children.size(); child_idx++) {
				auto &child = op.get().children[child_idx];
				if (!child) {
					throw InternalException("Detached logical plan subtree contains a null child");
				}
				auto child_entry = GetEntry(*child);
				if (!child_entry || child_entry->owner_parent.get() != &op.get() ||
				    child_entry->owner_child_index != child_idx) {
					throw InternalException("Detached logical plan ownership changed outside its indexed mutator");
				}
				pending.push_back(*child);
			}
		}
		for (auto op : registered) {
			auto entry = GetEntry(op);
			RemoveContributions(*entry);
			if (entry->flow_parent) {
				const bool cut = forest.CutFromParent(entry->forest_node);
				D_ASSERT(cut);
				entry->flow_parent = nullptr;
				entry->flow_child_index = DConstants::INVALID_INDEX;
			}
		}
		for (auto op : registered) {
			auto entry = GetEntry(op);
			forest_operators.erase(entry->forest_node);
			entries.erase(op);
		}
	}

	void RefreshFlowChildren(LogicalOperator &parent) {
		auto parent_entry = GetEntry(parent);
		if (!parent_entry) {
			throw InternalException("Cannot refresh an operator that is not indexed");
		}
		for (idx_t child_idx = 0; child_idx < parent.children.size(); child_idx++) {
			auto &child = parent.children[child_idx];
			if (!child) {
				throw InternalException("Cannot refresh an operator with a null child");
			}
			auto child_entry = GetEntry(*child);
			if (!child_entry || child_entry->owner_parent.get() != &parent ||
			    child_entry->owner_child_index != child_idx) {
				throw InternalException("Logical plan ownership changed outside its indexed mutator");
			}
			const bool should_link = IsFlowChild(parent, child_idx);
			const bool is_linked = child_entry->flow_parent.get() == &parent;
			const auto edge_to_parent = should_link ? GetEdgeValue(parent, child_idx) : LogicalPlanPathSummary();
			if (child_entry->flow_parent && !is_linked) {
				throw InternalException("Logical plan flow child already has another parent");
			}
			if (is_linked && !should_link) {
				const bool cut = forest.CutFromParent(child_entry->forest_node);
				D_ASSERT(cut);
				child_entry->flow_parent = nullptr;
				child_entry->flow_child_index = DConstants::INVALID_INDEX;
				child_entry->edge_to_parent = {};
			} else if (!is_linked && should_link) {
				child_entry->edge_to_parent = edge_to_parent;
				child_entry->flow_parent = parent;
				child_entry->flow_child_index = child_idx;
				const bool linked = forest.Link(child_entry->forest_node, parent_entry->forest_node,
				                                ToForestValue(child_entry->edge_to_parent));
				if (!linked) {
					throw InternalException("Cannot create a cyclic logical plan flow edge");
				}
			} else if (is_linked) {
				child_entry->edge_to_parent = edge_to_parent;
				child_entry->flow_child_index = child_idx;
				const bool updated =
				    forest.SetEdgeValue(child_entry->forest_node, ToForestValue(child_entry->edge_to_parent));
				D_ASSERT(updated);
			}
		}
	}

	void RefreshOperator(LogicalOperator &op) {
		auto entry = GetEntry(op);
		if (!entry) {
			throw InternalException("Cannot refresh an operator that is not indexed");
		}
		LogicalPlanDataFlowMetadata metadata;
		ValidateRefreshOperator(op, metadata);
		RemoveContributions(*entry);
		const bool added = AddContributions(*entry, std::move(metadata));
		D_ASSERT(added);
		entry->opaque = IsOpaque(op);
		entry->node_value = GetNodeValue(op);
		forest.SetNodeValue(entry->forest_node, ToForestValue(entry->node_value));
		RefreshFlowChildren(op);
	}

	void ValidateRefreshOperator(
	    LogicalOperator &op, LogicalPlanDataFlowMetadata &metadata,
	    optional_ptr<const reference_set_t<const LogicalOperator>> ignored_operators = nullptr) const {
		if (!GetMetadata(op, metadata)) {
			throw InternalException("Cannot refresh invalid logical plan operator metadata");
		}
		for (auto table_index : metadata.produced_table_indexes) {
			auto source = binding_sources.find(table_index);
			if (source != binding_sources.end() && &source->second.get() != &op &&
			    (!ignored_operators || ignored_operators->find(reference<const LogicalOperator>(source->second)) ==
			                               ignored_operators->end())) {
				throw InternalException("Cannot refresh duplicate logical plan table index %llu", table_index.index);
			}
		}
		if (metadata.cte_producer_index.IsValid()) {
			auto lineage = cte_lineage.find(metadata.cte_producer_index);
			if (lineage != cte_lineage.end() && lineage->second.producer_owner &&
			    lineage->second.producer_owner.get() != &op &&
			    (!ignored_operators || ignored_operators->find(reference<const LogicalOperator>(
			                               *lineage->second.producer_owner)) == ignored_operators->end())) {
				throw InternalException("Cannot refresh duplicate logical plan CTE producer %llu",
				                        metadata.cte_producer_index.index);
			}
		}
	}

	void ValidateChildren(LogicalOperator &parent) const {
		if (!GetEntry(parent)) {
			throw InternalException("Cannot mutate children of an operator that is not indexed");
		}
		for (idx_t child_idx = 0; child_idx < parent.children.size(); child_idx++) {
			auto &child = parent.children[child_idx];
			if (!child) {
				throw InternalException("Cannot mutate an operator with a null child");
			}
			auto child_entry = GetEntry(*child);
			if (!child_entry || child_entry->owner_parent.get() != &parent ||
			    child_entry->owner_child_index != child_idx) {
				throw InternalException("Logical plan ownership changed outside its indexed mutator");
			}
		}
	}

	void ValidateOperatorMetadata(LogicalOperator &parent) const {
		ValidateChildren(parent);
		LogicalPlanDataFlowMetadata metadata;
		auto entry = GetEntry(parent);
		if (!GetMetadata(parent, metadata) || metadata.produced_table_indexes != entry->produced_table_indexes ||
		    metadata.cte_producer_index != entry->cte_producer_index || metadata.cte_producer != entry->cte_producer ||
		    metadata.cte_reader_index != entry->cte_reader_index ||
		    metadata.binding_uses.size() != entry->binding_uses.size() ||
		    metadata.correlated_bindings != entry->correlated_bindings) {
			throw InternalException("Logical plan operator metadata changed without RefreshOperator");
		}
		for (idx_t use_idx = 0; use_idx < metadata.binding_uses.size(); use_idx++) {
			auto &expected = metadata.binding_uses[use_idx];
			auto &actual = entry->binding_uses[use_idx];
			if (expected.binding != actual.binding || expected.depth != actual.depth ||
			    &expected.consumer.get() != &actual.consumer.get()) {
				throw InternalException("Logical plan operator expressions changed without RefreshOperator");
			}
		}
	}

	void ValidateStructuralMutation(LogicalOperator &parent, idx_t future_child_count) const {
		ValidateOperatorMetadata(parent);
		ValidateFutureChildCount(parent, future_child_count);
	}

	void ValidateSlotReplacement(LogicalOperator &parent) const {
		ValidateOperatorMetadata(parent);
		if (parent.type != LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
			ValidateFutureChildCount(parent, parent.children.size());
		}
	}

	void ValidateFutureChildCount(const LogicalOperator &parent, idx_t future_child_count) const {
		if (parent.type == LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR) {
			throw InternalException("Cannot incrementally mutate children of an opaque extension operator");
		}
		if (parent.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && future_child_count != 2) {
			throw InternalException("A materialized CTE must retain its producer and continuation children");
		}
	}

	bool SupportsParentInsertion(const LogicalOperator &op) const {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
		case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
		case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
			return true;
		default:
			return false;
		}
	}

	void ValidateAttachChild(LogicalOperator &parent, idx_t child_index, LogicalOperator &subtree) const {
		ValidateChildren(parent);
		if (child_index > parent.children.size()) {
			throw InternalException("Cannot attach logical plan child index %llu", child_index);
		}
		auto subtree_entry = GetEntry(subtree);
		if (!subtree_entry || subtree_entry->owner_parent) {
			throw InternalException("Can only attach the root of a detached indexed subtree");
		}
		for (auto current = GetEntry(parent); current;
		     current = current->owner_parent ? GetEntry(*current->owner_parent) : nullptr) {
			if (&current->op.get() == &subtree) {
				throw InternalException("Cannot create a cycle in logical plan ownership");
			}
		}
	}

	unique_ptr<LogicalOperator> DetachChild(LogicalOperator &parent, idx_t child_index, bool refresh_parent) {
		ValidateChildren(parent);
		if (child_index >= parent.children.size()) {
			throw InternalException("Cannot detach logical plan child index %llu", child_index);
		}
		auto &child = *parent.children[child_index];
		auto child_entry = GetEntry(child);
		if (child_entry->flow_parent.get() == &parent) {
			const bool cut = forest.CutFromParent(child_entry->forest_node);
			D_ASSERT(cut);
			child_entry->flow_parent = nullptr;
			child_entry->flow_child_index = DConstants::INVALID_INDEX;
		}
		auto result = std::move(parent.children[child_index]);
		parent.children.erase(parent.children.begin() + NumericCast<int64_t>(child_index));
		child_entry->owner_parent = nullptr;
		child_entry->owner_child_index = DConstants::INVALID_INDEX;
		for (idx_t shifted_idx = child_index; shifted_idx < parent.children.size(); shifted_idx++) {
			auto shifted_entry = GetEntry(*parent.children[shifted_idx]);
			D_ASSERT(shifted_entry);
			shifted_entry->owner_child_index = shifted_idx;
			if (shifted_entry->flow_parent.get() == &parent) {
				shifted_entry->flow_child_index = shifted_idx;
			}
		}
		if (refresh_parent) {
			RefreshOperator(parent);
		}
		return result;
	}

	void AttachChild(LogicalOperator &parent, idx_t child_index, unique_ptr<LogicalOperator> subtree,
	                 bool refresh_parent) {
		if (!subtree) {
			throw InternalException("Cannot attach logical plan child index %llu", child_index);
		}
		ValidateAttachChild(parent, child_index, *subtree);
		parent.children.insert(parent.children.begin() + NumericCast<int64_t>(child_index), std::move(subtree));
		for (idx_t shifted_idx = child_index; shifted_idx < parent.children.size(); shifted_idx++) {
			auto shifted_entry = GetEntry(*parent.children[shifted_idx]);
			D_ASSERT(shifted_entry);
			shifted_entry->owner_parent = parent;
			shifted_entry->owner_child_index = shifted_idx;
			if (shifted_entry->flow_parent.get() == &parent) {
				shifted_entry->flow_child_index = shifted_idx;
			}
		}
		if (refresh_parent) {
			RefreshOperator(parent);
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
		reference_set_t<LogicalOperator> left_path;
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(left);
		while (current) {
			left_path.insert(current->op);
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		current = right;
		while (current) {
			if (left_path.find(current->op) != left_path.end()) {
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

	optional_ptr<LogicalPlanDataFlowEntry> ParentFirstPathOperator(LogicalPlanDataFlowEntry &ancestor,
	                                                               LogicalPlanDataFlowEntry &descendant,
	                                                               const LogicalPlanPathSummary &properties) const {
		if (properties.properties == 0 || !ParentIsAncestor(ancestor, descendant)) {
			return nullptr;
		}
		optional_ptr<LogicalPlanDataFlowEntry> result;
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(descendant);
		while (current) {
			if ((current->node_value.properties & properties.properties) != 0) {
				result = current;
			}
			if (&current->op.get() == &ancestor.op.get()) {
				return result;
			}
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		return nullptr;
	}

	optional_ptr<LogicalPlanDataFlowEntry> ParentLastPathOperator(LogicalPlanDataFlowEntry &ancestor,
	                                                              LogicalPlanDataFlowEntry &descendant,
	                                                              const LogicalPlanPathSummary &properties) const {
		if (properties.properties == 0 || !ParentIsAncestor(ancestor, descendant)) {
			return nullptr;
		}
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(descendant);
		while (current) {
			if ((current->node_value.properties & properties.properties) != 0) {
				return current;
			}
			if (&current->op.get() == &ancestor.op.get()) {
				return nullptr;
			}
			current = current->flow_parent ? GetEntry(*current->flow_parent) : nullptr;
		}
		return nullptr;
	}

	LogicalPlanDataFlowStatus BoundaryBetween(LogicalPlanDataFlowEntry &left, LogicalPlanDataFlowEntry &right) const {
		reference_set_t<LogicalOperator> left_path;
		auto current = optional_ptr<LogicalPlanDataFlowEntry>(left);
		while (current) {
			left_path.insert(current->op);
			current = current->owner_parent ? GetEntry(*current->owner_parent) : nullptr;
		}
		auto right_cursor = optional_ptr<LogicalPlanDataFlowEntry>(right);
		while (right_cursor && left_path.find(right_cursor->op) == left_path.end()) {
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

	idx_t ChildContainingSource(LogicalPlanDataFlowEntry &source, LogicalPlanDataFlowEntry &parent) {
		for (idx_t child_idx = 0; child_idx < 2 && child_idx < parent.op.get().children.size(); child_idx++) {
			auto child_entry = GetEntry(*parent.op.get().children[child_idx]);
			if (child_entry && forest.IsAncestor(child_entry->forest_node, source.forest_node)) {
				return child_idx;
			}
		}
		return DConstants::INVALID_INDEX;
	}

	bool JoinOutputContains(LogicalPlanDataFlowEntry &join_entry, LogicalPlanDataFlowEntry &source,
	                        const ColumnBinding &binding) {
		auto &join = join_entry.op.get().Cast<LogicalJoin>();
		if (join.join_type == JoinType::MARK && binding.table_index == join.mark_index) {
			return true;
		}
		auto child_index = ChildContainingSource(source, join_entry);
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
	                           const ColumnBinding &binding) {
		if (&entry.op.get() == &source.op.get()) {
			return ContainsBinding(entry.op.get().GetColumnBindings(), binding);
		}
		auto &op = entry.op.get();
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_PROJECTION:
		case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
		case LogicalOperatorType::LOGICAL_PIVOT:
		case LogicalOperatorType::LOGICAL_EXPRESSION_GET:
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

LogicalPlanDataFlow::LogicalPlanDataFlow(LogicalOperator &root)
    : state(make_shared_ptr<LogicalPlanDataFlowState>(root)) {
}

LogicalPlanDataFlow::~LogicalPlanDataFlow() {
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::ResolveSource(const ColumnBinding &binding, idx_t depth,
                                                                     LogicalOperator &consumer) const {
	state->EnsureQueryable();
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
	auto source_entry = state->GetEntry(source->second.get());
	if (!source_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr};
	}
	if (&source_entry->op.get() == &consumer) {
		if (!state->OutputContainsBinding(*source_entry, *source_entry, binding)) {
			return {LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE, nullptr};
		}
		return {LogicalPlanDataFlowStatus::SUCCESS, source->second.get()};
	}
	LogicalPlanPathSummary boundary;
	boundary.Add(LogicalPlanPathProperty::BINDING_AVAILABILITY_BOUNDARY);
	optional_ptr<RootedDynamicForestNode> boundary_node;
	optional_ptr<RootedDynamicForestNode> path_child_node;
	if (!state->forest.FindLastNodeOnPath(consumer_entry->forest_node, source_entry->forest_node,
	                                      ToForestValue(boundary), boundary_node, path_child_node)) {
		if (!state->forest.Connected(source_entry->forest_node, consumer_entry->forest_node)) {
			return {state->BoundaryBetween(*source_entry, *consumer_entry), nullptr};
		}
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, nullptr};
	}
	if (!state->OutputContainsBinding(*source_entry, *source_entry, binding)) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE, nullptr};
	}
	while (boundary_node) {
		auto boundary_op = state->GetForestOperator(*boundary_node);
		auto boundary_entry = boundary_op ? state->GetEntry(*boundary_op) : nullptr;
		if (!boundary_entry) {
			return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
		}
		if (&boundary_entry->op.get() == &consumer) {
			break;
		}
		if (!state->OutputContainsBinding(*boundary_entry, *source_entry, binding)) {
			return {LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE, nullptr};
		}
		auto current = boundary_entry->flow_parent ? state->GetEntry(*boundary_entry->flow_parent) : nullptr;
		if (!current) {
			return {LogicalPlanDataFlowStatus::DISCONNECTED, nullptr};
		}
		if (&current->op.get() == &consumer) {
			break;
		}
		const bool has_path = state->forest.FindLastNodeOnPath(consumer_entry->forest_node, current->forest_node,
		                                                       ToForestValue(boundary), boundary_node);
		D_ASSERT(has_path);
	}
	auto path_child_op = path_child_node ? state->GetForestOperator(*path_child_node) : nullptr;
	auto path_child_entry = path_child_op ? state->GetEntry(*path_child_op) : nullptr;
	if (!path_child_entry || path_child_entry->flow_parent.get() != &consumer) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, source->second.get(), path_child_entry->flow_child_index};
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::ResolveInputSource(const ColumnBinding &binding, idx_t depth,
                                                                          LogicalOperator &consumer) const {
	auto result = ResolveSource(binding, depth, consumer);
	if (result.status == LogicalPlanDataFlowStatus::SUCCESS || depth != 0) {
		return result;
	}
	auto source = state->binding_sources.find(binding.table_index);
	if (source == state->binding_sources.end()) {
		return result;
	}
	auto source_entry = state->GetEntry(source->second.get());
	if (!source_entry || &source_entry->op.get() == &consumer) {
		return result;
	}
	auto current = source_entry;
	while (current && current->owner_parent && current->owner_parent.get() != &consumer) {
		current = state->GetEntry(*current->owner_parent);
	}
	if (!current || current->owner_parent.get() != &consumer ||
	    current->owner_child_index >= consumer.children.size()) {
		return result;
	}
	auto &input = *consumer.children[current->owner_child_index];
	if (!state->ContainsBinding(input.GetColumnBindings(), binding)) {
		return result;
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, source->second.get(), current->owner_child_index};
}

LogicalPlanDataFlowParentResult LogicalPlanDataFlow::GetOwnershipParent(LogicalOperator &op) const {
	state->EnsureQueryable();
	auto entry = state->GetEntry(op);
	if (!entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr, DConstants::INVALID_INDEX};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, entry->owner_parent, entry->owner_child_index};
}

LogicalPlanDataFlowParentResult LogicalPlanDataFlow::GetFlowParent(LogicalOperator &op) const {
	state->EnsureQueryable();
	auto entry = state->GetEntry(op);
	if (!entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr, DConstants::INVALID_INDEX};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, entry->flow_parent, entry->flow_child_index};
}

LogicalPlanDataFlowBooleanResult LogicalPlanDataFlow::SameFlowTree(LogicalOperator &left,
                                                                   LogicalOperator &right) const {
	state->EnsureQueryable();
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
	state->EnsureQueryable();
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
	state->EnsureQueryable();
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
	state->EnsureQueryable();
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

LogicalPlanDataFlowOperatorResult
LogicalPlanDataFlow::FindFirstPathOperator(LogicalOperator &ancestor, LogicalOperator &descendant,
                                           const LogicalPlanPathSummary &properties) const {
	auto ancestor_entry = state->GetEntry(ancestor);
	auto descendant_entry = state->GetEntry(descendant);
	if (!ancestor_entry || !descendant_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, nullptr};
	}
	if (!state->forest.Connected(ancestor_entry->forest_node, descendant_entry->forest_node)) {
		return {state->BoundaryBetween(*ancestor_entry, *descendant_entry), nullptr};
	}
	if (!state->forest.IsAncestor(ancestor_entry->forest_node, descendant_entry->forest_node)) {
		return {LogicalPlanDataFlowStatus::NOT_ANCESTOR, nullptr};
	}
	auto first = state->forest.FindFirstNodeOnPath(ancestor_entry->forest_node, descendant_entry->forest_node,
	                                               ToForestValue(properties));
	if (!first) {
		return {LogicalPlanDataFlowStatus::PATH_PROPERTY_NOT_FOUND, nullptr};
	}
	auto first_operator = state->GetForestOperator(*first);
	if (!first_operator) {
		return {LogicalPlanDataFlowStatus::UNSUPPORTED, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, first_operator};
}

LogicalPlanDataFlowOperatorResult LogicalPlanDataFlow::GetCTEProducer(TableIndex cte_index) const {
	state->EnsureQueryable();
	auto lineage = state->cte_lineage.find(cte_index);
	if (lineage == state->cte_lineage.end() || !lineage->second.producer) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_FOUND, nullptr};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, lineage->second.producer};
}

LogicalPlanDataFlowReadersResult LogicalPlanDataFlow::GetCTEReaders(TableIndex cte_index) const {
	state->EnsureQueryable();
	auto lineage = state->cte_lineage.find(cte_index);
	if (lineage == state->cte_lineage.end()) {
		return {LogicalPlanDataFlowStatus::BINDING_NOT_FOUND, {}};
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, lineage->second.readers};
}

vector<reference<LogicalOperator>> LogicalPlanDataFlow::GetMaterializedCTEs() const {
	state->EnsureQueryable();
	vector<reference<LogicalOperator>> result;
	vector<reference<LogicalOperator>> pending {*state->root};
	while (!pending.empty()) {
		auto op = pending.back();
		pending.pop_back();
		if (op.get().type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			result.push_back(op);
		}
		for (idx_t child_idx = op.get().children.size(); child_idx > 0; child_idx--) {
			pending.push_back(*op.get().children[child_idx - 1]);
		}
	}
	return result;
}

const vector<LogicalPlanBindingUse> &LogicalPlanDataFlow::GetBindingUses() const {
	state->EnsureQueryable();
	if (state->binding_uses_dirty) {
		state->RebuildBindingUses();
	}
	return state->binding_uses;
}

vector<LogicalPlanCorrelatedUse> LogicalPlanDataFlow::GetCorrelatedUses() const {
	vector<LogicalPlanCorrelatedUse> result;
	for (auto &binding_use : GetBindingUses()) {
		if (binding_use.depth == 0) {
			continue;
		}
		LogicalPlanCorrelatedUse correlated_use {LogicalPlanDataFlowStatus::CORRELATED_REFERENCE,
		                                         binding_use.binding,
		                                         binding_use.depth,
		                                         binding_use.consumer,
		                                         nullptr,
		                                         nullptr};
		auto current = state->GetEntry(binding_use.consumer);
		if (!current) {
			correlated_use.status = LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED;
			result.push_back(correlated_use);
			continue;
		}
		if (current->opaque) {
			correlated_use.status = LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY;
			result.push_back(correlated_use);
			continue;
		}
		if (current->node_value.Has(LogicalPlanPathProperty::CTE_BOUNDARY)) {
			correlated_use.status = LogicalPlanDataFlowStatus::CTE_BOUNDARY;
			result.push_back(correlated_use);
			continue;
		}
		while (current->owner_parent) {
			auto parent = state->GetEntry(*current->owner_parent);
			if (!parent) {
				correlated_use.status = LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED;
				break;
			}
			if (parent->opaque) {
				correlated_use.status = LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY;
				break;
			}
			if (parent->node_value.Has(LogicalPlanPathProperty::CTE_BOUNDARY)) {
				correlated_use.status = LogicalPlanDataFlowStatus::CTE_BOUNDARY;
				break;
			}
			if (parent->op.get().type == LogicalOperatorType::LOGICAL_DEPENDENT_JOIN &&
			    current->owner_child_index == 1) {
				auto &dependent_join = parent->op.get().Cast<LogicalDependentJoin>();
				bool binding_matches = false;
				for (auto &column : dependent_join.correlated_columns) {
					if (column.binding == binding_use.binding) {
						binding_matches = true;
						break;
					}
				}
				if (binding_matches && dependent_join.children.size() != 2) {
					correlated_use.status = LogicalPlanDataFlowStatus::UNSUPPORTED;
					break;
				}
				if (binding_matches) {
					auto source = ResolveSource(binding_use.binding, 0, *dependent_join.children[0]);
					if (source.status == LogicalPlanDataFlowStatus::SUCCESS) {
						correlated_use.status = source.status;
						correlated_use.source = source.op;
						correlated_use.owning_join = dependent_join;
						break;
					}
					if (source.status == LogicalPlanDataFlowStatus::OPAQUE_BOUNDARY ||
					    source.status == LogicalPlanDataFlowStatus::CTE_BOUNDARY) {
						correlated_use.status = source.status;
					} else if (source.status != LogicalPlanDataFlowStatus::BINDING_NOT_FOUND &&
					           source.status != LogicalPlanDataFlowStatus::BINDING_NOT_AVAILABLE &&
					           source.status != LogicalPlanDataFlowStatus::DISCONNECTED &&
					           source.status != LogicalPlanDataFlowStatus::NOT_ANCESTOR) {
						correlated_use.status = source.status;
						break;
					}
				}
			}
			current = parent;
		}
		result.push_back(correlated_use);
	}
	return result;
}

idx_t LogicalPlanDataFlow::OperatorCount() const {
	state->EnsureQueryable();
	return state->entries.size();
}

bool LogicalPlanDataFlow::Verify() const {
	state->EnsureQueryable();
	if (!state->valid || state->entries.empty() || !state->root || !state->GetEntry(*state->root)) {
		return false;
	}
	if (state->binding_uses_dirty) {
		state->RebuildBindingUses();
	}
	idx_t ownership_count = 0;
	reference_set_t<LogicalOperator> visited;
	vector<reference<LogicalOperator>> pending;
	for (auto &entry_pair : state->entries) {
		if (!entry_pair.second->owner_parent) {
			pending.push_back(entry_pair.second->op);
		}
	}
	while (!pending.empty()) {
		auto op = pending.back();
		pending.pop_back();
		if (!visited.insert(op).second) {
			return false;
		}
		auto entry = state->GetEntry(op);
		if (!entry) {
			return false;
		}
		if (entry->node_value != state->GetNodeValue(op)) {
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
				if (child_entry->edge_to_parent != state->GetEdgeValue(op, child_idx)) {
					return false;
				}
			} else if (child_entry->flow_parent.get() == &op.get()) {
				return false;
			} else if (child_entry->edge_to_parent != LogicalPlanPathSummary()) {
				return false;
			}
			pending.push_back(*op.get().children[child_idx]);
		}
	}
	if (ownership_count != state->entries.size()) {
		return false;
	}
	idx_t produced_index_count = 0;
	idx_t cte_producer_count = 0;
	idx_t cte_reader_count = 0;
	idx_t binding_use_count = 0;
	idx_t correlated_binding_count = 0;
	for (auto &entry_pair : state->entries) {
		auto &entry = *entry_pair.second;
		LogicalPlanDataFlowMetadata metadata;
		if (!state->GetMetadata(entry.op, metadata) ||
		    metadata.produced_table_indexes != entry.produced_table_indexes ||
		    metadata.cte_producer_index != entry.cte_producer_index || metadata.cte_producer != entry.cte_producer ||
		    metadata.cte_reader_index != entry.cte_reader_index ||
		    metadata.binding_uses.size() != entry.binding_uses.size() ||
		    metadata.correlated_bindings != entry.correlated_bindings) {
			return false;
		}
		for (idx_t use_idx = 0; use_idx < metadata.binding_uses.size(); use_idx++) {
			auto &expected = metadata.binding_uses[use_idx];
			auto &actual = entry.binding_uses[use_idx];
			if (expected.binding != actual.binding || expected.depth != actual.depth ||
			    &expected.consumer.get() != &actual.consumer.get()) {
				return false;
			}
		}
		for (auto table_index : entry.produced_table_indexes) {
			auto source = state->binding_sources.find(table_index);
			if (source == state->binding_sources.end() || &source->second.get() != &entry.op.get()) {
				return false;
			}
			produced_index_count++;
		}
		if (entry.cte_producer_index.IsValid()) {
			auto lineage = state->cte_lineage.find(entry.cte_producer_index);
			if (lineage == state->cte_lineage.end() || lineage->second.producer_owner.get() != &entry.op.get() ||
			    lineage->second.producer != entry.cte_producer) {
				return false;
			}
			cte_producer_count++;
		}
		if (entry.cte_reader_index.IsValid()) {
			auto lineage = state->cte_lineage.find(entry.cte_reader_index);
			if (lineage == state->cte_lineage.end() ||
			    std::none_of(
			        lineage->second.readers.begin(), lineage->second.readers.end(),
			        [&](const reference<LogicalOperator> &reader) { return &reader.get() == &entry.op.get(); })) {
				return false;
			}
			cte_reader_count++;
		}
		for (auto &binding : entry.correlated_bindings) {
			auto correlated = state->correlated_operators.find(binding);
			if (correlated == state->correlated_operators.end() ||
			    correlated->second.find(entry.op) == correlated->second.end()) {
				return false;
			}
			correlated_binding_count++;
		}
		binding_use_count += entry.binding_uses.size();
	}
	idx_t actual_cte_producers = 0;
	idx_t actual_cte_readers = 0;
	idx_t actual_correlated_bindings = 0;
	for (auto &lineage : state->cte_lineage) {
		actual_cte_producers += lineage.second.producer ? 1 : 0;
		actual_cte_readers += lineage.second.readers.size();
	}
	for (auto &correlated : state->correlated_operators) {
		actual_correlated_bindings += correlated.second.size();
	}
	if (produced_index_count != state->binding_sources.size() || cte_producer_count != actual_cte_producers ||
	    cte_reader_count != actual_cte_readers || binding_use_count != state->binding_uses.size() ||
	    correlated_binding_count != actual_correlated_bindings) {
		return false;
	}
	for (auto &entry_pair : state->entries) {
		auto &entry = entry_pair.second;
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
		for (auto &left_pair : state->entries) {
			auto &left = left_pair.second;
			for (auto &right_pair : state->entries) {
				auto &right = right_pair.second;
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
				const LogicalPlanPathProperty guided_properties[] {
				    LogicalPlanPathProperty::OPAQUE_BOUNDARY,
				    LogicalPlanPathProperty::CTE_BOUNDARY,
				    LogicalPlanPathProperty::PROJECTION_BOUNDARY,
				    LogicalPlanPathProperty::AGGREGATE_BOUNDARY,
				    LogicalPlanPathProperty::SET_OPERATION_BOUNDARY,
				    LogicalPlanPathProperty::WINDOW_BOUNDARY,
				    LogicalPlanPathProperty::UNNEST_BOUNDARY,
				    LogicalPlanPathProperty::LIMIT_BOUNDARY,
				    LogicalPlanPathProperty::SIDE_EFFECT_BOUNDARY,
				    LogicalPlanPathProperty::FILTER_PUSHDOWN_BOUNDARY,
				    LogicalPlanPathProperty::NULLABILITY_BOUNDARY,
				    LogicalPlanPathProperty::BINDING_AVAILABILITY_BOUNDARY};
				for (auto property : guided_properties) {
					LogicalPlanPathSummary properties;
					properties.Add(property);
					auto expected_first = state->ParentFirstPathOperator(*left, *right, properties);
					auto forest_first = state->forest.FindFirstNodeOnPath(left->forest_node, right->forest_node,
					                                                      ToForestValue(properties));
					auto actual_first = forest_first ? state->GetForestOperator(*forest_first) : nullptr;
					if ((expected_first ? &expected_first->op.get() : nullptr) != actual_first.get()) {
						return false;
					}
					auto expected_last = state->ParentLastPathOperator(*left, *right, properties);
					auto forest_last = state->forest.FindLastNodeOnPath(left->forest_node, right->forest_node,
					                                                    ToForestValue(properties));
					auto actual_last = forest_last ? state->GetForestOperator(*forest_last) : nullptr;
					if ((expected_last ? &expected_last->op.get() : nullptr) != actual_last.get()) {
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

LogicalPlanDataFlowDetachedSubtree::LogicalPlanDataFlowDetachedSubtree(
    const shared_ptr<LogicalPlanDataFlowState> &state_p, unique_ptr<LogicalOperator> subtree_p)
    : state(state_p), subtree(std::move(subtree_p)) {
}

LogicalPlanDataFlowDetachedSubtree::~LogicalPlanDataFlowDetachedSubtree() {
	Reset();
}

LogicalPlanDataFlowDetachedSubtree::LogicalPlanDataFlowDetachedSubtree(
    LogicalPlanDataFlowDetachedSubtree &&other) noexcept
    : state(std::move(other.state)), subtree(std::move(other.subtree)) {
}

LogicalPlanDataFlowDetachedSubtree &
LogicalPlanDataFlowDetachedSubtree::operator=(LogicalPlanDataFlowDetachedSubtree &&other) noexcept {
	if (this != &other) {
		Reset();
		state = std::move(other.state);
		subtree = std::move(other.subtree);
	}
	return *this;
}

LogicalPlanDataFlowDetachedSubtree::operator bool() const {
	return subtree != nullptr;
}

LogicalOperator &LogicalPlanDataFlowDetachedSubtree::Get() {
	if (!subtree) {
		throw InternalException("Cannot access an empty detached logical plan subtree");
	}
	return *subtree;
}

void LogicalPlanDataFlowDetachedSubtree::Reset() {
	if (!subtree) {
		return;
	}
	auto current_state = state.lock();
	if (current_state) {
		current_state->UnregisterSubtree(*subtree);
	}
	subtree.reset();
	state.reset();
}

LogicalPlanDataFlowMutationScope::LogicalPlanDataFlowMutationScope(LogicalPlanDataFlowMutator &mutator_p)
    : mutator(mutator_p) {
	mutator_p.data_flow.state->active_mutation_scopes++;
}

LogicalPlanDataFlowMutationScope::~LogicalPlanDataFlowMutationScope() {
	Reset();
}

LogicalPlanDataFlowMutationScope::LogicalPlanDataFlowMutationScope(LogicalPlanDataFlowMutationScope &&other) noexcept
    : mutator(other.mutator) {
	other.mutator = nullptr;
}

LogicalPlanDataFlowMutationScope &
LogicalPlanDataFlowMutationScope::operator=(LogicalPlanDataFlowMutationScope &&other) noexcept {
	if (this != &other) {
		Reset();
		mutator = other.mutator;
		other.mutator = nullptr;
	}
	return *this;
}

void LogicalPlanDataFlowMutationScope::Reset() {
	if (!mutator) {
		return;
	}
	mutator->EndMutation();
	mutator = nullptr;
}

LogicalPlanDataFlowMutator::LogicalPlanDataFlowMutator(LogicalPlanDataFlow &data_flow_p) : data_flow(data_flow_p) {
	D_ASSERT(data_flow.Verify());
}

LogicalPlanDataFlowMutationScope LogicalPlanDataFlowMutator::BeginMutation() {
	return LogicalPlanDataFlowMutationScope(*this);
}

void LogicalPlanDataFlowMutator::EndMutation() {
	D_ASSERT(data_flow.state->active_mutation_scopes > 0);
	data_flow.state->active_mutation_scopes--;
	if (data_flow.state->active_mutation_scopes == 0 && !Exception::UncaughtException()) {
		D_ASSERT(data_flow.Verify());
	}
}

void LogicalPlanDataFlowMutator::VerifyAfterMutation() const {
	if (data_flow.state->active_mutation_scopes == 0) {
		D_ASSERT(data_flow.Verify());
	}
}

LogicalPlanDataFlowDetachedSubtree LogicalPlanDataFlowMutator::RegisterSubtree(unique_ptr<LogicalOperator> subtree) {
	if (!subtree) {
		throw InternalException("Cannot register a null logical plan subtree");
	}
	data_flow.state->RegisterSubtree(*subtree);
	VerifyAfterMutation();
	return LogicalPlanDataFlowDetachedSubtree(data_flow.state, std::move(subtree));
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::UnregisterSubtree(LogicalPlanDataFlowDetachedSubtree subtree) {
	auto subtree_state = subtree.state.lock();
	if (!subtree.subtree || subtree_state.get() != data_flow.state.get()) {
		throw InternalException("Cannot unregister a subtree belonging to another logical plan data flow");
	}
	data_flow.state->UnregisterSubtree(*subtree.subtree);
	auto result = std::move(subtree.subtree);
	subtree.state.reset();
	VerifyAfterMutation();
	return result;
}

struct LogicalPlanDataFlowSlot {
	optional_ptr<LogicalOperator> parent;
	idx_t child_index = DConstants::INVALID_INDEX;
	bool is_root = false;
};

static LogicalPlanDataFlowSlot GetMutationSlot(LogicalPlanDataFlowState &state, unique_ptr<LogicalOperator> &slot) {
	if (!slot) {
		throw InternalException("Cannot mutate a null logical plan slot");
	}
	auto entry = state.GetEntry(*slot);
	if (!entry) {
		throw InternalException("Cannot mutate a logical plan slot that is not indexed");
	}
	if (!entry->owner_parent) {
		if (state.root.get() != slot.get()) {
			throw InternalException("Cannot mutate a detached logical plan slot directly");
		}
		return {nullptr, DConstants::INVALID_INDEX, true};
	}
	auto &parent = *entry->owner_parent;
	if (entry->owner_child_index >= parent.children.size() || &parent.children[entry->owner_child_index] != &slot) {
		throw InternalException("Logical plan ownership changed outside its indexed mutator");
	}
	return {parent, entry->owner_child_index, false};
}

LogicalPlanDataFlowDetachedSubtree LogicalPlanDataFlowMutator::DetachChild(LogicalOperator &parent, idx_t child_index) {
	if (child_index >= parent.children.size()) {
		throw InternalException("Cannot detach logical plan child index %llu", child_index);
	}
	data_flow.state->ValidateStructuralMutation(parent, parent.children.size() - 1);
	auto subtree = data_flow.state->DetachChild(parent, child_index, false);
	data_flow.state->RefreshOperator(parent);
	return LogicalPlanDataFlowDetachedSubtree(data_flow.state, std::move(subtree));
}

void LogicalPlanDataFlowMutator::AttachChild(LogicalOperator &parent, idx_t child_index,
                                             LogicalPlanDataFlowDetachedSubtree subtree) {
	auto subtree_state = subtree.state.lock();
	if (!subtree.subtree || subtree_state.get() != data_flow.state.get()) {
		throw InternalException("Cannot attach a subtree belonging to another logical plan data flow");
	}
	data_flow.state->ValidateStructuralMutation(parent, parent.children.size() + 1);
	data_flow.state->ValidateAttachChild(parent, child_index, *subtree.subtree);
	auto owned_subtree = std::move(subtree.subtree);
	data_flow.state->AttachChild(parent, child_index, std::move(owned_subtree), false);
	data_flow.state->RefreshOperator(parent);
	subtree.state.reset();
	VerifyAfterMutation();
}

void LogicalPlanDataFlowMutator::AttachChild(LogicalOperator &parent, idx_t child_index,
                                             unique_ptr<LogicalOperator> subtree) {
	auto detached = RegisterSubtree(std::move(subtree));
	AttachChild(parent, child_index, std::move(detached));
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::EraseChild(LogicalOperator &parent, idx_t child_index) {
	if (child_index >= parent.children.size()) {
		throw InternalException("Cannot erase logical plan child index %llu", child_index);
	}
	data_flow.state->ValidateStructuralMutation(parent, parent.children.size() - 1);
	auto subtree = data_flow.state->DetachChild(parent, child_index, false);
	data_flow.state->RefreshOperator(parent);
	data_flow.state->UnregisterSubtree(*subtree);
	VerifyAfterMutation();
	return subtree;
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::ReplaceSubtree(unique_ptr<LogicalOperator> &slot,
                                                                       unique_ptr<LogicalOperator> replacement) {
	if (!replacement) {
		throw InternalException("Cannot replace a logical plan subtree with null");
	}
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	reference_set_t<const LogicalOperator> replaced_operators;
	vector<reference<LogicalOperator>> pending {*slot};
	while (!pending.empty()) {
		auto op = pending.back();
		pending.pop_back();
		replaced_operators.insert(op);
		for (auto &child : op.get().children) {
			pending.push_back(*child);
		}
	}
	data_flow.state->ValidateNewSubtree(*replacement, replaced_operators);
	unique_ptr<LogicalOperator> old_subtree;
	if (location.is_root) {
		old_subtree = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_subtree = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	data_flow.state->UnregisterSubtree(*old_subtree);
	data_flow.state->RegisterSubtree(*replacement);
	if (location.is_root) {
		slot = std::move(replacement);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(replacement), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
	return old_subtree;
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::ReplaceOperator(unique_ptr<LogicalOperator> &slot,
                                                                        unique_ptr<LogicalOperator> replacement) {
	if (!replacement || !replacement->children.empty()) {
		throw InternalException("Indexed operator replacement requires a childless replacement");
	}
	auto location = GetMutationSlot(*data_flow.state, slot);
	data_flow.state->ValidateChildren(*slot);
	data_flow.state->ValidateFutureChildCount(*slot, slot->children.size());
	data_flow.state->ValidateFutureChildCount(*replacement, slot->children.size());
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	reference_set_t<const LogicalOperator> replaced_operators;
	replaced_operators.insert(reference<const LogicalOperator>(*slot));
	data_flow.state->ValidateNewSubtree(*replacement, replaced_operators);

	unique_ptr<LogicalOperator> old_operator;
	if (location.is_root) {
		old_operator = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_operator = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	vector<unique_ptr<LogicalOperator>> children;
	children.reserve(old_operator->children.size());
	while (!old_operator->children.empty()) {
		children.push_back(data_flow.state->DetachChild(*old_operator, 0, false));
	}
	data_flow.state->UnregisterSubtree(*old_operator);
	data_flow.state->RegisterSubtree(*replacement);
	for (auto &child : children) {
		data_flow.state->AttachChild(*replacement, replacement->children.size(), std::move(child), false);
	}
	data_flow.state->RefreshOperator(*replacement);
	if (location.is_root) {
		slot = std::move(replacement);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(replacement), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
	return old_operator;
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::PromoteChild(unique_ptr<LogicalOperator> &slot,
                                                                     idx_t child_index) {
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (child_index >= slot->children.size()) {
		throw InternalException("Cannot promote logical plan child index %llu", child_index);
	}
	data_flow.state->ValidateStructuralMutation(*slot, 0);
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	reference_set_t<const LogicalOperator> removed_operators;
	removed_operators.insert(reference<const LogicalOperator>(*slot));
	for (idx_t current_child = 0; current_child < slot->children.size(); current_child++) {
		if (current_child == child_index) {
			continue;
		}
		vector<reference<LogicalOperator>> pending {*slot->children[current_child]};
		while (!pending.empty()) {
			auto current = pending.back();
			pending.pop_back();
			removed_operators.insert(reference<const LogicalOperator>(current));
			for (auto &child : current.get().children) {
				pending.push_back(*child);
			}
		}
	}
	LogicalPlanDataFlowMetadata promoted_metadata;
	data_flow.state->ValidateRefreshOperator(*slot->children[child_index], promoted_metadata, removed_operators);

	unique_ptr<LogicalOperator> old_operator;
	if (location.is_root) {
		old_operator = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_operator = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	auto promoted = data_flow.state->DetachChild(*old_operator, child_index, false);
	while (!old_operator->children.empty()) {
		auto discarded = data_flow.state->DetachChild(*old_operator, 0, false);
		data_flow.state->UnregisterSubtree(*discarded);
	}
	data_flow.state->UnregisterSubtree(*old_operator);
	data_flow.state->RefreshOperator(*promoted);
	if (location.is_root) {
		slot = std::move(promoted);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(promoted), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
	return old_operator;
}

void LogicalPlanDataFlowMutator::InsertUnary(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> wrapper) {
	if (!wrapper || !wrapper->children.empty()) {
		throw InternalException("Indexed unary insertion requires a childless wrapper");
	}
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (!data_flow.state->SupportsUnaryInsertion(*wrapper)) {
		throw InternalException("Indexed unary insertion requires an operator with known child semantics");
	}
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	data_flow.state->RegisterSubtree(*wrapper);
	data_flow.state->ValidateStructuralMutation(*wrapper, 1);
	unique_ptr<LogicalOperator> old_subtree;
	if (location.is_root) {
		old_subtree = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_subtree = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	data_flow.state->AttachChild(*wrapper, 0, std::move(old_subtree), false);
	data_flow.state->RefreshOperator(*wrapper);
	if (location.is_root) {
		slot = std::move(wrapper);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(wrapper), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
}

void LogicalPlanDataFlowMutator::InsertParent(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> parent,
                                              idx_t child_index) {
	if (!parent || parent->children.size() != 1 || child_index > 1) {
		throw InternalException("Indexed parent insertion requires one fresh sibling and a valid child slot");
	}
	if (!data_flow.state->SupportsParentInsertion(*parent)) {
		throw InternalException("Indexed parent insertion requires an operator with known binary child semantics");
	}
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	data_flow.state->ValidateFutureChildCount(*parent, 2);
	reference_set_t<const LogicalOperator> replaced_operators;
	data_flow.state->ValidateNewSubtree(*parent->children[0], replaced_operators);

	unique_ptr<LogicalOperator> old_subtree;
	if (location.is_root) {
		old_subtree = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_subtree = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	data_flow.state->UnregisterSubtree(*old_subtree);
	parent->children.insert(parent->children.begin() + NumericCast<int64_t>(child_index), std::move(old_subtree));
	data_flow.state->RegisterSubtree(*parent);
	if (location.is_root) {
		slot = std::move(parent);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(parent), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
}

void LogicalPlanDataFlowMutator::RotateParentWithChild(unique_ptr<LogicalOperator> &slot, idx_t child_index,
                                                       idx_t grandchild_index) {
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (child_index >= slot->children.size()) {
		throw InternalException("Cannot rotate logical plan child index %llu", child_index);
	}
	auto &child = *slot->children[child_index];
	if (grandchild_index >= child.children.size()) {
		throw InternalException("Cannot rotate logical plan grandchild index %llu", grandchild_index);
	}
	data_flow.state->ValidateStructuralMutation(*slot, slot->children.size());
	data_flow.state->ValidateStructuralMutation(child, child.children.size());
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}

	unique_ptr<LogicalOperator> old_parent;
	if (location.is_root) {
		old_parent = std::move(slot);
		data_flow.state->root = nullptr;
	} else {
		old_parent = data_flow.state->DetachChild(*location.parent, location.child_index, false);
	}
	auto new_parent = data_flow.state->DetachChild(*old_parent, child_index, false);
	auto replacement_child = data_flow.state->DetachChild(*new_parent, grandchild_index, false);
	data_flow.state->AttachChild(*old_parent, child_index, std::move(replacement_child), false);
	data_flow.state->RefreshOperator(*old_parent);
	data_flow.state->AttachChild(*new_parent, grandchild_index, std::move(old_parent), false);
	data_flow.state->RefreshOperator(*new_parent);
	if (location.is_root) {
		slot = std::move(new_parent);
		data_flow.state->root = slot.get();
	} else {
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(new_parent), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	VerifyAfterMutation();
}

void LogicalPlanDataFlowMutator::InsertUnary(unique_ptr<LogicalOperator> &slot, unique_ptr<LogicalOperator> wrapper,
                                             LogicalOperator &changed_parent) {
	if (!wrapper || !wrapper->children.empty()) {
		throw InternalException("Indexed unary insertion requires a childless wrapper");
	}
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (location.is_root || location.parent.get() != &changed_parent) {
		throw InternalException("Changed operator must own the indexed unary insertion slot");
	}
	if (!data_flow.state->SupportsUnaryInsertion(*wrapper)) {
		throw InternalException("Indexed unary insertion requires an operator with known child semantics");
	}
	data_flow.state->ValidateChildren(changed_parent);
	data_flow.state->ValidateFutureChildCount(changed_parent, changed_parent.children.size());
	LogicalPlanDataFlowMetadata parent_metadata;
	data_flow.state->ValidateRefreshOperator(changed_parent, parent_metadata);
	data_flow.state->RegisterSubtree(*wrapper);
	data_flow.state->ValidateStructuralMutation(*wrapper, 1);
	auto old_subtree = data_flow.state->DetachChild(changed_parent, location.child_index, false);
	data_flow.state->AttachChild(*wrapper, 0, std::move(old_subtree), false);
	data_flow.state->RefreshOperator(*wrapper);
	data_flow.state->AttachChild(changed_parent, location.child_index, std::move(wrapper), false);
	data_flow.state->RefreshOperator(changed_parent);
	D_ASSERT(data_flow.Verify());
}

unique_ptr<LogicalOperator> LogicalPlanDataFlowMutator::RemoveUnary(unique_ptr<LogicalOperator> &slot) {
	auto location = GetMutationSlot(*data_flow.state, slot);
	if (slot->children.size() != 1) {
		throw InternalException("Indexed unary removal requires exactly one child");
	}
	auto &wrapper = *slot;
	data_flow.state->ValidateChildren(wrapper);
	data_flow.state->ValidateFutureChildCount(wrapper, 0);
	if (!location.is_root) {
		data_flow.state->ValidateSlotReplacement(*location.parent);
	}
	auto child = data_flow.state->DetachChild(wrapper, 0, false);
	unique_ptr<LogicalOperator> removed;
	if (location.is_root) {
		removed = std::move(slot);
		data_flow.state->root = nullptr;
		slot = std::move(child);
		data_flow.state->root = slot.get();
	} else {
		removed = data_flow.state->DetachChild(*location.parent, location.child_index, false);
		data_flow.state->AttachChild(*location.parent, location.child_index, std::move(child), false);
		data_flow.state->RefreshOperator(*location.parent);
	}
	data_flow.state->UnregisterSubtree(*removed);
	VerifyAfterMutation();
	return removed;
}

void LogicalPlanDataFlowMutator::SwapChildren(LogicalOperator &parent, idx_t left_index, idx_t right_index) {
	data_flow.state->ValidateChildren(parent);
	if (left_index >= parent.children.size() || right_index >= parent.children.size()) {
		throw InternalException("Cannot swap logical plan child indexes %llu and %llu", left_index, right_index);
	}
	if (left_index == right_index) {
		return;
	}
	data_flow.state->ValidateStructuralMutation(parent, parent.children.size());
	std::swap(parent.children[left_index], parent.children[right_index]);
	auto left_entry = data_flow.state->GetEntry(*parent.children[left_index]);
	auto right_entry = data_flow.state->GetEntry(*parent.children[right_index]);
	left_entry->owner_child_index = left_index;
	right_entry->owner_child_index = right_index;
	data_flow.state->RefreshOperator(parent);
	VerifyAfterMutation();
}

void LogicalPlanDataFlowMutator::RefreshOperator(LogicalOperator &op) {
	data_flow.state->RefreshOperator(op);
	VerifyAfterMutation();
}

LogicalPlanDataFlowBooleanResult
LogicalPlanDataFlowMutator::HasCorrelatedBinding(LogicalOperator &subtree, const column_binding_set_t &bindings) const {
	auto subtree_entry = data_flow.state->GetEntry(subtree);
	if (!subtree_entry) {
		return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, false};
	}
	for (auto &binding : bindings) {
		auto correlated = data_flow.state->correlated_operators.find(binding);
		if (correlated == data_flow.state->correlated_operators.end()) {
			continue;
		}
		for (auto &correlated_op : correlated->second) {
			auto correlated_entry = data_flow.state->GetEntry(correlated_op);
			if (!correlated_entry) {
				return {LogicalPlanDataFlowStatus::OPERATOR_NOT_INDEXED, false};
			}
			if (&subtree == &correlated_op.get()) {
				return {LogicalPlanDataFlowStatus::SUCCESS, true};
			}
			if (data_flow.state->forest.Connected(subtree_entry->forest_node, correlated_entry->forest_node)) {
				if (data_flow.state->forest.IsAncestor(subtree_entry->forest_node, correlated_entry->forest_node)) {
					return {LogicalPlanDataFlowStatus::SUCCESS, true};
				}
				continue;
			}
			for (auto current = correlated_entry; current && current->owner_parent;
			     current = data_flow.state->GetEntry(*current->owner_parent)) {
				if (current->owner_parent.get() == &subtree) {
					return {LogicalPlanDataFlowStatus::SUCCESS, true};
				}
			}
		}
	}
	return {LogicalPlanDataFlowStatus::SUCCESS, false};
}

} // namespace duckdb
