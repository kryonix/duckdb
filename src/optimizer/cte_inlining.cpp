#include "duckdb/optimizer/cte_inlining.hpp"

#include "duckdb/common/map.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/cte_filter_pusher.hpp"
#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/optimizer/statistics_propagator.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/operator/logical_prepare.hpp"

namespace duckdb {

static constexpr double CTE_COST_IMPROVEMENT_THRESHOLD = 0.1;
static constexpr idx_t CTE_CANDIDATE_BUDGET = 32;

class TableIndexCheckpoint {
public:
	explicit TableIndexCheckpoint(Binder &binder_p)
	    : binder(binder_p), table_index_count(binder_p.GetTableIndexCount()) {
	}

	~TableIndexCheckpoint() {
		binder.get().RestoreTableIndexCount(table_index_count);
	}

private:
	reference<Binder> binder;
	idx_t table_index_count;
};

struct CTEConsumerInfo {
	reference<unique_ptr<LogicalOperator>> owner;
	bool filtered;
	bool below_delim_join;
};

static bool IsCTEReference(const LogicalOperator &op, TableIndex cte_index) {
	return op.type == LogicalOperatorType::LOGICAL_CTE_REF && op.Cast<LogicalCTERef>().cte_index == cte_index;
}

static void FindCTEConsumers(unique_ptr<LogicalOperator> &op, TableIndex cte_index, vector<CTEConsumerInfo> &consumers,
                             bool below_delim_join = false) {
	const bool child_below_delim_join = below_delim_join || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN;
	if (op->type == LogicalOperatorType::LOGICAL_FILTER && op->children.size() == 1 &&
	    IsCTEReference(*op->children[0], cte_index)) {
		consumers.push_back({op, true, child_below_delim_join});
		return;
	}
	if (IsCTEReference(*op, cte_index)) {
		consumers.push_back({op, false, below_delim_join});
		return;
	}
	for (auto &child : op->children) {
		FindCTEConsumers(child, cte_index, consumers, child_below_delim_join);
	}
}

static bool CostEstimateSupported(const LogicalOperator &op, TableIndex target_cte_index) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:
		return false;
	case LogicalOperatorType::LOGICAL_CTE_REF:
		if (op.Cast<LogicalCTERef>().cte_index != target_cte_index) {
			return false;
		}
		break;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (!CostEstimateSupported(*child, target_cte_index)) {
			return false;
		}
	}
	return true;
}

static void ResetEstimatedCardinalities(LogicalOperator &op) {
	op.has_estimated_cardinality = false;
	for (auto &child : op.children) {
		ResetEstimatedCardinalities(*child);
	}
}

static unique_ptr<LogicalOperator> OptimizeCandidate(Optimizer &optimizer, unique_ptr<LogicalOperator> plan) {
	ResetEstimatedCardinalities(*plan);
	CTEFilterPusher cte_filter_pusher(optimizer);
	plan = cte_filter_pusher.Optimize(std::move(plan));
	FilterPushdown pushdown(optimizer, true, FilterPushdown::ProjectionMode::PRESERVE_COMPUTED_EXPRESSIONS);
	plan = pushdown.Rewrite(std::move(plan));
	JoinOrderOptimizer join_order(optimizer.context);
	plan = join_order.Optimize(std::move(plan));
	StatisticsPropagator propagator(optimizer, *plan);
	propagator.PropagateStatistics(plan);
	return plan;
}

CTEInlining::CTEInlining(Optimizer &optimizer_p) : optimizer(optimizer_p) {
}

unique_ptr<LogicalOperator> CTEInlining::OptimizeStructural(unique_ptr<LogicalOperator> op) {
	has_changes = false;
	TryInlining(op, false);
	return op;
}

unique_ptr<LogicalOperator> CTEInlining::OptimizeCostAware(unique_ptr<LogicalOperator> op) {
	has_changes = false;
	TryInlining(op, true);
	return op;
}

bool CTEInlining::HasChanges() const {
	return has_changes;
}

static idx_t CountBaseTableReferences(const LogicalOperator &op) {
	idx_t number_of_references = 0;
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		number_of_references++;
	}
	for (auto &child : op.children) {
		number_of_references += CountBaseTableReferences(*child);
	}

	return number_of_references;
}

static idx_t CountCTEReferences(const LogicalOperator &op, TableIndex cte_index) {
	idx_t number_of_references = 0;
	if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		auto &cte = op.Cast<LogicalCTERef>();
		if (cte.cte_index == cte_index) {
			number_of_references = 1;
		}
	}
	for (auto &child : op.children) {
		number_of_references += CountCTEReferences(*child, cte_index);
	}

	return number_of_references;
}

static bool ContainsLimit(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_LIMIT || op.type == LogicalOperatorType::LOGICAL_TOP_N) {
		return true;
	}
	if (op.children.size() != 1) {
		return false;
	}
	for (auto &child : op.children) {
		if (ContainsLimit(*child)) {
			return true;
		}
	}
	return false;
}

bool CTEInlining::EndsInAggregateOrDistinct(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_WINDOW:
		return true;
	default:
		break;
	}
	if (op.children.size() != 1) {
		return false;
	}
	for (auto &child : op.children) {
		if (EndsInAggregateOrDistinct(*child)) {
			return true;
		}
	}
	return false;
}

static bool EndsInDummyScan(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DUMMY_SCAN || op.type == LogicalOperatorType::LOGICAL_EMPTY_RESULT ||
	    op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		return true;
	}
	if (op.children.size() != 1) {
		return false;
	}
	for (auto &child : op.children) {
		if (EndsInDummyScan(*child)) {
			return true;
		}
	}
	return false;
}

static bool ContainsDelimGet(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) {
		return true;
	}
	for (auto &child : op.children) {
		if (ContainsDelimGet(*child)) {
			return true;
		}
	}
	return false;
}

static bool ContainsCostSensitiveBlockingOperator(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
	case LogicalOperatorType::LOGICAL_DISTINCT:
	case LogicalOperatorType::LOGICAL_WINDOW:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
		return true;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (ContainsCostSensitiveBlockingOperator(*child)) {
			return true;
		}
	}
	return false;
}

static bool HasCTEReferenceBelowDelimJoin(const LogicalOperator &op, TableIndex cte_index,
                                          bool below_delim_join = false) {
	if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		auto &cteref = op.Cast<LogicalCTERef>();
		if (cteref.cte_index == cte_index) {
			return below_delim_join;
		}
	}
	auto child_below_delim_join = below_delim_join || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN;
	for (auto &child : op.children) {
		if (HasCTEReferenceBelowDelimJoin(*child, cte_index, child_below_delim_join)) {
			return true;
		}
	}
	return false;
}

void CTEInlining::TryInlining(unique_ptr<LogicalOperator> &op, bool cost_aware) {
	if (op->type == LogicalOperatorType::LOGICAL_PREPARE) {
		// we are in a prepare statement, if we have to copy an operator during inlining,
		// we have to be careful to use the correct parameter data
		auto &prepare = op->Cast<LogicalPrepare>();
		parameter_data = prepare.prepared->value_map;
	}

	// traverse children first, so we can inline the deepest CTEs first
	for (auto &child : op->children) {
		TryInlining(child, cost_aware);
	}

	if (op->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		auto &cte = op->Cast<LogicalMaterializedCTE>();
		auto ref_count = CountCTEReferences(*op, cte.table_index);
		if (ref_count == 0) {
			if (cte.children[0]->HasSideEffects()) {
				// Side-effecting CTEs must always execute even when unreferenced
				return;
			}
			// this CTE is not referenced, we can remove it
			op = std::move(op->children[1]);
			has_changes = true;
			return;
		}
		if (cte.children[0]->HasSideEffects()) {
			// Never inline a side-effecting CTE: the LOGICAL_MATERIALIZED_CTE guarantees
			// that it executes exactly once and before the query side.
			return;
		}
		if (ContainsDelimGet(*cte.children[0]) && HasCTEReferenceBelowDelimJoin(*op->children[1], cte.table_index)) {
			// Inlining a CTE that already contains a DELIM_GET stays safe while all matching CTE scans remain outside
			// DELIM_JOIN subtrees, but once a scan is nested below another DELIM_JOIN the inlined DELIM_GETs can attach
			// to the wrong duplicate-elimination source.
			return;
		}
		if (cte.materialize == CTEMaterialize::CTE_MATERIALIZE_ALWAYS) {
			// This CTE is always materialized, we cannot inline it
			return;
		}
		if (ref_count == 1) {
			// this CTE is only referenced once, we can inline it directly without copying
			bool success = Inline(op->children[1], *op, false);
			if (success) {
				op = std::move(op->children[1]);
				has_changes = true;
			}
			return;
		}
		if (ref_count > 1) {
			if (cte.materialize == CTEMaterialize::CTE_MATERIALIZE_NEVER) {
				// this CTE is referenced multiple times, but we are not allowed to materialize it
				// we have to inline it if possible
				bool success = Inline(op->children[1], *op, true);
				if (success) {
					op = std::move(op->children[1]);
					has_changes = true;
				}
				return;
			}
			if (!cost_aware) {
				return;
			}
			// check if we can inline this CTE
			PreventInlining prevent_inlining;
			prevent_inlining.VisitOperator(*op->children[0]);

			if (prevent_inlining.prevent_inlining) {
				// we cannot inline this CTE, we have to keep it materialized
				return;
			}
			if (TryCostAwareInlining(op)) {
				return;
			}

			// Prevent inlining if the CTE ends in an aggregate or distinct operator
			// This mimics the behavior of the CTE materialization in the binder
			if (EndsInAggregateOrDistinct(*op->children[0])) {
				return;
			}

			bool is_cheap_to_inline = op->children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT ||
			                          op->children[0]->type == LogicalOperatorType::LOGICAL_CTE_REF ||
			                          EndsInDummyScan(*op->children[0]);

			// Check how many base table references the CTE has
			auto base_table_references = CountBaseTableReferences(*op->children[0]);

			if (!is_cheap_to_inline && base_table_references > 2 && base_table_references * ref_count > 10) {
				return;
			}

			// CTEs require full materialization before the CTE scans begin,
			// LIMIT and TOP_N operators cannot abort the materialization,
			// even if only a part of the CTE result is needed.
			// Therefore, we check if the CTE Scans are below the LIMIT or TOP_N operator
			// and if so, we try to inline the CTE definition.
			if (is_cheap_to_inline || ContainsLimit(*op->children[1])) {
				// this CTE is referenced multiple times and has a limit, we want to inline it
				bool success = Inline(op->children[1], *op, true);
				if (success) {
					op = std::move(op->children[1]);
					has_changes = true;
				}
				return;
			}
		}
	}
}

bool CTEInlining::TryCostAwareInlining(unique_ptr<LogicalOperator> &op) {
	auto &cte = op->Cast<LogicalMaterializedCTE>();
	if (ContainsCostSensitiveBlockingOperator(*cte.children[0])) {
		return false;
	}
	vector<CTEConsumerInfo> consumers;
	FindCTEConsumers(op->children[1], cte.table_index, consumers);
	if (consumers.size() < 2) {
		return false;
	}
	if (optimizer.OptimizerDisabled(OptimizerType::EXPRESSION_REWRITER) ||
	    optimizer.OptimizerDisabled(OptimizerType::CTE_FILTER_PUSHER) ||
	    optimizer.OptimizerDisabled(OptimizerType::FILTER_PUSHDOWN) ||
	    optimizer.OptimizerDisabled(OptimizerType::JOIN_ORDER) ||
	    optimizer.OptimizerDisabled(OptimizerType::STATISTICS_PROPAGATION)) {
		return false;
	}
	if (!CostEstimateSupported(*op, cte.table_index)) {
		return false;
	}

	const idx_t consumer_count = consumers.size();
	set<idx_t> all_consumers;
	set<idx_t> mandatory_materialized;
	const bool producer_contains_delim_get = ContainsDelimGet(*cte.children[0]);
	for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
		all_consumers.insert(consumer_idx);
		if (producer_contains_delim_get && consumers[consumer_idx].below_delim_join) {
			mandatory_materialized.insert(consumer_idx);
		}
	}

	bool legacy_inline = false;
	if (!EndsInAggregateOrDistinct(*cte.children[0])) {
		const bool cheap = cte.children[0]->type == LogicalOperatorType::LOGICAL_EMPTY_RESULT ||
		                   cte.children[0]->type == LogicalOperatorType::LOGICAL_CTE_REF ||
		                   EndsInDummyScan(*cte.children[0]);
		const auto base_table_references = CountBaseTableReferences(*cte.children[0]);
		const bool too_expensive_to_copy =
		    !cheap && base_table_references > 2 && base_table_references * consumer_count > 10;
		legacy_inline = !too_expensive_to_copy && (cheap || ContainsLimit(*cte.children[1]));
	}
	const set<idx_t> legacy_set = legacy_inline ? all_consumers : set<idx_t>();
	auto valid_subset = [&](const set<idx_t> &inline_set) {
		for (auto consumer_idx : mandatory_materialized) {
			if (inline_set.find(consumer_idx) != inline_set.end()) {
				return false;
			}
		}
		const auto residual_count = consumer_count - inline_set.size();
		if (residual_count != 1) {
			return true;
		}
		for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
			if (inline_set.find(consumer_idx) == inline_set.end()) {
				return mandatory_materialized.find(consumer_idx) != mandatory_materialized.end();
			}
		}
		return false;
	};

	auto apply_set = [&](unique_ptr<LogicalOperator> &candidate, const set<idx_t> &inline_set) {
		if (inline_set.empty()) {
			return true;
		}
		auto &candidate_cte = candidate->Cast<LogicalMaterializedCTE>();
		vector<CTEConsumerInfo> candidate_consumers;
		FindCTEConsumers(candidate_cte.children[1], candidate_cte.table_index, candidate_consumers);
		if (candidate_consumers.size() != consumer_count) {
			return false;
		}
		candidate_cte.children[0]->ResolveOperatorTypes();
		vector<unique_ptr<LogicalOperator>> replacements;
		replacements.reserve(inline_set.size());
		for (auto consumer_idx : inline_set) {
			auto &consumer = candidate_consumers[consumer_idx];
			auto &consumer_op = *consumer.owner.get();
			auto &cteref =
			    consumer.filtered ? consumer_op.children[0]->Cast<LogicalCTERef>() : consumer_op.Cast<LogicalCTERef>();
			LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
			unique_ptr<LogicalOperator> definition_copy;
			try {
				definition_copy = deep_copy.DeepCopy(candidate_cte.children[0]);
			} catch (NotImplementedException &ex) {
				return false;
			}
			if (consumer.filtered) {
				vector<reference<LogicalOperator>> filters {consumer_op};
				auto filter_expression = CTEFilterPusher::BuildFilterExpression(*definition_copy, filters);
				auto filter = make_uniq_base<LogicalOperator, LogicalFilter>(std::move(filter_expression));
				LogicalFilter::SplitPredicates(filter->Cast<LogicalFilter>().expressions);
				optimizer.rewriter.VisitOperator(*filter);
				filter->children.push_back(std::move(definition_copy));
				FilterPushdown pushdown(optimizer, true, FilterPushdown::ProjectionMode::PRESERVE_COMPUTED_EXPRESSIONS);
				definition_copy = pushdown.Rewrite(std::move(filter));
			}
			vector<unique_ptr<Expression>> projection_expressions;
			auto bindings = definition_copy->GetColumnBindings();
			for (idx_t column_idx = 0; column_idx < bindings.size(); column_idx++) {
				projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(
				    candidate_cte.children[0]->types[column_idx], bindings[column_idx]));
			}
			auto projection = make_uniq<LogicalProjection>(cteref.table_index, std::move(projection_expressions));
			projection->children.push_back(std::move(definition_copy));
			replacements.push_back(std::move(projection));
		}

		idx_t replacement_idx = 0;
		for (auto consumer_idx : inline_set) {
			candidate_consumers[consumer_idx].owner.get() = std::move(replacements[replacement_idx++]);
		}
		if (inline_set.size() == consumer_count) {
			candidate = std::move(candidate_cte.children[1]);
		}
		return true;
	};

	map<set<idx_t>, double> costs;
	idx_t evaluated_candidates = 0;
	bool legacy_all_direct = false;
	auto estimate_subset = [&](const set<idx_t> &inline_set) -> optional<double> {
		auto existing = costs.find(inline_set);
		if (existing != costs.end()) {
			return existing->second;
		}
		if (evaluated_candidates >= CTE_CANDIDATE_BUDGET) {
			return optional<double>();
		}
		evaluated_candidates++;
		TableIndexCheckpoint checkpoint(optimizer.binder);
		try {
			auto candidate = op->Copy(optimizer.context);
			if (!apply_set(candidate, inline_set)) {
				return optional<double>();
			}
			candidate = OptimizeCandidate(optimizer, std::move(candidate));
			auto estimate =
			    PhysicalPlanGenerator::EstimateCandidateCost(optimizer.context, std::move(candidate), cte.table_index);
			if (estimate) {
				if (inline_set == legacy_set && estimate->target_cte_found) {
					legacy_all_direct = estimate->direct_cte_consumers == consumer_count &&
					                    estimate->buffered_cte_consumers == 0 &&
					                    estimate->materialized_cte_consumers == 0;
				}
				costs.emplace(inline_set, estimate->cost);
				return estimate->cost;
			}
			return optional<double>();
		} catch (NotImplementedException &ex) {
			return optional<double>();
		}
	};

	auto legacy_cost = estimate_subset(legacy_set);
	if (!legacy_cost) {
		return false;
	}
	if (legacy_set.empty() && legacy_all_direct) {
		return true;
	}
	estimate_subset(set<idx_t>());
	if (valid_subset(all_consumers)) {
		estimate_subset(all_consumers);
	}

	set<set<idx_t>> candidates;
	candidates.insert(legacy_set);
	candidates.insert(set<idx_t>());
	if (valid_subset(all_consumers)) {
		candidates.insert(all_consumers);
	}
	if (consumer_count <= 4) {
		const auto subset_count = uint64_t(1) << consumer_count;
		for (uint64_t mask = 0; mask < subset_count; mask++) {
			set<idx_t> inline_set;
			for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
				if (mask & (uint64_t(1) << consumer_idx)) {
					inline_set.insert(consumer_idx);
				}
			}
			if (valid_subset(inline_set)) {
				candidates.insert(std::move(inline_set));
			}
		}
	} else {
		set<idx_t> current_set;
		while (current_set.size() < consumer_count) {
			candidates.insert(current_set);
			optional_idx best_consumer;
			double best_next_cost = NumericLimits<double>::Maximum();
			for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
				if (current_set.find(consumer_idx) != current_set.end() ||
				    mandatory_materialized.find(consumer_idx) != mandatory_materialized.end()) {
					continue;
				}
				auto next_set = current_set;
				next_set.insert(consumer_idx);
				if (!valid_subset(next_set)) {
					continue;
				}
				auto cost = estimate_subset(next_set);
				if (cost && *cost < best_next_cost) {
					best_next_cost = *cost;
					best_consumer = consumer_idx;
				}
			}
			if (!best_consumer.IsValid()) {
				break;
			}
			current_set.insert(best_consumer.GetIndex());
		}

		current_set.clear();
		for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
			if (mandatory_materialized.find(consumer_idx) == mandatory_materialized.end()) {
				current_set.insert(consumer_idx);
			}
		}
		while (!current_set.empty()) {
			if (valid_subset(current_set)) {
				candidates.insert(current_set);
			}
			optional_idx best_consumer;
			double best_next_cost = NumericLimits<double>::Maximum();
			for (auto consumer_idx : current_set) {
				auto next_set = current_set;
				next_set.erase(consumer_idx);
				if (!valid_subset(next_set)) {
					continue;
				}
				auto cost = estimate_subset(next_set);
				if (cost && *cost < best_next_cost) {
					best_next_cost = *cost;
					best_consumer = consumer_idx;
				}
			}
			if (!best_consumer.IsValid()) {
				break;
			}
			current_set.erase(best_consumer.GetIndex());
		}
	}

	double best_cost = *legacy_cost;
	set<idx_t> best_set = legacy_set;
	for (const auto &inline_set : candidates) {
		if (!valid_subset(inline_set) && inline_set != legacy_set) {
			continue;
		}
		auto cost = estimate_subset(inline_set);
		if (inline_set != legacy_set && (inline_set.empty() || inline_set.size() == consumer_count)) {
			continue;
		}
		if (cost && *cost < best_cost) {
			best_cost = *cost;
			best_set = inline_set;
		}
	}

	set<idx_t> selected_set = legacy_set;
	if (best_set != legacy_set && best_cost <= *legacy_cost * (1.0 - CTE_COST_IMPROVEMENT_THRESHOLD)) {
		selected_set = std::move(best_set);
	}
	if (selected_set.empty()) {
		return true;
	}
	if (!apply_set(op, selected_set)) {
		return false;
	}
	has_changes = true;
	return true;
}

bool CTEInlining::Inline(unique_ptr<LogicalOperator> &op, LogicalOperator &materialized_cte, bool requires_copy) {
	if (op->type == LogicalOperatorType::LOGICAL_CTE_REF) {
		auto &cteref = op->Cast<LogicalCTERef>();
		auto &cte = materialized_cte.Cast<LogicalCTE>();
		if (cteref.cte_index == cte.table_index) {
			unique_ptr<LogicalOperator> &definition = cte.children[0];
			unique_ptr<LogicalOperator> copy;
			if (requires_copy) {
				// there are multiple references to the CTE, we need to copy it
				LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
				try {
					copy = deep_copy.DeepCopy(definition);
				} catch (NotImplementedException &ex) {
					// if we have to copy the lhs of a CTE, but we cannot copy the operator, we have to
					// stop inlining and keep the materialized CTE instead
					return false;
				}
			}
			vector<unique_ptr<Expression>> proj_expressions;
			definition->ResolveOperatorTypes();
			vector<LogicalType> types = definition->types;
			vector<ColumnBinding> bindings =
			    requires_copy ? copy->GetColumnBindings() : definition->GetColumnBindings();

			idx_t col_idx = 0;
			for (auto &col : bindings) {
				proj_expressions.push_back(make_uniq<BoundColumnRefExpression>(types[col_idx], col));
				col_idx++;
			}
			auto proj = make_uniq<LogicalProjection>(cteref.table_index, std::move(proj_expressions));

			if (requires_copy) {
				proj->children.push_back(std::move(copy));
			} else {
				proj->children.push_back(std::move(definition));
			}
			op = std::move(proj);
			return true;
		}
		return true;
	} else {
		bool success = true;
		for (auto &child : op->children) {
			success &= Inline(child, materialized_cte, requires_copy);
		}
		return success;
	}
}

void PreventInlining::VisitOperator(LogicalOperator &op) {
	VisitOperatorExpressions(op);
	// We can stop checking early if we already know that inlining is not possible
	if (!prevent_inlining) {
		VisitOperatorChildren(op);
	}
}

void PreventInlining::VisitExpression(unique_ptr<Expression> *expression) {
	auto &expr = *expression;

	if (expr->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
		auto &bound_function = expr->Cast<BoundFunctionExpression>();
		// if we encounter the ErrorFun function, we still want to inline
		if (bound_function.Function().GetName() == "error") {
			return;
		}

		if (expr->IsVolatile()) {
			prevent_inlining = true;
			return;
		}
	}
	VisitExpressionChildren(**expression);
}

} // namespace duckdb
