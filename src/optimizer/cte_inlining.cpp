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
static constexpr idx_t CTE_LOCAL_CANDIDATE_BUDGET = 4;

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
	optional_ptr<LogicalOperator> filter;
	bool filtered;
	bool early_stop;
	bool below_delim_join;
};

static bool IsCTEReference(const LogicalOperator &op, TableIndex cte_index) {
	return op.type == LogicalOperatorType::LOGICAL_CTE_REF && op.Cast<LogicalCTERef>().cte_index == cte_index;
}

static void FindCTEConsumers(unique_ptr<LogicalOperator> &op, TableIndex cte_index, vector<CTEConsumerInfo> &consumers,
                             optional_ptr<LogicalOperator> filter = nullptr, bool filtered = false,
                             bool early_stop = false, bool below_delim_join = false) {
	const bool child_below_delim_join = below_delim_join || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN;
	if (IsCTEReference(*op, cte_index)) {
		consumers.push_back({op, filter, filtered, early_stop, below_delim_join});
		return;
	}
	const bool unary_path = op->children.size() == 1;
	const bool child_filtered = (unary_path && filtered) || op->type == LogicalOperatorType::LOGICAL_FILTER;
	optional_ptr<LogicalOperator> child_filter;
	if (op->type == LogicalOperatorType::LOGICAL_FILTER && op->children.size() == 1 &&
	    IsCTEReference(*op->children[0], cte_index)) {
		child_filter = op.get();
	}
	const bool child_early_stop = (unary_path && early_stop) || op->type == LogicalOperatorType::LOGICAL_LIMIT;
	for (auto &child : op->children) {
		FindCTEConsumers(child, cte_index, consumers, child_filter, child_filtered, child_early_stop,
		                 child_below_delim_join);
	}
}

static bool CostEstimateSupported(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
	case LogicalOperatorType::LOGICAL_DELIM_GET:
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:
		return false;
	default:
		break;
	}
	for (auto &child : op.children) {
		if (!CostEstimateSupported(*child)) {
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

unique_ptr<LogicalOperator> CTEInlining::OptimizeCostAware(unique_ptr<LogicalOperator> op,
                                                           bool optimizer_generated_only) {
	has_changes = false;
	generated_only = optimizer_generated_only;
	TryInlining(op, true);
	return op;
}

bool CTEInlining::HasChanges() const {
	return has_changes;
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
			if (generated_only && !cte.optimizer_generated) {
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
			return;
		}
	}
}

bool CTEInlining::TryCostAwareInlining(unique_ptr<LogicalOperator> &op) {
	auto &cte = op->Cast<LogicalMaterializedCTE>();
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
	if (!CostEstimateSupported(*op)) {
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

	const set<idx_t> origin_set;
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
			auto &cteref = consumer_op.Cast<LogicalCTERef>();
			LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
			unique_ptr<LogicalOperator> definition_copy;
			try {
				definition_copy = deep_copy.DeepCopy(candidate_cte.children[0]);
			} catch (NotImplementedException &ex) {
				return false;
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
	map<idx_t, double> inline_producer_costs;
	idx_t evaluated_candidates = 0;
	optional<double> origin_producer_work;
	auto estimate_inline_producer = [&](idx_t consumer_idx) -> optional<double> {
		auto existing = inline_producer_costs.find(consumer_idx);
		if (existing != inline_producer_costs.end()) {
			return existing->second;
		}
		if (!consumers[consumer_idx].filter || !optimizer.ConsumeCTECandidateBudget()) {
			return optional<double>();
		}
		TableIndexCheckpoint checkpoint(optimizer.binder);
		try {
			LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
			auto producer = deep_copy.DeepCopy(cte.children[0]);
			producer->ResolveOperatorTypes();
			vector<reference<LogicalOperator>> filters {*consumers[consumer_idx].filter};
			auto filter_expression = CTEFilterPusher::BuildFilterExpression(*producer, filters);
			auto filter = make_uniq_base<LogicalOperator, LogicalFilter>(std::move(filter_expression));
			LogicalFilter::SplitPredicates(filter->Cast<LogicalFilter>().expressions);
			optimizer.rewriter.VisitOperator(*filter);
			filter->children.push_back(std::move(producer));
			producer = OptimizeCandidate(optimizer, std::move(filter));
			auto estimate =
			    PhysicalPlanGenerator::EstimateCandidateCost(optimizer.context, std::move(producer), cte.table_index);
			if (!estimate || !estimate->reliable) {
				return optional<double>();
			}
			inline_producer_costs.emplace(consumer_idx, estimate->cost);
			return estimate->cost;
		} catch (NotImplementedException &ex) {
			return optional<double>();
		} catch (InternalException &ex) {
			return optional<double>();
		}
	};
	auto estimate_subset = [&](const set<idx_t> &inline_set) -> optional<double> {
		auto existing = costs.find(inline_set);
		if (existing != costs.end()) {
			return existing->second;
		}
		if (evaluated_candidates >= CTE_LOCAL_CANDIDATE_BUDGET || !optimizer.ConsumeCTECandidateBudget()) {
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
			if (estimate && estimate->reliable) {
				double cost = estimate->cost;
				if (inline_set == origin_set && estimate->target_cte_found) {
					origin_producer_work = estimate->target_cte_producer_work;
				}
				if (origin_producer_work) {
					cost = estimate->target_cte_exchange_work + estimate->target_cte_producer_work;
					for (auto consumer_idx : inline_set) {
						auto inline_cost = estimate_inline_producer(consumer_idx);
						cost += inline_cost ? *inline_cost : *origin_producer_work;
					}
				}
				costs.emplace(inline_set, cost);
				return cost;
			}
			return optional<double>();
		} catch (NotImplementedException &ex) {
			return optional<double>();
		} catch (InternalException &ex) {
			return optional<double>();
		}
	};

	auto origin_cost = estimate_subset(origin_set);
	if (!origin_cost) {
		return false;
	}

	set<set<idx_t>> candidates;
	candidates.insert(origin_set);
	candidates.insert(set<idx_t>());
	if (valid_subset(all_consumers)) {
		candidates.insert(all_consumers);
	}
	set<idx_t> reader_local_set;
	optional_idx strongest_reader;
	idx_t strongest_score = 0;
	for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
		if (mandatory_materialized.find(consumer_idx) != mandatory_materialized.end()) {
			continue;
		}
		const idx_t score = (consumers[consumer_idx].filtered ? 1 : 0) + (consumers[consumer_idx].early_stop ? 2 : 0);
		if (score > 0) {
			reader_local_set.insert(consumer_idx);
		}
		if (score > strongest_score) {
			strongest_score = score;
			strongest_reader = consumer_idx;
		}
	}
	if (valid_subset(reader_local_set)) {
		candidates.insert(reader_local_set);
	}
	if (strongest_reader.IsValid()) {
		set<idx_t> strongest_set {strongest_reader.GetIndex()};
		if (valid_subset(strongest_set)) {
			candidates.insert(std::move(strongest_set));
		}
	}

	double best_cost = *origin_cost;
	set<idx_t> best_set = origin_set;
	for (const auto &inline_set : candidates) {
		if (!valid_subset(inline_set) && inline_set != origin_set) {
			continue;
		}
		auto cost = estimate_subset(inline_set);
		if (cost && *cost < best_cost) {
			best_cost = *cost;
			best_set = inline_set;
		}
	}

	set<idx_t> selected_set = origin_set;
	if (best_set != origin_set && best_cost <= *origin_cost * (1.0 - CTE_COST_IMPROVEMENT_THRESHOLD)) {
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
