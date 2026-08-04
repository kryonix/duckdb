#include "duckdb/optimizer/cte_inlining.hpp"

#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/row/tuple_data_layout.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/cte_filter_pusher.hpp"
#include "duckdb/optimizer/filter_pushdown.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/list.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"

#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/operator/logical_prepare.hpp"

#include "duckdb/function/scalar/generic_functions.hpp"

namespace duckdb {

static constexpr double CTE_COST_IMPROVEMENT_THRESHOLD = 0.1;
static constexpr double CTE_DIRECT_FANOUT_COST_PER_ROW = 1.0;

struct CTEPlanEstimate {
	double processed_bytes = 0;
	double output_rows = 0;
	double row_width = 0;
	bool reliable = false;
};

enum class CTEConsumerExchangeMode : uint8_t { DIRECT, BUFFERED, MATERIALIZED };

struct CTEConsumerInfo {
	reference<unique_ptr<LogicalOperator>> owner;
	optional_ptr<LogicalOperator> filter;
	bool below_delim_join;
	CTEConsumerExchangeMode exchange_mode;
};

using CTEReferenceCountMap = reference_map_t<const LogicalOperator, idx_t>;

static idx_t CountCTEReferences(const LogicalOperator &op, TableIndex cte_index,
                                optional_ptr<CTEReferenceCountMap> reference_counts = nullptr);

static bool IsCTEReference(const LogicalOperator &op, TableIndex cte_index) {
	return op.type == LogicalOperatorType::LOGICAL_CTE_REF && op.Cast<LogicalCTERef>().cte_index == cte_index;
}

static bool MaterializesLocalCTEConsumer(LogicalOperatorType type) {
	switch (type) {
	case LogicalOperatorType::LOGICAL_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
	case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
	case LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
	case LogicalOperatorType::LOGICAL_ORDER_BY:
	case LogicalOperatorType::LOGICAL_TOP_N:
		return true;
	default:
		return false;
	}
}

static void FindCTEConsumers(unique_ptr<LogicalOperator> &op, TableIndex cte_index,
                             const CTEReferenceCountMap &reference_counts, vector<CTEConsumerInfo> &consumers,
                             bool below_delim_join = false,
                             CTEConsumerExchangeMode exchange_mode = CTEConsumerExchangeMode::DIRECT) {
	auto reference_count = reference_counts.find(*op);
	D_ASSERT(reference_count != reference_counts.end());
	if (reference_count->second == 1) {
		if (MaterializesLocalCTEConsumer(op->type)) {
			exchange_mode = CTEConsumerExchangeMode::MATERIALIZED;
		} else if (op->type == LogicalOperatorType::LOGICAL_LIMIT && exchange_mode == CTEConsumerExchangeMode::DIRECT) {
			exchange_mode = CTEConsumerExchangeMode::BUFFERED;
		}
	}
	if (IsCTEReference(*op, cte_index)) {
		consumers.push_back({op, nullptr, below_delim_join, exchange_mode});
		return;
	}
	const bool child_below_delim_join = below_delim_join || op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN;
	if (op->type == LogicalOperatorType::LOGICAL_FILTER && op->children.size() == 1 &&
	    IsCTEReference(*op->children[0], cte_index)) {
		consumers.push_back({op->children[0], *op, child_below_delim_join, exchange_mode});
		return;
	}
	for (auto &child : op->children) {
		FindCTEConsumers(child, cte_index, reference_counts, consumers, child_below_delim_join, exchange_mode);
	}
}

static double EstimateRowWidth(const vector<LogicalType> &types) {
	if (types.empty()) {
		return 1;
	}
	TupleDataLayout tuple_layout;
	tuple_layout.Initialize(types, TupleDataValidityType::CAN_HAVE_NULL_VALUES);
	double row_width = static_cast<double>(tuple_layout.GetRowWidth());
	for (const auto &type : types) {
		TypeVisitor::VisitReplace(type, [&](const LogicalType &visited_type) {
			switch (visited_type.InternalType()) {
			case PhysicalType::VARCHAR:
				row_width += 8;
				break;
			case PhysicalType::LIST:
			case PhysicalType::ARRAY:
				row_width += 32;
				break;
			default:
				break;
			}
			row_width += 2;
			return visited_type;
		});
	}
	return row_width;
}

static bool CostEstimateSupported(const LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
	case LogicalOperatorType::LOGICAL_CTE_REF:
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

static double EstimateProcessedBytes(LogicalOperator &op, ClientContext &context) {
	double result = 0;
	for (auto &child : op.children) {
		result += EstimateProcessedBytes(*child, context);
	}
	const auto cardinality = op.has_estimated_cardinality ? op.estimated_cardinality : op.EstimateCardinality(context);
	result += static_cast<double>(cardinality) * EstimateRowWidth(op.types);
	return result;
}

static CTEPlanEstimate EstimateCTEPlan(Optimizer &optimizer, optional_ptr<bound_parameter_map_t> parameter_data,
                                       unique_ptr<LogicalOperator> &definition,
                                       const vector<reference<LogicalOperator>> &filters) {
	CTEPlanEstimate result;
	if (!CostEstimateSupported(*definition)) {
		return result;
	}

	unique_ptr<LogicalOperator> plan;
	try {
		// Preserve table indexes so rejected estimates do not consume binder state.
		plan = definition->Copy(optimizer.context);
		unordered_map<TableIndex, TableIndex> table_idx_replacements;
		TableBindingReplacer replacer(table_idx_replacements, parameter_data);
		replacer.VisitOperator(*plan);
	} catch (NotImplementedException &ex) {
		return result;
	}

	if (!filters.empty()) {
		auto filter_expression = CTEFilterPusher::BuildFilterExpression(*plan, filters);
		auto filter = make_uniq_base<LogicalOperator, LogicalFilter>(std::move(filter_expression));
		LogicalFilter::SplitPredicates(filter->Cast<LogicalFilter>().expressions);
		optimizer.rewriter.VisitOperator(*filter);
		filter->children.push_back(std::move(plan));

		FilterPushdown pushdown(optimizer, true, FilterPushdown::ProjectionMode::PRESERVE_COMPUTED_EXPRESSIONS);
		plan = pushdown.Rewrite(std::move(filter));
	}

	JoinOrderOptimizer join_order(optimizer.context);
	plan = join_order.Optimize(std::move(plan));
	plan->ResolveOperatorTypes();
	result.output_rows = static_cast<double>(plan->EstimateCardinality(optimizer.context));
	result.row_width = EstimateRowWidth(plan->types);
	result.processed_bytes = EstimateProcessedBytes(*plan, optimizer.context);
	result.reliable = true;
	return result;
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

static idx_t CountCTEReferences(const LogicalOperator &op, TableIndex cte_index,
                                optional_ptr<CTEReferenceCountMap> reference_counts) {
	idx_t number_of_references = 0;
	if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		auto &cte = op.Cast<LogicalCTERef>();
		if (cte.cte_index == cte_index) {
			number_of_references = 1;
		}
	}
	for (auto &child : op.children) {
		number_of_references += CountCTEReferences(*child, cte_index, reference_counts);
	}
	if (reference_counts) {
		(*reference_counts)[op] = number_of_references;
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
	CTEReferenceCountMap reference_counts;
	CountCTEReferences(*op->children[1], cte.table_index, reference_counts);
	vector<CTEConsumerInfo> consumers;
	FindCTEConsumers(op->children[1], cte.table_index, reference_counts, consumers);
	if (consumers.size() < 2 || consumers.size() > 63) {
		return false;
	}

	const idx_t consumer_count = consumers.size();
	const uint64_t all_consumers = (uint64_t(1) << consumer_count) - 1;
	uint64_t mandatory_materialized = 0;
	const bool producer_contains_delim_get = ContainsDelimGet(*cte.children[0]);
	for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
		if (producer_contains_delim_get && consumers[consumer_idx].below_delim_join) {
			mandatory_materialized |= uint64_t(1) << consumer_idx;
		}
	}
	bool has_materialized_consumer = false;
	for (auto &consumer : consumers) {
		if (consumer.exchange_mode == CTEConsumerExchangeMode::MATERIALIZED) {
			has_materialized_consumer = true;
			break;
		}
	}
	if (!has_materialized_consumer) {
		return false;
	}

	vector<CTEPlanEstimate> inline_estimates;
	inline_estimates.reserve(consumer_count);
	for (auto &consumer : consumers) {
		vector<reference<LogicalOperator>> filters;
		if (consumer.filter) {
			filters.push_back(*consumer.filter);
		}
		auto estimate = EstimateCTEPlan(optimizer, parameter_data, cte.children[0], filters);
		if (!estimate.reliable) {
			return false;
		}
		inline_estimates.push_back(estimate);
	}

	unordered_map<uint64_t, CTEPlanEstimate> residual_estimates;
	auto get_residual_estimate = [&](uint64_t inline_mask) -> CTEPlanEstimate & {
		auto entry = residual_estimates.find(inline_mask);
		if (entry != residual_estimates.end()) {
			return entry->second;
		}
		vector<reference<LogicalOperator>> filters;
		bool all_materialized_consumers_are_filtered = true;
		for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
			if (inline_mask & (uint64_t(1) << consumer_idx)) {
				continue;
			}
			if (!consumers[consumer_idx].filter) {
				all_materialized_consumers_are_filtered = false;
				break;
			}
			filters.push_back(*consumers[consumer_idx].filter);
		}
		if (!all_materialized_consumers_are_filtered) {
			filters.clear();
		}
		auto estimate = EstimateCTEPlan(optimizer, parameter_data, cte.children[0], filters);
		return residual_estimates.emplace(inline_mask, estimate).first->second;
	};

	auto estimate_subset = [&](uint64_t inline_mask, bool &reliable) {
		double cost = 0;
		idx_t materialized_count = 0;
		idx_t direct_count = 0;
		idx_t buffered_count = 0;
		for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
			if (inline_mask & (uint64_t(1) << consumer_idx)) {
				cost += inline_estimates[consumer_idx].processed_bytes;
			} else {
				switch (consumers[consumer_idx].exchange_mode) {
				case CTEConsumerExchangeMode::DIRECT:
					direct_count++;
					break;
				case CTEConsumerExchangeMode::BUFFERED:
					buffered_count++;
					break;
				case CTEConsumerExchangeMode::MATERIALIZED:
					materialized_count++;
					break;
				}
			}
		}
		const auto residual_count = direct_count + buffered_count + materialized_count;
		if (residual_count > 0) {
			auto &residual = get_residual_estimate(inline_mask);
			if (!residual.reliable) {
				reliable = false;
				return 0.0;
			}
			const auto output_bytes = residual.output_rows * residual.row_width;
			cost += residual.processed_bytes;
			cost += residual.output_rows * static_cast<double>(direct_count) * CTE_DIRECT_FANOUT_COST_PER_ROW;
			cost += output_bytes * static_cast<double>(buffered_count);
			if (materialized_count > 0) {
				cost += output_bytes * static_cast<double>(materialized_count + 1);
			}
		}
		reliable = true;
		return cost;
	};

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
	const uint64_t legacy_mask = legacy_inline ? all_consumers : 0;

	bool legacy_reliable;
	const auto legacy_cost = estimate_subset(legacy_mask, legacy_reliable);
	if (!legacy_reliable) {
		return false;
	}
	double best_cost = legacy_cost;
	uint64_t best_mask = legacy_mask;
	auto valid_subset = [&](uint64_t inline_mask) {
		if (inline_mask & mandatory_materialized) {
			return false;
		}
		const auto materialized_mask = all_consumers & ~inline_mask;
		if (materialized_mask && (materialized_mask & (materialized_mask - 1)) == 0 &&
		    !(materialized_mask & mandatory_materialized)) {
			return false;
		}
		return true;
	};

	set<uint64_t> candidates;
	candidates.insert(legacy_mask);
	if (consumer_count <= 4) {
		for (uint64_t inline_mask = 0; inline_mask <= all_consumers; inline_mask++) {
			if (valid_subset(inline_mask)) {
				candidates.insert(inline_mask);
			}
		}
	} else {
		uint64_t current_mask = 0;
		while (true) {
			candidates.insert(current_mask);
			optional_idx best_consumer;
			double best_next_cost = NumericLimits<double>::Maximum();
			for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
				const auto bit = uint64_t(1) << consumer_idx;
				if ((current_mask & bit) || (mandatory_materialized & bit)) {
					continue;
				}
				const auto next_mask = current_mask | bit;
				if (!valid_subset(next_mask)) {
					continue;
				}
				bool reliable;
				const auto cost = estimate_subset(next_mask, reliable);
				if (!reliable) {
					return false;
				}
				if (cost < best_next_cost) {
					best_next_cost = cost;
					best_consumer = consumer_idx;
				}
			}
			if (!best_consumer.IsValid()) {
				break;
			}
			current_mask |= uint64_t(1) << best_consumer.GetIndex();
		}

		current_mask = all_consumers & ~mandatory_materialized;
		if (mandatory_materialized == 0) {
			uint64_t best_seed = current_mask;
			double best_seed_cost = NumericLimits<double>::Maximum();
			vector<idx_t> seed_candidates;
			for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
				seed_candidates.push_back(consumer_idx);
			}
			std::sort(seed_candidates.begin(), seed_candidates.end(), [&](idx_t left, idx_t right) {
				return inline_estimates[left].processed_bytes > inline_estimates[right].processed_bytes;
			});
			seed_candidates.resize(MinValue<idx_t>(seed_candidates.size(), 8));
			for (idx_t left_pos = 0; left_pos < seed_candidates.size(); left_pos++) {
				for (idx_t right_pos = left_pos + 1; right_pos < seed_candidates.size(); right_pos++) {
					const auto left_idx = seed_candidates[left_pos];
					const auto right_idx = seed_candidates[right_pos];
					const auto seed_mask = current_mask & ~(uint64_t(1) << left_idx) & ~(uint64_t(1) << right_idx);
					bool reliable;
					const auto cost = estimate_subset(seed_mask, reliable);
					if (!reliable) {
						return false;
					}
					if (cost < best_seed_cost) {
						best_seed_cost = cost;
						best_seed = seed_mask;
					}
				}
			}
			candidates.insert(current_mask);
			current_mask = best_seed;
		}
		while (true) {
			if (valid_subset(current_mask)) {
				candidates.insert(current_mask);
			}
			optional_idx best_consumer;
			double best_next_cost = NumericLimits<double>::Maximum();
			for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
				const auto bit = uint64_t(1) << consumer_idx;
				if (!(current_mask & bit)) {
					continue;
				}
				const auto next_mask = current_mask & ~bit;
				if (!valid_subset(next_mask)) {
					continue;
				}
				bool reliable;
				const auto cost = estimate_subset(next_mask, reliable);
				if (!reliable) {
					return false;
				}
				if (cost < best_next_cost) {
					best_next_cost = cost;
					best_consumer = consumer_idx;
				}
			}
			if (!best_consumer.IsValid()) {
				break;
			}
			current_mask &= ~(uint64_t(1) << best_consumer.GetIndex());
		}
	}

	for (auto inline_mask : candidates) {
		if (!valid_subset(inline_mask) && inline_mask != legacy_mask) {
			continue;
		}
		bool reliable;
		const auto cost = estimate_subset(inline_mask, reliable);
		if (!reliable) {
			return false;
		}
		if (cost < best_cost) {
			best_cost = cost;
			best_mask = inline_mask;
		}
	}

	uint64_t selected_mask = legacy_mask;
	const bool mixed_plan = best_mask != 0 && best_mask != all_consumers;
	if (mixed_plan && best_mask != legacy_mask && best_cost <= legacy_cost * (1.0 - CTE_COST_IMPROVEMENT_THRESHOLD)) {
		selected_mask = best_mask;
	}
	if (selected_mask == 0) {
		return true;
	}

	cte.children[0]->ResolveOperatorTypes();
	vector<unique_ptr<LogicalOperator>> replacements;
	replacements.reserve(consumer_count);
	for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
		if (!(selected_mask & (uint64_t(1) << consumer_idx))) {
			continue;
		}
		auto &cteref = consumers[consumer_idx].owner.get()->Cast<LogicalCTERef>();
		LogicalOperatorDeepCopy deep_copy(optimizer.binder, parameter_data);
		unique_ptr<LogicalOperator> copy;
		try {
			copy = deep_copy.DeepCopy(cte.children[0]);
		} catch (NotImplementedException &ex) {
			return false;
		}

		vector<unique_ptr<Expression>> projection_expressions;
		auto bindings = copy->GetColumnBindings();
		for (idx_t column_idx = 0; column_idx < bindings.size(); column_idx++) {
			projection_expressions.push_back(
			    make_uniq<BoundColumnRefExpression>(cte.children[0]->types[column_idx], bindings[column_idx]));
		}
		auto projection = make_uniq<LogicalProjection>(cteref.table_index, std::move(projection_expressions));
		projection->children.push_back(std::move(copy));
		replacements.push_back(std::move(projection));
	}

	idx_t replacement_idx = 0;
	for (idx_t consumer_idx = 0; consumer_idx < consumer_count; consumer_idx++) {
		if (selected_mask & (uint64_t(1) << consumer_idx)) {
			consumers[consumer_idx].owner.get() = std::move(replacements[replacement_idx++]);
		}
	}
	if (selected_mask == all_consumers) {
		op = std::move(cte.children[1]);
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
