//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/filter_pushdown.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/unordered_set.hpp"
#include "duckdb/optimizer/filter_combiner.hpp"
#include "duckdb/optimizer/rule.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/logical_plan_data_flow.hpp"

namespace duckdb {

class Optimizer;
class CTEFilterPusher;

class FilterPushdown {
public:
	enum class ProjectionMode : uint8_t { ALLOW_COMPUTED_EXPRESSIONS, PRESERVE_COMPUTED_EXPRESSIONS };

	explicit FilterPushdown(Optimizer &optimizer, bool convert_mark_joins = true,
	                        ProjectionMode projection_mode = ProjectionMode::ALLOW_COMPUTED_EXPRESSIONS);

	//! Perform filter pushdown
	unique_ptr<LogicalOperator> Rewrite(unique_ptr<LogicalOperator> op);
	//! Return a reference to the client context (from the optimizer)
	ClientContext &GetContext();

	void CheckMarkToSemi(LogicalOperator &op, const unordered_set<TableIndex> &table_bindings);

	struct Filter {
		unordered_set<TableIndex> bindings;
		unique_ptr<Expression> filter;

		Filter() {
		}
		explicit Filter(unique_ptr<Expression> filter) : filter(std::move(filter)) {
		}

		void ExtractBindings();
	};

private:
	friend class CTEFilterPusher;

	struct RewriteContext {
		RewriteContext(LogicalPlanDataFlow &data_flow_p, LogicalPlanDataFlowMutator &mutator_p)
		    : data_flow(data_flow_p), mutator(mutator_p) {
		}

		LogicalPlanDataFlow &data_flow;
		LogicalPlanDataFlowMutator &mutator;
	};

	enum class JoinDecisionPolicy {
		CROSS_PRODUCT,
		LEFT_JOIN,
		ASOF_JOIN,
		SINGLE_JOIN,
		MARK_JOIN,
		GENERATED_RIGHT_FILTER
	};

	struct IndexedJoinSourceResult {
		LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::SUCCESS;
		bool left = false;
		bool right = false;
		bool join = false;
	};

	struct IndexedJoinSideResult {
		LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::SUCCESS;
		JoinSide side = JoinSide::NONE;
	};

	struct IndexedDelimJoinResult {
		LogicalPlanDataFlowStatus status = LogicalPlanDataFlowStatus::SUCCESS;
		bool should_push = false;
	};

	LogicalPlanDataFlowOperatorResult GetFilterConvergence(const Filter &filter, LogicalOperator &consumer,
	                                                       RewriteContext &context) const;
	LogicalPlanDataFlowOperatorResult GetIndexedFilterTarget(const Filter &filter, LogicalOperator &consumer,
	                                                         RewriteContext &context) const;
	bool TryRewriteAtIndexedTarget(unique_ptr<LogicalOperator> &op, RewriteContext &context);

	class JoinBindingState {
	public:
		JoinBindingState(LogicalOperator &join, LogicalPlanDataFlow &data_flow);

		LogicalOperator &Join();
		void AddRightBinding(TableIndex table_index);
		bool IsExtraRightBinding(TableIndex table_index) const;
		LogicalPlanDataFlowOperatorResult ResolveSource(const ColumnBinding &binding, idx_t depth);
#ifdef DEBUG
		JoinSide GetLegacyJoinSide(const Filter &filter);
		JoinSide GetLegacyJoinSide(const Expression &expression);
#endif

	private:
#ifdef DEBUG
		void Initialize();
		void InitializeLeft();
		void InitializeRight();
#endif

	private:
		reference<LogicalOperator> join;
		reference<LogicalPlanDataFlow> data_flow;
#ifdef DEBUG
		bool left_initialized = false;
		bool right_initialized = false;
		unordered_set<TableIndex> left_bindings;
		unordered_set<TableIndex> right_bindings;
#endif
		unordered_set<TableIndex> extra_right_bindings;
		column_binding_map_t<LogicalPlanDataFlowOperatorResult> indexed_sources;
	};

#ifdef DEBUG
	enum class JoinFilterDecision {
		PUSH_LEFT,
		PUSH_RIGHT,
		CREATE_JOIN_CONDITION,
		INSPECT_NULL_REJECTION,
		INSPECT_MARK,
		KEEP,
		DISCARD
	};

	LogicalPlanDataFlowOperatorResult GetLegacyFilterTarget(LogicalOperator &consumer, LogicalOperator &convergence,
	                                                        RewriteContext &context) const;
	static bool IsLegacyFilterPushdownBoundary(const LogicalOperator &op);
	void VerifyIndexedFilterTargets(LogicalOperator &consumer, RewriteContext &context) const;
#endif

	Optimizer &optimizer;
	FilterCombiner combiner;
	bool convert_mark_joins;
	ProjectionMode projection_mode;

	vector<unique_ptr<Filter>> filters;
	void Rewrite(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalAggregate op
	void PushdownAggregate(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a distinct operator
	void PushdownDistinct(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalFilter op
	void PushdownFilter(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalCrossProduct op
	void PushdownCrossProduct(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a join operator
	void PushdownJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalProjection op
	void PushdownProjection(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Split a projection so filters can reuse computed outputs without forcing all expressions to be evaluated early
	void SplitProjection(unique_ptr<LogicalOperator> &op, vector<unique_ptr<Expression>> split_expressions,
	                     RewriteContext &context);
	//! Push down a LogicalProjection op
	void PushdownUnnest(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalSetOperation op
	void PushdownSetOperation(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalGet op
	void PushdownGet(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalLimit op
	void PushdownLimit(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Push down a LogicalWindow op
	void PushdownWindow(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	// Pushdown an inner join
	void PushdownInnerJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	// Pushdown a left join
	void PushdownLeftJoin(unique_ptr<LogicalOperator> &op, JoinBindingState &binding_state, RewriteContext &context);

	// Pushdown an outer join
	void PushdownOuterJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	void PushdownSemiAntiJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	// Pushdown a mark join
	void PushdownMarkJoin(unique_ptr<LogicalOperator> &op, JoinBindingState &binding_state, RewriteContext &context);
	// Pushdown a single join
	void PushdownSingleJoin(unique_ptr<LogicalOperator> &op, JoinBindingState &binding_state, RewriteContext &context);

	// AddLogicalFilter used to add an extra LogicalFilter at this level,
	// because in some cases, some expressions can not be pushed down.
	void AddLogicalFilter(unique_ptr<LogicalOperator> &op, vector<unique_ptr<Expression>> expressions,
	                      RewriteContext &context);
	//! Push any remaining filters into a LogicalFilter at this level
	void PushFinalFilters(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	// Finish pushing down at this operator, creating a LogicalFilter to store any of the stored filters and recursively
	// pushing down into its children (if any)
	void FinishPushdown(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	//! Adds a filter to the set of filters. Returns FilterResult::UNSATISFIABLE if the subtree should be stripped, or
	//! FilterResult::SUCCESS otherwise

	bool PushFiltersIntoDelimJoin(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	void ReplaceWithEmptyResult(unique_ptr<LogicalOperator> &op, RewriteContext &context);
	IndexedJoinSourceResult ClassifyJoinSources(const Expression &expression, JoinBindingState &binding_state) const;
	static IndexedJoinSideResult GetIndexedJoinSide(const IndexedJoinSourceResult &sources, JoinDecisionPolicy policy);
	JoinSide GetJoinSide(const Filter &filter, JoinDecisionPolicy policy, JoinBindingState &binding_state) const;
	JoinSide GetJoinSide(const Expression &expression, JoinDecisionPolicy policy,
	                     JoinBindingState &binding_state) const;
	IndexedDelimJoinResult GetIndexedDelimJoinDecision(const Expression &expression, LogicalOperator &child,
	                                                   JoinBindingState &binding_state) const;
#ifdef DEBUG
	static bool GetLegacyDelimJoinDecision(const Filter &filter, LogicalOperator &child);
#endif
#ifdef DEBUG
	static JoinFilterDecision GetJoinFilterDecision(JoinSide side, JoinDecisionPolicy policy);
#endif
	FilterResult AddFilter(unique_ptr<Expression> expr);
	//! Extract filter bindings to compare them with expressions in an operator and determine if the filter
	//! can be pushed down
	void ExtractFilterBindings(const Expression &expr, vector<ColumnBinding> &bindings);
	//! Generate filters from the current set of filters stored in the FilterCombiner
	void GenerateFilters();
	//! if there are filters in this FilterPushdown node, push them into the combiner. Returns
	//! FilterResult::UNSATISFIABLE if the subtree should be stripped, or FilterResult::SUCCESS otherwise
	FilterResult PushFilters();
};

} // namespace duckdb
