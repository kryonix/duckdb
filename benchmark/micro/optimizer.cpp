#include "benchmark_runner.hpp"
#include "duckdb_benchmark_macro.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/optimizer/projection_pullup.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"

using namespace duckdb;

struct OptimizerPlanningState : public DuckDBBenchmarkState {
	explicit OptimizerPlanningState(string query_p) : DuckDBBenchmarkState(string()), query(std::move(query_p)) {
	}

	string query;
};

struct ProjectionPullupState : public DuckDBBenchmarkState {
	explicit ProjectionPullupState(const string &query) : DuckDBBenchmarkState(string()) {
		conn.BeginTransaction();
		Parser parser(conn.context->GetParserOptions());
		parser.ParseQuery(query);
		if (parser.statements.size() != 1) {
			throw InvalidInputException("Projection pullup benchmark requires one statement");
		}
		planner = make_uniq<Planner>(*conn.context);
		planner->CreatePlan(std::move(parser.statements[0]));
		plan = std::move(planner->plan);
	}

	unique_ptr<Planner> planner;
	unique_ptr<LogicalOperator> plan;
};

static string GenerateProjectionChain(idx_t depth, idx_t width) {
	string select_list;
	for (idx_t col_idx = 0; col_idx < width; col_idx++) {
		if (col_idx > 0) {
			select_list += ", ";
		}
		select_list += StringUtil::Format("i AS c%llu", col_idx);
	}
	string query = StringUtil::Format("SELECT %s FROM range(1) source(i)", select_list);
	for (idx_t level = 0; level < depth; level++) {
		string projected_columns;
		for (idx_t col_idx = 0; col_idx < width; col_idx++) {
			if (col_idx > 0) {
				projected_columns += ", ";
			}
			projected_columns += StringUtil::Format("c%llu", col_idx);
		}
		query =
		    StringUtil::Format("SELECT %s FROM (%s) projection_%llu WHERE c0 >= 0", projected_columns, query, level);
	}
	return query;
}

static string GenerateLeftDeepJoin(idx_t relation_count) {
	D_ASSERT(relation_count > 0);
	string query = "SELECT count(*) FROM range(1) relation_0(key_0)";
	for (idx_t relation_idx = 1; relation_idx < relation_count; relation_idx++) {
		query += StringUtil::Format(" JOIN range(1) relation_%llu(key_%llu) ON key_0 = key_%llu", relation_idx,
		                            relation_idx, relation_idx);
	}
	return query + " WHERE key_0 >= 0";
}

static string GenerateNestedCorrelation(idx_t depth) {
	D_ASSERT(depth > 0);
	string subquery = StringUtil::Format("SELECT 1 FROM range(1) correlated_%llu(value_%llu) "
	                                     "WHERE value_%llu = value_%llu",
	                                     depth, depth, depth, depth - 1);
	for (idx_t level = depth - 1; level > 0; level--) {
		subquery = StringUtil::Format("SELECT 1 FROM range(1) correlated_%llu(value_%llu) "
		                              "WHERE value_%llu = value_%llu AND EXISTS (%s)",
		                              level, level, level, level - 1, subquery);
	}
	return StringUtil::Format("SELECT count(*) FROM range(1) correlated_0(value_0) WHERE EXISTS (%s)", subquery);
}

static string GenerateCommonSubplans(idx_t occurrence_count) {
	D_ASSERT(occurrence_count > 1);
	const string subplan = "SELECT i, i + 1 AS j FROM range(100) source(i) WHERE i % 2 = 0";
	string query;
	for (idx_t occurrence = 0; occurrence < occurrence_count; occurrence++) {
		if (occurrence > 0) {
			query += " UNION ALL ";
		}
		query += StringUtil::Format("SELECT * FROM (%s) common_%llu", subplan, occurrence);
	}
	return query;
}

static string GenerateComputedProjectionBranches(idx_t branch_count) {
	D_ASSERT(branch_count > 1);
	string query;
	for (idx_t branch_idx = 0; branch_idx < branch_count; branch_idx++) {
		if (branch_idx > 0) {
			query += " UNION ALL ";
		}
		query += StringUtil::Format("SELECT computed FROM (SELECT i, i + %llu AS computed "
		                            "FROM range(1) source_%llu(i)) projected_%llu WHERE i >= 0",
		                            branch_idx, branch_idx, branch_idx);
	}
	return query;
}

#define OPTIMIZER_PLANNING_BENCHMARK(NAME, QUERY)                                                                      \
	DUCKDB_BENCHMARK(NAME, "[optimizer_planning]")                                                                     \
	unique_ptr<DuckDBBenchmarkState> CreateBenchmarkState() override {                                                 \
		return make_uniq<OptimizerPlanningState>(QUERY);                                                               \
	}                                                                                                                  \
	void Load(DuckDBBenchmarkState *state) override {                                                                  \
	}                                                                                                                  \
	void RunBenchmark(DuckDBBenchmarkState *state) override {                                                          \
		auto &planning_state = static_cast<OptimizerPlanningState &>(*state);                                          \
		auto prepared = planning_state.conn.Prepare(planning_state.query);                                             \
		if (prepared->HasError()) {                                                                                    \
			throw InvalidInputException("Optimizer planning benchmark failed: %s", prepared->GetError());              \
		}                                                                                                              \
	}                                                                                                                  \
	string VerifyResult(QueryResult *result) override {                                                                \
		return string();                                                                                               \
	}                                                                                                                  \
	string BenchmarkInfo() override {                                                                                  \
		return "Prepare a generated query without executing it";                                                       \
	}                                                                                                                  \
	FINISH_BENCHMARK(NAME)

#define PROJECTION_PULLUP_BENCHMARK(NAME, QUERY)                                                                       \
	DUCKDB_BENCHMARK(NAME, "[optimizer_planning]")                                                                     \
	unique_ptr<DuckDBBenchmarkState> CreateBenchmarkState() override {                                                 \
		return make_uniq<ProjectionPullupState>(QUERY);                                                                \
	}                                                                                                                  \
	void Load(DuckDBBenchmarkState *state) override {                                                                  \
	}                                                                                                                  \
	void RunBenchmark(DuckDBBenchmarkState *state) override {                                                          \
		auto &pullup_state = static_cast<ProjectionPullupState &>(*state);                                             \
		Optimizer optimizer(*pullup_state.planner->binder, *pullup_state.conn.context);                                \
		ProjectionPullup projection_pullup(optimizer, pullup_state.plan);                                              \
		projection_pullup.Optimize(pullup_state.plan);                                                                 \
	}                                                                                                                  \
	string VerifyResult(QueryResult *result) override {                                                                \
		return string();                                                                                               \
	}                                                                                                                  \
	string BenchmarkInfo() override {                                                                                  \
		return "Run projection pullup on a generated bound plan";                                                      \
	}                                                                                                                  \
	bool RequireReinit() override {                                                                                    \
		return true;                                                                                                   \
	}                                                                                                                  \
	FINISH_BENCHMARK(NAME)

OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningShallow, "SELECT i FROM range(1) source(i) WHERE i = 0")
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningProjectionNarrow100, GenerateProjectionChain(100, 1))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningProjectionNarrow300, GenerateProjectionChain(300, 1))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningProjectionWide30x64, GenerateProjectionChain(30, 64))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningJoin30, GenerateLeftDeepJoin(30))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningJoin50, GenerateLeftDeepJoin(50))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningCorrelated10, GenerateNestedCorrelation(10))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningCorrelated15, GenerateNestedCorrelation(15))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningCommonSubplan10, GenerateCommonSubplans(10))
OPTIMIZER_PLANNING_BENCHMARK(OptimizerPlanningCommonSubplan50, GenerateCommonSubplans(50))

PROJECTION_PULLUP_BENCHMARK(OptimizerProjectionPullupNarrow100, GenerateProjectionChain(100, 1))
PROJECTION_PULLUP_BENCHMARK(OptimizerProjectionPullupNarrow300, GenerateProjectionChain(300, 1))
PROJECTION_PULLUP_BENCHMARK(OptimizerProjectionPullupWide30x64, GenerateProjectionChain(30, 64))
PROJECTION_PULLUP_BENCHMARK(OptimizerProjectionPullupComputedBranches10, GenerateComputedProjectionBranches(10))
PROJECTION_PULLUP_BENCHMARK(OptimizerProjectionPullupComputedBranches50, GenerateComputedProjectionBranches(50))
