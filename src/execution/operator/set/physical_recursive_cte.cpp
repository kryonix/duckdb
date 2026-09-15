#include "duckdb/execution/operator/set/physical_recursive_cte_state.hpp"
#include "duckdb/execution/operator/set/physical_recursive_cte_delta.hpp"

#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/common/vector/flat_vector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/pipeline_executor.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/common/radix_partitioning.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include "duckdb/main/settings.hpp"

#include <functional>

namespace duckdb {

RecursiveCTEPartialKeySpec::RecursiveCTEPartialKeySpec(vector<idx_t> indices_p, idx_t full_key_count)
    : indices(std::move(indices_p)) {
	if (indices.empty() || indices.size() >= full_key_count || !std::is_sorted(indices.begin(), indices.end()) ||
	    std::adjacent_find(indices.begin(), indices.end()) != indices.end() || indices.back() >= full_key_count) {
		throw InternalException("Invalid USING KEY partial-key index specification");
	}
}

struct RecursiveCTEDistinctPartition {
	RecursiveCTEDistinctPartition(ClientContext &context, const vector<LogicalType> &types)
	    : ht(context, BufferAllocator::Get(context), types) {
	}

	mutex lock;
	GroupedAggregateHashTable ht;
};

PhysicalRecursiveCTE::PhysicalRecursiveCTE(PhysicalPlan &physical_plan, Identifier ctename, TableIndex table_index,
                                           vector<LogicalType> types, bool union_all, PhysicalOperator &top,
                                           PhysicalOperator &bottom, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::RECURSIVE_CTE, std::move(types), estimated_cardinality),
      ctename(std::move(ctename)), table_index(table_index), union_all(union_all),
      shared_executor_pool(make_shared_ptr<RecursiveExecutorPool>()) {
	children.push_back(top);
	children.push_back(bottom);
}

PhysicalRecursiveCTE::~PhysicalRecursiveCTE() {
}

idx_t PhysicalRecursiveCTE::NextMetricsInvocation() const {
	return metrics_invocations.fetch_add(1) + 1;
}

//===--------------------------------------------------------------------===//
// Sink State
//===--------------------------------------------------------------------===//
// Partition commits below this many rows in flight are not worth a task each.
static constexpr const idx_t KEYED_COMMIT_ROWS_PER_TASK = 4 * STANDARD_VECTOR_SIZE;
// Partitions are only worth their per-epoch bookkeeping once a commit has work for several tasks.
static constexpr const idx_t KEYED_PROMOTION_ROWS = 2 * KEYED_COMMIT_ROWS_PER_TASK;
// Routing and partition-major scans have a price too, so the serial commit must be at least this share of an epoch.
static constexpr const idx_t KEYED_PROMOTION_COMMIT_SHARE_DIVISOR = 5;
// Each partition costs a hash-table probe per routed chunk and a scan cursor per epoch, so cap the fan-out.
static constexpr const idx_t MAX_KEYED_PARTITIONS = 16;
// Rows of frozen state or epoch output that justify one more scan task, independent of the vector size so small
// results keep a single task at any vector size.
static constexpr const idx_t SCAN_ROWS_PER_TASK = 2048;

static idx_t ScanTasksForRows(idx_t rows) {
	return MaxValue<idx_t>((rows + SCAN_ROWS_PER_TASK - 1) / SCAN_ROWS_PER_TASK, 1);
}

static idx_t GetKeyedPartitionTarget(ClientContext &context) {
	const auto threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	if (threads <= 1) {
		return 1;
	}
	return MinValue<idx_t>(NextPowerOfTwo(threads), MAX_KEYED_PARTITIONS);
}

RecursiveCTEKeyedPartition::RecursiveCTEKeyedPartition(ClientContext &context, const PhysicalRecursiveCTE &op,
                                                       const vector<AggregateObject> &payload_aggregate_objects,
                                                       bool own_frontier)
    : candidates(context, op.internal_types),
      owned_frontier(own_frontier ? make_uniq<ColumnDataCollection>(context, op.working_table->Types()) : nullptr),
      frontier(owned_frontier ? *owned_frontier : *op.working_table), payload_executor(context),
      new_group_addresses(LogicalType::POINTER), new_groups(STANDARD_VECTOR_SIZE),
      preaggregation_hashes(LogicalType::HASH) {
	vector<LogicalType> aggr_input_types;
	for (auto &payload_aggregate : op.payload_aggregates) {
		auto &bound_aggr_expr = payload_aggregate->Cast<BoundAggregateExpression>();
		for (auto &child_expr : bound_aggr_expr.GetChildren()) {
			payload_executor.AddExpression(*child_expr);
			aggr_input_types.push_back(child_expr->GetReturnType());
		}
	}
	if (!op.key_normalizers.empty()) {
		key_executor = make_uniq<ExpressionExecutor>(context);
		for (auto &normalizer : op.key_normalizers) {
			key_executor->AddExpression(*normalizer);
		}
		raw_distinct_rows.Initialize(Allocator::DefaultAllocator(), op.distinct_types);
	}
	for (auto &comparison : op.payload_comparisons) {
		if (comparison) {
			payload_comparison_executors.push_back(make_uniq<ExpressionExecutor>(context, *comparison));
		} else {
			payload_comparison_executors.push_back(nullptr);
		}
	}
	ht = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.hash_key_types,
	                                          op.aggregate_types, payload_aggregate_objects);
	if (!op.union_all) {
		key_delta = make_uniq<RecursiveCTEKeyDeltaState>(context, op);
	}
	payload_rows.Initialize(Allocator::Get(context), aggr_input_types);
	distinct_rows.Initialize(Allocator::DefaultAllocator(), op.hash_key_types);
	update_rows.Initialize(Allocator::DefaultAllocator(), op.internal_types);
	InitializeCandidateAppend();
	InitializeFrontierAppend();
}

RecursiveCTEKeyedPartition::~RecursiveCTEKeyedPartition() {
}

void RecursiveCTEKeyedPartition::InitializeCandidateAppend() {
	candidates.InitializeAppend(candidate_append_state);
}

void RecursiveCTEKeyedPartition::InitializeFrontierAppend() {
	frontier.InitializeAppend(frontier_append_state);
}

bool RecursiveCTEKeyedPartition::HasWork() const {
	return candidates.Count() > 0 || !preaggregated.empty();
}

idx_t RecursiveCTEKeyedPartition::WorkRows() const {
	idx_t rows = candidates.Count();
	for (auto &preaggregate : preaggregated) {
		rows += preaggregate.candidate_rows;
	}
	return rows;
}

RecursiveCTEState::RecursiveCTEState(ClientContext &context, const PhysicalRecursiveCTE &op)
    : op(op), allow_executor_reuse(Settings::Get<EnableCachingOperatorsSetting>(context)), metrics(context, op),
      scheduler(op.shared_executor_pool, allow_executor_reuse),
      intermediate_table(context, op.using_key ? op.internal_types : op.GetTypes()), context(context) {
	if (metrics.Enabled()) {
		epoch_metrics = make_uniq<RecursiveCTEEpochMetrics>();
	}
	for (idx_t i = 0; i < op.payload_aggregates.size(); i++) {
		D_ASSERT(op.payload_aggregates[i]->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE);
		auto &bound_aggr_expr = op.payload_aggregates[i]->Cast<BoundAggregateExpression>();
		payload_aggregate_objects.emplace_back(bound_aggr_expr);
		finalize_requires_lock = finalize_requires_lock || !bound_aggr_expr.Function().FinalizeIsReadOnly();
	}
	for (auto &comparison : op.payload_comparisons) {
		if (comparison) {
			has_payload_comparison_executors = true;
		}
	}

	if (op.using_key) {
		for (auto &spec : op.partial_key_index_specs) {
			partial_key_indexes.push_back(
			    make_uniq<RecursiveCTEPartialKeyIndex>(Allocator::Get(context), op.hash_key_types, spec.Indices()));
		}
		if (!op.union_all) {
			can_preaggregate_using_key = true;
			for (idx_t payload_idx = 0; payload_idx < op.payload_types.size(); payload_idx++) {
				auto &aggregate = payload_aggregate_objects[payload_idx];
				if (!aggregate.function.HasStateCombineCallback() ||
				    aggregate.function.GetOrderDependent() == AggregateOrderDependent::ORDER_DEPENDENT) {
					can_preaggregate_using_key = false;
					break;
				}
			}
			can_reuse_new_group_candidates = op.internal_types == op.GetTypes();
			for (idx_t payload_idx = 0; can_reuse_new_group_candidates && payload_idx < op.payload_types.size();
			     payload_idx++) {
				auto &aggregate = op.payload_aggregates[payload_idx]->Cast<BoundAggregateExpression>();
				auto &children = aggregate.GetChildren();
				can_reuse_new_group_candidates =
				    aggregate.Function().HasSingleValueIdentity() && !children.empty() &&
				    children[0]->GetExpressionClass() == ExpressionClass::BOUND_REF &&
				    children[0]->Cast<BoundReferenceExpression>().Index() == op.payload_idx[payload_idx];
			}
			can_reuse_changed_group_candidates =
			    can_reuse_new_group_candidates && op.key_normalizers.empty() && !has_payload_comparison_executors;
			for (auto &key_type : op.hash_key_types) {
				switch (key_type.InternalType()) {
				case PhysicalType::FLOAT:
				case PhysicalType::DOUBLE:
				case PhysicalType::LIST:
				case PhysicalType::STRUCT:
				case PhysicalType::ARRAY:
				case PhysicalType::UNKNOWN:
					can_reuse_changed_group_candidates = false;
					break;
				default:
					break;
				}
			}
			for (auto &payload_type : op.payload_types) {
				switch (payload_type.InternalType()) {
				case PhysicalType::LIST:
				case PhysicalType::STRUCT:
				case PhysicalType::ARRAY:
				case PhysicalType::UNKNOWN:
					can_reuse_changed_group_candidates = false;
					break;
				default:
					break;
				}
			}
		}
		// Small recursions keep one partition writing the working table directly; PrepareKeyedCommit promotes
		keyed_partition_target = GetKeyedPartitionTarget(context);
		keyed_partitions.push_back(
		    make_uniq<RecursiveCTEKeyedPartition>(context, op, payload_aggregate_objects, false));
		keyed_layout = keyed_partitions[0]->ht->GetLayoutPtr();
		metrics.RecordKeyedPartitions(1);
	} else if (!op.union_all) {
		distinct_ht = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.distinct_types);
	}
	source_result.Initialize(Allocator::DefaultAllocator(), op.GetTypes());
	if (op.recurring_table) {
		op.recurring_table->InitializeAppend(recurring_append_state);
	}
}

unique_ptr<GroupedAggregateHashTable> RecursiveCTEState::CreateUsingKeyHashTable() const {
	return make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.hash_key_types,
	                                            op.aggregate_types, payload_aggregate_objects);
}

RecursiveCTEState::~RecursiveCTEState() {
	metrics.Log(partial_key_indexes);
	if (epoch_metrics) {
		metrics.LogEpochSummary(*epoch_metrics);
	}
}

const RecursiveCTEPartialKeyIndex &RecursiveCTEState::GetPartialKeyIndex(const vector<idx_t> &key_indices) const {
	for (auto &index : partial_key_indexes) {
		if (index->key_indices == key_indices) {
			return *index;
		}
	}
	throw InternalException("USING KEY partial-key index is missing");
}

void RecursiveCTEState::RecordSinkMetrics(idx_t wait_ns, idx_t work_ns, idx_t rows) {
	metrics.RecordSink(wait_ns, work_ns, rows);
}

void RecursiveCTEState::AppendOutput(DataChunk &chunk) {
	D_ASSERT(!op.using_key);
	const auto collect_metrics = metrics.Enabled();
	const auto before_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	lock_guard<mutex> guard(intermediate_table_lock);
	const auto after_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	CurrentOutputTable().Append(CurrentOutputAppendState(), chunk);
	if (collect_metrics) {
		const auto after_work = std::chrono::steady_clock::now();
		RecordSinkMetrics(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
		    chunk.size());
	}
}

void RecursiveCTEState::CombineOutput(ColumnDataCollection &output) {
	D_ASSERT(!op.using_key);
	const auto collect_metrics = metrics.Enabled();
	const auto before_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	lock_guard<mutex> guard(intermediate_table_lock);
	const auto after_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	const auto row_count = output.Count();
	CurrentOutputTable().Combine(output);
	if (collect_metrics) {
		const auto after_work = std::chrono::steady_clock::now();
		RecordSinkMetrics(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
		    row_count);
	}
}

void RecursiveCTEState::AppendCandidates(idx_t partition_idx, DataChunk &chunk) {
	D_ASSERT(op.using_key);
	auto &partition = *keyed_partitions[partition_idx];
	const auto collect_metrics = metrics.Enabled();
	const auto before_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	lock_guard<mutex> guard(partition.lock);
	const auto after_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	// Candidates are collected without mutating the hash state read by recurring.T in this epoch.
	partition.candidates.Append(partition.candidate_append_state, chunk);
	if (collect_metrics) {
		const auto after_work = std::chrono::steady_clock::now();
		RecordSinkMetrics(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
		    chunk.size());
	}
}

void RecursiveCTEState::CombineCandidates(idx_t partition_idx, ColumnDataCollection &output) {
	D_ASSERT(op.using_key);
	auto &partition = *keyed_partitions[partition_idx];
	const auto collect_metrics = metrics.Enabled();
	const auto before_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	lock_guard<mutex> guard(partition.lock);
	const auto after_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	const auto row_count = output.Count();
	partition.candidates.Combine(output);
	// Workers can resume shared appends after publishing local output.
	partition.InitializeCandidateAppend();
	if (collect_metrics) {
		const auto after_work = std::chrono::steady_clock::now();
		RecordSinkMetrics(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
		    row_count);
	}
}

void RecursiveCTEState::RegisterLocalPreaggregation(vector<RecursiveCTELocalPreaggregate> local_preaggregates,
                                                    idx_t classification_work_ns, idx_t preaggregation_work_ns) {
	D_ASSERT(local_preaggregates.size() == keyed_partitions.size());
	idx_t candidate_rows = 0;
	idx_t group_count = 0;
	for (idx_t partition_idx = 0; partition_idx < local_preaggregates.size(); partition_idx++) {
		auto &local_preaggregate = local_preaggregates[partition_idx];
		if (!local_preaggregate.ht) {
			continue;
		}
		D_ASSERT(local_preaggregate.candidate_rows > 0 &&
		         local_preaggregate.ht->Count() <= local_preaggregate.candidate_rows);
		candidate_rows += local_preaggregate.candidate_rows;
		group_count += local_preaggregate.ht->Count();
		auto &partition = *keyed_partitions[partition_idx];
		lock_guard<mutex> guard(partition.lock);
		partition.preaggregated.push_back(std::move(local_preaggregate));
	}
	if (metrics.Enabled()) {
		metrics.RecordHashRows(candidate_rows);
		GetEpochMetrics().RecordLocalKeyPreaggregationClassification(classification_work_ns);
		GetEpochMetrics().RecordLocalKeyPreaggregation(candidate_rows, group_count, preaggregation_work_ns);
	}
}

idx_t RecursiveCTEState::KeyedPartitionIndex(hash_t hash) const {
	return RadixPartitioning::ApplyMask(hash, keyed_radix_bits);
}

const GroupedAggregateHashTable &RecursiveCTEState::GetKeyedHashTable(idx_t partition_idx) const {
	D_ASSERT(partition_idx < keyed_partitions.size());
	return *keyed_partitions[partition_idx]->ht;
}

const TupleDataLayout &RecursiveCTEState::KeyedLayout() const {
	D_ASSERT(keyed_layout);
	return *keyed_layout;
}

shared_ptr<TupleDataLayout> RecursiveCTEState::KeyedLayoutPtr() const {
	return keyed_layout;
}

idx_t RecursiveCTEState::KeyedGroupCount() const {
	idx_t count = 0;
	for (auto &partition : keyed_partitions) {
		count += partition->ht->Count();
	}
	return count;
}

void RecursiveCTEState::InitializeKeyedScan(RecursiveCTEKeyedScanState &gstate) {
	gstate.partition_scans.clear();
	gstate.partition_scans.reserve(keyed_partitions.size());
	for (auto &partition : keyed_partitions) {
		auto scan = make_uniq<AggregateHTParallelScanState>();
		partition->ht->InitializeParallelScan(*scan);
		gstate.partition_scans.push_back(std::move(scan));
	}
}

bool RecursiveCTEState::ScanKeyedGroups(RecursiveCTEKeyedScanState &gstate, RecursiveCTEKeyedLocalScanState &lstate,
                                        DataChunk &groups) {
	if (lstate.partition_idx == DConstants::INVALID_INDEX) {
		lstate.partition_idx = 0;
		lstate.scan.partition_idx = DConstants::INVALID_INDEX;
	}
	while (lstate.partition_idx < keyed_partitions.size()) {
		auto &ht = *keyed_partitions[lstate.partition_idx]->ht;
		if (ht.ScanGroups(*gstate.partition_scans[lstate.partition_idx], lstate.scan, groups)) {
			return true;
		}
		// The next partition is a different collection, so its local scan state starts over
		lstate.partition_idx++;
		lstate.scan.partition_idx = DConstants::INVALID_INDEX;
	}
	return false;
}

Vector &RecursiveCTEState::ScannedKeyedRowLocations(RecursiveCTEKeyedLocalScanState &lstate) {
	return GroupedAggregateHashTable::ScannedRowLocations(lstate.scan);
}

void RecursiveCTEState::PreGrowKeyedState(idx_t expected_new) {
	if (expected_new == 0) {
		return;
	}
	const auto expected_per_partition = (expected_new + keyed_partitions.size() - 1) / keyed_partitions.size();
	for (auto &partition : keyed_partitions) {
		auto &ht = *partition->ht;
		const auto desired_capacity =
		    GroupedAggregateHashTable::GetCapacityForCount(ht.Count() + expected_per_partition);
		if (desired_capacity > ht.Capacity()) {
			ht.Resize(desired_capacity);
		}
	}
}

void RecursiveCTEState::AssembleStateRows(DataChunk &keys, DataChunk &aggregates, DataChunk &result) const {
	result.Reset();
	for (idx_t key_idx = 0; key_idx < op.distinct_idx.size(); key_idx++) {
		const auto representative_idx = op.key_representative_indices[key_idx];
		auto &source =
		    representative_idx == DConstants::INVALID_INDEX ? keys.data[key_idx] : aggregates.data[representative_idx];
		result.data[op.distinct_idx[key_idx]].Reference(source);
	}
	for (idx_t payload_idx = 0; payload_idx < op.payload_idx.size(); payload_idx++) {
		result.data[op.payload_idx[payload_idx]].Reference(aggregates.data[payload_idx]);
	}
	result.CheckCardinality(keys.size());
}

void RecursiveCTEState::FinalizeStateRows(RowOperationsState &row_state, Vector &addresses, DataChunk &keys,
                                          DataChunk &aggregates, DataChunk &result) {
	FinalizeAggregateRows(row_state, addresses, aggregates, keys.size());
	AssembleStateRows(keys, aggregates, result);
}

void RecursiveCTEState::FinalizeAggregateRows(RowOperationsState &row_state, Vector &addresses, DataChunk &aggregates,
                                              idx_t count) {
	aggregates.Reset();
	aggregates.SetChildCardinality(count);
	auto &layout = *keyed_layout;
	if (!finalize_requires_lock) {
		// Read-only finalizes use caller-local scratch, so concurrent readers need no exclusion
		RowOperations::FinalizeStates(row_state, layout, addresses, aggregates, 0);
		return;
	}
	lock_guard<mutex> guard(ht_finalize_lock);
	RowOperations::FinalizeStates(row_state, layout, addresses, aggregates, 0);
}

void RecursiveCTEState::ExtractUsingKeyKeys(RecursiveCTEKeyedPartition &partition, DataChunk &input) {
	auto &distinct_rows = partition.distinct_rows;
	distinct_rows.Reset();
	if (op.key_normalizers.empty()) {
		for (idx_t key_idx = 0; key_idx < op.distinct_idx.size(); key_idx++) {
			distinct_rows.data[key_idx].Reference(input.data[op.distinct_idx[key_idx]]);
		}
		distinct_rows.CheckCardinality(input.size());
		return;
	}
	auto &raw_distinct_rows = partition.raw_distinct_rows;
	raw_distinct_rows.Reset();
	for (idx_t key_idx = 0; key_idx < op.distinct_idx.size(); key_idx++) {
		raw_distinct_rows.data[key_idx].Reference(input.data[op.distinct_idx[key_idx]]);
	}
	raw_distinct_rows.CheckCardinality(input.size());
	D_ASSERT(partition.key_executor);
	partition.key_executor->Execute(raw_distinct_rows, distinct_rows);
}

void RecursiveCTEState::InitializeIntermediateAppend() {
	intermediate_table.InitializeAppend(intermediate_append_state);
}

void RecursiveCTEState::InitializeSharedOutputAppend() {
	CurrentOutputTable().InitializeAppend(CurrentOutputAppendState());
}

ColumnDataCollection &RecursiveCTEState::CurrentOutputTable() {
	D_ASSERT(!op.using_key);
	if (!output_is_working) {
		return intermediate_table;
	}
	D_ASSERT(op.working_table);
	return *op.working_table;
}

ColumnDataCollection &RecursiveCTEState::CurrentInputTable() {
	if (op.using_key) {
		D_ASSERT(op.working_table);
		return *op.working_table;
	}
	if (output_is_working) {
		return intermediate_table;
	}
	D_ASSERT(op.working_table);
	return *op.working_table;
}

const ColumnDataCollection &RecursiveCTEState::CurrentInputTable() const {
	if (op.using_key) {
		D_ASSERT(op.working_table);
		return *op.working_table;
	}
	if (output_is_working) {
		return intermediate_table;
	}
	D_ASSERT(op.working_table);
	return *op.working_table;
}

ColumnDataAppendState &RecursiveCTEState::CurrentOutputAppendState() {
	D_ASSERT(!op.using_key);
	if (!output_is_working) {
		return intermediate_append_state;
	}
	return working_append_state;
}

void RecursiveCTEState::AdvanceIterationBuffers() {
	if (!op.using_key) {
		output_is_working = !output_is_working;
	}
}

void RecursiveCTEState::ResetCurrentOutputTableForReuse() {
	D_ASSERT(!op.using_key);
	CurrentOutputTable().ResetForReuse();
}

void RecursiveCTEState::RebindRecursiveScans() {
	if (op.using_key) {
		return;
	}
	auto &input_table = CurrentInputTable();
	for (auto &scan_ref : op.recursive_scans) {
		auto &scan = scan_ref.get();
		scan.collection = input_table;
	}
}

unique_ptr<GlobalSinkState> PhysicalRecursiveCTE::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<RecursiveCTEState>(context, *this);
}

enum class RecursiveCTELocalPreaggregationDecision : uint8_t { DEFER, PREAGGREGATE, DIRECT };

//! Worker-local key extraction: routes candidates to keyed partitions and pre-aggregates fan-in locally.
class RecursiveCTELocalKeyState {
public:
	RecursiveCTELocalKeyState(ClientContext &context_p, const PhysicalRecursiveCTE &op_p)
	    : context(context_p), op(op_p), payload_executor(context), hashes(LogicalType::HASH) {
		vector<LogicalType> aggregate_input_types;
		for (auto &payload_aggregate : op.payload_aggregates) {
			auto &bound_aggregate = payload_aggregate->Cast<BoundAggregateExpression>();
			for (auto &child : bound_aggregate.GetChildren()) {
				payload_executor.AddExpression(*child);
				aggregate_input_types.push_back(child->GetReturnType());
			}
			aggregates.emplace_back(bound_aggregate);
		}
		if (!op.key_normalizers.empty()) {
			key_executor = make_uniq<ExpressionExecutor>(context);
			for (auto &normalizer : op.key_normalizers) {
				key_executor->AddExpression(*normalizer);
			}
			raw_keys.Initialize(Allocator::Get(context), op.distinct_types);
		}
		keys.Initialize(Allocator::Get(context), op.hash_key_types);
		payload.Initialize(Allocator::Get(context), aggregate_input_types);
		input.Initialize(Allocator::Get(context), op.internal_types);
	}

	RecursiveCTELocalPreaggregationDecision Classify(ColumnDataCollection &candidates) {
		const auto candidate_count = candidates.Count();
		D_ASSERT(candidate_count >= STANDARD_VECTOR_SIZE);
		sampled_candidate_count += candidate_count;
		ColumnDataScanState scan_state;
		candidates.InitializeScan(scan_state);
		while (candidates.Scan(scan_state, input)) {
			ExtractKeys(input);
			keys.Hash(hashes);
			cardinality.Update(hashes);
		}
		const auto distinct_upper_bound =
		    LossyNumericCast<idx_t>((1 + HyperLogLog::GetErrorRate()) * static_cast<double>(cardinality.Count()));
		if (distinct_upper_bound < sampled_candidate_count / 4) {
			return RecursiveCTELocalPreaggregationDecision::PREAGGREGATE;
		}
		// Keep sampling across vector boundaries, but bound the extra hashing for unique streams.
		static constexpr idx_t MAX_SAMPLE_VECTORS = 8;
		const auto sample_size = MaxValue<idx_t>(STANDARD_VECTOR_SIZE, 16);
		if (sampled_candidate_count >= sample_size * MAX_SAMPLE_VECTORS) {
			return RecursiveCTELocalPreaggregationDecision::DIRECT;
		}
		return RecursiveCTELocalPreaggregationDecision::DEFER;
	}

	void ResetClassification() {
		cardinality = HyperLogLog();
		sampled_candidate_count = 0;
	}

	//! Appends the chunk to the keyed partitions selected by the hash of its normalized key
	void Route(DataChunk &chunk, RecursiveCTEState &gstate) {
		const auto partition_count = gstate.KeyedPartitionCount();
		D_ASSERT(partition_count > 1);
		ExtractKeys(chunk);
		keys.Hash(hashes);
		PartitionRows(chunk.size(), gstate);
		for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
			const auto partition_size = partition_counts[partition_idx];
			if (partition_size == 0) {
				continue;
			}
			if (partition_size == chunk.size()) {
				gstate.AppendCandidates(partition_idx, chunk);
				return;
			}
			partition_chunk.Reset();
			partition_chunk.Slice(chunk, partition_selections[partition_idx], partition_size);
			gstate.AppendCandidates(partition_idx, partition_chunk);
		}
	}

	void Route(ColumnDataCollection &candidates, RecursiveCTEState &gstate) {
		ColumnDataScanState scan_state;
		candidates.InitializeScan(scan_state);
		while (candidates.Scan(scan_state, input)) {
			Route(input, gstate);
		}
	}

	//! Pre-aggregates the candidates into one local hash table per keyed partition
	vector<RecursiveCTELocalPreaggregate> Preaggregate(ColumnDataCollection &candidates, RecursiveCTEState &gstate) {
		const auto partition_count = gstate.KeyedPartitionCount();
		vector<RecursiveCTELocalPreaggregate> result(partition_count);
		ColumnDataScanState scan_state;
		candidates.InitializeScan(scan_state);
		while (candidates.Scan(scan_state, input)) {
			ExtractKeys(input);
			if (!payload_executor.expressions.empty()) {
				payload.Reset();
				payload_executor.Execute(input, payload);
			}
			if (partition_count == 1) {
				AddPreaggregated(result[0], keys, payload);
				continue;
			}
			keys.Hash(hashes);
			PartitionRows(input.size(), gstate);
			for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
				const auto partition_size = partition_counts[partition_idx];
				if (partition_size == 0) {
					continue;
				}
				if (partition_size == input.size()) {
					AddPreaggregated(result[partition_idx], keys, payload);
					break;
				}
				auto &selection = partition_selections[partition_idx];
				partition_keys.Reset();
				partition_keys.Slice(keys, selection, partition_size);
				partition_payload.Reset();
				partition_payload.Slice(payload, selection, partition_size);
				AddPreaggregated(result[partition_idx], partition_keys, partition_payload);
			}
		}
		return result;
	}

private:
	void ExtractKeys(DataChunk &source) {
		keys.Reset();
		if (!key_executor) {
			for (idx_t key_idx = 0; key_idx < op.distinct_idx.size(); key_idx++) {
				keys.data[key_idx].Reference(source.data[op.distinct_idx[key_idx]]);
			}
			keys.CheckCardinality(source.size());
			return;
		}
		raw_keys.Reset();
		for (idx_t key_idx = 0; key_idx < op.distinct_idx.size(); key_idx++) {
			raw_keys.data[key_idx].Reference(source.data[op.distinct_idx[key_idx]]);
		}
		raw_keys.CheckCardinality(source.size());
		key_executor->Execute(raw_keys, keys);
	}

	void PartitionRows(idx_t row_count, const RecursiveCTEState &gstate) {
		const auto partition_count = gstate.KeyedPartitionCount();
		if (partition_selections.size() != partition_count) {
			partition_selections.clear();
			partition_selections.reserve(partition_count);
			for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
				partition_selections.emplace_back(STANDARD_VECTOR_SIZE);
			}
			partition_counts.resize(partition_count);
			partition_chunk.Initialize(Allocator::Get(context), op.internal_types);
			partition_keys.Initialize(Allocator::Get(context), op.hash_key_types);
			partition_payload.Initialize(Allocator::Get(context), payload.GetTypes());
		}
		std::fill(partition_counts.begin(), partition_counts.end(), 0);
		const auto hash_data = FlatVector::GetData<hash_t>(hashes);
		for (idx_t row_idx = 0; row_idx < row_count; row_idx++) {
			const auto partition_idx = gstate.KeyedPartitionIndex(hash_data[row_idx]);
			partition_selections[partition_idx].set_index(partition_counts[partition_idx]++, row_idx);
		}
	}

	void AddPreaggregated(RecursiveCTELocalPreaggregate &target, DataChunk &group_keys, DataChunk &group_payload) {
		if (!target.ht) {
			target.ht = make_uniq<GroupedAggregateHashTable>(context, BufferAllocator::Get(context), op.hash_key_types,
			                                                 op.aggregate_types, aggregates);
		}
		target.candidate_rows += group_keys.size();
		target.ht->AddChunk(group_keys, group_payload, AggregateType::NON_DISTINCT);
	}

private:
	ClientContext &context;
	const PhysicalRecursiveCTE &op;
	ExpressionExecutor payload_executor;
	unique_ptr<ExpressionExecutor> key_executor;
	vector<AggregateObject> aggregates;
	DataChunk raw_keys;
	DataChunk keys;
	DataChunk payload;
	DataChunk input;
	DataChunk partition_chunk;
	DataChunk partition_keys;
	DataChunk partition_payload;
	Vector hashes;
	vector<SelectionVector> partition_selections;
	vector<idx_t> partition_counts;
	HyperLogLog cardinality;
	idx_t sampled_candidate_count = 0;
};

class RecursiveCTELocalState : public LocalSinkState {
public:
	RecursiveCTELocalState(ClientContext &context, const PhysicalRecursiveCTE &op)
	    : context(context), op(op), hashes(LogicalType::HASH), partition_hashes(LogicalType::HASH),
	      dummy_addresses(LogicalType::POINTER), new_groups(STANDARD_VECTOR_SIZE) {
		if (!op.using_key) {
			output = make_uniq<ColumnDataCollection>(context, op.GetTypes());
			output->InitializeAppend(append_state);
		}
		if (!op.using_key && !op.union_all) {
			partition_chunk.Initialize(Allocator::Get(context), op.GetTypes());
		}
	}

	ClientContext &context;
	const PhysicalRecursiveCTE &op;
	unique_ptr<ColumnDataCollection> output;
	ColumnDataAppendState append_state;
	Vector hashes;
	Vector partition_hashes;
	Vector dummy_addresses;
	SelectionVector new_groups;
	DataChunk partition_chunk;
	vector<SelectionVector> partition_selections;
	vector<idx_t> partition_counts;
	idx_t using_key_candidate_count = 0;
	idx_t using_key_classification_work_ns = 0;
	bool buffer_using_key_output = false;
	bool direct_using_key_output = false;
	unique_ptr<RecursiveCTELocalKeyState> using_key_state;

	RecursiveCTELocalKeyState &GetUsingKeyState() {
		if (!using_key_state) {
			using_key_state = make_uniq<RecursiveCTELocalKeyState>(context, op);
		}
		return *using_key_state;
	}

	//! Publishes candidates to the keyed partitions of the shared state
	void AppendUsingKeyCandidates(DataChunk &chunk, RecursiveCTEState &gstate) {
		D_ASSERT(op.using_key);
		if (gstate.KeyedPartitionCount() == 1) {
			gstate.AppendCandidates(0, chunk);
			return;
		}
		GetUsingKeyState().Route(chunk, gstate);
	}

	void CombineBufferedUsingKeyCandidates(RecursiveCTEState &gstate) {
		D_ASSERT(op.using_key && output);
		if (gstate.KeyedPartitionCount() == 1) {
			gstate.CombineCandidates(0, *output);
		} else {
			GetUsingKeyState().Route(*output, gstate);
		}
		output->ResetForReuse();
		output->InitializeAppend(append_state);
	}

	void SinkUsingKeyOutput(DataChunk &chunk, RecursiveCTEState &gstate) {
		D_ASSERT(op.using_key && !op.union_all);
		using_key_candidate_count += chunk.size();
		if (buffer_using_key_output) {
			BufferUsingKeyOutput(chunk);
			return;
		}
		const auto sample_size = MaxValue<idx_t>(STANDARD_VECTOR_SIZE, 16);
		const auto coalesce_small_chunks = chunk.size() < sample_size && (using_key_candidate_count >= sample_size ||
		                                                                  gstate.CurrentInputCount() >= sample_size);
		if (direct_using_key_output || !gstate.CanPreaggregateUsingKey()) {
			if (coalesce_small_chunks) {
				BufferUsingKeyOutput(chunk);
			} else {
				AppendUsingKeyCandidates(chunk, gstate);
			}
			return;
		}
		if (using_key_candidate_count <= gstate.CurrentInputCount() && !coalesce_small_chunks) {
			AppendUsingKeyCandidates(chunk, gstate);
			return;
		}
		BufferUsingKeyOutput(chunk);
		if (output->Count() < sample_size) {
			return;
		}
		ClassifyBufferedUsingKeyOutput();
		if (!buffer_using_key_output) {
			CombineBufferedUsingKeyCandidates(gstate);
		}
	}

	void ClassifyBufferedUsingKeyOutput() {
		D_ASSERT(output && output->Count() >= STANDARD_VECTOR_SIZE && !buffer_using_key_output &&
		         !direct_using_key_output);
		const auto classification_start = std::chrono::steady_clock::now();
		const auto decision = GetUsingKeyState().Classify(*output);
		const auto classification_end = std::chrono::steady_clock::now();
		using_key_classification_work_ns += NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(classification_end - classification_start).count());
		buffer_using_key_output = decision == RecursiveCTELocalPreaggregationDecision::PREAGGREGATE;
		direct_using_key_output = decision == RecursiveCTELocalPreaggregationDecision::DIRECT;
	}

	void BufferUsingKeyOutput(DataChunk &chunk) {
		D_ASSERT(op.using_key && !op.union_all);
		if (!output) {
			output = make_uniq<ColumnDataCollection>(context, op.internal_types);
			output->InitializeAppend(append_state);
		}
		output->Append(append_state, chunk);
	}

	vector<RecursiveCTELocalPreaggregate> Preaggregate(RecursiveCTEState &gstate, idx_t &preaggregation_work_ns) {
		D_ASSERT(output && op.using_key && !op.union_all);
		D_ASSERT(buffer_using_key_output && using_key_state);
		const auto preaggregation_start = std::chrono::steady_clock::now();
		auto result = using_key_state->Preaggregate(*output, gstate);
		const auto preaggregation_end = std::chrono::steady_clock::now();
		preaggregation_work_ns = NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(preaggregation_end - preaggregation_start).count());
		return result;
	}

	void InitializePartitions(idx_t partition_count) {
		if (partition_selections.size() == partition_count) {
			return;
		}
		partition_selections.clear();
		partition_selections.reserve(partition_count);
		for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
			partition_selections.emplace_back(STANDARD_VECTOR_SIZE);
		}
		partition_counts.resize(partition_count);
	}

	bool SupportsReuse() const override {
		return true;
	}

	void Reset(ExecutionContext &context, GlobalSinkState &gstate) override {
		using_key_candidate_count = 0;
		using_key_classification_work_ns = 0;
		buffer_using_key_output = false;
		direct_using_key_output = false;
		if (using_key_state) {
			using_key_state->ResetClassification();
		}
		if (!output) {
			return;
		}
		auto &recursive_state = gstate.Cast<RecursiveCTEState>();
		if (recursive_state.GetOperator().union_all && !recursive_state.UsesLocalUnionAllOutput()) {
			return;
		}
		output->ResetForReuse();
		output->InitializeAppend(append_state);
	}
};

unique_ptr<LocalSinkState> PhysicalRecursiveCTE::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<RecursiveCTELocalState>(context.client, *this);
}

void RecursiveCTEState::SinkSerialDistinct(DataChunk &chunk, RecursiveCTELocalState &lstate) {
	D_ASSERT(distinct_ht);
	const auto collect_metrics = metrics.Enabled();
	const auto candidate_count = chunk.size();
	const auto before_lock =
	    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	idx_t new_group_count;
	{
		lock_guard<mutex> guard(intermediate_table_lock);
		const auto after_lock =
		    collect_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		if (collect_metrics) {
			metrics.RecordHashRows(candidate_count);
		}
		new_group_count = distinct_ht->FindOrCreateGroups(chunk, lstate.dummy_addresses, lstate.new_groups);
		chunk.Slice(lstate.new_groups, new_group_count);
		if (collect_metrics) {
			const auto after_work = std::chrono::steady_clock::now();
			GetEpochMetrics().RecordDistinctGrouping(
			    candidate_count, new_group_count,
			    NumericCast<idx_t>(
			        std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()));
			RecordSinkMetrics(
			    NumericCast<idx_t>(
			        std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
			    NumericCast<idx_t>(
			        std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
			    candidate_count);
		}
	}
	if (new_group_count > 0) {
		lstate.output->Append(lstate.append_state, chunk);
	}
}

void RecursiveCTEState::SinkDistinct(DataChunk &chunk, RecursiveCTELocalState &lstate, bool emit_rows,
                                     bool record_sink_metrics) {
	auto &partitions = distinct_partitions;
	D_ASSERT(!partitions.empty());
	D_ASSERT((partitions.size() & (partitions.size() - 1)) == 0);
	lstate.InitializePartitions(partitions.size());
	std::fill(lstate.partition_counts.begin(), lstate.partition_counts.end(), 0);

	chunk.Hash(lstate.hashes);
	auto hash_data = FlatVector::GetData<hash_t>(lstate.hashes);
	for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
		const auto partition_idx = RadixPartitioning::ApplyMask(hash_data[row_idx], distinct_radix_bits);
		auto &partition_count = lstate.partition_counts[partition_idx];
		lstate.partition_selections[partition_idx].set_index(partition_count++, row_idx);
	}

	for (idx_t partition_idx = 0; partition_idx < partitions.size(); partition_idx++) {
		const auto partition_count = lstate.partition_counts[partition_idx];
		if (partition_count == 0) {
			continue;
		}
		lstate.partition_chunk.Reset();
		lstate.partition_chunk.Slice(chunk, lstate.partition_selections[partition_idx], partition_count);
		lstate.partition_hashes.Slice(lstate.hashes, lstate.partition_selections[partition_idx], partition_count);
		auto &partition = *partitions[partition_idx];
		const auto collect_sink_metrics = metrics.Enabled() && record_sink_metrics;
		const auto before_lock =
		    collect_sink_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		idx_t new_group_count;
		{
			lock_guard<mutex> guard(partition.lock);
			const auto after_lock =
			    collect_sink_metrics ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
			if (metrics.Enabled()) {
				metrics.RecordHashRows(partition_count);
			}
			new_group_count = partition.ht.FindOrCreateGroups(lstate.partition_chunk, lstate.partition_hashes,
			                                                  lstate.dummy_addresses, lstate.new_groups);
			lstate.partition_chunk.Slice(lstate.new_groups, new_group_count);
			if (collect_sink_metrics) {
				const auto after_work = std::chrono::steady_clock::now();
				GetEpochMetrics().RecordDistinctGrouping(
				    partition_count, new_group_count,
				    NumericCast<idx_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()));
				RecordSinkMetrics(
				    NumericCast<idx_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(after_lock - before_lock).count()),
				    NumericCast<idx_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(after_work - after_lock).count()),
				    partition_count);
			}
		}
		if (emit_rows && new_group_count > 0) {
			lstate.output->Append(lstate.append_state, lstate.partition_chunk);
		}
	}
}

void RecursiveCTEState::PromoteDistinctState(ClientContext &context, idx_t partition_count) {
	D_ASSERT(!op.using_key && !op.union_all);
	if (!distinct_partitions.empty() || partition_count <= 1) {
		return;
	}
	D_ASSERT(distinct_ht);
	const auto migrated_rows = distinct_ht->Count();
	const auto promotion_start =
	    metrics.Enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	distinct_radix_bits = RadixPartitioning::RadixBitsOfPowerOfTwo(partition_count);
	distinct_partitions.reserve(partition_count);
	for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
		distinct_partitions.push_back(make_uniq<RecursiveCTEDistinctPartition>(context, op.distinct_types));
	}

	RecursiveCTELocalState migration_state(context, op);
	DataChunk groups;
	groups.Initialize(Allocator::Get(context), op.distinct_types);
	AggregateHTScanState scan_state;
	distinct_ht->InitializeScan(scan_state);
	while (distinct_ht->ScanGroups(scan_state, groups)) {
		context.InterruptCheck();
		if (groups.size() > 0) {
			SinkDistinct(groups, migration_state, false, false);
		}
	}
	distinct_ht.reset();
	if (metrics.Enabled()) {
		const auto promotion_end = std::chrono::steady_clock::now();
		const auto elapsed_us = NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(promotion_end - promotion_start).count());
		metrics.LogDistinctPromotion(partition_count, migrated_rows, elapsed_us);
	}
}

bool RecursiveCTEState::ShouldPreaggregateUsingKeyUpdates(RecursiveCTEKeyedPartition &partition,
                                                          idx_t candidate_count) {
	D_ASSERT(can_preaggregate_using_key && candidate_count >= STANDARD_VECTOR_SIZE);
	// The frontier is still intact while partitions commit, so its share bounds the fan-in of this partition
	const auto frontier_rows = (op.working_table->Count() + keyed_partitions.size() - 1) / keyed_partitions.size();
	if (frontier_rows >= candidate_count) {
		return false;
	}
	HyperLogLog key_cardinality;
	const auto group_limit = candidate_count / 4;
	ColumnDataScanState sample_scan_state;
	partition.candidates.InitializeScan(sample_scan_state);
	while (partition.candidates.Scan(sample_scan_state, partition.update_rows)) {
		ExtractUsingKeyKeys(partition, partition.update_rows);
		partition.distinct_rows.Hash(partition.preaggregation_hashes);
		key_cardinality.Update(partition.preaggregation_hashes);
		const auto distinct_upper_bound =
		    LossyNumericCast<idx_t>((1 + HyperLogLog::GetErrorRate()) * static_cast<double>(key_cardinality.Count()));
		if (distinct_upper_bound >= group_limit) {
			return false;
		}
	}
	// A second keyed hash table only pays off when the estimated fan-in is substantial.
	return true;
}

template <bool COLLECT_METRICS>
void RecursiveCTEState::CommitUsingKeyUpdatesInternal(RecursiveCTEKeyedPartition &partition) {
	D_ASSERT(op.using_key);
	auto &candidates = partition.candidates;
	auto &frontier = partition.frontier;
	auto &ht = *partition.ht;
	auto &update_rows = partition.update_rows;
	auto &distinct_rows = partition.distinct_rows;
	auto &payload_rows = partition.payload_rows;
	auto &executor = partition.payload_executor;
	if (!partition.preaggregated.empty()) {
		D_ASSERT(!op.union_all);
		const auto combine_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		auto epoch_ht = std::move(partition.preaggregated[0].ht);
		auto preaggregated_candidate_count = partition.preaggregated[0].candidate_rows;
		for (idx_t local_idx = 1; local_idx < partition.preaggregated.size(); local_idx++) {
			epoch_ht->Combine(*partition.preaggregated[local_idx].ht);
			preaggregated_candidate_count += partition.preaggregated[local_idx].candidate_rows;
		}
		partition.preaggregated.clear();
		D_ASSERT(preaggregated_candidate_count > 0);
		if constexpr (COLLECT_METRICS) {
			const auto combine_end = std::chrono::steady_clock::now();
			GetEpochMetrics().RecordKeyPreaggregationCombine(NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(combine_end - combine_start).count()));
		}

		const auto raw_candidate_count = candidates.Count();
		bool preaggregate_raw_candidates = false;
		if (raw_candidate_count >= STANDARD_VECTOR_SIZE) {
			const auto classification_start =
			    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
			preaggregate_raw_candidates = ShouldPreaggregateUsingKeyUpdates(partition, raw_candidate_count);
			if constexpr (COLLECT_METRICS) {
				const auto classification_end = std::chrono::steady_clock::now();
				GetEpochMetrics().RecordKeyPreaggregationClassification(NumericCast<idx_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(classification_end - classification_start)
				        .count()));
			}
		}
		if (preaggregate_raw_candidates) {
			auto raw_ht = CreateUsingKeyHashTable();
			const auto preaggregation_work_ns = PreaggregateUsingKeyUpdates<COLLECT_METRICS>(partition, *raw_ht);
			if constexpr (COLLECT_METRICS) {
				GetEpochMetrics().RecordKeyPreaggregation(raw_candidate_count, raw_ht->Count(), preaggregation_work_ns);
			}
			const auto raw_combine_start =
			    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
			epoch_ht->Combine(*raw_ht);
			if constexpr (COLLECT_METRICS) {
				const auto raw_combine_end = std::chrono::steady_clock::now();
				GetEpochMetrics().RecordKeyPreaggregationCombine(NumericCast<idx_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(raw_combine_end - raw_combine_start).count()));
			}
			preaggregated_candidate_count += raw_candidate_count;
			candidates.ResetForReuse();
			partition.InitializeCandidateAppend();
		}
		CommitMixedUsingKeyUpdatesInternal<COLLECT_METRICS>(partition, std::move(epoch_ht),
		                                                    preaggregated_candidate_count);
		return;
	}
	const auto candidate_count = candidates.Count();
	bool use_preaggregation = false;
	if (can_preaggregate_using_key && candidate_count >= STANDARD_VECTOR_SIZE) {
		const auto classification_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		use_preaggregation = ShouldPreaggregateUsingKeyUpdates(partition, candidate_count);
		if constexpr (COLLECT_METRICS) {
			const auto classification_end = std::chrono::steady_clock::now();
			GetEpochMetrics().RecordKeyPreaggregationClassification(NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(classification_end - classification_start)
			        .count()));
		}
	}
	if (use_preaggregation) {
		CommitPreaggregatedUsingKeyUpdatesInternal<COLLECT_METRICS>(partition);
		return;
	}
	const auto delta_candidate_count = op.union_all ? idx_t(0) : candidate_count;
	idx_t delta_work_ns = 0;
	if (!op.union_all) {
		D_ASSERT(partition.key_delta);
		const auto delta_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		partition.key_delta->Reset();
		if constexpr (COLLECT_METRICS) {
			const auto delta_end = std::chrono::steady_clock::now();
			delta_work_ns += NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(delta_end - delta_start).count());
		}
	}
	ColumnDataScanState update_scan_state;
	candidates.InitializeScan(update_scan_state);
	while (candidates.Scan(update_scan_state, update_rows)) {
		if constexpr (COLLECT_METRICS) {
			metrics.RecordHashRows(update_rows.size());
		}
		const auto hash_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		idx_t snapshot_work_ns = 0;
		ExtractUsingKeyKeys(partition, update_rows);
		if (!executor.expressions.empty()) {
			payload_rows.Reset();
			executor.Execute(update_rows, payload_rows);
		}
		if (!op.union_all) {
			const auto new_group_count = ht.AddChunk(
			    distinct_rows, payload_rows, AggregateType::NON_DISTINCT,
			    [&](const Vector &group_addresses, const SelectionVector &new_groups, idx_t new_group_count) {
				    const auto delta_start =
				        COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
				    SnapshotUsingKeyDelta(partition, group_addresses, new_groups, new_group_count, update_rows.size());
				    if constexpr (COLLECT_METRICS) {
					    const auto delta_end = std::chrono::steady_clock::now();
					    snapshot_work_ns = NumericCast<idx_t>(
					        std::chrono::duration_cast<std::chrono::nanoseconds>(delta_end - delta_start).count());
					    delta_work_ns += snapshot_work_ns;
				    }
			    });
			if (partition.key_delta->deferred_previous_rows) {
				const auto delta_start =
				    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
				ValidateDeferredUsingKeyCandidateReuse(partition, update_rows);
				if constexpr (COLLECT_METRICS) {
					const auto delta_end = std::chrono::steady_clock::now();
					const auto elapsed_ns = NumericCast<idx_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(delta_end - delta_start).count());
					snapshot_work_ns += elapsed_ns;
					delta_work_ns += elapsed_ns;
				}
			}
			if constexpr (COLLECT_METRICS) {
				const auto hash_end = std::chrono::steady_clock::now();
				const auto hash_work_ns = NumericCast<idx_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - hash_start).count());
				D_ASSERT(snapshot_work_ns <= hash_work_ns);
				GetEpochMetrics().RecordKeyedHashCommit(hash_work_ns - snapshot_work_ns);
			}
			const auto index_start =
			    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
			for (auto &index : partial_key_indexes) {
				index->AddGroups(distinct_rows, partition.new_groups, partition.new_group_addresses,
				                 *FlatVector::IncrementalSelectionVector(), new_group_count);
			}
			if constexpr (COLLECT_METRICS) {
				const auto index_end = std::chrono::steady_clock::now();
				const auto elapsed_ns = NumericCast<idx_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(index_end - index_start).count());
				GetEpochMetrics().RecordPartialIndexMaintenance(elapsed_ns);
				metrics.RecordPartialIndexBuild(NumericCast<idx_t>(elapsed_ns / 1000));
			}
			continue;
		}
		if (partial_key_indexes.empty()) {
			ht.AddChunk(distinct_rows, payload_rows, AggregateType::NON_DISTINCT);
			if constexpr (COLLECT_METRICS) {
				const auto hash_end = std::chrono::steady_clock::now();
				const auto hash_work_ns = NumericCast<idx_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - hash_start).count());
				D_ASSERT(snapshot_work_ns <= hash_work_ns);
				GetEpochMetrics().RecordKeyedHashCommit(hash_work_ns - snapshot_work_ns);
			}
			continue;
		}
		const auto new_group_count =
		    ht.AddChunkAndGetNewGroups(distinct_rows, payload_rows, AggregateType::NON_DISTINCT,
		                               partition.new_group_addresses, partition.new_groups);
		if constexpr (COLLECT_METRICS) {
			const auto hash_end = std::chrono::steady_clock::now();
			const auto hash_work_ns =
			    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - hash_start).count());
			D_ASSERT(snapshot_work_ns <= hash_work_ns);
			GetEpochMetrics().RecordKeyedHashCommit(hash_work_ns - snapshot_work_ns);
		}
		const auto index_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		for (auto &index : partial_key_indexes) {
			index->AddGroups(distinct_rows, partition.new_groups, partition.new_group_addresses,
			                 *FlatVector::IncrementalSelectionVector(), new_group_count);
		}
		if constexpr (COLLECT_METRICS) {
			const auto index_end = std::chrono::steady_clock::now();
			const auto elapsed_ns = NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(index_end - index_start).count());
			GetEpochMetrics().RecordPartialIndexMaintenance(elapsed_ns);
			metrics.RecordPartialIndexBuild(NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::microseconds>(index_end - index_start).count()));
		}
	}
	if (op.union_all) {
		// Every candidate is a next-epoch row
		frontier.Reset();
		frontier.Combine(candidates);
		partition.InitializeCandidateAppend();
		return;
	}
	auto &delta = *partition.key_delta;
	const auto delta_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	if (can_reuse_new_group_candidates && delta.new_count == delta_candidate_count) {
		frontier.Reset();
		frontier.Combine(candidates);
		partition.InitializeCandidateAppend();
	} else if (TryReuseChangedGroupCandidates(partition, delta_candidate_count)) {
		frontier.Reset();
		frontier.Combine(candidates);
		partition.InitializeCandidateAppend();
	} else {
		frontier.ResetForReuse();
		partition.InitializeFrontierAppend();
		FinalizeUsingKeyDelta(partition, false, COLLECT_METRICS);
		candidates.ResetForReuse();
		partition.InitializeCandidateAppend();
	}
	if constexpr (COLLECT_METRICS) {
		const auto delta_end = std::chrono::steady_clock::now();
		delta_work_ns +=
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(delta_end - delta_start).count());
		GetEpochMetrics().RecordKeyDelta(delta_candidate_count, delta.touched_count, delta.new_count,
		                                 delta.changed_count, delta_work_ns);
	}
}

template <bool COLLECT_METRICS>
void RecursiveCTEState::ApplyPreaggregatedUsingKeyUpdates(RecursiveCTEKeyedPartition &partition,
                                                          GroupedAggregateHashTable &epoch_ht, idx_t &delta_work_ns) {
	auto &distinct_rows = partition.distinct_rows;
	AggregateHTScanState epoch_scan_state;
	epoch_ht.InitializeScan(epoch_scan_state);
	while (epoch_ht.ScanGroups(epoch_scan_state, distinct_rows)) {
		if (distinct_rows.size() == 0) {
			continue;
		}
		const auto snapshot_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		SnapshotPreaggregatedUsingKeyDeltaGroups(partition, distinct_rows);
		if constexpr (COLLECT_METRICS) {
			const auto snapshot_end = std::chrono::steady_clock::now();
			delta_work_ns += NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(snapshot_end - snapshot_start).count());
		}
	}

	const auto combine_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	partition.ht->Combine(epoch_ht);
	if constexpr (COLLECT_METRICS) {
		const auto combine_end = std::chrono::steady_clock::now();
		GetEpochMetrics().RecordKeyPreaggregationCombine(NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(combine_end - combine_start).count()));
	}
}

template <bool COLLECT_METRICS>
void RecursiveCTEState::CommitMixedUsingKeyUpdatesInternal(RecursiveCTEKeyedPartition &partition,
                                                           unique_ptr<GroupedAggregateHashTable> epoch_ht,
                                                           idx_t preaggregated_candidate_count) {
	D_ASSERT(op.using_key && !op.union_all && partition.key_delta && epoch_ht && preaggregated_candidate_count > 0);
	auto &delta = *partition.key_delta;
	auto &candidates = partition.candidates;
	auto &update_rows = partition.update_rows;
	auto &executor = partition.payload_executor;
	const auto raw_candidate_count = candidates.Count();
	const auto delta_candidate_count = raw_candidate_count + preaggregated_candidate_count;
	idx_t delta_work_ns = 0;
	const auto reset_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	delta.Reset();
	if constexpr (COLLECT_METRICS) {
		const auto reset_end = std::chrono::steady_clock::now();
		delta_work_ns +=
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(reset_end - reset_start).count());
	}

	ColumnDataScanState update_scan_state;
	candidates.InitializeScan(update_scan_state);
	while (candidates.Scan(update_scan_state, update_rows)) {
		if constexpr (COLLECT_METRICS) {
			metrics.RecordHashRows(update_rows.size());
		}
		const auto hash_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		idx_t snapshot_work_ns = 0;
		ExtractUsingKeyKeys(partition, update_rows);
		if (!executor.expressions.empty()) {
			partition.payload_rows.Reset();
			executor.Execute(update_rows, partition.payload_rows);
		}
		partition.ht->AddChunk(
		    partition.distinct_rows, partition.payload_rows, AggregateType::NON_DISTINCT,
		    [&](const Vector &group_addresses, const SelectionVector &new_groups, idx_t new_group_count) {
			    const auto snapshot_start =
			        COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
			    SnapshotUsingKeyDelta(partition, group_addresses, new_groups, new_group_count, update_rows.size(),
			                          false);
			    if constexpr (COLLECT_METRICS) {
				    const auto snapshot_end = std::chrono::steady_clock::now();
				    snapshot_work_ns = NumericCast<idx_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(snapshot_end - snapshot_start).count());
				    delta_work_ns += snapshot_work_ns;
			    }
		    });
		if constexpr (COLLECT_METRICS) {
			const auto hash_end = std::chrono::steady_clock::now();
			const auto hash_work_ns =
			    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - hash_start).count());
			D_ASSERT(snapshot_work_ns <= hash_work_ns);
			GetEpochMetrics().RecordKeyedHashCommit(hash_work_ns - snapshot_work_ns);
		}
	}

	ApplyPreaggregatedUsingKeyUpdates<COLLECT_METRICS>(partition, *epoch_ht, delta_work_ns);

	const auto finalize_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	partition.frontier.ResetForReuse();
	partition.InitializeFrontierAppend();
	const auto index_work_ns = FinalizeUsingKeyDelta(partition, !partial_key_indexes.empty(), COLLECT_METRICS);
	candidates.ResetForReuse();
	partition.InitializeCandidateAppend();
	if constexpr (COLLECT_METRICS) {
		const auto finalize_end = std::chrono::steady_clock::now();
		const auto finalize_work_ns = NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(finalize_end - finalize_start).count());
		D_ASSERT(index_work_ns <= finalize_work_ns);
		delta_work_ns += finalize_work_ns - index_work_ns;
		GetEpochMetrics().RecordKeyDelta(delta_candidate_count, delta.touched_count, delta.new_count,
		                                 delta.changed_count, delta_work_ns);
	}
}

template <bool COLLECT_METRICS>
idx_t RecursiveCTEState::PreaggregateUsingKeyUpdates(RecursiveCTEKeyedPartition &partition,
                                                     GroupedAggregateHashTable &epoch_ht) {
	auto &candidates = partition.candidates;
	auto &update_rows = partition.update_rows;
	auto &executor = partition.payload_executor;
	idx_t preaggregation_work_ns = 0;
	ColumnDataScanState update_scan_state;
	candidates.InitializeScan(update_scan_state);
	while (candidates.Scan(update_scan_state, update_rows)) {
		if constexpr (COLLECT_METRICS) {
			metrics.RecordHashRows(update_rows.size());
		}
		const auto hash_start =
		    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
		ExtractUsingKeyKeys(partition, update_rows);
		if (!executor.expressions.empty()) {
			partition.payload_rows.Reset();
			executor.Execute(update_rows, partition.payload_rows);
		}
		epoch_ht.AddChunk(partition.distinct_rows, partition.payload_rows, AggregateType::NON_DISTINCT);
		if constexpr (COLLECT_METRICS) {
			const auto hash_end = std::chrono::steady_clock::now();
			preaggregation_work_ns +=
			    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(hash_end - hash_start).count());
		}
	}
	return preaggregation_work_ns;
}

template <bool COLLECT_METRICS>
void RecursiveCTEState::CommitPreaggregatedUsingKeyUpdatesInternal(RecursiveCTEKeyedPartition &partition) {
	D_ASSERT(op.using_key && !op.union_all && partition.key_delta);
	auto &delta = *partition.key_delta;
	const auto delta_candidate_count = partition.candidates.Count();
	idx_t delta_work_ns = 0;
	const auto delta_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	delta.Reset();
	if constexpr (COLLECT_METRICS) {
		const auto delta_end = std::chrono::steady_clock::now();
		delta_work_ns +=
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(delta_end - delta_start).count());
	}

	auto epoch_ht = CreateUsingKeyHashTable();
	const auto preaggregation_work_ns = PreaggregateUsingKeyUpdates<COLLECT_METRICS>(partition, *epoch_ht);
	if constexpr (COLLECT_METRICS) {
		GetEpochMetrics().RecordKeyPreaggregation(delta_candidate_count, epoch_ht->Count(), preaggregation_work_ns);
	}
	ApplyPreaggregatedUsingKeyUpdates<COLLECT_METRICS>(partition, *epoch_ht, delta_work_ns);

	const auto finalize_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	partition.frontier.ResetForReuse();
	partition.InitializeFrontierAppend();
	const auto index_work_ns = FinalizeUsingKeyDelta(partition, !partial_key_indexes.empty(), COLLECT_METRICS);
	partition.candidates.ResetForReuse();
	partition.InitializeCandidateAppend();
	if constexpr (COLLECT_METRICS) {
		const auto finalize_end = std::chrono::steady_clock::now();
		const auto finalize_work_ns = NumericCast<idx_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(finalize_end - finalize_start).count());
		D_ASSERT(index_work_ns <= finalize_work_ns);
		delta_work_ns += finalize_work_ns - index_work_ns;
		GetEpochMetrics().RecordKeyDelta(delta_candidate_count, delta.touched_count, delta.new_count,
		                                 delta.changed_count, delta_work_ns);
	}
}

idx_t RecursiveCTEState::PrepareKeyedCommit() {
	D_ASSERT(op.using_key);
	keyed_commit_partitions.clear();
	keyed_commit_cursor = 0;
	if (keyed_partitions.size() == 1 && keyed_partition_target > 1) {
		const auto work_rows = keyed_partitions[0]->WorkRows();
		// Before an epoch completes there is nothing to compare the commit against, so a large early commit
		// partitions right away while the state is still cheap to split. Later commits are predicted from the
		// per-row cost of the previous one, so a growing frontier promotes before its big commit runs serially.
		bool promote = work_rows >= KEYED_PROMOTION_ROWS && last_epoch_ns == 0;
		if (work_rows >= KEYED_PROMOTION_ROWS && last_epoch_ns > 0 && last_commit_rows > 0) {
			const auto predicted_commit_ns = static_cast<double>(last_commit_ns) * static_cast<double>(work_rows) /
			                                 static_cast<double>(last_commit_rows);
			promote = predicted_commit_ns * static_cast<double>(KEYED_PROMOTION_COMMIT_SHARE_DIVISOR) >=
			          static_cast<double>(last_epoch_ns);
		}
		if (promote) {
			PromoteKeyedState();
		}
	}
	idx_t work_rows = 0;
	idx_t raw_candidate_rows = 0;
	bool has_preaggregates = false;
	for (idx_t partition_idx = 0; partition_idx < keyed_partitions.size(); partition_idx++) {
		auto &partition = *keyed_partitions[partition_idx];
		if (!partition.HasWork()) {
			continue;
		}
		keyed_commit_partitions.push_back(partition_idx);
		work_rows += partition.WorkRows();
		raw_candidate_rows += partition.candidates.Count();
		has_preaggregates = has_preaggregates || !partition.preaggregated.empty();
	}
	if (has_preaggregates && metrics.Enabled()) {
		// Raw candidates that arrived next to worker pre-aggregates, over every partition
		GetEpochMetrics().RecordLocalKeyPreaggregationResidual(raw_candidate_rows);
	}
	if (keyed_commit_partitions.size() <= 1) {
		return keyed_commit_partitions.size();
	}
	const auto threads = TaskScheduler::GetScheduler(context).NumberOfThreads();
	const auto row_tasks = MaxValue<idx_t>(work_rows / KEYED_COMMIT_ROWS_PER_TASK, 1);
	return MinValue<idx_t>(keyed_commit_partitions.size(), MinValue<idx_t>(threads, row_tasks));
}

idx_t RecursiveCTEState::PreparedKeyedCommitRows() const {
	idx_t rows = 0;
	for (auto partition_idx : keyed_commit_partitions) {
		rows += keyed_partitions[partition_idx]->WorkRows();
	}
	return rows;
}

idx_t RecursiveCTEState::NextKeyedCommitPartition() {
	const auto next = keyed_commit_cursor.fetch_add(1);
	if (next >= keyed_commit_partitions.size()) {
		return DConstants::INVALID_INDEX;
	}
	return keyed_commit_partitions[next];
}

void RecursiveCTEState::CommitKeyedPartition(idx_t partition_idx) {
	auto &partition = *keyed_partitions[partition_idx];
	if (metrics.Enabled()) {
		CommitUsingKeyUpdatesInternal<true>(partition);
	} else {
		CommitUsingKeyUpdatesInternal<false>(partition);
	}
}

void RecursiveCTEState::FinishKeyedCommit() {
	D_ASSERT(op.using_key && keyed_commit_cursor >= keyed_commit_partitions.size());
	if (keyed_commit_partitions.empty()) {
		// No partition produced a frontier
		op.working_table->Reset();
	} else if (keyed_partitions.size() > 1) {
		op.working_table->Reset();
		for (auto &partition : keyed_partitions) {
			op.working_table->Combine(partition->frontier);
			partition->InitializeFrontierAppend();
		}
	}
	if (metrics.Enabled()) {
		metrics.RecordKeyedCommit(keyed_commit_partitions.size());
	}
	keyed_commit_partitions.clear();
}

//! Moves the groups of `source` into the hash tables selected by the radix partition of their hash.
static void SplitKeyedHashTable(ClientContext &context, GroupedAggregateHashTable &source, idx_t radix_bits,
                                const std::function<GroupedAggregateHashTable &(idx_t)> &target) {
	auto layout_ptr = source.GetLayoutPtr();
	auto source_data = source.AcquirePartitionedData();
	if (source_data->Count() == 0) {
		return;
	}
	auto repartitioned = make_uniq<RadixPartitionedTupleData>(BufferManager::GetBufferManager(context), layout_ptr,
	                                                          MemoryTag::HASH_TABLE, radix_bits,
	                                                          layout_ptr->ColumnCount() - 1, QueryContext(context));
	source_data->Repartition(context, *repartitioned);
	auto &partitions = repartitioned->GetPartitions();
	for (idx_t partition_idx = 0; partition_idx < partitions.size(); partition_idx++) {
		auto &partition = *partitions[partition_idx];
		if (partition.Count() == 0) {
			continue;
		}
		auto &target_ht = target(partition_idx);
		target_ht.Combine(partition);
		// Combined states can still point into the arenas of the source
		target_ht.InheritAllocators(source);
	}
}

void RecursiveCTEState::RebuildPartialKeyIndexes() {
	if (partial_key_indexes.empty()) {
		return;
	}
	for (auto &index : partial_key_indexes) {
		index->Clear();
	}
	DataChunk keys;
	keys.Initialize(Allocator::Get(context), op.hash_key_types);
	RecursiveCTEKeyedScanState scan;
	RecursiveCTEKeyedLocalScanState local_scan;
	InitializeKeyedScan(scan);
	while (ScanKeyedGroups(scan, local_scan, keys)) {
		if (keys.size() == 0) {
			continue;
		}
		auto &addresses = ScannedKeyedRowLocations(local_scan);
		for (auto &index : partial_key_indexes) {
			index->AddGroups(keys, *FlatVector::IncrementalSelectionVector(), addresses,
			                 *FlatVector::IncrementalSelectionVector(), keys.size());
		}
	}
}

void RecursiveCTEState::PromoteKeyedState() {
	D_ASSERT(op.using_key && keyed_partitions.size() == 1 && keyed_partition_target > 1);
	const auto promotion_start =
	    metrics.Enabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	auto old_partition = std::move(keyed_partitions[0]);
	keyed_partitions.clear();
	const auto partition_count = keyed_partition_target;
	keyed_radix_bits = RadixPartitioning::RadixBitsOfPowerOfTwo(partition_count);
	keyed_partitions.reserve(partition_count);
	for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
		keyed_partitions.push_back(make_uniq<RecursiveCTEKeyedPartition>(context, op, payload_aggregate_objects, true));
	}
	// A new layout pointer makes direct probes rebind their lookup states
	keyed_layout = keyed_partitions[0]->ht->GetLayoutPtr();

	const auto migrated_rows = old_partition->ht->Count();
	SplitKeyedHashTable(
	    context, *old_partition->ht, keyed_radix_bits,
	    [&](idx_t partition_idx) -> GroupedAggregateHashTable & { return *keyed_partitions[partition_idx]->ht; });
	if (old_partition->candidates.Count() > 0) {
		RecursiveCTELocalKeyState router(context, op);
		router.Route(old_partition->candidates, *this);
	}
	for (auto &local_preaggregate : old_partition->preaggregated) {
		const auto local_groups = local_preaggregate.ht->Count();
		vector<unique_ptr<GroupedAggregateHashTable>> split(partition_count);
		SplitKeyedHashTable(context, *local_preaggregate.ht, keyed_radix_bits,
		                    [&](idx_t partition_idx) -> GroupedAggregateHashTable & {
			                    if (!split[partition_idx]) {
				                    split[partition_idx] = CreateUsingKeyHashTable();
			                    }
			                    return *split[partition_idx];
		                    });
		// The rows behind each group are only known in total, so attribute them by cumulative group share, which
		// keeps the sum over the partitions exact
		idx_t attributed_groups = 0;
		idx_t attributed_rows = 0;
		for (idx_t partition_idx = 0; partition_idx < partition_count; partition_idx++) {
			if (!split[partition_idx]) {
				continue;
			}
			RecursiveCTELocalPreaggregate entry;
			entry.ht = std::move(split[partition_idx]);
			attributed_groups += entry.ht->Count();
			const auto cumulative_rows = local_preaggregate.candidate_rows * attributed_groups / local_groups;
			entry.candidate_rows = cumulative_rows - attributed_rows;
			attributed_rows = cumulative_rows;
			keyed_partitions[partition_idx]->preaggregated.push_back(std::move(entry));
		}
		D_ASSERT(attributed_groups == local_groups && attributed_rows == local_preaggregate.candidate_rows);
	}
	// Row addresses changed, so the indexes over them are rebuilt from the new partitions
	RebuildPartialKeyIndexes();
	old_partition.reset();
	if (metrics.Enabled()) {
		metrics.RecordKeyedPartitions(partition_count);
		const auto promotion_end = std::chrono::steady_clock::now();
		metrics.LogKeyedPromotion(
		    partition_count, migrated_rows,
		    NumericCast<idx_t>(
		        std::chrono::duration_cast<std::chrono::microseconds>(promotion_end - promotion_start).count()));
	}
}

void RecursiveCTEState::CommitKeyedPartitionsInline() {
	while (true) {
		const auto partition_idx = NextKeyedCommitPartition();
		if (partition_idx == DConstants::INVALID_INDEX) {
			break;
		}
		CommitKeyedPartition(partition_idx);
	}
	FinishKeyedCommit();
}

class RecursiveCTEStateScanGlobalState : public GlobalSourceState {
public:
	RecursiveCTEStateScanGlobalState(ClientContext &context, RecursiveCTEState &state) {
		state.InitializeKeyedScan(scan);
		max_threads = MinValue<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads(),
		                              ScanTasksForRows(state.KeyedGroupCount()));
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

	RecursiveCTEKeyedScanState scan;
	idx_t max_threads = 1;
};

class RecursiveCTEStateScanLocalState : public LocalSourceState {
public:
	RecursiveCTEStateScanLocalState(ClientContext &context, const PhysicalRecursiveCTE &op)
	    : arena(Allocator::Get(context)), row_state(arena) {
		distinct_rows.Initialize(Allocator::Get(context), op.hash_key_types);
		aggregate_rows.Initialize(Allocator::Get(context), op.aggregate_types);
	}

	DataChunk distinct_rows;
	DataChunk aggregate_rows;
	RecursiveCTEKeyedLocalScanState scan;
	ArenaAllocator arena;
	RowOperationsState row_state;
};

PhysicalRecursiveCTEStateScan::PhysicalRecursiveCTEStateScan(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                                             idx_t estimated_cardinality, TableIndex cte_index)
    : PhysicalColumnDataScan(physical_plan, std::move(types), PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN,
                             estimated_cardinality, cte_index) {
}

unique_ptr<GlobalSourceState> PhysicalRecursiveCTEStateScan::GetGlobalSourceState(ClientContext &context) const {
	return GetGlobalSourceState(context, OperatorPartitionInfo::NoPartitionInfo());
}

unique_ptr<GlobalSourceState>
PhysicalRecursiveCTEStateScan::GetGlobalSourceState(ClientContext &context,
                                                    const OperatorPartitionInfo &partition_info) const {
	(void)partition_info;
	if (!recursive_cte || !recursive_cte->sink_state) {
		throw InternalException("USING KEY state scan has no recursive state");
	}
	// The state is frozen for the epoch once the scan is scheduled, so its chunk count bounds the scan tasks
	auto &recursive_state = recursive_cte->sink_state->Cast<RecursiveCTEState>();
	return make_uniq<RecursiveCTEStateScanGlobalState>(context, recursive_state);
}

unique_ptr<LocalSourceState> PhysicalRecursiveCTEStateScan::GetLocalSourceState(ExecutionContext &context,
                                                                                GlobalSourceState &gstate) const {
	if (!recursive_cte) {
		throw InternalException("USING KEY state scan is not linked to its recursive CTE");
	}
	return make_uniq<RecursiveCTEStateScanLocalState>(context.client, *recursive_cte);
}

SourceResultType PhysicalRecursiveCTEStateScan::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                                OperatorSourceInput &input) const {
	if (!recursive_cte || !recursive_cte->sink_state) {
		throw InternalException("USING KEY state scan has no recursive state");
	}
	auto &recursive_state = recursive_cte->sink_state->Cast<RecursiveCTEState>();
	if (!recursive_state.GetMetrics().Enabled()) {
		return GetDataFromState(chunk, input, recursive_state);
	}
	const auto scan_start = std::chrono::steady_clock::now();
	const auto result = GetDataFromState(chunk, input, recursive_state);
	const auto scan_end = std::chrono::steady_clock::now();
	recursive_state.GetEpochMetrics().RecordRecurringScan(
	    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(scan_end - scan_start).count()));
	return result;
}

SourceResultType PhysicalRecursiveCTEStateScan::GetDataFromState(DataChunk &chunk, OperatorSourceInput &input,
                                                                 RecursiveCTEState &recursive_state) const {
	auto &gstate = input.global_state.Cast<RecursiveCTEStateScanGlobalState>();
	auto &lstate = input.local_state.Cast<RecursiveCTEStateScanLocalState>();
	while (recursive_state.ScanKeyedGroups(gstate.scan, lstate.scan, lstate.distinct_rows)) {
		if (lstate.distinct_rows.size() == 0) {
			continue;
		}
		// The scan already located every row, finalize straight from its addresses
		recursive_state.FinalizeStateRows(lstate.row_state, RecursiveCTEState::ScannedKeyedRowLocations(lstate.scan),
		                                  lstate.distinct_rows, lstate.aggregate_rows, chunk);
		if (recursive_state.GetMetrics().Enabled()) {
			recursive_state.GetMetrics().RecordRecurringScanRows(chunk.size());
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	return SourceResultType::FINISHED;
}

InsertionOrderPreservingMap<string> PhysicalRecursiveCTEStateScan::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["CTE Index"] = StringUtil::Format("%llu", cte_index.index);
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

SinkResultType PhysicalRecursiveCTE::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<RecursiveCTEState>();

	if (!using_key && union_all) {
		if (!gstate.UsesLocalUnionAllOutput()) {
			gstate.AppendOutput(chunk);
			return SinkResultType::NEED_MORE_INPUT;
		}
		auto &lstate = input.local_state.Cast<RecursiveCTELocalState>();
		D_ASSERT(lstate.output);
		lstate.output->Append(lstate.append_state, chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}
	if (!using_key) {
		auto &lstate = input.local_state.Cast<RecursiveCTELocalState>();
		D_ASSERT(lstate.output);
		if (!gstate.HasDistinctPartitions()) {
			gstate.SinkSerialDistinct(chunk, lstate);
		} else {
			gstate.SinkDistinct(chunk, lstate);
		}
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &lstate = input.local_state.Cast<RecursiveCTELocalState>();
	if (union_all) {
		lstate.AppendUsingKeyCandidates(chunk, gstate);
		return SinkResultType::NEED_MORE_INPUT;
	}
	lstate.SinkUsingKeyOutput(chunk, gstate);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalRecursiveCTE::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<RecursiveCTEState>();
	if (using_key) {
		if (!union_all) {
			auto &lstate = input.local_state.Cast<RecursiveCTELocalState>();
			if (!lstate.buffer_using_key_output && !lstate.direct_using_key_output && lstate.output &&
			    lstate.output->Count() >= STANDARD_VECTOR_SIZE &&
			    lstate.using_key_candidate_count > gstate.CurrentInputCount() && gstate.CanPreaggregateUsingKey()) {
				lstate.ClassifyBufferedUsingKeyOutput();
			}
			if (!lstate.buffer_using_key_output) {
				if (gstate.GetMetrics().Enabled() && lstate.using_key_classification_work_ns > 0) {
					gstate.GetEpochMetrics().RecordLocalKeyPreaggregationClassification(
					    lstate.using_key_classification_work_ns);
				}
				if (lstate.output && lstate.output->Count() > 0) {
					lstate.CombineBufferedUsingKeyCandidates(gstate);
				}
				return SinkCombineResultType::FINISHED;
			}
			D_ASSERT(lstate.output && lstate.output->Count() > 0 && gstate.CanPreaggregateUsingKey());
			idx_t preaggregation_work_ns = 0;
			auto local_preaggregates = lstate.Preaggregate(gstate, preaggregation_work_ns);
			gstate.RegisterLocalPreaggregation(std::move(local_preaggregates), lstate.using_key_classification_work_ns,
			                                   preaggregation_work_ns);
			return SinkCombineResultType::FINISHED;
		}
	} else {
		if (union_all && !gstate.UsesLocalUnionAllOutput()) {
			return SinkCombineResultType::FINISHED;
		}
		auto &lstate = input.local_state.Cast<RecursiveCTELocalState>();
		D_ASSERT(lstate.output);
		gstate.CombineOutput(*lstate.output);
	}
	return SinkCombineResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
RecursiveCTESourceState::RecursiveCTESourceState(ClientContext &context, const PhysicalRecursiveCTE &op_p)
    : op(op_p), thread_count(TaskScheduler::GetScheduler(context).NumberOfThreads()) {
}

idx_t RecursiveCTESourceState::MaxThreads() {
	if (!op.sink_state) {
		return 1;
	}
	auto &state = op.sink_state->Cast<RecursiveCTEState>();
	return MinValue<idx_t>(thread_count, ScanTasksForRows(state.AnchorOutputRows()));
}

RecursiveCTESourceLocalState::RecursiveCTESourceLocalState(ClientContext &context, const PhysicalRecursiveCTE &op)
    : arena(Allocator::Get(context)), row_state(arena) {
	if (op.using_key) {
		distinct_rows.Initialize(Allocator::Get(context), op.hash_key_types);
		aggregate_rows.Initialize(Allocator::Get(context), op.aggregate_types);
	}
}

unique_ptr<GlobalSourceState> PhysicalRecursiveCTE::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<RecursiveCTESourceState>(context, *this);
}

unique_ptr<GlobalSourceState>
PhysicalRecursiveCTE::GetGlobalSourceState(ClientContext &context, const OperatorPartitionInfo &partition_info) const {
	return make_uniq<RecursiveCTESourceState>(context, *this);
}

unique_ptr<LocalSourceState> PhysicalRecursiveCTE::GetLocalSourceState(ExecutionContext &context,
                                                                       GlobalSourceState &gstate) const {
	return make_uniq<RecursiveCTESourceLocalState>(context.client, *this);
}

SourceResultType PhysicalRecursiveCTE::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                       OperatorSourceInput &input) const {
	auto &gstate = sink_state->Cast<RecursiveCTEState>();
	return gstate.GetData(context, chunk, input);
}

idx_t RecursiveCTEState::AnchorOutputRows() const {
	return op.using_key ? KeyedGroupCount() : intermediate_table.Count();
}

SourceResultType RecursiveCTEState::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) {
	auto &lstate = input.local_state.Cast<RecursiveCTESourceLocalState>();
	if (!lstate.counted) {
		lstate.counted = true;
		if (metrics.Enabled()) {
			metrics.RecordSourceTask();
		}
	}
	return op.using_key ? GetUsingKeyData(context, chunk, input) : GetUnionData(context, chunk, input);
}

void RecursiveCTEState::RunUsingKeyRecursion(ExecutionContext &context) {
	D_ASSERT(op.using_key);
	while (true) {
		// The commit of the previous epoch left the next frontier in the working table
		const auto expected_new = op.working_table->Count();
		if (!op.union_all && expected_new == 0) {
			return;
		}
		PreGrowKeyedState(expected_new);

		const auto epoch_start = std::chrono::steady_clock::now();
		op.ExecuteRecursivePipelines(context);
		const auto epoch_end = std::chrono::steady_clock::now();
		RecordKeyedEpochTime(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(epoch_end - epoch_start).count()));
		if (op.working_table->Count() == 0) {
			return;
		}
	}
}

template <bool COLLECT_METRICS>
SourceResultType RecursiveCTEState::DrainUsingKeyState(DataChunk &chunk, RecursiveCTESourceState &gstate,
                                                       RecursiveCTESourceLocalState &lstate) {
	const auto drain_start =
	    COLLECT_METRICS ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
	while (ScanKeyedGroups(gstate.drain_scan, lstate.drain_scan, lstate.distinct_rows)) {
		if (lstate.distinct_rows.size() == 0) {
			continue;
		}
		// Every task finalizes its own rows once, from the addresses the scan located
		FinalizeAggregateRows(lstate.row_state, ScannedKeyedRowLocations(lstate.drain_scan), lstate.aggregate_rows,
		                      lstate.distinct_rows.size());
		AssembleStateRows(lstate.distinct_rows, lstate.aggregate_rows, chunk);
		if constexpr (COLLECT_METRICS) {
			if (!lstate.drained) {
				lstate.drained = true;
				metrics.RecordDrainTask();
			}
			metrics.RecordFinalStateRows(chunk.size());
			const auto drain_end = std::chrono::steady_clock::now();
			GetEpochMetrics().RecordFinalStateDrain(NumericCast<idx_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(drain_end - drain_start).count()));
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
	if constexpr (COLLECT_METRICS) {
		const auto drain_end = std::chrono::steady_clock::now();
		GetEpochMetrics().RecordFinalStateDrain(
		    NumericCast<idx_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(drain_end - drain_start).count()));
	}
	return SourceResultType::FINISHED;
}

SourceResultType RecursiveCTEState::GetUsingKeyData(ExecutionContext &context, DataChunk &chunk,
                                                    OperatorSourceInput &input) {
	D_ASSERT(op.using_key);
	auto &gstate = input.global_state.Cast<RecursiveCTESourceState>();
	auto &lstate = input.local_state.Cast<RecursiveCTESourceLocalState>();
	{
		annotated_unique_lock<annotated_mutex> guard(gstate.lock);
		switch (gstate.phase) {
		case RecursiveCTESourcePhase::INITIAL: {
			// The first task runs the whole recursion; the frozen state is drained by every task afterwards
			gstate.phase = RecursiveCTESourcePhase::RECURSING_KEY;
			gstate.driver_active = true;
			guard.unlock();
			RunUsingKeyRecursion(context);
			guard.lock();
			gstate.driver_active = false;
			InitializeKeyedScan(gstate.drain_scan);
			gstate.phase = RecursiveCTESourcePhase::DRAINING_FINAL_KEY_STATE;
			if (ScanTasksForRows(KeyedGroupCount()) > 1) {
				gstate.UnblockTasks();
			}
			break;
		}
		case RecursiveCTESourcePhase::RECURSING_KEY:
			if (metrics.Enabled()) {
				metrics.RecordBlockedSourceTask();
			}
			return gstate.BlockSource(input.interrupt_state);
		case RecursiveCTESourcePhase::DRAINING_FINAL_KEY_STATE:
			break;
		case RecursiveCTESourcePhase::FINISHED:
			return SourceResultType::FINISHED;
		default:
			throw InternalException("Unsupported recursive CTE key source phase");
		}
	}
	const auto result = metrics.Enabled() ? DrainUsingKeyState<true>(chunk, gstate, lstate)
	                                      : DrainUsingKeyState<false>(chunk, gstate, lstate);
	if (result == SourceResultType::FINISHED) {
		annotated_lock_guard<annotated_mutex> guard(gstate.lock);
		gstate.phase = RecursiveCTESourcePhase::FINISHED;
		gstate.UnblockTasks();
	}
	return result;
}

bool RecursiveCTEState::RunUnionEpoch(ExecutionContext &context) {
	D_ASSERT(!op.using_key);
	// The scanned output becomes the next iteration input
	auto &current_output = CurrentOutputTable();

	// After an iteration, we reset the recurring table
	// and fill it up with the new hash table rows for the next iteration.
	if (op.ref_recurring && current_output.Count() != 0) {
		// we need to populate the recurring table from the intermediate table
		// careful: we can not just use Combine here, because this destroys the intermediate table
		// instead we need to scan and append to create a copy
		// Note: as we are in the "normal" recursion case here, not the USING KEY case,
		// we can just scan the intermediate table directly, instead of going through the HT
		ColumnDataScanState recurring_scan_state;
		current_output.InitializeScan(recurring_scan_state);
		while (current_output.Scan(recurring_scan_state, source_result)) {
			op.recurring_table->Append(recurring_append_state, source_result);
		}
	}

	AdvanceIterationBuffers();
	ResetCurrentOutputTableForReuse();
	RebindRecursiveScans();

	// Pre-grow the dedup HT to avoid costly Resize + ReinsertTuples during the next Sink phase.
	// current_output.Count() is the count of rows output in the previous iteration — an upper bound
	// on the number of new unique rows the next iteration can add (since the recursion is converging).
	if (!op.union_all) {
		const idx_t expected_new = current_output.Count();
		if (expected_new > 0) {
			if (distinct_partitions.empty()) {
				const idx_t desired_capacity =
				    GroupedAggregateHashTable::GetCapacityForCount(distinct_ht->Count() + expected_new);
				if (desired_capacity > distinct_ht->Capacity()) {
					distinct_ht->Resize(desired_capacity);
				}
			} else {
				const auto expected_per_partition =
				    (expected_new + distinct_partitions.size() - 1) / distinct_partitions.size();
				for (auto &partition : distinct_partitions) {
					const auto desired_capacity =
					    GroupedAggregateHashTable::GetCapacityForCount(partition->ht.Count() + expected_per_partition);
					if (desired_capacity > partition->ht.Capacity()) {
						partition->ht.Resize(desired_capacity);
					}
				}
			}
		}
	}

	// now we need to re-execute all of the pipelines that depend on the recursion
	op.ExecuteRecursivePipelines(context);

	// check if we obtained any results
	// if not, we are done
	return CurrentOutputTable().Count() != 0;
}

SourceResultType RecursiveCTEState::GetUnionData(ExecutionContext &context, DataChunk &chunk,
                                                 OperatorSourceInput &input) {
	D_ASSERT(!op.using_key);
	auto &gstate = input.global_state.Cast<RecursiveCTESourceState>();
	auto &lstate = input.local_state.Cast<RecursiveCTESourceLocalState>();
	while (true) {
		annotated_unique_lock<annotated_mutex> guard(gstate.lock);
		if (lstate.holds_chunk) {
			// The chunk handed out last time has been consumed
			lstate.holds_chunk = false;
			gstate.in_flight--;
		}
		switch (gstate.phase) {
		case RecursiveCTESourcePhase::INITIAL:
			CurrentOutputTable().InitializeScan(gstate.output_scan);
			gstate.epoch++;
			gstate.phase = RecursiveCTESourcePhase::SCANNING_UNION;
			break;
		case RecursiveCTESourcePhase::SCANNING_UNION:
			break;
		case RecursiveCTESourcePhase::FINISHED:
			return SourceResultType::FINISHED;
		default:
			throw InternalException("Unsupported recursive CTE union source phase");
		}
		if (!gstate.driver_active) {
			if (lstate.seen_epoch != gstate.epoch) {
				// The output collection changed, so the pinned chunk state of the previous epoch is stale
				lstate.output_scan = ColumnDataLocalScanState();
				lstate.seen_epoch = gstate.epoch;
			}
			// Scanners are counted so the epoch cannot switch under a chunk copy that runs outside the lock
			gstate.scanning++;
			guard.unlock();
			const auto scanned = CurrentOutputTable().Scan(gstate.output_scan, lstate.output_scan, chunk);
			guard.lock();
			gstate.scanning--;
			if (scanned) {
				lstate.holds_chunk = true;
				gstate.in_flight++;
				return SourceResultType::HAVE_MORE_OUTPUT;
			}
			if (gstate.in_flight == 0 && gstate.scanning == 0 && !gstate.driver_active) {
				// The last task out of the epoch runs the next one, without holding the lock, because the recursive
				// pipelines may execute other tasks of this pipeline on this thread
				gstate.driver_active = true;
				guard.unlock();
				const auto has_output = RunUnionEpoch(context);
				guard.lock();
				gstate.driver_active = false;
				if (!has_output) {
					gstate.phase = RecursiveCTESourcePhase::FINISHED;
					gstate.UnblockTasks();
					return SourceResultType::FINISHED;
				}
				CurrentOutputTable().InitializeScan(gstate.output_scan);
				gstate.epoch++;
				// Narrow epochs are consumed by the driver alone; parked tasks stay parked
				if (ScanTasksForRows(CurrentOutputTable().Count()) > 1) {
					gstate.UnblockTasks();
				}
				continue;
			}
		}
		// Another task drives the epoch or still holds one of its chunks: park until the next output is exposed
		if (metrics.Enabled()) {
			metrics.RecordBlockedSourceTask();
		}
		return gstate.BlockSource(input.interrupt_state);
	}
}

vector<const_reference<PhysicalOperator>> PhysicalRecursiveCTE::GetSources() const {
	return {*this};
}

InsertionOrderPreservingMap<string> PhysicalRecursiveCTE::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["CTE Name"] = ctename.GetIdentifierName();
	result["Table Index"] = StringUtil::Format("%llu", table_index.index);
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

} // namespace duckdb
