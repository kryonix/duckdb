#include "duckdb/main/settings.hpp"
#include "duckdb/function/partition_stats.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"

#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/join/physical_index_group_join.hpp"
#include "duckdb/execution/operator/join/physical_hash_group_join.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/index/art/art.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/aggregate/distributive_function_utils.hpp"
#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/hash_group_join.hpp"
#include "duckdb/execution/group_join_strategy.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_group_join.hpp"

namespace duckdb {

struct IndexGroupJoinInfo {
	DuckTableEntry *table;
	Identifier index_name;
	vector<StorageIndex> probe_column_ids;
	vector<LogicalType> probe_scan_types;
	vector<idx_t> probe_projection_ids;
	unique_ptr<TableFilterSet> probe_filters;
	vector<unique_ptr<Expression>> probe_residual_filters;
	vector<idx_t> index_key_map;
	vector<LogicalType> index_key_types;
};

static optional<StorageIndex> GetIndexGroupJoinStorageIndex(PhysicalTableScan &scan, DuckTableEntry &table,
                                                            idx_t output_index) {
	if (output_index >= scan.GetTypes().size()) {
		return nullopt;
	}
	const auto scan_index = scan.projection_ids.empty() ? output_index : scan.projection_ids[output_index];
	if (scan_index >= scan.column_ids.size()) {
		return nullopt;
	}
	auto &column = scan.column_ids[scan_index];
	if (!column.HasPrimaryIndex() || column.HasChildren() || column.IsVirtualColumn()) {
		return nullopt;
	}
	return table.GetStorageIndex(column);
}

static optional<IndexGroupJoinInfo> TryGetIndexGroupJoinInfo(ClientContext &context, PhysicalOperator &probe,
                                                             const HashGroupJoinCandidate &candidate) {
	if (candidate.routed) {
		return nullopt;
	}
	vector<unique_ptr<Expression>> residual_filters;
	reference<PhysicalOperator> probe_child(probe);
	while (probe_child.get().type == PhysicalOperatorType::FILTER) {
		auto &filter = probe_child.get().Cast<PhysicalFilter>();
		if (filter.expression->IsVolatile() || filter.children.size() != 1) {
			return nullopt;
		}
		residual_filters.push_back(filter.expression->Copy());
		probe_child = filter.children[0];
	}
	if (probe_child.get().type != PhysicalOperatorType::TABLE_SCAN ||
	    probe_child.get().GetTypes() != probe.GetTypes()) {
		return nullopt;
	}
	auto &scan = probe_child.get().Cast<PhysicalTableScan>();
	if (scan.function.name != "seq_scan" || (scan.dynamic_filters && scan.dynamic_filters->HasFilters())) {
		return nullopt;
	}
	auto &bind_data = scan.bind_data->Cast<TableScanBindData>();
	if (bind_data.partitions_to_scan || bind_data.table.type != CatalogType::TABLE_ENTRY) {
		return nullopt;
	}
	auto &table = bind_data.table.Cast<DuckTableEntry>();
	vector<StorageIndex> probe_column_ids;
	vector<LogicalType> probe_scan_types;
	probe_column_ids.reserve(scan.column_ids.size());
	probe_scan_types.reserve(scan.column_ids.size());
	for (auto &column : scan.column_ids) {
		if (!column.HasPrimaryIndex() || column.HasChildren() || column.IsVirtualColumn()) {
			return nullopt;
		}
		probe_column_ids.push_back(table.GetStorageIndex(column));
		probe_scan_types.push_back(column.HasType() ? column.GetScanType()
		                                            : scan.returned_types[column.GetPrimaryIndex()]);
	}
	vector<idx_t> probe_projection_ids = scan.projection_ids;
	if (probe_projection_ids.empty()) {
		if (scan.GetTypes().size() != scan.column_ids.size()) {
			return nullopt;
		}
		for (idx_t column_idx = 0; column_idx < scan.column_ids.size(); column_idx++) {
			probe_projection_ids.push_back(column_idx);
		}
	}

	vector<idx_t> probe_key_columns;
	probe_key_columns.reserve(candidate.probe_key_indices.size());
	for (auto output_idx : candidate.probe_key_indices) {
		auto storage_index = GetIndexGroupJoinStorageIndex(scan, table, output_idx);
		if (!storage_index || !storage_index->HasPrimaryIndex() || storage_index->HasChildren()) {
			return nullopt;
		}
		probe_key_columns.push_back(storage_index->GetPrimaryIndex());
	}

	auto &info = table.GetStorage().GetDataTableInfo();
	info->BindIndexes(context, ART::TYPE_NAME);
	for (auto &index : info->GetIndexes().Indexes()) {
		if (!index.IsBound() || index.GetIndexType() != ART::TYPE_NAME) {
			continue;
		}
		auto &art = index.Cast<ART>();
		if (art.unbound_expressions.size() != probe_key_columns.size()) {
			continue;
		}
		vector<idx_t> index_key_map;
		vector<bool> used_keys(probe_key_columns.size(), false);
		bool matches = true;
		for (idx_t expression_idx = 0; expression_idx < art.unbound_expressions.size(); expression_idx++) {
			auto &expression = *art.unbound_expressions[expression_idx];
			if (expression.GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				matches = false;
				break;
			}
			auto &column = expression.Cast<BoundColumnRefExpression>();
			auto index_column = column.Binding().column_index;
			if (column.Depth() != 0 || index_column >= art.GetColumnIds().size()) {
				matches = false;
				break;
			}
			auto physical_column = art.GetColumnIds()[index_column];
			auto key_entry = std::find(probe_key_columns.begin(), probe_key_columns.end(), physical_column);
			if (key_entry == probe_key_columns.end()) {
				matches = false;
				break;
			}
			auto key_idx = NumericCast<idx_t>(key_entry - probe_key_columns.begin());
			if (used_keys[key_idx]) {
				matches = false;
				break;
			}
			used_keys[key_idx] = true;
			index_key_map.push_back(key_idx);
		}
		if (matches) {
			return IndexGroupJoinInfo {&table,
			                           art.GetIndexName(),
			                           std::move(probe_column_ids),
			                           std::move(probe_scan_types),
			                           std::move(probe_projection_ids),
			                           scan.table_filters ? scan.table_filters->Copy() : nullptr,
			                           std::move(residual_filters),
			                           std::move(index_key_map),
			                           art.logical_types};
		}
	}
	return nullopt;
}

static void RewriteGroupJoinProbeReferences(unique_ptr<Expression> &expression, idx_t probe_child,
                                            idx_t left_column_count) {
	if (expression->GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto &reference = expression->Cast<BoundReferenceExpression>();
		if (probe_child == 1) {
			if (reference.Index() < left_column_count) {
				throw InternalException(
				    "HASH_GROUP_JOIN aggregate reference %llu unexpectedly referenced the owner child "
				    "(probe child %llu, left columns %llu)",
				    reference.Index(), probe_child, left_column_count);
			}
			reference.IndexMutable() -= left_column_count;
		} else if (reference.Index() >= left_column_count) {
			throw InternalException("HASH_GROUP_JOIN aggregate reference %llu unexpectedly referenced the owner child "
			                        "(probe child %llu, left columns %llu)",
			                        reference.Index(), probe_child, left_column_count);
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(*expression, [&](unique_ptr<Expression> &child) {
		RewriteGroupJoinProbeReferences(child, probe_child, left_column_count);
	});
}

static uint32_t RequiredBitsForValue(uint32_t n) {
	idx_t required_bits = 0;
	while (n > 0) {
		n >>= 1;
		required_bits++;
	}
	return UnsafeNumericCast<uint32_t>(required_bits);
}

template <class T>
hugeint_t GetRangeHugeint(const BaseStatistics &nstats) {
	return Hugeint::Convert(NumericStats::GetMax<T>(nstats)) - Hugeint::Convert(NumericStats::GetMin<T>(nstats));
}

static bool CanUsePartitionedAggregate(ClientContext &context, LogicalAggregate &op, PhysicalOperator &child,
                                       vector<column_t> &partition_columns) {
	if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) {
		return false;
	}
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (aggregate.IsDistinct()) {
			// distinct aggregates are not supported in partitioned hash aggregates
			return false;
		}
	}

	return PhysicalPlanGenerator::HasSingleValuePartitions(context, op.groups, child, partition_columns);
}

bool PhysicalPlanGenerator::HasSingleValuePartitions(ClientContext &context,
                                                     const vector<unique_ptr<Expression>> &partitions,
                                                     PhysicalOperator &child, vector<column_t> &partition_columns) {
	// check if the source is partitioned by the aggregate columns
	// figure out the columns we are grouping by
	for (auto &group_expr : partitions) {
		// only support bound reference here
		if (group_expr->GetExpressionType() != ExpressionType::BOUND_REF) {
			return false;
		}
		auto &ref = group_expr->Cast<BoundReferenceExpression>();
		partition_columns.push_back(ref.Index());
	}
	// traverse the children of the aggregate to find the source operator
	reference<PhysicalOperator> child_ref(child);
	while (child_ref.get().type != PhysicalOperatorType::TABLE_SCAN) {
		auto &child_op = child_ref.get();
		switch (child_op.type) {
		case PhysicalOperatorType::PROJECTION: {
			// recompute partition columns
			auto &projection = child_op.Cast<PhysicalProjection>();
			vector<column_t> new_columns;
			for (auto &partition_col : partition_columns) {
				// we only support bound reference here
				auto &expr = projection.select_list[partition_col];
				if (expr->GetExpressionType() != ExpressionType::BOUND_REF) {
					return false;
				}
				auto &ref = expr->Cast<BoundReferenceExpression>();
				new_columns.push_back(ref.Index());
			}
			// continue into child node with new columns
			partition_columns = std::move(new_columns);
			child_ref = child_op.children[0];
			break;
		}
		case PhysicalOperatorType::FILTER:
			// continue into child operators
			child_ref = child_op.children[0];
			break;
		default:
			// unsupported operator for partition pass-through
			return false;
		}
	}
	auto &table_scan = child_ref.get().Cast<PhysicalTableScan>();
	if (!table_scan.function.get_partition_info) {
		// this source does not expose partition information - skip
		return false;
	}
	// get the base columns by projecting over the projection_ids/column_ids
	if (!table_scan.projection_ids.empty()) {
		for (auto &partition_col : partition_columns) {
			partition_col = table_scan.projection_ids[partition_col];
		}
	}
	vector<column_t> base_columns;
	for (const auto &partition_idx : partition_columns) {
		auto col_idx = partition_idx;
		col_idx = table_scan.column_ids[col_idx].GetPrimaryIndex();
		base_columns.push_back(col_idx);
	}
	// check if the source operator is partitioned by the grouping columns
	TableFunctionPartitionInput input(table_scan.bind_data.get(), base_columns);
	auto partition_info = table_scan.function.get_partition_info(context, input);
	if (partition_info != TablePartitionInfo::SINGLE_VALUE_PARTITIONS) {
		// we only support single-value partitions currently
		return false;
	}
	// we have single value partitions!
	return true;
}

static bool CanUsePerfectHashAggregate(ClientContext &context, LogicalAggregate &op, vector<idx_t> &bits_per_group) {
	if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) {
		return false;
	}
	idx_t perfect_hash_bits = 0;
	for (idx_t group_idx = 0; group_idx < op.groups.size(); group_idx++) {
		auto &group = op.groups[group_idx];
		auto &stats = op.group_stats[group_idx];

		switch (group->GetReturnType().InternalType()) {
		case PhysicalType::INT8:
		case PhysicalType::INT16:
		case PhysicalType::INT32:
		case PhysicalType::INT64:
		case PhysicalType::UINT8:
		case PhysicalType::UINT16:
		case PhysicalType::UINT32:
		case PhysicalType::UINT64:
			break;
		default:
			// we only support simple integer types for perfect hashing
			return false;
		}
		// check if the group has stats available
		auto &group_type = group->GetReturnType();
		if (!stats) {
			// no stats, but we might still be able to use perfect hashing if the type is small enough
			// for small types we can just set the stats to [type_min, type_max]
			switch (group_type.InternalType()) {
			case PhysicalType::INT8:
			case PhysicalType::INT16:
			case PhysicalType::UINT8:
			case PhysicalType::UINT16:
				break;
			default:
				// type is too large and there are no stats: skip perfect hashing
				return false;
			}
			// construct stats with the min and max value of the type
			stats = NumericStats::CreateUnknown(group_type).ToUnique();
			NumericStats::SetMin(*stats, Value::MinimumValue(group_type));
			NumericStats::SetMax(*stats, Value::MaximumValue(group_type));
		}
		auto &nstats = *stats;

		if (!NumericStats::HasMinMax(nstats)) {
			return false;
		}

		if (NumericStats::Max(*stats) < NumericStats::Min(*stats)) {
			// May result in underflow
			return false;
		}

		// we have a min and a max value for the stats: use that to figure out how many bits we have
		// we add two here, one for the NULL value, and one to make the computation one-indexed
		// (e.g. if min and max are the same, we still need one entry in total)
		hugeint_t range_h;
		switch (group_type.InternalType()) {
		case PhysicalType::INT8:
			range_h = GetRangeHugeint<int8_t>(nstats);
			break;
		case PhysicalType::INT16:
			range_h = GetRangeHugeint<int16_t>(nstats);
			break;
		case PhysicalType::INT32:
			range_h = GetRangeHugeint<int32_t>(nstats);
			break;
		case PhysicalType::INT64:
			range_h = GetRangeHugeint<int64_t>(nstats);
			break;
		case PhysicalType::UINT8:
			range_h = GetRangeHugeint<uint8_t>(nstats);
			break;
		case PhysicalType::UINT16:
			range_h = GetRangeHugeint<uint16_t>(nstats);
			break;
		case PhysicalType::UINT32:
			range_h = GetRangeHugeint<uint32_t>(nstats);
			break;
		case PhysicalType::UINT64:
			range_h = GetRangeHugeint<uint64_t>(nstats);
			break;
		default:
			throw InternalException("Unsupported type for perfect hash (should be caught before)");
		}

		uint64_t range;
		if (!Hugeint::TryCast(range_h, range)) {
			return false;
		}

		// bail out on any range bigger than 2^32
		if (range >= NumericLimits<int32_t>::Maximum()) {
			return false;
		}

		range += 2;
		// figure out how many bits we need
		idx_t required_bits = RequiredBitsForValue(UnsafeNumericCast<uint32_t>(range));
		bits_per_group.push_back(required_bits);
		perfect_hash_bits += required_bits;
		// check if we have exceeded the bits for the hash
		if (perfect_hash_bits > Settings::Get<PerfectHtThresholdSetting>(context)) {
			// too many bits for perfect hash
			return false;
		}
	}
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (aggregate.IsDistinct() || !aggregate.Function().HasStateCombineCallback()) {
			// distinct aggregates are not supported in perfect hash aggregates
			return false;
		}
	}
	return true;
}

static HashGroupJoinCandidate GetHashGroupJoinCandidate(LogicalGroupJoin &op) {
	D_ASSERT(op.output_group_sources.size() == op.output_group_indices.size());
	vector<HashGroupJoinOutputColumn> output_groups;
	for (idx_t group_idx = 0; group_idx < op.output_group_sources.size(); group_idx++) {
		output_groups.push_back({op.output_group_sources[group_idx], op.output_group_indices[group_idx]});
	}
	return HashGroupJoinCandidate {op.owner_child,       op.probe_child,           op.owner_key_indices,
	                               op.probe_key_indices, op.owner_payload_indices, std::move(output_groups),
	                               op.unique_owner,      op.unmatched_policy,      op.routed,
	                               op.single_match};
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalGroupJoin &op) {
	D_ASSERT(op.children.size() == 2);
	auto candidate = GetHashGroupJoinCandidate(op);
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (aggregate.GetOrderBys()) {
			FunctionBinder::BindSortedAggregate(context, aggregate, op.groups, op.grouping_sets);
		}
	}
	reference<PhysicalOperator> probe = CreatePlan(*op.children[candidate.probe_child]);
	reference<PhysicalOperator> owner = CreatePlan(*op.children[candidate.owner_child]);

	vector<unique_ptr<Expression>> owner_expressions;
	vector<LogicalType> owner_types;
	vector<unique_ptr<Expression>> probe_expressions;
	vector<LogicalType> probe_types;
	vector<unique_ptr<Expression>> groups;
	for (idx_t key_idx = 0; key_idx < candidate.owner_key_indices.size(); key_idx++) {
		auto owner_idx = candidate.owner_key_indices[key_idx];
		auto probe_idx = candidate.probe_key_indices[key_idx];
		auto type = owner.get().GetTypes()[owner_idx];
		owner_types.push_back(type);
		owner_expressions.push_back(make_uniq<BoundReferenceExpression>(type, owner_idx));
		probe_types.push_back(type);
		probe_expressions.push_back(make_uniq<BoundReferenceExpression>(type, probe_idx));
		groups.push_back(make_uniq<BoundReferenceExpression>(type, key_idx));
	}

	FunctionBinder function_binder(context);
	vector<unique_ptr<Expression>> owner_payload_aggregates;
	for (idx_t payload_idx = 0; payload_idx < candidate.owner_payload_indices.size(); payload_idx++) {
		auto owner_idx = candidate.owner_payload_indices[payload_idx];
		auto type = owner.get().GetTypes()[owner_idx];
		owner_types.push_back(type);
		owner_expressions.push_back(make_uniq<BoundReferenceExpression>(type, owner_idx));
		vector<unique_ptr<Expression>> first_children;
		first_children.push_back(make_uniq<BoundReferenceExpression>(type, payload_idx));
		owner_payload_aggregates.push_back(function_binder.BindAggregateFunction(
		    FirstFunctionGetter::GetFunction(type), std::move(first_children), nullptr, AggregateType::NON_DISTINCT));
	}

	vector<unique_ptr<Expression>> aggregates;
	aggregates.push_back(
	    function_binder.BindAggregateFunction(CountStarFun::GetFunction(), {}, nullptr, AggregateType::NON_DISTINCT));
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		for (auto &child : aggregate.GetChildrenMutable()) {
			RewriteGroupJoinProbeReferences(child, candidate.probe_child, op.left_column_count);
			auto type = child->GetReturnType();
			auto projection_index = probe_expressions.size();
			probe_types.push_back(type);
			probe_expressions.push_back(std::move(child));
			child = make_uniq<BoundReferenceExpression>(type, projection_index);
		}
	}
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (!aggregate.GetFilter()) {
			continue;
		}
		auto &filter = aggregate.GetFilterMutable();
		RewriteGroupJoinProbeReferences(filter, candidate.probe_child, op.left_column_count);
		auto type = filter->GetReturnType();
		auto payload_index = probe_expressions.size() - candidate.probe_key_indices.size();
		probe_types.push_back(type);
		probe_expressions.push_back(std::move(filter));
		filter = make_uniq<BoundReferenceExpression>(type, payload_index);
	}
	for (auto &expression : op.expressions) {
		aggregates.push_back(std::move(expression));
	}

	auto &owner_projection = Make<PhysicalProjection>(std::move(owner_types), std::move(owner_expressions),
	                                                  owner.get().estimated_cardinality);
	owner_projection.children.push_back(owner);
	auto execution_mode = Settings::Get<DebugGroupJoinExecutionSetting>(context);
	if (execution_mode == GroupJoinExecutionMode::INDEX ||
	    (execution_mode == GroupJoinExecutionMode::AUTO && op.use_index)) {
		auto index_info = TryGetIndexGroupJoinInfo(context, probe, candidate);
		if (index_info) {
			auto &group_join = Make<PhysicalIndexGroupJoin>(
			    op, owner_projection, *index_info->table, std::move(index_info->index_name),
			    std::move(index_info->probe_column_ids), std::move(index_info->probe_scan_types),
			    std::move(index_info->probe_projection_ids), probe.get().GetTypes(),
			    std::move(index_info->probe_filters), std::move(index_info->probe_residual_filters),
			    std::move(probe_expressions), std::move(index_info->index_key_map),
			    std::move(index_info->index_key_types), std::move(aggregates), std::move(owner_payload_aggregates),
			    std::move(groups), std::move(candidate.output_groups), candidate.unmatched_policy,
			    op.estimated_cardinality);
			return group_join;
		}
	}
	auto &probe_projection = Make<PhysicalProjection>(std::move(probe_types), std::move(probe_expressions),
	                                                  probe.get().estimated_cardinality);
	probe_projection.children.push_back(probe);
	return Make<PhysicalHashGroupJoin>(op, probe_projection, owner_projection, std::move(aggregates),
	                                   std::move(owner_payload_aggregates), std::move(groups),
	                                   std::move(candidate.output_groups), candidate.unmatched_policy, candidate.routed,
	                                   candidate.unique_owner, candidate.single_match, std::move(op.filter_pushdown),
	                                   op.execution_mode, op.estimated_cardinality);
}

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalAggregate &op) {
	D_ASSERT(op.children.size() == 1);

	reference<PhysicalOperator> plan = CreatePlan(*op.children[0]);
	plan = ExtractAggregateExpressions(plan, op.expressions, op.groups, op.grouping_sets);

	bool can_use_simple_aggregation = true;
	for (auto &expression : op.expressions) {
		auto &aggregate = expression->Cast<BoundAggregateExpression>();
		if (!aggregate.Function().GetStateClusterUpdateCallback()) {
			// unsupported aggregate for simple aggregation: use hash aggregation
			can_use_simple_aggregation = false;
			break;
		}
	}

	// Check if all groups are valid
	if (op.group_stats.empty()) {
		op.group_stats.resize(op.groups.size());
	}
	auto group_validity = TupleDataValidityType::CANNOT_HAVE_NULL_VALUES;
	for (const auto &stats : op.group_stats) {
		if (stats && !stats->CanHaveNull()) {
			continue;
		}
		group_validity = TupleDataValidityType::CAN_HAVE_NULL_VALUES;
		break;
	}

	if (op.groups.empty() && op.grouping_sets.size() <= 1) {
		// no groups: use the dedicated ungrouped aggregate path
		auto &group_by = Make<PhysicalUngroupedAggregate>(op.types, std::move(op.expressions), op.estimated_cardinality,
		                                                  op.distinct_validity);
		group_by.children.push_back(plan);
		return group_by;
	}

	// groups! create a GROUP BY aggregator
	// use a partitioned or perfect hash aggregate if possible
	vector<column_t> partition_columns;
	vector<idx_t> required_bits;
	if (can_use_simple_aggregation && CanUsePartitionedAggregate(context, op, plan, partition_columns)) {
		auto &group_by =
		    Make<PhysicalPartitionedAggregate>(context, op.types, std::move(op.expressions), std::move(op.groups),
		                                       std::move(partition_columns), op.estimated_cardinality);
		group_by.children.push_back(plan);
		return group_by;
	}

	if (CanUsePerfectHashAggregate(context, op, required_bits)) {
		auto &group_by = Make<PhysicalPerfectHashAggregate>(context, op.types, std::move(op.expressions),
		                                                    std::move(op.groups), std::move(op.group_stats),
		                                                    std::move(required_bits), op.estimated_cardinality);
		group_by.children.push_back(plan);
		return group_by;
	}

	auto &group_by = Make<PhysicalHashAggregate>(context, op.types, std::move(op.expressions), std::move(op.groups),
	                                             std::move(op.grouping_sets), std::move(op.grouping_functions),
	                                             op.estimated_cardinality, group_validity, op.distinct_validity);
	group_by.children.push_back(plan);
	return group_by;
}

PhysicalOperator &PhysicalPlanGenerator::ExtractAggregateExpressions(PhysicalOperator &child,
                                                                     vector<unique_ptr<Expression>> &aggregates,
                                                                     vector<unique_ptr<Expression>> &groups,
                                                                     optional_ptr<vector<GroupingSet>> grouping_sets) {
	vector<unique_ptr<Expression>> expressions;
	vector<LogicalType> types;

	// bind sorted aggregates
	for (auto &aggr : aggregates) {
		auto &bound_aggr = aggr->Cast<BoundAggregateExpression>();
		if (bound_aggr.GetOrderBys()) {
			// sorted aggregate!
			FunctionBinder::BindSortedAggregate(context, bound_aggr, groups, grouping_sets);
		}
	}
	for (auto &group : groups) {
		auto ref = make_uniq<BoundReferenceExpression>(group->GetReturnType(), expressions.size());
		types.push_back(group->GetReturnType());
		expressions.push_back(std::move(group));
		group = std::move(ref);
	}
	for (auto &aggr : aggregates) {
		auto &bound_aggr = aggr->Cast<BoundAggregateExpression>();
		for (auto &child_expr : bound_aggr.GetChildrenMutable()) {
			auto ref = make_uniq<BoundReferenceExpression>(child_expr->GetReturnType(), expressions.size());
			types.push_back(child_expr->GetReturnType());
			expressions.push_back(std::move(child_expr));
			child_expr = std::move(ref);
		}
		if (bound_aggr.GetFilter()) {
			auto &filter = bound_aggr.GetFilterMutable();
			auto ref = make_uniq<BoundReferenceExpression>(filter->GetReturnType(), expressions.size());
			types.push_back(filter->GetReturnType());
			expressions.push_back(std::move(filter));
			bound_aggr.GetFilterMutable() = std::move(ref);
		}
	}
	if (expressions.empty()) {
		return child;
	}
	auto &proj = Make<PhysicalProjection>(std::move(types), std::move(expressions), child.estimated_cardinality);
	proj.children.push_back(child);
	return proj;
}

} // namespace duckdb
