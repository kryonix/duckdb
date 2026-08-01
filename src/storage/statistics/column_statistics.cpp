#include "duckdb/storage/statistics/column_statistics.hpp"

#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"

namespace duckdb {

ColumnStatistics::ColumnStatistics(BaseStatistics stats_p) : stats(std::move(stats_p)) {
	if (DistinctStatistics::TypeIsSupported(stats.GetType())) {
		distinct_stats = make_uniq<DistinctStatistics>();
	}
}
ColumnStatistics::ColumnStatistics(BaseStatistics stats_p, unique_ptr<DistinctStatistics> distinct_stats_p,
                                   unique_ptr<NumericMoments> numeric_moments_p)
    : stats(std::move(stats_p)), distinct_stats(std::move(distinct_stats_p)),
      numeric_moments(std::move(numeric_moments_p)) {
}

shared_ptr<ColumnStatistics> ColumnStatistics::CreateEmptyStats(const LogicalType &type) {
	return make_shared_ptr<ColumnStatistics>(BaseStatistics::CreateEmpty(type));
}

void ColumnStatistics::Merge(ColumnStatistics &other) {
	stats.Merge(other.stats);
	if (distinct_stats && other.distinct_stats) {
		distinct_stats->Merge(*other.distinct_stats);
	}
	if (numeric_moments && other.numeric_moments) {
		numeric_moments->Merge(*other.numeric_moments);
	} else {
		numeric_moments.reset();
	}
}

BaseStatistics &ColumnStatistics::Statistics() {
	return stats;
}

bool ColumnStatistics::HasDistinctStats() {
	return distinct_stats.get();
}

DistinctStatistics &ColumnStatistics::DistinctStats() {
	if (!distinct_stats) {
		throw InternalException("DistinctStats called without distinct_stats");
	}
	return *distinct_stats;
}

void ColumnStatistics::SetDistinct(unique_ptr<DistinctStatistics> distinct) {
	this->distinct_stats = std::move(distinct);
}

bool ColumnStatistics::HasNumericMoments() const {
	return numeric_moments != nullptr;
}

const NumericMoments &ColumnStatistics::GetNumericMoments() const {
	if (!numeric_moments) {
		throw InternalException("GetNumericMoments called without numeric moments");
	}
	return *numeric_moments;
}

void ColumnStatistics::SetNumericMoments(unique_ptr<NumericMoments> moments) {
	numeric_moments = std::move(moments);
}

void ColumnStatistics::ClearNumericMoments() {
	numeric_moments.reset();
}

void ColumnStatistics::UpdateDistinctStatistics(const Vector &v, idx_t count, Vector &hashes) {
	if (!distinct_stats) {
		return;
	}
	// we use a sample to update the distinct statistics for performance reasons
	distinct_stats->UpdateSample(v, count, hashes);
}

shared_ptr<ColumnStatistics> ColumnStatistics::Copy() const {
	return make_shared_ptr<ColumnStatistics>(stats.Copy(), distinct_stats ? distinct_stats->Copy() : nullptr,
	                                         numeric_moments ? numeric_moments->Copy() : nullptr);
}

void ColumnStatistics::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(100, "statistics", stats);
	serializer.WritePropertyWithDefault(101, "distinct", distinct_stats, unique_ptr<DistinctStatistics>());
	serializer.WritePropertyWithDefault(102, "numeric_moments", numeric_moments, unique_ptr<NumericMoments>());
}

shared_ptr<ColumnStatistics> ColumnStatistics::Deserialize(Deserializer &deserializer) {
	auto stats = deserializer.ReadProperty<BaseStatistics>(100, "statistics");
	auto distinct_stats = deserializer.ReadPropertyWithExplicitDefault<unique_ptr<DistinctStatistics>>(
	    101, "distinct", unique_ptr<DistinctStatistics>());
	auto numeric_moments = deserializer.ReadPropertyWithExplicitDefault<unique_ptr<NumericMoments>>(
	    102, "numeric_moments", unique_ptr<NumericMoments>());
	return make_shared_ptr<ColumnStatistics>(std::move(stats), std::move(distinct_stats), std::move(numeric_moments));
}

} // namespace duckdb
