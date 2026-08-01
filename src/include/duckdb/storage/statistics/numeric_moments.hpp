//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/storage/statistics/numeric_moments.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {
class Deserializer;
class Serializer;
class Vector;
struct NumericMomentsUpdate;

//! Mergeable moments collected by ANALYZE for numeric columns.
class NumericMoments {
public:
	NumericMoments();
	NumericMoments(idx_t count, double mean, double m2, double m3, bool valid);

public:
	void Update(const Vector &input, idx_t count);
	void Merge(const NumericMoments &other);

	unique_ptr<NumericMoments> Copy() const;

	bool IsValid() const;
	idx_t Count() const;
	double Mean() const;
	double PopulationVariance() const;
	double Skewness() const;

	Value ToStruct() const;

	void Serialize(Serializer &serializer) const;
	static unique_ptr<NumericMoments> Deserialize(Deserializer &deserializer);

	static bool TypeIsSupported(const LogicalType &type);

private:
	friend struct NumericMomentsUpdate;
	void UpdateValue(double value);

private:
	idx_t count;
	double mean;
	double m2;
	double m3;
	bool valid;
};

} // namespace duckdb
