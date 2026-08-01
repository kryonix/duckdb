#include "duckdb/storage/statistics/numeric_moments.hpp"

#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/uhugeint.hpp"
#include "duckdb/common/types/vector.hpp"

#include <cmath>

namespace duckdb {

NumericMoments::NumericMoments() : count(0), mean(0), m2(0), m3(0), valid(true) {
}

NumericMoments::NumericMoments(idx_t count_p, double mean_p, double m2_p, double m3_p, bool valid_p)
    : count(count_p), mean(mean_p), m2(m2_p), m3(m3_p), valid(valid_p) {
}

bool NumericMoments::TypeIsSupported(const LogicalType &type) {
	return type.IsNumeric();
}

void NumericMoments::UpdateValue(double value) {
	if (!std::isfinite(value)) {
		valid = false;
		return;
	}

	auto previous_count = static_cast<double>(count);
	count++;
	auto current_count = static_cast<double>(count);
	auto delta = value - mean;
	auto delta_n = delta / current_count;
	auto term = delta * delta_n * previous_count;
	m3 += term * delta_n * (current_count - 2.0) - 3.0 * delta_n * m2;
	m2 += term;
	mean += delta_n;
}

struct NumericMomentsUpdate {
	template <class T, class CONVERTER>
	static void Update(const UnifiedVectorFormat &format, idx_t count, NumericMoments &result, CONVERTER converter) {
		auto data = UnifiedVectorFormat::GetData<T>(format);
		for (idx_t i = 0; i < count; i++) {
			auto index = format.sel->get_index(i);
			if (!format.validity.RowIsValid(index)) {
				continue;
			}
			result.UpdateValue(converter(data[index]));
		}
	}

	template <class T>
	static void UpdateScaled(const UnifiedVectorFormat &format, idx_t count, NumericMoments &result,
	                         double scale = 1.0) {
		Update<T>(format, count, result, [scale](T value) { return static_cast<double>(value) / scale; });
	}
};

void NumericMoments::Update(const Vector &input, idx_t update_count) {
	D_ASSERT(TypeIsSupported(input.GetType()));
	if (!valid) {
		return;
	}

	UnifiedVectorFormat format;
	input.ToUnifiedFormat(format);
	double scale = 1.0;
	if (input.GetType().id() == LogicalTypeId::DECIMAL) {
		scale = std::pow(10.0, DecimalType::GetScale(input.GetType()));
	}

	switch (input.GetType().InternalType()) {
	case PhysicalType::INT8:
		NumericMomentsUpdate::UpdateScaled<int8_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::INT16:
		NumericMomentsUpdate::UpdateScaled<int16_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::INT32:
		NumericMomentsUpdate::UpdateScaled<int32_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::INT64:
		NumericMomentsUpdate::UpdateScaled<int64_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::UINT8:
		NumericMomentsUpdate::UpdateScaled<uint8_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::UINT16:
		NumericMomentsUpdate::UpdateScaled<uint16_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::UINT32:
		NumericMomentsUpdate::UpdateScaled<uint32_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::UINT64:
		NumericMomentsUpdate::UpdateScaled<uint64_t>(format, update_count, *this, scale);
		break;
	case PhysicalType::INT128:
		NumericMomentsUpdate::Update<hugeint_t>(
		    format, update_count, *this, [scale](hugeint_t value) { return Hugeint::Cast<double>(value) / scale; });
		break;
	case PhysicalType::UINT128:
		NumericMomentsUpdate::Update<uhugeint_t>(
		    format, update_count, *this, [scale](uhugeint_t value) { return Uhugeint::Cast<double>(value) / scale; });
		break;
	case PhysicalType::FLOAT:
		NumericMomentsUpdate::UpdateScaled<float>(format, update_count, *this);
		break;
	case PhysicalType::DOUBLE:
		NumericMomentsUpdate::UpdateScaled<double>(format, update_count, *this);
		break;
	default:
		throw InternalException("Unsupported physical type for numeric moments");
	}
}

void NumericMoments::Merge(const NumericMoments &other) {
	if (!valid || !other.valid) {
		valid = false;
		return;
	}
	if (other.count == 0) {
		return;
	}
	if (count == 0) {
		count = other.count;
		mean = other.mean;
		m2 = other.m2;
		m3 = other.m3;
		return;
	}

	auto left_count = static_cast<double>(count);
	auto right_count = static_cast<double>(other.count);
	auto total_count = left_count + right_count;
	auto delta = other.mean - mean;
	auto combined_m2 = m2 + other.m2 + delta * delta * left_count * right_count / total_count;
	auto combined_m3 =
	    m3 + other.m3 +
	    delta * delta * delta * left_count * right_count * (left_count - right_count) / (total_count * total_count) +
	    3.0 * delta * (left_count * other.m2 - right_count * m2) / total_count;

	mean += delta * right_count / total_count;
	m2 = combined_m2;
	m3 = combined_m3;
	count += other.count;
}

unique_ptr<NumericMoments> NumericMoments::Copy() const {
	return make_uniq<NumericMoments>(count, mean, m2, m3, valid);
}

bool NumericMoments::IsValid() const {
	return valid;
}

idx_t NumericMoments::Count() const {
	return count;
}

double NumericMoments::Mean() const {
	return mean;
}

double NumericMoments::PopulationVariance() const {
	if (count == 0) {
		return 0;
	}
	return MaxValue<double>(m2 / static_cast<double>(count), 0);
}

double NumericMoments::Skewness() const {
	if (count < 2 || m2 <= 0) {
		return 0;
	}
	return std::sqrt(static_cast<double>(count)) * m3 / std::pow(m2, 1.5);
}

Value NumericMoments::ToStruct() const {
	child_list_t<Value> children;
	children.emplace_back("count", Value::UBIGINT(count));
	children.emplace_back("mean", Value::DOUBLE(mean));
	children.emplace_back("variance", Value::DOUBLE(PopulationVariance()));
	children.emplace_back("skewness", Value::DOUBLE(Skewness()));
	children.emplace_back("valid", Value::BOOLEAN(valid));
	return Value::STRUCT(std::move(children));
}

void NumericMoments::Serialize(Serializer &serializer) const {
	serializer.WriteProperty(100, "count", count);
	serializer.WriteProperty(101, "mean", mean);
	serializer.WriteProperty(102, "m2", m2);
	serializer.WriteProperty(103, "m3", m3);
	serializer.WriteProperty(104, "valid", valid);
}

unique_ptr<NumericMoments> NumericMoments::Deserialize(Deserializer &deserializer) {
	auto count = deserializer.ReadProperty<idx_t>(100, "count");
	auto mean = deserializer.ReadProperty<double>(101, "mean");
	auto m2 = deserializer.ReadProperty<double>(102, "m2");
	auto m3 = deserializer.ReadProperty<double>(103, "m3");
	auto valid = deserializer.ReadProperty<bool>(104, "valid");
	return make_uniq<NumericMoments>(count, mean, m2, m3, valid);
}

} // namespace duckdb
