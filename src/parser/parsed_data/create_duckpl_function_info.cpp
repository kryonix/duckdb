#include "duckdb/parser/parsed_data/create_duckpl_function_info.hpp"

namespace duckdb {

CreateDuckPLFunctionInfo::CreateDuckPLFunctionInfo(CatalogType type) : CreateFunctionInfo(type, INVALID_SCHEMA) {
}

unique_ptr<CreateInfo> CreateDuckPLFunctionInfo::Copy() const {
	auto result = make_uniq<CreateDuckPLFunctionInfo>(type);
	result->definition = definition;
	CopyFunctionProperties(*result);
	return std::move(result);
}

string CreateDuckPLFunctionInfo::ToString() const {
	return sql.empty() ? "CREATE FUNCTION " + name + " ... LANGUAGE duckpl" : sql;
}

} // namespace duckdb
