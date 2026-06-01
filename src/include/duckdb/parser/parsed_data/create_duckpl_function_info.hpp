//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/parser/parsed_data/create_duckpl_function_info.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/parser/duckpl/parse_definition.hpp"
#include "duckdb/parser/parsed_data/create_function_info.hpp"

namespace duckdb {

struct CreateDuckPLFunctionInfo : public CreateFunctionInfo {
	explicit CreateDuckPLFunctionInfo(CatalogType type = CatalogType::SCALAR_FUNCTION_ENTRY);

	shared_ptr<DuckPLFunctionDefinition> definition;

public:
	unique_ptr<CreateInfo> Copy() const override;
	string ToString() const override;
};

} // namespace duckdb
