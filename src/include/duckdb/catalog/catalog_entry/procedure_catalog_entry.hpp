//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/catalog/catalog_entry/procedure_catalog_entry.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/catalog/catalog_entry/macro_catalog_entry.hpp"

namespace duckdb {

//! A procedure in the catalog: a macro-family entry whose members expand at a top-level CALL
class ProcedureCatalogEntry : public MacroCatalogEntry {
public:
	static constexpr const CatalogType Type = CatalogType::PROCEDURE_ENTRY;
	static constexpr const char *Name = "procedure";

public:
	ProcedureCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info);

	unique_ptr<CatalogEntry> Copy(ClientContext &context) const override;
};

} // namespace duckdb
