#include "duckdb/catalog/catalog_entry/scalar_macro_catalog_entry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_macro_catalog_entry.hpp"
#include "duckdb/function/scalar_macro_function.hpp"
#include "duckdb/function/language_macro_function.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

MacroCatalogEntry::MacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : FunctionEntry(
          (info.macros[0]->type == MacroType::TABLE_MACRO || info.macros[0]->type == MacroType::LANGUAGE_TABLE_MACRO
               ? CatalogType::TABLE_MACRO_ENTRY
               : CatalogType::MACRO_ENTRY),
          catalog, schema, info),
      macros(std::move(info.macros)) {
	this->temporary = info.temporary;
	this->internal = info.internal;
	this->extension_name = info.extension_name;
	this->dependencies = info.dependencies;
	this->comment = info.comment;
	this->tags = info.tags;
}

unique_ptr<MacroCatalogEntry> MacroCatalogEntry::Create(Catalog &catalog, SchemaCatalogEntry &schema,
                                                        CreateMacroInfo &info) {
	// every producer funnels through here: binder dispatch and member selection rely on an entry
	// holding members of one kind and, for language members, of one language and the entry's kind,
	// so enforce it at the boundary
	for (auto &member_ptr : info.macros) {
		auto &member = *member_ptr;
		if (!member.IsLanguageMember()) {
			continue;
		}
		const auto member_entry_type =
		    member.type == MacroType::LANGUAGE_TABLE_MACRO ? CatalogType::TABLE_MACRO_ENTRY : CatalogType::MACRO_ENTRY;
		if (member_entry_type != info.type) {
			throw CatalogException("Macro %s cannot combine scalar and table language function members in one entry",
			                       info.GetFunctionName());
		}
	}
	for (idx_t member_idx = 1; member_idx < info.macros.size(); member_idx++) {
		auto &first = *info.macros[0];
		auto &member = *info.macros[member_idx];
		if (first.IsLanguageMember() != member.IsLanguageMember()) {
			throw CatalogException("Macro %s cannot mix language function members and plain macro members",
			                       info.GetFunctionName());
		}
		if (first.IsLanguageMember() && !StringUtil::CIEquals(LanguageMacroFunction::Get(first).language,
		                                                      LanguageMacroFunction::Get(member).language)) {
			throw CatalogException("Macro %s cannot mix members of different languages", info.GetFunctionName());
		}
	}
	if (info.type == CatalogType::TABLE_MACRO_ENTRY) {
		return make_uniq<TableMacroCatalogEntry>(catalog, schema, info);
	}
	return make_uniq<ScalarMacroCatalogEntry>(catalog, schema, info);
}

ScalarMacroCatalogEntry::ScalarMacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : MacroCatalogEntry(catalog, schema, info) {
}

unique_ptr<CatalogEntry> ScalarMacroCatalogEntry::Copy(ClientContext &context) const {
	auto info_copy = GetInfo();
	auto &cast_info = info_copy->Cast<CreateMacroInfo>();
	auto result = make_uniq<ScalarMacroCatalogEntry>(catalog, schema, cast_info);
	return std::move(result);
}

TableMacroCatalogEntry::TableMacroCatalogEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateMacroInfo &info)
    : MacroCatalogEntry(catalog, schema, info) {
}

unique_ptr<CatalogEntry> TableMacroCatalogEntry::Copy(ClientContext &context) const {
	auto info_copy = GetInfo();
	auto &cast_info = info_copy->Cast<CreateMacroInfo>();
	auto result = make_uniq<TableMacroCatalogEntry>(catalog, schema, cast_info);
	return std::move(result);
}

unique_ptr<CreateInfo> MacroCatalogEntry::GetInfo() const {
	auto info = make_uniq<CreateMacroInfo>(type);
	info->SetQualifiedName(schema.GetQualifiedName(name));
	for (auto &function : macros) {
		info->macros.push_back(function->Copy());
	}
	info->extension_name = extension_name;
	info->dependencies = dependencies;
	info->descriptions = descriptions;
	info->comment = comment;
	info->tags = tags;
	return std::move(info);
}

optional_idx MacroCatalogEntry::FindMember(const vector<LogicalType> &signature) const {
	// a fully typed member matching the type list exactly is unique by the create-time conflict rules
	for (idx_t i = 0; i < macros.size(); i++) {
		if (macros[i]->MatchesSignature(signature)) {
			return i;
		}
	}
	// otherwise an untyped slot acts as a wildcard, but a destructive operation must name one
	// member: several compatible candidates are ambiguous
	optional_idx found;
	for (idx_t i = 0; i < macros.size(); i++) {
		auto &member = *macros[i];
		if (member.parameters.size() != signature.size() || !member.HasUntypedParameter()) {
			continue;
		}
		bool compatible = true;
		for (idx_t type_idx = 0; type_idx < signature.size(); type_idx++) {
			if (type_idx < member.types.size() && member.types[type_idx].id() != LogicalTypeId::UNKNOWN &&
			    member.types[type_idx] != signature[type_idx]) {
				compatible = false;
				break;
			}
		}
		if (!compatible) {
			continue;
		}
		if (found.IsValid()) {
			throw CatalogException("The signature is ambiguous: it can name overload %s or %s of function %s",
			                       macros[found.GetIndex()]->SignatureToString(), member.SignatureToString(), name);
		}
		found = i;
	}
	return found;
}

string MacroCatalogEntry::ToSQL() const {
	auto create_info = GetInfo();
	create_info->StripCatalogQualification();
	return create_info->ToString();
}

} // namespace duckdb
