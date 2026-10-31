#include "duckdb/function/language_macro_function.hpp"

#include "duckdb/main/language_function_binder.hpp"

#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/common/sql_identifier.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/function/language_table_macro_function.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"

namespace duckdb {

LanguageMacroFunction::LanguageMacroFunction(string language, string source)
    : MacroFunction(MacroType::LANGUAGE_MACRO), language(std::move(language)), source(std::move(source)) {
}

LanguageMacroFunction::LanguageMacroFunction() : MacroFunction(MacroType::LANGUAGE_MACRO) {
}

LanguageMacroFunction::LanguageMacroFunction(MacroType type) : MacroFunction(type) {
}

LanguageMacroFunction &LanguageMacroFunction::Get(MacroFunction &member) {
	if (!member.IsLanguageMember()) {
		throw InternalException("Failed to get the language member view - macro type mismatch");
	}
	return reinterpret_cast<LanguageMacroFunction &>(member);
}

const LanguageMacroFunction &LanguageMacroFunction::Get(const MacroFunction &member) {
	if (!member.IsLanguageMember()) {
		throw InternalException("Failed to get the language member view - macro type mismatch");
	}
	return reinterpret_cast<const LanguageMacroFunction &>(member);
}

unique_ptr<MacroFunction> LanguageMacroFunction::Copy() const {
	auto result = make_uniq<LanguageMacroFunction>(language, source);
	result->language_version = language_version;
	CopyProperties(*result);
	return std::move(result);
}

string LanguageMacroFunction::ToSQL() const {
	// MacroFunction::ToSQL renders "(params) [RETURNS type] AS ", splice the LANGUAGE in front of AS
	auto prefix = MacroFunction::ToSQL();
	D_ASSERT(StringUtil::EndsWith(prefix, " AS "));
	prefix.erase(prefix.size() - 4);
	return prefix + " LANGUAGE " + SQLIdentifier::ToString(language) + " AS " + SQLString::ToString(source);
}

//! A complete identity for a frozen result column type: ordinary type equality ignores collations
//! and treats extension metadata as a wildcard, so compare the serialized representation, which
//! captures aliases, collations, modifiers, and nested types exactly
static string ReturnColumnTypeIdentity(const LogicalType &type) {
	Allocator allocator;
	MemoryStream stream(allocator);
	SerializationOptions options;
	options.storage_compatibility = StorageCompatibility::Latest();
	BinarySerializer serializer(stream, options);
	serializer.Begin();
	serializer.WriteProperty(100, "type", type);
	serializer.End();
	return string(char_ptr_cast(stream.GetData()), stream.GetPosition());
}

LanguageFunctionCreateInfo::LanguageFunctionCreateInfo(const QualifiedName &qualified_name, const string &sql,
                                                       vector<unique_ptr<MacroFunction>> &members,
                                                       vector<FunctionDescription> &descriptions)
    : qualified_name(qualified_name), sql(sql), members(members), descriptions(descriptions) {
	// freeze the declared state before the callback sees it
	for (auto &member : members) {
		FrozenMemberState state;
		auto &language_member = LanguageMacroFunction::Get(*member);
		state.language_version = language_member.language_version;
		state.relation_parameters = member->relation_parameters;
		if (member->type == MacroType::LANGUAGE_TABLE_MACRO) {
			auto &table_member = member->Cast<LanguageTableMacroFunction>();
			state.return_names = table_member.return_names;
			for (auto &type : table_member.return_types) {
				state.return_type_identities.push_back(ReturnColumnTypeIdentity(type));
			}
		}
		frozen_members.push_back(std::move(state));
	}
}

const string &LanguageFunctionCreateInfo::MemberLanguage(idx_t member_idx) const {
	return LanguageMacroFunction::Get(*members[member_idx]).language;
}

const string &LanguageFunctionCreateInfo::MemberSource(idx_t member_idx) const {
	return LanguageMacroFunction::Get(*members[member_idx]).source;
}

bool LanguageFunctionCreateInfo::MemberIsTableFunction(idx_t member_idx) const {
	return members[member_idx]->type == MacroType::LANGUAGE_TABLE_MACRO;
}

idx_t LanguageFunctionCreateInfo::MemberParameterCount(idx_t member_idx) const {
	return members[member_idx]->parameters.size();
}

string LanguageFunctionCreateInfo::MemberParameterName(idx_t member_idx, idx_t param_idx) const {
	return members[member_idx]->parameters[param_idx]->Cast<ColumnRefExpression>().GetColumnName().GetIdentifierName();
}

LogicalType LanguageFunctionCreateInfo::MemberParameterType(idx_t member_idx, idx_t param_idx) const {
	auto &member = *members[member_idx];
	if (param_idx >= member.types.size()) {
		return LogicalType::UNKNOWN;
	}
	return member.types[param_idx];
}

LogicalType LanguageFunctionCreateInfo::MemberReturnType(idx_t member_idx) const {
	return members[member_idx]->return_type;
}

idx_t LanguageFunctionCreateInfo::MemberReturnColumnCount(idx_t member_idx) const {
	auto &member = *members[member_idx];
	if (member.type != MacroType::LANGUAGE_TABLE_MACRO) {
		return 0;
	}
	return member.Cast<LanguageTableMacroFunction>().return_names.size();
}

const Identifier &LanguageFunctionCreateInfo::MemberReturnColumnName(idx_t member_idx, idx_t col_idx) const {
	return members[member_idx]->Cast<LanguageTableMacroFunction>().return_names[col_idx];
}

LogicalType LanguageFunctionCreateInfo::MemberReturnColumnType(idx_t member_idx, idx_t col_idx) const {
	return members[member_idx]->Cast<LanguageTableMacroFunction>().return_types[col_idx];
}

const string &LanguageFunctionCreateInfo::MemberLanguageVersion(idx_t member_idx) const {
	return LanguageMacroFunction::Get(*members[member_idx]).language_version;
}

void LanguageFunctionCreateInfo::SetMemberSource(idx_t member_idx, string source) {
	LanguageMacroFunction::Get(*members[member_idx]).source = std::move(source);
}

void LanguageFunctionCreateInfo::PublishMemberSchema(idx_t member_idx, vector<Identifier> names,
                                                     vector<LogicalType> types) {
	auto &member = *members[member_idx];
	auto &language = LanguageMacroFunction::Get(member).language;
	if (member.type != MacroType::LANGUAGE_TABLE_MACRO) {
		throw InternalException("The language binder for \"%s\" published a result schema for %s(), which is not a "
		                        "table function member",
		                        language, GetFunctionName());
	}
	auto &table_member = member.Cast<LanguageTableMacroFunction>();
	if (!table_member.return_names.empty()) {
		// publish-once: only an unresolved schema may be set, and never changed
		throw InternalException("The language binder for \"%s\" tried to change the result schema of %s() during "
		                        "validation, which is not allowed",
		                        language, GetFunctionName());
	}
	if (names.empty() || names.size() != types.size()) {
		throw InternalException("The language binder for \"%s\" published a result schema for %s() with %llu column "
		                        "names and %llu column types, which is not allowed",
		                        language, GetFunctionName(), names.size(), types.size());
	}
	// a published schema must be as valid as a declared one
	identifier_set_t published_names;
	for (auto &name : names) {
		if (!published_names.insert(name).second) {
			throw InternalException("The language binder for \"%s\" published a result schema for %s() with a "
			                        "duplicate column name %s, which is not allowed",
			                        language, GetFunctionName(), name);
		}
	}
	for (auto &type : types) {
		// recursively: an unresolved type may hide below a concrete container
		const auto unresolved = [](const LogicalType &child) {
			return child.id() == LogicalTypeId::UNKNOWN || child.id() == LogicalTypeId::UNBOUND ||
			       child.id() == LogicalTypeId::INVALID;
		};
		if (TypeVisitor::Contains(type, unresolved)) {
			throw InternalException("The language binder for \"%s\" published a result schema for %s() with an "
			                        "unresolved column type, which is not allowed",
			                        language, GetFunctionName());
		}
	}
	// freeze the published schema, like a declared one: the callback may retain references into the
	// adopted vectors and into the shared type metadata
	auto &state = frozen_members[member_idx];
	D_ASSERT(state.return_names.empty() && state.return_type_identities.empty());
	state.return_names = names;
	for (auto &type : types) {
		state.return_type_identities.push_back(ReturnColumnTypeIdentity(type));
	}
	table_member.return_names = std::move(names);
	table_member.return_types = std::move(types);
}

unique_ptr<ParsedExpression> LanguageFunctionCreateInfo::MemberParameterDefault(idx_t member_idx,
                                                                                idx_t param_idx) const {
	auto &member = *members[member_idx];
	const auto &param_name = member.parameters[param_idx]->Cast<ColumnRefExpression>().GetColumnName();
	auto entry = member.default_parameters.find(param_name);
	if (entry == member.default_parameters.end()) {
		return nullptr;
	}
	return entry->second->Copy();
}

const vector<idx_t> &LanguageFunctionCreateInfo::MemberRelationParameters(idx_t member_idx) const {
	return members[member_idx]->relation_parameters;
}

void LanguageFunctionCreateInfo::PublishMemberRelationParameters(idx_t member_idx, vector<idx_t> relation_parameters) {
	auto &member = *members[member_idx];
	auto &language = LanguageMacroFunction::Get(member).language;
	if (!member.relation_parameters.empty()) {
		// publish-once: only unset markers may be published, and never changed
		throw InternalException("The language binder for \"%s\" tried to change the relation parameters of %s() "
		                        "during validation, which is not allowed",
		                        language, GetFunctionName());
	}
	for (idx_t marker_idx = 0; marker_idx < relation_parameters.size(); marker_idx++) {
		const auto param_idx = relation_parameters[marker_idx];
		if (param_idx >= member.parameters.size() ||
		    (marker_idx > 0 && relation_parameters[marker_idx - 1] >= param_idx)) {
			throw InternalException("The language binder for \"%s\" published relation parameters for %s() that are "
			                        "not strictly increasing declared parameter indexes",
			                        language, GetFunctionName());
		}
	}
	// freeze the published markers: the callback may retain references into the adopted vector
	frozen_members[member_idx].relation_parameters = relation_parameters;
	member.relation_parameters = std::move(relation_parameters);
}

void LanguageFunctionCreateInfo::PublishMemberLanguageVersion(idx_t member_idx, string version) {
	auto &member = LanguageMacroFunction::Get(*members[member_idx]);
	if (!member.language_version.empty()) {
		// publish-once: only an empty version may be set, and never changed
		throw InternalException("The language binder for \"%s\" tried to change the published language version of "
		                        "%s() during validation, which is not allowed",
		                        member.language, GetFunctionName());
	}
	// freeze the published version: the callback may retain a reference into the adopted buffer
	frozen_members[member_idx].language_version = version;
	member.language_version = std::move(version);
}

void LanguageFunctionCreateInfo::VerifyMemberState() const {
	for (idx_t member_idx = 0; member_idx < members.size(); member_idx++) {
		auto &member = *members[member_idx];
		auto &language_member = LanguageMacroFunction::Get(member);
		auto &state = frozen_members[member_idx];
		if (language_member.language_version != state.language_version) {
			throw InternalException("The language binder for \"%s\" changed the language version of %s() during "
			                        "validation, which is not allowed",
			                        language_member.language, GetFunctionName());
		}
		if (member.relation_parameters != state.relation_parameters) {
			throw InternalException("The language binder for \"%s\" changed the relation parameters of %s() during "
			                        "validation, which is not allowed",
			                        language_member.language, GetFunctionName());
		}
		if (member.type != MacroType::LANGUAGE_TABLE_MACRO) {
			continue;
		}
		auto &table_member = member.Cast<LanguageTableMacroFunction>();
		bool changed = state.return_names != table_member.return_names ||
		               state.return_type_identities.size() != table_member.return_types.size();
		for (idx_t col_idx = 0; !changed && col_idx < state.return_type_identities.size(); col_idx++) {
			changed =
			    state.return_type_identities[col_idx] != ReturnColumnTypeIdentity(table_member.return_types[col_idx]);
		}
		if (changed) {
			throw InternalException("The language binder for \"%s\" changed a declared or published result schema of "
			                        "%s() during validation, which is not allowed",
			                        table_member.language, GetFunctionName());
		}
	}
}

} // namespace duckdb
