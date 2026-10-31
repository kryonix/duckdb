//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/language_function_binder.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/identifier.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/parser/parsed_data/create_function_info.hpp"

namespace duckdb {

class Binder;
class ClientContext;
class ExpressionBinder;
class FunctionExpression;
class LanguageMacroFunction;
class LanguageTableMacroFunction;
class MacroFunction;
class ParsedExpression;
class ScalarMacroCatalogEntry;
class SelectStatement;
class TableMacroCatalogEntry;
struct BindResult;

//! The slice of a definition a ValidateCreate callback can act on: it reads the members, may
//! rewrite a member's source, publishes documentation, and publishes, exactly once, what the
//! definition left unresolved (a table member's result schema, a language version). Everything
//! else about the definition (the target, the member count, the resolved signatures, the
//! defaults) is out of reach by type: members are read through scalar accessors, so the
//! definition's structure cannot be changed through the view. Type values share DuckDB's interned
//! type metadata like any LogicalType copy; declared and published result column types are
//! additionally frozen by serialized identity, so an in-place metadata mutation reachable through
//! a returned type value is rejected after the callback.
//! The view borrows from the definition under validation: it is only valid inside the callback.
class LanguageFunctionCreateInfo {
public:
	LanguageFunctionCreateInfo(const Identifier &function_name, vector<unique_ptr<MacroFunction>> &members,
	                           vector<FunctionDescription> &descriptions);

	//! The unqualified function name, for error messages
	const Identifier &GetFunctionName() const {
		return function_name;
	}
	//! The number of members in the definition under validation
	idx_t MemberCount() const {
		return members.size();
	}
	//! One member's language
	const string &MemberLanguage(idx_t member_idx) const;
	//! One member's stored source
	const string &MemberSource(idx_t member_idx) const;
	//! Whether a member is the table-valued variant
	bool MemberIsTableFunction(idx_t member_idx) const;
	//! The number of parameters a member declares
	idx_t MemberParameterCount(idx_t member_idx) const;
	//! A parameter's declared name
	string MemberParameterName(idx_t member_idx, idx_t param_idx) const;
	//! A parameter's resolved type, UNKNOWN for an untyped parameter
	LogicalType MemberParameterType(idx_t member_idx, idx_t param_idx) const;
	//! A member's declared RETURNS type, UNKNOWN when none was declared
	LogicalType MemberReturnType(idx_t member_idx) const;
	//! The number of declared or published RETURNS TABLE columns of a table member, 0 while the
	//! schema is unresolved
	idx_t MemberReturnColumnCount(idx_t member_idx) const;
	//! A declared or published result column's name
	const Identifier &MemberReturnColumnName(idx_t member_idx, idx_t col_idx) const;
	//! A declared or published result column's type
	LogicalType MemberReturnColumnType(idx_t member_idx, idx_t col_idx) const;
	//! A member's published language version, empty while unset
	const string &MemberLanguageVersion(idx_t member_idx) const;
	//! Rewrite one member's stored source
	void SetMemberSource(idx_t member_idx, string source);
	//! Publish the resolved result schema of an unresolved table member, exactly once: legal only
	//! while the stored schema is empty. A published schema is held to the declared-schema rules:
	//! non-empty, equally many names and types, unique names, recursively resolved types
	void PublishMemberSchema(idx_t member_idx, vector<Identifier> names, vector<LogicalType> types);
	//! Publish a member's language version, exactly once: legal only while the stored version is
	//! empty. Informational, surfaced by duckdb_functions, not re-exported by SQL EXPORT
	void PublishMemberLanguageVersion(idx_t member_idx, string version);
	//! Publish documentation. A description must carry a member's parameter types, with ANY for
	//! untyped parameters: documentation in any other shape is not released when its member is
	//! replaced or dropped
	void AddDescription(FunctionDescription description) {
		descriptions.push_back(std::move(description));
	}

	//! Called by the CREATE binder after the callback: result column types share interned type
	//! metadata like any LogicalType copy, so an in-place mutation needs no setter. The serialized
	//! identities captured at construction (declared schemas) and at publication (published
	//! schemas) make the detached-copy convention enforceable
	void VerifyReturnSchemaIdentity() const;

private:
	const Identifier &function_name;
	vector<unique_ptr<MacroFunction>> &members;
	vector<FunctionDescription> &descriptions;
	//! Per member, the serialized identities of its result column types (empty for scalar members)
	vector<vector<string>> return_type_identities;
};

//! Interface an extension registers to bind calls to macro members written in its language. It
//! returns an ordinary bound expression, so a language may translate its functions to SQL or
//! dispatch to its own interpreter. Core stores only the declarative definition.
class LanguageFunctionBinder {
public:
	virtual ~LanguageFunctionBinder() = default;

	//! Validate a definition at CREATE time. Only called when the language is registered,
	//! definitions created or replayed without it are stored unvalidated. The info exposes exactly
	//! what the callback may do: read the members, rewrite member sources, publish documentation,
	//! and publish, once, an unresolved table member's result schema and a language version.
	virtual void ValidateCreate(ClientContext &context, LanguageFunctionCreateInfo &info) = 0;

	//! Bind a call to the already-selected language member in the caller's context. Core has resolved
	//! the overload, named arguments, defaults, and parameter casts: arguments holds one unbound
	//! expression per member parameter, in declaration order. A declared RETURNS type is cast by core.
	virtual BindResult BindScalarCall(ExpressionBinder &binder, ScalarMacroCatalogEntry &entry,
	                                  LanguageMacroFunction &member, vector<unique_ptr<ParsedExpression>> arguments,
	                                  idx_t depth) = 0;

	//! Compile a call to the already-selected table member into a query. Core has resolved the
	//! overload, named arguments, defaults, and declared-type boundary casts: arguments holds one
	//! UNBOUND expression per member parameter, in declaration order, so correlated (lateral)
	//! arguments pass through. The returned statement is bound by core in a child binder of the
	//! caller with correlation enabled, exactly like a table macro expansion, and its result is
	//! cast column-wise to the member's return schema when one is declared. The default raises a
	//! BinderException: a language that registers no table support rejects table calls cleanly.
	virtual unique_ptr<SelectStatement> BindTableCall(Binder &binder, TableMacroCatalogEntry &entry,
	                                                  LanguageTableMacroFunction &member,
	                                                  vector<unique_ptr<ParsedExpression>> arguments);
};

} // namespace duckdb
