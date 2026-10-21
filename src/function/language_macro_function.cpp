#include "duckdb/function/language_macro_function.hpp"

#include "duckdb/parser/keyword_helper.hpp"

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
	return prefix + " LANGUAGE " + KeywordHelper::WriteOptionallyQuoted(language) + " AS " +
	       KeywordHelper::WriteQuoted(source);
}

} // namespace duckdb
