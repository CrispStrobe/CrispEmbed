#include "spacy_v2.h"

#include "core/json_value.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "spacy_ggml.h"
#ifdef CRISPEMBED_SPACY_TRF
#include "spacy_transformer.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace crispembed {
namespace spacy {
namespace {

using Json = core_json_value::value;

constexpr std::size_t max_tokens = 2048;
constexpr std::size_t parser_features = 8;
constexpr std::size_t parser_pieces = 2;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::invalid_argument(message);
}

void check_cancel(const CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw NumericCancelled();
}

const Json & member(const Json & object, const char * key) {
    require(object.kind == Json::object_t, std::string("Expected spaCy v2 resource object for ") + key);
    const auto * value = object.get(key);
    require(value != nullptr, std::string("Missing spaCy v2 resource field ") + key);
    return *value;
}

const std::string & text(const Json & value, const char * what) {
    require(value.kind == Json::string_t, std::string("Expected string: ") + what);
    return value.s;
}

std::vector<std::string> strings(const Json & value, const char * what, std::size_t maximum) {
    require(value.kind == Json::array_t && !value.a.empty() && value.a.size() <= maximum,
            std::string("Invalid string list: ") + what);
    std::vector<std::string> result;
    std::set<std::string> seen;
    for (const auto & item : value.a) {
        result.push_back(text(item, what));
        require(seen.insert(result.back()).second, std::string("Duplicate entry in ") + what);
    }
    return result;
}

// ---------------------------------------------------------------- attribute ruler

enum class Field { TAG, POS, LOWER, DEP, IS_SPACE, ORTH };

struct Predicate {
    enum Kind { EQUAL, IN, NOT_IN, REGEX } kind = EQUAL;
    Field field = Field::TAG;
    std::string value;
    bool flag = false;
    std::unordered_set<std::string> set;
    std::shared_ptr<PythonRegex> regex;
};

struct Rule {
    std::vector<std::vector<std::vector<Predicate>>> patterns; // pattern -> token -> predicates
    std::vector<std::pair<std::string, std::string>> attrs;    // POS, TAG, MORPH, LEMMA, DEP
    std::size_t index = 0;
};

Field field_of(const std::string & name) {
    if (name == "TAG") return Field::TAG;
    if (name == "POS") return Field::POS;
    if (name == "LOWER") return Field::LOWER;
    if (name == "DEP") return Field::DEP;
    if (name == "IS_SPACE") return Field::IS_SPACE;
    if (name == "ORTH" || name == "TEXT") return Field::ORTH;
    throw std::invalid_argument("Unsupported attribute-ruler pattern key " + name);
}

std::vector<Rule> parse_rules(const Json & value) {
    require(value.kind == Json::array_t && value.a.size() <= 100000, "Invalid attribute-ruler rules");
    std::vector<Rule> rules;
    for (const auto & entry : value.a) {
        Rule rule;
        const auto & index = member(entry, "index");
        require(index.kind == Json::int_t && index.i >= 0 && index.i < 16, "Unsupported attribute-ruler index");
        rule.index = static_cast<std::size_t>(index.i);
        const auto & attrs = member(entry, "attrs");
        require(attrs.kind == Json::object_t && !attrs.o.empty(), "Invalid attribute-ruler attrs");
        for (const auto & item : attrs.o) {
            require(item.first == "POS" || item.first == "TAG" || item.first == "MORPH" || item.first == "LEMMA" ||
                        item.first == "DEP",
                    "Unsupported attribute-ruler attribute " + item.first);
            rule.attrs.emplace_back(item.first, text(item.second, "attribute value"));
        }
        const auto & patterns = member(entry, "patterns");
        require(patterns.kind == Json::array_t && !patterns.a.empty(), "Invalid attribute-ruler patterns");
        for (const auto & pattern : patterns.a) {
            require(pattern.kind == Json::array_t && !pattern.a.empty() && pattern.a.size() <= 8 &&
                        rule.index < pattern.a.size(),
                    "Invalid attribute-ruler pattern length/index");
            std::vector<std::vector<Predicate>> tokens;
            for (const auto & token : pattern.a) {
                require(token.kind == Json::object_t && !token.o.empty(), "Invalid attribute-ruler token pattern");
                std::vector<Predicate> predicates;
                for (const auto & item : token.o) {
                    require(item.first != "OP", "Attribute-ruler quantifiers are unsupported");
                    Predicate predicate;
                    predicate.field = field_of(item.first);
                    const auto & spec = item.second;
                    if (predicate.field == Field::IS_SPACE) {
                        require(spec.kind == Json::bool_t, "IS_SPACE needs a boolean");
                        predicate.flag = spec.b;
                    } else if (spec.kind == Json::string_t) {
                        predicate.value = spec.s;
                    } else {
                        require(spec.kind == Json::object_t && spec.o.size() == 1, "Unsupported pattern predicate");
                        const auto & op = spec.o[0];
                        if (op.first == "IN" || op.first == "NOT_IN") {
                            predicate.kind = op.first == "IN" ? Predicate::IN : Predicate::NOT_IN;
                            require(op.second.kind == Json::array_t, "IN/NOT_IN needs a list");
                            for (const auto & member_value : op.second.a)
                                predicate.set.insert(text(member_value, "IN/NOT_IN member"));
                        } else if (op.first == "REGEX") {
                            predicate.kind = Predicate::REGEX;
                            predicate.regex = std::make_shared<PythonRegex>(text(op.second, "REGEX"));
                        } else {
                            throw std::invalid_argument("Unsupported pattern predicate " + op.first);
                        }
                    }
                    predicates.push_back(std::move(predicate));
                }
                tokens.push_back(std::move(predicates));
            }
            rule.patterns.push_back(std::move(tokens));
        }
        rules.push_back(std::move(rule));
    }
    return rules;
}

// ------------------------------------------------------------------- lemmatizer

struct RuleLemmatizer {
    std::string language;
    std::unordered_map<std::string, std::unordered_set<std::string>> index;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::string>>> exceptions;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> rules;
    std::unordered_map<std::uint64_t, std::string> lookup;
    bool present = false;
};

const Json & pos_entries(const Json & tables, const char * name) {
    static const Json empty = [] {
        Json value;
        value.kind = Json::object_t;
        return value;
    }();
    const auto * table = tables.get(name);
    if (!table) return empty;
    require(text(member(*table, "key_kind"), "key_kind") == "pos", std::string(name) + ": expected POS keys");
    const auto & entries = member(*table, "entries");
    require(entries.kind == Json::object_t, std::string(name) + ": invalid entries");
    return entries;
}

// spaCy EditTreeLemmatizer (top_k=1, backoff="orth", overwrite=False).
struct EditTreeLemmatizer {
    std::vector<std::uint32_t> class_to_tree;
    std::vector<EditTree> trees;
    bool present = false;
};

EditTreeLemmatizer parse_edit_trees(const Json & value) {
    EditTreeLemmatizer result;
    result.present = true;
    const auto & top_k = member(value, "top_k");
    const auto & overwrite = member(value, "overwrite");
    require(top_k.kind == Json::int_t && top_k.i == 1 && text(member(value, "backoff"), "backoff") == "orth" &&
                overwrite.kind == Json::bool_t && !overwrite.b,
            "Unsupported edit-tree lemmatizer settings");
    auto index = [](const Json & item, std::uint64_t bound) {
        require(item.kind == Json::int_t && item.i >= 0 && static_cast<std::uint64_t>(item.i) <= bound,
                "Invalid edit-tree integer");
        return static_cast<std::uint32_t>(item.i);
    };
    const auto & trees = member(value, "trees");
    require(trees.kind == Json::array_t && !trees.a.empty() && trees.a.size() <= 1000000, "Invalid edit trees");
    for (const auto & entry : trees.a) {
        require(entry.kind == Json::object_t, "Invalid edit tree");
        EditTree tree;
        if (entry.get("orig")) {
            tree.is_match = false;
            tree.original = text(member(entry, "orig"), "edit-tree orig");
            tree.replacement = text(member(entry, "subst"), "edit-tree subst");
        } else {
            tree.prefix_length = index(member(entry, "prefix_len"), 4096);
            tree.suffix_length = index(member(entry, "suffix_len"), 4096);
            tree.prefix_tree = index(member(entry, "prefix_tree"), EditTree::none);
            tree.suffix_tree = index(member(entry, "suffix_tree"), EditTree::none);
        }
        result.trees.push_back(std::move(tree));
    }
    validate_edit_trees(result.trees);
    const auto & classes = member(value, "class_to_tree");
    require(classes.kind == Json::array_t && !classes.a.empty(), "Invalid edit-tree classes");
    for (const auto & item : classes.a) result.class_to_tree.push_back(index(item, result.trees.size() - 1));
    return result;
}

RuleLemmatizer parse_lemmatizer(const Json & value, const std::string & language) {
    RuleLemmatizer result;
    result.present = true;
    require(text(member(value, "mode"), "mode") == "rule", "Only rule lemmatizers use the v2 profile");
    result.language = text(member(value, "language"), "lemmatizer language");
    require(result.language == language && (language == "en" || language == "fr"),
            "Unsupported rule lemmatizer language");
    const auto & overwrite = member(value, "overwrite");
    require(overwrite.kind == Json::bool_t && !overwrite.b, "Unsupported lemmatizer overwrite");
    const auto & tables = member(value, "tables");
    for (const auto & item : pos_entries(tables, "lemma_index").o) {
        require(item.second.kind == Json::array_t, "Invalid lemma_index");
        auto & words = result.index[item.first];
        for (const auto & word : item.second.a) words.insert(text(word, "lemma_index word"));
    }
    for (const auto & item : pos_entries(tables, "lemma_exc").o) {
        require(item.second.kind == Json::object_t, "Invalid lemma_exc");
        auto & table = result.exceptions[item.first];
        for (const auto & entry : item.second.o) {
            require(entry.second.kind == Json::array_t, "Invalid lemma_exc forms");
            auto & forms = table[entry.first];
            for (const auto & form : entry.second.a) forms.push_back(text(form, "lemma_exc form"));
        }
    }
    for (const auto & item : pos_entries(tables, "lemma_rules").o) {
        require(item.second.kind == Json::array_t, "Invalid lemma_rules");
        auto & rules = result.rules[item.first];
        for (const auto & pair : item.second.a) {
            require(pair.kind == Json::array_t && pair.a.size() == 2, "Invalid lemma rule");
            rules.emplace_back(text(pair.a[0], "rule old"), text(pair.a[1], "rule new"));
        }
    }
    if (const auto * table = tables.get("lemma_lookup")) {
        require(text(member(*table, "key_kind"), "key_kind") == "hash_uint64_decimal", "Invalid lemma_lookup keys");
        for (const auto & item : member(*table, "entries").o) {
            std::uint64_t key = 0;
            for (const char c : item.first) {
                require(c >= '0' && c <= '9', "Invalid lemma_lookup key");
                key = key * 10 + static_cast<std::uint64_t>(c - '0');
            }
            // spaCy stores one-element lists; both lookup paths use element 0.
            const auto & value = item.second;
            if (value.kind == Json::array_t) {
                require(!value.a.empty(), "Empty lemma_lookup value");
                result.lookup.emplace(key, text(value.a[0], "lemma_lookup value"));
            } else {
                result.lookup.emplace(key, text(value, "lemma_lookup value"));
            }
        }
    }
    return result;
}

std::map<std::string, std::string> morph_dict(const std::string & morph) {
    std::map<std::string, std::string> result;
    for (std::size_t at = 0; at < morph.size();) {
        const auto end = morph.find('|', at);
        const auto field = morph.substr(at, end == std::string::npos ? std::string::npos : end - at);
        const auto equal = field.find('=');
        if (equal != std::string::npos) result[field.substr(0, equal)] = field.substr(equal + 1);
        if (end == std::string::npos) break;
        at = end + 1;
    }
    return result;
}

std::string base64_decode(const std::string & input) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    require(input.size() % 4 == 0, "Invalid base64 length");
    std::string out;
    out.reserve(input.size() / 4 * 3);
    for (std::size_t i = 0; i < input.size(); i += 4) {
        std::uint32_t chunk = 0;
        int padding = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            const char c = input[i + j];
            std::uint32_t value = 0;
            if (c == '=') {
                require(i + 4 == input.size() && j >= 2, "Invalid base64 padding");
                ++padding;
            } else {
                const auto at = alphabet.find(c);
                require(at != std::string::npos && padding == 0, "Invalid base64 character");
                value = static_cast<std::uint32_t>(at);
            }
            chunk = (chunk << 6) | value;
        }
        out += static_cast<char>((chunk >> 16) & 0xFF);
        if (padding < 2) out += static_cast<char>((chunk >> 8) & 0xFF);
        if (padding < 1) out += static_cast<char>(chunk & 0xFF);
    }
    return out;
}

std::string ascii_lower(std::string value) {
    for (auto & c : value)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return value;
}

bool ends_with(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

struct PipelineV2::Impl {
    const LexicalModel * lexical = nullptr;
    std::string language;
    std::vector<std::string> components;
    std::vector<std::string> tags;
    std::vector<MorphLabel> morphology; // morphologizer: label -> (POS, canonical morph)
    std::vector<std::string> parser_moves;
    std::vector<Rule> rules;
    RuleLemmatizer lemmatizer;
    EditTreeLemmatizer edit_trees;
    std::unique_ptr<GgmlNumeric> numeric;
#ifdef CRISPEMBED_SPACY_TRF
    std::unique_ptr<TransformerPipeline> transformer;
#endif
    // Parser geometry: hidden width from the lower layer; trf parsers have no
    // upper layer (the maxout output is the move score vector).
    std::size_t parser_hidden = 0, parser_slot = 0, parser_columns = 0;
    bool parser_upper = true;
    std::unique_ptr<NerModel> ner;
    std::vector<float> lower_bias, lower_pad, upper_w, upper_b;

    std::vector<float> parser_scores(const Matrix & cache, const std::array<std::int32_t, 8> & ids) const {
        std::vector<float> summed(parser_slot, 0.0f);
        for (std::size_t f = 0; f < parser_features; ++f) {
            const auto row = ids[f] < 0 ? 0 : static_cast<std::size_t>(ids[f]) + 1;
            require(row < cache.rows, "spaCy v2 parser feature out of range");
            const float * source = cache.data.data() + row * parser_columns + f * parser_slot;
            for (std::size_t i = 0; i < parser_slot; ++i) summed[i] += source[i];
        }
        for (std::size_t i = 0; i < parser_slot; ++i) summed[i] += lower_bias[i];
        std::vector<float> hidden(parser_hidden);
        for (std::size_t h = 0; h < parser_hidden; ++h) {
            const float first = summed[h * parser_pieces];
            const float second = summed[h * parser_pieces + 1];
            hidden[h] = second > first ? second : first;
        }
        if (!parser_upper) {
            for (const float value : hidden) require(std::isfinite(value), "Nonfinite spaCy v2 parser score");
            return hidden;
        }
        std::vector<float> scores(parser_moves.size());
        for (std::size_t c = 0; c < scores.size(); ++c) {
            const float * w = upper_w.data() + c * parser_hidden;
            float sum = 0.0f;
            for (std::size_t h = 0; h < parser_hidden; ++h) sum += hidden[h] * w[h];
            scores[c] = sum + upper_b[c];
            require(std::isfinite(scores[c]), "Nonfinite spaCy v2 parser score");
        }
        return scores;
    }

    bool matches(const Predicate & predicate, const TokenAnnotation & annotation, const LexicalToken & token,
                 const std::string & lower) const {
        if (predicate.field == Field::IS_SPACE) return token.is_space == predicate.flag;
        const std::string & value = predicate.field == Field::TAG     ? annotation.tag
                                    : predicate.field == Field::POS   ? annotation.pos
                                    : predicate.field == Field::DEP   ? annotation.dep
                                    : predicate.field == Field::LOWER ? lower
                                                                      : token.text;
        switch (predicate.kind) {
        case Predicate::EQUAL:
            return value == predicate.value;
        case Predicate::IN:
            return predicate.set.count(value) != 0;
        case Predicate::NOT_IN:
            return predicate.set.count(value) == 0;
        case Predicate::REGEX:
            return predicate.regex->search(value);
        }
        return false;
    }

    // spaCy AttributeRuler: match every rule against the unmodified document,
    // sort by (rule index, start, end), then assign in order (later rules win).
    void apply_rules(std::vector<TokenAnnotation> & annotations, const std::vector<LexicalToken> & tokens,
                     const CancelCheck & cancelled) const {
        std::vector<std::string> lowered;
        lowered.reserve(tokens.size());
        for (const auto & token : tokens) lowered.push_back(lexical->python_lower(token.text));
        std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> found;
        for (std::size_t r = 0; r < rules.size(); ++r) {
            check_cancel(cancelled);
            std::set<std::pair<std::size_t, std::size_t>> spans;
            for (const auto & pattern : rules[r].patterns) {
                if (pattern.size() > tokens.size()) continue;
                for (std::size_t start = 0; start + pattern.size() <= tokens.size(); ++start) {
                    bool ok = true;
                    for (std::size_t t = 0; t < pattern.size() && ok; ++t)
                        for (const auto & predicate : pattern[t])
                            if (!matches(predicate, annotations[start + t], tokens[start + t], lowered[start + t])) {
                                ok = false;
                                break;
                            }
                    if (ok) spans.emplace(start, start + pattern.size());
                }
            }
            for (const auto & span : spans) found.emplace_back(r, span.first, span.second);
        }
        std::sort(found.begin(), found.end());
        for (const auto & match : found) {
            const auto & rule = rules[std::get<0>(match)];
            auto & target = annotations[std::get<1>(match) + rule.index];
            for (const auto & attr : rule.attrs) {
                if (attr.first == "POS")
                    target.pos = attr.second;
                else if (attr.first == "TAG")
                    target.tag = attr.second;
                else if (attr.first == "MORPH")
                    target.morph = attr.second;
                else if (attr.first == "LEMMA")
                    target.lemma = attr.second;
                else if (attr.first == "DEP")
                    target.dep = attr.second;
            }
        }
    }

    bool english_base_form(const TokenAnnotation & annotation) const {
        const auto pos = ascii_lower(annotation.pos);
        const auto morph = morph_dict(annotation.morph);
        auto get = [&](const char * key) -> const std::string * {
            const auto found = morph.find(key);
            return found == morph.end() ? nullptr : &found->second;
        };
        auto is = [&](const char * key, const char * value) {
            const auto * found = get(key);
            return found && *found == value;
        };
        if (pos == "noun" && is("Number", "Sing")) return true;
        if (pos == "verb" && is("VerbForm", "Inf")) return true;
        if (pos == "verb" && is("VerbForm", "Fin") && is("Tense", "Pres") && !get("Number")) return true;
        if (pos == "adj" && is("Degree", "Pos")) return true;
        if (is("VerbForm", "Inf")) return true;
        if (is("VerbForm", "None")) return true;
        if (is("Degree", "Pos")) return true;
        return false;
    }

    // spaCy FrenchLemmatizer.rule_lemmatize / Lemmatizer.lookup_lemmatize.
    std::string french_lemma(const TokenAnnotation & annotation, const LexicalToken & token) const {
        const auto pos = ascii_lower(annotation.pos);
        if (pos.empty() || pos == "eol" || pos == "space") return lexical->python_lower(token.text);
        static const std::set<std::string> rule_pos = { "noun",  "verb", "adj",  "adp",   "adv",  "aux",
                                                        "cconj", "det",  "pron", "punct", "sconj" };
        if (lemmatizer.rules.empty() || !rule_pos.count(pos)) {
            const auto found = lemmatizer.lookup.find(spacy_string_id(token.text)); // original case
            return found == lemmatizer.lookup.end() ? token.text : found->second;
        }
        const auto string = lexical->python_lower(token.text);
        const auto index = lemmatizer.index.find(pos);
        const bool has_index = index != lemmatizer.index.end();
        if (has_index && index->second.count(string)) return string;
        std::vector<std::string> forms;
        const auto exceptions = lemmatizer.exceptions.find(pos);
        if (exceptions != lemmatizer.exceptions.end()) {
            const auto found = exceptions->second.find(string);
            if (found != exceptions->second.end()) forms = found->second;
        }
        std::vector<std::string> oov_forms;
        const auto rules = lemmatizer.rules.find(pos);
        if (forms.empty() && rules != lemmatizer.rules.end()) {
            for (const auto & rule : rules->second) {
                if (!ends_with(string, rule.first)) continue;
                const auto form = string.substr(0, string.size() - rule.first.size()) + rule.second;
                if (form.empty()) continue;
                if ((has_index && index->second.count(form)) || !lexical->python_isalpha(form))
                    forms.push_back(form);
                else
                    oov_forms.push_back(form);
            }
        }
        if (forms.empty()) forms = oov_forms;
        if (forms.empty()) {
            const auto found = lemmatizer.lookup.find(spacy_string_id(string));
            forms.push_back(found == lemmatizer.lookup.end() ? string : found->second);
        }
        return forms.front(); // dict.fromkeys dedupe keeps the first element first
    }

    // spaCy Lemmatizer.rule_lemmatize (+ EnglishLemmatizer.is_base_form).
    std::string rule_lemma(const TokenAnnotation & annotation, const LexicalToken & token) const {
        if (language == "fr") return french_lemma(annotation, token);
        const auto & string = token.text;
        const auto pos = ascii_lower(annotation.pos);
        if (pos.empty() || pos == "eol" || pos == "space") return lexical->python_lower(string);
        if (language == "en" && english_base_form(annotation)) return lexical->python_lower(string);
        const auto index = lemmatizer.index.find(pos);
        const auto exceptions = lemmatizer.exceptions.find(pos);
        const auto rules = lemmatizer.rules.find(pos);
        const bool has_index = index != lemmatizer.index.end() && !index->second.empty();
        const bool has_exceptions = exceptions != lemmatizer.exceptions.end() && !exceptions->second.empty();
        const bool has_rules = rules != lemmatizer.rules.end() && !rules->second.empty();
        if (!has_index && !has_exceptions && !has_rules) return pos == "propn" ? string : lexical->python_lower(string);
        const auto lower = lexical->python_lower(string);
        std::vector<std::string> forms, oov_forms;
        if (has_rules) {
            for (const auto & rule : rules->second) {
                if (!ends_with(lower, rule.first)) continue;
                const auto form = lower.substr(0, lower.size() - rule.first.size()) + rule.second;
                if (form.empty()) continue;
                const bool known = has_index && index->second.count(form);
                if (known)
                    forms.insert(forms.begin(), form);
                else if (!lexical->python_isalpha(form))
                    forms.push_back(form);
                else
                    oov_forms.push_back(form);
            }
        }
        std::vector<std::string> unique;
        for (const auto & form : forms)
            if (std::find(unique.begin(), unique.end(), form) == unique.end()) unique.push_back(form);
        forms = std::move(unique);
        if (has_exceptions) {
            const auto found = exceptions->second.find(lower);
            if (found != exceptions->second.end())
                for (const auto & form : found->second)
                    if (std::find(forms.begin(), forms.end(), form) == forms.end()) forms.insert(forms.begin(), form);
        }
        if (forms.empty()) forms = oov_forms;
        if (forms.empty()) forms.push_back(string);
        return forms.front();
    }
};

PipelineV2::PipelineV2(const std::string & resources_json, std::map<std::string, ggml_tensor *> weights,
                       ggml_backend_t backend, const LexicalModel & lexical)
    : impl_(std::make_unique<Impl>()) {
    auto & m = *impl_;
    m.lexical = &lexical;
    require(resources_json.size() <= 512u * 1024u * 1024u, "spaCy v2 resources exceed bound");
    const auto root = core_json_value::parse(resources_json);
    const auto & format = member(root, "format_version");
    require(format.kind == Json::int_t && format.i == 2, "Unsupported spaCy v2 resource format");
    m.language = text(member(root, "language"), "language");
    require(m.language == "en" || m.language == "fr" || m.language == "de", "Unsupported spaCy v2 language");
    m.components = strings(member(root, "component_order"), "component_order", 16);
    const std::vector<std::string> known = { "tok2vec", "transformer",     "tagger",     "morphologizer",
                                             "parser",  "attribute_ruler", "lemmatizer", "ner" };
    for (const auto & component : m.components)
        require(std::find(known.begin(), known.end(), component) != known.end(),
                "Unsupported spaCy v2 component " + component);
    require((m.components.front() == "tok2vec" || m.components.front() == "transformer") &&
                std::find(m.components.begin(), m.components.end(), "parser") != m.components.end(),
            "spaCy v2 needs tok2vec and parser");
    const auto & labels = member(root, "labels");
    auto has = [&](const char * component) {
        return std::find(m.components.begin(), m.components.end(), component) != m.components.end();
    };
    if (has("tagger")) m.tags = strings(member(labels, "tagger"), "tagger labels", 4096);
    if (has("morphologizer")) {
        const auto & morph = member(labels, "morphologizer");
        const auto & overwrite = member(morph, "overwrite");
        const auto & extend = member(morph, "extend");
        require(overwrite.kind == Json::bool_t && overwrite.b && extend.kind == Json::bool_t && !extend.b,
                "Only overwrite/no-extend morphologizers are supported");
        const auto names = strings(member(morph, "labels"), "morphologizer labels", 8192);
        const auto & assign = member(morph, "assign");
        require(assign.kind == Json::array_t && assign.a.size() == names.size(), "Invalid morphologizer assignment");
        for (const auto & item : assign.a)
            m.morphology.push_back({ text(member(item, "pos"), "pos"), text(member(item, "morph"), "morph") });
    }
    const auto & parser = member(labels, "parser");
    m.parser_moves = strings(member(parser, "moves"), "parser moves", 137);
    (void)ParserState(0, m.parser_moves);
    if (has("attribute_ruler")) m.rules = parse_rules(member(labels, "attribute_ruler"));
    if (has("lemmatizer")) {
        const auto & lemma = member(labels, "lemmatizer");
        if (text(member(lemma, "mode"), "mode") == "edit_tree")
            m.edit_trees = parse_edit_trees(lemma);
        else
            m.lemmatizer = parse_lemmatizer(lemma, m.language);
    }

    // Numeric engine over the shared tok2vec, tagger and parser lower layer.
    std::map<std::string, ggml_tensor *> numeric_weights, ner_weights;
    for (const auto & item : weights) {
        require(item.second && item.second->type == GGML_TYPE_F32, "spaCy v2 tensors must be F32: " + item.first);
        (item.first.rfind("ner.", 0) == 0 ? ner_weights : numeric_weights).insert(item);
    }
    auto host = [&](const char * name, std::size_t count) {
        const auto found = numeric_weights.find(name);
        require(found != numeric_weights.end() && static_cast<std::size_t>(ggml_nelements(found->second)) == count,
                std::string("Invalid spaCy v2 tensor ") + name);
        std::vector<float> values(count);
        ggml_backend_tensor_get(found->second, values.data(), 0, count * sizeof(float));
        for (const float value : values) require(std::isfinite(value), std::string("Nonfinite tensor ") + name);
        return values;
    };
    {
        const auto found = numeric_weights.find("parser.lower.W");
        require(found != numeric_weights.end() && found->second->ne[3] == static_cast<std::int64_t>(parser_features) &&
                    found->second->ne[1] == static_cast<std::int64_t>(parser_pieces),
                "Invalid spaCy v2 parser lower layer");
        m.parser_hidden = static_cast<std::size_t>(found->second->ne[2]);
        m.parser_slot = m.parser_hidden * parser_pieces;
        m.parser_columns = parser_features * m.parser_slot;
        const auto * upper = parser.get("has_upper");
        m.parser_upper = !upper || (upper->kind == Json::bool_t && upper->b);
    }
    m.lower_bias = host("parser.lower.b", m.parser_slot);
    m.lower_pad = host("parser.lower.pad", m.parser_columns);
    if (m.parser_upper) {
        m.upper_w = host("parser.upper.W", m.parser_moves.size() * m.parser_hidden);
        m.upper_b = host("parser.upper.b", m.parser_moves.size());
    } else {
        require(m.parser_hidden == m.parser_moves.size() &&
                    numeric_weights.find("parser.upper.W") == numeric_weights.end(),
                "Upper-less spaCy parser must score every move");
    }
    if (has("tagger")) {
        const auto found = numeric_weights.find("tagger.W");
        require(found != numeric_weights.end() && found->second->ne[1] == static_cast<std::int64_t>(m.tags.size()),
                "Tagger weights do not match its labels");
    }
    if (has("morphologizer")) {
        const auto found = numeric_weights.find("morphologizer.W");
        require(found != numeric_weights.end() &&
                    found->second->ne[1] == static_cast<std::int64_t>(m.morphology.size()),
                "Morphologizer weights do not match its labels");
    } else {
        require(numeric_weights.find("morphologizer.W") == numeric_weights.end(), "Unexpected morphologizer weights");
    }
    if (m.edit_trees.present) {
        const auto found = numeric_weights.find("lemmatizer.W");
        require(found != numeric_weights.end() &&
                    found->second->ne[1] == static_cast<std::int64_t>(m.edit_trees.class_to_tree.size()),
                "Edit-tree lemmatizer weights do not match its labels");
        require(has("transformer"), "Edit-tree lemmatizers are admitted on the transformer path only");
    } else {
        require(numeric_weights.find("lemmatizer.W") == numeric_weights.end(),
                "Unexpected trainable lemmatizer weights");
    }
    if (m.components.front() == "transformer") {
#ifdef CRISPEMBED_SPACY_TRF
        const auto & spec = member(labels, "transformer");
        const auto & architecture = text(member(spec, "architecture"), "architecture");
        const auto & piece_encoder = text(member(spec, "piece_encoder"), "piece_encoder");
        require(((architecture == "camembert" && piece_encoder == "camembert_sentencepiece") ||
                 (architecture == "roberta" && piece_encoder == "roberta_byte_bpe") ||
                 (architecture == "bert" && piece_encoder == "bert_wordpiece")) &&
                    text(member(spec, "listener"), "listener") == "last_layer_reduce_mean",
                "Unsupported spaCy transformer profile");
        auto number = [&](const char * key) {
            const auto & value = member(spec, key);
            require(value.kind == Json::int_t && value.i > 0 && value.i < 100000, std::string("Invalid ") + key);
            return static_cast<int>(value.i);
        };
        TransformerConfig config;
        config.layers = number("layers");
        config.width = number("width");
        config.heads = number("heads");
        config.intermediate = number("intermediate");
        {
            const auto & value = member(spec, "padding_idx");
            require(value.kind == Json::int_t && value.i >= 0 && value.i < 100000, "Invalid padding_idx");
            config.padding_idx = static_cast<int>(value.i);
        }
        if (architecture == "bert") {
            const auto & value = member(spec, "position_offset");
            require(value.kind == Json::int_t && value.i == 0, "Invalid BERT position_offset");
            config.position_offset = 0;
        }
        config.max_positions = number("max_positions");
        config.model_max_length = number("model_max_length");
        config.window = number("window");
        config.stride = number("stride");
        const auto & eps = member(spec, "layer_norm_eps");
        // bert-base-german-cased ships layer_norm_eps = 0.0 (torch uses it as is).
        require(eps.is_number() && eps.number() >= 0 && eps.number() < 1, "Invalid layer_norm_eps");
        config.layer_norm_eps = static_cast<float>(eps.number());
        config.piece_encoder = piece_encoder;
        if (piece_encoder == "camembert_sentencepiece") {
            config.sentencepiece_model = base64_decode(text(member(spec, "sentencepiece_model_base64"), "spm model"));
        } else if (piece_encoder == "bert_wordpiece") {
            const auto & wordpiece = member(spec, "wordpiece");
            const auto & pieces = member(wordpiece, "pieces");
            require(pieces.kind == Json::array_t && !pieces.a.empty(), "Invalid WordPiece pieces");
            for (const auto & piece : pieces.a) config.wordpiece_pieces.push_back(text(piece, "WordPiece piece"));
            auto special = [&](const char * key) {
                const auto & value = member(wordpiece, key);
                require(value.kind == Json::int_t && value.i >= 0, std::string("Invalid WordPiece ") + key);
                return static_cast<int>(value.i);
            };
            config.bos_id = special("bos");
            config.eos_id = special("eos");
            config.unk_id = special("unk");
        } else {
            const auto & bpe = member(spec, "byte_bpe");
            const auto & pieces = member(bpe, "pieces");
            require(pieces.kind == Json::array_t && !pieces.a.empty(), "Invalid byte-BPE pieces");
            for (const auto & piece : pieces.a) config.bpe_pieces.push_back(text(piece, "byte-BPE piece"));
            const auto & merges = member(bpe, "merges");
            require(merges.kind == Json::array_t, "Invalid byte-BPE merges");
            for (const auto & merge : merges.a) {
                require(merge.kind == Json::array_t && merge.a.size() == 2, "Invalid byte-BPE merge");
                config.bpe_merges.emplace_back(text(merge.a[0], "merge"), text(merge.a[1], "merge"));
            }
            config.bpe_split_pattern = text(member(bpe, "split_pattern"), "split_pattern");
            auto special = [&](const char * key) {
                const auto & value = member(bpe, key);
                require(value.kind == Json::int_t && value.i >= 0, std::string("Invalid byte-BPE ") + key);
                return static_cast<int>(value.i);
            };
            config.bos_id = special("bos");
            config.eos_id = special("eos");
            config.unk_id = special("unk");
        }
        // A transformer-listener NER's reduce/lower layers run in the heads graph.
        for (const auto & item : ner_weights)
            if (item.first.rfind("ner.reduce.", 0) == 0 || item.first.rfind("ner.lower.W", 0) == 0)
                numeric_weights.insert(item);
        m.transformer = std::make_unique<TransformerPipeline>(std::move(config), std::move(numeric_weights), backend);
#else
        throw std::invalid_argument("This build has no spaCy transformer support (CRISPEMBED_SPACY_TRF=OFF)");
#endif
    } else {
        m.numeric = std::make_unique<GgmlNumeric>(std::move(numeric_weights), backend);
    }
    if (has("ner")) {
        const auto ner_moves = strings(member(member(labels, "ner"), "moves"), "ner moves", 1024);
        m.ner = std::make_unique<NerModel>(std::move(ner_weights), backend, ner_moves);
    } else {
        require(ner_weights.empty(), "NER weights without an NER component");
    }
}

PipelineV2::~PipelineV2() = default;

const std::string & PipelineV2::language() const {
    return impl_->language;
}

bool PipelineV2::has_ner() const {
    return impl_->ner != nullptr;
}

PipelineV2::Output PipelineV2::run(const std::vector<LexicalToken> & tokens, const TraceSink & trace,
                                   const CancelCheck & cancelled) const {
    const auto & m = *impl_;
    check_cancel(cancelled);
    const auto n = tokens.size();
    require(n <= max_tokens, "spaCy token count exceeds the admitted limit");
    Output output;
    if (n == 0) return output;
    std::vector<FeatureRow> features;
    std::vector<std::uint8_t> is_space;
    for (const auto & token : tokens) {
        features.push_back(token.features);
        is_space.push_back(token.is_space);
    }
#ifdef CRISPEMBED_SPACY_TRF
    const auto numeric = m.transformer ? m.transformer->run(tokens, trace, cancelled)
                                       : m.numeric->run(features, max_tokens, trace, cancelled);
#else
    const auto numeric = m.numeric->run(features, max_tokens, trace, cancelled);
#endif
    auto & annotations = output.annotations.tokens;
    annotations.resize(n);
    for (const auto & component : m.components) {
        check_cancel(cancelled);
        if (component == "tagger") {
            require(numeric.tagger.rows == n && numeric.tagger.cols == m.tags.size(), "Invalid tagger scores");
            for (std::size_t i = 0; i < n; ++i) {
                const float * row = numeric.tagger.data.data() + i * numeric.tagger.cols;
                std::size_t best = 0;
                for (std::size_t c = 1; c < numeric.tagger.cols; ++c)
                    if (row[c] > row[best]) best = c;
                annotations[i].tag_class = best;
                annotations[i].tag = m.tags[best];
            }
        } else if (component == "morphologizer") {
            // overwrite=True, extend=False: the label replaces POS and morph.
            require(numeric.morphologizer.rows == n && numeric.morphologizer.cols == m.morphology.size(),
                    "Invalid morphologizer scores");
            for (std::size_t i = 0; i < n; ++i) {
                const float * row = numeric.morphologizer.data.data() + i * numeric.morphologizer.cols;
                std::size_t best = 0;
                for (std::size_t c = 1; c < numeric.morphologizer.cols; ++c)
                    if (row[c] > row[best]) best = c;
                annotations[i].morph_class = best;
                annotations[i].pos = m.morphology[best].pos;
                annotations[i].morph = m.morphology[best].morph;
            }
        } else if (component == "parser") {
            ParserState state(n, m.parser_moves);
            std::size_t steps = 0;
            while (!state.is_final()) {
                check_cancel(cancelled);
                require(steps < 8 * n + 32, "spaCy parser transition budget exceeded");
                const auto ids = state.features();
                const auto valid = state.valid_actions();
                const auto scores = m.parser_scores(numeric.parser_cache, ids);
                std::size_t chosen = 0;
                bool selected = false;
                for (std::size_t c = 0; c < scores.size(); ++c)
                    if (valid[c] && (!selected || scores[c] > scores[chosen])) {
                        chosen = c;
                        selected = true;
                    }
                require(selected, "spaCy parser has no legal transition");
                if (trace) {
                    const auto prefix = "parser.step." + std::to_string(steps) + ".";
                    trace(prefix + "scores", TensorView{ { scores.size() }, scores.data(), scores.size() });
                }
                state.step(chosen);
                ++steps;
            }
            output.annotations.parser_steps = steps;
            const auto parsed = state.finish(is_space, cancelled);
            for (std::size_t i = 0; i < n; ++i) {
                annotations[i].head = parsed.heads[i];
                annotations[i].dep = parsed.dependencies[i];
                annotations[i].is_sent_start = parsed.sentence_starts[i] != 0;
            }
        } else if (component == "attribute_ruler") {
            m.apply_rules(annotations, tokens, cancelled);
        } else if (component == "lemmatizer" && m.edit_trees.present) {
            const auto & scores = numeric.lemmatizer;
            require(scores.rows == n && scores.cols == m.edit_trees.class_to_tree.size(), "Invalid lemmatizer scores");
            for (std::size_t i = 0; i < n; ++i) {
                const float * row = scores.data.data() + i * scores.cols;
                std::size_t best = 0;
                for (std::size_t c = 1; c < scores.cols; ++c)
                    if (row[c] > row[best]) best = c;
                // Inapplicable tree -> backoff orth; existing lemma kept.
                annotations[i].lemma = apply_lemma(m.edit_trees.trees, m.edit_trees.class_to_tree[best], tokens[i].text,
                                                   annotations[i].lemma)
                                           .lemma;
            }
        } else if (component == "lemmatizer") {
            for (std::size_t i = 0; i < n; ++i)
                if (annotations[i].lemma.empty()) annotations[i].lemma = m.rule_lemma(annotations[i], tokens[i]);
        } else if (component == "ner") {
            std::vector<std::uint8_t> starts;
            for (const auto & annotation : annotations) starts.push_back(annotation.is_sent_start ? 1 : 0);
            if (m.ner->listener()) {
                std::vector<std::uint8_t> spaces;
                for (const auto & token : tokens) spaces.push_back(token.is_space);
                output.entities = m.ner->run_cached(numeric.ner_lower.data, spaces, starts, trace, cancelled);
            } else {
                output.entities = m.ner->run(features, starts, trace, cancelled);
            }
            output.entity_labels = m.ner->labels();
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (annotations[i].is_sent_start) {
            if (!output.annotations.sentences.empty()) output.annotations.sentences.back().end_token = i;
            output.annotations.sentences.push_back({ i, n });
        }
    }
    check_cancel(cancelled);
    return output;
}

} // namespace spacy
} // namespace crispembed
