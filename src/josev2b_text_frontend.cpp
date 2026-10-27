// Publisher schema semantics: mayflowergmbh/Josev-2B@6f0a31c8 (Apache-2.0).
// Independent implementation; see docs/josev2b-tokenizer-schema.md.
#include "josev2b_text_frontend.h"
#include "josev2b_packed_profile.h"
#include "core/bpe.h"
#include "core/decision_schema.h"
#include "core/env_gate.h"
#include "core/nfc.h"
#include "gguf.h"
#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace josev2b {
namespace {
using V = core_decision::Value;
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
bool utf8(const std::string & s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i++];
        if (c < 128) continue;
        const int n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
        if (!n || i + n > s.size()) return false;
        for (int k = 0; k < n; ++k) {
            const unsigned char d = s[i++];
            if (d < 0x80 || d > 0xbf ||
                (!k && ((c == 0xe0 && d < 0xa0) || (c == 0xed && d >= 0xa0) || (c == 0xf0 && d < 0x90) ||
                        (c == 0xf4 && d >= 0x90))))
                return false;
        }
    }
    return true;
}
void strict_strings(const std::string & s) {
    bool inside = false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"')
            inside = !inside;
        else if (inside && s[i] == '\\') {
            require(++i < s.size() && std::string("\"\\/bfnrtu").find(s[i]) != std::string::npos,
                    "invalid JSON escape");
            if (s[i] == 'u')
                for (int k = 0; k < 4; ++k) {
                    require(++i < s.size() && std::string("0123456789abcdefABCDEF").find(s[i]) != std::string::npos,
                            "invalid Unicode escape");
                }
        }
    }
}
void scalar_strings(const V & v) {
    if (v.kind == V::string) require(utf8(v.text), "non-scalar Unicode string");
    for (const auto & p : v.fields) {
        require(utf8(p.first), "non-scalar Unicode key");
        scalar_strings(p.second);
    }
    for (const auto & item : v.list) scalar_strings(item);
}
std::string quote(const std::string & s) {
    require(utf8(s), "Josev requires Unicode scalar strings");
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
std::string render(const V & v, bool literal = true) {
    if (v.kind == V::string) return literal ? (require(utf8(v.text), "invalid Unicode"), v.text) : quote(v.text);
    if (v.kind == V::null_value) return "null";
    if (v.kind == V::boolean) return v.flag ? "true" : "false";
    if (v.kind == V::number) return core_decision::python_str(v);
    std::string out = v.kind == V::array ? "[" : "{";
    if (v.kind == V::array) {
        for (size_t i = 0; i < v.list.size(); ++i) {
            if (i) out += ',';
            out += render(v.list[i], false);
        }
    } else {
        auto fields = v.fields;
        std::sort(fields.begin(), fields.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        for (size_t i = 0; i < fields.size(); ++i) {
            if (i) out += ',';
            out += quote(fields[i].first) + ":" + render(fields[i].second, false);
        }
    }
    return out + (v.kind == V::array ? "]" : "}");
}
const V & member(const V & v, const char * name) {
    require(v.kind == V::object, "Josev expected an object");
    for (const auto & p : v.fields)
        if (p.first == name) return p.second;
    throw std::runtime_error(std::string("missing Josev member: ") + name);
}
V string(const std::string & s) {
    V v;
    v.kind = V::string;
    v.text = s;
    return v;
}
int key(const gguf_context * g, const char * name, gguf_type type) {
    const auto k = gguf_find_key(g, name);
    require(k >= 0 && gguf_get_kv_type(g, k) == type, "missing or mistyped tokenizer metadata");
    return int(k);
}
std::string str(const gguf_context * g, const char * name) {
    return gguf_get_val_str(g, key(g, name, GGUF_TYPE_STRING));
}
std::vector<std::string> strings(const gguf_context * g, const char * name) {
    int k = key(g, name, GGUF_TYPE_ARRAY);
    require(gguf_get_arr_type(g, k) == GGUF_TYPE_STRING, "mistyped string array");
    const size_t n = gguf_get_arr_n(g, k);
    require(n <= 1000000, "tokenizer array too large");
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(g, k, i));
    return out;
}
std::vector<int32_t> ints(const gguf_context * g, const char * name) {
    int k = key(g, name, GGUF_TYPE_ARRAY);
    require(gguf_get_arr_type(g, k) == GGUF_TYPE_INT32, "mistyped int array");
    const size_t n = gguf_get_arr_n(g, k);
    require(n <= 256, "added-token array too large");
    const auto * p = static_cast<const int32_t *>(gguf_get_arr_data(g, k));
    return std::vector<int32_t>(p, p + n);
}
const char * pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
const char * prefix = "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. Each answer "
                      "must be exactly one of that field's allowed options.<|im_end|>\n<|im_start|>user\nSTATE:\n";
const char * suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";
} // namespace

bool text_frontend::load(const char * path, std::string & error) {
    return load_file(path, false, error);
}
bool text_frontend::load_packed(const char * path, std::string & error) {
    return load_file(path, true, error);
}
bool text_frontend::load_file(const char * path, bool packed, std::string & error) {
    *this = {};
    error.clear();
    if (!core_env::on("CRISPEMBED_JOSEV_TEXT_FRONTEND")) {
        error = "Josev text frontend gate is off";
        return false;
    }
    try {
        require(!packed || core_env::on("CRISPEMBED_JOSEV_PACKED"), "Josev packed gate is off");
        require(path, "null tokenizer path");
        if (packed) require(josev2b_packed::file_bounds(path, error), "packed file bounds rejected");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(file && file.tellg() > 0 &&
                    uint64_t(file.tellg()) <= (packed ? uint64_t(8) * 1024 * 1024 * 1024 : uint64_t(64) * 1024 * 1024),
                "tokenizer component size invalid");
        gguf_init_params params = { true, nullptr };
        std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_from_file(path, params), gguf_free);
        require(bool(g), "cannot open tokenizer GGUF metadata");
        if (packed)
            require(josev2b_packed::metadata_valid(g.get(), error), "Josev packed metadata rejected");
        else
            require(gguf_get_n_tensors(g.get()) == 0, "expected metadata-only tokenizer GGUF");
        require(str(g.get(), "general.architecture") == (packed ? josev2b_packed::architecture : "josev2b_tokenizer") &&
                    str(g.get(), "josev2b.tokenizer.format") == "crispembed-josev-tokenizer-v1" &&
                    str(g.get(), "general.source.huggingface.repository") == "mayflowergmbh/Josev-2B" &&
                    str(g.get(), "general.source.huggingface.revision") == "6f0a31c8ed683aba79097eb3f48df2c9ab560afd" &&
                    str(g.get(), "josev2b.tokenizer.source_sha256") ==
                        "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4" &&
                    str(g.get(), "josev2b.tokenizer.config_sha256") ==
                        "50470511708577d43f7c18c2d8e2ab02fb8abf3add1c40f60b74579d6d16e27e" &&
                    str(g.get(), "josev2b.serializer.source_sha256") ==
                        "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3" &&
                    str(g.get(), "josev2b.tokenizer.normalizer") == "NFC" &&
                    str(g.get(), "josev2b.tokenizer.pattern") == pattern,
                "tokenizer source/profile differs");
        text_frontend next;
        next.vocab_ = strings(g.get(), "tokenizer.ggml.tokens");
        next.added_ = ints(g.get(), "josev2b.tokenizer.added_ids");
        const auto flags = ints(g.get(), "josev2b.tokenizer.added_flags");
        const auto merges = strings(g.get(), "tokenizer.ggml.merges");
        require(next.vocab_.size() >= 200000 && next.vocab_.size() <= 250000 && !merges.empty() &&
                    next.added_.size() == 33 && flags.size() == next.added_.size(),
                "vocab/merge/added count differs");
        for (size_t i = 0; i < next.vocab_.size(); ++i)
            if (!next.vocab_[i].empty())
                require(next.tokens_.emplace(next.vocab_[i], int32_t(i)).second, "duplicate token");
        for (size_t i = 0; i < merges.size(); ++i)
            require(merges[i].find(' ') != std::string::npos && next.merges_.emplace(merges[i], int32_t(i)).second,
                    "invalid or duplicate merge");
        std::vector<int32_t> seen;
        for (size_t i = 0; i < next.added_.size(); ++i) {
            const auto id = next.added_[i];
            require(id >= 0 && size_t(id) < next.vocab_.size() && !next.vocab_[id].empty() &&
                        (flags[i] == 0 || flags[i] == 16) && std::find(seen.begin(), seen.end(), id) == seen.end(),
                    "unsupported added token flags/ID");
            seen.push_back(id);
        }
        for (int b = 0; b < 256; ++b) {
            char c = char(b);
            require(next.tokens_.count(core_bpe::bytes_to_unicode(&c, 1)), "missing byte token");
        }
        next.enabled_ = true;
        *this = std::move(next);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
std::vector<int32_t> text_frontend::tokens(const std::string & text) const {
    require(utf8(text), "invalid Unicode tokenizer input");
    std::vector<int32_t> ids;
    for (size_t start = 0; start < text.size();) {
        size_t next = text.size();
        int selected = -1;
        for (int id : added_) {
            const auto found = text.find(vocab_[id], start);
            if (found != std::string::npos &&
                (found < next || (found == next && (selected < 0 || vocab_[id].size() > vocab_[selected].size())))) {
                next = found;
                selected = id;
            }
        }
        const auto part = core_bpe::tokenize_pretokenized(
            tokens_, merges_,
            core_bpe::bytelevel_pretokenize(core_nfc::normalize(text.substr(start, next - start)), 1, true));
        ids.insert(ids.end(), part.begin(), part.end());
        if (selected < 0) break;
        ids.push_back(selected);
        start = next + vocab_[selected].size();
    }
    return ids;
}
bool text_frontend::encode_record(const std::string & json, encoded_text & output, std::string & error,
                                  int max_length) const {
    output = {};
    error.clear();
    try {
        require(enabled_, "Josev text frontend context is disabled");
        require(max_length >= 1 && max_length <= 512 && json.size() <= 65536 && utf8(json),
                "request bounds/Unicode invalid");
        strict_strings(json);
        const auto root = core_decision::Parser(json).parse();
        scalar_strings(root);
        require(!root.get("images").truth() && !root.get("videos").truth(), "Josev text frontend rejects media");
        const auto & questions = member(root, "questions");
        require(questions.kind == V::object && !questions.fields.empty() && questions.fields.size() <= 16,
                "question bounds");
        encoded_text result;
        auto schema = tokens("\n\nSCHEMA FIELDS:\n");
        size_t total_options = 0;
        auto append = [&](const std::string & s) {
            auto ids = tokens(s);
            schema.insert(schema.end(), ids.begin(), ids.end());
        };
        for (const auto & item : questions.fields) {
            const auto & q = item.second;
            const auto & type = member(q, "type");
            require(type.kind == V::string && (type.text == "noul" || type.text == "choice" || type.text == "score"),
                    "unsupported question type");
            text_field field;
            field.question_id = item.first;
            field.question_type = type.text == "noul" ? 0 : type.text == "choice" ? 1 : 2;
            append("\nFIELD " + std::to_string(result.questions.size() + 1) + "\nID: " + item.first +
                   "\nTYPE: " + type.text + "\nINSTRUCTION: ");
            field.question_span.start = int(schema.size());
            const auto & instruction = q.get("instructions");
            append(instruction.kind == V::null_value || (instruction.kind == V::string && instruction.text.empty())
                       ? item.first
                       : render(instruction));
            field.question_span.end = int(schema.size());
            append("\nALLOWED OPTIONS:\n");
            std::vector<std::pair<std::string, V>> options;
            if (field.question_type == 0) {
                V yes = string("The proposition is true or the answer is yes.");
                V no = string("The proposition is false or the answer is no.");
                const auto & criteria = q.get("criteria");
                if (criteria.truth()) {
                    require(criteria.kind == V::object, "noul criteria must be an object");
                    for (const auto & p : criteria.fields) {
                        if (p.first == "true") yes = p.second;
                        if (p.first == "false") no = p.second;
                    }
                }
                options = { { "true", yes }, { "false", no } };
            } else {
                const auto & criteria = member(q, "criteria");
                require(criteria.kind == (field.question_type == 1 ? V::object : V::array),
                        "unsupported criteria kind");
                if (field.question_type == 1) {
                    options = criteria.fields;
                    std::sort(options.begin(), options.end(),
                              [](const auto & a, const auto & b) { return a.first < b.first; });
                } else
                    for (size_t i = 0; i < criteria.list.size(); ++i)
                        options.emplace_back(std::to_string(i), criteria.list[i]);
            }
            require(!options.empty() && (total_options += options.size()) <= 128, "option bounds");
            for (const auto & option : options) {
                append("OPTION " + std::to_string(field.option_ids.size() + 1) + ": ");
                V semantics;
                semantics.kind = V::object;
                semantics.fields.emplace_back("option_id", string(option.first));
                if (option.second.kind != V::null_value) semantics.fields.emplace_back("description", option.second);
                text_span span;
                span.start = int(schema.size());
                append(render(semantics));
                span.end = int(schema.size());
                field.option_ids.push_back(option.first);
                field.option_spans.push_back(span);
                field.lexical_ids.emplace_back(schema.begin() + span.start, schema.begin() + span.end);
                append("\n");
            }
            append("END FIELD\n");
            result.questions.push_back(std::move(field));
        }
        auto before = tokens(prefix), after = tokens(suffix), state = tokens(render(member(root, "state")));
        const auto fixed = before.size() + schema.size() + after.size();
        require(fixed <= size_t(max_length), "schema requires more tokens than maximum before state");
        state.resize(std::min(state.size(), size_t(max_length) - fixed));
        const int offset = int(before.size() + state.size());
        for (auto & q : result.questions) {
            q.question_span.start += offset;
            q.question_span.end += offset;
            for (auto & s : q.option_spans) {
                s.start += offset;
                s.end += offset;
            }
        }
        result.input_ids = std::move(before);
        result.input_ids.insert(result.input_ids.end(), state.begin(), state.end());
        result.input_ids.insert(result.input_ids.end(), schema.begin(), schema.end());
        result.input_ids.insert(result.input_ids.end(), after.begin(), after.end());
        result.attention_mask.assign(result.input_ids.size(), 1);
        bool has_id = false;
        for (const auto & p : root.fields)
            if (p.first == "id") has_id = true;
        result.record_id = has_id ? core_decision::python_str(root.get("id")) : "unknown";
        output = std::move(result);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
} // namespace josev2b
