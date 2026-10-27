// Independent pinned tokenizers0.22.2 Unigram/role implementation. No generic
// SentencePiece/WordPiece behavior changes. See docs/moedern-unigram-frontend.md.
#include "moedern_text_frontend.h"
#include "core/env_gate.h"
#include "core/unicode_categ.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>

namespace moedern_colbert {
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
uint32_t next(const std::string & text, size_t & at) {
    const size_t start = at;
    const unsigned char c = text.at(at++);
    if (c < 128) return c;
    const int n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
    require(n && at + n <= text.size(), "invalid UTF8 scalar");
    for (int k = 0; k < n; ++k) {
        const unsigned char d = text[at++];
        require(d >= 0x80 && d <= 0xbf &&
                    (k || !((c == 0xe0 && d < 0xa0) || (c == 0xed && d >= 0xa0) || (c == 0xf0 && d < 0x90) ||
                            (c == 0xf4 && d >= 0x90))),
                "invalid UTF8 scalar");
    }
    size_t copy = start;
    return core_unicode::utf8_next(text.data(), text.size(), copy);
}
bool whitespace(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || cp == 32 || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
           cp == 0x3000;
}
int key(gguf_context * g, const char * name, gguf_type type) {
    const int k = int(gguf_find_key(g, name));
    require(k >= 0 && gguf_get_kv_type(g, k) == type, "missing/mistyped tokenizer metadata");
    return k;
}
std::string string(gguf_context * g, const char * name) {
    return gguf_get_val_str(g, key(g, name, GGUF_TYPE_STRING));
}
int array_key(gguf_context * g, const char * name, gguf_type type, size_t cap) {
    int k = key(g, name, GGUF_TYPE_ARRAY);
    require(gguf_get_arr_type(g, k) == type && gguf_get_arr_n(g, k) <= cap, "invalid tokenizer array type/budget");
    return k;
}
std::vector<std::string> strings(gguf_context * g, const char * name, size_t cap) {
    const int k = array_key(g, name, GGUF_TYPE_STRING, cap);
    std::vector<std::string> out;
    for (size_t i = 0; i < gguf_get_arr_n(g, k); ++i) out.emplace_back(gguf_get_arr_str(g, k, i));
    return out;
}
std::vector<int32_t> ints(gguf_context * g, const char * name, size_t cap) {
    const int k = array_key(g, name, GGUF_TYPE_INT32, cap);
    std::vector<int32_t> out(gguf_get_arr_n(g, k));
    if (!out.empty()) std::memcpy(out.data(), gguf_get_arr_data(g, k), out.size() * sizeof(int32_t));
    return out;
}
void replace(std::string & text, const std::string & from, const std::string & to) {
    for (size_t at = 0; (at = text.find(from, at)) != std::string::npos; at += to.size())
        text.replace(at, from.size(), to);
}
const char * marker = "\xe2\x96\x81";
} // namespace
bool text_frontend::load(const char * path, std::string & error) {
    *this = {};
    error.clear();
    try {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
                          std::numeric_limits<double>::digits == 53,
                      "exact binary64 scores required");
        require(core_env::on("CRISPEMBED_MOEDERN_TEXT_FRONTEND"), "MoEdern text frontend gate is OFF");
        require(path && *path, "empty tokenizer component path");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(file && file.tellg() > 0 && file.tellg() <= 16 * 1024 * 1024, "bounded tokenizer component required");
        const gguf_init_params params = { true, nullptr };
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_from_file(path, params), gguf_free);
        require(meta && gguf_get_n_tensors(meta.get()) == 0, "metadata-only tokenizer component required");
        auto * g = meta.get();
        require(string(g, "general.architecture") == "moedern_unigram_tokenizer" &&
                    string(g, "moedern.tokenizer.format") == "crispembed-moedern-unigram-v1" &&
                    string(g, "moedern.tokenizer.revision") == "9d360b8b88ae11bea5784bf9012ff509fdd5c2da" &&
                    string(g, "moedern.tokenizer.source_sha256") ==
                        "7357662aa3be57e8fd953365a37f892dab4ee702b817e1efcef9f193a388aff0" &&
                    string(g, "moedern.tokenizer.original_report_sha256") ==
                        "b50ca1369f454d61b8c95583d8eb77a847c47ede29e52ed66b575784f4e4660f" &&
                    string(g, "moedern.tokenizer.normalizer") == "rust0.22.2-two-nfkd-stripaccents-lowercase-passes" &&
                    string(g, "moedern.tokenizer.pretokenizer") == "WhitespaceSplit+Metaspace-always-split" &&
                    string(g, "moedern.tokenizer.model") == "Unigram-f64-unk1-fuse-unknown-no-byte-fallback",
                "tokenizer source/profile differs");
        text_frontend result;
        result.vocab_ = strings(g, "tokenizer.ggml.tokens", 30002);
        const int scores = array_key(g, "moedern.tokenizer.scores_f64", GGUF_TYPE_FLOAT64, 30000);
        result.scores_.resize(gguf_get_arr_n(g, scores));
        if (!result.scores_.empty())
            std::memcpy(result.scores_.data(), gguf_get_arr_data(g, scores), result.scores_.size() * sizeof(double));
        require(result.vocab_.size() == 30002 && result.scores_.size() == 30000 && result.vocab_[0] == "<pad>" &&
                    result.vocab_[1] == "<unk>" && result.vocab_[2] == "[CLS]" && result.vocab_[3] == "[SEP]" &&
                    result.vocab_[4] == "[MASK]" && result.vocab_[30000] == "[Q] " && result.vocab_[30001] == "[D] ",
                "Unigram vocabulary layout differs");
        result.added_ids_ = ints(g, "moedern.tokenizer.added_ids", 7);
        result.added_flags_ = ints(g, "moedern.tokenizer.added_flags", 7);
        result.added_content_ = strings(g, "moedern.tokenizer.added_content", 7);
        require(result.added_ids_ == std::vector<int32_t>({ 0, 1, 2, 3, 4, 30000, 30001 }) &&
                    result.added_flags_.size() == 7 && result.added_content_.size() == 7,
                "added token cardinality/IDs differ");
        for (size_t i = 0; i < 7; ++i) {
            const int expected = i == 4 ? 18 : i < 5 ? 16 : result.added_flags_[i];
            require(result.added_content_[i] == result.vocab_[result.added_ids_[i]] &&
                        result.added_flags_[i] == expected && (i < 5 || expected == 8 || expected == 16),
                    "unsupported effective added token flags");
        }
        result.skip_ = ints(g, "moedern.tokenizer.skip_ids", 32);
        require(!result.skip_.empty() && std::is_sorted(result.skip_.begin(), result.skip_.end()) &&
                    std::adjacent_find(result.skip_.begin(), result.skip_.end()) == result.skip_.end(),
                "invalid skiplist");
        for (int id : result.skip_) require(id >= 0 && id < 30002 && id != 1, "invalid skiplist ID");
        const auto cp = ints(g, "moedern.tokenizer.normalizer_codepoints", 65536);
        const auto values = strings(g, "moedern.tokenizer.normalizer_values", 65536);
        require(cp.size() == values.size() && cp.size() > 10000, "normalizer map cardinality differs");
        int32_t previous = -1;
        for (size_t i = 0; i < cp.size(); ++i) {
            require(cp[i] > previous && cp[i] <= 0x10ffff && !(cp[i] >= 0xd800 && cp[i] <= 0xdfff) &&
                        values[i].size() <= 256,
                    "invalid scalar normalizer map");
            for (size_t at = 0; at < values[i].size();) next(values[i], at);
            result.normalization_.emplace(uint32_t(cp[i]), values[i]);
            previous = cp[i];
        }
        result.trie_.push_back({});
        double minimum = std::numeric_limits<double>::infinity();
        for (size_t id = 0; id < result.scores_.size(); ++id) {
            require(std::isfinite(result.scores_[id]) && !result.vocab_[id].empty(), "nonfinite/empty Unigram item");
            minimum = std::min(minimum, result.scores_[id]);
            int node_id = 0;
            for (unsigned char c : result.vocab_[id]) {
                auto found = result.trie_[node_id].next.find(c);
                if (found == result.trie_[node_id].next.end()) {
                    const int target = int(result.trie_.size());
                    result.trie_[node_id].next.emplace(c, target);
                    result.trie_.push_back({});
                    node_id = target;
                } else
                    node_id = found->second;
            }
            require(result.trie_[node_id].id < 0, "duplicate Unigram piece");
            result.trie_[node_id].id = int(id);
        }
        result.unknown_score_ = minimum - 10.0;
        result.enabled_ = true;
        *this = std::move(result);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
std::string text_frontend::normalize(const std::string & text) const {
    std::string input = text;
    replace(input, "``", "\"");
    replace(input, "''", "\"");
    std::string out;
    for (size_t at = 0; at < input.size();) {
        const size_t begin = at;
        const uint32_t cp = next(input, at);
        const auto item = normalization_.find(cp);
        const std::string part = item == normalization_.end() ? input.substr(begin, at - begin) : item->second;
        for (char c : part) {
            if (c == ' ' && !out.empty() && out.back() == ' ') continue;
            out += c;
        }
    }
    return out;
}
std::vector<std::string> text_frontend::pretokens(const std::string & text) const {
    std::vector<std::string> result;
    for (size_t at = 0; at < text.size();) {
        size_t begin = at;
        uint32_t cp = next(text, at);
        if (whitespace(cp)) continue;
        while (at < text.size()) {
            size_t end = at;
            cp = next(text, end);
            if (whitespace(cp)) break;
            at = end;
        }
        auto word = text.substr(begin, at - begin);
        if (word.compare(0, 3, marker)) word = std::string(marker) + word;
        size_t start = 0;
        while (start < word.size()) {
            size_t end = word.find(marker, start + 3);
            if (end == std::string::npos) end = word.size();
            result.push_back(word.substr(start, end - start));
            start = end;
        }
    }
    return result;
}
void text_frontend::unigram(const std::string & text, std::vector<int32_t> & ids) const {
    struct path {
        double score = 0;
        int previous = -1, id = -1;
    };
    std::vector<path> best(text.size() + 1);
    for (size_t at = 0; at < text.size();) {
        size_t single_end = at;
        next(text, single_end);
        bool single = false;
        int node_id = 0;
        for (size_t end = at; end < text.size();) {
            const auto child = trie_[node_id].next.find(static_cast<unsigned char>(text[end++]));
            if (child == trie_[node_id].next.end()) break;
            node_id = child->second;
            const int id = trie_[node_id].id;
            if (id < 0) continue;
            const double score = scores_[id] + best[at].score;
            if (best[end].previous < 0 || score > best[end].score) best[end] = { score, int(at), id };
            if (end == single_end) single = true;
        }
        if (!single) {
            const double score = unknown_score_ + best[at].score;
            if (best[single_end].previous < 0 || score > best[single_end].score)
                best[single_end] = { score, int(at), 1 };
        }
        at = single_end;
    }
    std::vector<int32_t> reverse;
    for (int end = int(text.size()); end > 0;) {
        const auto & p = best[end];
        require(p.previous >= 0 && p.previous < end, "Unigram path unreachable");
        if (p.id != 1 || reverse.empty() || reverse.back() != 1) reverse.push_back(p.id);
        end = p.previous;
    }
    ids.insert(ids.end(), reverse.rbegin(), reverse.rend());
}
void text_frontend::normalized(const std::string & text, std::vector<int32_t> & ids) const {
    const auto value = normalize(text);
    size_t start = 0;
    while (start < value.size()) {
        size_t end = value.size();
        int selected = -1;
        std::string content;
        for (size_t i = 0; i < added_ids_.size(); ++i)
            if (added_flags_[i] & 8) {
                const auto token = normalize(added_content_[i]);
                const auto found = value.find(token, start);
                if (found != std::string::npos && (found < end || (found == end && token.size() > content.size()))) {
                    end = found;
                    selected = int(i);
                    content = token;
                }
            }
        for (const auto & segment : pretokens(value.substr(start, end - start))) unigram(segment, ids);
        if (selected < 0) break;
        ids.push_back(added_ids_[selected]);
        start = end + content.size();
    }
}
void text_frontend::raw(const std::string & text, std::vector<int32_t> & ids) const {
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.size();
        int selected = -1;
        for (size_t i = 0; i < added_ids_.size(); ++i)
            if (!(added_flags_[i] & 8)) {
                const auto found = text.find(added_content_[i], start);
                if (found != std::string::npos &&
                    (found < end ||
                     (found == end && (selected < 0 || added_content_[i].size() > added_content_[selected].size())))) {
                    end = found;
                    selected = int(i);
                }
            }
        size_t ordinary_end = end;
        if (selected >= 0 && (added_flags_[selected] & 2)) {
            size_t whitespace_begin = start;
            for (size_t at = start; at < end;) {
                const auto cp = next(text, at);
                if (!whitespace(cp)) whitespace_begin = at;
            }
            ordinary_end = whitespace_begin;
        }
        normalized(text.substr(start, ordinary_end - start), ids);
        if (selected < 0) break;
        ids.push_back(added_ids_[selected]);
        start = end + added_content_[selected].size();
    }
}
bool text_frontend::encode(const std::string & text, role task, text_tokens & out, std::string & error) const {
    out = {};
    error.clear();
    try {
        require(enabled_, "MoEdern text frontend context is disabled");
        require(text.size() <= 65536, "text exceeds 64KiB");
        require(task == role::query || task == role::document, "explicit query/document role required");
        for (size_t at = 0; at < text.size();) next(text, at);
        std::vector<int32_t> body;
        raw(std::string(task == role::query ? "[Q] " : "[D] ") + text, body);
        const size_t maximum = task == role::query ? 32 : 512;
        if (body.size() > maximum - 2) body.resize(maximum - 2);
        text_tokens result;
        result.ids.push_back(2);
        result.ids.insert(result.ids.end(), body.begin(), body.end());
        result.ids.push_back(3);
        result.supplied_mask.assign(result.ids.size(), 1);
        if (task == role::query) {
            result.ids.resize(32, 4);
            result.supplied_mask.resize(32, 0);
        }
        result.effective_mask = result.supplied_mask;
        result.scoring_mask = result.supplied_mask;
        if (task == role::query) {
            result.expansion_positions.resize(32);
            for (size_t i = 0; i < 32; ++i) result.expansion_positions[i] = uint8_t(!result.supplied_mask[i]);
            std::fill(result.effective_mask.begin(), result.effective_mask.end(), 1);
            std::fill(result.scoring_mask.begin(), result.scoring_mask.end(), 1);
        } else
            for (size_t i = 0; i < result.ids.size(); ++i)
                if (std::binary_search(skip_.begin(), skip_.end(), result.ids[i])) result.scoring_mask[i] = 0;
        out = std::move(result);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
#ifdef CRISPEMBED_MOEDERN_TEXT_TESTING
std::vector<int32_t> text_frontend::test_unigram(const std::string & text) const {
    require(enabled_, "disabled Unigram");
    std::vector<int32_t> ids;
    unigram(text, ids);
    return ids;
}
std::string text_frontend::test_normalize(const std::string & text) const {
    require(enabled_, "disabled normalizer");
    return normalize(text);
}
std::vector<std::string> text_frontend::test_pretokens(const std::string & text) const {
    require(enabled_, "disabled pretokenizer");
    return pretokens(text);
}
#endif
} // namespace moedern_colbert
