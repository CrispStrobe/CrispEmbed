// Native port of spaCy 3.8.18 tokenizer/lex_attrs/StringStore semantics (MIT).
// Source: 43311999028032fd81bef77a75fdb4fa96803c98. Original notices are
// installed from tools/spacy/notices. Model data stays in authenticated resources.
#include "spacy_lexical.h"

#include "core/json_value.h"
#include "spacy_lexical_data.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <utility>

namespace crispembed {
namespace spacy {
namespace {

using Json = core_json_value::value;
constexpr std::size_t max_text_bytes = 1024 * 1024;
constexpr std::size_t max_tokens = 2048;

void require(bool ok, const char * message) {
    if (!ok) throw std::invalid_argument(message);
}

void check_cancel(const CancelCheck & cancelled) {
    if (cancelled && cancelled()) throw NumericCancelled();
}

struct Utf8 {
    std::vector<std::uint32_t> scalars;
    std::vector<std::size_t> bytes{ 0 };
    std::vector<std::size_t> utf16{ 0 };
};

Utf8 decode_utf8(std::string_view text) {
    Utf8 result;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i++]);
        std::uint32_t cp = lead;
        unsigned tails = 0;
        if (lead >= 0xC2 && lead <= 0xDF) {
            cp = lead & 0x1F;
            tails = 1;
        } else if (lead >= 0xE0 && lead <= 0xEF) {
            cp = lead & 0x0F;
            tails = 2;
        } else if (lead >= 0xF0 && lead <= 0xF4) {
            cp = lead & 0x07;
            tails = 3;
        } else {
            require(lead < 0x80, "Invalid UTF-8 leading byte");
        }
        require(text.size() - i >= tails, "Incomplete UTF-8 sequence");
        for (unsigned j = 0; j < tails; ++j) {
            const auto byte = static_cast<unsigned char>(text[i++]);
            require((byte & 0xC0) == 0x80, "Invalid UTF-8 continuation byte");
            cp = (cp << 6) | (byte & 0x3F);
        }
        require((tails != 1 || cp >= 0x80) && (tails != 2 || cp >= 0x800) && (tails != 3 || cp >= 0x10000) &&
                    cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF),
                "Noncanonical or nonscalar UTF-8 sequence");
        result.scalars.push_back(cp);
        result.bytes.push_back(i);
        result.utf16.push_back(result.utf16.back() + (cp > 0xFFFF ? 2 : 1));
    }
    return result;
}

std::string encode_utf8(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

template <std::size_t N> bool in_ranges(std::uint32_t cp, const std::uint32_t (&ranges)[N][2]) {
    const auto it = std::upper_bound(std::begin(ranges), std::end(ranges), cp,
                                     [](std::uint32_t value, const auto & pair) { return value < pair[0]; });
    return it != std::begin(ranges) && cp <= (*(it - 1))[1];
}

using Ranges = std::vector<std::pair<std::uint32_t, std::uint32_t>>;
bool in_ranges(std::uint32_t cp, const Ranges & ranges) {
    const auto it = std::upper_bound(ranges.begin(), ranges.end(), cp,
                                     [](std::uint32_t value, const auto & pair) { return value < pair.first; });
    return it != ranges.begin() && cp <= (it - 1)->second;
}

const Json & field(const Json & object, const char * name, Json::kind_t kind) {
    require(object.kind == Json::object_t, "Expected resource object");
    const auto * value = object.get(name);
    require(value && value->kind == kind, "Missing or mistyped lexical resource field");
    return *value;
}

void unique_finite_json(const Json & value) {
    if (value.kind == Json::object_t) {
        std::set<std::string> keys;
        for (const auto & item : value.o) {
            require(keys.insert(item.first).second, "Duplicate resource JSON field");
            unique_finite_json(item.second);
        }
    } else if (value.kind == Json::array_t) {
        for (const auto & item : value.a) unique_finite_json(item);
    } else if (value.kind == Json::float_t) {
        require(std::isfinite(value.f), "Nonfinite lexical resource value");
    } else if (value.kind == Json::string_t) {
        decode_utf8(value.s);
    }
}

std::uint64_t decimal_id(const std::string & text) {
    require(!text.empty() && (text.size() == 1 || text[0] != '0'), "Invalid decimal resource ID");
    std::uint64_t value = 0;
    for (const char c : text) {
        require(c >= '0' && c <= '9', "Invalid decimal resource ID");
        const unsigned digit = static_cast<unsigned>(c - '0');
        require(value <= (std::numeric_limits<std::uint64_t>::max() - digit) / 10, "Resource ID overflow");
        value = value * 10 + digit;
    }
    return value;
}

std::uint64_t murmur64(std::string_view text) {
    constexpr std::uint64_t m = 0xC6A4A7935BD1E995ULL;
    std::uint64_t h = 1 ^ (static_cast<std::uint64_t>(text.size()) * m);
    std::size_t at = 0;
    while (text.size() - at >= 8) {
        std::uint64_t k = 0;
        for (unsigned i = 0; i < 8; ++i) k |= std::uint64_t(static_cast<unsigned char>(text[at + i])) << (8 * i);
        k *= m;
        k ^= k >> 47;
        k *= m;
        h ^= k;
        h *= m;
        at += 8;
    }
    if (at != text.size()) {
        for (unsigned shift = 0; at < text.size(); ++at, shift += 8)
            h ^= std::uint64_t(static_cast<unsigned char>(text[at])) << shift;
        h *= m;
    }
    h ^= h >> 47;
    h *= m;
    h ^= h >> 47;
    return h;
}

template <std::size_t N> std::string regex_ranges(const std::uint32_t (&ranges)[N][2]) {
    std::ostringstream out;
    out << std::hex;
    for (const auto & pair : ranges) {
        out << "\\x{" << pair[0] << '}';
        if (pair[0] != pair[1]) out << "-\\x{" << pair[1] << '}';
    }
    return out.str();
}

std::string python_regex(const std::string & source) {
    // These are the only shorthands in the authenticated German patterns.
    // Expand from pinned CPython data rather than PCRE2's Unicode database.
    static const std::string word = regex_ranges(spacy_lexical_data::word);
    static const std::string digit = regex_ranges(spacy_lexical_data::decimal);
    static const std::string space = regex_ranges(spacy_lexical_data::space);
    std::string out;
    bool in_class = false;
    for (std::size_t i = 0; i < source.size(); ++i) {
        if (source.compare(i, 4, "(?u)") == 0) {
            i += 3;
            continue;
        }
        const char c = source[i];
        if (c == '[') in_class = true;
        if (c == ']') in_class = false;
        if (c != '\\') {
            out += c;
            continue;
        }
        require(i + 1 < source.size(), "Trailing regex escape");
        const char e = source[++i];
        if (e == 'u' || e == 'U') {
            const std::size_t count = e == 'u' ? 4 : 8;
            require(source.size() - i - 1 >= count, "Truncated Unicode regex escape");
            for (std::size_t j = i + 1; j <= i + count; ++j)
                require((source[j] >= '0' && source[j] <= '9') || (source[j] >= 'a' && source[j] <= 'f') ||
                            (source[j] >= 'A' && source[j] <= 'F'),
                        "Invalid Unicode regex escape");
            out += "\\x{" + source.substr(i + 1, count) + "}";
            i += count;
        } else if (e == 'w' || e == 'd' || e == 's' || e == 'S') {
            const auto & ranges = e == 'w' ? word : e == 'd' ? digit : space;
            require(e != 'S' || !in_class, "Unsupported complemented shorthand inside regex class");
            if (!in_class) out += e == 'S' ? "[^" : "[";
            out += ranges;
            if (!in_class) out += ']';
        } else {
            out += '\\';
            out += e;
        }
    }
    return out;
}

struct Match {
    std::size_t start = 0, end = 0;
    bool found = false;
};

class Regex {
public:
    explicit Regex(const std::string & source) {
        const std::string translated = python_regex(source);
        int code = 0;
        PCRE2_SIZE offset = 0;
        regex_.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(translated.data()), translated.size(), PCRE2_UTF, &code,
                                   &offset, nullptr));
        if (!regex_) {
            PCRE2_UCHAR buffer[256]{};
            pcre2_get_error_message(code, buffer, sizeof(buffer));
            throw std::invalid_argument("Unsupported tokenizer regex at " + std::to_string(offset) + ": " +
                                        reinterpret_cast<const char *>(buffer));
        }
    }

    Match search(std::string_view text, std::size_t offset = 0, std::uint32_t flags = 0) const {
        std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> data(
            pcre2_match_data_create_from_pattern(regex_.get(), nullptr), pcre2_match_data_free);
        std::unique_ptr<pcre2_match_context, decltype(&pcre2_match_context_free)> context(
            pcre2_match_context_create(nullptr), pcre2_match_context_free);
        if (!data || !context) throw std::bad_alloc();
        pcre2_set_match_limit(context.get(), 1000000);
        pcre2_set_depth_limit(context.get(), 1000);
        pcre2_set_heap_limit(context.get(), 8192); // KiB; per match, not model storage.
        const int result = pcre2_match(regex_.get(), reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset,
                                       flags, data.get(), context.get());
        if (result == PCRE2_ERROR_NOMATCH) return {};
        if (result < 0) throw std::runtime_error("Tokenizer regex failed its bounded match: " + std::to_string(result));
        const auto * bounds = pcre2_get_ovector_pointer(data.get());
        return { bounds[0], bounds[1], true };
    }

    std::vector<Match> find_all(std::string_view text, const CancelCheck & cancelled) const {
        std::vector<Match> matches;
        std::size_t offset = 0;
        std::uint32_t flags = 0;
        while (offset <= text.size()) {
            check_cancel(cancelled);
            const auto match = search(text, offset, flags);
            if (!match.found) {
                if (!flags || offset == text.size()) break;
                // Python finditer permits a nonempty alternative immediately
                // after a zero-width match. Only advance if that retry fails.
                ++offset;
                while (offset < text.size() && (static_cast<unsigned char>(text[offset]) & 0xC0) == 0x80) ++offset;
                flags = 0;
                continue;
            }
            matches.push_back(match);
            require(matches.size() <= max_tokens * 4, "Tokenizer infix match count exceeds bound");
            offset = match.end;
            flags = match.start == match.end ? PCRE2_NOTEMPTY_ATSTART | PCRE2_ANCHORED : 0;
        }
        return matches;
    }

private:
    std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> regex_{ nullptr, pcre2_code_free };
};

struct ExceptionPiece {
    std::string orth, norm;
    bool has_norm = false;
};

struct Piece {
    std::size_t start = 0, end = 0;
    bool space_after = false;
    const ExceptionPiece * exception = nullptr;
};

struct SpecialPattern {
    std::vector<std::string> words;
};

} // namespace

std::vector<std::size_t> utf8_boundaries(std::string_view text) {
    return decode_utf8(text).bytes;
}

std::uint64_t spacy_string_id(std::string_view text) {
    static const std::map<std::string_view, std::uint64_t> symbols(std::begin(spacy_lexical_data::symbols),
                                                                   std::end(spacy_lexical_data::symbols));
    const auto found = symbols.find(text);
    return found == symbols.end() ? murmur64(text) : found->second;
}

struct LexicalModel::Impl {
    Regex prefix, suffix, infix, url;
    std::map<std::string, std::vector<ExceptionPiece>, std::less<>> exceptions;
    std::map<std::uint64_t, std::string> hash_norms;
    std::map<std::string, std::string, std::less<>> text_norms;
    std::map<std::uint32_t, std::string> lower;
    Ranges alpha, digit, upper, space;
    std::map<std::string, std::vector<SpecialPattern>, std::less<>> specials_by_first;

    explicit Impl(const Json & root)
        : prefix(field(field(root, "tokenizer", Json::object_t), "prefix_search", Json::string_t).s),
          suffix(field(field(root, "tokenizer", Json::object_t), "suffix_search", Json::string_t).s),
          infix(field(field(root, "tokenizer", Json::object_t), "infix_finditer", Json::string_t).s),
          url(field(field(root, "tokenizer", Json::object_t), "url_match", Json::string_t).s) {
        // Format 1 is the closed German de_core_news_sm profile (exact inventory
        // counts). Format 2 (models/convert-spacy-pipeline-to-gguf.py) admits
        // other spaCy 3.8 pipelines with the same tokenizer/NORM structure and
        // generic engineering bounds instead of fixed counts.
        const auto format = field(root, "format_version", Json::int_t).i;
        const auto & language = field(root, "language", Json::string_t).s;
        require((format == 1 && language == "de") || (format == 2 && (language == "en" || language == "fr")),
                "Unsupported lexical resource profile");
        const bool closed = format == 1;
        const auto & source = field(root, "lexical_semantics_source", Json::object_t);
        require(field(source, "revision", Json::string_t).s == "43311999028032fd81bef77a75fdb4fa96803c98",
                "Unsupported lexical source revision");
        const auto & tokenizer = field(root, "tokenizer", Json::object_t);
        field(tokenizer, "token_match", Json::null_t);
        require(field(tokenizer, "faster_heuristics", Json::bool_t).b, "Unsupported tokenizer special-case policy");
        const auto & cases = field(tokenizer, "exceptions", Json::object_t);
        require(closed ? cases.o.size() == 417 : !cases.o.empty() && cases.o.size() <= 20000,
                "Changed tokenizer exception inventory");
        for (const auto & item : cases.o) {
            require(!item.first.empty() && item.first.size() <= 16384 && item.second.kind == Json::array_t &&
                        !item.second.a.empty() && item.second.a.size() <= 3,
                    "Invalid tokenizer exception");
            auto & pieces = exceptions[item.first];
            std::string combined;
            for (const auto & part : item.second.a) {
                ExceptionPiece piece;
                piece.orth = field(part, "ORTH", Json::string_t).s;
                require(!piece.orth.empty(), "Empty tokenizer exception piece");
                if (const auto * norm = part.get("NORM")) {
                    require(norm->kind == Json::string_t && part.o.size() == 2, "Invalid exception NORM");
                    piece.norm = norm->s;
                    piece.has_norm = true;
                } else {
                    require(part.o.size() == 1, "Unsupported tokenizer exception attribute");
                }
                combined += piece.orth;
                pieces.push_back(std::move(piece));
            }
            require(combined == item.first, "Exception does not reconstruct original source");
        }
        const auto & norms = field(root, "norm_lookup_precedence", Json::array_t).a;
        require(norms.size() == 2 && field(norms[0], "key_kind", Json::string_t).s == "hash_uint64_decimal" &&
                    field(norms[1], "key_kind", Json::string_t).s == "text",
                "Unsupported normalization precedence");
        for (const auto & item : field(norms[0], "entries", Json::object_t).o) {
            require(item.second.kind == Json::string_t, "Invalid normalization value");
            hash_norms.emplace(decimal_id(item.first), item.second.s);
        }
        for (const auto & item : field(norms[1], "entries", Json::object_t).o) {
            require(item.second.kind == Json::string_t, "Invalid normalization value");
            text_norms.emplace(item.first, item.second.s);
        }
        require(closed ? hash_norms.size() == 2 && text_norms.size() == 45
                       : hash_norms.size() <= 1000000 && text_norms.size() <= 100000,
                "Changed normalization inventory");
        const auto & unicode = field(root, "unicode", Json::object_t);
        require(field(unicode, "unicode_version", Json::string_t).s == "15.0.0" &&
                    field(unicode, "normalization", Json::string_t).s == "none; preserve original UTF-8 source" &&
                    field(unicode, "lower_semantics", Json::string_t).s ==
                        "CPython str.lower; contextual final sigma must also be implemented",
                "Unsupported Unicode profile");
        const auto & ranges = field(unicode, "ranges_inclusive", Json::object_t);
        auto read_ranges = [&](const char * name, Ranges & result, std::size_t count) {
            const auto & values = field(ranges, name, Json::array_t).a;
            require(values.size() == count, "Changed Unicode range count");
            for (const auto & bounds : values) {
                require(bounds.kind == Json::array_t && bounds.a.size() == 2 && bounds.a[0].kind == Json::int_t &&
                            bounds.a[1].kind == Json::int_t && bounds.a[0].i >= 0 && bounds.a[0].i <= bounds.a[1].i &&
                            bounds.a[1].i <= 0x10FFFF && (bounds.a[1].i < 0xD800 || bounds.a[0].i > 0xDFFF),
                        "Invalid Unicode range");
                require(result.empty() || result.back().second < bounds.a[0].i, "Unsorted Unicode ranges");
                result.emplace_back(static_cast<std::uint32_t>(bounds.a[0].i),
                                    static_cast<std::uint32_t>(bounds.a[1].i));
            }
        };
        read_ranges("isalpha", alpha, 659);
        read_ranges("isdigit", digit, 83);
        read_ranges("isupper", upper, 651);
        read_ranges("isspace", space, 10);
        for (const auto & item : field(unicode, "scalar_lower", Json::object_t).o) {
            const auto cp = decimal_id(item.first);
            require(cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF) && item.second.kind == Json::string_t,
                    "Invalid Unicode lowercase entry");
            lower.emplace(static_cast<std::uint32_t>(cp), item.second.s);
        }
        require(lower.size() == 1433, "Changed Unicode lowercase count");
        for (const auto & item : exceptions) {
            const auto & surface = item.first;
            if (!prefix.search(surface).found && !suffix.search(surface).found && !infix.search(surface).found &&
                surface.find(' ') == std::string::npos)
                continue;
            const auto split = tokenize_affixes(surface, decode_utf8(surface), false, {});
            SpecialPattern pattern;
            for (const auto & piece : split)
                pattern.words.emplace_back(surface.substr(piece.start, piece.end - piece.start));
            if (!pattern.words.empty()) specials_by_first[pattern.words.front()].push_back(std::move(pattern));
        }
    }

    void push(std::vector<Piece> & out, Piece piece) const {
        if (piece.end == piece.start) return;
        require(out.size() < max_tokens, "spaCy token count exceeds 2048; input was not truncated");
        out.push_back(piece);
    }

    bool append_special(std::string_view text, std::size_t start, std::size_t end, std::vector<Piece> & out,
                        bool with_special) const {
        if (!with_special) return false;
        const auto found = exceptions.find(text.substr(start, end - start));
        if (found == exceptions.end()) return false;
        for (const auto & entry : found->second) {
            push(out, { start, start + entry.orth.size(), false, &entry });
            start += entry.orth.size();
        }
        require(start == end, "Internal exception span mismatch");
        return true;
    }

    void split_span(std::string_view text, std::size_t start, std::size_t end, bool with_special,
                    std::vector<Piece> & out, const CancelCheck & cancelled) const {
        if (append_special(text, start, end, out, with_special)) return;
        std::vector<Piece> before, after;
        while (start < end) {
            check_cancel(cancelled);
            const auto current = text.substr(start, end - start);
            if (with_special && exceptions.find(current) != exceptions.end()) break;
            const auto pre = prefix.search(current);
            const auto pre_len = pre.found ? pre.end - pre.start : 0;
            if (pre_len && start + pre_len < end && with_special &&
                exceptions.find(text.substr(start + pre_len, end - start - pre_len)) != exceptions.end()) {
                before.push_back({ start, start + pre_len, false, nullptr });
                start += pre_len;
                break;
            }
            const auto suf = suffix.search(text.substr(start + pre_len, end - start - pre_len));
            const auto suf_len = suf.found ? suf.end - suf.start : 0;
            if (suf_len && start < end - suf_len && with_special &&
                exceptions.find(text.substr(start, end - start - suf_len)) != exceptions.end()) {
                after.push_back({ end - suf_len, end, false, nullptr });
                end -= suf_len;
                break;
            }
            if (!pre_len && !suf_len) break;
            if (pre_len) {
                before.push_back({ start, start + pre_len, false, nullptr });
                start += pre_len;
            }
            if (suf_len) {
                after.push_back({ end - suf_len, end, false, nullptr });
                end -= suf_len;
            }
            require(before.size() + after.size() + out.size() <= max_tokens, "spaCy affix count exceeds bound");
        }
        for (const auto & piece : before) push(out, piece);
        if (start < end && !append_special(text, start, end, out, with_special)) {
            const auto current = text.substr(start, end - start);
            const auto url_match = url.search(current);
            if (url_match.found && url_match.start == 0) {
                push(out, { start, end, false, nullptr });
            } else {
                std::size_t offset = 0;
                for (const auto & match : infix.find_all(current, cancelled)) {
                    check_cancel(cancelled);
                    if (match.start == 0) continue;
                    if (match.start != offset) push(out, { start + offset, start + match.start, false, nullptr });
                    if (match.start != match.end) push(out, { start + match.start, start + match.end, false, nullptr });
                    offset = match.end;
                }
                if (offset < current.size()) push(out, { start + offset, end, false, nullptr });
            }
        }
        for (auto it = after.rbegin(); it != after.rend(); ++it) push(out, *it);
    }

    std::vector<Piece> tokenize_affixes(std::string_view text, const Utf8 & decoded, bool with_special,
                                        const CancelCheck & cancelled) const {
        std::vector<Piece> out;
        if (decoded.scalars.empty()) return out;
        bool in_ws = in_ranges(decoded.scalars[0], space);
        std::size_t start = 0;
        for (std::size_t i = 0; i < decoded.scalars.size(); ++i) {
            if ((i & 255) == 0) check_cancel(cancelled);
            const auto cp = decoded.scalars[i];
            if (in_ranges(cp, space) == in_ws) continue;
            const auto at = decoded.bytes[i];
            if (start < at) split_span(text, start, at, with_special, out, cancelled);
            if (cp == ' ') {
                require(!out.empty(), "Invalid tokenizer whitespace transition");
                out.back().space_after = true;
                start = at + 1;
            } else {
                start = at;
            }
            in_ws = !in_ws;
        }
        if (start < text.size()) {
            split_span(text, start, text.size(), with_special, out, cancelled);
            require(!out.empty(), "Tokenizer lost final span");
            out.back().space_after = text.back() == ' ' && !in_ws;
        }
        return out;
    }

    void apply_specials(std::string_view text, std::vector<Piece> & pieces, const CancelCheck & cancelled) const {
        struct Span {
            std::size_t start, end;
        };
        std::vector<Span> matches;
        for (std::size_t i = 0; i < pieces.size(); ++i) {
            check_cancel(cancelled);
            const auto & piece = pieces[i];
            const auto found = specials_by_first.find(text.substr(piece.start, piece.end - piece.start));
            if (found == specials_by_first.end()) continue;
            for (const auto & pattern : found->second) {
                if (pattern.words.size() > pieces.size() - i) continue;
                bool equal = true;
                for (std::size_t j = 0; j < pattern.words.size(); ++j) {
                    const auto & other = pieces[i + j];
                    if (pattern.words[j] != text.substr(other.start, other.end - other.start)) {
                        equal = false;
                        break;
                    }
                }
                if (equal) matches.push_back({ i, i + pattern.words.size() });
            }
        }
        std::sort(matches.begin(), matches.end(), [](const Span & a, const Span & b) {
            return a.end - a.start != b.end - b.start ? a.end - a.start > b.end - b.start : a.start < b.start;
        });
        std::vector<bool> seen(pieces.size(), false);
        std::map<std::size_t, std::size_t> selected;
        for (const auto & match : matches) {
            if (!seen[match.start] && !seen[match.end - 1]) selected.emplace(match.start, match.end);
            for (std::size_t i = match.start; i < match.end; ++i) seen[i] = true;
        }
        std::vector<Piece> retokenized;
        for (std::size_t i = 0; i < pieces.size();) {
            const auto found = selected.find(i);
            if (found == selected.end()) {
                push(retokenized, pieces[i++]);
                continue;
            }
            const auto end = found->second;
            if (append_special(text, pieces[i].start, pieces[end - 1].end, retokenized, true)) {
                retokenized.back().space_after = pieces[end - 1].space_after;
            } else {
                for (std::size_t j = i; j < end; ++j) push(retokenized, pieces[j]);
            }
            i = end;
        }
        pieces = std::move(retokenized);
    }

    std::string lowered(const Utf8 & decoded, std::size_t start, std::size_t end) const {
        std::string result;
        for (std::size_t i = start; i < end; ++i) {
            const auto cp = decoded.scalars[i];
            if (cp == 0x3A3) {
                std::size_t before = i, after = i + 1;
                while (before > start && in_ranges(decoded.scalars[before - 1], spacy_lexical_data::sigma_ignorable))
                    --before;
                while (after < end && in_ranges(decoded.scalars[after], spacy_lexical_data::sigma_ignorable)) ++after;
                const bool final =
                    before > start && in_ranges(decoded.scalars[before - 1], spacy_lexical_data::sigma_cased) &&
                    (after == end || !in_ranges(decoded.scalars[after], spacy_lexical_data::sigma_cased));
                result += encode_utf8(final ? 0x3C2 : 0x3C3);
            } else {
                const auto found = lower.find(cp);
                result += found == lower.end() ? encode_utf8(cp) : found->second;
            }
        }
        return result;
    }

    LexicalToken describe(std::string_view text, const Utf8 & decoded, const Piece & piece) const {
        LexicalToken token;
        token.byte_start = piece.start;
        token.byte_end = piece.end;
        const auto begin = std::lower_bound(decoded.bytes.begin(), decoded.bytes.end(), piece.start);
        const auto end = std::lower_bound(decoded.bytes.begin(), decoded.bytes.end(), piece.end);
        require(begin != decoded.bytes.end() && end != decoded.bytes.end() && *begin == piece.start &&
                    *end == piece.end,
                "Tokenizer span is not aligned to original UTF-8 scalars");
        token.scalar_start = static_cast<std::size_t>(begin - decoded.bytes.begin());
        token.scalar_end = static_cast<std::size_t>(end - decoded.bytes.begin());
        token.utf16_start = decoded.utf16[token.scalar_start];
        token.utf16_end = decoded.utf16[token.scalar_end];
        token.text = std::string(text.substr(piece.start, piece.end - piece.start));
        token.orth_id = spacy_string_id(token.text);
        token.space_after = piece.space_after;
        token.is_space = true;
        for (std::size_t i = token.scalar_start; i < token.scalar_end; ++i)
            token.is_space = token.is_space && in_ranges(decoded.scalars[i], space);
        if (piece.exception && piece.exception->has_norm) {
            token.norm = piece.exception->norm;
        } else if (const auto found = hash_norms.find(token.orth_id); found != hash_norms.end()) {
            token.norm = found->second;
        } else if (const auto found = text_norms.find(token.text); found != text_norms.end()) {
            token.norm = found->second;
        } else {
            token.norm = lowered(decoded, token.scalar_start, token.scalar_end);
        }
        token.prefix = std::string(text.substr(piece.start, decoded.bytes[token.scalar_start + 1] - piece.start));
        const auto suffix_start =
            decoded.bytes[std::max(token.scalar_start, token.scalar_end >= 3 ? token.scalar_end - 3 : 0)];
        token.suffix = std::string(text.substr(suffix_start, piece.end - suffix_start));
        if (token.scalar_end - token.scalar_start >= 100) {
            token.shape = "LONG";
        } else {
            std::uint32_t previous = 0xFFFFFFFF;
            unsigned repeated = 0;
            for (std::size_t i = token.scalar_start; i < token.scalar_end; ++i) {
                const auto cp = decoded.scalars[i];
                const auto shape = in_ranges(cp, alpha)
                                       ? (in_ranges(cp, upper) ? std::uint32_t('X') : std::uint32_t('x'))
                                   : in_ranges(cp, digit) ? std::uint32_t('d')
                                                          : cp;
                if (shape == previous)
                    ++repeated;
                else {
                    previous = shape;
                    repeated = 0;
                }
                if (repeated < 4) token.shape += encode_utf8(shape);
            }
        }
        token.features = {
            spacy_string_id(token.norm),  spacy_string_id(token.prefix),   spacy_string_id(token.suffix),
            spacy_string_id(token.shape), token.space_after ? 1ULL : 0ULL, token.is_space ? 1ULL : 0ULL
        };
        return token;
    }
};

struct PythonRegex::Impl {
    Regex regex;
    explicit Impl(const std::string & pattern) : regex(pattern) {}
};

PythonRegex::PythonRegex(const std::string & pattern) : impl_(std::make_unique<Impl>(pattern)) {}
PythonRegex::~PythonRegex() = default;
PythonRegex::PythonRegex(PythonRegex &&) noexcept = default;
PythonRegex & PythonRegex::operator=(PythonRegex &&) noexcept = default;

bool PythonRegex::search(std::string_view text) const {
    (void)decode_utf8(text);
    return impl_->regex.search(text).found;
}

std::string LexicalModel::python_lower(std::string_view text) const {
    require(impl_ != nullptr, "Moved-from lexical model");
    const auto decoded = decode_utf8(text);
    return impl_->lowered(decoded, 0, decoded.scalars.size());
}

bool LexicalModel::python_isalpha(std::string_view text) const {
    require(impl_ != nullptr, "Moved-from lexical model");
    const auto decoded = decode_utf8(text);
    if (decoded.scalars.empty()) return false;
    for (const auto cp : decoded.scalars)
        if (!in_ranges(cp, impl_->alpha)) return false;
    return true;
}

LexicalModel::LexicalModel(const std::string & authenticated_resource_json) {
    require(authenticated_resource_json.size() <= 32 * 1024 * 1024, "Lexical resource file exceeds bound");
    const auto root = core_json_value::parse(authenticated_resource_json);
    unique_finite_json(root);
    impl_ = std::make_unique<Impl>(root);
}
LexicalModel::~LexicalModel() = default;
LexicalModel::LexicalModel(LexicalModel &&) noexcept = default;
LexicalModel & LexicalModel::operator=(LexicalModel &&) noexcept = default;

std::vector<LexicalToken> LexicalModel::tokenize(std::string_view text, const CancelCheck & cancelled) const {
    require(impl_ != nullptr, "Moved-from lexical model");
    require(text.size() <= max_text_bytes, "spaCy text exceeds 1 MiB; input was not truncated");
    check_cancel(cancelled);
    const auto decoded = decode_utf8(text);
    auto pieces = impl_->tokenize_affixes(text, decoded, true, cancelled);
    impl_->apply_specials(text, pieces, cancelled);
    std::vector<LexicalToken> tokens;
    tokens.reserve(pieces.size());
    std::size_t expected = 0;
    for (const auto & piece : pieces) {
        check_cancel(cancelled);
        require(piece.start == expected && piece.end <= text.size(), "Tokenizer lost original source coverage");
        expected = piece.end + (piece.space_after ? 1 : 0);
        require(expected <= text.size() && (!piece.space_after || text[piece.end] == ' '),
                "Tokenizer changed original whitespace");
        tokens.push_back(impl_->describe(text, decoded, piece));
    }
    require(expected == text.size(), "Tokenizer lost trailing source coverage");
    return tokens;
}

} // namespace spacy
} // namespace crispembed
