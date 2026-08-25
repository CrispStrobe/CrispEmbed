// core_json — shared JSON string escaping + extraction helpers.
//
// Centralized in core/ (like core_util/core_gguf) because the server AND the CLI
// each hand-rolled their own json_escape and they had already DIVERGED (server: 3
// chars; CLI: 5) — both echoing OCR/KIE/NER text, both missing control-char escapes.
// One correct, tested implementation, per the pcs.cpp 'two copies drift' lesson.
//
// The server hand-parses request bodies (no JSON library dependency). These two
// helpers replace the previous delimiter-scan approach, which mis-split any
// payload whose string values contained ']', an escaped quote (\"), or a
// backslash (\\) — corrupting the parsed input cardinality (issue #34,
// "returned 7 embeddings for 6 inputs"). They honor JSON string escaping so
// such values round-trip correctly.
//
// Header-only + inline so both server.cpp and the unit test share one impl.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace core_json {

// Decode a JSON string literal starting at body[i] (which must be the opening
// '"'). Honors JSON escape sequences (\" \\ \/ \n \t \r \b \f \uXXXX, including
// surrogate pairs). On success, appends the decoded text to `out` and sets `end`
// to the index just past the closing quote; returns false if there is no valid
// closing quote.
inline bool json_decode_string(const std::string & body, size_t i, std::string & out, size_t & end) {
    if (i >= body.size() || body[i] != '"') return false;
    out.clear();
    size_t j = i + 1;
    while (j < body.size()) {
        char c = body[j];
        if (c == '"') {
            end = j + 1;
            return true;
        }
        if (c == '\\') {
            if (j + 1 >= body.size()) return false;
            char e = body[j + 1];
            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'n':
                out += '\n';
                break;
            case 't':
                out += '\t';
                break;
            case 'r':
                out += '\r';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'u': {
                // \uXXXX — decode BMP code point (and surrogate pairs) to UTF-8.
                if (j + 5 >= body.size()) return false;
                auto hex4 = [&](size_t at, unsigned & cp) -> bool {
                    cp = 0;
                    for (size_t k = 0; k < 4; k++) {
                        char h = body[at + k];
                        cp <<= 4;
                        if (h >= '0' && h <= '9')
                            cp |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f')
                            cp |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F')
                            cp |= (unsigned)(h - 'A' + 10);
                        else
                            return false;
                    }
                    return true;
                };
                unsigned cp = 0;
                if (!hex4(j + 2, cp)) return false;
                j += 6;
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    // High surrogate — consume the following \uXXXX low surrogate.
                    unsigned lo = 0;
                    if (j + 5 < body.size() && body[j] == '\\' && body[j + 1] == 'u' && hex4(j + 2, lo) &&
                        lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        j += 6;
                    }
                }
                if (cp < 0x80) {
                    out += (char)cp;
                } else if (cp < 0x800) {
                    out += (char)(0xC0 | (cp >> 6));
                    out += (char)(0x80 | (cp & 0x3F));
                } else if (cp < 0x10000) {
                    out += (char)(0xE0 | (cp >> 12));
                    out += (char)(0x80 | ((cp >> 6) & 0x3F));
                    out += (char)(0x80 | (cp & 0x3F));
                } else {
                    out += (char)(0xF0 | (cp >> 18));
                    out += (char)(0x80 | ((cp >> 12) & 0x3F));
                    out += (char)(0x80 | ((cp >> 6) & 0x3F));
                    out += (char)(0x80 | (cp & 0x3F));
                }
                continue; // j already advanced past the escape
            }
            default:
                out += e;
                break;
            }
            j += 2;
        } else {
            out += c;
            j++;
        }
    }
    return false;
}

// Locate the VALUE of a top-level object key structurally, returning the index
// of the first non-whitespace value character (or npos).
//
// This replaces a plain body.find("\"key\""), which latches onto any occurrence
// of the token — including a decoy string VALUE that happens to equal the key
// name. A JSON string element like ["input"] is legal unescaped, so
//   {"labels":["input"],"other":["x","y"]}  asked for "input"
// made find() match the array element, skip to the NEXT colon ("other"), and
// return another key's values (or values for a key that is not even present).
//
// We only match "key" when it sits at object depth 1 in KEY position (right
// after '{' or a depth-1 ',') and is immediately followed by ':'. Strings are
// scanned with escape awareness so quotes/brackets inside values never confuse
// the depth/position tracking.
inline size_t json_find_key_value(const std::string & body, const char * key) {
    const std::string want(key);
    int depth = 0;
    bool expect_key = false; // at depth 1, is the next string a key?
    size_t i = 0;
    while (i < body.size()) {
        const char c = body[i];
        if (c == '"') {
            std::string s;
            size_t end = 0;
            if (!json_decode_string(body, i, s, end)) return std::string::npos; // malformed
            const bool is_key = (depth == 1 && expect_key);
            i = end;
            // Skip whitespace to see whether a ':' follows (confirms key position).
            size_t j = i;
            while (j < body.size() && (body[j] == ' ' || body[j] == '\t' || body[j] == '\n' || body[j] == '\r')) j++;
            if (is_key && j < body.size() && body[j] == ':') {
                expect_key = false; // a value follows this colon
                if (s == want) {
                    j++; // past ':'
                    while (j < body.size() && (body[j] == ' ' || body[j] == '\t' || body[j] == '\n' || body[j] == '\r'))
                        j++;
                    return j < body.size() ? j : std::string::npos;
                }
            }
            continue;
        }
        switch (c) {
        case '{':
            depth++;
            if (depth == 1) expect_key = true;
            break;
        case '[':
            depth++;
            break;
        case '}':
        case ']':
            depth--;
            break;
        case ',':
            if (depth == 1) expect_key = true;
            break;
        default:
            break;
        }
        i++;
    }
    return std::string::npos;
}

// Extract the string value(s) of a top-level JSON key into `out`. Handles both
//   "key": "single string"   and   "key": ["a", "b", ...]
// with full JSON string escaping, so values containing ], \" or \\ no longer
// corrupt parsing or the input cardinality. Returns the number of strings added.
inline size_t json_extract_strings_escaped(const std::string & body, const char * key, std::vector<std::string> & out) {
    size_t p = json_find_key_value(body, key);
    if (p == std::string::npos) return 0;
    const size_t before = out.size();
    if (body[p] == '[') {
        p++;
        while (p < body.size()) {
            // Advance to the next string or the closing bracket, skipping commas/space.
            while (p < body.size() && body[p] != '"' && body[p] != ']') p++;
            if (p >= body.size() || body[p] == ']') break;
            std::string s;
            size_t end = 0;
            if (!json_decode_string(body, p, s, end)) break;
            out.push_back(std::move(s));
            p = end;
        }
    } else if (body[p] == '"') {
        std::string s;
        size_t end = 0;
        if (json_decode_string(body, p, s, end)) out.push_back(std::move(s));
    }
    return out.size() - before;
}

// ---------------------------------------------------------------------------
// Legacy delimiter-scan parser (the pre-#34 behaviour), kept ONLY as an A/B /
// regression-bisection escape hatch behind CRISPEMBED_SERVER_LEGACY_JSON=1.
//
// It is KNOWN-BUGGY by construction and is retained deliberately (per the
// env-gate rule: never remove a gate — it is the bisection mechanism). It takes
// the first ']' even when that bracket sits inside a string value, and pairs
// bare '"' characters, so it mis-splits any payload containing ']', \" or \\.
// Do not "fix" it; its whole purpose is to reproduce the old behaviour.
// ---------------------------------------------------------------------------
inline size_t json_extract_strings_legacy(const std::string & body, const char * key, std::vector<std::string> & out) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t before = out.size();
    size_t pos = body.find(needle);
    if (pos == std::string::npos) return 0;
    size_t arr_start = body.find('[', pos);
    size_t str_start = body.find('"', pos + needle.size());
    if (arr_start != std::string::npos && (str_start == std::string::npos || arr_start < str_start)) {
        size_t arr_end = body.find(']', arr_start); // BUG (kept): first ']' wins, even inside a string
        if (arr_end == std::string::npos) return 0;
        const std::string arr = body.substr(arr_start + 1, arr_end - arr_start - 1);
        size_t i = 0;
        while (i < arr.size()) {
            size_t q1 = arr.find('"', i);
            if (q1 == std::string::npos) break;
            size_t q2 = arr.find('"', q1 + 1); // BUG (kept): ignores \" escapes
            if (q2 == std::string::npos) break;
            out.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
            i = q2 + 1;
        }
    } else if (str_start != std::string::npos) {
        size_t q2 = body.find('"', str_start + 1);
        if (q2 != std::string::npos) out.push_back(body.substr(str_start + 1, q2 - str_start - 1));
    }
    return out.size() - before;
}

// True when CRISPEMBED_SERVER_LEGACY_JSON=1 selects the legacy scan. Read once.
inline bool json_legacy_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("CRISPEMBED_SERVER_LEGACY_JSON");
        return e && e[0] == '1';
    }();
    return on;
}

// Gated entry point — every server endpoint routes through this, so one env var
// A/Bs the whole request-parsing surface without recompiling.
inline size_t json_extract_strings(const std::string & body, const char * key, std::vector<std::string> & out) {
    return json_legacy_enabled() ? json_extract_strings_legacy(body, key, out)
                                 : json_extract_strings_escaped(body, key, out);
}

// ---------------------------------------------------------------------------
// Structural value skipping + the Cohere/Jina `documents` shape (issue #51).
//
// json_extract_strings flattens EVERY string it meets inside the array, which is
// correct for `["a","b"]` but wrong for the object form the de-facto rerank API
// also accepts:
//     "documents": [{"text": "a"}, {"text": "b"}]
// there it would yield ["text","a","text","b"] — four documents, all garbage.
// So the object form needs an element-aware walk, which needs a value skipper.
// ---------------------------------------------------------------------------

// Skip exactly one JSON value beginning at body[i] (i must be its first
// character, not whitespace). Returns the index just past the value, or npos if
// the value is malformed/unterminated. String-aware at every depth, so a '}' or
// ']' inside a string never closes a container.
inline size_t json_skip_value(const std::string & body, size_t i) {
    if (i >= body.size()) return std::string::npos;
    const char c = body[i];
    if (c == '"') {
        std::string s;
        size_t end = 0;
        return json_decode_string(body, i, s, end) ? end : std::string::npos;
    }
    if (c == '{' || c == '[') {
        int depth = 0;
        size_t j = i;
        while (j < body.size()) {
            const char d = body[j];
            if (d == '"') {
                std::string s;
                size_t end = 0;
                if (!json_decode_string(body, j, s, end)) return std::string::npos;
                j = end;
                continue;
            }
            if (d == '{' || d == '[') depth++;
            if (d == '}' || d == ']') {
                depth--;
                if (depth == 0) return j + 1;
            }
            j++;
        }
        return std::string::npos;
    }
    // Number / true / false / null — run to the first structural delimiter.
    size_t j = i;
    while (j < body.size() && body[j] != ',' && body[j] != '}' && body[j] != ']') j++;
    return j > i ? j : std::string::npos;
}

// Extract rerank documents from `key`, accepting BOTH shapes the Cohere / Jina
// API allows (and the single-value degenerate forms):
//     "documents": ["a", "b"]              -> ["a", "b"]
//     "documents": [{"text":"a"}, {"text":"b"}] -> ["a", "b"]
//     "documents": "a"                     -> ["a"]
//     "documents": {"text":"a"}            -> ["a"]
// Only the `text` field of an object element is read (Cohere's `rank_fields`
// multi-field form is not supported). An object element WITHOUT a `text` field
// yields an empty string rather than being dropped: the response's `index`
// values must line up with the caller's array positions, and silently shrinking
// the list would shift every index after it. Returns the number appended.
//
// Routed through json_extract_strings for the plain-string case so the
// CRISPEMBED_SERVER_LEGACY_JSON A/B gate still covers it.
inline size_t json_extract_documents(const std::string & body, const char * key, std::vector<std::string> & out) {
    const size_t p = json_find_key_value(body, key);
    if (p == std::string::npos) return 0;

    // No object elements anywhere in the value -> the plain-string path, which
    // keeps the legacy gate in play for the shape that predates this helper.
    const size_t vend = json_skip_value(body, p);
    if (vend == std::string::npos) return 0;
    if (body[p] != '{' && body.find('{', p) >= vend) return json_extract_strings(body, key, out);

    const size_t before = out.size();
    auto push_object = [&](const std::string & obj) {
        std::vector<std::string> t;
        json_extract_strings_escaped(obj, "text", t);
        out.push_back(t.empty() ? std::string() : std::move(t.front()));
    };

    if (body[p] == '{') {
        push_object(body.substr(p, vend - p));
    } else if (body[p] == '"') {
        std::string s;
        size_t end = 0;
        if (json_decode_string(body, p, s, end)) out.push_back(std::move(s));
    } else if (body[p] == '[') {
        size_t q = p + 1;
        while (q < body.size()) {
            while (q < body.size() &&
                   (body[q] == ',' || body[q] == ' ' || body[q] == '\t' || body[q] == '\n' || body[q] == '\r'))
                q++;
            if (q >= body.size() || body[q] == ']') break;
            const size_t e = json_skip_value(body, q);
            if (e == std::string::npos) break;
            if (body[q] == '"') {
                std::string s;
                size_t end = 0;
                if (json_decode_string(body, q, s, end)) out.push_back(std::move(s));
            } else if (body[q] == '{') {
                push_object(body.substr(q, e - q));
            }
            // Anything else (a bare number/null element) is not a document; skip.
            q = e;
        }
    }
    return out.size() - before;
}

// ---------------------------------------------------------------------------
// Boolean extraction — same depth-1 structural lookup as the number/array
// readers, so a string VALUE equal to the key name cannot be mistaken for the
// key. Accepts JSON `true`/`false` and, leniently, the numeric 1/0 that some
// clients send. Anything else (including a quoted "true") returns `def`.
// ---------------------------------------------------------------------------
inline bool json_extract_bool(const std::string & body, const char * key, bool def) {
    const size_t p = json_find_key_value(body, key);
    if (p == std::string::npos) return def;
    if (body.compare(p, 4, "true") == 0) return true;
    if (body.compare(p, 5, "false") == 0) return false;
    if (body[p] == '1') return true;
    if (body[p] == '0') return false;
    return def;
}

// ---------------------------------------------------------------------------
// Scalar number extraction — the server hand-rolled ~14 of these as
// body.find("\"key\"") + atof/atoi, which carry the SAME key-decoy bug B2 fixed
// for arrays (a string value equal to the key name is matched as the key). Route
// them through the depth-1 finder so every JSON field read is location-correct.
// ---------------------------------------------------------------------------
inline double json_extract_number_structural(const std::string & body, const char * key, double def) {
    const size_t p = json_find_key_value(body, key);
    if (p == std::string::npos) return def;
    // A JSON number starts with '-', a digit, or (leniently) '+'/'.'; a string
    // ('"'), true/false/null, object/array here means "not a number" -> default.
    const char c = body[p];
    if (!(c == '-' || c == '+' || c == '.' || (c >= '0' && c <= '9'))) return def;
    char * end = nullptr;
    const double v = std::strtod(body.c_str() + p, &end);
    if (end == body.c_str() + p) return def; // no conversion
    return v;
}

// Legacy scalar parse (pre-fix): naive find of the key token, then the first
// number-looking run after it. Reproduces the old behaviour (incl. the decoy
// bug) for A/B behind CRISPEMBED_SERVER_LEGACY_JSON=1. Known-weak by design.
inline double json_extract_number_legacy(const std::string & body, const char * key, double def) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t k = body.find(needle);
    if (k == std::string::npos) return def;
    const size_t start = body.find_first_of("-0123456789.", k + needle.size());
    if (start == std::string::npos) return def;
    char * end = nullptr;
    const double v = std::strtod(body.c_str() + start, &end);
    if (end == body.c_str() + start) return def;
    return v;
}

// Gated entry point. Callers cast to float/int as before.
inline double json_extract_number(const std::string & body, const char * key, double def) {
    return json_legacy_enabled() ? json_extract_number_legacy(body, key, def)
                                 : json_extract_number_structural(body, key, def);
}

// ---------------------------------------------------------------------------
// Output escaping — the symmetric half of the input parser.
//
// The exact inverse of json_decode_string: escape ", \, and EVERY control
// character U+0000..U+001F. RFC 8259 forbids a raw control char inside a JSON
// string, so emitting a tab/CR/etc. unescaped produces output that strict
// parsers (Python json, Go encoding/json, JS JSON.parse) reject outright. OCR /
// KIE / OMR results routinely contain tabs and newlines (tables, kern/LilyPond
// structure), so this was a live break on the OCR echo endpoints.
//
// The short forms match what json_decode_string decodes, so
// json_decode_string(json_escape_strict(x)) == x for ANY byte string (bytes
// >= 0x80 pass through as UTF-8, which JSON permits). That round-trip is the
// property test.
// ---------------------------------------------------------------------------
inline std::string json_escape_strict(const std::string & s) {
    static const char * HEX = "0123456789abcdef";
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out += HEX[(c >> 4) & 0xF];
                out += HEX[c & 0xF];
            } else {
                out += (char)c;
            }
        }
    }
    return out;
}

// Legacy escaper (pre-fix): only ", \, \n — leaves \t, \r and other control
// chars RAW, which is invalid JSON. Kept behind the gate as the bisection
// escape hatch; known-buggy by construction, do not "fix".
inline std::string json_escape_legacy(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"')
            out += "\\\"";
        else if (c == '\\')
            out += "\\\\";
        else if (c == '\n')
            out += "\\n";
        else
            out += c;
    }
    return out;
}

// Gated entry point (shares CRISPEMBED_SERVER_LEGACY_JSON with the parser, so one
// switch reverts the whole JSON surface — input parse + key location + output
// escaping — to pre-fix behaviour for A/B).
inline std::string json_escape(const std::string & s) {
    return json_legacy_enabled() ? json_escape_legacy(s) : json_escape_strict(s);
}

} // namespace core_json
