// Minimal JSON-string extraction helpers for the HTTP server.
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

namespace crispembed_server {

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

// Extract the string value(s) of a top-level JSON key into `out`. Handles both
//   "key": "single string"   and   "key": ["a", "b", ...]
// with full JSON string escaping, so values containing ], \" or \\ no longer
// corrupt parsing or the input cardinality. Returns the number of strings added.
inline size_t json_extract_strings_escaped(const std::string & body, const char * key, std::vector<std::string> & out) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t k = body.find(needle);
    if (k == std::string::npos) return 0;
    size_t p = body.find(':', k + needle.size());
    if (p == std::string::npos) return 0;
    p++;
    while (p < body.size() && (body[p] == ' ' || body[p] == '\t' || body[p] == '\n' || body[p] == '\r')) p++;
    if (p >= body.size()) return 0;
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

} // namespace crispembed_server
