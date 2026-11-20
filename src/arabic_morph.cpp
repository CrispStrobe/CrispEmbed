// Native port of the camel_tools morphological analyzer for ALMOR databases.
//
// Ported from camel_tools 1.6.0 (https://github.com/CAMeL-Lab/camel_tools):
// camel_tools/morphology/database.py (MorphologyDB, analysis flag only),
// camel_tools/morphology/analyzer.py (Analyzer.analyze and its backoff
// paths), camel_tools/morphology/utils.py (merge_features, the rewrite_*
// helpers, strip_lex, simple_ar_to_caphi), camel_tools/utils/dediac.py
// (dediac_ar) and camel_tools/utils/charsets.py (AR_* and the Unicode
// punctuation/symbol sets). The port keeps the Python control flow and order
// of operations so analyses, feature values and dict insertion order match.
//
// The ported camel_tools code is distributed under the MIT License:
//
//   MIT License
//
//   Copyright 2018-2026 New York University Abu Dhabi
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit
//   persons to whom the Software is furnished to do so, subject to the
//   following conditions:
//
//   The above copyright notice and this permission notice shall be included
//   in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Port notes (deliberate, documented differences):
// - merge_features appends keys from Python frozensets (_JOIN_FEATS,
//   _CONCAT_FEATS, _CONCAT_FEATS_NONE, _LOGPROB_FEATS, ...) when the stem
//   lacks them. Their relative order follows CPython's string hashing and so
//   changes with PYTHONHASHSEED; this port appends them in the order they are
//   written in camel_tools' source. Every other key position is identical.
// - The loader is stricter than camel_tools: every section marker must be
//   present, sizes are bounded, and *_logprob values must be plain decimals
//   (camel_tools would call float() on them at analysis time).
// - camel_tools raises (KeyError, re.error) for some database shapes and for
//   backoff stems that are invalid re.sub templates; those cases return
//   CRISPEMBED_AR_MORPH_INPUT_ERROR instead of an analysis list.

#include "arabic_morph.h"

#include "arabic_morph_tables.h"
#include "core/env_gate.h"
#include "core/json.h"
#include "core/json_value.h"
#include "core/py_str.h"
#include "core/sha256.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using u32 = std::u32string;
using py_str::isspace;

constexpr size_t k_max_db_bytes = size_t(512) << 20;
constexpr size_t k_max_manifest_bytes = size_t(64) << 10;
constexpr size_t k_max_aux_file_bytes = size_t(1) << 20;
constexpr size_t k_max_entry_feats = 4096;
constexpr size_t k_max_entries = size_t(1) << 24;
constexpr size_t k_max_form_cps = 256;

struct data_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct input_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------
// UTF-8 helpers over validated text.

bool valid_utf8(std::string_view s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            ++i;
            continue;
        }
        int k;
        uint32_t cp;
        if (c >= 0xC2 && c <= 0xDF) {
            k = 1;
            cp = c & 0x1F;
        } else if (c >= 0xE0 && c <= 0xEF) {
            k = 2;
            cp = c & 0x0F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            k = 3;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + size_t(k) >= n) return false;
        for (int j = 1; j <= k; ++j) {
            const unsigned char d = (unsigned char)s[i + size_t(j)];
            if ((d & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((k == 2 && cp < 0x800) || (k == 3 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF))
            return false;
        i += size_t(k) + 1;
    }
    return true;
}

// Decodes the code point at s[i] (valid UTF-8) and advances i.
char32_t next_cp(std::string_view s, size_t & i) {
    const unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        ++i;
        return c;
    }
    int k = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
    uint32_t cp = c & (k == 3 ? 0x07 : k == 2 ? 0x0F : 0x1F);
    for (int j = 1; j <= k; ++j) cp = (cp << 6) | ((unsigned char)s[i + size_t(j)] & 0x3F);
    i += size_t(k) + 1;
    return char32_t(cp);
}

size_t cp_count(std::string_view s) {
    size_t n = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) ++n;
    return n;
}

// str.strip() with no arguments on valid UTF-8.
std::string_view py_strip(std::string_view s) {
    size_t a = 0;
    while (a < s.size()) {
        size_t j = a;
        if (!isspace(next_cp(s, j))) break;
        a = j;
    }
    size_t b = s.size();
    while (b > a) {
        size_t start = b - 1;
        while (start > a && ((unsigned char)s[start] & 0xC0) == 0x80) --start;
        size_t j = start;
        if (!isspace(next_cp(s, j))) break;
        b = start;
    }
    return s.substr(a, b - a);
}

// str.split(' ')
std::vector<std::string_view> split_char(std::string_view s, char sep) {
    std::vector<std::string_view> out;
    size_t start = 0;
    for (;;) {
        const size_t p = s.find(sep, start);
        if (p == std::string_view::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, p - start));
        start = p + 1;
    }
}

// str.split() with no arguments (runs of Unicode whitespace).
std::vector<std::string_view> split_ws(std::string_view s) {
    std::vector<std::string_view> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        if (isspace(next_cp(s, j))) {
            i = j;
            continue;
        }
        const size_t start = i;
        while (i < s.size()) {
            j = i;
            if (isspace(next_cp(s, j))) break;
            i = j;
        }
        out.push_back(s.substr(start, i - start));
    }
    return out;
}

void append_u32(std::string & out, const u32 & s) {
    for (char32_t c : s) py_str::append_utf8(out, c);
}

std::string to_utf8(const u32 & s) {
    std::string out;
    out.reserve(s.size() * 2);
    append_u32(out, s);
    return out;
}

u32 from_utf8_valid(std::string_view s) {
    u32 out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) out.push_back(next_cp(s, i));
    return out;
}

// ---------------------------------------------------------------------------
// camel_tools.utils.charsets / dediac / analyzer predicates.

bool is_ar_diac(char32_t c) {
    return (c >= 0x064B && c <= 0x0652) || c == 0x0670;
}

bool is_ar_letter(char32_t c) {
    return (c >= 0x0621 && c <= 0x063A) || (c >= 0x0640 && c <= 0x064A) || c == 0x0671 || c == 0x067E || c == 0x0686 ||
           c == 0x06A4 || c == 0x06AF;
}

bool is_punct_symbol(char32_t c) {
    using arabic_morph_tables::k_punct_symbol;
    size_t lo = 0, hi = sizeof(k_punct_symbol) / sizeof(k_punct_symbol[0]);
    const uint32_t cp = uint32_t(c);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < k_punct_symbol[mid].lo)
            hi = mid;
        else if (cp > k_punct_symbol[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

u32 dediac_ar(const u32 & s) {
    u32 out;
    out.reserve(s.size());
    for (char32_t c : s)
        if (!is_ar_diac(c)) out.push_back(c);
    return out;
}

std::string dediac_ar_utf8(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const size_t start = i;
        const char32_t c = next_cp(s, i);
        if (!is_ar_diac(c)) out.append(s.data() + start, i - start);
    }
    return out;
}

// DEFAULT_NORMALIZE_MAP
u32 normalize(const u32 & s) {
    u32 out;
    out.reserve(s.size());
    for (char32_t c : s) {
        switch (c) {
        case 0x0625:
        case 0x0623:
        case 0x0622:
        case 0x0671:
            out.push_back(0x0627);
            break;
        case 0x0649:
            out.push_back(0x064A);
            break;
        case 0x0629:
            out.push_back(0x0647);
            break;
        case 0x0640:
            break;
        default:
            out.push_back(c);
        }
    }
    return out;
}

// _IS_DIGIT_RE = '^.*[0-9٠-٩]+.*$' with re.match: '.' excludes only
// '\n' and '$' also matches before a final '\n'.
bool is_digit_word(const u32 & w) {
    size_t limit = w.size();
    if (limit && w[limit - 1] == U'\n') --limit;
    bool digit = false;
    for (size_t i = 0; i < limit; ++i) {
        if (w[i] == U'\n') return false;
        if ((w[i] >= U'0' && w[i] <= U'9') || (w[i] >= 0x0660 && w[i] <= 0x0669)) digit = true;
    }
    return digit;
}

// '^[...]+$' anchored classes: every character but an optional final '\n'.
template <typename Pred> bool all_class(const u32 & w, Pred pred) {
    size_t limit = w.size();
    if (limit && w[limit - 1] == U'\n' && !pred(U'\n')) --limit;
    if (limit == 0) return false;
    for (size_t i = 0; i < limit; ++i)
        if (!pred(w[i])) return false;
    return true;
}

// ---------------------------------------------------------------------------
// camel_tools.morphology.utils rewrite helpers (re.sub ports, on code points).

bool is_sun_letter(char32_t c) {
    switch (c) {
    case 0x062A:
    case 0x062B:
    case 0x062F:
    case 0x0630:
    case 0x0631:
    case 0x0632:
    case 0x0633:
    case 0x0634:
    case 0x0635:
    case 0x0636:
    case 0x0637:
    case 0x0638:
    case 0x0644:
    case 0x0646:
        return true;
    default:
        return false;
    }
}

// _REWRITE_DIAC_RE_1: '#\+*([sun])' -> '\1ّ'
u32 diac_re1(const u32 & w) {
    u32 out;
    out.reserve(w.size() + 2);
    size_t i = 0;
    while (i < w.size()) {
        if (w[i] == U'#') {
            size_t j = i + 1;
            while (j < w.size() && w[j] == U'+') ++j;
            if (j < w.size() && is_sun_letter(w[j])) {
                out.push_back(w[j]);
                out.push_back(0x0651);
                i = j + 1;
                continue;
            }
        }
        out.push_back(w[i++]);
    }
    return out;
}

// _REWRITE_DIAC_RE_2: '#\+*' -> ''
u32 diac_re2(const u32 & w) {
    u32 out;
    out.reserve(w.size());
    size_t i = 0;
    while (i < w.size()) {
        if (w[i] == U'#') {
            ++i;
            while (i < w.size() && w[i] == U'+') ++i;
            continue;
        }
        out.push_back(w[i++]);
    }
    return out;
}

// _REWRITE_DIAC_RE_3: 'ا\+?َ([ةت])' -> 'ا\1'
u32 diac_re3(const u32 & w) {
    u32 out;
    out.reserve(w.size());
    size_t i = 0;
    while (i < w.size()) {
        if (w[i] == 0x0627) {
            size_t j = i + 1;
            if (j < w.size() && w[j] == U'+') ++j;
            if (j + 1 < w.size() && w[j] == 0x064E && (w[j + 1] == 0x0629 || w[j + 1] == 0x062A)) {
                out.push_back(0x0627);
                out.push_back(w[j + 1]);
                i = j + 2;
                continue;
            }
        }
        out.push_back(w[i++]);
    }
    return out;
}

u32 rewrite_diac(const u32 & in) {
    u32 w = diac_re3(diac_re2(diac_re1(in)));
    u32 out;
    out.reserve(w.size());
    for (size_t i = 0; i < w.size(); ++i) {
        char32_t c = w[i];
        if (c == 0x0671) c = 0x0627;                                       // RE_4
        if (c == U'+') continue;                                           // RE_5
        if (c == 0x0651 && !out.empty() && out.back() == 0x0651) continue; // RE_6
        out.push_back(c);
    }
    // RE_6 collapses runs present after RE_5 removed the '+'s; the loop above
    // compares against the emitted text, which is that string.
    return out; // normalize_tanwyn(..., 'AF') replaces each pattern by itself.
}

u32 rewrite_tok_1(const u32 & w) {
    return diac_re3(diac_re2(diac_re1(w)));
}

u32 rewrite_tok_2(const u32 & w) {
    return diac_re3(w);
}

u32 rewrite_pattern(const u32 & w) {
    return diac_re2(w);
}

bool at(const u32 & w, size_t i, const char * lit) {
    for (size_t k = 0; lit[k]; ++k)
        if (i + k >= w.size() || w[i + k] != char32_t((unsigned char)lit[k])) return false;
    return true;
}

bool in_set(char32_t c, const char * set) {
    for (size_t k = 0; set[k]; ++k)
        if (c == char32_t((unsigned char)set[k])) return true;
    return false;
}

// '$' matches at the end or before a final '\n'.
bool at_end(const u32 & w, size_t i) {
    return i == w.size() || (i + 1 == w.size() && w[i] == U'\n');
}

u32 rewrite_caphi(const u32 & input) {
    u32 w = input;
    u32 out;
    // RE_1: '(l-)\+(t\_|th\_|d\_|th\.\_|r\_|z\_|s\_|sh\_|s\.\_|d\.\_|t\.\_|dh\.\_|l\_|n\_|dh\_)' -> '\2\2'
    {
        static const char * const alts[] = { "t_",  "th_", "d_",  "th._", "r_", "z_", "s_", "sh_",
                                             "s._", "d._", "t._", "dh._", "l_", "n_", "dh_" };
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (at(w, i, "l-+")) {
                bool done = false;
                for (const char * alt : alts) {
                    if (at(w, i + 3, alt)) {
                        const u32 a = py_str::U(alt);
                        out += a;
                        out += a;
                        i += 3 + a.size();
                        done = true;
                        break;
                    }
                }
                if (done) continue;
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_2: '(\S)[-]*\+~' -> '\1_\1'
    {
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (!isspace(w[i])) {
                size_t j = i + 1;
                while (j < w.size() && w[j] == U'-') ++j;
                if (j + 1 < w.size() && w[j] == U'+' && w[j + 1] == U'~') {
                    out.push_back(w[i]);
                    out.push_back(U'_');
                    out.push_back(w[i]);
                    i = j + 2;
                    continue;
                }
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_3 / RE_4: 'i\_y-\+([^iau]+|$)' -> 'ii_\1', 'u\_w-\+([^iau]+|$)' -> 'uu_\1'
    for (int pass = 0; pass < 2; ++pass) {
        const char * lit = pass == 0 ? "i_y-+" : "u_w-+";
        const char * rep = pass == 0 ? "ii_" : "uu_";
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (at(w, i, lit)) {
                const size_t j = i + 5;
                if (j < w.size() && !in_set(w[j], "iau")) {
                    size_t k = j;
                    while (k < w.size() && !in_set(w[k], "iau")) ++k;
                    out += py_str::U(rep);
                    out.append(w, j, k - j);
                    i = k;
                    continue;
                }
                if (at_end(w, j)) {
                    out += py_str::U(rep);
                    i = j;
                    continue;
                }
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_5: '([iua])\+-2_[iua]' -> '\1'
    {
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (in_set(w[i], "iua") && at(w, i + 1, "+-2_") && i + 5 < w.size() && in_set(w[i + 5], "iua")) {
                out.push_back(w[i]);
                i += 6;
                continue;
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_6: '(.+)\+-2_([iua])' -> '\1_\2' (greedy: the last occurrence on the line)
    {
        out.clear();
        size_t p = 0;
        while (p < w.size()) {
            size_t limit = p;
            while (limit < w.size() && w[limit] != U'\n') ++limit;
            size_t found = std::u32string::npos;
            for (size_t q = limit; q-- > p + 1;) {
                if (at(w, q, "+-2_") && q + 4 < w.size() && in_set(w[q + 4], "iua")) {
                    found = q;
                    break;
                }
            }
            if (found != std::u32string::npos) {
                out.append(w, p, found - p);
                out.push_back(U'_');
                out.push_back(w[found + 4]);
                p = found + 5;
                continue;
            }
            out.push_back(w[p++]);
        }
        w.swap(out);
    }
    // RE_7: 'u\+w(_+[^ioua])' -> 'uu\1'
    {
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (at(w, i, "u+w")) {
                const size_t j = i + 3;
                size_t e = j;
                while (e < w.size() && w[e] == U'_') ++e;
                if (e > j) {
                    if (e < w.size() && !in_set(w[e], "ioua")) {
                        out += py_str::U("uu");
                        out.append(w, j, e + 1 - j);
                        i = e + 1;
                        continue;
                    }
                    if (e - j >= 2) {
                        out += py_str::U("uu");
                        out.append(w, j, e - j);
                        i = e;
                        continue;
                    }
                }
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_8: 'p-\+([iua])' -> 't_\1'
    {
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (at(w, i, "p-+") && i + 3 < w.size() && in_set(w[i + 3], "iua")) {
                out += py_str::U("t_");
                out.push_back(w[i + 3]);
                i += 4;
                continue;
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_9: 'aa\+a[_]*' -> 'aa_'
    {
        out.clear();
        size_t i = 0;
        while (i < w.size()) {
            if (at(w, i, "aa+a")) {
                i += 4;
                while (i < w.size() && w[i] == U'_') ++i;
                out += py_str::U("aa_");
                continue;
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    // RE_10: '[\+-]' -> '_'; RE_11: '_+' -> '_'
    out.clear();
    for (char32_t c : w) {
        if (c == U'+' || c == U'-') c = U'_';
        if (c == U'_' && !out.empty() && out.back() == U'_') continue;
        out.push_back(c);
    }
    w.swap(out);
    // RE_12: '((^\_+)|(\_p?\_*$))' -> ''
    {
        out.clear();
        size_t i = 0;
        if (!w.empty() && w[0] == U'_') {
            while (i < w.size() && w[i] == U'_') ++i;
        }
        while (i < w.size()) {
            if (w[i] == U'_') {
                size_t k = i + 1;
                bool matched = false;
                if (k < w.size() && w[k] == U'p') {
                    size_t m = k + 1;
                    while (m < w.size() && w[m] == U'_') ++m;
                    if (at_end(w, m)) {
                        i = m;
                        matched = true;
                    }
                }
                if (!matched) {
                    size_t m = k;
                    while (m < w.size() && w[m] == U'_') ++m;
                    if (at_end(w, m)) {
                        i = m;
                        matched = true;
                    }
                }
                if (matched) continue;
            }
            out.push_back(w[i++]);
        }
        w.swap(out);
    }
    return w;
}

// simple_ar_to_caphi
std::string simple_ar_to_caphi(const u32 & in) {
    u32 s = in;
    if (!s.empty() && s[0] == 0x0627) s[0] = 0x0623;
    std::string out;
    bool first = true;
    for (char32_t c : s) {
        const char * v = nullptr;
        switch (c) {
        case 0x0621:
            v = "2";
            break;
        case 0x0622:
            v = "2_aa";
            break;
        case 0x0623:
            v = "2";
            break;
        case 0x0624:
            v = "2";
            break;
        case 0x0625:
            v = "2";
            break;
        case 0x0626:
            v = "2";
            break;
        case 0x0627:
            v = "aa";
            break;
        case 0x0628:
            v = "b";
            break;
        case 0x062A:
            v = "t";
            break;
        case 0x062B:
            v = "th";
            break;
        case 0x062C:
            v = "j";
            break;
        case 0x062D:
            v = "7";
            break;
        case 0x062E:
            v = "kh";
            break;
        case 0x062F:
            v = "d";
            break;
        case 0x0630:
            v = "dh";
            break;
        case 0x0631:
            v = "r";
            break;
        case 0x0632:
            v = "z";
            break;
        case 0x0633:
            v = "s";
            break;
        case 0x0634:
            v = "sh";
            break;
        case 0x0635:
            v = "s.";
            break;
        case 0x0636:
            v = "d.";
            break;
        case 0x0637:
            v = "t.";
            break;
        case 0x0638:
            v = "dh.";
            break;
        case 0x0639:
            v = "3";
            break;
        case 0x063A:
            v = "gh";
            break;
        case 0x0641:
            v = "f";
            break;
        case 0x0642:
            v = "q";
            break;
        case 0x0643:
            v = "k";
            break;
        case 0x0644:
            v = "l";
            break;
        case 0x0645:
            v = "m";
            break;
        case 0x0646:
            v = "n";
            break;
        case 0x0647:
            v = "h";
            break;
        case 0x0648:
            v = "w";
            break;
        case 0x0649:
            v = "aa";
            break;
        case 0x064A:
            v = "y";
            break;
        default:
            break;
        }
        if (!v) continue;
        if (!first) out.push_back('_');
        out += v;
        first = false;
    }
    return out;
}

// Python re.sub('NOAN', repl, s) with repl parsed as a template exactly like
// CPython 3.12's re._parser.parse_template for a pattern with no groups.
// Throws input_error where CPython raises re.error / IndexError.
std::string expand_noan_template(const u32 & repl) {
    std::string out;
    size_t i = 0;
    auto is_oct = [](char32_t c) { return c >= U'0' && c <= U'7'; };
    auto is_dig = [](char32_t c) { return c >= U'0' && c <= U'9'; };
    while (i < repl.size()) {
        const char32_t c0 = repl[i];
        if (c0 != U'\\') {
            py_str::append_utf8(out, c0);
            ++i;
            continue;
        }
        if (i + 1 >= repl.size()) throw input_error("camel_tools raises re.error: bad escape (end of pattern)");
        const char32_t c = repl[i + 1];
        i += 2;
        if (c == U'g') {
            if (i >= repl.size() || repl[i] != U'<') throw input_error("camel_tools raises re.error: missing <");
            const size_t close = repl.find(U'>', i + 1);
            if (close == u32::npos) throw input_error("camel_tools raises re.error: missing >");
            const u32 name = repl.substr(i + 1, close - i - 1);
            i = close + 1;
            bool ascii_digits = !name.empty();
            for (char32_t d : name) ascii_digits = ascii_digits && is_dig(d);
            size_t nz = 0;
            while (ascii_digits && nz < name.size() && name[nz] == U'0') ++nz;
            if (!ascii_digits || nz != name.size())
                throw input_error("camel_tools raises re.error: invalid group reference");
            out += "NOAN"; // \g<0>: the whole match
        } else if (c == U'0') {
            unsigned v = 0;
            int taken = 0;
            while (taken < 2 && i < repl.size() && is_oct(repl[i])) {
                v = v * 8 + unsigned(repl[i] - U'0');
                ++i;
                ++taken;
            }
            py_str::append_utf8(out, char32_t(v & 0xFF));
        } else if (is_dig(c)) {
            if (i < repl.size() && is_dig(repl[i])) {
                const char32_t c2 = repl[i];
                if (is_oct(c) && is_oct(c2) && i + 1 < repl.size() && is_oct(repl[i + 1])) {
                    const unsigned v = unsigned(c - U'0') * 64 + unsigned(c2 - U'0') * 8 + unsigned(repl[i + 1] - U'0');
                    if (v > 0377) throw input_error("camel_tools raises re.error: octal escape out of range");
                    py_str::append_utf8(out, char32_t(v));
                    i += 2;
                    continue;
                }
            }
            throw input_error("camel_tools raises re.error: invalid group reference");
        } else {
            char32_t mapped = 0;
            switch (c) {
            case U'a':
                mapped = 7;
                break;
            case U'b':
                mapped = 8;
                break;
            case U'f':
                mapped = 12;
                break;
            case U'n':
                mapped = 10;
                break;
            case U'r':
                mapped = 13;
                break;
            case U't':
                mapped = 9;
                break;
            case U'v':
                mapped = 11;
                break;
            case U'\\':
                mapped = U'\\';
                break;
            default:
                break;
            }
            if (mapped) {
                py_str::append_utf8(out, mapped);
            } else if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')) {
                throw input_error("camel_tools raises re.error: bad escape");
            } else {
                out.push_back('\\');
                py_str::append_utf8(out, c);
            }
        }
    }
    return out;
}

std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
    std::string out;
    size_t start = 0;
    for (;;) {
        const size_t p = s.find(from, start);
        if (p == std::string_view::npos) break;
        out.append(s.data() + start, p - start);
        out.append(to.data(), to.size());
        start = p + from.size();
    }
    out.append(s.data() + start, s.size() - start);
    return out;
}

// float() of a validated plain decimal: an integer mantissa below 2^53 and a
// power of ten up to 10^22 are exact doubles, and IEEE division rounds the
// quotient correctly, so this equals Python's correctly rounded float().
bool parse_plain_decimal(std::string_view s, double & out) {
    size_t i = 0;
    bool neg = false;
    if (i < s.size() && s[i] == '-') {
        neg = true;
        ++i;
    }
    uint64_t mant = 0;
    int digits = 0, frac = 0;
    bool seen_dot = false, any = false;
    for (; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '.' && !seen_dot && any) {
            seen_dot = true;
            continue;
        }
        if (c < '0' || c > '9') return false;
        any = true;
        if (mant || c != '0') ++digits;
        if (digits > 15) return false;
        mant = mant * 10 + uint64_t(c - '0');
        if (seen_dot) ++frac;
    }
    if (!any || (seen_dot && frac == 0) || frac > 22) return false;
    double v = double(mant);
    if (frac) {
        double p = 1;
        for (int k = 0; k < frac; ++k) p *= 10;
        v /= p;
    }
    out = neg ? -v : v;
    return true;
}

// ---------------------------------------------------------------------------
// Values and ordered result dicts.

struct value {
    enum kind_t : uint8_t { str_t, none_t, float_t } kind = str_t;
    std::string s;
    double f = 0;
};

struct rfeat {
    uint32_t key;
    value v;
};

using result_t = std::vector<rfeat>;

value str_value(std::string s) {
    value v;
    v.s = std::move(s);
    return v;
}

value none_value() {
    value v;
    v.kind = value::none_t;
    return v;
}

value float_value(double f) {
    value v;
    v.kind = value::float_t;
    v.f = f;
    return v;
}

rfeat * find(result_t & r, uint32_t key) {
    for (auto & f : r)
        if (f.key == key) return &f;
    return nullptr;
}

void set(result_t & r, uint32_t key, value v) {
    if (rfeat * f = find(r, key)) {
        f->v = std::move(v);
        return;
    }
    r.push_back({ key, std::move(v) });
}

// ---------------------------------------------------------------------------
// Database (MorphologyDB with flags='a').

struct feat {
    uint32_t key;
    uint32_t val; // string id
};

struct entry {
    uint32_t cat;
    uint32_t begin;
    uint32_t count;
};

// Ordered key->string view, the shape of a camel_tools analysis line dict.
using fview = std::vector<std::pair<uint32_t, std::string_view>>;

const std::string_view * fget(const fview & f, uint32_t key) {
    for (const auto & kv : f)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

enum known_key : uint32_t {
    K_DIAC,
    K_LEX,
    K_GLOSS,
    K_BW,
    K_POS,
    K_PATTERN,
    K_CAPHI,
    K_CATIB6,
    K_UD,
    K_ROOT,
    K_D3TOK,
    K_D3SEG,
    K_ATBSEG,
    K_D2SEG,
    K_D1SEG,
    K_D1TOK,
    K_D2TOK,
    K_ATBTOK,
    K_BWTOK,
    K_STEM,
    K_STEMGLOSS,
    K_STEMCAT,
    K_SOURCE,
    K_GEN,
    K_NUM,
    K_FORM_GEN,
    K_FORM_NUM,
    K_POS_LOGPROB,
    K_LEX_LOGPROB,
    K_POS_LEX_LOGPROB,
    K_COUNT
};

const char * const k_known_names[K_COUNT] = {
    "diac",  "lex",      "gloss",    "bw",          "pos",         "pattern",        "caphi",  "catib6",
    "ud",    "root",     "d3tok",    "d3seg",       "atbseg",      "d2seg",          "d1seg",  "d1tok",
    "d2tok", "atbtok",   "bwtok",    "stem",        "stemgloss",   "stemcat",        "source", "gen",
    "num",   "form_gen", "form_num", "pos_logprob", "lex_logprob", "pos_lex_logprob"
};

// camel_tools frozensets, in source order (see the port notes above).
const uint32_t k_join_feats[] = { K_GLOSS, K_BW };
const uint32_t k_concat_feats[] = { K_DIAC, K_PATTERN, K_CAPHI, K_CATIB6, K_UD };
const uint32_t k_concat_feats_none[] = { K_D3TOK, K_D3SEG, K_ATBSEG, K_D2SEG, K_D1SEG,
                                         K_D1TOK, K_D2TOK, K_ATBTOK, K_BWTOK };
const uint32_t k_tok_schemes_1[] = { K_D1TOK, K_D2TOK, K_ATBTOK, K_D1SEG, K_D2SEG, K_D3SEG, K_ATBSEG };
const uint32_t k_tok_schemes_2[] = { K_D3TOK, K_D3SEG };
const uint32_t k_logprob_feats[] = { K_POS_LOGPROB, K_LEX_LOGPROB, K_POS_LEX_LOGPROB };
const uint32_t k_copy_feats[] = { K_GLOSS, K_ATBTOK, K_ATBSEG, K_D1TOK, K_D1SEG,
                                  K_D2TOK, K_D2SEG,  K_D3TOK,  K_D3SEG, K_BWTOK };
const uint32_t k_undefined_lex_feats[] = { K_ROOT, K_PATTERN, K_CAPHI };

// What the analyzer needs from either database representation.
struct db_common {
    std::vector<std::string> key_names;
    std::unordered_map<std::string, uint32_t> key_ids;
    std::vector<uint8_t> defined; // by key id
    // Output keys by id; empty means every key (the text database).
    std::vector<uint8_t> output;
    bool pattern_in_order = false;
    std::unordered_map<std::string, result_t> defaults;
    std::unordered_map<std::string, std::vector<std::string>> stem_backoffs;
    size_t max_prefix_size = 0, max_suffix_size = 0;

    bool def(uint32_t key) const { return key < defined.size() && defined[key]; }

    bool keep(uint32_t key) const { return output.empty() || (key < output.size() && output[key]); }

    uint32_t key_id(std::string_view name) {
        auto it = key_ids.find(std::string(name));
        if (it != key_ids.end()) return it->second;
        const uint32_t id = uint32_t(key_names.size());
        key_names.emplace_back(name);
        key_ids.emplace(std::string(name), id);
        return id;
    }

    static uint64_t pair_key(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }
};

// Entries of one form: explicit ids (text) or a contiguous row range (binary).
struct elist {
    const uint32_t * ids = nullptr;
    uint32_t first = 0;
    uint32_t n = 0;

    uint32_t size() const { return n; }

    uint32_t operator[](uint32_t i) const { return ids ? ids[i] : first + i; }

    bool empty() const { return n == 0; }
};

// The ALMOR text database parsed into heap structures.
struct morph_db : db_common {
    // Interned strings.
    std::string pool;
    std::vector<std::pair<uint32_t, uint32_t>> strs;
    std::vector<std::string> cat_names;
    std::unordered_map<std::string, uint32_t> cat_ids;

    std::vector<feat> feats;
    std::vector<entry> entries;
    std::unordered_map<std::string, std::vector<uint32_t>> prefix_hash, suffix_hash, stem_hash;

    std::unordered_set<uint64_t> ab, bc, ac;
    std::vector<uint8_t> ab_has, bc_has, ac_has;

    std::string_view str(uint32_t id) const { return std::string_view(pool.data() + strs[id].first, strs[id].second); }

    uint32_t cat_id(std::string_view name) {
        auto it = cat_ids.find(std::string(name));
        if (it != cat_ids.end()) return it->second;
        const uint32_t id = uint32_t(cat_names.size());
        cat_names.emplace_back(name);
        cat_ids.emplace(std::string(name), id);
        return id;
    }

    fview view(uint32_t e) const {
        fview out;
        const entry & en = entries[e];
        out.reserve(en.count);
        for (uint32_t k = 0; k < en.count; ++k) {
            const feat & f = feats[en.begin + k];
            out.emplace_back(f.key, str(f.val));
        }
        return out;
    }

    static elist lookup(const std::unordered_map<std::string, std::vector<uint32_t>> & hash, const std::string & form) {
        auto it = hash.find(form);
        elist l;
        if (it != hash.end()) {
            l.ids = it->second.data();
            l.n = uint32_t(it->second.size());
        }
        return l;
    }

    elist prefixes(const std::string & form) const { return lookup(prefix_hash, form); }

    elist suffixes(const std::string & form) const { return lookup(suffix_hash, form); }

    elist stems(const std::string & form) const { return lookup(stem_hash, form); }

    uint32_t cat(uint32_t e) const { return entries[e].cat; }

    std::string_view cat_name(uint32_t c) const { return cat_names[c]; }

    bool ab_has_cat(uint32_t a) const { return ab_has[a] != 0; }

    bool bc_has_cat(uint32_t b) const { return bc_has[b] != 0; }

    bool ac_has_cat(uint32_t a) const { return ac_has[a] != 0; }

    bool ab_ok(uint32_t a, uint32_t b) const { return ab.count(pair_key(a, b)) != 0; }

    bool bc_ok(uint32_t b, uint32_t c) const { return bc.count(pair_key(b, c)) != 0; }

    bool ac_ok(uint32_t a, uint32_t c) const { return ac.count(pair_key(a, c)) != 0; }

    size_t prefix_forms() const { return prefix_hash.size(); }

    size_t suffix_forms() const { return suffix_hash.size(); }

    size_t stem_forms() const { return stem_hash.size(); }

    size_t stem_entries() const {
        size_t n = 0;
        for (const auto & kv : stem_hash) n += kv.second.size();
        return n;
    }
};

class db_parser {
public:
    db_parser(std::string text, morph_db & db) : text_(std::move(text)), db_(db) {}

    void parse() {
        for (uint32_t k = 0; k < K_COUNT; ++k) db_.key_id(k_known_names[k]);
        if (!valid_utf8(text_)) throw data_error("database is not valid UTF-8");
        universal_newlines();
        split_lines();
        size_t li = 0;
        parse_defines(li);
        parse_defaults(li);
        parse_order(li);
        parse_tokenizations(li);
        parse_stembackoffs(li);
        parse_affixes(li, "###SUFFIXES###", db_.prefix_hash, "PREFIXES");
        parse_affixes(li, "###STEMS###", db_.suffix_hash, "SUFFIXES");
        parse_stems(li);
        parse_table(li, "###TABLE BC###", db_.ab, db_.ab_has, "TABLE AB", true);
        parse_table(li, "###TABLE AC###", db_.bc, db_.bc_has, "TABLE BC", true);
        parse_table(li, nullptr, db_.ac, db_.ac_has, "TABLE AC", false);
        for (const auto & kv : db_.prefix_hash) db_.max_prefix_size = std::max(db_.max_prefix_size, cp_count(kv.first));
        for (const auto & kv : db_.suffix_hash) db_.max_suffix_size = std::max(db_.max_suffix_size, cp_count(kv.first));
        db_.defined.resize(db_.key_names.size(), 0);
        db_.ab_has.resize(db_.cat_names.size(), 0);
        db_.bc_has.resize(db_.cat_names.size(), 0);
        db_.ac_has.resize(db_.cat_names.size(), 0);
        intern_.clear();
    }

private:
    std::string text_;
    morph_db & db_;
    std::vector<std::string_view> lines_; // each includes its '\n' when present
    std::unordered_map<std::string_view, uint32_t> intern_;
    std::unordered_set<std::string> defines_;

    [[noreturn]] void fail(const std::string & what) { throw data_error("invalid database: " + what); }

    // open(..., encoding='utf-8') uses universal newlines.
    void universal_newlines() {
        if (text_.find('\r') == std::string::npos) return;
        std::string out;
        out.reserve(text_.size());
        for (size_t i = 0; i < text_.size(); ++i) {
            if (text_[i] == '\r') {
                out.push_back('\n');
                if (i + 1 < text_.size() && text_[i + 1] == '\n') ++i;
            } else {
                out.push_back(text_[i]);
            }
        }
        text_.swap(out);
    }

    void split_lines() {
        std::string_view t(text_);
        size_t start = 0;
        while (start < t.size()) {
            const size_t p = t.find('\n', start);
            const size_t end = p == std::string_view::npos ? t.size() : p + 1;
            lines_.push_back(t.substr(start, end - start));
            start = end;
        }
    }

    uint32_t intern(std::string_view s) {
        auto it = intern_.find(s);
        if (it != intern_.end()) return it->second;
        if (db_.pool.size() + s.size() > 0xFFFFFFFFu) fail("string pool overflow");
        const uint32_t id = uint32_t(db_.strs.size());
        db_.strs.emplace_back(uint32_t(db_.pool.size()), uint32_t(s.size()));
        db_.pool.append(s.data(), s.size());
        intern_.emplace(s, id);
        return id;
    }

    void expect_more(size_t li, const char * section) {
        if (li >= lines_.size()) fail(std::string("missing section after ") + section);
    }

    void parse_defines(size_t & li) {
        for (;; ++li) {
            expect_more(li, "DEFINES");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###DEFINES###") continue;
            if (line == "###DEFAULTS###") {
                ++li;
                return;
            }
            const auto toks = split_char(line, ' ');
            if (toks.size() < 3 || toks[0] != "DEFINE") fail("DEFINES line");
            for (size_t t = 2; t < toks.size(); ++t) {
                const auto sub = split_char(toks[t], ':');
                if (sub.size() != 2 && sub[0] != toks[1]) fail("DEFINES value");
                if (sub.size() < 2) fail("DEFINES value");
                if (toks.size() == 3 && sub[1] == "*open*") break;
            }
            defines_.emplace(toks[1]);
            const uint32_t id = db_.key_id(toks[1]);
            if (db_.defined.size() <= id) db_.defined.resize(id + 1, 0);
            db_.defined[id] = 1;
        }
    }

    void parse_defaults(size_t & li) {
        for (;; ++li) {
            expect_more(li, "DEFAULTS");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###ORDER###") {
                ++li;
                return;
            }
            const auto toks = split_char(line, ' ');
            if (toks.size() < 2 || toks[0] != "DEFAULT") fail("DEFAULTS line");
            result_t parsed;
            for (size_t t = 1; t < toks.size(); ++t) {
                const size_t colon = toks[t].find(':');
                if (colon == std::string_view::npos) fail("DEFAULTS value");
                const std::string_view val = toks[t].substr(colon + 1);
                set(parsed, db_.key_id(toks[t].substr(0, colon)),
                    val == "*" ? none_value() : str_value(std::string(val)));
            }
            const rfeat * pos = find(parsed, K_POS);
            if (!pos) fail("DEFAULTS line without pos");
            // defaults[None] (pos:*) is never read by the analyzer.
            if (pos->v.kind == value::str_t) db_.defaults[pos->v.s] = parsed;
        }
    }

    void parse_order(size_t & li) {
        bool have = false;
        for (;; ++li) {
            expect_more(li, "ORDER");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###TOKENIZATIONS###") {
                if (!have) fail("missing ORDER");
                ++li;
                return;
            }
            const auto toks = split_char(line, ' ');
            if (toks.size() < 2 || !defines_.count(std::string(toks[1]))) fail("ORDER line");
            db_.pattern_in_order = false;
            for (size_t t = 1; t < toks.size(); ++t)
                if (toks[t] == "pattern") db_.pattern_in_order = true;
            have = true;
        }
    }

    void parse_tokenizations(size_t & li) {
        for (;; ++li) {
            expect_more(li, "TOKENIZATIONS");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###STEMBACKOFF###") {
                ++li;
                return;
            }
            const auto toks = split_char(line, ' ');
            if (toks.size() < 2 || !defines_.count(std::string(toks[1]))) fail("TOKENIZATION line");
        }
    }

    void parse_stembackoffs(size_t & li) {
        for (;; ++li) {
            expect_more(li, "STEMBACKOFF");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###PREFIXES###") {
                ++li;
                return;
            }
            const auto toks = split_char(line, ' ');
            if (toks.size() < 3 || toks[0] != "STEMBACKOFF") fail("STEMBACKOFF line");
            std::vector<std::string> cats;
            for (size_t t = 2; t < toks.size(); ++t) cats.emplace_back(toks[t]);
            db_.stem_backoffs[std::string(toks[1])] = std::move(cats);
        }
    }

    // _parse_analysis_line_toks
    uint32_t add_entry(std::string_view category, const std::vector<std::string_view> & toks, bool is_stem) {
        if (db_.entries.size() >= k_max_entries) fail("too many entries");
        entry e;
        e.cat = db_.cat_id(category);
        e.begin = uint32_t(db_.feats.size());
        e.count = 0;
        for (std::string_view tok : toks) {
            if (tok.empty()) continue;
            const size_t colon = tok.find(':');
            if (colon == std::string_view::npos) fail("invalid key value pair");
            const uint32_t key = db_.key_id(tok.substr(0, colon));
            std::string_view val = tok.substr(colon + 1);
            if (key == K_POS_LOGPROB || key == K_LEX_LOGPROB || key == K_POS_LEX_LOGPROB) {
                double ignored;
                if (!parse_plain_decimal(val, ignored)) fail("unsupported *_logprob literal");
            }
            bool replaced = false;
            for (uint32_t k = 0; k < e.count; ++k) {
                feat & f = db_.feats[e.begin + k];
                if (f.key == key) {
                    f.val = intern(val);
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                if (e.count >= k_max_entry_feats) fail("too many features in one entry");
                db_.feats.push_back({ key, intern(val) });
                ++e.count;
            }
        }
        if (is_stem) {
            // analysis['lex'] = strip_lex(analysis['lex'])
            bool have_lex = false;
            for (uint32_t k = 0; k < e.count; ++k) {
                feat & f = db_.feats[e.begin + k];
                if (f.key != K_LEX) continue;
                const std::string_view lex = db_.str(f.val);
                const size_t cut = lex.find_first_of("_-");
                if (cut != std::string_view::npos) {
                    // Views into the pool are not stable across intern(); copy first.
                    const std::string prefix(lex.substr(0, cut));
                    f.val = intern_owned(prefix);
                }
                have_lex = true;
            }
            if (!have_lex) fail("STEMS entry without lex");
        }
        db_.entries.push_back(e);
        return uint32_t(db_.entries.size() - 1);
    }

    // Interns a string that does not live in text_ (keys must stay valid).
    uint32_t intern_owned(const std::string & s) {
        owned_.push_back(std::make_unique<std::string>(s));
        return intern(*owned_.back());
    }
    std::vector<std::unique_ptr<std::string>> owned_;

    void parse_affixes(size_t & li, const char * end_marker,
                       std::unordered_map<std::string, std::vector<uint32_t>> & hash, const char * section) {
        for (;; ++li) {
            expect_more(li, section);
            const std::string_view line = lines_[li];
            const auto parts = split_char(line, '\t');
            if (parts.size() != 3) {
                if (py_strip(line) == end_marker) {
                    ++li;
                    return;
                }
                fail(std::string("invalid ") + section + " line");
            }
            const std::string form(py_strip(parts[0]));
            if (cp_count(form) > k_max_form_cps) fail("affix too long");
            const uint32_t e = add_entry(parts[1], split_char(py_strip(parts[2]), ' '), false);
            hash[form].push_back(e);
        }
    }

    void parse_stems(size_t & li) {
        for (;; ++li) {
            expect_more(li, "STEMS");
            const std::string_view line = py_strip(lines_[li]);
            if (line == "###TABLE AB###") {
                ++li;
                return;
            }
            const auto parts = split_char(line, '\t');
            if (parts.size() != 3) fail("invalid STEMS line");
            if (cp_count(parts[0]) > k_max_form_cps) fail("stem too long");
            const uint32_t e = add_entry(parts[1], split_char(parts[2], ' '), true);
            db_.stem_hash[std::string(parts[0])].push_back(e);
        }
    }

    void parse_table(size_t & li, const char * end_marker, std::unordered_set<uint64_t> & table,
                     std::vector<uint8_t> & has, const char * section, bool need_marker) {
        for (;; ++li) {
            if (li >= lines_.size()) {
                if (need_marker) fail(std::string("missing marker after ") + section);
                return;
            }
            const std::string_view line = py_strip(lines_[li]);
            if (end_marker && line == end_marker) {
                ++li;
                return;
            }
            const auto toks = split_ws(line);
            if (toks.size() != 2) fail(std::string("invalid ") + section + " line");
            const uint32_t a = db_.cat_id(toks[0]);
            const uint32_t b = db_.cat_id(toks[1]);
            table.insert(db_common::pair_key(a, b));
            if (has.size() <= a) has.resize(a + 1, 0);
            has[a] = 1;
        }
    }
};

// ---------------------------------------------------------------------------
// Binary database (docs/arabic-morph-binary-format.md): read in place from a
// memory-mapped (or, on hosts without mmap, heap-held) file. Every offset,
// id and string is validated once at load; lookups never allocate.

constexpr char k_bin_magic[8] = { 'C', 'E', 'M', 'O', 'R', 'P', 'H', 'B' };
constexpr uint32_t k_bin_version = 1;
constexpr uint32_t k_bin_header_bytes = 64;
constexpr uint32_t k_bin_cat_column = 0xFFFFFFFFu;
constexpr uint32_t k_bin_none = 0xFFFFFFFFu;
constexpr uint32_t k_key_defined = 1, k_key_output = 2;

uint32_t crc32_bytes(const uint8_t * p, size_t n) {
    static uint32_t table[256];
    static const bool ready = [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return true;
    }();
    (void)ready;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

inline uint32_t rd32(const uint8_t * p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

inline uint64_t rd64(const uint8_t * p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline uint32_t rd_width(const uint8_t * p, uint32_t width) {
    if (width == 1) return p[0];
    if (width == 2) return uint32_t(p[0]) | (uint32_t(p[1]) << 8);
    return rd32(p);
}

bool little_endian_host() {
    const uint32_t one = 1;
    uint8_t b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

// Bounds-checked cursor over one section.
class bin_reader {
public:
    bin_reader(const uint8_t * p, size_t n, const char * what) : p_(p), n_(n), what_(what) {}

    uint32_t u32() {
        need(4);
        const uint32_t v = rd32(p_ + pos_);
        pos_ += 4;
        return v;
    }

    const uint8_t * take(size_t bytes) {
        need(bytes);
        const uint8_t * at = p_ + pos_;
        pos_ += bytes;
        return at;
    }

    // n items of item_bytes each, guarding the multiplication.
    const uint8_t * take_array(uint64_t n, uint64_t item_bytes) {
        if (item_bytes && n > (uint64_t(1) << 40) / item_bytes) fail();
        return take(size_t(n * item_bytes));
    }

    void align8() { pos_ = std::min(n_, (pos_ + 7) & ~size_t(7)); }

    [[noreturn]] void fail() const { throw data_error(std::string("binary database: truncated ") + what_); }

private:
    void need(size_t bytes) const {
        if (bytes > n_ - pos_) fail();
    }

    const uint8_t * p_;
    size_t n_;
    size_t pos_ = 0;
    const char * what_;
};

struct bin_column {
    uint32_t key;   // morph key id, or k_bin_cat_column
    uint32_t width; // 1, 2 or 4 bytes
    uint32_t offset;
    uint32_t dict_count;
    const uint8_t * dict; // dict_count little-endian u32 string (or category) ids
};

struct bin_table {
    uint32_t rows = 0, forms = 0, row_bytes = 0, base = 0;
    std::vector<bin_column> columns; // value columns in stored-key order
    bin_column cat{};
    const uint8_t * index = nullptr; // forms x {form string id, first row, row count}
    const uint8_t * data = nullptr;  // rows x row_bytes
};

struct bin_db : db_common {
    // The mapped or heap-held file; the database points into it.
    std::shared_ptr<const void> owner;
    const uint8_t * base = nullptr;
    size_t size = 0;

    uint32_t n_strings = 0;
    const uint8_t * str_offsets = nullptr;
    const uint8_t * str_bytes = nullptr;
    uint32_t n_cats = 0;
    const uint8_t * cat_ids = nullptr;
    bin_table prefix, suffix, stem;
    const uint8_t * compat[3] = {}; // AB, BC, AC sorted (a << 32 | b)
    uint32_t compat_n[3] = {};

    std::string_view str(uint32_t id) const {
        const uint32_t a = rd32(str_offsets + 4 * size_t(id)), b = rd32(str_offsets + 4 * size_t(id) + 4);
        return std::string_view(reinterpret_cast<const char *>(str_bytes) + a, b - a);
    }

    const bin_table & table_of(uint32_t e, uint32_t & row) const {
        if (e < suffix.base) {
            row = e;
            return prefix;
        }
        if (e < stem.base) {
            row = e - suffix.base;
            return suffix;
        }
        row = e - stem.base;
        return stem;
    }

    static uint32_t local(const bin_table & t, uint32_t row, const bin_column & c) {
        return rd_width(t.data + size_t(row) * t.row_bytes + c.offset, c.width);
    }

    uint32_t cat(uint32_t e) const {
        uint32_t row;
        const bin_table & t = table_of(e, row);
        return rd32(t.cat.dict + 4 * size_t(local(t, row, t.cat)));
    }

    std::string_view cat_name(uint32_t c) const { return str(rd32(cat_ids + 4 * size_t(c))); }

    fview view(uint32_t e) const {
        uint32_t row;
        const bin_table & t = table_of(e, row);
        fview out;
        out.reserve(t.columns.size());
        const uint8_t * r = t.data + size_t(row) * t.row_bytes;
        for (const bin_column & c : t.columns) {
            const uint32_t v = rd_width(r + c.offset, c.width);
            if (v >= c.dict_count) continue; // absent
            out.emplace_back(c.key, str(rd32(c.dict + 4 * size_t(v))));
        }
        return out;
    }

    elist find(const bin_table & t, std::string_view form) const {
        uint32_t lo = 0, hi = t.forms;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            const int c = str(rd32(t.index + 12 * size_t(mid))).compare(form);
            if (c < 0)
                lo = mid + 1;
            else if (c > 0)
                hi = mid;
            else {
                elist l;
                l.first = t.base + rd32(t.index + 12 * size_t(mid) + 4);
                l.n = rd32(t.index + 12 * size_t(mid) + 8);
                return l;
            }
        }
        return elist();
    }

    elist prefixes(std::string_view form) const { return find(prefix, form); }

    elist suffixes(std::string_view form) const { return find(suffix, form); }

    elist stems(std::string_view form) const { return find(stem, form); }

    // First pair index >= key in table t.
    uint32_t lower(int t, uint64_t key) const {
        uint32_t lo = 0, hi = compat_n[t];
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            if (rd64(compat[t] + 8 * size_t(mid)) < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    bool has_first(int t, uint32_t a) const {
        const uint32_t i = lower(t, pair_key(a, 0));
        return i < compat_n[t] && uint32_t(rd64(compat[t] + 8 * size_t(i)) >> 32) == a;
    }

    bool has_pair(int t, uint32_t a, uint32_t b) const {
        const uint64_t key = pair_key(a, b);
        const uint32_t i = lower(t, key);
        return i < compat_n[t] && rd64(compat[t] + 8 * size_t(i)) == key;
    }

    bool ab_has_cat(uint32_t a) const { return has_first(0, a); }

    bool bc_has_cat(uint32_t b) const { return has_first(1, b); }

    bool ac_has_cat(uint32_t a) const { return has_first(2, a); }

    bool ab_ok(uint32_t a, uint32_t b) const { return has_pair(0, a, b); }

    bool bc_ok(uint32_t b, uint32_t c) const { return has_pair(1, b, c); }

    bool ac_ok(uint32_t a, uint32_t c) const { return has_pair(2, a, c); }

    size_t prefix_forms() const { return prefix.forms; }

    size_t suffix_forms() const { return suffix.forms; }

    size_t stem_forms() const { return stem.forms; }

    size_t stem_entries() const { return stem.rows; }
};

// Validates the file and builds the small heap-side tables (keys, defaults,
// stem backoffs, column descriptors). Throws data_error.
void load_bin(bin_db & db) {
    if (!little_endian_host()) throw data_error("binary database requires a little-endian host");
    const uint8_t * f = db.base;
    const size_t n = db.size;
    if (n < k_bin_header_bytes || std::memcmp(f, k_bin_magic, 8) != 0) throw data_error("not a binary database");
    if (rd32(f + 8) != k_bin_version) throw data_error("binary database: unsupported format version");
    if (rd32(f + 12) != k_bin_header_bytes) throw data_error("binary database: header size");
    const uint32_t n_sections = rd32(f + 16);
    const uint32_t flags = rd32(f + 20);
    if (rd64(f + 24) != uint64_t(n)) throw data_error("binary database: file size mismatch");
    {
        uint8_t header[k_bin_header_bytes];
        std::memcpy(header, f, k_bin_header_bytes);
        std::memset(header + 36, 0, 4);
        if (crc32_bytes(header, k_bin_header_bytes) != rd32(f + 36))
            throw data_error("binary database: header checksum");
    }
    if (n_sections == 0 || n_sections > 64 || uint64_t(k_bin_header_bytes) + 24ull * n_sections > n)
        throw data_error("binary database: section directory");
    const uint8_t * dir = f + k_bin_header_bytes;
    if (crc32_bytes(dir, 24 * size_t(n_sections)) != rd32(f + 32))
        throw data_error("binary database: directory checksum");
    db.pattern_in_order = (flags & 1) != 0;

    struct section {
        const uint8_t * p = nullptr;
        size_t n = 0;
    };

    auto tag_of = [](const char * t) { return rd32(reinterpret_cast<const uint8_t *>(t)); };
    static const char * const tags[] = { "STRS", "KEYS", "CATS", "DFLT", "BKOF", "PREF",
                                         "SUFF", "STEM", "TBAB", "TBBC", "TBAC" };
    section sec[11];
    for (uint32_t i = 0; i < n_sections; ++i) {
        const uint8_t * d = dir + 24 * size_t(i);
        const uint64_t off = rd64(d + 8), len = rd64(d + 16);
        if (off % 8 != 0 || off > n || len > n - off) throw data_error("binary database: section bounds");
        if (crc32_bytes(f + off, size_t(len)) != rd32(d + 4)) throw data_error("binary database: section checksum");
        for (size_t t = 0; t < 11; ++t) {
            if (rd32(d) != tag_of(tags[t])) continue;
            if (sec[t].p) throw data_error("binary database: duplicate section");
            sec[t].p = f + off;
            sec[t].n = size_t(len);
        }
    }
    for (size_t t = 0; t < 11; ++t)
        if (!sec[t].p) throw data_error(std::string("binary database: missing section ") + tags[t]);

    // STRS: sorted, unique, valid UTF-8.
    {
        bin_reader r(sec[0].p, sec[0].n, "STRS");
        db.n_strings = r.u32();
        db.str_offsets = r.take_array(uint64_t(db.n_strings) + 1, 4);
        const uint32_t total = rd32(db.str_offsets + 4 * size_t(db.n_strings));
        db.str_bytes = r.take(total);
        uint32_t prev = 0;
        for (uint32_t i = 0; i <= db.n_strings; ++i) {
            const uint32_t o = rd32(db.str_offsets + 4 * size_t(i));
            if ((i == 0 && o != 0) || o < prev) throw data_error("binary database: string offsets");
            prev = o;
        }
        for (uint32_t i = 0; i < db.n_strings; ++i) {
            if (!valid_utf8(db.str(i))) throw data_error("binary database: string is not valid UTF-8");
            if (i && !(db.str(i - 1) < db.str(i))) throw data_error("binary database: string table order");
        }
    }
    auto string_id = [&](uint32_t id) {
        if (id >= db.n_strings) throw data_error("binary database: string id");
        return id;
    };

    // KEYS: file key index -> morph key id.
    for (uint32_t k = 0; k < K_COUNT; ++k) db.key_id(k_known_names[k]);
    std::vector<uint32_t> key_map;
    {
        bin_reader r(sec[1].p, sec[1].n, "KEYS");
        const uint32_t count = r.u32();
        if (count > 4096) throw data_error("binary database: key count");
        std::vector<std::pair<uint32_t, uint32_t>> keys;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t name = string_id(r.u32());
            keys.emplace_back(name, r.u32());
        }
        std::unordered_set<uint32_t> seen;
        for (const auto & k : keys) {
            if (!seen.insert(k.first).second) throw data_error("binary database: duplicate key");
            const uint32_t id = db.key_id(db.str(k.first));
            key_map.push_back(id);
            if (db.defined.size() <= id) db.defined.resize(id + 1, 0);
            if (db.output.size() <= id) db.output.resize(id + 1, 0);
            db.defined[id] = (k.second & k_key_defined) ? 1 : 0;
            db.output[id] = (k.second & k_key_output) ? 1 : 0;
        }
        db.defined.resize(db.key_names.size(), 0);
        db.output.resize(db.key_names.size(), 0);
        if (!db.keep(K_DIAC) || !db.keep(K_LEX)) throw data_error("binary database: diac and lex must be kept");
    }
    auto key_of = [&](uint32_t index) {
        if (index >= key_map.size()) throw data_error("binary database: key index");
        return key_map[index];
    };

    // CATS
    {
        bin_reader r(sec[2].p, sec[2].n, "CATS");
        db.n_cats = r.u32();
        db.cat_ids = r.take_array(db.n_cats, 4);
        for (uint32_t i = 0; i < db.n_cats; ++i) string_id(rd32(db.cat_ids + 4 * size_t(i)));
    }

    // DFLT
    {
        bin_reader r(sec[3].p, sec[3].n, "DFLT");
        const uint32_t count = r.u32();
        for (uint32_t i = 0; i < count; ++i) {
            const std::string pos(db.str(string_id(r.u32())));
            const uint32_t nf = r.u32();
            if (nf > k_max_entry_feats) throw data_error("binary database: default size");
            result_t parsed;
            for (uint32_t k = 0; k < nf; ++k) {
                const uint32_t key = key_of(r.u32());
                const uint32_t v = r.u32();
                set(parsed, key, v == k_bin_none ? none_value() : str_value(std::string(db.str(string_id(v)))));
            }
            db.defaults[pos] = std::move(parsed);
        }
    }

    // BKOF
    {
        bin_reader r(sec[4].p, sec[4].n, "BKOF");
        const uint32_t count = r.u32();
        for (uint32_t i = 0; i < count; ++i) {
            const std::string name(db.str(string_id(r.u32())));
            const uint32_t nc = r.u32();
            if (nc > 65536) throw data_error("binary database: backoff size");
            std::vector<std::string> cats;
            for (uint32_t k = 0; k < nc; ++k) cats.emplace_back(db.str(string_id(r.u32())));
            db.stem_backoffs[name] = std::move(cats);
        }
    }

    // Entry tables.
    uint32_t base = 0;
    bin_table * tables[3] = { &db.prefix, &db.suffix, &db.stem };
    for (int t = 0; t < 3; ++t) {
        bin_table & tb = *tables[t];
        bin_reader r(sec[5 + t].p, sec[5 + t].n, tags[5 + t]);
        tb.rows = r.u32();
        tb.forms = r.u32();
        const uint32_t n_cols = r.u32();
        tb.row_bytes = r.u32();
        if (n_cols == 0 || n_cols > 4096 || tb.row_bytes == 0 || tb.forms > tb.rows ||
            uint64_t(base) + tb.rows > k_max_entries)
            throw data_error("binary database: table header");
        tb.base = base;
        base += tb.rows;
        const uint8_t * desc = r.take_array(n_cols, 16);
        bool have_cat = false;
        std::unordered_set<uint32_t> keys_seen;
        for (uint32_t c = 0; c < n_cols; ++c) {
            bin_column col;
            const uint32_t key = rd32(desc + 16 * size_t(c));
            col.width = rd32(desc + 16 * size_t(c) + 4);
            col.offset = rd32(desc + 16 * size_t(c) + 8);
            col.dict_count = rd32(desc + 16 * size_t(c) + 12);
            if ((col.width != 1 && col.width != 2 && col.width != 4) || uint64_t(col.offset) + col.width > tb.row_bytes)
                throw data_error("binary database: column layout");
            const uint64_t absent = col.width == 4 ? 0xFFFFFFFFull : (1ull << (8 * col.width)) - 1;
            if (col.dict_count >= absent) throw data_error("binary database: column dictionary");
            col.dict = r.take_array(col.dict_count, 4);
            for (uint32_t v = 0; v < col.dict_count; ++v) {
                const uint32_t id = rd32(col.dict + 4 * size_t(v));
                if (key == k_bin_cat_column ? id >= db.n_cats : id >= db.n_strings)
                    throw data_error("binary database: column value id");
            }
            if (key == k_bin_cat_column) {
                if (have_cat) throw data_error("binary database: two category columns");
                have_cat = true;
                col.key = key;
                tb.cat = col;
            } else {
                col.key = key_of(key);
                if (!keys_seen.insert(col.key).second) throw data_error("binary database: duplicate column");
                tb.columns.push_back(col);
            }
        }
        if (!have_cat) throw data_error("binary database: no category column");
        r.align8();
        tb.index = r.take_array(tb.forms, 12);
        uint32_t next = 0;
        for (uint32_t i = 0; i < tb.forms; ++i) {
            const uint8_t * e = tb.index + 12 * size_t(i);
            const uint32_t form = string_id(rd32(e));
            if (i && form <= rd32(e - 12)) throw data_error("binary database: form index order");
            if (rd32(e + 4) != next || rd32(e + 8) == 0 || rd32(e + 8) > tb.rows - next)
                throw data_error("binary database: form index rows");
            next += rd32(e + 8);
            if (cp_count(db.str(form)) > k_max_form_cps) throw data_error("binary database: form too long");
        }
        if (next != tb.rows) throw data_error("binary database: form index coverage");
        r.align8();
        tb.data = r.take_array(tb.rows, tb.row_bytes);
        for (uint32_t row = 0; row < tb.rows; ++row) {
            if (bin_db::local(tb, row, tb.cat) >= tb.cat.dict_count) throw data_error("binary database: row category");
            for (const bin_column & c : tb.columns) {
                const uint32_t v = bin_db::local(tb, row, c);
                const uint32_t absent = c.width == 4 ? 0xFFFFFFFFu : (1u << (8 * c.width)) - 1;
                if (v >= c.dict_count && v != absent) throw data_error("binary database: row value");
            }
        }
    }
    for (uint32_t i = 0; i < db.prefix.forms; ++i)
        db.max_prefix_size = std::max(db.max_prefix_size, cp_count(db.str(rd32(db.prefix.index + 12 * size_t(i)))));
    for (uint32_t i = 0; i < db.suffix.forms; ++i)
        db.max_suffix_size = std::max(db.max_suffix_size, cp_count(db.str(rd32(db.suffix.index + 12 * size_t(i)))));

    // Compatibility tables.
    for (int t = 0; t < 3; ++t) {
        bin_reader r(sec[8 + t].p, sec[8 + t].n, tags[8 + t]);
        db.compat_n[t] = r.u32();
        r.u32(); // padding
        db.compat[t] = r.take_array(db.compat_n[t], 8);
        for (uint32_t i = 0; i < db.compat_n[t]; ++i) {
            const uint64_t v = rd64(db.compat[t] + 8 * size_t(i));
            if (uint32_t(v >> 32) >= db.n_cats || uint32_t(v) >= db.n_cats ||
                (i && v <= rd64(db.compat[t] + 8 * size_t(i) - 8)))
                throw data_error("binary database: compatibility table");
        }
    }
}

// ---------------------------------------------------------------------------
// Analyzer.

enum class backoff_t { none, noan_all, noan_prop, add_all, add_prop };

// The analyzer behind the C ABI, whatever the database representation.
class analyzer_base {
public:
    virtual ~analyzer_base() = default;
    virtual std::vector<result_t> analyze(const u32 & raw) const = 0;
    virtual const db_common & common() const = 0;
};

// camel_tools' Analyzer over a text (morph_db) or binary (bin_db) database.
// With the binary database only the profile features are stored; every
// assignment of a dropped feature is skipped (keep()), which leaves the kept
// features unchanged because no kept value is computed from a dropped one.
template <typename DB> class analyzer final : public analyzer_base {
public:
    analyzer(std::unique_ptr<DB> db, backoff_t backoff) : db_(std::move(db)), backoff_(backoff) {}

    const db_common & common() const override { return *db_; }

    std::vector<result_t> analyze(const u32 & raw) const override {
        const u32 word = py_str::strip(raw);
        std::vector<result_t> analyses;
        if (word.empty()) return analyses;

        const u32 word_dediac = dediac_ar(word);
        const u32 word_normal = normalize(word_dediac);

        if (is_digit_word(word)) return { special(word, "digit", "/NOUN_NUM", "digit", "DIGIT", "NOM", "NUM", false) };
        if (all_class(word, is_punct_symbol))
            return { special(word, "punc", "/PUNC", "punc", "PUNC", "PNX", "PUNCT", false) };
        bool has_punc = false;
        for (char32_t c : word) has_punc = has_punc || is_punct_symbol(c);
        if (!has_punc) {
            if (!all_class(word, [](char32_t c) { return is_ar_letter(c) || is_ar_diac(c); }))
                return { special(word, "latin", "/FOREIGN", "foreign", "FOREIGN", "FOREIGN", "X", true) };
        }

        // UTF-8 of word_normal with code point offsets, for hash lookups.
        std::string wn;
        std::vector<size_t> off;
        off.reserve(word_normal.size() + 1);
        for (char32_t c : word_normal) {
            off.push_back(wn.size());
            py_str::append_utf8(wn, c);
        }
        off.push_back(wn.size());
        auto sub = [&](size_t a, size_t b) { return wn.substr(off[a], off[b] - off[a]); };
        // camel_tools' tokenization keys are only concatenated strings; with
        // the binary database they are dropped and never computed.

        std::string wd_no_tatweel;
        for (char32_t c : word_dediac)
            if (c != 0x0640) py_str::append_utf8(wd_no_tatweel, c);

        const long long w = (long long)word_normal.size();
        const long long max_p = (long long)db_->max_prefix_size, max_s = (long long)db_->max_suffix_size;
        if (!has_punc) {
            for (long long p = 0; p <= std::min(max_p, w - 1); ++p) {
                for (long long s = std::max(1LL, w - p - max_s); s <= w - p; ++s) {
                    const elist pre = db_->prefixes(sub(0, size_t(p)));
                    const elist suf = db_->suffixes(sub(size_t(p + s), size_t(w)));
                    if (pre.empty() || suf.empty()) continue;
                    const elist stem = db_->stems(sub(size_t(p), size_t(p + s)));
                    if (stem.empty()) continue;
                    combined(wd_no_tatweel, pre, stem, suf, analyses);
                }
            }
        }

        const bool noan = backoff_ == backoff_t::noan_all || backoff_ == backoff_t::noan_prop;
        const bool add = backoff_ == backoff_t::add_all || backoff_ == backoff_t::add_prop;
        if ((noan && analyses.empty()) || add) {
            const bool prop = backoff_ == backoff_t::noan_prop || backoff_ == backoff_t::add_prop;
            auto bo = db_->stem_backoffs.find(prop ? "PROP" : "ALL");
            if (bo == db_->stem_backoffs.end()) throw input_error("camel_tools raises KeyError: stem backoff mode");
            const elist noan_stems = db_->stems("NOAN");
            if (noan_stems.empty()) throw input_error("camel_tools raises KeyError: 'NOAN'");
            std::vector<uint32_t> stem_ids;
            for (uint32_t k = 0; k < noan_stems.size(); ++k) {
                const uint32_t e = noan_stems[k];
                const std::string_view cat = db_->cat_name(db_->cat(e));
                if (std::find(bo->second.begin(), bo->second.end(), cat) != bo->second.end()) stem_ids.push_back(e);
            }
            elist stems;
            stems.ids = stem_ids.data();
            stems.n = uint32_t(stem_ids.size());
            for (long long p = 0; p <= std::min(max_p, w - 1); ++p) {
                for (long long s = std::max(1LL, w - p - max_s); s <= w - p; ++s) {
                    const elist pre = db_->prefixes(sub(0, size_t(p)));
                    const elist suf = db_->suffixes(sub(size_t(p + s), size_t(w)));
                    if (pre.empty() || suf.empty()) continue;
                    const u32 stem = word_normal.substr(size_t(p), size_t(s));
                    combined_backoff(stem, prop, pre, stems, suf, analyses);
                }
            }
        }
        return analyses;
    }

private:
    std::unique_ptr<DB> db_;
    backoff_t backoff_;

    const std::string_view & need(const fview & f, uint32_t key) const {
        const std::string_view * v = fget(f, key);
        if (!v) throw input_error("camel_tools raises KeyError: '" + db_->key_names[key] + "'");
        return *v;
    }

    static std::string get_or(const fview & f, uint32_t key, std::string_view dflt) {
        const std::string_view * v = fget(f, key);
        return std::string(v ? *v : dflt);
    }

    const value & need(result_t & r, uint32_t key) const {
        rfeat * f = find(r, key);
        if (!f) throw input_error("camel_tools raises KeyError: '" + db_->key_names[key] + "'");
        return f->v;
    }

    // set() for keys the database outputs; a dropped key is never stored.
    void put(result_t & r, uint32_t key, value v) const {
        if (db_->keep(key)) set(r, key, std::move(v));
    }

    // Digit, punctuation and foreign words.
    result_t special(const u32 & word_u, const char * dkey, const char * bw_suffix, const char * source,
                     const char * undefined_lex, const char * catib6, const char * ud, bool foreign) const {
        auto it = db_->defaults.find(dkey);
        if (it == db_->defaults.end()) throw input_error(std::string("camel_tools raises KeyError: '") + dkey + "'");
        result_t r = it->second;
        const std::string word = to_utf8(word_u);
        if (foreign) put(r, K_POS, str_value("foreign"));
        put(r, K_DIAC, str_value(word));
        put(r, K_STEM, str_value(word));
        put(r, K_STEMGLOSS, str_value(word));
        put(r, K_STEMCAT, none_value());
        put(r, K_LEX, str_value(word));
        put(r, K_BW, str_value(word + bw_suffix));
        put(r, K_SOURCE, str_value(source));
        for (uint32_t k : k_copy_feats)
            if (db_->def(k)) put(r, k, str_value(word));
        for (uint32_t k : k_undefined_lex_feats)
            if (db_->def(k)) put(r, k, str_value(undefined_lex));
        if (db_->def(K_CATIB6)) put(r, K_CATIB6, str_value(catib6));
        if (db_->def(K_UD)) put(r, K_UD, str_value(ud));
        put(r, K_POS_LOGPROB, float_value(-99.0));
        put(r, K_LEX_LOGPROB, float_value(-99.0));
        put(r, K_POS_LEX_LOGPROB, float_value(-99.0));
        fix_form(r);
        return r;
    }

    static bool is_dash(const value & v) { return v.kind == value::str_t && v.s == "-"; }

    void fix_form(result_t & r) const {
        if (db_->def(K_FORM_GEN) && is_dash(need(r, K_GEN))) {
            value v = need(r, K_FORM_GEN);
            set(r, K_GEN, std::move(v));
        }
        if (db_->def(K_FORM_NUM) && is_dash(need(r, K_NUM))) {
            value v = need(r, K_FORM_NUM);
            set(r, K_NUM, std::move(v));
        }
    }

    static std::string utf8_of(const value & v) { return v.kind == value::str_t ? v.s : std::string(); }

    // merge_features(db, prefix_feats, stem_feats, suffix_feats, 'AF')
    result_t merge(const fview & P, const fview & S, const fview & X) const {
        result_t r;
        r.reserve(S.size() + 16);
        for (const auto & kv : S) r.push_back({ kv.first, str_value(std::string(kv.second)) });
        for (const auto & kv : S) {
            const std::string_view * sv = fget(X, kv.first);
            if (sv && *sv != "-" && !sv->empty()) set(r, kv.first, str_value(std::string(*sv)));
            const std::string_view * pv = fget(P, kv.first);
            if (pv && *pv != "-" && !pv->empty()) set(r, kv.first, str_value(std::string(*pv)));
        }
        for (uint32_t k : k_join_feats) {
            if (!db_->def(k)) continue;
            std::string joined;
            for (const fview * f : { &P, &S, &X }) {
                const std::string_view * v = fget(*f, k);
                if (!v || v->empty()) continue;
                append_join(joined, *v);
            }
            set(r, k, str_value(finish_join(joined)));
        }
        for (uint32_t k : k_concat_feats) {
            if (!db_->def(k)) continue;
            std::string joined;
            for (const fview * f : { &P, &S, &X }) {
                const std::string_view * v = fget(*f, k);
                if (!v || v->empty()) continue;
                append_join(joined, *v);
            }
            set(r, k, str_value(finish_join(joined)));
        }
        const std::string_view * sdiac = fget(S, K_DIAC);
        for (uint32_t k : k_concat_feats_none) {
            if (!db_->def(k)) continue;
            std::string v = get_or(P, k, "");
            const std::string_view * sv = fget(S, k);
            v += sv ? std::string(*sv) : std::string(sdiac ? *sdiac : std::string_view());
            v += get_or(X, k, "");
            set(r, k, str_value(std::move(v)));
        }
        if (db_->keep(K_STEM)) set(r, K_STEM, str_value(std::string(need(S, K_DIAC))));
        put(r, K_STEMGLOSS, str_value(get_or(S, K_GLOSS, "")));
        {
            const value & d = need(r, K_DIAC);
            set(r, K_DIAC, str_value(to_utf8(rewrite_diac(from_utf8_valid(utf8_of(d))))));
        }
        for (uint32_t k : k_tok_schemes_1) {
            if (!db_->def(k)) continue;
            rfeat * f = find(r, k);
            const std::string cur = f ? utf8_of(f->v) : std::string();
            set(r, k, str_value(to_utf8(rewrite_tok_1(from_utf8_valid(cur)))));
        }
        for (uint32_t k : k_tok_schemes_2) {
            if (!db_->def(k)) continue;
            rfeat * f = find(r, k);
            const std::string cur = f ? utf8_of(f->v) : std::string();
            set(r, k, str_value(to_utf8(rewrite_tok_2(from_utf8_valid(cur)))));
        }
        if (db_->def(K_CAPHI)) {
            rfeat * f = find(r, K_CAPHI);
            const std::string cur = f ? utf8_of(f->v) : std::string();
            set(r, K_CAPHI, str_value(to_utf8(rewrite_caphi(from_utf8_valid(cur)))));
        }
        fix_form(r);
        if (db_->pattern_in_order) {
            std::string v = get_or(P, K_DIAC, "");
            const std::string_view * sp = fget(S, K_PATTERN);
            v += sp ? std::string(*sp) : std::string(sdiac ? *sdiac : std::string_view());
            v += get_or(X, K_DIAC, "");
            set(r, K_PATTERN, str_value(to_utf8(rewrite_pattern(from_utf8_valid(v)))));
        }
        for (uint32_t k : k_logprob_feats) {
            if (!db_->def(k)) continue;
            rfeat * f = find(r, k);
            double d = -99.0;
            if (f) {
                if (f->v.kind == value::float_t)
                    d = f->v.f;
                else if (f->v.kind != value::str_t || !parse_plain_decimal(f->v.s, d))
                    throw input_error("camel_tools float() rejects a *_logprob value");
            }
            set(r, k, float_value(d));
        }
        return r;
    }

    // '+'.join helper: a leading marker byte records "non-empty list".
    static void append_join(std::string & joined, std::string_view v) {
        if (joined.empty())
            joined.push_back('\x01');
        else
            joined.push_back('+');
        joined.append(v.data(), v.size());
    }

    static std::string finish_join(const std::string & joined) {
        return joined.empty() ? std::string() : joined.substr(1);
    }

    void combined(const std::string & wd_no_tatweel, const elist & prefixes, const elist & stems,
                  const elist & suffixes, std::vector<result_t> & out) const {
        for (uint32_t pi = 0; pi < prefixes.size(); ++pi) {
            const uint32_t pe = prefixes[pi];
            const uint32_t pcat = db_->cat(pe);
            fview P;
            bool have_p = false;
            for (uint32_t si = 0; si < stems.size(); ++si) {
                const uint32_t se = stems[si];
                const uint32_t scat = db_->cat(se);
                if (!db_->ab_has_cat(pcat)) throw input_error("camel_tools raises KeyError: prefix category");
                if (!db_->ab_ok(pcat, scat)) continue;
                fview S;
                bool have_s = false;
                for (uint32_t xi = 0; xi < suffixes.size(); ++xi) {
                    const uint32_t xe = suffixes[xi];
                    const uint32_t xcat = db_->cat(xe);
                    if (!db_->bc_has_cat(scat) || !db_->ac_has_cat(pcat) || !db_->bc_ok(scat, xcat) ||
                        !db_->ac_ok(pcat, xcat))
                        continue;
                    if (!have_p) {
                        P = db_->view(pe);
                        have_p = true;
                    }
                    if (!have_s) {
                        S = db_->view(se);
                        have_s = true;
                    }
                    const fview X = db_->view(xe);
                    result_t merged = merge(P, S, X);
                    if (db_->keep(K_STEM)) set(merged, K_STEM, str_value(std::string(need(S, K_DIAC))));
                    if (db_->keep(K_STEMCAT)) set(merged, K_STEMCAT, str_value(std::string(db_->cat_name(scat))));
                    const std::string merged_dediac = dediac_ar_utf8(utf8_of(need(merged, K_DIAC)));
                    if (wd_no_tatweel != merged_dediac) put(merged, K_SOURCE, str_value("spvar"));
                    out.push_back(std::move(merged));
                }
            }
        }
    }

    void combined_backoff(const u32 & stem, bool prop, const elist & prefixes, const elist & stems,
                          const elist & suffixes, std::vector<result_t> & out) const {
        std::string expanded;
        bool expanded_ready = false;
        for (uint32_t pi = 0; pi < prefixes.size(); ++pi) {
            const uint32_t pe = prefixes[pi];
            const uint32_t pcat = db_->cat(pe);
            const fview P = db_->view(pe);
            for (uint32_t si = 0; si < stems.size(); ++si) {
                const uint32_t se = stems[si];
                const uint32_t scat = db_->cat(se);
                // stem_feats = copy.copy(...): owned, mutated across suffixes.
                std::vector<std::pair<uint32_t, std::string>> sf;
                for (const auto & kv : db_->view(se)) sf.emplace_back(kv.first, std::string(kv.second));
                auto sget = [&](uint32_t key) -> std::string * {
                    for (auto & kv : sf)
                        if (kv.first == key) return &kv.second;
                    return nullptr;
                };
                auto sneed = [&](uint32_t key) -> std::string & {
                    std::string * v = sget(key);
                    if (!v) throw input_error("camel_tools raises KeyError: '" + db_->key_names[key] + "'");
                    return *v;
                };
                if (!db_->ab_has_cat(pcat)) throw input_error("camel_tools raises KeyError: prefix category");
                if (!db_->ab_ok(pcat, scat)) continue;
                for (uint32_t xi = 0; xi < suffixes.size(); ++xi) {
                    const uint32_t xe = suffixes[xi];
                    const uint32_t xcat = db_->cat(xe);
                    if (!db_->bc_has_cat(scat)) throw input_error("camel_tools raises KeyError: stem category");
                    if (!db_->bc_ok(scat, xcat) || !db_->ac_has_cat(pcat) || !db_->ac_ok(pcat, xcat)) continue;
                    if (prop && sneed(K_BW).find("NOUN_PROP") == std::string::npos) continue;
                    if (!expanded_ready) {
                        expanded = expand_noan_template(stem);
                        expanded_ready = true;
                    }
                    for (uint32_t k : { uint32_t(K_BW), uint32_t(K_DIAC), uint32_t(K_LEX) }) {
                        std::string & v = sneed(k);
                        v = replace_all(v, "NOAN", expanded);
                    }
                    if (db_->keep(K_CAPHI)) {
                        if (std::string * c = sget(K_CAPHI))
                            *c = simple_ar_to_caphi(stem);
                        else
                            sf.emplace_back(K_CAPHI, simple_ar_to_caphi(stem));
                    }
                    fview S;
                    S.reserve(sf.size());
                    for (const auto & kv : sf) S.emplace_back(kv.first, std::string_view(kv.second));
                    const fview X = db_->view(xe);
                    result_t merged = merge(P, S, X);
                    if (db_->keep(K_STEM)) set(merged, K_STEM, str_value(sneed(K_DIAC)));
                    if (db_->keep(K_STEMCAT)) set(merged, K_STEMCAT, str_value(std::string(db_->cat_name(scat))));
                    put(merged, K_SOURCE, str_value("backoff"));
                    put(merged, K_PATTERN, str_value("backoff"));
                    // The binary converter requires gloss on every NOAN stem,
                    // so skipping this dropped key cannot hide a KeyError.
                    if (db_->keep(K_GLOSS)) set(merged, K_GLOSS, str_value(sneed(K_GLOSS)));
                    out.push_back(std::move(merged));
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Files, hashes and the bundle manifest.

bool is_sha256_hex(const char * s) {
    if (!s || std::strlen(s) != 64) return false;
    for (size_t i = 0; i < 64; ++i)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
    return true;
}

std::string read_bounded(const std::string & path, size_t max_bytes) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw data_error("cannot open " + path);
    std::string out;
    char buf[1 << 16];
    for (;;) {
        const size_t n = std::fread(buf, 1, sizeof buf, f);
        if (out.size() + n > max_bytes) {
            std::fclose(f);
            throw data_error("file exceeds its size bound: " + path);
        }
        out.append(buf, n);
        if (n < sizeof buf) break;
    }
    const bool ok = !std::ferror(f);
    std::fclose(f);
    if (!ok) throw data_error("cannot read " + path);
    return out;
}

std::string dirname_of(const std::string & path) {
    const size_t p = path.find_last_of("/\\");
    return p == std::string::npos ? std::string() : path.substr(0, p + 1);
}

bool plain_file_name(const std::string & name) {
    if (name.empty() || name == "." || name == "..") return false;
    for (char c : name)
        if (c == '/' || c == '\\' || c == ':' || c == '\0') return false;
    return true;
}

// Read-only bytes of a whole file: a private read-only mapping (POSIX mmap,
// Windows MapViewOfFile) or, on hosts without one (Emscripten) and with
// CRISPEMBED_AR_MORPH_MMAP=0, one heap buffer.
struct file_bytes {
    std::shared_ptr<const void> owner;
    const uint8_t * data = nullptr;
    size_t size = 0;
    bool mapped = false;
};

file_bytes read_file_heap(const std::string & path, size_t max_bytes) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw data_error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    const long end = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (end < 0 || size_t(end) > max_bytes) {
        std::fclose(f);
        throw data_error("file exceeds its size bound: " + path);
    }
    const size_t n = size_t(end);
    std::shared_ptr<uint8_t> buf(static_cast<uint8_t *>(std::malloc(n ? n : 1)), [](uint8_t * p) { std::free(p); });
    if (!buf) {
        std::fclose(f);
        throw std::bad_alloc();
    }
    const size_t got = n ? std::fread(buf.get(), 1, n, f) : 0;
    const bool ok = got == n && !std::ferror(f);
    std::fclose(f);
    if (!ok) throw data_error("cannot read " + path);
    file_bytes out;
    out.data = buf.get();
    out.size = n;
    out.owner = std::move(buf);
    return out;
}

file_bytes map_file(const std::string & path, size_t max_bytes) {
#if defined(_WIN32)
    // CRISPEMBED_AR_MORPH_MMAP=0 reads the binary database into one heap buffer instead of mapping it.
    if (!core_env::explicitly_off("CRISPEMBED_AR_MORPH_MMAP")) {
        const int wn = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, nullptr, 0);
        if (wn <= 0) throw data_error("cannot open " + path);
        std::wstring wpath(size_t(wn), L'\0');
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.c_str(), -1, &wpath[0], wn);
        HANDLE file = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw data_error("cannot open " + path);
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(file, &sz) || sz.QuadPart <= 0 || uint64_t(sz.QuadPart) > max_bytes) {
            CloseHandle(file);
            throw data_error("file is empty or exceeds its size bound: " + path);
        }
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(file);
        if (!mapping) throw data_error("cannot map " + path);
        void * view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mapping);
        if (!view) throw data_error("cannot map " + path);
        file_bytes out;
        out.owner = std::shared_ptr<const void>(view, [](const void * p) { UnmapViewOfFile(p); });
        out.data = static_cast<const uint8_t *>(view);
        out.size = size_t(sz.QuadPart);
        out.mapped = true;
        return out;
    }
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
    // CRISPEMBED_AR_MORPH_MMAP=0 reads the binary database into one heap buffer instead of mapping it.
    if (!core_env::explicitly_off("CRISPEMBED_AR_MORPH_MMAP")) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw data_error("cannot open " + path);
        struct stat st {};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0 || uint64_t(st.st_size) > max_bytes) {
            ::close(fd);
            throw data_error("file is empty or exceeds its size bound: " + path);
        }
        const size_t n = size_t(st.st_size);
        void * p = ::mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) throw data_error("cannot map " + path);
        file_bytes out;
        out.owner = std::shared_ptr<const void>(p, [n](const void * q) { ::munmap(const_cast<void *>(q), n); });
        out.data = static_cast<const uint8_t *>(p);
        out.size = n;
        out.mapped = true;
        return out;
    }
#endif
    return read_file_heap(path, max_bytes);
}

bool has_bin_magic(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char head[8] = {};
    const size_t n = std::fread(head, 1, 8, f);
    std::fclose(f);
    return n == 8 && std::memcmp(head, k_bin_magic, 8) == 0;
}

struct opened {
    std::unique_ptr<analyzer_base> an;
    std::string info;
};

backoff_t to_backoff(int b) {
    switch (b) {
    case CRISPEMBED_AR_MORPH_BACKOFF_NONE:
        return backoff_t::none;
    case CRISPEMBED_AR_MORPH_BACKOFF_NOAN_ALL:
        return backoff_t::noan_all;
    case CRISPEMBED_AR_MORPH_BACKOFF_NOAN_PROP:
        return backoff_t::noan_prop;
    case CRISPEMBED_AR_MORPH_BACKOFF_ADD_ALL:
        return backoff_t::add_all;
    case CRISPEMBED_AR_MORPH_BACKOFF_ADD_PROP:
        return backoff_t::add_prop;
    default:
        throw std::invalid_argument("invalid backoff mode");
    }
}

const char * backoff_name(int b) {
    static const char * const names[] = { "NONE", "NOAN_ALL", "NOAN_PROP", "ADD_ALL", "ADD_PROP" };
    return names[b];
}

template <typename DB>
opened finish_build(std::unique_ptr<DB> db, const char * format, const std::string & db_sha256, int backoff,
                    const std::string & variant, const std::string & manifest_sha256, const char * storage) {
    std::string info = "{\"abi_version\":" + std::to_string(CRISPEMBED_AR_MORPH_ABI_VERSION) + ",\"variant\":\"" +
                       core_json::json_escape_strict(variant) + "\",\"db_format\":\"" + format + "\",\"db_sha256\":\"" +
                       db_sha256 + "\"";
    if (!manifest_sha256.empty()) info += ",\"manifest_sha256\":\"" + manifest_sha256 + "\"";
    info += ",\"backoff\":\"" + std::string(backoff_name(backoff)) +
            "\",\"prefix_forms\":" + std::to_string(db->prefix_forms()) +
            ",\"suffix_forms\":" + std::to_string(db->suffix_forms()) +
            ",\"stem_forms\":" + std::to_string(db->stem_forms()) +
            ",\"stem_entries\":" + std::to_string(db->stem_entries()) + ",\"unicode_version\":\"" +
            arabic_morph_tables::k_unicode_version + "\",\"storage\":\"" + storage + "\",\"keys\":[";
    for (size_t k = 0; k < db->key_names.size(); ++k) {
        if (k) info += ',';
        info += '"' + core_json::json_escape_strict(db->key_names[k]) + '"';
    }
    info += "],\"profile\":";
    if (db->output.empty()) {
        info += "null";
    } else {
        info += '[';
        bool first = true;
        for (size_t k = 0; k < db->key_names.size(); ++k) {
            if (!db->keep(uint32_t(k))) continue;
            if (!first) info += ',';
            first = false;
            info += '"' + core_json::json_escape_strict(db->key_names[k]) + '"';
        }
        info += ']';
    }
    info += '}';
    opened o;
    o.an = std::make_unique<analyzer<DB>>(std::move(db), to_backoff(backoff));
    o.info = std::move(info);
    return o;
}

opened build(std::string db_text, const std::string & db_sha256, int backoff, const std::string & variant,
             const std::string & manifest_sha256) {
    to_backoff(backoff);
    auto db = std::make_unique<morph_db>();
    db_parser(std::move(db_text), *db).parse();
    return finish_build(std::move(db), "text", db_sha256, backoff, variant, manifest_sha256, "heap");
}

opened build_bin(file_bytes bytes, const std::string & db_sha256, int backoff, const std::string & variant,
                 const std::string & manifest_sha256) {
    to_backoff(backoff);
    auto db = std::make_unique<bin_db>();
    db->owner = bytes.owner;
    db->base = bytes.data;
    db->size = bytes.size;
    load_bin(*db);
    return finish_build(std::move(db), "binary", db_sha256, backoff, variant, manifest_sha256,
                        bytes.mapped ? "mapped" : "heap");
}

void put_error(char * error, size_t cap, const std::string & msg) {
    if (!error || cap == 0) return;
    const size_t n = std::min(cap - 1, msg.size());
    std::memcpy(error, msg.data(), n);
    error[n] = '\0';
}

void append_json(std::string & out, const db_common & db, const std::vector<result_t> & analyses) {
    out.push_back('[');
    for (size_t a = 0; a < analyses.size(); ++a) {
        if (a) out.push_back(',');
        out.push_back('{');
        bool first = true;
        for (const rfeat & f : analyses[a]) {
            if (!db.keep(f.key)) continue;
            if (!first) out.push_back(',');
            first = false;
            out.push_back('"');
            out += core_json::json_escape_strict(db.key_names[f.key]);
            out += "\":";
            switch (f.v.kind) {
            case value::str_t:
                out.push_back('"');
                out += core_json::json_escape_strict(f.v.s);
                out.push_back('"');
                break;
            case value::none_t:
                out += "null";
                break;
            case value::float_t:
                out += py_str::float_repr(f.v.f);
                break;
            }
        }
        out.push_back('}');
    }
    out.push_back(']');
}

void put_u32(std::string & out, uint32_t v) {
    char b[4] = { char(v & 0xFF), char((v >> 8) & 0xFF), char((v >> 16) & 0xFF), char((v >> 24) & 0xFF) };
    out.append(b, 4);
}

void put_u16(std::string & out, uint32_t v) {
    out.push_back(char(v & 0xFF));
    out.push_back(char((v >> 8) & 0xFF));
}

// One word's compact record (see crispembed_ar_morph_analyze_compact).
// `kept`, when given, receives the index of every analysis written (all of
// them unless dedup dropped some).
void append_compact(std::string & out, const db_common & db, const std::vector<result_t> & analyses, bool dedup,
                    std::vector<uint32_t> * kept = nullptr) {
    constexpr uint32_t none_ref = 0xFFFFFFFFu;
    std::unordered_map<std::string, uint32_t> index; // kind byte + text
    std::vector<const std::string *> strings;
    std::vector<uint8_t> kinds;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> rows;
    std::unordered_set<std::string> seen;
    uint32_t max_key = 0;
    std::string scratch;
    if (kept) kept->clear();
    for (size_t ai = 0; ai < analyses.size(); ++ai) {
        const result_t & r = analyses[ai];
        std::vector<std::pair<uint32_t, uint32_t>> row;
        row.reserve(r.size());
        for (const rfeat & f : r) {
            if (!db.keep(f.key)) continue;
            uint32_t ref = none_ref;
            if (f.v.kind != value::none_t) {
                scratch.assign(1, f.v.kind == value::float_t ? '\1' : '\0');
                if (f.v.kind == value::float_t)
                    scratch += py_str::float_repr(f.v.f);
                else
                    scratch += f.v.s;
                auto it = index.find(scratch);
                if (it == index.end()) {
                    it = index.emplace(scratch, uint32_t(strings.size())).first;
                    strings.push_back(&it->first);
                    kinds.push_back(uint8_t(scratch[0]));
                }
                ref = it->second;
            }
            max_key = std::max(max_key, f.key);
            row.emplace_back(f.key, ref);
        }
        if (dedup) {
            std::string sig;
            for (const auto & kv : row) {
                put_u32(sig, kv.first);
                put_u32(sig, kv.second);
            }
            if (!seen.insert(std::move(sig)).second) continue;
        }
        if (kept) kept->push_back(uint32_t(ai));
        rows.push_back(std::move(row));
    }
    if (rows.size() > 0xFFFFFFFFull || strings.size() >= 0xFFFFFFFFull) throw std::runtime_error("result too large");
    const bool wide = strings.size() >= 0xFFFF || max_key > 0xFF;
    put_u32(out, uint32_t(rows.size()));
    put_u32(out, uint32_t(strings.size()));
    out.push_back(wide ? '\1' : '\0');
    out.append(3, '\0');
    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string & s = *strings[i];
        out.push_back(char(kinds[i]));
        put_u32(out, uint32_t(s.size() - 1));
        out.append(s, 1, std::string::npos);
    }
    for (const auto & row : rows) {
        put_u16(out, uint32_t(row.size()));
        for (const auto & kv : row) {
            if (wide) {
                put_u16(out, kv.first);
                put_u32(out, kv.second);
            } else {
                out.push_back(char(kv.first));
                put_u16(out, kv.second == none_ref ? 0xFFFFu : kv.second);
            }
        }
    }
}

const std::string * check_word(std::string_view in) {
    static const std::string too_long = "word exceeds CRISPEMBED_AR_MORPH_MAX_WORD_BYTES";
    static const std::string bad_utf8 = "word is not valid UTF-8";
    if (in.size() > CRISPEMBED_AR_MORPH_MAX_WORD_BYTES) return &too_long;
    if (!valid_utf8(in)) return &bad_utf8;
    return nullptr;
}

// ---------------------------------------------------------------------------
// Context disambiguation (docs/arabic-morph-disambig-format.md).
//
// A bigram hidden Markov model over candidates. A candidate groups the
// analyses of a token that share (tag, lex, pos), where the tag joins the
// model's tag features. A sentence scores
//   prod_i T(tag_i | tag_i-1) * L(lex_i, pos_i | pos_i)
// with the boundary tag before the first and after the last token. The
// forward-backward posterior of each candidate is its score. A token without
// analyses is the single candidate (<oov>, <oov>, <oov>).

constexpr char k_dis_magic[8] = { 'C', 'E', 'M', 'D', 'I', 'S', 'A', 'M' };
constexpr uint32_t k_dis_version = 1;
constexpr size_t k_max_model_bytes = size_t(64) << 20;
constexpr uint32_t k_max_tag_features = 32;
constexpr uint32_t k_max_tags = 8192;
// Log-probability of a lexeme whose pos the model has never seen.
constexpr double k_dis_unknown_logp = -30.0;
// Probability of a transition from or to a tag the model has never seen.
constexpr double k_dis_unknown_trans = 1e-12;
constexpr uint32_t k_scores_magic = 0x31534D41u; // "AMS1" little-endian
constexpr uint32_t k_max_sentence_words = 1u << 16;

struct disambig_model {
    struct tag_feature {
        std::string name;
        bool flag = false; // presence (0/1) instead of the value
    };
    std::vector<tag_feature> spec;
    std::unordered_map<std::string, uint32_t> tag_ids;
    uint32_t n_tags = 0;
    // T(cur | prev): the listed entry, else backoff[prev] * uni[cur].
    std::vector<double> uni, backoff;
    std::unordered_map<uint64_t, double> trans;
    std::unordered_map<std::string, double> unseen; // pos -> log-probability of an unlisted lexeme
    std::unordered_map<std::string, double> lexp;   // pos '\t' lex -> log-probability
    std::string info;
};

void load_disambig(disambig_model & m, const uint8_t * f, size_t n, const std::string & sha256,
                   const std::string & manifest_sha256) {
    if (!little_endian_host()) throw data_error("disambiguation model requires a little-endian host");
    if (n < k_bin_header_bytes || std::memcmp(f, k_dis_magic, 8) != 0) throw data_error("not a disambiguation model");
    if (rd32(f + 8) != k_dis_version) throw data_error("disambiguation model: unsupported format version");
    if (rd32(f + 12) != k_bin_header_bytes) throw data_error("disambiguation model: header size");
    const uint32_t n_sections = rd32(f + 16);
    if (rd32(f + 20) != 0) throw data_error("disambiguation model: flags");
    if (rd64(f + 24) != uint64_t(n)) throw data_error("disambiguation model: file size mismatch");
    {
        uint8_t header[k_bin_header_bytes];
        std::memcpy(header, f, k_bin_header_bytes);
        std::memset(header + 36, 0, 4);
        if (crc32_bytes(header, k_bin_header_bytes) != rd32(f + 36))
            throw data_error("disambiguation model: header checksum");
    }
    if (n_sections == 0 || n_sections > 64 || uint64_t(k_bin_header_bytes) + 24ull * n_sections > n)
        throw data_error("disambiguation model: section directory");
    const uint8_t * dir = f + k_bin_header_bytes;
    if (crc32_bytes(dir, 24 * size_t(n_sections)) != rd32(f + 32))
        throw data_error("disambiguation model: directory checksum");
    static const char * const names[] = { "STRS", "SPEC", "TAGS", "TRAN", "UNSN", "LEXP", "META" };
    constexpr size_t n_names = sizeof names / sizeof names[0];
    struct section {
        const uint8_t * p = nullptr;
        size_t n = 0;
    } sec[n_names];
    for (uint32_t i = 0; i < n_sections; ++i) {
        const uint8_t * d = dir + 24 * size_t(i);
        const uint64_t off = rd64(d + 8), len = rd64(d + 16);
        if (off % 8 != 0 || off > n || len > n - off) throw data_error("disambiguation model: section bounds");
        if (crc32_bytes(f + off, size_t(len)) != rd32(d + 4))
            throw data_error("disambiguation model: section checksum");
        // The padding to the next 8-byte boundary is zero (and so covered).
        for (uint64_t q = off + len; q < n && q % 8 != 0; ++q)
            if (f[q] != 0) throw data_error("disambiguation model: section padding");
        for (size_t t = 0; t < n_names; ++t) {
            if (rd32(d) != rd32(reinterpret_cast<const uint8_t *>(names[t]))) continue;
            if (sec[t].p) throw data_error("disambiguation model: duplicate section");
            sec[t].p = f + off;
            sec[t].n = size_t(len);
        }
    }
    for (size_t t = 0; t < n_names; ++t)
        if (!sec[t].p) throw data_error(std::string("disambiguation model: missing section ") + names[t]);
    auto fail = [](const char * what) { throw data_error(std::string("disambiguation model: ") + what); };

    // STRS: unique, sorted by bytes, valid UTF-8.
    std::vector<std::string_view> strings;
    {
        bin_reader r(sec[0].p, sec[0].n, "STRS");
        const uint32_t count = r.u32();
        const uint8_t * offs = r.take_array(uint64_t(count) + 1, 4);
        const uint32_t total = rd32(offs + 4 * size_t(count));
        const char * bytes = reinterpret_cast<const char *>(r.take(total));
        uint32_t prev = 0;
        strings.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t a = rd32(offs + 4 * size_t(i)), b = rd32(offs + 4 * size_t(i) + 4);
            if ((i == 0 && a != 0) || a < prev || b < a || b > total) fail("string offsets");
            prev = a;
            const std::string_view sv(bytes + a, b - a);
            if (!valid_utf8(sv)) fail("string encoding");
            if (!strings.empty() && !(strings.back() < sv)) fail("string order");
            strings.push_back(sv);
        }
    }
    auto str = [&](uint32_t id) -> std::string_view {
        if (id >= strings.size()) fail("string id");
        return strings[id];
    };
    // SPEC: the tag features; the first is pos by value.
    {
        bin_reader r(sec[1].p, sec[1].n, "SPEC");
        const uint32_t count = r.u32();
        if (count == 0 || count > k_max_tag_features) fail("tag features");
        for (uint32_t i = 0; i < count; ++i) {
            disambig_model::tag_feature tf;
            tf.name = std::string(str(r.u32()));
            const uint32_t mode = r.u32();
            if (mode > 1) fail("tag feature mode");
            tf.flag = mode == 1;
            m.spec.push_back(std::move(tf));
        }
        if (m.spec[0].name != "pos" || m.spec[0].flag) fail("the first tag feature must be pos");
    }
    // TAGS: unique; tag 0 is the sentence boundary <s>.
    {
        bin_reader r(sec[2].p, sec[2].n, "TAGS");
        m.n_tags = r.u32();
        if (m.n_tags == 0 || m.n_tags > k_max_tags) fail("tag count");
        for (uint32_t i = 0; i < m.n_tags; ++i)
            if (!m.tag_ids.emplace(std::string(str(r.u32())), i).second) fail("duplicate tag");
        if (m.tag_ids.find("<s>") == m.tag_ids.end() || m.tag_ids.at("<s>") != 0) fail("boundary tag");
    }
    auto read_logp = [&](bin_reader & r) {
        float lp;
        std::memcpy(&lp, r.take(4), 4);
        if (!(lp <= 0.0f) || !std::isfinite(lp)) fail("log-probability");
        return double(lp);
    };
    // TRAN: unigram and backoff weights per tag, then the listed transitions
    // (prev, cur, log-probability), strictly ascending.
    {
        bin_reader r(sec[3].p, sec[3].n, "TRAN");
        if (r.u32() != m.n_tags) fail("transition size");
        m.uni.resize(m.n_tags);
        m.backoff.resize(m.n_tags);
        for (double & u : m.uni) u = std::exp(read_logp(r));
        for (double & b : m.backoff) b = std::exp(read_logp(r));
        const uint32_t count = r.u32();
        if (uint64_t(count) > uint64_t(m.n_tags) * m.n_tags) fail("transition count");
        m.trans.reserve(count);
        uint64_t prev = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t a = r.u32(), b = r.u32();
            if (a >= m.n_tags || b >= m.n_tags) fail("transition tag");
            const uint64_t k = (uint64_t(a) << 32) | b;
            if (i && k <= prev) fail("transition order");
            prev = k;
            m.trans.emplace(k, std::exp(read_logp(r)));
        }
    }
    // UNSN: per pos, the log-probability of a lexeme the model does not list.
    {
        bin_reader r(sec[4].p, sec[4].n, "UNSN");
        const uint32_t count = r.u32();
        for (uint32_t i = 0; i < count; ++i) {
            const std::string pos(str(r.u32()));
            if (!m.unseen.emplace(pos, read_logp(r)).second) fail("duplicate pos");
        }
    }
    // LEXP: (pos, lex, log-probability), strictly ascending by (pos, lex).
    {
        bin_reader r(sec[5].p, sec[5].n, "LEXP");
        const uint32_t count = r.u32();
        uint64_t prev = 0;
        m.lexp.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t pos = r.u32(), lex = r.u32();
            const uint64_t k = (uint64_t(pos) << 32) | lex;
            if (i && k <= prev) fail("lexeme order");
            prev = k;
            std::string key(str(pos));
            key += '\t';
            key += str(lex);
            m.lexp.emplace(std::move(key), read_logp(r));
        }
    }
    // META: the training description, a JSON object.
    const std::string meta(reinterpret_cast<const char *>(sec[6].p), sec[6].n);
    try {
        if (core_json_value::parse(meta).kind != core_json_value::value::object_t) fail("META");
    } catch (const data_error &) {
        throw;
    } catch (const std::exception &) {
        fail("META");
    }
    m.info = "{\"abi_version\":" + std::to_string(CRISPEMBED_AR_MORPH_ABI_VERSION) +
             ",\"model_format\":\"crispembed-camel-morph-disambig-v1\",\"model_sha256\":\"" + sha256 + "\"";
    if (!manifest_sha256.empty()) m.info += ",\"manifest_sha256\":\"" + manifest_sha256 + "\"";
    m.info += ",\"tags\":" + std::to_string(m.n_tags) + ",\"transitions\":" + std::to_string(m.trans.size()) +
              ",\"lexemes\":" + std::to_string(m.lexp.size()) + ",\"tag_features\":[";
    for (size_t i = 0; i < m.spec.size(); ++i) {
        if (i) m.info += ',';
        m.info += '"' + core_json::json_escape_strict(m.spec[i].name) + (m.spec[i].flag ? ":flag" : "") + '"';
    }
    m.info += "],\"meta\":" + meta + '}';
}

// One token's candidates and their posteriors.
struct dis_token {
    std::vector<uint32_t> cand_of; // per analysis
    std::vector<int64_t> tag;      // per candidate; -1 = unknown to the model
    std::vector<double> emit;      // per candidate
    std::vector<double> alpha, beta, post;
};

void build_candidates(const disambig_model & m, const db_common & db, const std::vector<result_t> & analyses,
                      dis_token & t) {
    static const std::string absent = "-";
    std::vector<uint32_t> keys;
    keys.reserve(m.spec.size());
    for (const auto & tf : m.spec) {
        auto it = db.key_ids.find(tf.name);
        keys.push_back(it == db.key_ids.end() ? 0xFFFFFFFFu : it->second);
    }
    const auto lex_it = db.key_ids.find("lex");
    const auto pos_it = db.key_ids.find("pos");
    auto feature = [](const result_t & r, uint32_t key) -> const std::string * {
        for (const rfeat & f : r)
            if (f.key == key) return f.v.kind == value::str_t ? &f.v.s : nullptr;
        return nullptr;
    };
    std::unordered_map<std::string, uint32_t> index;
    std::vector<std::string> lp_keys, poses;
    std::vector<std::string> tag_keys;
    t.cand_of.clear();
    auto add = [&](std::string tag, std::string pos, const std::string & lex) {
        std::string key = tag + '\n' + pos + '\t' + lex;
        auto it = index.find(key);
        if (it != index.end()) return it->second;
        const uint32_t id = uint32_t(tag_keys.size());
        index.emplace(std::move(key), id);
        lp_keys.push_back(pos + '\t' + lex);
        poses.push_back(std::move(pos));
        tag_keys.push_back(std::move(tag));
        return id;
    };
    if (analyses.empty()) {
        add("<oov>", "<oov>", "<oov>");
    } else {
        for (const result_t & r : analyses) {
            std::string tag;
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i) tag += '|';
                const std::string * v = keys[i] == 0xFFFFFFFFu ? nullptr : feature(r, keys[i]);
                if (m.spec[i].flag)
                    tag += (!v || v->empty() || *v == "0" || *v == "na") ? '0' : '1';
                else
                    tag += v ? *v : absent;
            }
            const std::string * lex = lex_it == db.key_ids.end() ? nullptr : feature(r, lex_it->second);
            const std::string * pos = pos_it == db.key_ids.end() ? nullptr : feature(r, pos_it->second);
            t.cand_of.push_back(add(std::move(tag), pos ? *pos : std::string(), lex ? *lex : std::string()));
        }
    }
    const size_t n = tag_keys.size();
    t.tag.assign(n, -1);
    t.emit.assign(n, 0.0);
    for (size_t c = 0; c < n; ++c) {
        auto tg = m.tag_ids.find(tag_keys[c]);
        if (tg != m.tag_ids.end()) t.tag[c] = tg->second;
        double lp = k_dis_unknown_logp;
        auto lx = m.lexp.find(lp_keys[c]);
        if (lx != m.lexp.end()) {
            lp = lx->second;
        } else {
            auto un = m.unseen.find(poses[c]);
            if (un != m.unseen.end()) lp = un->second;
        }
        t.emit[c] = std::exp(lp);
    }
}

double dis_trans(const disambig_model & m, int64_t prev, int64_t cur) {
    if (prev < 0 || cur < 0) return k_dis_unknown_trans;
    auto it = m.trans.find((uint64_t(prev) << 32) | uint64_t(cur));
    return it != m.trans.end() ? it->second : m.backoff[size_t(prev)] * m.uni[size_t(cur)];
}

// Scaled forward-backward; fills every token's post (sums to 1).
void forward_backward(const disambig_model & m, std::vector<dis_token> & toks) {
    const size_t len = toks.size();
    if (!len) return;
    std::vector<double> scale(len, 1.0);
    for (size_t i = 0; i < len; ++i) {
        dis_token & t = toks[i];
        const size_t nc = t.tag.size();
        t.alpha.assign(nc, 0.0);
        double sum = 0;
        for (size_t c = 0; c < nc; ++c) {
            double in = 0;
            if (i == 0) {
                in = dis_trans(m, 0, t.tag[c]);
            } else {
                const dis_token & p = toks[i - 1];
                for (size_t q = 0; q < p.tag.size(); ++q) in += p.alpha[q] * dis_trans(m, p.tag[q], t.tag[c]);
            }
            t.alpha[c] = in * t.emit[c];
            sum += t.alpha[c];
        }
        if (!(sum > 0) || !std::isfinite(sum)) {
            // Underflow: restart from uniform weights at this token.
            for (size_t c = 0; c < nc; ++c) t.alpha[c] = 1.0 / double(nc);
            sum = 1.0;
        } else {
            for (double & a : t.alpha) a /= sum;
        }
        scale[i] = sum;
    }
    dis_token & last = toks[len - 1];
    double end = 0;
    for (size_t c = 0; c < last.tag.size(); ++c) end += last.alpha[c] * dis_trans(m, last.tag[c], 0);
    if (!(end > 0) || !std::isfinite(end)) end = 1.0;
    last.beta.assign(last.tag.size(), 0.0);
    for (size_t c = 0; c < last.tag.size(); ++c) last.beta[c] = dis_trans(m, last.tag[c], 0) / end;
    for (size_t i = len - 1; i > 0; --i) {
        const dis_token & t = toks[i];
        dis_token & p = toks[i - 1];
        p.beta.assign(p.tag.size(), 0.0);
        for (size_t q = 0; q < p.tag.size(); ++q) {
            double b = 0;
            for (size_t c = 0; c < t.tag.size(); ++c) b += dis_trans(m, p.tag[q], t.tag[c]) * t.emit[c] * t.beta[c];
            p.beta[q] = b / scale[i];
        }
        double mx = 0;
        for (double b : p.beta) mx = std::max(mx, b);
        if (!(mx > 0) || !std::isfinite(mx)) std::fill(p.beta.begin(), p.beta.end(), 1.0);
    }
    for (dis_token & t : toks) {
        const size_t nc = t.tag.size();
        t.post.assign(nc, 0.0);
        double z = 0;
        for (size_t c = 0; c < nc; ++c) z += t.alpha[c] * t.beta[c];
        for (size_t c = 0; c < nc; ++c)
            t.post[c] = z > 0 && std::isfinite(z) ? t.alpha[c] * t.beta[c] / z : 1.0 / double(nc);
    }
}

void put_f32(std::string & out, float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    put_u32(out, u);
}

} // namespace

struct crispembed_ar_morph {
    std::unique_ptr<analyzer_base> an;
    std::string info;
};

extern "C" {

uint32_t crispembed_ar_morph_abi_version(void) {
    return CRISPEMBED_AR_MORPH_ABI_VERSION;
}

static crispembed_ar_morph * finish_open(opened o) {
    auto * m = new crispembed_ar_morph;
    m->an = std::move(o.an);
    m->info = std::move(o.info);
    return m;
}

crispembed_ar_morph * crispembed_ar_morph_open_db(const char * db_path, const char * db_sha256, int backoff,
                                                  char * error, size_t error_capacity) {
    put_error(error, error_capacity, "");
    // Kill switch: CRISPEMBED_AR_MORPH=0 disables the Camel Morph analyzer service.
    if (core_env::explicitly_off("CRISPEMBED_AR_MORPH")) {
        put_error(error, error_capacity, "disabled by CRISPEMBED_AR_MORPH");
        return nullptr;
    }
    if (!db_path || !is_sha256_hex(db_sha256) || backoff < 0 || backoff > 4) {
        put_error(error, error_capacity, "invalid argument (path, lowercase SHA-256 or backoff)");
        return nullptr;
    }
    try {
        if (has_bin_magic(db_path)) {
            file_bytes bytes = map_file(db_path, k_max_db_bytes);
            const std::string digest = core_sha256::bytes(bytes.data, bytes.size);
            if (digest != db_sha256) throw data_error("database SHA-256 mismatch");
            return finish_open(build_bin(std::move(bytes), digest, backoff, "db", ""));
        }
        std::string text = read_bounded(db_path, k_max_db_bytes);
        const std::string digest = core_sha256::bytes(text.data(), text.size());
        if (digest != db_sha256) throw data_error("database SHA-256 mismatch");
        return finish_open(build(std::move(text), digest, backoff, "db", ""));
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
    }
    return nullptr;
}

crispembed_ar_morph * crispembed_ar_morph_open(const char * manifest_path, const char * manifest_sha256, int backoff,
                                               char * error, size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (core_env::explicitly_off("CRISPEMBED_AR_MORPH")) {
        put_error(error, error_capacity, "disabled by CRISPEMBED_AR_MORPH");
        return nullptr;
    }
    if (!manifest_path || !is_sha256_hex(manifest_sha256) || backoff < 0 || backoff > 4) {
        put_error(error, error_capacity, "invalid argument (path, lowercase SHA-256 or backoff)");
        return nullptr;
    }
    try {
        const std::string mtext = read_bounded(manifest_path, k_max_manifest_bytes);
        if (core_sha256::bytes(mtext.data(), mtext.size()) != manifest_sha256)
            throw data_error("manifest SHA-256 mismatch");
        const core_json_value::value m = core_json_value::parse(mtext);
        auto get_str = [&](const char * key) {
            const core_json_value::value * v = m.get(key);
            if (!v || v->kind != core_json_value::value::string_t) throw data_error(std::string("manifest: ") + key);
            return v->s;
        };
        if (get_str("format") != "crispembed-camel-morph-bundle") throw data_error("manifest: unknown format");
        const core_json_value::value * fv = m.get("format_version");
        if (!fv || fv->kind != core_json_value::value::int_t || (fv->i != 1 && fv->i != 2))
            throw data_error("manifest: unsupported format_version");
        // Version 1 bundles hold the ALMOR text database, version 2 the binary one.
        const bool binary = fv->i == 2;
        if (binary && get_str("db_format") != "crispembed-camel-morph-bin-v1")
            throw data_error("manifest: unsupported db_format");
        const std::string db_name = get_str("db");
        const std::string variant = get_str("variant");
        const core_json_value::value * files = m.get("files");
        if (!files || files->kind != core_json_value::value::array_t || files->a.size() > 16)
            throw data_error("manifest: files");
        const std::string dir = dirname_of(manifest_path);
        std::string db_text, db_digest;
        file_bytes db_bytes;
        bool have_db = false;
        for (const auto & f : files->a) {
            const core_json_value::value * p = f.get("path");
            const core_json_value::value * h = f.get("sha256");
            const core_json_value::value * s = f.get("size");
            if (!p || p->kind != core_json_value::value::string_t || !plain_file_name(p->s) || !h ||
                h->kind != core_json_value::value::string_t || !is_sha256_hex(h->s.c_str()) || !s ||
                s->kind != core_json_value::value::int_t || s->i < 0)
                throw data_error("manifest: file entry");
            const bool is_db = p->s == db_name;
            const size_t bound = is_db ? k_max_db_bytes : k_max_aux_file_bytes;
            if (size_t(s->i) > bound) throw data_error("manifest: file too large");
            if (is_db && binary) {
                db_bytes = map_file(dir + p->s, size_t(s->i));
                if (db_bytes.size != size_t(s->i)) throw data_error("size mismatch: " + p->s);
                const std::string digest = core_sha256::bytes(db_bytes.data, db_bytes.size);
                if (digest != h->s) throw data_error("SHA-256 mismatch: " + p->s);
                db_digest = digest;
                have_db = true;
                continue;
            }
            std::string bytes = read_bounded(dir + p->s, size_t(s->i));
            if (bytes.size() != size_t(s->i)) throw data_error("size mismatch: " + p->s);
            const std::string digest = core_sha256::bytes(bytes.data(), bytes.size());
            if (digest != h->s) throw data_error("SHA-256 mismatch: " + p->s);
            if (is_db) {
                db_text = std::move(bytes);
                db_digest = digest;
                have_db = true;
            }
        }
        if (!have_db) throw data_error("manifest: database not listed");
        if (binary) return finish_open(build_bin(std::move(db_bytes), db_digest, backoff, variant, manifest_sha256));
        return finish_open(build(std::move(db_text), db_digest, backoff, variant, manifest_sha256));
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
    }
    return nullptr;
}

int crispembed_ar_morph_analyze(const crispembed_ar_morph * morph, const char * word, size_t word_len, char ** json,
                                size_t * json_len, char * error, size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (json) *json = nullptr;
    if (json_len) *json_len = 0;
    if (!morph || !json || (!word && word_len)) {
        put_error(error, error_capacity, "invalid argument");
        return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
    }
    if (core_env::explicitly_off("CRISPEMBED_AR_MORPH")) {
        put_error(error, error_capacity, "disabled by CRISPEMBED_AR_MORPH");
        return CRISPEMBED_AR_MORPH_DISABLED;
    }
    try {
        const std::string_view in(word ? word : "", word_len);
        if (const std::string * why = check_word(in)) {
            put_error(error, error_capacity, *why);
            return CRISPEMBED_AR_MORPH_INPUT_ERROR;
        }
        const std::vector<result_t> analyses = morph->an->analyze(from_utf8_valid(in));
        std::string out;
        append_json(out, morph->an->common(), analyses);
        char * buf = static_cast<char *>(std::malloc(out.size() + 1));
        if (!buf) {
            put_error(error, error_capacity, "allocation failed");
            return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
        }
        std::memcpy(buf, out.data(), out.size());
        buf[out.size()] = '\0';
        *json = buf;
        if (json_len) *json_len = out.size();
        return CRISPEMBED_AR_MORPH_OK;
    } catch (const input_error & e) {
        put_error(error, error_capacity, e.what());
        return CRISPEMBED_AR_MORPH_INPUT_ERROR;
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
        return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
        return CRISPEMBED_AR_MORPH_RUNTIME_ERROR;
    }
}

int crispembed_ar_morph_analyze_compact(const crispembed_ar_morph * morph, const char * words,
                                        const uint32_t * word_lens, uint32_t n_words, uint32_t flags, uint8_t ** out,
                                        size_t * out_len, char * error, size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (out) *out = nullptr;
    if (out_len) *out_len = 0;
    if (!morph || !out || (n_words && !word_lens) || (flags & ~uint32_t(CRISPEMBED_AR_MORPH_COMPACT_DEDUP))) {
        put_error(error, error_capacity, "invalid argument");
        return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_words; ++i) total += word_lens[i];
    if (total && !words) {
        put_error(error, error_capacity, "invalid argument");
        return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
    }
    if (core_env::explicitly_off("CRISPEMBED_AR_MORPH")) {
        put_error(error, error_capacity, "disabled by CRISPEMBED_AR_MORPH");
        return CRISPEMBED_AR_MORPH_DISABLED;
    }
    try {
        std::string buf;
        put_u32(buf, CRISPEMBED_AR_MORPH_COMPACT_MAGIC);
        put_u32(buf, n_words);
        size_t pos = 0;
        std::string record;
        for (uint32_t i = 0; i < n_words; ++i) {
            const std::string_view in(words ? words + pos : "", word_lens[i]);
            pos += word_lens[i];
            record.clear();
            uint32_t status = CRISPEMBED_AR_MORPH_OK;
            if (check_word(in)) {
                status = CRISPEMBED_AR_MORPH_INPUT_ERROR;
            } else {
                try {
                    const std::vector<result_t> analyses = morph->an->analyze(from_utf8_valid(in));
                    append_compact(record, morph->an->common(), analyses,
                                   (flags & CRISPEMBED_AR_MORPH_COMPACT_DEDUP) != 0);
                } catch (const input_error &) {
                    status = CRISPEMBED_AR_MORPH_INPUT_ERROR;
                    record.clear();
                }
            }
            put_u32(buf, status);
            put_u32(buf, uint32_t(record.size()));
            buf += record;
        }
        uint8_t * mem = static_cast<uint8_t *>(std::malloc(buf.size()));
        if (!mem) {
            put_error(error, error_capacity, "allocation failed");
            return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
        }
        std::memcpy(mem, buf.data(), buf.size());
        *out = mem;
        if (out_len) *out_len = buf.size();
        return CRISPEMBED_AR_MORPH_OK;
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
        return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
        return CRISPEMBED_AR_MORPH_RUNTIME_ERROR;
    }
}

void crispembed_ar_morph_string_free(char * text) {
    std::free(text);
}

void crispembed_ar_morph_buffer_free(uint8_t * buffer) {
    std::free(buffer);
}

const char * crispembed_ar_morph_info(const crispembed_ar_morph * morph) {
    return morph ? morph->info.c_str() : nullptr;
}

void crispembed_ar_morph_free(crispembed_ar_morph * morph) {
    delete morph;
}

} // extern "C"

struct crispembed_ar_morph_disambig {
    disambig_model model;
};

namespace {

bool disambig_gate(char * error, size_t error_capacity) {
    if (core_env::explicitly_off("CRISPEMBED_AR_MORPH")) {
        put_error(error, error_capacity, "disabled by CRISPEMBED_AR_MORPH");
        return false;
    }
    // Default-off gate of the context disambiguator: CRISPEMBED_AR_MORPH_DISAMBIG=1 enables it.
    if (!core_env::on("CRISPEMBED_AR_MORPH_DISAMBIG")) {
        put_error(error, error_capacity, "disabled: set CRISPEMBED_AR_MORPH_DISAMBIG=1 to enable");
        return false;
    }
    return true;
}

crispembed_ar_morph_disambig * finish_disambig(const uint8_t * p, size_t n, const std::string & sha256,
                                               const std::string & manifest_sha256) {
    auto d = std::make_unique<crispembed_ar_morph_disambig>();
    load_disambig(d->model, p, n, sha256, manifest_sha256);
    return d.release();
}

} // namespace

extern "C" {

crispembed_ar_morph_disambig * crispembed_ar_morph_disambig_open_file(const char * model_path,
                                                                      const char * model_sha256, char * error,
                                                                      size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (!disambig_gate(error, error_capacity)) return nullptr;
    if (!model_path || !is_sha256_hex(model_sha256)) {
        put_error(error, error_capacity, "invalid argument (path or lowercase SHA-256)");
        return nullptr;
    }
    try {
        const std::string bytes = read_bounded(model_path, k_max_model_bytes);
        const std::string digest = core_sha256::bytes(bytes.data(), bytes.size());
        if (digest != model_sha256) throw data_error("model SHA-256 mismatch");
        return finish_disambig(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), digest, "");
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
    }
    return nullptr;
}

crispembed_ar_morph_disambig * crispembed_ar_morph_disambig_open(const char * manifest_path,
                                                                 const char * manifest_sha256, char * error,
                                                                 size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (!disambig_gate(error, error_capacity)) return nullptr;
    if (!manifest_path || !is_sha256_hex(manifest_sha256)) {
        put_error(error, error_capacity, "invalid argument (path or lowercase SHA-256)");
        return nullptr;
    }
    try {
        const std::string mtext = read_bounded(manifest_path, k_max_manifest_bytes);
        if (core_sha256::bytes(mtext.data(), mtext.size()) != manifest_sha256)
            throw data_error("manifest SHA-256 mismatch");
        const core_json_value::value m = core_json_value::parse(mtext);
        auto get_str = [&](const char * key) {
            const core_json_value::value * v = m.get(key);
            if (!v || v->kind != core_json_value::value::string_t) throw data_error(std::string("manifest: ") + key);
            return v->s;
        };
        if (get_str("format") != "crispembed-camel-morph-disambig-bundle") throw data_error("manifest: unknown format");
        const core_json_value::value * fv = m.get("format_version");
        if (!fv || fv->kind != core_json_value::value::int_t || fv->i != 1)
            throw data_error("manifest: unsupported format_version");
        if (get_str("model_format") != "crispembed-camel-morph-disambig-v1")
            throw data_error("manifest: unsupported model_format");
        const std::string model_name = get_str("model");
        const core_json_value::value * files = m.get("files");
        if (!files || files->kind != core_json_value::value::array_t || files->a.size() > 16)
            throw data_error("manifest: files");
        const std::string dir = dirname_of(manifest_path);
        std::string model_bytes, model_digest;
        bool have_model = false;
        for (const auto & f : files->a) {
            const core_json_value::value * p = f.get("path");
            const core_json_value::value * h = f.get("sha256");
            const core_json_value::value * s = f.get("size");
            if (!p || p->kind != core_json_value::value::string_t || !plain_file_name(p->s) || !h ||
                h->kind != core_json_value::value::string_t || !is_sha256_hex(h->s.c_str()) || !s ||
                s->kind != core_json_value::value::int_t || s->i < 0)
                throw data_error("manifest: file entry");
            const bool is_model = p->s == model_name;
            if (size_t(s->i) > (is_model ? k_max_model_bytes : k_max_aux_file_bytes))
                throw data_error("manifest: file too large");
            std::string bytes = read_bounded(dir + p->s, size_t(s->i));
            if (bytes.size() != size_t(s->i)) throw data_error("size mismatch: " + p->s);
            const std::string digest = core_sha256::bytes(bytes.data(), bytes.size());
            if (digest != h->s) throw data_error("SHA-256 mismatch: " + p->s);
            if (is_model) {
                model_bytes = std::move(bytes);
                model_digest = digest;
                have_model = true;
            }
        }
        if (!have_model) throw data_error("manifest: model not listed");
        return finish_disambig(reinterpret_cast<const uint8_t *>(model_bytes.data()), model_bytes.size(), model_digest,
                               manifest_sha256);
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
    }
    return nullptr;
}

int crispembed_ar_morph_disambiguate(const crispembed_ar_morph * morph, const crispembed_ar_morph_disambig * model,
                                     const char * words, const uint32_t * word_lens, uint32_t n_words,
                                     const uint32_t * sentence_lens, uint32_t n_sentences, uint32_t flags,
                                     uint8_t ** out, size_t * out_len, char * error, size_t error_capacity) {
    put_error(error, error_capacity, "");
    if (out) *out = nullptr;
    if (out_len) *out_len = 0;
    if (!morph || !model || !out || (n_words && !word_lens) || (n_sentences && !sentence_lens) ||
        (flags & ~uint32_t(CRISPEMBED_AR_MORPH_COMPACT_DEDUP))) {
        put_error(error, error_capacity, "invalid argument");
        return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
    }
    uint64_t total = 0, covered = 0;
    for (uint32_t i = 0; i < n_words; ++i) total += word_lens[i];
    for (uint32_t i = 0; i < n_sentences; ++i) {
        if (sentence_lens[i] > k_max_sentence_words) {
            put_error(error, error_capacity, "sentence exceeds 65536 words");
            return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
        }
        covered += sentence_lens[i];
    }
    if ((total && !words) || covered != n_words) {
        put_error(error, error_capacity, "invalid argument (sentence lengths must sum to n_words)");
        return CRISPEMBED_AR_MORPH_INVALID_ARGUMENT;
    }
    if (!disambig_gate(error, error_capacity)) return CRISPEMBED_AR_MORPH_DISABLED;
    try {
        const db_common & db = morph->an->common();
        const bool dedup = (flags & CRISPEMBED_AR_MORPH_COMPACT_DEDUP) != 0;
        // Per distinct word: status, compact record, kept analyses, candidates.
        struct word_result {
            uint32_t status = CRISPEMBED_AR_MORPH_OK;
            std::string record;
            std::vector<result_t> analyses;
            std::vector<uint32_t> kept;
        };
        std::unordered_map<std::string_view, size_t> seen;
        std::vector<word_result> results;
        std::vector<size_t> word_of(n_words);
        size_t at = 0;
        for (uint32_t i = 0; i < n_words; ++i) {
            const std::string_view in(words ? words + at : "", word_lens[i]);
            at += word_lens[i];
            auto it = seen.find(in);
            if (it != seen.end()) {
                word_of[i] = it->second;
                continue;
            }
            word_result r;
            if (check_word(in)) {
                r.status = CRISPEMBED_AR_MORPH_INPUT_ERROR;
            } else {
                try {
                    std::vector<result_t> all = morph->an->analyze(from_utf8_valid(in));
                    append_compact(r.record, db, all, dedup, &r.kept);
                    for (uint32_t k : r.kept) r.analyses.push_back(std::move(all[k]));
                } catch (const input_error &) {
                    r.status = CRISPEMBED_AR_MORPH_INPUT_ERROR;
                    r.record.clear();
                    r.analyses.clear();
                }
            }
            seen.emplace(in, results.size());
            word_of[i] = results.size();
            results.push_back(std::move(r));
        }
        std::string buf;
        put_u32(buf, CRISPEMBED_AR_MORPH_COMPACT_MAGIC);
        put_u32(buf, n_words);
        for (uint32_t i = 0; i < n_words; ++i) {
            const word_result & r = results[word_of[i]];
            put_u32(buf, r.status);
            put_u32(buf, uint32_t(r.record.size()));
            buf += r.record;
        }
        put_u32(buf, k_scores_magic);
        put_u32(buf, n_words);
        std::vector<dis_token> toks;
        uint32_t first = 0;
        for (uint32_t s = 0; s < n_sentences; ++s) {
            const uint32_t len = sentence_lens[s];
            toks.assign(len, dis_token());
            for (uint32_t i = 0; i < len; ++i)
                build_candidates(model->model, db, results[word_of[first + i]].analyses, toks[i]);
            forward_backward(model->model, toks);
            for (uint32_t i = 0; i < len; ++i) {
                const dis_token & t = toks[i];
                std::vector<uint32_t> order(t.post.size()), rank(t.post.size());
                for (uint32_t c = 0; c < order.size(); ++c) order[c] = c;
                std::stable_sort(order.begin(), order.end(),
                                 [&](uint32_t a, uint32_t b) { return t.post[a] > t.post[b]; });
                for (uint32_t r = 0; r < order.size(); ++r) rank[order[r]] = r;
                put_u32(buf, uint32_t(t.cand_of.size()));
                put_u32(buf, uint32_t(t.post.size()));
                for (uint32_t c : t.cand_of) {
                    put_u32(buf, rank[c]);
                    put_f32(buf, float(t.post[c]));
                }
            }
            first += len;
        }
        uint8_t * mem = static_cast<uint8_t *>(std::malloc(buf.size()));
        if (!mem) {
            put_error(error, error_capacity, "allocation failed");
            return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
        }
        std::memcpy(mem, buf.data(), buf.size());
        *out = mem;
        if (out_len) *out_len = buf.size();
        return CRISPEMBED_AR_MORPH_OK;
    } catch (const std::bad_alloc &) {
        put_error(error, error_capacity, "allocation failed");
        return CRISPEMBED_AR_MORPH_ALLOCATION_ERROR;
    } catch (const std::exception & e) {
        put_error(error, error_capacity, e.what());
        return CRISPEMBED_AR_MORPH_RUNTIME_ERROR;
    }
}

const char * crispembed_ar_morph_disambig_info(const crispembed_ar_morph_disambig * model) {
    return model ? model->model.info.c_str() : nullptr;
}

void crispembed_ar_morph_disambig_free(crispembed_ar_morph_disambig * model) {
    delete model;
}

} // extern "C"
