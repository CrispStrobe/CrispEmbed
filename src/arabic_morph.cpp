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

struct morph_db {
    // Interned strings.
    std::string pool;
    std::vector<std::pair<uint32_t, uint32_t>> strs;
    std::vector<std::string> key_names;
    std::unordered_map<std::string, uint32_t> key_ids;
    std::vector<std::string> cat_names;
    std::unordered_map<std::string, uint32_t> cat_ids;

    std::vector<feat> feats;
    std::vector<entry> entries;
    std::unordered_map<std::string, std::vector<uint32_t>> prefix_hash, suffix_hash, stem_hash;

    std::unordered_set<uint64_t> ab, bc, ac;
    std::vector<uint8_t> ab_has, bc_has, ac_has;

    std::vector<uint8_t> defined; // by key id
    bool pattern_in_order = false;
    std::unordered_map<std::string, result_t> defaults;
    std::unordered_map<std::string, std::vector<std::string>> stem_backoffs;
    size_t max_prefix_size = 0, max_suffix_size = 0;

    std::string_view str(uint32_t id) const { return std::string_view(pool.data() + strs[id].first, strs[id].second); }

    bool def(uint32_t key) const { return key < defined.size() && defined[key]; }

    uint32_t key_id(std::string_view name) {
        auto it = key_ids.find(std::string(name));
        if (it != key_ids.end()) return it->second;
        const uint32_t id = uint32_t(key_names.size());
        key_names.emplace_back(name);
        key_ids.emplace(std::string(name), id);
        return id;
    }

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

    static uint64_t pair_key(uint32_t a, uint32_t b) { return (uint64_t(a) << 32) | b; }
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
            table.insert(morph_db::pair_key(a, b));
            if (has.size() <= a) has.resize(a + 1, 0);
            has[a] = 1;
        }
    }
};

// ---------------------------------------------------------------------------
// Analyzer.

enum class backoff_t { none, noan_all, noan_prop, add_all, add_prop };

class analyzer {
public:
    analyzer(std::unique_ptr<morph_db> db, backoff_t backoff) : db_(std::move(db)), backoff_(backoff) {}

    const morph_db & db() const { return *db_; }

    std::vector<result_t> analyze(const u32 & raw) const {
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

        std::string wd_no_tatweel;
        for (char32_t c : word_dediac)
            if (c != 0x0640) py_str::append_utf8(wd_no_tatweel, c);

        const long long w = (long long)word_normal.size();
        const long long max_p = (long long)db_->max_prefix_size, max_s = (long long)db_->max_suffix_size;
        if (!has_punc) {
            for (long long p = 0; p <= std::min(max_p, w - 1); ++p) {
                for (long long s = std::max(1LL, w - p - max_s); s <= w - p; ++s) {
                    auto pre = db_->prefix_hash.find(sub(0, size_t(p)));
                    auto suf = db_->suffix_hash.find(sub(size_t(p + s), size_t(w)));
                    if (pre == db_->prefix_hash.end() || suf == db_->suffix_hash.end()) continue;
                    auto stem = db_->stem_hash.find(sub(size_t(p), size_t(p + s)));
                    if (stem == db_->stem_hash.end()) continue;
                    combined(wd_no_tatweel, pre->second, stem->second, suf->second, analyses);
                }
            }
        }

        const bool noan = backoff_ == backoff_t::noan_all || backoff_ == backoff_t::noan_prop;
        const bool add = backoff_ == backoff_t::add_all || backoff_ == backoff_t::add_prop;
        if ((noan && analyses.empty()) || add) {
            const bool prop = backoff_ == backoff_t::noan_prop || backoff_ == backoff_t::add_prop;
            auto bo = db_->stem_backoffs.find(prop ? "PROP" : "ALL");
            if (bo == db_->stem_backoffs.end()) throw input_error("camel_tools raises KeyError: stem backoff mode");
            auto noan_stems = db_->stem_hash.find("NOAN");
            if (noan_stems == db_->stem_hash.end()) throw input_error("camel_tools raises KeyError: 'NOAN'");
            std::vector<uint32_t> stems;
            for (uint32_t e : noan_stems->second) {
                const std::string & cat = db_->cat_names[db_->entries[e].cat];
                if (std::find(bo->second.begin(), bo->second.end(), cat) != bo->second.end()) stems.push_back(e);
            }
            for (long long p = 0; p <= std::min(max_p, w - 1); ++p) {
                for (long long s = std::max(1LL, w - p - max_s); s <= w - p; ++s) {
                    auto pre = db_->prefix_hash.find(sub(0, size_t(p)));
                    auto suf = db_->suffix_hash.find(sub(size_t(p + s), size_t(w)));
                    if (pre == db_->prefix_hash.end() || suf == db_->suffix_hash.end()) continue;
                    const u32 stem = word_normal.substr(size_t(p), size_t(s));
                    combined_backoff(stem, prop, pre->second, stems, suf->second, analyses);
                }
            }
        }
        return analyses;
    }

private:
    std::unique_ptr<morph_db> db_;
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

    // Digit, punctuation and foreign words.
    result_t special(const u32 & word_u, const char * dkey, const char * bw_suffix, const char * source,
                     const char * undefined_lex, const char * catib6, const char * ud, bool foreign) const {
        auto it = db_->defaults.find(dkey);
        if (it == db_->defaults.end()) throw input_error(std::string("camel_tools raises KeyError: '") + dkey + "'");
        result_t r = it->second;
        const std::string word = to_utf8(word_u);
        if (foreign) set(r, K_POS, str_value("foreign"));
        set(r, K_DIAC, str_value(word));
        set(r, K_STEM, str_value(word));
        set(r, K_STEMGLOSS, str_value(word));
        set(r, K_STEMCAT, none_value());
        set(r, K_LEX, str_value(word));
        set(r, K_BW, str_value(word + bw_suffix));
        set(r, K_SOURCE, str_value(source));
        for (uint32_t k : k_copy_feats)
            if (db_->def(k)) set(r, k, str_value(word));
        for (uint32_t k : k_undefined_lex_feats)
            if (db_->def(k)) set(r, k, str_value(undefined_lex));
        if (db_->def(K_CATIB6)) set(r, K_CATIB6, str_value(catib6));
        if (db_->def(K_UD)) set(r, K_UD, str_value(ud));
        set(r, K_POS_LOGPROB, float_value(-99.0));
        set(r, K_LEX_LOGPROB, float_value(-99.0));
        set(r, K_POS_LEX_LOGPROB, float_value(-99.0));
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
        set(r, K_STEM, str_value(std::string(need(S, K_DIAC))));
        set(r, K_STEMGLOSS, str_value(get_or(S, K_GLOSS, "")));
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

    void combined(const std::string & wd_no_tatweel, const std::vector<uint32_t> & prefixes,
                  const std::vector<uint32_t> & stems, const std::vector<uint32_t> & suffixes,
                  std::vector<result_t> & out) const {
        for (uint32_t pe : prefixes) {
            const uint32_t pcat = db_->entries[pe].cat;
            fview P;
            bool have_p = false;
            for (uint32_t se : stems) {
                const uint32_t scat = db_->entries[se].cat;
                if (!db_->ab_has[pcat]) throw input_error("camel_tools raises KeyError: prefix category");
                if (!db_->ab.count(morph_db::pair_key(pcat, scat))) continue;
                fview S;
                bool have_s = false;
                for (uint32_t xe : suffixes) {
                    const uint32_t xcat = db_->entries[xe].cat;
                    if (!db_->bc_has[scat] || !db_->ac_has[pcat] || !db_->bc.count(morph_db::pair_key(scat, xcat)) ||
                        !db_->ac.count(morph_db::pair_key(pcat, xcat)))
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
                    set(merged, K_STEM, str_value(std::string(need(S, K_DIAC))));
                    set(merged, K_STEMCAT, str_value(db_->cat_names[scat]));
                    const std::string merged_dediac = dediac_ar_utf8(utf8_of(need(merged, K_DIAC)));
                    if (wd_no_tatweel != merged_dediac) set(merged, K_SOURCE, str_value("spvar"));
                    out.push_back(std::move(merged));
                }
            }
        }
    }

    void combined_backoff(const u32 & stem, bool prop, const std::vector<uint32_t> & prefixes,
                          const std::vector<uint32_t> & stems, const std::vector<uint32_t> & suffixes,
                          std::vector<result_t> & out) const {
        std::string expanded;
        bool expanded_ready = false;
        for (uint32_t pe : prefixes) {
            const uint32_t pcat = db_->entries[pe].cat;
            const fview P = db_->view(pe);
            for (uint32_t se : stems) {
                const uint32_t scat = db_->entries[se].cat;
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
                if (!db_->ab_has[pcat]) throw input_error("camel_tools raises KeyError: prefix category");
                if (!db_->ab.count(morph_db::pair_key(pcat, scat))) continue;
                for (uint32_t xe : suffixes) {
                    const uint32_t xcat = db_->entries[xe].cat;
                    if (!db_->bc_has[scat]) throw input_error("camel_tools raises KeyError: stem category");
                    if (!db_->bc.count(morph_db::pair_key(scat, xcat)) || !db_->ac_has[pcat] ||
                        !db_->ac.count(morph_db::pair_key(pcat, xcat)))
                        continue;
                    if (prop && sneed(K_BW).find("NOUN_PROP") == std::string::npos) continue;
                    if (!expanded_ready) {
                        expanded = expand_noan_template(stem);
                        expanded_ready = true;
                    }
                    for (uint32_t k : { uint32_t(K_BW), uint32_t(K_DIAC), uint32_t(K_LEX) }) {
                        std::string & v = sneed(k);
                        v = replace_all(v, "NOAN", expanded);
                    }
                    if (std::string * c = sget(K_CAPHI))
                        *c = simple_ar_to_caphi(stem);
                    else
                        sf.emplace_back(K_CAPHI, simple_ar_to_caphi(stem));
                    fview S;
                    S.reserve(sf.size());
                    for (const auto & kv : sf) S.emplace_back(kv.first, std::string_view(kv.second));
                    const fview X = db_->view(xe);
                    result_t merged = merge(P, S, X);
                    set(merged, K_STEM, str_value(sneed(K_DIAC)));
                    set(merged, K_STEMCAT, str_value(db_->cat_names[scat]));
                    set(merged, K_SOURCE, str_value("backoff"));
                    set(merged, K_PATTERN, str_value("backoff"));
                    set(merged, K_GLOSS, str_value(sneed(K_GLOSS)));
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

struct opened {
    std::unique_ptr<analyzer> an;
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

opened build(std::string db_text, const std::string & db_sha256, int backoff, const std::string & variant,
             const std::string & manifest_sha256) {
    const backoff_t bo = to_backoff(backoff);
    auto db = std::make_unique<morph_db>();
    db_parser(std::move(db_text), *db).parse();
    size_t stems = 0;
    for (const auto & kv : db->stem_hash) stems += kv.second.size();
    std::string info = "{\"abi_version\":" + std::to_string(CRISPEMBED_AR_MORPH_ABI_VERSION) + ",\"variant\":\"" +
                       core_json::json_escape_strict(variant) + "\",\"db_sha256\":\"" + db_sha256 + "\"";
    if (!manifest_sha256.empty()) info += ",\"manifest_sha256\":\"" + manifest_sha256 + "\"";
    info += ",\"backoff\":\"" + std::string(backoff_name(backoff)) +
            "\",\"prefix_forms\":" + std::to_string(db->prefix_hash.size()) +
            ",\"suffix_forms\":" + std::to_string(db->suffix_hash.size()) +
            ",\"stem_forms\":" + std::to_string(db->stem_hash.size()) + ",\"stem_entries\":" + std::to_string(stems) +
            ",\"unicode_version\":\"" + arabic_morph_tables::k_unicode_version + "\"}";
    opened o;
    o.an = std::make_unique<analyzer>(std::move(db), bo);
    o.info = std::move(info);
    return o;
}

void put_error(char * error, size_t cap, const std::string & msg) {
    if (!error || cap == 0) return;
    const size_t n = std::min(cap - 1, msg.size());
    std::memcpy(error, msg.data(), n);
    error[n] = '\0';
}

void append_json(std::string & out, const morph_db & db, const std::vector<result_t> & analyses) {
    out.push_back('[');
    for (size_t a = 0; a < analyses.size(); ++a) {
        if (a) out.push_back(',');
        out.push_back('{');
        bool first = true;
        for (const rfeat & f : analyses[a]) {
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

} // namespace

struct crispembed_ar_morph {
    std::unique_ptr<analyzer> an;
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
        if (!fv || fv->kind != core_json_value::value::int_t || fv->i != 1)
            throw data_error("manifest: unsupported format_version");
        const std::string db_name = get_str("db");
        const std::string variant = get_str("variant");
        const core_json_value::value * files = m.get("files");
        if (!files || files->kind != core_json_value::value::array_t || files->a.size() > 16)
            throw data_error("manifest: files");
        const std::string dir = dirname_of(manifest_path);
        std::string db_text, db_digest;
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
    if (word_len > CRISPEMBED_AR_MORPH_MAX_WORD_BYTES) {
        put_error(error, error_capacity, "word exceeds CRISPEMBED_AR_MORPH_MAX_WORD_BYTES");
        return CRISPEMBED_AR_MORPH_INPUT_ERROR;
    }
    try {
        const std::string_view in(word ? word : "", word_len);
        if (!valid_utf8(in)) {
            put_error(error, error_capacity, "word is not valid UTF-8");
            return CRISPEMBED_AR_MORPH_INPUT_ERROR;
        }
        const std::vector<result_t> analyses = morph->an->analyze(from_utf8_valid(in));
        std::string out;
        append_json(out, morph->an->db(), analyses);
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

void crispembed_ar_morph_string_free(char * text) {
    std::free(text);
}

const char * crispembed_ar_morph_info(const crispembed_ar_morph * morph) {
    return morph ? morph->info.c_str() : nullptr;
}

void crispembed_ar_morph_free(crispembed_ar_morph * morph) {
    delete morph;
}

} // extern "C"
