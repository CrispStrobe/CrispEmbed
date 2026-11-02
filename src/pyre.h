// pyre.h - CPython `str` and `re` semantics over Unicode code points.
//
// Native ports of Python text post-processing must agree with CPython on
// code-point indexing, Unicode whitespace/word/digit classes, str.lower(),
// str.splitlines() and `re`'s leftmost-first backtracking. This header gives a
// small, dependency-free backtracking VM (explicit stack, no recursion per
// character) for the Python regular-expression subset used by ported code:
// literals, classes, . \s \S \d \D \w \W \b \B \uXXXX, groups (capturing,
// (?:), (?P<name>)), alternation, greedy/lazy * + ? {m} {m,} {m,n},
// ^ $ \A \Z, lookahead and fixed-width lookbehind, backreferences, and the
// DOTALL / IGNORECASE flags. Unsupported syntax throws at compile time so a
// port can never silently change pattern meaning.
//
// Character predicates come from tools/gen/gen_pyunicode_tables.py, generated
// with the CPython version recorded in pyunicode_tables.h.
#pragma once

#include "pyunicode_tables.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pyre {

using ustring = std::u32string;

// ---------------------------------------------------------------- UTF-8 ----
// Invalid bytes round-trip via surrogateescape (U+DC80..U+DCFF), like
// Python's bytes.decode("utf-8", "surrogateescape").
inline ustring from_utf8(const std::string & s) {
    ustring out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) {
            out.push_back(b);
            ++i;
            continue;
        }
        int more = 0;
        uint32_t cp = 0, min = 0;
        if (b >= 0xc2 && b < 0xe0)
            more = 1, cp = b & 31, min = 0x80;
        else if (b >= 0xe0 && b < 0xf0)
            more = 2, cp = b & 15, min = 0x800;
        else if (b >= 0xf0 && b <= 0xf4)
            more = 3, cp = b & 7, min = 0x10000;
        bool ok = more > 0 && i + static_cast<size_t>(more) < s.size();
        for (int k = 1; ok && k <= more; ++k) {
            const unsigned c = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
            if ((c & 0xc0) != 0x80)
                ok = false;
            else
                cp = (cp << 6) | (c & 63);
        }
        if (ok && (cp < min || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))) ok = false;
        if (!ok) {
            out.push_back(0xdc00 + b);
            ++i;
            continue;
        }
        out.push_back(cp);
        i += static_cast<size_t>(more) + 1;
    }
    return out;
}

inline void append_utf8(std::string & out, char32_t cp) {
    if (cp >= 0xdc80 && cp <= 0xdcff) {
        out.push_back(static_cast<char>(cp - 0xdc00));
    } else if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 63)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 63)));
        out.push_back(static_cast<char>(0x80 | (cp & 63)));
    }
}

inline std::string to_utf8(const ustring & s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t c : s) append_utf8(out, c);
    return out;
}

inline ustring U(const char * utf8) {
    return from_utf8(utf8);
}

// ----------------------------------------------------------- predicates ----
inline bool in_ranges(const uint32_t * r, size_t n_pairs, uint32_t c) {
    size_t lo = 0, hi = n_pairs;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (c < r[mid * 2])
            hi = mid;
        else if (c > r[mid * 2 + 1])
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

inline bool isspace(char32_t c) {
    if (c < 128) return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x1f);
    for (uint32_t s : pyunicode_tables::space_points)
        if (s == c) return true;
    return false;
}

inline bool isalnum(char32_t c) {
    if (c < 128) return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    return in_ranges(pyunicode_tables::alnum_ranges, sizeof(pyunicode_tables::alnum_ranges) / 8, c);
}

inline bool isword(char32_t c) {
    return c == '_' || isalnum(c);
}

inline bool isdecimal(char32_t c) {
    if (c < 128) return c >= '0' && c <= '9';
    return in_ranges(pyunicode_tables::decimal_ranges, sizeof(pyunicode_tables::decimal_ranges) / 8, c);
}

inline char32_t simple_lower(char32_t c) {
    if (c < 128) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    const uint32_t * m = pyunicode_tables::lower_simple_map;
    size_t lo = 0, hi = sizeof(pyunicode_tables::lower_simple_map) / 8;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (m[mid * 2] == c) return m[mid * 2 + 1];
        if (m[mid * 2] < c)
            lo = mid + 1;
        else
            hi = mid;
    }
    return c;
}

// str.lower() (full mapping, e.g. U+0130 -> "i̇").
inline ustring lower(const ustring & s) {
    ustring out;
    out.reserve(s.size());
    const uint32_t * m = pyunicode_tables::lower_full_index;
    const size_t n = sizeof(pyunicode_tables::lower_full_index) / 12;
    for (size_t pos = 0; pos < s.size(); ++pos) {
        const char32_t c = s[pos];
        if (c < 128) {
            out.push_back((c >= 'A' && c <= 'Z') ? c + 32 : c);
            continue;
        }
        if (c == 0x3a3) { // CPython handle_capital_sigma (Final_Sigma)
            auto ignorable = [](char32_t x) {
                return in_ranges(pyunicode_tables::sigma_ignorable_ranges,
                                 sizeof(pyunicode_tables::sigma_ignorable_ranges) / 8, x);
            };
            auto cased = [](char32_t x) {
                return in_ranges(pyunicode_tables::sigma_cased_ranges, sizeof(pyunicode_tables::sigma_cased_ranges) / 8,
                                 x);
            };
            long j = static_cast<long>(pos) - 1;
            while (j >= 0 && ignorable(s[static_cast<size_t>(j)])) --j;
            bool final_sigma = j >= 0 && cased(s[static_cast<size_t>(j)]);
            if (final_sigma && pos + 1 < s.size()) {
                size_t k = pos + 1;
                while (k < s.size() && ignorable(s[k])) ++k;
                final_sigma = k == s.size() || !cased(s[k]);
            }
            out.push_back(final_sigma ? 0x3c2 : 0x3c3);
            continue;
        }
        size_t lo = 0, hi = n;
        bool found = false;
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (m[mid * 3] == c) {
                for (uint32_t k = 0; k < m[mid * 3 + 2]; ++k)
                    out.push_back(pyunicode_tables::lower_full_pool[m[mid * 3 + 1] + k]);
                found = true;
                break;
            }
            if (m[mid * 3] < c)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (!found) out.push_back(c);
    }
    return out;
}

// ------------------------------------------------------- str helpers -------
inline ustring strip(const ustring & s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace(s[a])) ++a;
    while (b > a && isspace(s[b - 1])) --b;
    return s.substr(a, b - a);
}
inline ustring lstrip(const ustring & s) {
    size_t a = 0;
    while (a < s.size() && isspace(s[a])) ++a;
    return s.substr(a);
}
inline ustring rstrip(const ustring & s) {
    size_t b = s.size();
    while (b > 0 && isspace(s[b - 1])) --b;
    return s.substr(0, b);
}
inline ustring strip_chars(const ustring & s, const ustring & chars) {
    size_t a = 0, b = s.size();
    while (a < b && chars.find(s[a]) != ustring::npos) ++a;
    while (b > a && chars.find(s[b - 1]) != ustring::npos) --b;
    return s.substr(a, b - a);
}
inline bool starts(const ustring & s, const ustring & p, size_t at = 0) {
    return at <= s.size() && s.size() - at >= p.size() && s.compare(at, p.size(), p) == 0;
}
inline bool ends(const ustring & s, const ustring & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
inline bool contains(const ustring & s, const ustring & p) {
    return s.find(p) != ustring::npos;
}
// str.count(sub) for nonempty sub (non-overlapping).
inline size_t count(const ustring & s, const ustring & sub) {
    if (sub.empty()) return s.size() + 1;
    size_t n = 0;
    for (size_t at = s.find(sub); at != ustring::npos; at = s.find(sub, at + sub.size())) ++n;
    return n;
}
inline ustring replace(const ustring & s, const ustring & from, const ustring & to) {
    if (from.empty()) throw std::invalid_argument("pyre::replace empty pattern unsupported");
    ustring out;
    size_t last = 0;
    for (size_t at = s.find(from); at != ustring::npos; at = s.find(from, last)) {
        out.append(s, last, at - last);
        out += to;
        last = at + from.size();
    }
    out.append(s, last, ustring::npos);
    return out;
}
inline bool is_line_break(char32_t c) {
    return c == '\n' || c == '\r' || c == 0x0b || c == 0x0c || c == 0x1c || c == 0x1d || c == 0x1e || c == 0x85 ||
           c == 0x2028 || c == 0x2029;
}
// str.splitlines() without keepends.
inline std::vector<ustring> splitlines(const ustring & s) {
    std::vector<ustring> out;
    size_t i = 0, start = 0;
    while (i < s.size()) {
        if (is_line_break(s[i])) {
            out.push_back(s.substr(start, i - start));
            if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') ++i;
            ++i;
            start = i;
        } else
            ++i;
    }
    if (start < s.size()) out.push_back(s.substr(start));
    return out;
}
// str.split() (whitespace, maxsplit=-1) and str.split(None, maxsplit).
inline std::vector<ustring> split_ws(const ustring & s, int maxsplit = -1) {
    std::vector<ustring> out;
    size_t i = 0;
    while (true) {
        while (i < s.size() && isspace(s[i])) ++i;
        if (i >= s.size()) break;
        if (maxsplit >= 0 && static_cast<int>(out.size()) == maxsplit) {
            // Remainder, with trailing whitespace stripped (CPython behaviour).
            size_t b = s.size();
            while (b > i && isspace(s[b - 1])) --b;
            out.push_back(s.substr(i, b - i));
            break;
        }
        size_t j = i;
        while (j < s.size() && !isspace(s[j])) ++j;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}
// str.split(sep) for nonempty sep.
inline std::vector<ustring> split(const ustring & s, const ustring & sep) {
    std::vector<ustring> out;
    size_t last = 0;
    for (size_t at = s.find(sep); at != ustring::npos; at = s.find(sep, last)) {
        out.push_back(s.substr(last, at - last));
        last = at + sep.size();
    }
    out.push_back(s.substr(last));
    return out;
}
inline ustring join(const std::vector<ustring> & parts, const ustring & sep) {
    ustring out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

// --------------------------------------------------------------- regex -----
enum flags : unsigned { NONE = 0, DOTALL = 1, IGNORECASE = 2 };

struct match {
    std::vector<long> slots; // 2 per group, -1 when unset; group 0 = whole match.
    const ustring * subject = nullptr;
    bool ok = false;
    long start(int g = 0) const { return slots[g * 2]; }
    long end(int g = 0) const { return slots[g * 2 + 1]; }
    bool has(int g) const { return slots[g * 2] >= 0; }
    ustring str(int g = 0) const {
        return has(g) ? subject->substr(static_cast<size_t>(start(g)), static_cast<size_t>(end(g) - start(g)))
                      : ustring();
    }
};

class regex {
public:
    regex(const char * pattern_utf8, unsigned f = NONE) : flags_(f) {
        pat_ = from_utf8(pattern_utf8);
        pos_ = 0;
        node_ptr root = parse_alt();
        if (pos_ != pat_.size()) throw std::invalid_argument("pyre: unbalanced ')' in pattern");
        n_groups_ = group_count_;
        prog_.clear();
        emit(*root, prog_);
        prog_.push_back({ op::match });
    }

    int groups() const { return n_groups_; }
    int group_index(const std::string & name) const {
        for (size_t i = 0; i < names_.size(); ++i)
            if (names_[i] == name) return static_cast<int>(i) + 1;
        throw std::invalid_argument("pyre: unknown group " + name);
    }

    // re.match at `pos` (anchored start).
    bool match_at(const ustring & s, size_t pos, match & m, bool full = false) const {
        return run(prog_, s, pos, m, full, false, pos);
    }
    bool fullmatch(const ustring & s, match & m) const { return match_at(s, 0, m, true); }
    bool fullmatch(const ustring & s) const {
        match m;
        return fullmatch(s, m);
    }
    // re.search from `pos`. `must_advance` rejects an empty match at `pos`.
    bool search(const ustring & s, size_t pos, match & m, bool must_advance = false) const {
        for (size_t p = pos; p <= s.size(); ++p)
            if (run(prog_, s, p, m, false, must_advance && p == pos, p)) return true;
        return false;
    }
    bool search(const ustring & s) const {
        match m;
        return search(s, 0, m);
    }

    // Iterate matches with CPython's finditer/sub progression rules.
    template <typename F> void for_each(const ustring & s, F && f) const {
        size_t pos = 0;
        bool must_advance = false;
        match m;
        while (pos <= s.size() && search(s, pos, m, must_advance)) {
            f(m);
            const size_t e = static_cast<size_t>(m.end());
            must_advance = e == static_cast<size_t>(m.start());
            pos = e;
        }
    }
    std::vector<match> finditer(const ustring & s) const {
        std::vector<match> out;
        for_each(s, [&](const match & m) { out.push_back(m); });
        return out;
    }
    // findall for patterns with zero or one group.
    std::vector<ustring> findall(const ustring & s) const {
        if (n_groups_ > 1) throw std::invalid_argument("pyre::findall supports at most one group");
        std::vector<ustring> out;
        for_each(s, [&](const match & m) { out.push_back(m.str(n_groups_)); });
        return out;
    }
    template <typename F> ustring sub_fn(const ustring & s, F && f) const {
        ustring out;
        size_t last = 0;
        for_each(s, [&](const match & m) {
            out.append(s, last, static_cast<size_t>(m.start()) - last);
            out += f(m);
            last = static_cast<size_t>(m.end());
        });
        out.append(s, last, ustring::npos);
        return out;
    }
    // re.sub with a template: supports \1..\9, \g<n>, \g<name>, \\ and
    // CPython's other escapes (\n, \t, ...). Unknown non-letter escapes are
    // kept verbatim, matching CPython.
    ustring sub(const ustring & s, const char * repl_utf8) const {
        const ustring repl = from_utf8(repl_utf8);
        return sub_fn(s, [&](const match & m) { return expand(m, repl); });
    }
    // re.split; non-participating groups become empty strings (CPython: None).
    std::vector<ustring> split(const ustring & s) const {
        std::vector<ustring> out;
        size_t last = 0;
        for_each(s, [&](const match & m) {
            out.push_back(s.substr(last, static_cast<size_t>(m.start()) - last));
            for (int g = 1; g <= n_groups_; ++g) out.push_back(m.str(g));
            last = static_cast<size_t>(m.end());
        });
        out.push_back(s.substr(last));
        return out;
    }

    ustring expand(const match & m, const ustring & repl) const {
        ustring out;
        for (size_t i = 0; i < repl.size(); ++i) {
            const char32_t c = repl[i];
            if (c != '\\' || i + 1 >= repl.size()) {
                if (c == '\\') throw std::invalid_argument("pyre: bad escape at end of template");
                out.push_back(c);
                continue;
            }
            const char32_t n = repl[++i];
            if (n >= '0' && n <= '9') {
                int g = static_cast<int>(n - '0');
                if (i + 1 < repl.size() && repl[i + 1] >= '0' && repl[i + 1] <= '9' &&
                    g * 10 + static_cast<int>(repl[i + 1] - '0') <= n_groups_) {
                    g = g * 10 + static_cast<int>(repl[++i] - '0');
                }
                if (g > n_groups_) throw std::invalid_argument("pyre: invalid group reference");
                out += m.str(g);
            } else if (n == 'g') {
                const size_t close = repl.find('>', i);
                if (i + 1 >= repl.size() || repl[i + 1] != '<' || close == ustring::npos)
                    throw std::invalid_argument("pyre: bad \\g escape");
                const std::string name = to_utf8(repl.substr(i + 2, close - i - 2));
                int g = -1;
                if (!name.empty() &&
                    std::all_of(name.begin(), name.end(), [](char ch) { return ch >= '0' && ch <= '9'; }))
                    g = std::stoi(name);
                else
                    g = group_index(name);
                out += m.str(g);
                i = close;
            } else if (n == '\\') {
                out.push_back('\\');
            } else if (n == 'n') {
                out.push_back('\n');
            } else if (n == 't') {
                out.push_back('\t');
            } else if (n == 'r') {
                out.push_back('\r');
            } else if (n == 'f') {
                out.push_back('\f');
            } else if (n == 'v') {
                out.push_back('\v');
            } else if (n == 'a') {
                out.push_back('\a');
            } else if (n == 'b') {
                out.push_back('\b');
            } else if ((n >= 'a' && n <= 'z') || (n >= 'A' && n <= 'Z')) {
                throw std::invalid_argument("pyre: bad template escape");
            } else {
                out.push_back('\\');
                out.push_back(n);
            }
        }
        return out;
    }

private:
    enum class op : uint8_t {
        chr,
        any,
        cls,
        split,
        jmp,
        save,
        bol,
        eol,
        bos,
        eos,
        wordb,
        nwordb,
        look,
        backref,
        mark,
        progress,
        match
    };
    struct inst {
        op o;
        char32_t c = 0;
        int x = 0, y = 0;  // targets / indices
        bool flag = false; // icase for chr/cls/backref; negate for look/cls
        bool behind = false;
    };
    struct cls_item {
        char32_t lo, hi;
        int kind; // 0 range, 1 \s, 2 \S, 3 \d, 4 \D, 5 \w, 6 \W
    };
    struct char_class {
        std::vector<cls_item> items;
        bool negate = false;
    };
    struct node;
    using node_ptr = std::shared_ptr<node>;
    enum class nk {
        empty,
        chr,
        any,
        cls,
        cat,
        alt,
        rep,
        group,
        ncgroup,
        bol,
        eol,
        bos,
        eos,
        wordb,
        nwordb,
        look,
        backref
    };
    struct node {
        nk k = nk::empty;
        char32_t c = 0;
        int cls = -1;
        std::vector<node_ptr> kids;
        int min = 0, max = -1; // max -1 = inf
        bool lazy = false;
        int group = 0;
        bool negate = false, behind = false;
    };

    ustring pat_;
    size_t pos_ = 0;
    unsigned flags_;
    int group_count_ = 0, n_groups_ = 0, n_regs_ = 0;
    std::vector<std::string> names_;
    std::vector<char_class> classes_;
    std::vector<std::vector<inst>> looks_;
    std::vector<int> look_width_;
    std::vector<inst> prog_;

    [[noreturn]] void fail(const char * what) const {
        throw std::invalid_argument(std::string("pyre: ") + what + " in pattern " + to_utf8(pat_));
    }
    bool more() const { return pos_ < pat_.size(); }
    char32_t peek() const { return pat_[pos_]; }

    node_ptr make(nk k) {
        auto n = std::make_shared<node>();
        n->k = k;
        return n;
    }

    node_ptr parse_alt() {
        std::vector<node_ptr> alts{ parse_cat() };
        while (more() && peek() == '|') {
            ++pos_;
            alts.push_back(parse_cat());
        }
        if (alts.size() == 1) return alts[0];
        auto n = make(nk::alt);
        n->kids = alts;
        return n;
    }

    node_ptr parse_cat() {
        auto n = make(nk::cat);
        while (more() && peek() != '|' && peek() != ')') {
            node_ptr atom = parse_atom();
            atom = parse_quant(atom);
            n->kids.push_back(atom);
        }
        return n;
    }

    bool parse_brace(int & mn, int & mx) {
        // CPython: '{' starts a quantifier only for {m}, {m,}, {,n}, {m,n}.
        size_t p = pos_ + 1;
        auto digits = [&](int & v) {
            size_t s = p;
            v = 0;
            while (p < pat_.size() && pat_[p] >= '0' && pat_[p] <= '9') v = v * 10 + static_cast<int>(pat_[p++] - '0');
            return p > s;
        };
        int a = 0, b = 0;
        const bool has_a = digits(a);
        if (p < pat_.size() && pat_[p] == '}') {
            if (!has_a) return false;
            mn = mx = a;
            pos_ = p + 1;
            return true;
        }
        if (p >= pat_.size() || pat_[p] != ',') return false;
        ++p;
        const bool has_b = digits(b);
        if (p >= pat_.size() || pat_[p] != '}') return false;
        mn = has_a ? a : 0;
        mx = has_b ? b : -1;
        pos_ = p + 1;
        return true;
    }

    node_ptr parse_quant(node_ptr atom) {
        while (more()) {
            int mn = 0, mx = -1;
            const char32_t c = peek();
            if (c == '*') {
                ++pos_;
                mn = 0, mx = -1;
            } else if (c == '+') {
                ++pos_;
                mn = 1, mx = -1;
            } else if (c == '?') {
                ++pos_;
                mn = 0, mx = 1;
            } else if (c == '{') {
                if (!parse_brace(mn, mx)) return atom;
            } else
                return atom;
            if (atom->k == nk::rep) fail("multiple repeat");
            auto r = make(nk::rep);
            r->min = mn;
            r->max = mx;
            if (more() && peek() == '?') {
                r->lazy = true;
                ++pos_;
            } else if (more() && peek() == '+')
                fail("possessive quantifier unsupported");
            r->kids.push_back(atom);
            atom = r;
            if (mx >= 0 && mn > mx) fail("min repeat greater than max repeat");
        }
        return atom;
    }

    int hexval(size_t n) {
        if (pos_ + n > pat_.size()) fail("truncated \\u escape");
        int v = 0;
        for (size_t i = 0; i < n; ++i) {
            const char32_t h = pat_[pos_++];
            v *= 16;
            if (h >= '0' && h <= '9')
                v += static_cast<int>(h - '0');
            else if (h >= 'a' && h <= 'f')
                v += static_cast<int>(h - 'a' + 10);
            else if (h >= 'A' && h <= 'F')
                v += static_cast<int>(h - 'A' + 10);
            else
                fail("bad hex escape");
        }
        return v;
    }

    // Parse an escape after '\\'. Returns literal in `lit` or class kind (>0).
    int parse_escape(char32_t & lit, bool in_class) {
        if (!more()) fail("trailing backslash");
        const char32_t c = pat_[pos_++];
        switch (c) {
        case 's':
            return 1;
        case 'S':
            return 2;
        case 'd':
            return 3;
        case 'D':
            return 4;
        case 'w':
            return 5;
        case 'W':
            return 6;
        case 'n':
            lit = '\n';
            return 0;
        case 't':
            lit = '\t';
            return 0;
        case 'r':
            lit = '\r';
            return 0;
        case 'f':
            lit = '\f';
            return 0;
        case 'v':
            lit = '\v';
            return 0;
        case 'a':
            lit = '\a';
            return 0;
        case 'u':
            lit = static_cast<char32_t>(hexval(4));
            return 0;
        case 'U':
            lit = static_cast<char32_t>(hexval(8));
            return 0;
        case 'x':
            lit = static_cast<char32_t>(hexval(2));
            return 0;
        case 'b':
            if (in_class) {
                lit = '\b';
                return 0;
            }
            return 10;
        case 'B':
            return in_class ? (fail("bad escape \\B in class"), 0) : 11;
        case 'A':
            return in_class ? (fail("bad escape"), 0) : 12;
        case 'Z':
            return in_class ? (fail("bad escape"), 0) : 13;
        default:
            if (c >= '0' && c <= '9') {
                if (in_class || c == '0') fail("octal escapes unsupported");
                lit = c;
                return 20; // backreference digit
            }
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) fail("bad escape");
            lit = c;
            return 0;
        }
    }

    int parse_class() {
        char_class cc;
        if (more() && peek() == '^') {
            cc.negate = true;
            ++pos_;
        }
        bool first = true;
        while (true) {
            if (!more()) fail("unterminated character set");
            char32_t c = peek();
            if (c == ']' && !first) {
                ++pos_;
                break;
            }
            first = false;
            char32_t lo = 0;
            ++pos_;
            if (c == '\\') {
                const int kind = parse_escape(lo, true);
                if (kind) {
                    cc.items.push_back({ 0, 0, kind });
                    continue;
                }
            } else
                lo = c;
            if (more() && peek() == '-' && pos_ + 1 < pat_.size() && pat_[pos_ + 1] != ']') {
                ++pos_;
                char32_t hi = pat_[pos_++];
                if (hi == '\\') {
                    const int kind = parse_escape(hi, true);
                    if (kind) fail("bad character range");
                }
                if (hi < lo) fail("bad character range");
                cc.items.push_back({ lo, hi, 0 });
            } else
                cc.items.push_back({ lo, lo, 0 });
        }
        classes_.push_back(cc);
        return static_cast<int>(classes_.size()) - 1;
    }

    node_ptr parse_atom() {
        const char32_t c = pat_[pos_++];
        if (c == '(') {
            node_ptr n;
            if (more() && peek() == '?') {
                ++pos_;
                if (!more()) fail("bad group");
                const char32_t t = pat_[pos_++];
                if (t == ':') {
                    n = make(nk::ncgroup);
                } else if (t == 'P' && more() && peek() == '<') {
                    const size_t close = pat_.find('>', pos_);
                    if (close == ustring::npos) fail("bad group name");
                    const std::string name = to_utf8(pat_.substr(pos_ + 1, close - pos_ - 1));
                    pos_ = close + 1;
                    n = make(nk::group);
                    n->group = ++group_count_;
                    names_.resize(static_cast<size_t>(group_count_));
                    names_[static_cast<size_t>(group_count_) - 1] = name;
                } else if (t == '=' || t == '!') {
                    n = make(nk::look);
                    n->negate = t == '!';
                } else if (t == '<' && more() && (peek() == '=' || peek() == '!')) {
                    n = make(nk::look);
                    n->behind = true;
                    n->negate = pat_[pos_++] == '!';
                } else
                    fail("unsupported group syntax");
            } else {
                n = make(nk::group);
                n->group = ++group_count_;
                names_.resize(static_cast<size_t>(group_count_));
            }
            n->kids.push_back(parse_alt());
            if (!more() || peek() != ')') fail("missing )");
            ++pos_;
            return n;
        }
        if (c == '[') {
            auto n = make(nk::cls);
            n->cls = parse_class();
            return n;
        }
        if (c == '.') return make(nk::any);
        if (c == '^') return make(nk::bol);
        if (c == '$') return make(nk::eol);
        if (c == '*' || c == '+' || c == '?') fail("nothing to repeat");
        if (c == '\\') {
            char32_t lit = 0;
            const int kind = parse_escape(lit, false);
            if (kind == 0) {
                auto n = make(nk::chr);
                n->c = lit;
                return n;
            }
            if (kind >= 1 && kind <= 6) {
                char_class cc;
                cc.items.push_back({ 0, 0, kind });
                classes_.push_back(cc);
                auto n = make(nk::cls);
                n->cls = static_cast<int>(classes_.size()) - 1;
                return n;
            }
            if (kind == 10) return make(nk::wordb);
            if (kind == 11) return make(nk::nwordb);
            if (kind == 12) return make(nk::bos);
            if (kind == 13) return make(nk::eos);
            auto n = make(nk::backref);
            n->group = static_cast<int>(lit - '0');
            if (more() && peek() >= '0' && peek() <= '9') fail("multi-digit backreference unsupported");
            if (n->group > group_count_) fail("invalid group reference");
            return n;
        }
        auto n = make(nk::chr);
        n->c = c;
        return n;
    }

    // Fixed width of a node (for lookbehind), or -1 if variable.
    int width(const node & n) const {
        switch (n.k) {
        case nk::chr:
        case nk::any:
        case nk::cls:
            return 1;
        case nk::empty:
        case nk::bol:
        case nk::eol:
        case nk::bos:
        case nk::eos:
        case nk::wordb:
        case nk::nwordb:
        case nk::look:
            return 0;
        case nk::cat: {
            int w = 0;
            for (const auto & k : n.kids) {
                const int x = width(*k);
                if (x < 0) return -1;
                w += x;
            }
            return w;
        }
        case nk::alt: {
            int w = -2;
            for (const auto & k : n.kids) {
                const int x = width(*k);
                if (x < 0 || (w != -2 && x != w)) return -1;
                w = x;
            }
            return w < 0 ? 0 : w;
        }
        case nk::group:
        case nk::ncgroup:
            return width(*n.kids[0]);
        case nk::rep: {
            if (n.max != n.min) return -1;
            const int x = width(*n.kids[0]);
            return x < 0 ? -1 : x * n.min;
        }
        default:
            return -1;
        }
    }

    void emit(const node & n, std::vector<inst> & p) {
        const bool icase = (flags_ & IGNORECASE) != 0;
        switch (n.k) {
        case nk::empty:
            break;
        case nk::chr:
            p.push_back({ op::chr, n.c, 0, 0, icase });
            break;
        case nk::any:
            p.push_back({ op::any, 0, 0, 0, (flags_ & DOTALL) != 0 });
            break;
        case nk::cls:
            p.push_back({ op::cls, 0, n.cls, 0, icase });
            break;
        case nk::bol:
            p.push_back({ op::bol });
            break;
        case nk::eol:
            p.push_back({ op::eol });
            break;
        case nk::bos:
            p.push_back({ op::bos });
            break;
        case nk::eos:
            p.push_back({ op::eos });
            break;
        case nk::wordb:
            p.push_back({ op::wordb });
            break;
        case nk::nwordb:
            p.push_back({ op::nwordb });
            break;
        case nk::backref:
            p.push_back({ op::backref, 0, n.group, 0, icase });
            break;
        case nk::cat:
            for (const auto & k : n.kids) emit(*k, p);
            break;
        case nk::group:
            p.push_back({ op::save, 0, n.group * 2 });
            emit(*n.kids[0], p);
            p.push_back({ op::save, 0, n.group * 2 + 1 });
            break;
        case nk::ncgroup:
            emit(*n.kids[0], p);
            break;
        case nk::alt: {
            std::vector<size_t> jumps;
            for (size_t i = 0; i < n.kids.size(); ++i) {
                if (i + 1 < n.kids.size()) {
                    const size_t s = p.size();
                    p.push_back({ op::split });
                    p[s].x = static_cast<int>(p.size());
                    emit(*n.kids[i], p);
                    jumps.push_back(p.size());
                    p.push_back({ op::jmp });
                    p[s].y = static_cast<int>(p.size());
                } else
                    emit(*n.kids[i], p);
            }
            for (size_t j : jumps) p[j].x = static_cast<int>(p.size());
            break;
        }
        case nk::look: {
            std::vector<inst> sub;
            emit(*n.kids[0], sub);
            sub.push_back({ op::match });
            int w = 0;
            if (n.behind) {
                w = width(*n.kids[0]);
                if (w < 0) fail("look-behind requires fixed-width pattern");
            }
            looks_.push_back(sub);
            look_width_.push_back(w);
            inst i{ op::look };
            i.x = static_cast<int>(looks_.size()) - 1;
            i.flag = n.negate;
            i.behind = n.behind;
            p.push_back(i);
            break;
        }
        case nk::rep: {
            const node & body = *n.kids[0];
            for (int i = 0; i < n.min; ++i) emit(body, p);
            if (n.max < 0) {
                const int reg = n_regs_++;
                const size_t loop = p.size();
                p.push_back({ op::split });
                const size_t body_at = p.size();
                p.push_back({ op::mark, 0, reg });
                emit(body, p);
                p.push_back({ op::progress, 0, reg });
                p.push_back({ op::jmp, 0, static_cast<int>(loop) });
                const size_t out = p.size();
                p[loop].x = static_cast<int>(n.lazy ? out : body_at);
                p[loop].y = static_cast<int>(n.lazy ? body_at : out);
            } else {
                std::vector<size_t> splits;
                for (int i = n.min; i < n.max; ++i) {
                    splits.push_back(p.size());
                    p.push_back({ op::split });
                    p.back().x = static_cast<int>(p.size()); // tentative
                    emit(body, p);
                }
                const size_t out = p.size();
                for (size_t s : splits) {
                    const int body_at = static_cast<int>(s + 1);
                    p[s].x = n.lazy ? static_cast<int>(out) : body_at;
                    p[s].y = n.lazy ? body_at : static_cast<int>(out);
                }
            }
            break;
        }
        }
    }

    bool class_has(int idx, char32_t c, bool icase) const {
        const char_class & cc = classes_[static_cast<size_t>(idx)];
        auto test = [&](char32_t ch) {
            for (const auto & it : cc.items) {
                switch (it.kind) {
                case 0:
                    if (ch >= it.lo && ch <= it.hi) return true;
                    break;
                case 1:
                    if (isspace(ch)) return true;
                    break;
                case 2:
                    if (!isspace(ch)) return true;
                    break;
                case 3:
                    if (isdecimal(ch)) return true;
                    break;
                case 4:
                    if (!isdecimal(ch)) return true;
                    break;
                case 5:
                    if (isword(ch)) return true;
                    break;
                case 6:
                    if (!isword(ch)) return true;
                    break;
                }
            }
            return false;
        };
        bool hit = test(c);
        if (!hit && icase) {
            const char32_t l = simple_lower(c);
            hit = test(l) || (l >= 'a' && l <= 'z' && test(l - 32));
        }
        return cc.negate ? !hit : hit;
    }

    struct frame {
        int pc;
        long pos;
        int kind; // 0 branch, 1 restore slot, 2 restore reg
        int idx;
        long old;
    };

    bool run(const std::vector<inst> & prog, const ustring & s, size_t start, match & m, bool full, bool must_advance,
             size_t origin, long limit = -1) const {
        std::vector<long> slots(static_cast<size_t>(n_groups_ + 1) * 2, -1), regs(static_cast<size_t>(n_regs_), -2);
        std::vector<frame> stack;
        int pc = 0;
        long pos = static_cast<long>(start);
        const long len = limit >= 0 ? limit : static_cast<long>(s.size());
        auto backtrack = [&]() -> bool {
            while (!stack.empty()) {
                frame f = stack.back();
                stack.pop_back();
                if (f.kind == 1)
                    slots[static_cast<size_t>(f.idx)] = f.old;
                else if (f.kind == 2)
                    regs[static_cast<size_t>(f.idx)] = f.old;
                else {
                    pc = f.pc;
                    pos = f.pos;
                    return true;
                }
            }
            return false;
        };
        while (true) {
            const inst & in = prog[static_cast<size_t>(pc)];
            bool ok = true;
            switch (in.o) {
            case op::chr:
                if (pos < len && (s[static_cast<size_t>(pos)] == in.c ||
                                  (in.flag && simple_lower(s[static_cast<size_t>(pos)]) == simple_lower(in.c)))) {
                    ++pos;
                    ++pc;
                } else
                    ok = false;
                break;
            case op::any:
                if (pos < len && (in.flag || s[static_cast<size_t>(pos)] != '\n')) {
                    ++pos;
                    ++pc;
                } else
                    ok = false;
                break;
            case op::cls:
                if (pos < len && class_has(in.x, s[static_cast<size_t>(pos)], in.flag)) {
                    ++pos;
                    ++pc;
                } else
                    ok = false;
                break;
            case op::split:
                stack.push_back({ in.y, pos, 0, 0, 0 });
                pc = in.x;
                break;
            case op::jmp:
                pc = in.x;
                break;
            case op::save:
                stack.push_back({ 0, 0, 1, in.x, slots[static_cast<size_t>(in.x)] });
                slots[static_cast<size_t>(in.x)] = pos;
                ++pc;
                break;
            case op::mark:
                stack.push_back({ 0, 0, 2, in.x, regs[static_cast<size_t>(in.x)] });
                regs[static_cast<size_t>(in.x)] = pos;
                ++pc;
                break;
            case op::progress:
                if (regs[static_cast<size_t>(in.x)] == pos)
                    ok = false; // empty iteration: stop looping (CPython)
                else
                    ++pc;
                break;
            case op::bol:
                if (pos == 0)
                    ++pc;
                else
                    ok = false;
                break;
            case op::eol:
                if (pos == len || (pos == len - 1 && s[static_cast<size_t>(pos)] == '\n'))
                    ++pc;
                else
                    ok = false;
                break;
            case op::bos:
                if (pos == 0)
                    ++pc;
                else
                    ok = false;
                break;
            case op::eos:
                if (pos == len)
                    ++pc;
                else
                    ok = false;
                break;
            case op::wordb:
            case op::nwordb: {
                const bool a = pos > 0 && isword(s[static_cast<size_t>(pos - 1)]);
                const bool b = pos < len && isword(s[static_cast<size_t>(pos)]);
                const bool boundary = a != b;
                if (boundary == (in.o == op::wordb))
                    ++pc;
                else
                    ok = false;
                break;
            }
            case op::backref: {
                const long a = slots[static_cast<size_t>(in.x * 2)], b = slots[static_cast<size_t>(in.x * 2 + 1)];
                if (a < 0 || b < 0) {
                    ok = false;
                    break;
                }
                const long n = b - a;
                if (pos + n > len) {
                    ok = false;
                    break;
                }
                for (long k = 0; k < n && ok; ++k) {
                    const char32_t x = s[static_cast<size_t>(a + k)], y = s[static_cast<size_t>(pos + k)];
                    if (x != y && !(in.flag && simple_lower(x) == simple_lower(y))) ok = false;
                }
                if (ok) {
                    pos += n;
                    ++pc;
                }
                break;
            }
            case op::look: {
                match sub;
                bool hit = false;
                const int w = look_width_[static_cast<size_t>(in.x)];
                if (in.behind) {
                    if (pos - w >= 0)
                        hit = run_sub(looks_[static_cast<size_t>(in.x)], s, static_cast<size_t>(pos - w), pos, sub);
                } else
                    hit = run_sub(looks_[static_cast<size_t>(in.x)], s, static_cast<size_t>(pos), -1, sub);
                if (hit != in.flag)
                    ++pc;
                else
                    ok = false;
                break;
            }
            case op::match:
                if ((full && pos != len) || (must_advance && pos == static_cast<long>(origin)))
                    ok = false;
                else {
                    slots[0] = static_cast<long>(start);
                    slots[1] = pos;
                    m.slots = slots;
                    m.subject = &s;
                    m.ok = true;
                    return true;
                }
                break;
            }
            if (!ok && !backtrack()) return false;
        }
    }

    // Lookaround subprogram: captures inside are discarded (not used by ports).
    // A lookbehind must end exactly at the current position.
    bool run_sub(const std::vector<inst> & sub, const ustring & s, size_t start, long must_end, match & m) const {
        if (must_end >= 0) return run(sub, s, start, m, true, false, start, must_end);
        return run(sub, s, start, m, false, false, start);
    }
};

} // namespace pyre
