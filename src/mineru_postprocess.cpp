// mineru_postprocess.cpp - C++ translation of mineru-vl-utils post_process/
// (c9797cde8f54cc48306196fc6aebe73ea31dd4a3, Apache-2.0, MinerU Team).
// Modified: translated to C++; debug logging removed. Every function keeps
// the upstream name, order of operations and regular expressions verbatim.
#include "mineru_postprocess.h"

#include "mineru_equation_big_table.h"
#include "otsl_table.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace mineru_pp {

using pyre::regex;
using pyre::U;

namespace {
struct python_type_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

ustring u(const char * s) {
    return pyre::from_utf8(s);
}
} // namespace

// ------------------------------------------------------------ block --------
bool block::has_key(const std::string & k) const {
    return std::find(extra_keys.begin(), extra_keys.end(), k) != extra_keys.end();
}
void block::add_key(const std::string & k) {
    if (!has_key(k)) extra_keys.push_back(k);
}
void block::remove_key(const std::string & k) {
    extra_keys.erase(std::remove(extra_keys.begin(), extra_keys.end(), k), extra_keys.end());
}
void block::set_type(const std::string & value) {
    const bool prev = has_key("merge_prev") ? merge_prev : false;
    type = value;
    if (value == "text") {
        merge_prev = prev;
        add_key("merge_prev");
    } else {
        remove_key("merge_prev");
        merge_prev = false;
    }
}
block_ptr make_block(const std::string & type, const std::array<double, 4> & bbox, int angle, bool merge_prev) {
    auto b = std::make_shared<block>();
    b->type = type;
    b->bbox = bbox;
    b->angle = angle;
    if (type == "text") {
        b->merge_prev = merge_prev;
        b->extra_keys.push_back("merge_prev");
    }
    return b;
}

// ----------------------------------------------------------- otsl2html -----
ustring html_escape(const ustring & s) {
    ustring out;
    for (char32_t c : s) {
        switch (c) {
        case '&':
            out += U"&amp;";
            break;
        case '<':
            out += U"&lt;";
            break;
        case '>':
            out += U"&gt;";
            break;
        case '"':
            out += U"&quot;";
            break;
        case '\'':
            out += U"&#x27;";
            break;
        default:
            out.push_back(c);
        }
    }
    return out;
}

ustring convert_otsl_to_html(const ustring & otsl) {
    return otsl_table::to_html_upstream(otsl);
}

// --------------------------------------------------- equation repairs ------
ustring try_fix_equation_delimeters(const ustring & latex) {
    ustring s = pyre::strip(latex);
    if (s.substr(0, 2) == U"\\[") s = s.substr(2);
    if (s.size() >= 2 && s.substr(s.size() - 2) == U"\\]")
        s = s.substr(0, s.size() - 2);
    else if (s.size() < 2 && s == U"\\]")
        s.clear();
    return pyre::strip(s);
}

ustring try_fix_equation_leq(const ustring & latex) {
    return pyre::replace(latex, U"<", U"< ");
}

ustring try_fix_equation_double_subscript(const ustring & latex) {
    static const regex p(R"(_\s*\{([^{}]|\{[^{}]*\})*\}\s*_\s*\{([^{}]|\{[^{}]*\})*\})");
    if (!p.search(latex)) return latex;
    return p.sub(latex, "");
}

ustring try_fix_equation_eqqcolon(const ustring & latex) {
    ustring s = pyre::replace(latex, U"\\eqqcolon", U"=:");
    return pyre::replace(s, U"\\coloneqq", U":=");
}

ustring try_fix_equation_big(const ustring & latex) {
    ustring s = latex;
    for (const auto & row : mineru_equation_big_table) s = pyre::replace(s, u(row[0]), u(row[1]));
    return s;
}

ustring try_fix_unbalanced_braces(const ustring & f) {
    std::vector<size_t> stack;
    std::set<size_t> unmatched;
    for (size_t i = 0; i < f.size(); ++i) {
        if (f[i] != '{' && f[i] != '}') continue;
        size_t bs = 0;
        for (size_t j = i; j > 0 && f[j - 1] == '\\'; --j) ++bs;
        if (bs % 2 == 1) continue;
        if (f[i] == '{')
            stack.push_back(i);
        else if (!stack.empty())
            stack.pop_back();
        else
            unmatched.insert(i);
    }
    unmatched.insert(stack.begin(), stack.end());
    ustring out;
    for (size_t i = 0; i < f.size(); ++i)
        if (!unmatched.count(i)) out.push_back(f[i]);
    return out;
}

// equation_left_right ---------------------------------------------------------
namespace {
const std::vector<ustring> & valid_left() {
    static const std::vector<ustring> v = { U"\\left\\lbrace", U"\\left\\lVert",  U"\\left\\lvert", U"\\left\\rvert",
                                            U"\\left\\rVert",  U"\\left\\vert",   U"\\left\\Vert",  U"\\left\\lfloor",
                                            U"\\left\\lbrack", U"\\left\\langle", U"\\left|",       U"\\left\\|",
                                            U"\\left[",        U"\\left]",        U"\\left(",       U"\\left)",
                                            U"\\left\\{",      U"\\left\\}",      U"\\left.",       U"\\left/" };
    return v;
}
const std::vector<ustring> & valid_right() {
    static const std::vector<ustring> v = {
        U"\\right\\rbrace", U"\\right\\lVert", U"\\right\\lvert",  U"\\right\\rvert",  U"\\right\\rVert",
        U"\\right\\vert",   U"\\right\\Vert",  U"\\right\\rfloor", U"\\right\\rbrack", U"\\right\\rangle",
        U"\\right|",        U"\\right\\|",     U"\\right]",        U"\\right[",        U"\\right)",
        U"\\right(",        U"\\right\\}",     U"\\right\\{",      U"\\right.",        U"\\right/"
    };
    return v;
}
#define MINERU_LEFT_PATTERN                                                                                            \
    R"(\\left\\lbrace|\\left\\lVert|\\left\\lvert|\\left\\rvert|\\left\\rVert|\\left\\vert|\\left\\Vert|\\left\\lfloor|\\left\\lbrack|\\left\\langle|\\left\||\\left\\\||\\left\[|\\left\]|\\left\(|\\left\\{|\\left\\}|\\left\.)"
#define MINERU_RIGHT_PATTERN                                                                                           \
    R"(\\right\\rbrace|\\right\\lVert|\\right\\lvert|\\right\\rvert|\\right\\rVert|\\right\\vert|\\right\\Vert|\\right\\rfloor|\\right\\rbrack|\\right\\rangle|\\right\||\\right\\\||\\right\]|\\right\[|\\right\)|\\right\\}|\\right\\{|\\right.)"

size_t count_left(const ustring & s) {
    static const regex p(MINERU_LEFT_PATTERN);
    return p.findall(s).size();
}
size_t count_right(const ustring & s) {
    static const regex p(MINERU_RIGHT_PATTERN);
    return p.findall(s).size();
}
bool in_list(const std::vector<ustring> & v, const ustring & s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}
bool is_pair_left_right(const ustring & l, const ustring & r) {
    static const char * pairs[][2] = {
        { "\\left\\lbrace", "\\right\\rbrace" },
        { "\\left\\lVert", "\\right\\lVert" },
        { "\\left\\lvert", "\\right\\lvert" },
        { "\\left\\vert", "\\right\\vert" },
        { "\\left\\Vert", "\\right\\Vert" },
        { "\\left\\lfloor", "\\right\\rfloor" },
        { "\\left\\lbrack", "\\right\\rbrack" },
        { "\\left\\langle", "\\right\\rangle" },
        { "\\left|", "\\right|" },
        { "\\left\\|", "\\right\\|" },
        { "\\left[", "\\right]" },
        { "\\left]", "\\right[" },
        { "\\left(", "\\right)" },
        { "\\left)", "\\right(" },
        { "\\left\\{", "\\right\\}" },
        { "\\left\\}", "\\right\\{" },
        { "\\left/", "\\right/" },
    };
    for (const auto & p : pairs)
        if ((l == u(p[0]) || l == U"\\left.") && (r == u(p[1]) || r == U"\\right.")) return true;
    return false;
}
// node tag: (array_tag, element_idx) or None
struct tag {
    bool none = true;
    long a = 0, e = 0;
    bool operator==(const tag & o) const { return none == o.none && (none || (a == o.a && e == o.e)); }
    bool operator<(const tag & o) const { return std::tie(none, a, e) < std::tie(o.none, o.a, o.e); }
};
struct arr {
    long tag, start, end;
    std::vector<long> contain;
};
bool tag_array(const std::vector<ustring> & nodes, std::vector<arr> & out) {
    std::vector<std::pair<long, long>> stack;
    long array_tag = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (pyre::contains(nodes[i], U"\\begin{array}")) {
            ++array_tag;
            stack.push_back({ array_tag, static_cast<long>(i) });
        } else if (nodes[i] == U"\\end{array}") {
            if (stack.empty()) return false;
            auto top = stack.back();
            stack.pop_back();
            array_tag = top.first; // upstream rebinds array_tag on pop
            out.push_back({ top.first, top.second, static_cast<long>(i), {} });
        }
    }
    if (!stack.empty()) return false;
    for (size_t a = 0; a < out.size(); ++a)
        for (size_t b = 0; b < out.size(); ++b) {
            if (a == b) continue;
            if (out[a].start < out[b].start && out[a].end > out[b].end) out[a].contain.push_back(out[b].tag);
        }
    return true;
}
std::vector<tag> tag_element(const std::vector<ustring> & nodes, const std::vector<arr> & arrays) {
    if (arrays.empty()) return { tag{} };
    std::map<long, const arr *> tag2arr;
    for (const auto & a : arrays) tag2arr[a.tag] = &a; // dict: later duplicates win
    std::vector<std::vector<tag>> all;
    for (const auto & a : arrays) {
        std::vector<bool> masked(nodes.size(), false);
        std::vector<tag> cur(nodes.size());
        for (long t : a.contain) {
            const arr * c = tag2arr[t];
            for (long i = c->start; i <= c->end; ++i) masked[static_cast<size_t>(i)] = true;
        }
        for (size_t i = 0; i < nodes.size(); ++i)
            if (static_cast<long>(i) < a.start || static_cast<long>(i) > a.end) masked[i] = true;
        long element = 0;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (masked[i]) continue; // None (checked after & and \\ in upstream: masked nodes are None)
            if (nodes[i] == U"&" || nodes[i] == U"\\\\")
                ++element;
            else {
                cur[i].none = false;
                cur[i].a = a.tag;
                cur[i].e = element;
            }
        }
        all.push_back(cur);
    }
    std::vector<tag> out(all[0].size());
    for (size_t i = 0; i < out.size(); ++i)
        for (const auto & lst : all)
            if (!lst[i].none) {
                out[i] = lst[i];
                break;
            }
    return out;
}
std::vector<ustring> split_with_left_right(const ustring & s) {
    static const regex p("(" MINERU_LEFT_PATTERN "|" MINERU_RIGHT_PATTERN ")");
    return p.split(s);
}
bool structural(const ustring & n) {
    return n == U"&" || n == U"\\\\" || pyre::contains(n, U"\\begin{array}") || pyre::contains(n, U"\\end{array}");
}
std::vector<ustring> clean_span(const std::vector<ustring> & nodes, const std::vector<tag> & tags) {
    std::vector<ustring> out = nodes;
    std::vector<size_t> valid;
    std::set<tag> spans;
    for (size_t i = 0; i < tags.size(); ++i)
        if (!structural(nodes[i])) {
            valid.push_back(i);
            spans.insert(tags[i]);
        }
    // Upstream iterates a set; spans own disjoint node indices so order is immaterial.
    for (const auto & span : spans) {
        std::vector<size_t> same;
        for (size_t i : valid)
            if (tags[i] == span) same.push_back(i);
        size_t nl = 0, nr = 0;
        for (size_t i : same) {
            nl += count_left(nodes[i]);
            nr += count_right(nodes[i]);
        }
        if (nl == nr) continue;
        std::vector<std::vector<ustring>> fixed;
        for (size_t i : same) {
            std::vector<ustring> parts;
            for (auto & p : split_with_left_right(nodes[i]))
                if (!pyre::strip(p).empty()) parts.push_back(p);
            fixed.push_back(parts);
        }
        std::vector<ustring> lr;
        std::vector<std::pair<size_t, size_t>> lr_idx;
        for (size_t a = 0; a < fixed.size(); ++a)
            for (size_t b = 0; b < fixed[a].size(); ++b) {
                const ustring & t = fixed[a][b];
                if (!in_list(valid_left(), t) && !in_list(valid_right(), t)) continue;
                if (lr.empty() && lr_idx.empty()) {
                    lr.push_back(t);
                    lr_idx.push_back({ a, b });
                } else if (is_pair_left_right(lr.back(), t)) {
                    lr.pop_back();
                    lr_idx.pop_back();
                } else {
                    lr.push_back(t);
                    lr_idx.push_back({ a, b });
                }
            }
        for (size_t k = 0; k < lr.size(); ++k) {
            auto & cell = fixed[lr_idx[k].first][lr_idx[k].second];
            if (pyre::contains(lr[k], U"\\left"))
                cell = cell + U" \\right.";
            else if (pyre::contains(lr[k], U"\\right"))
                cell = U"\\left. " + cell;
        }
        for (size_t k = 0; k < same.size(); ++k) out[same[k]] = pyre::join(fixed[k], U"");
    }
    return out;
}
} // namespace

ustring try_match_equation_left_right(const ustring & latex) {
    if (count_left(latex) == count_right(latex)) return latex;
    if (pyre::count(latex, U"\\begin{array}") != pyre::count(latex, U"\\end{array}")) return latex;
    // fix_left_right_mismatch
    static const regex delim(R"((\&|\\\\|\\begin{array}\s*{[a-zA-Z\s]*}|\\end{array}))");
    std::vector<ustring> nodes;
    for (auto & n : delim.split(pyre::strip(pyre::strip(latex))))
        if (!n.empty()) nodes.push_back(pyre::strip(n));
    std::vector<arr> arrays;
    if (!tag_array(nodes, arrays)) return latex;
    std::stable_sort(arrays.begin(), arrays.end(),
                     [](const arr & a, const arr & b) { return a.contain.size() > b.contain.size(); });
    const auto tags = tag_element(nodes, arrays);
    return pyre::join(clean_span(nodes, tags), U"");
}

ustring process_equation(const ustring & content) {
    ustring c = try_fix_equation_delimeters(content);
    c = try_match_equation_left_right(c);
    c = try_fix_equation_double_subscript(c);
    c = try_fix_equation_eqqcolon(c);
    c = try_fix_equation_big(c);
    c = try_fix_equation_leq(c);
    c = try_fix_unbalanced_braces(c);
    return c;
}

ustring add_equation_brackets(const ustring & content) {
    ustring c = pyre::strip(content);
    if (!pyre::starts(c, U"\\[")) c = U"\\[\n" + c;
    if (!pyre::ends(c, U"\\]")) c = c + U"\n\\]";
    return c;
}

// ------------------------------------------------------- text repairs ------
ustring try_convert_display_to_inline(const ustring & text) {
    static const regex p(R"(\\\[(.*?)\\\])", pyre::DOTALL);
    static const regex digits(u8"[–\\d\\-,\\s]+");
    return p.sub_fn(text, [&](const pyre::match & m) {
        const ustring inner = m.str(1);
        if (digits.fullmatch(inner)) return U"\\[" + inner + U"\\]";
        return U"\\(" + inner + U"\\)";
    });
}

ustring try_fix_macro_spacing_in_markdown(const ustring & text) {
    static const std::vector<ustring> known = { U"\\top", U"\\int", U"\\inf" };
    static const std::vector<std::pair<ustring, std::shared_ptr<regex>>> targets = [] {
        std::vector<std::pair<ustring, std::shared_ptr<regex>>> v;
        // re.escape(macro) + r"([a-zA-Z])(?![a-zA-Z])"
        for (const char * m : { "cong", "to", "times", "subset", "in" })
            v.push_back({ U"\\" + pyre::from_utf8(m),
                          std::make_shared<regex>((std::string("\\\\") + m + "([a-zA-Z])(?![a-zA-Z])").c_str()) });
        return v;
    }();
    static const regex splitter(R"((\\\(.*?\\\)))", pyre::DOTALL);
    ustring result;
    for (const auto & part : splitter.split(text)) {
        if (pyre::starts(part, U"\\(") && pyre::ends(part, U"\\)")) {
            ustring inner = part.size() >= 4 ? part.substr(2, part.size() - 4) : ustring();
            for (const auto & t : targets) {
                inner = t.second->sub_fn(inner, [&](const pyre::match & m) {
                    const ustring letter = m.str(1);
                    if (in_list(known, t.first + letter)) return m.str(0);
                    return t.first + U" " + letter;
                });
            }
            result += U"\\(" + inner + U"\\)";
        } else
            result += part;
    }
    return result;
}

ustring try_move_underscores_outside(const ustring & text) {
    static const regex p(R"(\\\((.+?)\\\))", pyre::DOTALL);
    static const regex us(R"((_{3,}))");
    static const regex us_full(R"(_{3,})");
    return p.sub_fn(text, [&](const pyre::match & m) {
        const ustring inner = m.str(1);
        const auto parts = us.split(inner);
        if (parts.size() == 1) return m.str(0);
        std::vector<ustring> result;
        for (const auto & part : parts) {
            if (us_full.fullmatch(part))
                result.push_back(part);
            else if (!pyre::strip(part).empty())
                result.push_back(U"\\(" + part + U"\\)");
        }
        return pyre::join(result, U" ");
    });
}

// ------------------------------------------------- table image tokens ------
ustring replace_table_image_tokens(const ustring & content, const std::vector<std::pair<ustring, ustring>> & map) {
    if (content.empty() || map.empty()) return content;
    ustring c = content;
    for (const auto & kv : map) {
        const ustring inner = kv.first.substr(1, kv.first.size() - 2);
        // Tokens are drawn from [A-Z2-8]; re.escape leaves them unchanged.
        for (char32_t ch : inner)
            if (!((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')))
                throw std::invalid_argument("unexpected table image token");
        const regex p(("\\[\\s*" + pyre::to_utf8(inner) + "\\s*\\]").c_str());
        const ustring repl = U"<img src=\"" + kv.second + U"\"/>";
        c = p.sub_fn(c, [&](const pyre::match &) { return repl; });
    }
    return c;
}

ustring replace_table_formula_delimiters(const ustring & content, bool enabled) {
    if (!enabled || content.empty()) return content;
    static const regex inl(R"(\\\((.+?)\\\))", pyre::DOTALL);
    static const regex blk(R"(\\\[(.+?)\\\])", pyre::DOTALL);
    static const regex eqt(R"((<eq>.*?</eq>))", pyre::DOTALL);
    auto wrap = [](const regex & p, const ustring & t) {
        return p.sub_fn(t, [](const pyre::match & m) { return U"<eq>" + pyre::strip(m.str(1)) + U"</eq>"; });
    };
    auto parts = eqt.split(content);
    for (auto & part : parts) {
        if (part.empty() || eqt.fullmatch(part)) continue;
        part = wrap(blk, wrap(inl, part));
    }
    return pyre::join(parts, U"");
}

// ------------------------------------------- image_analysis_postprocess ----
namespace {
ustring extract_tagged(const ustring & text, const char * st, const char * en) {
    const ustring s = u(st), e = u(en);
    size_t a = text.find(s);
    if (a == ustring::npos) return ustring();
    a += s.size();
    const size_t b = text.find(e, a);
    if (b == ustring::npos) return ustring();
    return pyre::strip(text.substr(a, b - a));
}
size_t count_md_cols(const ustring & line) {
    ustring s = pyre::strip(line);
    if (pyre::starts(s, U"|")) s = s.substr(1);
    if (pyre::ends(s, U"|")) s = s.substr(0, s.size() - 1);
    return pyre::split(s, U"|").size();
}
bool is_md_sep_line(const ustring & line) {
    static const regex p(R"(^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$)");
    pyre::match m;
    return p.match_at(line, 0, m);
}
bool is_md_row_candidate(const ustring & line) {
    const ustring s = pyre::strip(line);
    if (s.empty()) return false;
    return pyre::contains(s, U"|") && count_md_cols(s) >= 2;
}
std::vector<ustring> split_md_row(const ustring & line) {
    ustring s = pyre::strip(line);
    if (pyre::starts(s, U"|")) s = s.substr(1);
    if (pyre::ends(s, U"|")) s = s.substr(0, s.size() - 1);
    std::vector<ustring> cells;
    ustring cur;
    bool escaped = false;
    for (char32_t c : s) {
        if (escaped) {
            if (c == '|')
                cur.push_back('|');
            else {
                cur.push_back('\\');
                cur.push_back(c);
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '|') {
            cells.push_back(pyre::strip(cur));
            cur.clear();
        } else
            cur.push_back(c);
    }
    if (escaped) cur.push_back('\\');
    cells.push_back(pyre::strip(cur));
    return cells;
}
} // namespace

bool convert_markdown_table_to_html(const ustring & content, ustring & out) {
    if (content.empty() || pyre::strip(content).empty()) return false;
    std::vector<ustring> lines;
    for (auto & l : pyre::splitlines(pyre::strip(content))) lines.push_back(pyre::strip(l));
    if (lines.size() < 2) return false;
    for (auto & l : lines)
        if (l.empty()) return false;
    if (!is_md_row_candidate(lines[0]) || !is_md_sep_line(lines[1])) return false;
    const auto header = split_md_row(lines[0]);
    const auto sep = split_md_row(lines[1]);
    if (header.size() < 2 || sep.size() != header.size()) return false;
    std::vector<std::vector<ustring>> body;
    for (size_t i = 2; i < lines.size(); ++i) {
        if (!is_md_row_candidate(lines[i])) return false;
        auto row = split_md_row(lines[i]);
        if (row.size() != header.size()) return false;
        body.push_back(row);
    }
    out = U"<table><tr>";
    for (auto & c : header) out += U"<th>" + html_escape(c) + U"</th>";
    out += U"</tr>";
    for (auto & row : body) {
        out += U"<tr>";
        for (auto & c : row) out += U"<td>" + html_escape(c) + U"</td>";
        out += U"</tr>";
    }
    out += U"</table>";
    return true;
}

bool has_malformed_markdown_table(const ustring & content) {
    const auto lines = pyre::splitlines(content);
    std::vector<size_t> seps;
    size_t candidates = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (is_md_sep_line(lines[i])) seps.push_back(i);
        if (is_md_row_candidate(lines[i])) ++candidates;
    }
    bool found_valid = false;
    for (size_t sep : seps) {
        long h = static_cast<long>(sep) - 1;
        while (h >= 0 && pyre::strip(lines[static_cast<size_t>(h)]).empty()) --h;
        if (h < 0 || !is_md_row_candidate(lines[static_cast<size_t>(h)])) return true;
        const size_t hc = count_md_cols(lines[static_cast<size_t>(h)]), sc = count_md_cols(lines[sep]);
        if (hc < 2 || sc != hc) return true;
        for (size_t r = sep + 1; r < lines.size(); ++r) {
            if (pyre::strip(lines[r]).empty()) break;
            if (!is_md_row_candidate(lines[r])) break;
            if (count_md_cols(lines[r]) != hc) return true;
        }
        found_valid = true;
    }
    return candidates >= 2 && !found_valid;
}

ustring extract_and_validate_mermaid_strict(const ustring & content_in) {
    if (content_in.empty() || pyre::strip(content_in).empty()) return ustring();
    const ustring content = pyre::strip(content_in);
    static const regex mm(R"(```mermaid\s*(.*?)\s*```)", pyre::DOTALL);
    static const regex cb(R"(```\s*(.*?)\s*```)", pyre::DOTALL);
    pyre::match m;
    ustring code;
    if (mm.search(content, 0, m))
        code = pyre::strip(m.str(1));
    else if (cb.search(content, 0, m))
        code = pyre::strip(m.str(1));
    else
        code = pyre::strip(content);
    if (pyre::starts(pyre::lower(code), U"mermaid")) code = pyre::strip(code.size() > 7 ? code.substr(7) : ustring());
    static const regex g1(R"(^(grap|grapg|graphh)\b)", pyre::IGNORECASE);
    static const regex g2(R"(^(flowchar|flowchartt)\b)", pyre::IGNORECASE);
    code = g1.sub(code, "graph");
    code = g2.sub(code, "flowchart");
    const ustring bidir = U"__MERMAID_BIDIR_ARROW__";
    static const regex b1(R"(<\s*-\s*-\s*>)");
    static const regex a1(R"(-\s+->)"), a2(R"(--\s+>)"), a3(R"(-\s+-\s+>)"), a4(R"((?<![-=])\s+->\s+)");
    code = b1.sub_fn(code, [&](const pyre::match &) { return bidir; });
    code = a1.sub(code, "-->");
    code = a2.sub(code, "-->");
    code = a3.sub(code, "-->");
    code = a4.sub(code, " --> ");
    code = pyre::replace(code, bidir, U"<-->");
    static const regex ws(R"([ \t]+)");
    static const regex node(R"(([a-zA-Z0-9_\u4e00-\u9fa5\-]+(?:[ \t]+[a-zA-Z0-9_\u4e00-\u9fa5\-]+)*)[ \t]*\[(.*?)\])",
                            pyre::DOTALL);
    const ustring processed = node.sub_fn(code, [&](const pyre::match & mt) {
        const ustring raw_id = pyre::strip(mt.str(1));
        if (pyre::starts(pyre::lower(raw_id), U"subgraph ")) return mt.str(0);
        const ustring id = ws.sub(raw_id, "_");
        ustring raw = mt.str(2);
        if (pyre::starts(raw, U"\"[")) return mt.str(0);
        if (raw.empty()) return id + U"[]";
        if (pyre::starts(raw, U"\"") && pyre::ends(raw, U"\""))
            raw = raw.size() >= 2 ? raw.substr(1, raw.size() - 2) : ustring();
        ustring safe = pyre::replace(raw, U"\"", U"&quot;");
        safe = pyre::replace(safe, U"\n", U"<br>");
        return id + U"[\"" + safe + U"\"]";
    });
    return U"```mermaid\n" + processed + U"\n```";
}

// ---------------------------------------------------- mermaid_repetition ----
namespace {
ustring normalize_escaped_newlines(const ustring & text) {
    static const regex bs(R"(\\+n)");
    static const regex br(R"(<br\s*/?>)", pyre::IGNORECASE);
    return br.sub(bs.sub(text, "\n"), "\n");
}
std::pair<size_t, ustring> max_consecutive_repeat(const std::vector<ustring> & lines) {
    if (lines.empty()) return { 0, ustring() };
    size_t best = 1, cur = 1;
    ustring best_line = lines[0], cur_line = lines[0];
    for (size_t i = 1; i < lines.size(); ++i) {
        if (lines[i] == cur_line)
            ++cur;
        else {
            if (cur > best) best = cur, best_line = cur_line;
            cur_line = lines[i];
            cur = 1;
        }
    }
    if (cur > best) best = cur, best_line = cur_line;
    return { best, best_line };
}
size_t max_freq(const std::vector<ustring> & lines) {
    std::map<ustring, size_t> f;
    size_t best = 0;
    for (auto & l : lines) best = std::max(best, ++f[l]);
    return best;
}
ustring normalize_free_text_line(const ustring & line_in) {
    static const regex lead(R"re(^[\[\]\(\)"'\s>]+)re");
    static const regex trail(R"re([\[\]\(\)"'\s,.;:]+$)re");
    static const regex ws(R"(\s+)");
    ustring line = pyre::strip(line_in);
    line = lead.sub(line, "");
    line = trail.sub(line, "");
    return pyre::strip(ws.sub(line, " "));
}
bool is_mermaid_structure_line(const ustring & line) {
    const ustring s = pyre::strip(line);
    if (s.empty()) return true;
    const ustring l = pyre::lower(s);
    if (pyre::starts(l, U"graph ") || pyre::starts(l, U"flowchart ")) return true;
    if (pyre::starts(l, U"subgraph ") || l == U"end") return true;
    if (pyre::starts(l, U"direction ") || pyre::starts(l, U"%%")) return true;
    return pyre::contains(s, U"-->");
}
bool detect_raw_line_repetition(const ustring & content, size_t min_consecutive, size_t min_total) {
    std::vector<ustring> lines;
    for (auto & raw : pyre::splitlines(normalize_escaped_newlines(content))) {
        if (is_mermaid_structure_line(raw)) continue;
        ustring c = normalize_free_text_line(raw);
        if (!c.empty()) lines.push_back(c);
    }
    if (lines.empty()) return false;
    if (max_consecutive_repeat(lines).first >= min_consecutive) return true;
    return max_freq(lines) >= min_total;
}
bool node_id_issues(const ustring & id, size_t min_token_repeats, size_t max_len) {
    if (id.empty()) return false;
    if (id.size() > max_len) return true;
    static const regex pair_re(R"([A-Z][a-z]|[A-Z]{2})");
    static const regex alpha8(R"([A-Za-z]{8,})");
    std::map<ustring, size_t> freq;
    size_t best = 0;
    for (auto & t : pair_re.findall(id)) best = std::max(best, ++freq[t]);
    if (freq.empty())
        for (auto & chunk : alpha8.findall(id))
            for (size_t i = 0; i + 1 < chunk.size(); ++i) best = std::max(best, ++freq[chunk.substr(i, 2)]);
    return !freq.empty() && best >= min_token_repeats;
}
const regex & edge_line_re() {
    static const regex r(
        R"(^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->\s*(?:\|[^|\n]*\|\s*)?(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)(?:\s*(?:\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}))?\s*$)");
    return r;
}
bool same_label_edge_repetition(const ustring & content, size_t min_edges) {
    static const regex labeled(
        R"(^\s*(?P<src>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*-->\s*(?P<dst>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*\[\"(?P<label>(?:[^\"\\]|\\\\.)*)\"\]\s*$)");
    std::map<ustring, size_t> counter;
    bool has_cons = false;
    ustring cons;
    size_t run = 0, best = 0;
    for (auto & raw : pyre::splitlines(normalize_escaped_newlines(content))) {
        pyre::match m;
        if (!labeled.match_at(raw, 0, m)) {
            has_cons = false;
            run = 0;
            continue;
        }
        const ustring label = pyre::strip(m.str(3));
        if (label.empty()) continue;
        if (++counter[label] >= min_edges) return true;
        if (has_cons && label == cons)
            ++run;
        else {
            has_cons = true;
            cons = label;
            run = 1;
        }
        best = std::max(best, run);
    }
    return best >= min_edges;
}
ustring edge_endpoint_id(const ustring & segment) {
    ustring c = pyre::strip(segment);
    if (c.empty()) return ustring();
    static const regex lab(R"(^\|[^|\n]*\|\s*)");
    static const regex lead(R"(^([A-Za-z0-9_\u4e00-\u9fa5\-]+))");
    c = lab.sub(c, "");
    pyre::match m;
    if (lead.match_at(c, 0, m)) return m.str(1);
    return ustring();
}
bool malformed_edge_line_issues(const ustring & content, size_t min_rep, size_t max_len) {
    for (auto & raw : pyre::splitlines(normalize_escaped_newlines(content))) {
        const ustring s = pyre::strip(raw);
        if (!pyre::contains(s, U"-->")) continue;
        if (pyre::count(s, U"-->") >= 2) {
            for (auto & seg : pyre::split(s, U"-->"))
                if (node_id_issues(edge_endpoint_id(seg), min_rep, max_len)) return true;
            continue;
        }
        pyre::match m;
        if (!edge_line_re().match_at(raw, 0, m)) continue;
        if (node_id_issues(m.str(1), min_rep, max_len) || node_id_issues(m.str(2), min_rep, max_len)) return true;
    }
    return false;
}
} // namespace

bool detect_mermaid_repetition_issues(const ustring & content) {
    const size_t min_consecutive = 8, min_total = 20, max_label = 2000, min_same_edge = 8, max_id = 128;
    static const regex label_re(R"re((?P<node>[A-Za-z0-9_\u4e00-\u9fa5\-]+)\s*\["(?P<label>(?:[^"\\]|\\.)*)"\])re",
                                pyre::DOTALL);
    for (const auto & m : label_re.finditer(content)) {
        if (node_id_issues(m.str(1), min_consecutive, max_id)) return true;
        const ustring label = normalize_escaped_newlines(m.str(2));
        std::vector<ustring> lines;
        for (auto & l : pyre::splitlines(label))
            if (!pyre::strip(l).empty()) lines.push_back(pyre::strip(l));
        if (lines.empty()) continue;
        if (max_consecutive_repeat(lines).first >= min_consecutive) return true;
        if (max_freq(lines) >= min_total) return true;
        std::set<ustring> uniq(lines.begin(), lines.end());
        const double ratio = static_cast<double>(uniq.size()) / static_cast<double>(std::max<size_t>(lines.size(), 1));
        if (label.size() >= max_label && ratio <= 0.2) return true;
    }
    std::map<ustring, size_t> edges;
    for (auto & raw : pyre::splitlines(normalize_escaped_newlines(content))) {
        pyre::match m;
        if (!edge_line_re().match_at(raw, 0, m)) continue;
        if (++edges[m.str(1) + U"-->" + m.str(2)] >= min_same_edge) return true;
    }
    if (malformed_edge_line_issues(content, min_consecutive, max_id)) return true;
    if (same_label_edge_repetition(content, min_same_edge)) return true;
    return detect_raw_line_repetition(content, min_consecutive, min_total);
}

// ------------------------------------------------ mermaid_node_id_legalize --
namespace {
ustring sanitize_node_id(const ustring & raw) {
    static const regex alnum(R"([A-Za-z0-9]+)");
    static const regex other(R"([^A-Za-z0-9_])");
    const auto parts = alnum.findall(raw);
    ustring cand;
    if (!parts.empty()) {
        cand = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) {
            ustring p = parts[i];
            for (size_t k = 0; k < p.size(); ++k) {
                char32_t & c = p[k];
                if (k == 0 && c >= 'a' && c <= 'z') c -= 32;
                if (k > 0 && c >= 'A' && c <= 'Z') c += 32;
            }
            cand += p;
        }
    } else
        cand = other.sub(raw, "");
    if (cand.empty()) cand = U"node";
    if (pyre::isdecimal(cand[0]) && cand[0] < 128) cand = U"n" + cand;
    return cand;
}
ustring sanitize_edge_label(const ustring & raw) {
    static const regex spaces(R"( {2,})");
    ustring l = pyre::replace(pyre::replace(raw, U"(", U""), U")", U"");
    return pyre::strip(spaces.sub(l, " "));
}
bool looks_plain(const ustring & t) {
    if (t.empty()) return false;
    for (char32_t c : t)
        if (pyre::from_utf8("[](){}\"|").find(c) != ustring::npos) return false;
    return true;
}
struct id_state {
    std::vector<std::pair<ustring, ustring>> id_map; // insertion-ordered dict
    std::set<ustring> declared;
    const ustring * get(const ustring & k) const {
        for (auto & kv : id_map)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool has_value(const ustring & v) const {
        for (auto & kv : id_map)
            if (kv.second == v) return true;
        return false;
    }
};
ustring convert_endpoint(const ustring & token, id_state & st) {
    static const regex valid(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
    const ustring raw = pyre::strip(token);
    if (!looks_plain(raw)) return token;
    if (valid.fullmatch(raw)) return raw;
    ustring fixed;
    if (const ustring * f = st.get(raw))
        fixed = *f;
    else {
        fixed = sanitize_node_id(raw);
        const ustring base = fixed;
        long idx = 2;
        while (st.declared.count(fixed) || st.has_value(fixed)) {
            fixed = base + pyre::from_utf8(std::to_string(idx).c_str());
            ++idx;
        }
        st.id_map.push_back({ raw, fixed });
    }
    if (!st.declared.count(fixed)) {
        st.declared.insert(fixed);
        return fixed + U"[\"" + raw + U"\"]";
    }
    return fixed;
}
template <bool Chain> std::vector<ustring> split_top_level(const ustring & line, bool * trailing) {
    std::vector<ustring> out;
    size_t start = 0, i = 0;
    bool in_quote = false, escaped = false, in_pipe = false;
    long depth = 0;
    while (i < line.size()) {
        const char32_t ch = line[i];
        if (escaped) {
            escaped = false;
            ++i;
            continue;
        }
        if (in_quote) {
            if (ch == '\\')
                escaped = true;
            else if (ch == '"')
                in_quote = false;
            ++i;
            continue;
        }
        if (ch == '"') {
            in_quote = true;
            ++i;
            continue;
        }
        if (ch == '[' || ch == '(' || ch == '{') {
            ++depth;
            ++i;
            continue;
        }
        if (ch == ']' || ch == ')' || ch == '}') {
            if (depth > 0) --depth;
            ++i;
            continue;
        }
        if (ch == '|' && depth == 0) {
            in_pipe = !in_pipe;
            ++i;
            continue;
        }
        if (Chain) {
            if (pyre::starts(line, U"-->", i) && depth == 0 && !in_pipe) {
                out.push_back(line.substr(start, i - start));
                start = i + 3;
                i += 3;
                continue;
            }
        } else if (ch == ';' && depth == 0 && !in_pipe) {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
        ++i;
    }
    out.push_back(line.substr(start));
    if (!Chain) {
        const bool t = out.size() > 1 && pyre::strip(out.back()).empty();
        if (t) out.pop_back();
        *trailing = t;
    }
    return out;
}
ustring fix_edge_target(const ustring & segment, id_state & st) {
    static const regex lab(R"(^\|(?P<label>.*?)\|\s*(?P<dst>.+?)\s*$)");
    const ustring s = pyre::strip(segment);
    pyre::match m;
    if (lab.match_at(s, 0, m)) {
        const ustring label = sanitize_edge_label(m.str(1));
        return U"|" + label + U"| " + convert_endpoint(m.str(2), st);
    }
    return convert_endpoint(s, st);
}
ustring fix_edge_statement(const ustring & statement, id_state & st) {
    const ustring s = pyre::strip(statement);
    if (!pyre::contains(s, U"-->") || pyre::starts(s, U"%%")) return s;
    const auto parts = split_top_level<true>(s, nullptr);
    if (parts.size() < 2) return s;
    ustring fixed = convert_endpoint(pyre::strip(parts[0]), st);
    for (size_t i = 1; i < parts.size(); ++i) {
        const ustring t = fix_edge_target(parts[i], st);
        fixed += pyre::starts(t, U"|") ? U" -->" + t : U" --> " + t;
    }
    return fixed;
}
} // namespace

ustring try_fix_mermaid_node_id_legalization(const ustring & content) {
    static const regex decl(R"(^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[\[\(\{])");
    static const regex bidir(R"(<\s*-\s*-\s*>)");
    const auto lines = pyre::splitlines(content);
    id_state st;
    for (auto & line : lines) {
        pyre::match m;
        if (decl.match_at(pyre::strip(line), 0, m)) st.declared.insert(m.str(1));
    }
    std::vector<ustring> out;
    for (auto & line : lines) {
        const ustring s = pyre::strip(line);
        if (bidir.search(line)) {
            out.push_back(line);
            continue;
        }
        if (!pyre::contains(line, U"-->") || pyre::starts(s, U"%%")) {
            out.push_back(line);
            continue;
        }
        bool trailing = false;
        const auto statements = split_top_level<false>(line, &trailing);
        std::vector<ustring> fixed;
        for (auto & stmt : statements)
            if (!pyre::strip(stmt).empty()) fixed.push_back(fix_edge_statement(stmt, st));
        if (fixed.empty()) {
            out.push_back(line);
            continue;
        }
        ustring nl = U"  " + pyre::join(fixed, U"; ");
        if (trailing) nl += U";";
        out.push_back(nl);
    }
    ustring r = pyre::join(out, U"\n");
    if (pyre::ends(content, U"\n")) r += U"\n";
    return r;
}

// ----------------------------------------------- mermaid_invalid_edge_lines --
ustring try_fix_mermaid_invalid_edge_lines(const ustring & content) {
    static const regex valid(R"(-->\s*\|[^|\n]+\|\s*\S)");
    static const regex suspect(R"(-->\s*[^|\n]*\|\s*[^|\n]+$)");
    auto pipes = [](const ustring & line) {
        bool s1 = false, s2 = false, esc = false;
        size_t n = 0;
        for (char32_t c : line) {
            if (esc) {
                esc = false;
                continue;
            }
            if (c == '\\') {
                esc = true;
                continue;
            }
            if (c == '\'' && !s2) {
                s1 = !s1;
                continue;
            }
            if (c == '"' && !s1) {
                s2 = !s2;
                continue;
            }
            if (c == '|' && !s1 && !s2) ++n;
        }
        return n;
    };
    std::vector<ustring> kept;
    for (auto & line : pyre::splitlines(content)) {
        const ustring s = pyre::strip(line);
        bool invalid = false;
        if (!s.empty() && !pyre::starts(s, U"%%") && pyre::contains(line, U"-->") && pyre::contains(line, U"|") &&
            !valid.search(line))
            invalid = pipes(line) == 1 && suspect.search(s);
        if (!invalid) kept.push_back(line);
    }
    ustring r = pyre::join(kept, U"\n");
    if (pyre::ends(content, U"\n")) r += U"\n";
    return r;
}

// -------------------------------------------- mermaid_invalid_subgraph_lines --
ustring try_fix_mermaid_invalid_subgraph_lines(const ustring & content) {
    static const regex line_re(R"(^(\s*)subgraph\s+(.+?)\s*$)");
    static const regex valid(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
    static const regex explicit_re(R"(^([A-Za-z_][A-Za-z0-9_]*)\s*(?=[\[\(]))");
    static const regex nonid(R"([^A-Za-z0-9_]+)");
    static const regex unders(R"(_+)");
    std::set<ustring> seen;
    std::vector<ustring> out;
    for (auto & line : pyre::splitlines(content)) {
        pyre::match m;
        if (!line_re.match_at(line, 0, m)) {
            out.push_back(line);
            continue;
        }
        const ustring indent = m.str(1), body = m.str(2);
        pyre::match em;
        if (explicit_re.match_at(pyre::strip(body), 0, em) && !em.str(1).empty()) {
            seen.insert(em.str(1));
            out.push_back(line);
            continue;
        }
        const auto parts = pyre::split_ws(body, 1);
        if (parts.empty()) {
            out.push_back(line);
            continue;
        }
        pyre::match vm;
        if (valid.match_at(parts[0], 0, vm)) {
            seen.insert(parts[0]);
            out.push_back(line);
            continue;
        }
        ustring sid = pyre::strip_chars(unders.sub(nonid.sub(body, "_"), "_"), U"_");
        if (sid.empty()) sid = U"sg";
        if (sid[0] >= '0' && sid[0] <= '9') sid = U"sg_" + sid;
        ustring id = sid;
        if (seen.count(id)) {
            long idx = 2;
            while (seen.count(sid + U"_" + pyre::from_utf8(std::to_string(idx).c_str()))) ++idx;
            id = sid + U"_" + pyre::from_utf8(std::to_string(idx).c_str());
        }
        seen.insert(id);
        out.push_back(indent + U"subgraph " + id + U"[\"" + pyre::replace(body, U"\"", U"\\\"") + U"\"]");
    }
    ustring r = pyre::join(out, U"\n");
    if (pyre::ends(content, U"\n")) r += U"\n";
    return r;
}

// ------------------------------------------------ chart_invalid_repetition --
namespace {
bool chart_split_row(const ustring & line, std::vector<ustring> & cells) {
    ustring t = pyre::strip(line);
    if (t.empty() || !pyre::contains(t, U"|")) return false;
    if (pyre::starts(t, U"|")) t = t.substr(1);
    if (pyre::ends(t, U"|")) t = t.substr(0, t.size() - 1);
    cells.clear();
    ustring buf;
    bool esc = false, code = false;
    for (char32_t c : t) {
        if (esc) {
            buf.push_back(c);
            esc = false;
            continue;
        }
        if (c == '\\') {
            esc = true;
            buf.push_back(c);
            continue;
        }
        if (c == '`') {
            code = !code;
            buf.push_back(c);
            continue;
        }
        if (c == '|' && !code) {
            cells.push_back(pyre::strip(buf));
            buf.clear();
            continue;
        }
        buf.push_back(c);
    }
    cells.push_back(pyre::strip(buf));
    return true;
}
bool is_sep_row(const std::vector<ustring> & cells) {
    static const regex sep(R"(^:?-{3,}:?$)");
    if (cells.empty()) return false;
    for (auto & c : cells)
        if (!sep.fullmatch(pyre::replace(c, U" ", U""))) return false;
    return true;
}
bool chart_sep_line(const ustring & line) {
    std::vector<ustring> cells;
    if (!chart_split_row(line, cells)) return false;
    return is_sep_row(cells);
}
// Lowest i>=1 with (c+c)[i:].startswith(c) (KMP), i.e. (c+c).find(c, 1).
size_t rotation_period(const ustring & c) {
    const size_t n = c.size();
    std::vector<size_t> pi(n, 0);
    for (size_t i = 1; i < n; ++i) {
        size_t k = pi[i - 1];
        while (k > 0 && c[i] != c[k]) k = pi[k - 1];
        if (c[i] == c[k]) ++k;
        pi[i] = k;
    }
    const ustring t = c + c;
    size_t k = 0;
    for (size_t i = 1; i < t.size(); ++i) {
        while (k > 0 && t[i] != c[k]) k = pi[k - 1];
        if (t[i] == c[k]) ++k;
        if (k == n) return i + 1 - n;
    }
    return n;
}
ustring normalize_fragment(const ustring & f) {
    static const regex ws(R"(\s+)");
    return pyre::lower(pyre::strip(ws.sub(f, " ")));
}
bool cell_issue(const ustring & cell) {
    if (cell.size() > 300 || pyre::count(cell, U"::") > 20 || pyre::count(cell, U";") > 30) return true;
    static const regex frag(R"(\s*(?:::|;|,|\|)\s*)");
    std::map<ustring, size_t> counter;
    for (auto & f : frag.split(cell)) {
        const ustring n = normalize_fragment(f);
        if (n.size() < 3) continue;
        if (++counter[n] >= 8) return true;
    }
    return false;
}
bool cell_repetition_pattern(const ustring & content) {
    bool in_table = false, sep_seen = false;
    for (auto & line : pyre::splitlines(content)) {
        std::vector<ustring> row;
        if (!chart_split_row(line, row)) {
            in_table = sep_seen = false;
            continue;
        }
        if (!in_table) {
            in_table = true;
            sep_seen = false;
            continue;
        }
        if (!sep_seen) {
            if (is_sep_row(row)) sep_seen = true;
            continue;
        }
        for (auto & c : row) {
            const ustring t = pyre::strip(c);
            if (!t.empty() && cell_issue(t)) return true;
        }
    }
    return false;
}
bool noisy_row(const std::vector<ustring> & cells) {
    if (is_sep_row(cells)) return false;
    for (auto & c : cells) {
        const ustring t = pyre::strip(c);
        if (!t.empty() && cell_issue(t)) return true;
    }
    return false;
}
bool repair_row(const ustring & line, std::vector<ustring> & cells) {
    ustring n = pyre::strip(line);
    n = pyre::replace(n, U"\uff5c", U"|");
    n = pyre::replace(n, U"\u00a6", U"|");
    n = pyre::replace(n, U"\\|", U"|");
    if (!pyre::contains(n, U"|")) return false;
    if (!chart_split_row(n, cells) || cells.size() < 2) return false;
    return true;
}
ustring clean_cell(const ustring & cell) {
    static const regex ws(R"(\s+)");
    ustring c = pyre::strip(ws.sub(pyre::replace(cell, U"\t", U" "), " "));
    if (pyre::count(c, U"`") % 2 == 1) c += U"`";
    return pyre::replace(c, U"|", U"\\|");
}
ustring fmt_row(const std::vector<ustring> & cells) {
    std::vector<ustring> c;
    for (auto & x : cells) c.push_back(clean_cell(x));
    return U"| " + pyre::join(c, U" | ") + U" |";
}
ustring fmt_sep(size_t n) {
    std::vector<ustring> c(n, U"---");
    return U"| " + pyre::join(c, U" | ") + U" |";
}
bool pipe_bounds(const ustring & l) {
    const ustring s = pyre::strip(l);
    return pyre::starts(s, U"|") && pyre::ends(s, U"|");
}
bool validate_repaired(const ustring & table) {
    const auto lines = pyre::splitlines(pyre::strip(table));
    if (lines.size() < 3) return false;
    std::vector<ustring> h, s;
    if (!chart_split_row(lines[0], h) || !chart_split_row(lines[1], s) || !is_sep_row(s)) return false;
    if (!pipe_bounds(lines[0]) || !pipe_bounds(lines[1])) return false;
    if (h.size() != s.size()) return false;
    for (size_t i = 2; i < lines.size(); ++i) {
        std::vector<ustring> r;
        if (!chart_split_row(lines[i], r) || !pipe_bounds(lines[i]) || r.size() != h.size()) return false;
    }
    return true;
}
} // namespace

bool validate_content_repetition(const ustring & content) {
    if (pyre::contains(content, U"\\|")) return true;
    static const regex nl5(R"(\n{5,})");
    if (nl5.search(content)) return true;
    {
        static const regex ws(R"(\s+)");
        const ustring compact = pyre::strip(ws.sub(content, " "));
        const size_t size = compact.size();
        if (size >= 80) {
            const size_t period = rotation_period(compact);
            if (period > 0 && period < size && size % period == 0 && size / period >= 2) return true;
        }
    }
    static const regex run(R"(([A-Za-z0-9])\1{19,})");
    if (run.search(content)) return true;
    if (cell_repetition_pattern(content)) return true;
    std::vector<ustring> lines;
    for (auto & l : pyre::splitlines(pyre::strip_chars(content, U"\n"))) lines.push_back(pyre::strip(l));
    for (size_t i = 0; i < lines.size();) {
        if (lines[i].empty()) {
            ++i;
            continue;
        }
        size_t j = i + 1;
        while (j < lines.size() && lines[j] == lines[i]) ++j;
        const size_t n = j - i;
        if ((n >= 2 && lines[i].size() >= 30) || n >= 3) return true;
        i = j;
    }
    {
        std::map<ustring, size_t> pos;
        for (auto & l : lines)
            if (l.size() >= 30 && !chart_sep_line(l))
                if (++pos[l] >= 4) return true;
    }
    std::vector<ustring> kept;
    for (auto & l : pyre::splitlines(content))
        if (!chart_sep_line(l)) kept.push_back(l);
    static const regex tok(R"([A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|[^\s])");
    const auto tokens = tok.findall(pyre::join(kept, U"\n"));
    for (size_t i = 0; i < tokens.size();) {
        const ustring low = pyre::lower(tokens[i]);
        size_t j = i + 1;
        while (j < tokens.size() && pyre::lower(tokens[j]) == low) ++j;
        if (j - i >= 8) return true;
        i = j;
    }
    return false;
}

ustring extract_largest_legal_markdown_table(const ustring & content) {
    // parsed_table_blocks
    std::vector<std::vector<std::vector<ustring>>> blocks;
    std::vector<std::vector<ustring>> cur;
    bool sep_seen = false;
    for (auto & line : pyre::splitlines(content)) {
        std::vector<ustring> cells;
        if (!repair_row(line, cells)) {
            if (!cur.empty()) {
                blocks.push_back(cur);
                cur.clear();
            }
            sep_seen = false;
            continue;
        }
        if (cur.empty()) {
            cur.push_back(cells);
            sep_seen = false;
            continue;
        }
        if (!sep_seen) {
            cur.push_back(cells);
            if (is_sep_row(cells)) sep_seen = true;
            continue;
        }
        if (noisy_row(cells)) continue;
        cur.push_back(cells);
    }
    if (!cur.empty()) blocks.push_back(cur);
    std::vector<ustring> candidates;
    for (auto & rows : blocks) {
        for (size_t idx = 0; idx < rows.size(); ++idx) {
            if (!is_sep_row(rows[idx])) continue;
            // build_table_candidate
            if (idx == 0) continue;
            const auto & header = rows[idx - 1];
            if (header.size() < 2 || is_sep_row(header)) continue;
            std::vector<std::vector<ustring>> data;
            for (size_t r = idx + 1; r < rows.size(); ++r) {
                if (is_sep_row(rows[r]) || rows[r].size() != header.size()) break;
                data.push_back(rows[r]);
            }
            if (data.empty()) continue;
            std::vector<ustring> lines = { fmt_row(header), fmt_sep(header.size()) };
            for (auto & d : data) lines.push_back(fmt_row(d));
            const ustring cand = pyre::join(lines, U"\n");
            if (validate_repaired(cand)) candidates.push_back(cand);
        }
        // build_fallback_table_candidates
        size_t index = 0;
        while (index < rows.size()) {
            if (is_sep_row(rows[index]) || rows[index].size() < 2) {
                ++index;
                continue;
            }
            const size_t cols = rows[index].size();
            std::vector<std::vector<ustring>> run = { rows[index] };
            size_t next = index + 1;
            while (next < rows.size()) {
                if (is_sep_row(rows[next]) || rows[next].size() != cols) break;
                run.push_back(rows[next]);
                ++next;
            }
            if (run.size() >= 2) {
                std::vector<ustring> lines = { fmt_row(run[0]), fmt_sep(cols) };
                for (size_t k = 1; k < run.size(); ++k) lines.push_back(fmt_row(run[k]));
                const ustring cand = pyre::join(lines, U"\n");
                if (validate_repaired(cand)) candidates.push_back(cand);
            }
            index = std::max(next, index + 1);
        }
    }
    if (candidates.empty()) return ustring();
    auto score = [](const ustring & t) {
        const auto rows = pyre::splitlines(t);
        std::vector<ustring> first;
        if (!rows.empty()) chart_split_row(rows[0], first);
        return std::make_tuple(rows.size(), first.size(), t.size());
    };
    size_t best = 0;
    auto best_score = score(candidates[0]);
    for (size_t i = 1; i < candidates.size(); ++i) {
        const auto s = score(candidates[i]);
        if (s > best_score) best = i, best_score = s;
    }
    return candidates[best];
}

ustring try_fix_chart_invalid_repetition(const ustring & content) {
    if (!validate_content_repetition(content)) return content;
    const ustring fixed = extract_largest_legal_markdown_table(content);
    return fixed.empty() ? content : fixed;
}

image_analysis process_image_or_chart(const ustring & content) {
    image_analysis v;
    v.cls = extract_tagged(content, "<|class_start|>", "<|class_end|>");
    v.sub_class = extract_tagged(content, "<|sub_class_start|>", "<|sub_class_end|>");
    v.caption = extract_tagged(content, "<|caption_start|>", "<|caption_end|>");
    v.content = extract_tagged(content, "<|content_start|>", "<|content_end|>");
    v.cls = pyre::lower(pyre::strip(v.cls));
    v.sub_class = pyre::lower(pyre::strip(v.sub_class));
    ustring c = v.content;
    if (c.empty() && pyre::count(content, U"<|content_start|>") == 1 && pyre::count(content, U"<|content_end|>") == 0)
        c = pyre::split(content, U"<|content_start|>").back();
    if (v.cls == U"chemical") {
        c.clear();
    } else if (v.cls == U"flowchart") {
        try {
            if (detect_mermaid_repetition_issues(c))
                c.clear();
            else {
                c = try_fix_mermaid_node_id_legalization(c);
                c = try_fix_mermaid_invalid_edge_lines(c);
                c = try_fix_mermaid_invalid_subgraph_lines(c);
            }
        } catch (const std::exception &) {
        }
        c = extract_and_validate_mermaid_strict(c);
    } else if (v.cls == U"chart") {
        try {
            c = try_fix_chart_invalid_repetition(c);
        } catch (const std::exception &) {
        }
        if (!c.empty() && has_malformed_markdown_table(c)) c.clear();
    }
    v.content = c;
    return v;
}

ustring convert_pure_table_content_to_html(const ustring & content) {
    if (content.empty() || pyre::strip(content).empty()) return ustring();
    const ustring stripped = pyre::strip(content);
    const ustring low = pyre::lower(stripped);
    if (pyre::starts(low, U"<table") && pyre::ends(low, U"</table>")) return stripped;
    ustring md;
    if (convert_markdown_table_to_html(content, md)) return md;
    for (const char32_t * t : { U"<nl>", U"<fcel>", U"<ecel>", U"<lcel>", U"<ucel>", U"<xcel>" })
        if (pyre::contains(content, t)) {
            try {
                const ustring html = convert_otsl_to_html(content);
                if (html.empty() || pyre::strip(html).empty()) return ustring();
                return html;
            } catch (const std::exception &) {
                return ustring();
            }
        }
    return ustring();
}

// ---------------------------------------------------------- equation_block --
ustring combined_equations(std::vector<ustring> contents) {
    static const regex tag_re(R"(\\tag\s*\{[^}]*\})");
    static const regex tag_sub(R"(\\tag\s*\{([^}]*)\})");
    size_t tags = 0;
    for (auto & c : contents) tags += tag_re.findall(c).size();
    if (tags > 1)
        for (auto & c : contents) c = tag_sub.sub(c, "(\\1)");
    ustring out = U"\\begin{array}{l} ";
    for (auto & c : contents) out += c + U" \\\\ ";
    return out + U"\\end{array}";
}

namespace {
double eq_cover_ratio(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    const double xa = std::max(a[0], b[0]), ya = std::max(a[1], b[1]);
    const double xb = std::min(a[2], b[2]), yb = std::min(a[3], b[3]);
    // Python: max(0, xB - xA) * max(0, yB - yA)
    const double w = (xb - xa) > 0 ? (xb - xa) : 0.0, h = (yb - ya) > 0 ? (yb - ya) : 0.0;
    const double inter = w * h;
    const double area_b = (b[2] - b[0]) * (b[3] - b[1]);
    if (area_b == 0) return 0.0;
    return inter / area_b;
}

block_list do_handle_equation_block(const block_list & blocks) {
    std::vector<size_t> sem, span;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]->type == "equation_block")
            sem.push_back(i);
        else if (blocks[i]->type == "equation")
            span.push_back(i);
    }
    std::map<size_t, std::vector<size_t>> sem_spans;
    for (size_t s : sem) {
        std::vector<size_t> covered;
        for (size_t p : span)
            if (eq_cover_ratio(blocks[s]->bbox, blocks[p]->bbox) > 0.9) covered.push_back(p);
        if (covered.size() > 1) sem_spans[s] = covered;
    }
    block_list out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        bool member = false;
        for (auto & kv : sem_spans)
            if (std::find(kv.second.begin(), kv.second.end(), i) != kv.second.end()) member = true;
        if (member) continue;
        auto it = sem_spans.find(i);
        if (it != sem_spans.end()) {
            std::vector<ustring> contents;
            for (size_t p : it->second) {
                // Upstream concatenates/regex-scans None and raises TypeError.
                if (!blocks[p]->has_content) throw python_type_error("equation span content is None");
                contents.push_back(blocks[p]->content);
            }
            auto b = make_block("equation", blocks[i]->bbox, blocks[i]->angle);
            b->set_content(combined_equations(contents));
            out.push_back(b);
            continue;
        }
        if (blocks[i]->type == "equation_block") continue;
        out.push_back(blocks[i]);
    }
    return out;
}

bool paratext(const std::string & t) {
    return t == "header" || t == "footer" || t == "page_number" || t == "aside_text" || t == "page_footnote" ||
           t == "unknown";
}

void simple_process(const block_list & blocks, bool eq_wrap) {
    for (auto & b : blocks) {
        if (b->type == "table" && b->truthy()) {
            ustring content = b->content;
            try {
                content = convert_otsl_to_html(content);
            } catch (const std::exception &) {
            }
            content = replace_table_image_tokens(
                content, b->has_token_map ? b->table_image_tokens : std::vector<std::pair<ustring, ustring>>{});
            b->set_content(replace_table_formula_delimiters(content, eq_wrap));
        }
        if ((b->type == "image" || b->type == "chart") && b->truthy()) {
            try {
                const image_analysis r = process_image_or_chart(b->content);
                if (r.cls == U"pure_table") {
                    b->set_type("table");
                    const ustring html = convert_pure_table_content_to_html(r.content);
                    b->set_content(html.empty() ? ustring() : replace_table_formula_delimiters(html, eq_wrap));
                } else if (r.cls == U"pure_formula") {
                    b->set_type("equation");
                    b->set_content(r.content);
                } else if (r.cls == U"chart") {
                    b->set_type("chart");
                    b->sub_type = r.sub_class;
                    b->add_key("sub_type");
                    b->set_content(r.content);
                } else {
                    b->set_type("image");
                    b->sub_type = r.cls;
                    b->add_key("sub_type");
                    if (r.cls == U"natural_image" || r.content.empty())
                        b->set_content(r.caption);
                    else
                        b->set_content(r.content);
                }
            } catch (const std::exception &) {
                b->set_none();
            }
        }
    }
}

block_list cleanup_metadata(const block_list & blocks) {
    for (auto & b : blocks) {
        b->has_token_map = false;
        b->table_image_tokens.clear();
        b->absorbed = false;
    }
    return blocks;
}

block_list post_process(const block_list & input, const options & o) {
    simple_process(input, o.enable_table_formula_eq_wrap);
    for (auto & b : input)
        if (b->type == "list_item") b->set_type("text");
    block_list blocks;
    for (auto & b : input) {
        if (!b->has_content || pyre::strip(b->content) != U"[Non-Text]") {
            blocks.push_back(b);
            continue;
        }
        if (b->type == "header" || b->type == "footer") {
            b->set_content(ustring());
            blocks.push_back(b);
            continue;
        }
        static const std::set<std::string> protect = { "image", "chart",       "table",         "equation",
                                                       "list",  "image_block", "equation_block" };
        if (protect.count(b->type)) blocks.push_back(b);
    }
    if (o.simple_post_process) {
        block_list out;
        for (auto & b : blocks)
            if (!(b->type == "image" && b->absorbed)) out.push_back(b);
        return cleanup_metadata(out);
    }
    for (auto & b : blocks) {
        if (b->type == "equation" && b->truthy()) {
            try {
                b->set_content(process_equation(b->content));
            } catch (const std::exception &) {
            }
        } else if (b->type == "text" && b->truthy()) {
            try {
                b->set_content(try_convert_display_to_inline(b->content));
                b->set_content(try_fix_macro_spacing_in_markdown(b->content));
                b->set_content(try_move_underscores_outside(b->content));
            } catch (const std::exception &) {
            }
        }
    }
    if (o.handle_equation_block) blocks = do_handle_equation_block(blocks);
    for (auto & b : blocks)
        if (b->type == "equation" && b->truthy()) b->set_content(add_equation_brackets(b->content));
    block_list out;
    for (auto & b : blocks) {
        if (b->type == "equation_block") continue;
        if (b->type == "image" && b->absorbed) continue;
        if (o.abandon_list && b->type == "list") continue;
        if (o.abandon_paratext && paratext(b->type)) continue;
        out.push_back(b);
    }
    return cleanup_metadata(out);
}
} // namespace

block_list post_process_helper(const block_list & blocks, const options & opts) {
    try {
        return post_process(blocks, opts);
    } catch (const python_type_error &) {
        block_list clean;
        for (auto & b : blocks)
            if (!(b->type == "image" && b->absorbed)) clean.push_back(b);
        return cleanup_metadata(clean);
    }
}

// ------------------------------------------------------------ rendering ----
std::string python_float_repr(double x) {
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
    if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
    char buf[64];
    int prec = 1;
    for (; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, x);
        if (std::strtod(buf, nullptr) == x) break;
    }
    // buf = [-]d.ddde[+-]XX
    std::string s(buf);
    const bool neg = s[0] == '-';
    if (neg) s = s.substr(1);
    const size_t e = s.find('e');
    std::string digits = s.substr(0, 1) + (e > 1 ? s.substr(2, e - 2) : "");
    const int exp = std::atoi(s.c_str() + e + 1);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    std::string out;
    if (exp < -4 || exp >= 16) {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        char eb[16];
        std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
        out += eb;
    } else if (exp < 0) {
        out = "0." + std::string(static_cast<size_t>(-exp - 1), '0') + digits;
    } else {
        const size_t int_len = static_cast<size_t>(exp) + 1;
        if (digits.size() <= int_len)
            out = digits + std::string(int_len - digits.size(), '0') + ".0";
        else
            out = digits.substr(0, int_len) + "." + digits.substr(int_len);
    }
    return neg ? "-" + out : out;
}

std::string json_quote(const ustring & value) {
    static const char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (char32_t c : value) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
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
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (c < 0x20) {
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 15];
            } else
                pyre::append_utf8(out, c);
        }
    }
    return out + "\"";
}

std::string blocks_json(const block_list & blocks) {
    std::string out = "[";
    for (size_t i = 0; i < blocks.size(); ++i) {
        const block & b = *blocks[i];
        if (i) out += ",";
        out += "{\"type\":" + json_quote(pyre::from_utf8(b.type)) + ",\"bbox\":[";
        for (int k = 0; k < 4; ++k) out += (k ? "," : "") + python_float_repr(b.bbox[static_cast<size_t>(k)]);
        out += "],\"angle\":" + (b.angle < 0 ? std::string("null") : std::to_string(b.angle));
        out += ",\"content\":" + (b.has_content ? json_quote(b.content) : std::string("null"));
        for (const auto & k : b.extra_keys) {
            if (k == "merge_prev")
                out += std::string(",\"merge_prev\":") + (b.merge_prev ? "true" : "false");
            else if (k == "sub_type")
                out += ",\"sub_type\":" + json_quote(b.sub_type);
        }
        out += "}";
    }
    return out + "]";
}

std::string json2md(const block_list & blocks) {
    std::vector<ustring> parts;
    long last = -1;
    static const regex cjk(R"([\u4e00-\u9fff\u3400-\u4dbf])");
    for (const auto & bp : blocks) {
        const block & b = *bp;
        if (!b.truthy()) continue;
        ustring content = b.content;
        const bool mp = b.has_key("merge_prev") && b.merge_prev;
        if (mp && last >= 0) {
            if (!cjk.search(content)) content = U" " + content;
            parts[static_cast<size_t>(last)] += content;
        } else {
            parts.push_back(content);
            if (b.type == "text") last = static_cast<long>(parts.size()) - 1;
        }
    }
    return pyre::to_utf8(pyre::join(parts, U"\n\n"));
}

} // namespace mineru_pp
