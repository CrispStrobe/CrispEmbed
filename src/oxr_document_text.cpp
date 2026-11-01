// oxr_document_text.cpp — OXR content postprocessing (code-point exact).
//
// Translated from SoMarkAI/OXR @ c1a1dcb3: oxr/pipeline/postprocess/
// {text,formula,code,table,chemical_structure}_postprocess.py and
// oxr/utils/table.py (strip_table_code_fence). Apache-2.0 plus the OXR
// additional terms (licenses/oxr-LICENSE). Changes: Python regular
// expressions are expanded into explicit scanners with identical
// backtracking results; strings are processed as UTF-32 code points.
#include "oxr_document_internal.h"

#include <algorithm>

namespace oxr_document {
namespace text {

using namespace py_str;

namespace {
const char32_t BS = U'\\';

bool in_text_commands(const u32 & name) {
    static const char * names[] = { "text", "textrm", "textnormal", "textbf", "textit", "mbox", "hbox", "verb" };
    for (const char * n : names)
        if (name == U(n)) return true;
    return false;
}
} // namespace

// _LATEX_COMMAND = re.compile(r"\\(?:[A-Za-z]+|.)", re.DOTALL); returns end or npos.
size_t latex_command(const u32 & s, size_t i) {
    if (i >= s.size() || s[i] != BS || i + 1 >= s.size()) return u32::npos;
    if (is_ascii_alpha(s[i + 1])) {
        size_t j = i + 1;
        while (j < s.size() && is_ascii_alpha(s[j])) ++j;
        return j;
    }
    return i + 2;
}

bool is_escaped(const u32 & s, size_t index) {
    size_t count = 0;
    size_t cursor = index;
    while (cursor > 0 && s[cursor - 1] == BS) {
        ++count;
        --cursor;
    }
    return count % 2 == 1;
}

// re.search(r"```<header>\n(.*?)\n```", DOTALL). header_mode: 0 = "latex"/"html"
// literal (case-insensitive when ci), 1 = (?:[^\n`]*), 2 = (?:.*?).
std::optional<u32> fenced_search(const u32 & s, int header_mode, const char * literal, bool ci) {
    const u32 fence = U("```");
    const u32 close = U("\n```");
    for (size_t i = 0; i + 3 <= s.size(); ++i) {
        if (s.compare(i, 3, fence) != 0) continue;
        size_t h = i + 3;
        if (header_mode == 0) {
            const u32 lit = U(literal);
            if (h + lit.size() > s.size()) continue;
            bool ok = true;
            for (size_t k = 0; k < lit.size(); ++k) {
                char32_t c = s[h + k];
                if (ci && c >= U'A' && c <= U'Z') c = c - U'A' + U'a';
                if (c != lit[k]) ok = false;
            }
            if (!ok) continue;
            h += lit.size();
            if (h >= s.size() || s[h] != U'\n') continue;
        } else if (header_mode == 1) {
            while (h < s.size() && s[h] != U'\n' && s[h] != U'`') ++h;
            if (h >= s.size() || s[h] != U'\n') continue;
        } else {
            while (h < s.size() && s[h] != U'\n') ++h; // lazy header stops at the first newline
            if (h >= s.size()) continue;
        }
        const size_t body = h + 1;
        const size_t end = s.find(close, body);
        if (end == u32::npos) continue;
        return s.substr(body, end - body);
    }
    return std::nullopt;
}

// re.match(r"^OPEN(.*?|.*)CLOSE$", text, DOTALL): Python '$' also matches
// before one trailing newline.
std::optional<u32> anchored_wrapper(const u32 & s, const u32 & open, const u32 & close, bool lazy) {
    if (!startswith(s, open)) return std::nullopt;
    const size_t lo = open.size();
    if (s.size() < lo + close.size()) return std::nullopt;
    auto ok = [&](size_t e) {
        if (s.compare(e, close.size(), close) != 0) return false;
        const size_t after = e + close.size();
        return after == s.size() || (after + 1 == s.size() && s[after] == U'\n');
    };
    const size_t hi = s.size() - close.size();
    if (lazy) {
        for (size_t e = lo; e <= hi; ++e)
            if (ok(e)) return s.substr(lo, e - lo);
    } else {
        for (size_t e = hi + 1; e-- > lo;)
            if (ok(e)) return s.substr(lo, e - lo);
    }
    return std::nullopt;
}

u32 space_latex_row_breaks(const u32 & latex) {
    u32 result;
    size_t index = 0;
    while (index < latex.size()) {
        if (latex[index] == U'%') {
            size_t end = latex.find(U'\n', index);
            end = end == u32::npos ? latex.size() : end;
            result += latex.substr(index, end - index);
            index = end;
            continue;
        }
        const size_t cmd_end = latex_command(latex, index);
        if (cmd_end == u32::npos) {
            result.push_back(latex[index]);
            index += 1;
            continue;
        }
        const u32 command = latex.substr(index, cmd_end - index);
        size_t end = cmd_end;
        if (in_text_commands(command.substr(1))) {
            size_t cursor = end;
            while (cursor < latex.size() && isspace(latex[cursor])) ++cursor;
            if (cursor < latex.size() && latex[cursor] == U'{') {
                int depth = 1;
                cursor += 1;
                while (cursor < latex.size() && depth) {
                    const size_t esc = latex_command(latex, cursor);
                    if (esc != u32::npos) {
                        cursor = esc;
                        continue;
                    }
                    if (latex[cursor] == U'{')
                        depth += 1;
                    else if (latex[cursor] == U'}')
                        depth -= 1;
                    cursor += 1;
                }
                if (depth) return latex;
                end = cursor;
            }
        }
        result += latex.substr(index, end - index);
        if (command == U("\\\\") && end < latex.size() &&
            (latex[end] == U'(' || latex[end] == U')' || latex[end] == U'{' || latex[end] == U'}'))
            result.push_back(U' ');
        index = end;
    }
    return result;
}

namespace {
// _MARKDOWN_CODE.match: (?P<ticks>`+)[\s\S]*?(?P=ticks)|~~~[\s\S]*?~~~
size_t markdown_code(const u32 & s, size_t i) {
    if (i < s.size() && s[i] == U'`') {
        size_t run = 0;
        while (i + run < s.size() && s[i + run] == U'`') ++run;
        for (size_t n = run; n >= 1; --n) {
            const u32 ticks(n, U'`');
            const size_t p = s.find(ticks, i + n);
            if (p != u32::npos) return p + n;
        }
        return u32::npos;
    }
    if (startswith(s, U("~~~"), i)) {
        const size_t p = s.find(U("~~~"), i + 3);
        if (p != u32::npos) return p + 3;
    }
    return u32::npos;
}

char32_t fold_ascii(char32_t c) {
    return (c >= U'A' && c <= U'Z') ? c - U'A' + U'a' : c;
}
// [A-Za-z] under re.IGNORECASE also matches U+0130, U+0131, U+017F, U+212A.
bool ci_ascii_letter(char32_t c) {
    return is_ascii_alpha(c) || c == 0x130 || c == 0x131 || c == 0x17F || c == 0x212A;
}
bool ci_lit(const u32 & s, size_t i, const char * lit) {
    size_t k = 0;
    for (; lit[k]; ++k) {
        if (i + k >= s.size()) return false;
        if (fold_ascii(s[i + k]) != char32_t((unsigned char)lit[k])) return false;
    }
    return true;
}
// &(?:lt|#0*60|#x0*3c);  (or the gt/62/3e closing form), IGNORECASE; returns end.
size_t entity(const u32 & s, size_t i, bool open) {
    if (i >= s.size() || s[i] != U'&') return u32::npos;
    size_t j = i + 1;
    if (ci_lit(s, j, open ? "lt" : "gt")) {
        j += 2;
    } else if (j < s.size() && s[j] == U'#') {
        ++j;
        if (j < s.size() && fold_ascii(s[j]) == U'x') {
            ++j;
            while (j < s.size() && s[j] == U'0') ++j;
            if (!ci_lit(s, j, open ? "3c" : "3e")) return u32::npos;
            j += 2;
        } else {
            while (j < s.size() && s[j] == U'0') ++j;
            if (!ci_lit(s, j, open ? "60" : "62")) return u32::npos;
            j += 2;
        }
    } else {
        return u32::npos;
    }
    if (j >= s.size() || s[j] != U';') return u32::npos;
    return j + 1;
}
size_t html_comment(const u32 & s, size_t i) {
    if (!startswith(s, U("<!--"), i)) return u32::npos;
    const size_t p = s.find(U("-->"), i + 4);
    return p == u32::npos ? s.size() : p + 3;
}
size_t html_tag(const u32 & s, size_t i) {
    if (i >= s.size() || s[i] != U'<') return u32::npos;
    // </?[A-Za-z][A-Za-z0-9-]*(?:\s[^>]*)?/?>
    {
        size_t j = i + 1;
        if (j < s.size() && s[j] == U'/' && j + 1 < s.size() && is_ascii_alpha(s[j + 1])) ++j;
        if (j < s.size() && is_ascii_alpha(s[j])) {
            ++j;
            while (j < s.size() && (is_ascii_alpha(s[j]) || is_ascii_digit(s[j]) || s[j] == U'-')) ++j;
            if (j < s.size() && isspace(s[j])) {
                const size_t gt = s.find(U'>', j + 1);
                if (gt != u32::npos) return gt + 1;
            } else if (j < s.size() && s[j] == U'>') {
                return j + 1;
            } else if (j + 1 < s.size() && s[j] == U'/' && s[j + 1] == U'>') {
                return j + 2;
            }
        }
    }
    // <![A-Z][^>]*(?:>|\Z)
    if (i + 2 < s.size() && s[i + 1] == U'!' && s[i + 2] >= U'A' && s[i + 2] <= U'Z') {
        const size_t gt = s.find(U'>', i + 3);
        return gt == u32::npos ? s.size() : gt + 1;
    }
    // <\?[^>]*(?:>|\Z)
    if (i + 1 < s.size() && s[i + 1] == U'?') {
        const size_t gt = s.find(U'>', i + 2);
        return gt == u32::npos ? s.size() : gt + 1;
    }
    return u32::npos;
}
size_t html_entity_tag(const u32 & s, size_t i) {
    size_t j = entity(s, i, true);
    if (j == u32::npos) return j;
    if (j < s.size() && s[j] == U'/' && j + 1 < s.size() && ci_ascii_letter(s[j + 1])) ++j;
    if (j >= s.size() || !ci_ascii_letter(s[j])) return u32::npos;
    ++j;
    for (size_t q = j; q < s.size(); ++q) {
        const size_t e = entity(s, q, false);
        if (e != u32::npos) return e;
    }
    return u32::npos;
}
size_t protected_markup(const u32 & s, size_t i) {
    size_t e = html_comment(s, i);
    if (e != u32::npos) return e;
    e = html_tag(s, i);
    if (e != u32::npos) return e;
    return html_entity_tag(s, i);
}
bool valid_math_open(const u32 & s, size_t i, int width) {
    if (width == 2) return !startswith(s, U("$$$"), i);
    const size_t end = i + 1;
    return end < s.size() && !isspace(s[end]) && s[end] != U'$';
}
bool valid_math_close(const u32 & s, size_t i, int width) {
    if (width == 2) return i > 0 && !startswith(s, U("$$$"), i);
    return i > 0 && !isspace(s[i - 1]) && !startswith(s, U("$$"), i);
}
} // namespace

u32 space_markdown_math(const u32 & md) {
    u32 result;
    size_t index = 0;
    while (index < md.size()) {
        size_t e = markdown_code(md, index);
        if (e != u32::npos) {
            result += md.substr(index, e - index);
            index = e;
            continue;
        }
        e = protected_markup(md, index);
        if (e != u32::npos) {
            result += md.substr(index, e - index);
            index = e;
            continue;
        }
        if (md[index] != U'$' || is_escaped(md, index)) {
            result.push_back(md[index]);
            index += 1;
            continue;
        }
        const int width = startswith(md, U("$$"), index) ? 2 : 1;
        if (!valid_math_open(md, index, width)) {
            result += md.substr(index, size_t(width));
            index += size_t(width);
            continue;
        }
        size_t cursor = index + size_t(width);
        std::optional<size_t> end;
        while (cursor < md.size()) {
            if (markdown_code(md, cursor) != u32::npos || protected_markup(md, cursor) != u32::npos) break;
            if (md[cursor] == U'$' && !is_escaped(md, cursor)) {
                const int found = startswith(md, U("$$"), cursor) ? 2 : 1;
                if (found == width && valid_math_close(md, cursor, width)) end = cursor;
                break;
            }
            cursor += 1;
        }
        if (!end) {
            result += md.substr(index, size_t(width));
            index += size_t(width);
            continue;
        }
        const u32 delim(size_t(width), U'$');
        result +=
            delim + space_latex_row_breaks(md.substr(index + size_t(width), *end - index - size_t(width))) + delim;
        index = *end + size_t(width);
    }
    return result;
}

namespace {
std::optional<std::pair<u32, size_t>> parse_braced_group(const u32 & s, size_t open) {
    if (open >= s.size() || s[open] != U'{') return std::nullopt;
    int depth = 1;
    size_t index = open + 1;
    while (index < s.size()) {
        const char32_t c = s[index];
        if (c == BS) {
            index += 2;
            continue;
        }
        if (c == U'{')
            depth += 1;
        else if (c == U'}') {
            depth -= 1;
            if (depth == 0) return std::make_pair(s.substr(open + 1, index - open - 1), index + 1);
        }
        index += 1;
    }
    return std::nullopt;
}
bool command_boundary(const u32 & s, size_t i) {
    return i >= s.size() || !isalpha(s[i]);
}
// returns 1 tag*, 2 tag, 3 leqno, 4 eqno, 0 none
int numbering_command_at(const u32 & s, size_t i) {
    if (startswith(s, U("\\tag*"), i) && command_boundary(s, i + 5)) return 1;
    if (startswith(s, U("\\tag"), i) && command_boundary(s, i + 4)) return 2;
    if (startswith(s, U("\\leqno"), i) && command_boundary(s, i + 6)) return 3;
    if (startswith(s, U("\\eqno"), i) && command_boundary(s, i + 5)) return 4;
    return 0;
}
u32 join_space(const std::vector<u32> & v) {
    u32 out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out.push_back(U' ');
        out += v[i];
    }
    return out;
}
void append_numbering(std::vector<u32> & target, const u32 & value) {
    const u32 v = strip(value);
    if (!v.empty()) target.push_back(v);
}
} // namespace

std::array<u32, 3> extract_formula_numbering(const u32 & text) {
    u32 parts;
    std::vector<u32> before, after;
    size_t index = 0;
    while (index < text.size()) {
        const int command = numbering_command_at(text, index);
        if (command == 1 || command == 2) {
            size_t group_start = index + (command == 1 ? 5 : 4);
            while (group_start < text.size() && isspace(text[group_start])) ++group_start;
            const auto parsed = parse_braced_group(text, group_start);
            if (!parsed) {
                parts.push_back(text[index]);
                index += 1;
                continue;
            }
            const u32 number = strip(parsed->first);
            index = parsed->second;
            if (command == 2)
                append_numbering(after, U("(") + number + U(")"));
            else
                append_numbering(after, number);
            continue;
        }
        if (command == 3 || command == 4) {
            size_t start = index + (command == 3 ? 6 : 5);
            while (start < text.size() && isspace(text[start])) ++start;
            size_t end = start;
            while (end < text.size() && !numbering_command_at(text, end)) ++end;
            const u32 number = strip(text.substr(start, end - start));
            if (command == 3)
                append_numbering(before, number);
            else
                append_numbering(after, number);
            index = end;
            continue;
        }
        parts.push_back(text[index]);
        index += 1;
    }
    return { strip(parts), join_space(before), join_space(after) };
}

u32 clean_formula_markdown_delimiters(const u32 & latex) {
    std::vector<size_t> dollars;
    u32 visible;
    int depth = 0;
    size_t index = 0;
    while (index < latex.size()) {
        if (latex[index] == U'%') {
            const size_t end = latex.find(U'\n', index);
            index = end == u32::npos ? latex.size() : end;
            continue;
        }
        const size_t cmd = latex_command(latex, index);
        if (cmd != u32::npos) {
            const u32 name = latex.substr(index + 1, cmd - index - 1);
            index = cmd;
            if (in_text_commands(name)) {
                if (name == U("verb")) return latex;
                size_t cursor = index;
                while (cursor < latex.size() && isspace(latex[cursor])) ++cursor;
                const auto group = parse_braced_group(latex, cursor);
                if (!group) return latex;
                index = group->second;
            }
            visible.push_back(U' ');
            continue;
        }
        const char32_t c = latex[index];
        if (c == U'{')
            depth += 1;
        else if (c == U'}') {
            depth -= 1;
            if (depth < 0) return latex;
        } else if (c == U'$') {
            if (depth || startswith(latex, U("$$"), index) || (index && latex[index - 1] == U'$')) return latex;
            dollars.push_back(index);
        }
        visible.push_back(c);
        index += 1;
    }
    if (depth || dollars.empty()) return latex;
    // "`" in prose or re.search(r"[㐀-鿿]|\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\b", prose)
    for (char32_t c : visible)
        if (c == U'`' || (c >= 0x3400 && c <= 0x9FFF)) return latex;
    for (size_t p = 0; p < visible.size(); ++p) {
        if (!is_ascii_alpha(visible[p]) || (p > 0 && is_word(visible[p - 1]))) continue;
        size_t q = p;
        while (q < visible.size() && is_ascii_alpha(visible[q])) ++q;
        if (q - p < 2 || q >= visible.size() || !isspace(visible[q])) continue;
        size_t r = q;
        while (r < visible.size() && isspace(visible[r])) ++r;
        size_t t = r;
        while (t < visible.size() && is_ascii_alpha(visible[t])) ++t;
        if (t - r < 2) continue;
        if (t < visible.size() && is_word(visible[t])) continue;
        return latex;
    }
    const bool wrapper = dollars.size() == 2 && dollars[0] == 0 && dollars[1] == latex.size() - 1;
    bool seams = dollars.size() % 2 == 0;
    for (size_t k = 0; seams && k + 1 < dollars.size(); k += 2)
        if (!all_space(latex.substr(dollars[k] + 1, dollars[k + 1] - dollars[k] - 1))) seams = false;
    if (!wrapper && !seams) return latex;
    if (wrapper) return latex.substr(1, latex.size() - 2);
    auto ends_op = [](const u32 & s) {
        if (s.empty()) return false;
        const char32_t c = s.back();
        return c == U'=' || c == U'+' || c == U'-' || c == U'*' || c == U'/' || c == U'<' || c == U'>';
    };
    auto starts_op = [](const u32 & s) {
        if (s.empty()) return false;
        const char32_t c = s.front();
        return c == U'=' || c == U'+' || c == U'-' || c == U'*' || c == U'/' || c == U'<' || c == U'>';
    };
    for (size_t k = 0; k + 1 < dollars.size(); k += 2) {
        const size_t start = dollars[k], end = dollars[k + 1];
        if (!(ends_op(rstrip(latex.substr(0, start))) || starts_op(lstrip(latex.substr(end + 1))))) return latex;
    }
    u32 out;
    size_t cursor = 0;
    for (size_t k = 0; k + 1 < dollars.size(); k += 2) {
        const size_t start = dollars[k], end = dollars[k + 1];
        out += latex.substr(cursor, start - cursor);
        out += latex.substr(start + 1, end - start - 1);
        cursor = end + 1;
    }
    out += latex.substr(cursor);
    return out;
}

namespace {
size_t find_unescaped(const u32 & s, const u32 & marker, size_t start) {
    size_t index = start;
    while (true) {
        index = s.find(marker, index);
        if (index == u32::npos) return index;
        if (!is_escaped(s, index)) return index;
        index += marker.size();
    }
}
u32 rewrite_numbered(const u32 & open, const u32 & formula, const u32 & close) {
    const auto parts = extract_formula_numbering(formula);
    if (parts[1].empty() && parts[2].empty()) return open + formula + close;
    std::vector<u32> pieces;
    if (!parts[1].empty()) pieces.push_back(parts[1]);
    pieces.push_back(open + parts[0] + close);
    if (!parts[2].empty()) pieces.push_back(parts[2]);
    return join_space(pieces);
}
} // namespace

u32 move_formula_numbering_outside_math(const u32 & text) {
    u32 result;
    size_t index = 0;
    while (index < text.size()) {
        if (startswith(text, U("\\["), index)) {
            const size_t close = text.find(U("\\]"), index + 2);
            if (close != u32::npos) {
                result += rewrite_numbered(U("\\["), text.substr(index + 2, close - index - 2), U("\\]"));
                index = close + 2;
                continue;
            }
        }
        if (startswith(text, U("\\("), index)) {
            const size_t close = text.find(U("\\)"), index + 2);
            if (close != u32::npos) {
                result += rewrite_numbered(U("\\("), text.substr(index + 2, close - index - 2), U("\\)"));
                index = close + 2;
                continue;
            }
        }
        if (startswith(text, U("$$"), index) && !is_escaped(text, index)) {
            const size_t close = find_unescaped(text, U("$$"), index + 2);
            if (close != u32::npos) {
                result += rewrite_numbered(U("$$"), text.substr(index + 2, close - index - 2), U("$$"));
                index = close + 2;
                continue;
            }
        }
        if (text[index] == U'$' && !is_escaped(text, index)) {
            const size_t close = find_unescaped(text, U("$"), index + 1);
            if (close != u32::npos) {
                result += rewrite_numbered(U("$"), text.substr(index + 1, close - index - 1), U("$"));
                index = close + 1;
                continue;
            }
        }
        result.push_back(text[index]);
        index += 1;
    }
    return result;
}

u32 postprocess_text(const u32 & raw) {
    const u32 stripped = strip(raw);
    u32 collapsed;
    for (size_t i = 0; i < stripped.size(); ++i) {
        if (stripped[i] == U'\n' && !collapsed.empty() && collapsed.back() == U'\n' && i > 0 &&
            stripped[i - 1] == U'\n')
            continue;
        collapsed.push_back(stripped[i]);
    }
    return move_formula_numbering_outside_math(collapsed);
}

std::array<u32, 3> postprocess_formula(const u32 & raw) {
    u32 text = strip(raw);
    if (auto m = fenced_search(text, 0, "latex", false)) {
        text = *m;
    } else if (contains(text, U("```"))) {
        if (auto m2 = fenced_search(text, 2, nullptr, false)) text = *m2;
    }
    const std::pair<const char *, const char *> wrappers[] = {
        { "$$", "$$" }, { "$", "$" }, { "\\[", "\\]" }, { "\\(", "\\)" }
    };
    for (const auto & w : wrappers) {
        if (auto m = anchored_wrapper(text, U(w.first), U(w.second), true)) {
            text = strip(*m);
            break;
        }
    }
    auto parts = extract_formula_numbering(text);
    parts[0] = space_latex_row_breaks(parts[0]);
    return parts;
}

u32 postprocess_code(const u32 & raw) {
    u32 text = strip(raw);
    if (auto m = fenced_search(text, 1, nullptr, false)) text = strip(*m);
    if (text.empty()) return text;
    return U("```\n") + text + U("\n```");
}

u32 postprocess_chemical(const u32 & raw) {
    u32 text = strip(raw);
    if (auto m = fenced_search(text, 1, nullptr, false)) text = strip(*m);
    const std::pair<const char *, const char *> wrappers[] = {
        { "$$", "$$" }, { "$", "$" }, { "\\[", "\\]" }, { "\\(", "\\)" }
    };
    for (const auto & w : wrappers)
        if (auto m = anchored_wrapper(text, U(w.first), U(w.second), false)) text = strip(*m);
    if (auto m = anchored_wrapper(text, U("\\smiles{"), U("}"), false)) text = strip(*m);
    if (text.empty()) return text;
    return U("$\\smiles{") + text + U("}$");
}

u32 strip_table_code_fence(const u32 & raw) {
    const u32 text = strip(raw);
    if (auto m = fenced_search(text, 0, "html", true)) return strip(*m);
    if (contains(text, U("```")))
        if (auto m = fenced_search(text, 2, nullptr, false)) return strip(*m);
    return text;
}

} // namespace text
} // namespace oxr_document
