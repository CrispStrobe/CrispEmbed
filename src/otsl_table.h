// otsl_table.h - one shared OTSL (Optimized Table Structure Language) to HTML
// converter for every CrispEmbed caller, with each caller's semantics kept as
// an explicit profile. Header-only so weight-free tests can use it directly.
//
//   upstream_docling  Byte-exact docling/mineru-vl-utils convert_otsl_to_html
//                     (mineru-vl-utils c9797cde post_process/otsl2html.py,
//                     Apache-2.0; translated to C++). Permissive: never rejects,
//                     ragged rows padded, spans follow the upstream next-token
//                     rules. Used by the MinerU full document route.
//   mineru_basic_v1   The MinerU native *basic* route (pre-2026-10-31). Moved
//                     verbatim from mineru_layout.h; output unchanged. Differs
//                     from upstream in span detection for malformed input.
//   marker_strict     Marker/TeleOCR validating parser: rejects text before
//                     cells, unknown tokens, orphan/overlapping spans, with
//                     warnings. Moved verbatim from marker_document.cpp; the
//                     caller keeps its trim/escape/limit and <table> handling.
//
// The profiles intentionally differ on malformed input; tests/test_otsl_table.cpp
// records each one's behaviour against the pre-extraction implementations.
#pragma once

#include "pyre.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <regex>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace otsl_table {

using pyre::ustring;

enum class profile { upstream_docling, mineru_basic_v1, marker_strict };

// ---------------------------------------------------- mineru_basic_v1 ------
inline bool python_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
           cp == 0x3000;
}

inline std::string python_trim(const std::string & s) {
    size_t first = s.size(), last = 0;
    for (size_t i = 0; i < s.size();) {
        const size_t begin = i;
        const uint32_t b = static_cast<unsigned char>(s[i++]);
        uint32_t cp = b;
        int more = 0;
        if (b >= 0xc2 && b < 0xe0) {
            cp = b & 31;
            more = 1;
        } else if (b >= 0xe0 && b < 0xf0) {
            cp = b & 15;
            more = 2;
        } else if (b >= 0xf0 && b <= 0xf4) {
            cp = b & 7;
            more = 3;
        }
        bool valid = i + static_cast<size_t>(more) <= s.size();
        for (int k = 0; valid && k < more; ++k) {
            const unsigned c = static_cast<unsigned char>(s[i + k]);
            if ((c & 0xc0) != 0x80)
                valid = false;
            else
                cp = (cp << 6) | (c & 63);
        }
        if (valid) i += more;
        if (!valid || !python_space(cp)) {
            first = std::min(first, begin);
            last = i;
        }
    }
    return first == s.size() ? "" : s.substr(first, last - first);
}

inline std::string html_escape_python(const std::string & s) {
    std::string out;
    for (char c : s) {
        switch (c) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        case '"':
            out += "&quot;";
            break;
        case '\'':
            out += "&#x27;";
            break;
        default:
            out += c;
        }
    }
    return out;
}

inline bool otsl_token(const std::string & s) {
    return s == "<nl>" || s == "<fcel>" || s == "<ecel>" || s == "<lcel>" || s == "<ucel>" || s == "<xcel>";
}

inline std::string to_html_basic_v1(const std::string & input) {
    if (input.compare(0, 6, "<table") == 0 && input.size() >= 8 && input.compare(input.size() - 8, 8, "</table>") == 0)
        return input;
    static const std::regex token_re(R"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)");
    std::vector<std::string> tokens, mixed;
    size_t last = 0;
    for (std::sregex_iterator it(input.begin(), input.end(), token_re), end; it != end; ++it) {
        const size_t pos = static_cast<size_t>(it->position());
        const std::string text = input.substr(last, pos - last);
        if (!python_trim(text).empty()) mixed.push_back(text);
        tokens.push_back(it->str());
        mixed.push_back(it->str());
        last = pos + static_cast<size_t>(it->length());
    }
    if (!python_trim(input.substr(last)).empty()) mixed.push_back(input.substr(last));
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string> row;
    for (const auto & t : tokens) {
        if (t == "<nl>") {
            if (!row.empty()) {
                rows.push_back(row);
                row.clear();
            }
        } else
            row.push_back(t);
    }
    if (!row.empty()) rows.push_back(row);
    if (rows.empty()) return "";
    size_t cols = 0;
    for (const auto & r : rows) cols = std::max(cols, r.size());
    for (auto & r : rows) r.resize(cols, "<ecel>");
    // Faithful to the blueprint's ragged-row reconstruction, including its
    // handling of non-token text. Only <fcel> gives a cell textual content.
    std::vector<std::string> rebuilt;
    size_t at = 0;
    for (const auto & r : rows) {
        for (const auto & t : r) {
            rebuilt.push_back(t);
            if (at < mixed.size() && mixed[at] == t) {
                ++at;
                if (at < mixed.size() && !otsl_token(mixed[at])) rebuilt.push_back(mixed[at++]);
            }
        }
        rebuilt.push_back("<nl>");
        if (at < mixed.size() && mixed[at] == "<nl>") ++at;
    }
    struct cell {
        size_t r, c, rs, cs;
        std::string text;
    };
    std::vector<cell> cells;
    size_t r = 0, c = 0;
    for (size_t i = 0; i < rebuilt.size(); ++i) {
        const auto & t = rebuilt[i];
        if (t == "<fcel>" || t == "<ecel>") {
            cell v{ r, c, 1, 1, "" };
            if (t == "<fcel>" && i + 1 < rebuilt.size() && !otsl_token(rebuilt[i + 1]))
                v.text = python_trim(rebuilt[i + 1]);
            while (c + v.cs < cols && (rows[r][c + v.cs] == "<lcel>" || rows[r][c + v.cs] == "<xcel>")) ++v.cs;
            while (r + v.rs < rows.size() && (rows[r + v.rs][c] == "<ucel>" || rows[r + v.rs][c] == "<xcel>")) ++v.rs;
            cells.push_back(v);
        }
        if (t == "<nl>") {
            ++r;
            c = 0;
        } else if (otsl_token(t))
            ++c;
    }
    if (cells.empty()) return "";
    // The Python TableData starts with an empty independent cell at every
    // coordinate then overwrites spans in source order (also for malformed OTSL).
    std::vector<cell> grid;
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols; ++j) grid.push_back({ i, j, 1, 1, "" });
    for (const auto & v : cells)
        for (size_t i = v.r; i < std::min(v.r + v.rs, rows.size()); ++i)
            for (size_t j = v.c; j < std::min(v.c + v.cs, cols); ++j) grid[i * cols + j] = v;
    std::ostringstream html;
    html << "<table>";
    for (size_t i = 0; i < rows.size(); ++i) {
        html << "<tr>";
        for (size_t j = 0; j < cols; ++j) {
            const auto & v = grid[i * cols + j];
            if (v.r != i || v.c != j) continue;
            html << "<td";
            if (v.rs > 1) html << " rowspan=\"" << v.rs << "\"";
            if (v.cs > 1) html << " colspan=\"" << v.cs << "\"";
            html << ">" << html_escape_python(v.text) << "</td>";
        }
        html << "</tr>";
    }
    html << "</table>";
    return html.str();
}


// --------------------------------------------------- upstream_docling ------
inline ustring upstream_html_escape(const ustring & s) {
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
namespace detail {
inline bool otsl_tok(const ustring & s) {
    return s == U"<nl>" || s == U"<fcel>" || s == U"<ecel>" || s == U"<lcel>" || s == U"<ucel>" || s == U"<xcel>";
}
struct table_cell {
    long row_span = 1, col_span = 1, r0 = 0, r1 = 0, c0 = 0, c1 = 0;
    ustring text;
};
} // namespace detail

inline ustring to_html_upstream(const ustring & otsl) {
    if (pyre::starts(otsl, U"<table") && pyre::ends(otsl, U"</table>")) return otsl;
    static const pyre::regex pattern("(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)");
    const std::vector<ustring> tokens = pattern.findall(otsl);
    std::vector<ustring> texts;
    for (auto & part : pattern.split(otsl))
        if (!pyre::strip(part).empty()) texts.push_back(part);

    // otsl_parse_texts
    std::vector<std::vector<ustring>> rows;
    {
        std::vector<ustring> cur;
        bool in_group = false;
        for (const auto & t : tokens) {
            if (t == U"<nl>") {
                if (in_group) rows.push_back(cur);
                cur.clear();
                in_group = false;
            } else {
                cur.push_back(t);
                in_group = true;
            }
        }
        if (in_group) rows.push_back(cur);
    }
    std::vector<detail::table_cell> cells;
    if (!rows.empty()) {
        size_t max_cols = 0;
        for (auto & r : rows) max_cols = std::max(max_cols, r.size());
        for (auto & r : rows)
            while (r.size() < max_cols) r.push_back(U"<ecel>");
        std::vector<ustring> rebuilt;
        size_t ti = 0;
        for (const auto & r : rows) {
            for (const auto & t : r) {
                rebuilt.push_back(t);
                if (ti < texts.size() && texts[ti] == t) {
                    ++ti;
                    if (ti < texts.size() && !detail::otsl_tok(texts[ti])) rebuilt.push_back(texts[ti++]);
                }
            }
            rebuilt.push_back(U"<nl>");
            if (ti < texts.size() && texts[ti] == U"<nl>") ++ti;
        }
        texts = rebuilt;
    }
    auto count_right = [&](size_t c, size_t r) {
        long span = 0;
        size_t it = c;
        while (true) {
            if (it >= rows.at(r).size()) throw std::out_of_range("otsl count_right");
            const auto & t = rows[r][it];
            if (!(t == U"<lcel>" || t == U"<xcel>")) break;
            ++it;
            ++span;
            if (it >= rows[r].size()) return span;
        }
        return span;
    };
    auto count_down = [&](size_t c, size_t r) {
        long span = 0;
        size_t it = r;
        while (true) {
            if (c >= rows.at(it).size()) throw std::out_of_range("otsl count_down");
            const auto & t = rows[it][c];
            if (!(t == U"<ucel>" || t == U"<xcel>")) break;
            ++it;
            ++span;
            if (it >= rows.size()) return span;
        }
        return span;
    };
    size_t r_idx = 0, c_idx = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        const ustring & text = texts[i];
        ustring cell_text;
        if (text == U"<fcel>" || text == U"<ecel>") {
            long row_span = 1, col_span = 1;
            size_t right_offset = 1;
            if (text != U"<ecel>") {
                if (i + 1 >= texts.size()) throw std::out_of_range("otsl texts[i+1]");
                if (!detail::otsl_tok(texts[i + 1])) {
                    cell_text = texts[i + 1];
                    right_offset = 2;
                }
            }
            ustring next_right;
            if (i + right_offset < texts.size()) next_right = texts[i + right_offset];
            ustring next_bottom;
            if (r_idx + 1 < rows.size() && c_idx < rows[r_idx + 1].size()) next_bottom = rows[r_idx + 1][c_idx];
            if (next_right == U"<lcel>" || next_right == U"<xcel>") col_span += count_right(c_idx + 1, r_idx);
            if (next_bottom == U"<ucel>" || next_bottom == U"<xcel>") row_span += count_down(c_idx, r_idx + 1);
            detail::table_cell cell;
            cell.text = pyre::strip(cell_text);
            cell.row_span = row_span;
            cell.col_span = col_span;
            cell.r0 = static_cast<long>(r_idx);
            cell.r1 = static_cast<long>(r_idx) + row_span;
            cell.c0 = static_cast<long>(c_idx);
            cell.c1 = static_cast<long>(c_idx) + col_span;
            cells.push_back(cell);
        }
        if (detail::otsl_tok(text) && text != U"<nl>") ++c_idx;
        if (text == U"<nl>") {
            ++r_idx;
            c_idx = 0;
        }
    }
    const long nrows = static_cast<long>(rows.size());
    long ncols = 0;
    for (auto & r : rows) ncols = std::max(ncols, static_cast<long>(r.size()));
    if (cells.empty()) return ustring();
    std::vector<detail::table_cell> grid;
    grid.reserve(static_cast<size_t>(nrows * ncols));
    for (long i = 0; i < nrows; ++i)
        for (long j = 0; j < ncols; ++j) {
            detail::table_cell e;
            e.text.clear();
            e.r0 = i, e.r1 = i + 1, e.c0 = j, e.c1 = j + 1;
            grid.push_back(e);
        }
    for (const auto & cell : cells)
        for (long i = std::min(cell.r0, nrows); i < std::min(cell.r1, nrows); ++i)
            for (long j = std::min(cell.c0, ncols); j < std::min(cell.c1, ncols); ++j)
                grid[static_cast<size_t>(i * ncols + j)] = cell;
    ustring html = U"<table>";
    for (long i = 0; i < nrows; ++i) {
        html += U"<tr>";
        for (long j = 0; j < ncols; ++j) {
            const auto & cell = grid[static_cast<size_t>(i * ncols + j)];
            if (cell.r0 != i || cell.c0 != j) continue;
            html += U"<td";
            if (cell.row_span > 1)
                html += pyre::from_utf8((" rowspan=\"" + std::to_string(cell.row_span) + "\"").c_str());
            if (cell.col_span > 1)
                html += pyre::from_utf8((" colspan=\"" + std::to_string(cell.col_span) + "\"").c_str());
            html += U">";
            html += upstream_html_escape(pyre::strip(cell.text));
            html += U"</td>";
        }
        html += U"</tr>";
    }
    html += U"</table>";
    return html;
}


// ------------------------------------------------------- marker_strict -----
struct marker_hooks {
    std::string (*trim)(const std::string &);
    std::string (*escape)(const std::string &);
    void (*warn)(std::vector<std::string> &, const std::string &);
    size_t cell_limit;
};
namespace detail {
struct table_token {
    std::string kind, text;
};
struct marker_cell {
    size_t row, column, rows, columns;
    std::string text;
};
} // namespace detail

// Input: already trimmed, size-checked, not starting with "<table".
inline std::string marker_grid_to_html(const std::string & text, std::vector<std::string> & warnings,
                                       const marker_hooks & hooks) {
    using detail::table_token;
    using table_cell = detail::marker_cell;
    const std::set<std::string> tokens = { "fcel", "ecel", "lcel", "ucel", "xcel", "nl" };
    std::vector<std::vector<table_token>> rows;
    std::vector<table_token> row;
    size_t cursor = 0, cells_count = 0;
    while (cursor < text.size()) {
        const size_t at = text.find('<', cursor);
        if (at == std::string::npos) {
            if (!hooks.trim(text.substr(cursor)).empty() && (row.empty() || row.back().kind != "fcel"))
                hooks.warn(warnings, "Text outside an OTSL content cell ignored; raw output retained.");
            break;
        }
        if (!hooks.trim(text.substr(cursor, at - cursor)).empty()) {
            hooks.warn(warnings, "Text before an OTSL cell rejected; raw output retained.");
            return {};
        }
        const size_t end = text.find('>', at);
        if (end == std::string::npos) {
            hooks.warn(warnings, "Truncated OTSL token.");
            return {};
        }
        const std::string kind = text.substr(at + 1, end - at - 1);
        if (!tokens.count(kind)) {
            hooks.warn(warnings, "Unknown OTSL token; raw output retained.");
            return {};
        }
        cursor = end + 1;
        if (kind == "nl") {
            if (!row.empty()) {
                rows.push_back(std::move(row));
                row.clear();
            }
            continue;
        }
        table_token token{ kind, "" };
        if (kind == "fcel") {
            size_t next = cursor;
            while ((next = text.find('<', next)) != std::string::npos) {
                bool delimiter = false;
                for (const auto & candidate : tokens) {
                    if (text.compare(next + 1, candidate.size(), candidate) == 0 &&
                        next + candidate.size() + 1 < text.size() && text[next + candidate.size() + 1] == '>') {
                        delimiter = true;
                        break;
                    }
                }
                if (delimiter) break;
                ++next;
            }
            if (next == std::string::npos) next = text.size();
            token.text = hooks.trim(text.substr(cursor, next - cursor));
            cursor = next;
        }
        row.push_back(std::move(token));
        if (++cells_count > hooks.cell_limit) {
            hooks.warn(warnings, "Table exceeds the 16384-cell limit.");
            return {};
        }
    }
    if (!row.empty()) rows.push_back(std::move(row));
    if (rows.empty()) {
        hooks.warn(warnings, "Table output contains no OTSL cells.");
        return {};
    }
    size_t width = 0;
    for (const auto & r : rows) width = std::max(width, r.size());
    if (width > hooks.cell_limit / rows.size()) {
        hooks.warn(warnings, "Padded table exceeds the 16384-cell limit.");
        return {};
    }
    bool padded = false;
    for (auto & r : rows)
        while (r.size() < width) {
            r.push_back({ "ecel", "" });
            padded = true;
        }
    if (padded) hooks.warn(warnings, "Ragged OTSL rows padded with empty cells.");
    std::vector<int> owners(width * rows.size(), -1);
    std::vector<table_cell> cells;
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t c = 0; c < width; ++c) {
            const auto & token = rows[r][c];
            if (token.kind != "fcel" && token.kind != "ecel") continue;
            size_t columns = 1, height = 1;
            while (c + columns < width && rows[r][c + columns].kind == "lcel") ++columns;
            while (r + height < rows.size() && rows[r + height][c].kind == "ucel") ++height;
            const int owner = static_cast<int>(cells.size());
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < columns; ++x) {
                    const std::string expected = !y && !x ? token.kind : !y ? "lcel" : !x ? "ucel" : "xcel";
                    if (rows[r + y][c + x].kind != expected || owners[(r + y) * width + c + x] != -1) {
                        hooks.warn(warnings, "Invalid or overlapping OTSL spanning cell; raw output retained.");
                        return {};
                    }
                    owners[(r + y) * width + c + x] = owner;
                }
            cells.push_back({ r, c, height, columns, token.text });
        }
    }
    if (std::find(owners.begin(), owners.end(), -1) != owners.end()) {
        hooks.warn(warnings, "Orphan OTSL continuation cell; raw output retained.");
        return {};
    }
    std::string html = "<table>";
    for (size_t r = 0; r < rows.size(); ++r) {
        html += "<tr>";
        for (size_t c = 0; c < width; ++c) {
            const auto & cell = cells[static_cast<size_t>(owners[r * width + c])];
            if (cell.row != r || cell.column != c) continue;
            html += "<td";
            if (cell.rows > 1) html += " rowspan=\"" + std::to_string(cell.rows) + "\"";
            if (cell.columns > 1) html += " colspan=\"" + std::to_string(cell.columns) + "\"";
            html += ">" + hooks.escape(cell.text) + "</td>";
        }
        html += "</tr>";
    }
    return html + "</table>";
}


// ------------------------------------------------------------ dispatch -----
// Profiles that need no caller hooks.
inline std::string to_html(const std::string & otsl, profile p) {
    switch (p) {
    case profile::mineru_basic_v1:
        return to_html_basic_v1(otsl);
    case profile::upstream_docling:
        return pyre::to_utf8(to_html_upstream(pyre::from_utf8(otsl)));
    case profile::marker_strict:
        break;
    }
    throw std::invalid_argument("otsl_table::to_html: marker_strict requires marker_hooks");
}

} // namespace otsl_table
