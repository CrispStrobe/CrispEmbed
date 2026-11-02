// SPDX-License-Identifier: Apache-2.0
// Native document processing adapted from Marker 2.1.0 (Datalab contributors)
// and teleocr-rs (Ramahatratra Zafiamy and contributors). See NOTICE.marker.
// This is a deterministic alternative pipeline, not Surya model parity.
#include "marker_document.h"
#include "otsl_table.h"
#include "core/json.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace crisp_marker {
namespace {

constexpr size_t warning_limit = 128;
constexpr size_t table_cell_limit = 16384;
constexpr size_t text_limit = 4 * 1024 * 1024;
constexpr int recursion_limit = 24;

void warn(std::vector<std::string> & out, const std::string & message) {
    if (out.size() < warning_limit)
        out.push_back(message);
    else if (out.size() == warning_limit)
        out.push_back("Further warnings suppressed (128-warning limit).");
}

bool space(unsigned char c) {
    return std::isspace(c) != 0;
}
std::string trim(const std::string & value) {
    size_t a = 0, b = value.size();
    while (a < b && space(static_cast<unsigned char>(value[a]))) ++a;
    while (b > a && space(static_cast<unsigned char>(value[b - 1]))) --b;
    return value.substr(a, b - a);
}
std::string rtrim(const std::string & value) {
    size_t end = value.size();
    while (end && space(static_cast<unsigned char>(value[end - 1]))) --end;
    return value.substr(0, end);
}
std::string lower(std::string value) {
    for (char & c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}
bool starts(const std::string & value, const std::string & prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}
bool ends(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}
std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    size_t first = 0;
    while (first < text.size()) {
        size_t last = text.find('\n', first);
        if (last == std::string::npos) last = text.size();
        std::string part = text.substr(first, last - first);
        if (!part.empty() && part.back() == '\r') part.pop_back();
        lines.push_back(std::move(part));
        first = last + 1;
    }
    return lines;
}
std::string collapse_space(const std::string & text) {
    std::string out;
    bool pending = false;
    for (unsigned char c : text) {
        if (space(c))
            pending = !out.empty();
        else {
            if (pending) out += ' ';
            out += static_cast<char>(c);
            pending = false;
        }
    }
    return out;
}
std::string number(double value) {
    if (!std::isfinite(value)) return "0";
    if (std::abs(value) < 0.0000005) value = 0;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(6) << value;
    std::string out = stream.str();
    while (!out.empty() && out.back() == '0') out.pop_back();
    if (!out.empty() && out.back() == '.') out.pop_back();
    return out.empty() ? "0" : out;
}
std::string valid_utf8(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        const unsigned char first = static_cast<unsigned char>(value[i]);
        if (first < 0x80) {
            out += value[i++];
            continue;
        }
        size_t length = first >= 0xC2 && first <= 0xDF   ? 2
                        : first >= 0xE0 && first <= 0xEF ? 3
                        : first >= 0xF0 && first <= 0xF4 ? 4
                                                         : 0;
        uint32_t cp = length == 2 ? first & 0x1F : length == 3 ? first & 0x0F : first & 7;
        bool valid = length && i + length <= value.size();
        for (size_t j = 1; valid && j < length; ++j) {
            const unsigned char next = static_cast<unsigned char>(value[i + j]);
            valid = (next & 0xC0) == 0x80;
            cp = (cp << 6) | (next & 0x3F);
        }
        valid = valid && !(length == 2 && cp < 0x80) && !(length == 3 && cp < 0x800) &&
                !(length == 4 && cp < 0x10000) && cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
        if (valid) {
            out.append(value, i, length);
            i += length;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}
bool disallowed_control(unsigned char value) {
    return (value < 32 && value != '\n' && value != '\r' && value != '\t') || value == 127;
}
std::string markdown_literal(const std::string & value) {
    // Public render results use a NUL-terminated C ABI. Preserve layout
    // whitespace, but never let model controls terminate or alter the string.
    std::string out;
    for (unsigned char c : valid_utf8(value)) {
        if (disallowed_control(c))
            out += "\xEF\xBF\xBD";
        else
            out += static_cast<char>(c);
    }
    return out;
}
std::string bytes_hex(const std::string & value) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() * 2);
    for (unsigned char c : value) {
        out += digits[c >> 4];
        out += digits[c & 15];
    }
    return out;
}
std::string quote(const std::string & value) {
    return "\"" + core_json::json_escape_strict(valid_utf8(value)) + "\"";
}
std::string malformed_bytes_field(const std::string & field, const std::string & value) {
    return valid_utf8(value) == value ? "" : "," + quote(field + "_bytes_hex") + ":" + quote(bytes_hex(value));
}
bool good_box(const box & value) {
    return std::isfinite(value.x0) && std::isfinite(value.y0) && std::isfinite(value.x1) && std::isfinite(value.y1) &&
           value.x1 > value.x0 && value.y1 > value.y0;
}
box united(const box & a, const box & b) {
    if (!good_box(a)) return b;
    if (!good_box(b)) return a;
    return { std::min(a.x0, b.x0), std::min(a.y0, b.y0), std::max(a.x1, b.x1), std::max(a.y1, b.y1) };
}
double intersection(const box & a, const box & b) {
    return std::max(0.0, std::min(a.x1, b.x1) - std::max(a.x0, b.x0)) *
           std::max(0.0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
}
std::vector<point> rectangle(const box & value) {
    return { { value.x0, value.y0 }, { value.x1, value.y0 }, { value.x1, value.y1 }, { value.x0, value.y1 } };
}
double median(std::vector<double> values, double fallback = 10) {
    if (values.empty()) return fallback;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double result = values[middle];
    if (values.size() % 2 == 0) result = (result + *std::max_element(values.begin(), values.begin() + middle)) * 0.5;
    return result;
}
double line_font(const line & value) {
    double total = 0, weight = 0;
    for (const auto & item : value.spans) {
        if (item.font_size > 0 && std::isfinite(item.font_size)) {
            const double size = static_cast<double>(std::max<size_t>(1, trim(item.text).size()));
            total += item.font_size * size;
            weight += size;
        }
    }
    return weight ? total / weight : std::max(1.0, value.bbox.height());
}
double block_font(const block & value) {
    std::vector<double> sizes;
    for (const auto & item : value.lines) sizes.push_back(line_font(item));
    return median(sizes, value.bbox.height() / std::max<size_t>(1, split_lines(value.text).size()));
}
bool mostly_font(const line & value, bool bold) {
    size_t total = 0, matched = 0;
    for (const auto & item : value.spans) {
        const size_t count = trim(item.text).size();
        total += count;
        if (bold ? item.bold : item.monospace) matched += count;
    }
    return total && matched * 4 >= total * 3;
}
bool numeric(const std::string & text) {
    bool digit = false;
    for (unsigned char c : trim(text)) {
        if (std::isdigit(c))
            digit = true;
        else if (c != ',' && c != '.' && c != '-' && c != '+' && c != '%' && c != '(' && c != ')' && !space(c))
            return false;
    }
    return digit;
}
bool ascii_lower_start(const std::string & text) {
    const std::string t = trim(text);
    return !t.empty() && t.front() >= 'a' && t.front() <= 'z';
}
bool hyphenated(const std::string & text) {
    const std::string t = trim(text);
    return t.size() >= 2 && t.back() == '-' &&
           ((t[t.size() - 2] >= 'a' && t[t.size() - 2] <= 'z') || (t[t.size() - 2] >= '0' && t[t.size() - 2] <= '9'));
}
std::string join_paragraph(const std::string & a, const std::string & b) {
    std::string left = trim(a), right = trim(b);
    if (left.empty()) return right;
    if (right.empty()) return left;
    if (hyphenated(left) && ascii_lower_start(right)) {
        left.pop_back();
        return left + right;
    }
    return left + " " + right;
}

struct bullet_info {
    bool found = false, ordered = false;
    size_t prefix = 0;
};
bullet_info bullet(const std::string & value) {
    size_t first = 0;
    while (first < value.size() && space(static_cast<unsigned char>(value[first]))) ++first;
    const std::string tail = value.substr(first);
    for (const std::string mark : { "•", "●", "○", "◦", "■", "▪", "▫", "–", "—", "-", "*" }) {
        if (starts(tail, mark) && tail.size() > mark.size() && space(static_cast<unsigned char>(tail[mark.size()]))) {
            size_t n = first + mark.size();
            while (n < value.size() && space(static_cast<unsigned char>(value[n]))) ++n;
            return { true, false, n };
        }
    }
    size_t n = first;
    if (n < value.size() && value[n] == '(') ++n;
    size_t word = n;
    while (n < value.size() && std::isalnum(static_cast<unsigned char>(value[n])) && n - word < 6) ++n;
    if (n > word && n < value.size() && (value[n] == '.' || value[n] == ')') && n + 1 < value.size() &&
        space(static_cast<unsigned char>(value[n + 1]))) {
        const std::string token = value.substr(word, n - word);
        const bool digits = std::all_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); });
        const bool roman = token.size() <= 5 && token.find_first_not_of("ivxlcIVXLC") == std::string::npos;
        if ((digits && token.size() <= 3) || token.size() == 1 || roman) {
            ++n;
            while (n < value.size() && space(static_cast<unsigned char>(value[n]))) ++n;
            return { true, true, n };
        }
    }
    return {};
}

std::string canonical_label(const std::string & kind) {
    static const std::map<std::string, std::string> labels = { { "text", "Text" },
                                                               { "title", "SectionHeader" },
                                                               { "sectionheader", "SectionHeader" },
                                                               { "table", "Table" },
                                                               { "image", "Picture" },
                                                               { "picture", "Picture" },
                                                               { "figure", "Figure" },
                                                               { "char", "Table" },
                                                               { "diagram", "Diagram" },
                                                               { "code", "Code" },
                                                               { "algorithm", "Code" },
                                                               { "header", "PageHeader" },
                                                               { "pageheader", "PageHeader" },
                                                               { "footer", "PageFooter" },
                                                               { "pagefooter", "PageFooter" },
                                                               { "page_number", "PageFooter" },
                                                               { "page_footnote", "Footnote" },
                                                               { "footnote", "Footnote" },
                                                               { "aside_text", "Text" },
                                                               { "equation", "Equation" },
                                                               { "equation_block", "Equation" },
                                                               { "ref_text", "Bibliography" },
                                                               { "bibliography", "Bibliography" },
                                                               { "list", "ListGroup" },
                                                               { "listgroup", "ListGroup" },
                                                               { "phonetic", "Text" },
                                                               { "table_caption", "Caption" },
                                                               { "image_caption", "Caption" },
                                                               { "code_caption", "Caption" },
                                                               { "caption", "Caption" },
                                                               { "table_footnote", "Footnote" },
                                                               { "image_footnote", "Footnote" },
                                                               { "unknown", "Text" },
                                                               { "seal", "Text" },
                                                               { "form", "Form" },
                                                               { "tableofcontents", "TableOfContents" },
                                                               { "chemicalblock", "ChemicalBlock" },
                                                               { "blankpage", "BlankPage" } };
    auto it = labels.find(lower(kind));
    return it == labels.end() ? std::string() : it->second;
}
double cross(const point & a, const point & b, const point & c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
bool simple_polygon(const std::vector<point> & points) {
    double area = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        const point &a = points[i], &b = points[(i + 1) % points.size()];
        area += a.x * b.y - a.y * b.x;
        if (a.x == b.x && a.y == b.y) return false;
        for (size_t j = i + 2; j < points.size(); ++j) {
            if (i == 0 && j + 1 == points.size()) continue;
            const point &c = points[j], &d = points[(j + 1) % points.size()];
            const double ac = cross(a, b, c), ad = cross(a, b, d), ca = cross(c, d, a), cb = cross(c, d, b);
            if (ac * ad < 0 && ca * cb < 0) return false;
            auto on = [](const point & p, const point & q, const point & r, double orientation) {
                return std::abs(orientation) < 1e-9 && r.x >= std::min(p.x, q.x) && r.x <= std::max(p.x, q.x) &&
                       r.y >= std::min(p.y, q.y) && r.y <= std::max(p.y, q.y);
            };
            if (on(a, b, c, ac) || on(a, b, d, ad) || on(c, d, a, ca) || on(c, d, b, cb)) return false;
        }
    }
    return std::abs(area) > 1e-8;
}

// Only generated table markup is accepted. This deliberately has no general
// HTML parser, CSS, URL, event-handler, or embedded-resource support.
std::string decode_entities(const std::string & text) {
    std::string out;
    static const std::vector<std::pair<std::string, std::string>> entities = { { "&amp;", "&" }, { "&lt;", "<" },
                                                                               { "&gt;", ">" },  { "&quot;", "\"" },
                                                                               { "&#39;", "'" }, { "&#x27;", "'" } };
    for (size_t i = 0; i < text.size();) {
        if (text[i] != '&') {
            out += text[i++];
            continue;
        }
        bool found = false;
        for (const auto & pair : entities) {
            if (text.compare(i, pair.first.size(), pair.first) == 0) {
                out += pair.second;
                i += pair.first.size();
                found = true;
                break;
            }
        }
        if (!found) out += text[i++];
    }
    return out;
}
std::string safe_table(const std::string & input) {
    if (input.size() > text_limit) return {};
    const std::string text = trim(input);
    if (!starts(text, "<table") || !ends(text, "</table>")) return {};
    const std::set<std::string> allowed = { "table", "thead", "tbody", "tfoot", "tr", "td", "th" };
    std::vector<std::string> stack;
    std::string out;
    size_t pos = 0, cell_count = 0;
    while (pos < text.size()) {
        if (text[pos] != '<') {
            size_t end = text.find('<', pos);
            if (end == std::string::npos) end = text.size();
            std::string data = text.substr(pos, end - pos);
            if (!trim(data).empty() && (stack.empty() || (stack.back() != "td" && stack.back() != "th"))) return {};
            out += escape_html(decode_entities(data));
            pos = end;
            continue;
        }
        const size_t end = text.find('>', pos);
        if (end == std::string::npos || end - pos > 256) return {};
        std::string tag = text.substr(pos + 1, end - pos - 1);
        const bool closing = !tag.empty() && tag.front() == '/';
        if (closing) tag.erase(0, 1);
        size_t length = 0;
        while (length < tag.size() && std::isalpha(static_cast<unsigned char>(tag[length]))) ++length;
        const std::string name = tag.substr(0, length);
        if (!allowed.count(name)) return {};
        std::string attrs = trim(tag.substr(length));
        if (closing) {
            if (!attrs.empty() || stack.empty() || stack.back() != name) return {};
            stack.pop_back();
            out += "</" + name + ">";
        } else {
            if (stack.size() > 8) return {};
            const std::string parent = stack.empty() ? "" : stack.back();
            if ((name == "table" && !stack.empty()) ||
                ((name == "thead" || name == "tbody" || name == "tfoot") && parent != "table") ||
                (name == "tr" && parent != "table" && parent != "thead" && parent != "tbody" && parent != "tfoot") ||
                ((name == "td" || name == "th") && parent != "tr"))
                return {};
            if ((name == "td" || name == "th") && ++cell_count > table_cell_limit) return {};
            out += "<" + name;
            std::set<std::string> seen;
            while (!attrs.empty()) {
                const size_t equals = attrs.find('=');
                if (equals == std::string::npos) return {};
                const std::string key = trim(attrs.substr(0, equals));
                if ((name != "td" && name != "th") || (key != "rowspan" && key != "colspan") ||
                    !seen.insert(key).second)
                    return {};
                size_t a = equals + 1;
                while (a < attrs.size() && space(static_cast<unsigned char>(attrs[a]))) ++a;
                char q = 0;
                if (a < attrs.size() && (attrs[a] == '\'' || attrs[a] == '"')) q = attrs[a++];
                size_t b = a;
                unsigned value = 0;
                while (b < attrs.size() && std::isdigit(static_cast<unsigned char>(attrs[b])) && b - a < 5) {
                    value = value * 10 + static_cast<unsigned>(attrs[b++] - '0');
                }
                if (b == a || value < 1 || value > 4096) return {};
                if (q) {
                    if (b >= attrs.size() || attrs[b] != q) return {};
                    ++b;
                } else if (b < attrs.size() && !space(static_cast<unsigned char>(attrs[b])))
                    return {};
                out += " " + key + "=\"" + std::to_string(value) + "\"";
                attrs = trim(attrs.substr(b));
            }
            out += ">";
            stack.push_back(name);
        }
        pos = end + 1;
        if (stack.empty() && pos != text.size()) return {};
    }
    return stack.empty() && cell_count ? out : std::string();
}

struct table_token {
    std::string kind, text;
};
struct table_cell {
    size_t row, column, rows, columns;
    std::string text;
};

std::vector<span> row_cells(const line & input) {
    std::vector<span> spans;
    for (auto item : input.spans) {
        item.text = trim(item.text);
        if (!item.text.empty() && good_box(item.bbox)) spans.push_back(std::move(item));
    }
    std::stable_sort(spans.begin(), spans.end(), [](const span & a, const span & b) { return a.bbox.x0 < b.bbox.x0; });
    std::vector<span> cells;
    const double gap = std::max(2.5, line_font(input) * 0.85);
    for (auto item : spans) {
        if (!cells.empty() && item.bbox.x0 - cells.back().bbox.x1 < gap) {
            cells.back().text = join_paragraph(cells.back().text, item.text);
            cells.back().bbox = united(cells.back().bbox, item.bbox);
            cells.back().bold = cells.back().bold && item.bold;
        } else
            cells.push_back(std::move(item));
    }
    return cells;
}

// PDFium sometimes emits separate lines for the cells of one table row.
std::vector<line> table_rows(std::vector<line> lines) {
    std::stable_sort(lines.begin(), lines.end(), [](const line & a, const line & b) {
        return a.bbox.y0 < b.bbox.y0 || (a.bbox.y0 == b.bbox.y0 && a.bbox.x0 < b.bbox.x0);
    });
    std::vector<line> rows;
    for (auto & item : lines) {
        if (!good_box(item.bbox) || trim(text_of(item)).empty()) continue;
        if (!rows.empty()) {
            line & previous = rows.back();
            const double overlap = std::min(previous.bbox.y1, item.bbox.y1) - std::max(previous.bbox.y0, item.bbox.y0);
            if (overlap >= 0.6 * std::min(previous.bbox.height(), item.bbox.height())) {
                previous.spans.insert(previous.spans.end(), item.spans.begin(), item.spans.end());
                previous.bbox = united(previous.bbox, item.bbox);
                continue;
            }
        }
        rows.push_back(std::move(item));
    }
    return rows;
}

std::string make_pdf_table(const std::vector<line> & input, double * quality = nullptr) {
    if (input.size() < 3 || input.size() > table_cell_limit) return {};
    const auto lines = table_rows(input);
    if (lines.size() < 3) return {};
    std::vector<std::vector<span>> rows;
    std::map<size_t, size_t> counts;
    for (const auto & item : lines) {
        rows.push_back(row_cells(item));
        if (rows.back().size() >= 2 && rows.back().size() <= 64) ++counts[rows.back().size()];
    }
    size_t width = 0, anchors = 0;
    for (const auto & entry : counts)
        if (entry.second > anchors) {
            width = entry.first;
            anchors = entry.second;
        }
    if (anchors < 3 || width < 2 || width * rows.size() > table_cell_limit) return {};
    std::vector<double> starts_at;
    for (size_t c = 0; c < width; ++c) {
        std::vector<double> points;
        for (const auto & row : rows)
            if (row.size() == width) points.push_back(row[c].bbox.x0);
        starts_at.push_back(median(points));
    }
    std::vector<double> font_sizes;
    for (const auto & item : lines) font_sizes.push_back(line_font(item));
    const double tolerance = std::max(3.0, median(font_sizes) * 1.2);
    for (size_t c = 1; c < width; ++c)
        if (starts_at[c] - starts_at[c - 1] < tolerance * 2) return {};
    size_t aligned = 0, nonempty = 0, total = 0;
    std::vector<std::vector<std::string>> grid;
    for (const auto & row : rows) {
        std::vector<std::string> cells(width);
        bool row_aligned = true;
        for (const auto & item : row) {
            size_t nearest = 0;
            double distance = std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < width; ++c) {
                const double candidate = std::abs(starts_at[c] - item.bbox.x0);
                if (candidate < distance) {
                    distance = candidate;
                    nearest = c;
                }
            }
            if (distance > tolerance || !cells[nearest].empty()) row_aligned = false;
            cells[nearest] = join_paragraph(cells[nearest], item.text);
            ++total;
            if (numeric(item.text)) ++nonempty;
        }
        if (row_aligned && row.size() >= 2) ++aligned;
        grid.push_back(std::move(cells));
    }
    // Aligned text columns can also be prose; require compact rows or numeric
    // data. This conservative gate is deliberately narrower than upstream's
    // multi-candidate grid scorer and does not fabricate spanning cells.
    size_t short_cells = 0;
    for (const auto & row : rows)
        for (const auto & item : row)
            if (item.text.size() <= 48) ++short_cells;
    const double score = static_cast<double>(aligned) / rows.size();
    if (score < 0.8 || (nonempty * 5 < total && short_cells * 10 < total * 9)) return {};
    if (quality) *quality = score;
    bool header = !rows.front().empty();
    for (const auto & item : rows.front())
        if (!item.bold) header = false;
    std::string html = "<table>";
    for (size_t r = 0; r < grid.size(); ++r) {
        html += "<tr>";
        const std::string tag = header && r == 0 ? "th" : "td";
        for (const auto & cell : grid[r]) html += "<" + tag + ">" + escape_html(cell) + "</" + tag + ">";
        html += "</tr>";
    }
    return html + "</table>";
}

} // namespace

std::string escape_html(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : valid_utf8(text)) {
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
            out += "&#39;";
            break;
        default:
            if (c >= 32 || c == '\n' || c == '\r' || c == '\t')
                out += static_cast<char>(c);
            else
                out += "&#xfffd;";
        }
    }
    return out;
}

std::string text_of(const line & value) {
    std::string out;
    double last_x = 0;
    bool have_previous = false;
    for (const auto & item : value.spans) {
        if (item.text.empty()) continue;
        if (have_previous && !out.empty() && !space(static_cast<unsigned char>(out.back())) &&
            !space(static_cast<unsigned char>(item.text.front())) &&
            item.bbox.x0 - last_x > std::max(0.5, item.font_size * 0.12))
            out += ' ';
        out += item.text;
        last_x = item.bbox.x1;
        have_previous = true;
    }
    return out;
}

bool text_is_usable(const std::string & value) {
    if (value.empty() || value.size() > text_limit) return false;
    size_t count = 0, bad = 0, meaningful = 0, run = 0, longest_run = 0;
    uint32_t previous = 0;
    for (size_t i = 0; i < value.size();) {
        const unsigned char first = static_cast<unsigned char>(value[i]);
        uint32_t cp = first;
        size_t size = 1;
        if (first >= 0xC2 && first <= 0xDF) {
            cp = first & 0x1F;
            size = 2;
        } else if (first >= 0xE0 && first <= 0xEF) {
            cp = first & 0x0F;
            size = 3;
        } else if (first >= 0xF0 && first <= 0xF4) {
            cp = first & 7;
            size = 4;
        } else if (first >= 0x80) {
            ++bad;
            ++count;
            ++i;
            continue;
        }
        bool valid = i + size <= value.size();
        for (size_t j = 1; valid && j < size; ++j) {
            unsigned char next = static_cast<unsigned char>(value[i + j]);
            valid = (next & 0xC0) == 0x80;
            cp = (cp << 6) | (next & 0x3F);
        }
        valid = valid && !(size == 2 && cp < 0x80) && !(size == 3 && cp < 0x800) && !(size == 4 && cp < 0x10000) &&
                cp <= 0x10FFFF && !(cp >= 0xD800 && cp <= 0xDFFF);
        if (!valid) {
            ++bad;
            ++count;
            ++i;
            continue;
        }
        i += size;
        ++count;
        if ((cp < 32 && cp != '\t' && cp != '\r' && cp != '\n') || cp == 0xFFFD || (cp >= 0xE000 && cp <= 0xF8FF) ||
            (cp >= 0xF0000 && cp <= 0xFFFFD) || (cp >= 0x100000 && cp <= 0x10FFFD))
            ++bad;
        else if ((cp < 128 && std::isalnum(static_cast<unsigned char>(cp))) ||
                 (cp >= 0xC0 && !(cp >= 0x2000 && cp <= 0x206F) && cp != 0xFEFF))
            ++meaningful;
        run = cp == previous && cp != ' ' && cp != '\n' ? run + 1 : 1;
        longest_run = std::max(longest_run, run);
        previous = cp;
    }
    return count && meaningful && bad * 50 <= count && longest_run <= 48;
}

std::vector<block> parse_teleocr_layout(const std::string & text, double width, double height, int max_blocks,
                                        std::vector<std::string> & warnings) {
    if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0)
        throw std::invalid_argument("Layout page dimensions must be finite and positive.");
    if (max_blocks < 1 || max_blocks > 100000) throw std::invalid_argument("Invalid layout block limit.");
    if (text.size() > text_limit) throw std::length_error("Layout output exceeds the 4 MiB limit.");
    std::vector<block> out;
    size_t line_number = 0;
    std::istringstream layout_stream(text);
    std::string value;
    while (std::getline(layout_stream, value)) {
        ++line_number;
        if (line_number > 65536) {
            warn(warnings, "Layout exceeds the 65536-line parsing limit; remaining lines omitted.");
            break;
        }
        value = trim(value);
        if (value.empty() || value == "<|im_end|>" || value == "<|endoftext|>") continue;
        const std::string prefix = "Layout line " + std::to_string(line_number) + ": ";
        const size_t box_end = value.find('>');
        if (!starts(value, "<box:") || box_end == std::string::npos || value.compare(box_end + 1, 7, "<label:") != 0) {
            warn(warnings, prefix + "invalid syntax.");
            continue;
        }
        const size_t label_end = value.find('>', box_end + 8);
        if (label_end == std::string::npos || label_end + 2 >= value.size() || value[label_end + 1] != '<' ||
            value.back() != '>' || value.find('>', label_end + 2) != value.size() - 1) {
            warn(warnings, prefix + "missing or invalid orientation.");
            continue;
        }
        const std::string kind = lower(value.substr(box_end + 8, label_end - box_end - 8));
        const std::string canonical = canonical_label(kind);
        if (canonical.empty()) {
            warn(warnings, prefix + "unknown label '" + kind.substr(0, 80) + "'.");
            continue;
        }
        if (canonical == "BlankPage") continue;
        std::vector<unsigned> coordinates;
        bool valid = true;
        size_t pos = 5;
        while (pos < box_end) {
            while (pos < box_end && space(static_cast<unsigned char>(value[pos]))) ++pos;
            if (pos == box_end) break;
            unsigned coord = 0;
            size_t digits = 0;
            while (pos < box_end && std::isdigit(static_cast<unsigned char>(value[pos]))) {
                if (++digits > 4) valid = false;
                if (digits <= 4) coord = coord * 10 + static_cast<unsigned>(value[pos] - '0');
                ++pos;
            }
            if (digits == 0 || coord > 1000 || (pos < box_end && !space(static_cast<unsigned char>(value[pos])))) {
                valid = false;
                break;
            }
            coordinates.push_back(coord);
            if (coordinates.size() > 256) {
                valid = false;
                break;
            }
        }
        if (!valid || coordinates.size() < 4 || coordinates.size() % 2) {
            warn(warnings, prefix + "coordinates must be 4..256 paired integers in 0..1000.");
            continue;
        }
        block item;
        item.type = canonical;
        item.raw_label = kind;
        item.raw = value;
        item.source = "teleocr-layout";
        item.order = static_cast<int>(out.size());
        for (size_t i = 0; i < coordinates.size(); i += 2)
            item.polygon.push_back({ width * coordinates[i] / 1000.0, height * coordinates[i + 1] / 1000.0 });
        if (coordinates.size() == 4) {
            if (coordinates[0] >= coordinates[2] || coordinates[1] >= coordinates[3]) {
                warn(warnings, prefix + "empty or reversed rectangle.");
                continue;
            }
            item.bbox = { item.polygon[0].x, item.polygon[0].y, item.polygon[1].x, item.polygon[1].y };
            item.polygon = rectangle(item.bbox);
        } else {
            if (item.polygon.front().x == item.polygon.back().x && item.polygon.front().y == item.polygon.back().y)
                item.polygon.pop_back();
            if (item.polygon.size() < 3 || !simple_polygon(item.polygon)) {
                warn(warnings, prefix + "degenerate or self-intersecting polygon.");
                continue;
            }
            item.bbox = { width, height, 0, 0 };
            for (const auto & p : item.polygon) {
                item.bbox.x0 = std::min(item.bbox.x0, p.x);
                item.bbox.x1 = std::max(item.bbox.x1, p.x);
                item.bbox.y0 = std::min(item.bbox.y0, p.y);
                item.bbox.y1 = std::max(item.bbox.y1, p.y);
            }
        }
        std::string orientation = lower(value.substr(label_end + 2, value.size() - label_end - 3));
        for (char & c : orientation)
            if (!std::isalpha(static_cast<unsigned char>(c))) c = ' ';
        std::istringstream words(orientation);
        std::string word;
        bool found = false;
        while (words >> word) {
            if (word == "up" || word == "right" || word == "down" || word == "left") {
                item.angle = word == "right" ? 90 : word == "down" ? 180 : word == "left" ? 270 : 0;
                found = true;
                break;
            }
        }
        if (!found) warn(item.warnings, "Unrecognized orientation; preserved page orientation.");
        if (kind == "unknown") warn(item.warnings, "Unknown source region retained as Text.");
        if (kind == "char")
            warn(item.warnings, "Scientific figure's implied-table task mapped to Table; source label retained.");
        if (out.size() >= static_cast<size_t>(max_blocks)) {
            warn(warnings,
                 "Layout exceeds the " + std::to_string(max_blocks) + "-block limit; remaining regions omitted.");
            break;
        }
        out.push_back(std::move(item));
    }
    return out;
}

std::string otsl_to_html(const std::string & input, std::vector<std::string> & warnings) {
    const std::string text = trim(input);
    if (text.empty()) return {};
    if (text.size() > text_limit) {
        warn(warnings, "Table output exceeds the 4 MiB limit.");
        return {};
    }
    if (starts(text, "<table")) {
        const std::string safe = safe_table(text);
        if (safe.empty()) warn(warnings, "Unsafe or malformed HTML table rejected; raw output retained.");
        return safe;
    }
    // OTSL grid parsing is shared (otsl_table.h, profile marker_strict); the
    // algorithm moved verbatim and keeps this caller's output byte-identical.
    const otsl_table::marker_hooks hooks{ &trim, &escape_html, &warn, table_cell_limit };
    return otsl_table::marker_grid_to_html(text, warnings, hooks);
}

std::string pdf_table_html(const std::vector<line> & lines, std::vector<std::string> & warnings) {
    const std::string html = make_pdf_table(lines);
    if (html.empty() && !lines.empty())
        warn(warnings,
             "PDF table reconstruction found no reliable aligned grid; OCR or escaped text fallback required.");
    return html;
}

namespace {

struct reading_unit {
    block value;
    bool wide = false;
};

bool caption_text(const std::string & text) {
    const std::string t = lower(trim(text));
    for (const std::string prefix : { "figure ", "fig. ", "table ", "plate ", "algorithm ", "listing " }) {
        if (starts(t, prefix) && t.size() > prefix.size() &&
            (std::isdigit(static_cast<unsigned char>(t[prefix.size()])) || t[prefix.size()] == '('))
            return true;
    }
    return false;
}

std::vector<line> split_wide_gaps(const line & value, double page_width) {
    // Rotation can put word boxes at subtly different x coordinates due to
    // ascenders/descenders. Preserve PDFium's character stream on vertical
    // lines instead of sorting words by their ink-box left edge.
    if (value.bbox.height() > value.bbox.width() * 2 && value.bbox.height() > line_font(value) * 3) return { value };
    std::vector<span> sorted = value.spans;
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const span & a, const span & b) { return a.bbox.x0 < b.bbox.x0; });
    const double threshold = std::max(page_width * 0.03, line_font(value) * 2.0);
    std::vector<line> result;
    for (const auto & part : sorted) {
        if (trim(part.text).empty() || !good_box(part.bbox)) continue;
        if (result.empty() || part.bbox.x0 - result.back().bbox.x1 > threshold)
            result.push_back({ part.bbox, { part } });
        else {
            result.back().spans.push_back(part);
            result.back().bbox = united(result.back().bbox, part.bbox);
        }
    }
    return result;
}

double column_cut(const std::vector<reading_unit> & units, double width, double body_font) {
    if (width <= 0 || units.size() < 4) return -1;
    double best = -1, best_score = 0;
    for (int index = 20; index <= 80; ++index) {
        const double cut = width * index / 100.0;
        size_t left = 0, right = 0, crossing = 0;
        double ly0 = 1e30, ly1 = 0, ry0 = 1e30, ry1 = 0;
        for (const auto & unit : units) {
            const box & b = unit.value.bbox;
            if (b.width() > width * 0.70 || unit.value.type == "SectionHeader" || unit.value.type == "Table") continue;
            if (b.x1 <= cut - width * 0.008) {
                ++left;
                ly0 = std::min(ly0, b.y0);
                ly1 = std::max(ly1, b.y1);
            } else if (b.x0 >= cut + width * 0.008) {
                ++right;
                ry0 = std::min(ry0, b.y0);
                ry1 = std::max(ry1, b.y1);
            } else
                ++crossing;
        }
        const double overlap = std::min(ly1, ry1) - std::max(ly0, ry0);
        if (left < 2 || right < 2 || overlap < body_font * 2) continue;
        const double score = std::min(left, right) * 3.0 - crossing * 8.0 - std::abs(index - 50) * 0.001;
        if (score > best_score) {
            best_score = score;
            best = cut;
        }
    }
    return best_score >= 6 ? best : -1;
}

bool y_less(const reading_unit & a, const reading_unit & b) {
    if (a.value.bbox.y0 != b.value.bbox.y0) return a.value.bbox.y0 < b.value.bbox.y0;
    return a.value.bbox.x0 < b.value.bbox.x0;
}

std::vector<reading_unit> reading_order(std::vector<reading_unit> units, double width, double height, double body_font,
                                        std::vector<std::string> & warnings, bool infer_rotation = true) {
    size_t vertical = 0, text_units = 0;
    int direction_votes = 0, direction_count = 0;
    for (const auto & unit : units) {
        if (unit.value.lines.empty()) continue;
        ++text_units;
        const line & first = unit.value.lines.front();
        if (first.bbox.height() > first.bbox.width() * 2 && first.bbox.height() > line_font(first) * 3) {
            ++vertical;
            if (first.spans.size() >= 2) {
                const box &a = first.spans.front().bbox, &b = first.spans.back().bbox;
                const double delta_y = ((b.y0 + b.y1) - (a.y0 + a.y1)) * 0.5;
                if (std::abs(delta_y) > line_font(first) * 2) {
                    direction_votes += delta_y > 0 ? 1 : -1;
                    ++direction_count;
                }
            }
        }
    }
    if (infer_rotation && text_units && vertical * 2 > text_units) {
        if (direction_count && std::abs(direction_votes) * 4 >= direction_count * 3) {
            const bool clockwise = direction_votes > 0;
            warn(warnings, clockwise ? "Vertical word-span progression indicates a 90-degree quarter-turn; reading "
                                       "order uses unrotated geometry."
                                     : "Vertical word-span progression indicates a 270-degree quarter-turn; reading "
                                       "order uses unrotated geometry.");
            std::vector<box> originals;
            originals.reserve(units.size());
            for (size_t i = 0; i < units.size(); ++i) {
                auto & item = units[i].value;
                originals.push_back(item.bbox);
                const box b = item.bbox;
                item.bbox = clockwise ? box{ b.y0, width - b.x1, b.y1, width - b.x0 }
                                      : box{ height - b.y1, b.x0, height - b.y0, b.x1 };
                item.order = static_cast<int>(i);
            }
            units = reading_order(std::move(units), height, width, body_font, warnings, false);
            for (auto & unit : units) unit.value.bbox = originals[static_cast<size_t>(unit.value.order)];
            return units;
        }
        warn(warnings,
             "Vertical direction is ambiguous; PDFium character-stream order retained without geometric reordering.");
        return units;
    }
    const double cut = column_cut(units, width, body_font);
    std::stable_sort(units.begin(), units.end(), y_less);
    if (cut < 0) return units;
    warn(warnings, "Reading order uses a deterministic two-column whitespace heuristic.");
    std::vector<reading_unit> output, left, right;
    auto flush = [&]() {
        std::stable_sort(left.begin(), left.end(), y_less);
        std::stable_sort(right.begin(), right.end(), y_less);
        for (auto & item : left) output.push_back(std::move(item));
        for (auto & item : right) output.push_back(std::move(item));
        left.clear();
        right.clear();
    };
    for (auto & item : units) {
        const box & b = item.value.bbox;
        if (b.x0 < cut && b.x1 > cut) {
            flush();
            output.push_back(std::move(item));
        } else if ((b.x0 + b.x1) * 0.5 < cut)
            left.push_back(std::move(item));
        else
            right.push_back(std::move(item));
    }
    flush();
    return output;
}

void merge_block(block & into, const block & added, bool newline) {
    into.text = newline ? into.text + "\n" + added.text : join_paragraph(into.text, added.text);
    if (!added.raw.empty()) into.raw += (into.raw.empty() ? "" : "\n") + added.raw;
    into.lines.insert(into.lines.end(), added.lines.begin(), added.lines.end());
    into.bbox = united(into.bbox, added.bbox);
    into.polygon = rectangle(into.bbox);
}

void assign_ids(block & value, int page_index, int & next, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Document structure exceeds the nesting limit.");
    value.id = "/page/" + std::to_string(page_index) + "/" + value.type + "/" + std::to_string(next++);
    for (auto & child : value.children) assign_ids(child, page_index, next, depth + 1);
}

std::string margin_signature(const std::string & value) {
    std::string out;
    bool digits = false;
    for (unsigned char c : lower(collapse_space(value))) {
        if (std::isdigit(c)) {
            if (!digits) out += '#';
            digits = true;
        } else {
            out += static_cast<char>(c);
            digits = false;
        }
    }
    return out;
}

void normalize_block(block & value, double width, double height, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Document structure exceeds the nesting limit.");
    if (value.text.size() > text_limit || value.raw.size() > text_limit || value.html.size() > text_limit)
        throw std::length_error("Document block exceeds the 4 MiB text limit.");
    if (value.text.empty() && !value.lines.empty()) {
        for (const auto & item : value.lines)
            value.text = value.type == "Code" ? value.text + (value.text.empty() ? "" : "\n") + trim(text_of(item))
                                              : join_paragraph(value.text, text_of(item));
    }
    if (valid_utf8(value.text) != value.text || valid_utf8(value.raw) != value.raw)
        warn(value.warnings,
             "Malformed UTF-8 rendered with replacement characters; exact bytes retained in JSON hex fields.");
    if ((value.type == "Code" || value.type == "ChemicalBlock" || value.type == "Equation") &&
        std::any_of(value.text.begin(), value.text.end(), [](unsigned char c) { return disallowed_control(c); }))
        warn(value.warnings, "Control characters replaced in Markdown; original text and raw values retained in JSON.");
    if (value.polygon.empty() && good_box(value.bbox)) value.polygon = rectangle(value.bbox);
    if (!good_box(value.bbox)) warn(value.warnings, "Invalid block geometry retained in diagnostics.");
    if (value.bbox.x0 < 0 || value.bbox.y0 < 0 || value.bbox.x1 > width + 0.01 || value.bbox.y1 > height + 0.01)
        warn(value.warnings, "Block geometry extends outside the page.");
    if (!std::isfinite(value.confidence)) {
        value.confidence = 0;
        warn(value.warnings, "Non-finite confidence replaced by 0.");
    }
    if (value.generated_tokens < 0) {
        value.generated_tokens = 0;
        warn(value.warnings, "Negative generated-token count replaced by 0.");
    }
    if (value.token_cap_reached) warn(value.warnings, "Generation reached the token limit; content may be incomplete.");
    if (value.type == "Table") {
        if (!value.html.empty()) {
            const std::string safe = safe_table(value.html);
            if (safe.empty()) warn(value.warnings, "Unsafe or malformed table HTML removed; raw text preserved.");
            value.html = safe;
        }
        if (value.html.empty() && !value.lines.empty()) value.html = make_pdf_table(value.lines);
        if (value.html.empty() &&
            (value.text.find("<fcel>") != std::string::npos || starts(trim(value.text), "<table")))
            value.html = otsl_to_html(value.text, value.warnings);
    } else if (!value.html.empty()) {
        // The native backend returns text, not trusted HTML. No non-table
        // model markup is executed even if a caller fills this field.
        if (value.text.empty()) value.text = value.html;
        warn(value.warnings, "Non-table supplied HTML rendered as escaped text.");
        value.html.clear();
    }
    for (auto & child : value.children) normalize_block(child, width, height, depth + 1);
}

void split_list(block & group) {
    if (!group.children.empty()) return;
    std::vector<block> items;
    if (!group.lines.empty()) {
        for (const auto & item : group.lines) {
            const std::string text = trim(text_of(item));
            const auto marker = bullet(text);
            if (marker.found || items.empty()) {
                block child;
                child.type = "ListItem";
                child.text = marker.found ? text.substr(marker.prefix) : text;
                child.raw = text;
                child.raw_label = marker.ordered ? "ordered" : "unordered";
                child.lines = { item };
                child.bbox = item.bbox;
                child.polygon = rectangle(item.bbox);
                child.source = group.source;
                items.push_back(std::move(child));
            } else {
                items.back().text = join_paragraph(items.back().text, text);
                items.back().lines.push_back(item);
                items.back().bbox = united(items.back().bbox, item.bbox);
                items.back().polygon = rectangle(items.back().bbox);
            }
        }
    } else {
        for (const auto & source : split_lines(group.text)) {
            const std::string text = trim(source);
            if (text.empty()) continue;
            const auto marker = bullet(text);
            if (marker.found || items.empty()) {
                block child;
                child.type = "ListItem";
                child.text = marker.found ? text.substr(marker.prefix) : text;
                child.raw = text;
                child.raw_label = marker.ordered ? "ordered" : "unordered";
                child.bbox = group.bbox;
                child.polygon = group.polygon;
                child.source = group.source;
                items.push_back(std::move(child));
            } else
                items.back().text = join_paragraph(items.back().text, text);
        }
    }
    bool any_bullet = false;
    for (const auto & item : items)
        if (bullet(item.raw).found) any_bullet = true;
    if (!any_bullet) {
        group.type = "Text";
        warn(group.warnings, "List label had no item markers; retained as text.");
        return;
    }
    group.children = std::move(items);
}

void nest_list(block & group, double page_width) {
    if (group.children.empty()) return;
    std::vector<block> flat = std::move(group.children);
    std::vector<double> indents;
    std::vector<int> depths;
    for (const auto & item : flat) {
        const double x = item.bbox.x0;
        while (indents.size() > 1 && x < indents.back() - page_width * 0.012) indents.pop_back();
        if (indents.empty() || (x > indents.back() + page_width * 0.015 && indents.size() < 7)) indents.push_back(x);
        depths.push_back(static_cast<int>(indents.size() - 1));
    }
    size_t position = 0;
    std::function<std::vector<block>(int)> build = [&](int level) {
        std::vector<block> result;
        while (position < flat.size() && depths[position] >= level) {
            if (depths[position] > level && !result.empty()) {
                block nested;
                nested.type = "ListGroup";
                nested.source = group.source;
                nested.children = build(depths[position]);
                for (const auto & item : nested.children) nested.bbox = united(nested.bbox, item.bbox);
                nested.polygon = rectangle(nested.bbox);
                result.back().children.push_back(std::move(nested));
            } else {
                result.push_back(std::move(flat[position++]));
            }
        }
        return result;
    };
    group.children = build(0);
}

void group_lists(page & value) {
    std::vector<block> output;
    for (size_t i = 0; i < value.blocks.size();) {
        if (value.blocks[i].type == "ListGroup") {
            split_list(value.blocks[i]);
            if (value.blocks[i].type == "ListGroup") nest_list(value.blocks[i], value.width);
            output.push_back(std::move(value.blocks[i++]));
            continue;
        }
        if (value.blocks[i].type != "ListItem") {
            output.push_back(std::move(value.blocks[i++]));
            continue;
        }
        block group;
        group.type = "ListGroup";
        group.source = value.blocks[i].source;
        double previous_y = value.blocks[i].bbox.y1;
        while (i < value.blocks.size() && value.blocks[i].type == "ListItem" &&
               (group.children.empty() || value.blocks[i].bbox.y0 - previous_y <= value.height * 0.08)) {
            block item = std::move(value.blocks[i++]);
            const auto marker = bullet(item.text);
            if (marker.found) {
                if (item.raw.empty()) item.raw = item.text;
                item.text = item.text.substr(marker.prefix);
                item.raw_label = marker.ordered ? "ordered" : "unordered";
            }
            group.bbox = united(group.bbox, item.bbox);
            previous_y = item.bbox.y1;
            group.children.push_back(std::move(item));
        }
        group.polygon = rectangle(group.bbox);
        nest_list(group, value.width);
        output.push_back(std::move(group));
    }
    value.blocks = std::move(output);
}

bool image_type(const std::string & type) {
    return type == "Picture" || type == "Figure" || type == "Diagram";
}
bool caption_host(const std::string & type) {
    return image_type(type) || type == "Table";
}
bool nearby_caption(const block & caption, const block & host, double height) {
    if (caption.type != "Caption") return false;
    const double gap = std::max(0.0, std::max(caption.bbox.y0 - host.bbox.y1, host.bbox.y0 - caption.bbox.y1));
    const double overlap = std::min(caption.bbox.x1, host.bbox.x1) - std::max(caption.bbox.x0, host.bbox.x0);
    return gap <= height * 0.05 && overlap > 0;
}

void group_captions(page & value) {
    std::vector<block> output;
    for (size_t i = 0; i < value.blocks.size(); ++i) {
        block item = std::move(value.blocks[i]);
        if (!caption_host(item.type)) {
            output.push_back(std::move(item));
            continue;
        }
        const bool before = !output.empty() && nearby_caption(output.back(), item, value.height);
        const bool after = i + 1 < value.blocks.size() && nearby_caption(value.blocks[i + 1], item, value.height);
        if (!before && !after) {
            output.push_back(std::move(item));
            continue;
        }
        block group;
        group.type = item.type + "Group";
        group.source = item.source;
        group.bbox = item.bbox;
        if (before) {
            group.bbox = united(group.bbox, output.back().bbox);
            group.children.push_back(std::move(output.back()));
            output.pop_back();
        }
        group.children.push_back(std::move(item));
        if (after) {
            group.bbox = united(group.bbox, value.blocks[i + 1].bbox);
            group.children.push_back(std::move(value.blocks[++i]));
        }
        group.polygon = rectangle(group.bbox);
        output.push_back(std::move(group));
    }
    value.blocks = std::move(output);
}

void walk_blocks(std::vector<block> & blocks, const std::function<void(block &)> & fn, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Document structure exceeds the nesting limit.");
    for (auto & value : blocks) {
        fn(value);
        walk_blocks(value.children, fn, depth + 1);
    }
}

void running_margins(document & value, const options & opts) {
    std::map<std::pair<std::string, bool>, std::set<int>> occurrences;
    std::map<int, double> body_font;
    std::set<int> body_present;
    for (const auto & p : value.pages) {
        std::vector<double> fonts;
        for (const auto & item : p.blocks) {
            if (!trim(item.text).empty() && item.bbox.y0 >= p.height * 0.08 && item.bbox.y1 <= p.height * 0.87) {
                body_present.insert(p.index);
                for (const auto & row : item.lines) fonts.push_back(line_font(row));
            }
        }
        body_font[p.index] = median(fonts, 12);
    }
    auto candidate = [&](const block & item, const page & p) {
        const bool explicit_label = item.type == "PageHeader" || item.type == "PageFooter";
        if (!explicit_label && (!body_present.count(p.index) || block_font(item) > body_font[p.index] * 1.15))
            return false;
        return (item.type == "Text" || item.type == "SectionHeader" || item.type == "Caption" ||
                item.type == "PageHeader" || item.type == "PageFooter") &&
               item.bbox.height() <= p.height * 0.035 && trim(item.text).size() <= 150 && !trim(item.text).empty() &&
               (item.bbox.y1 <= p.height * 0.08 || item.bbox.y0 >= p.height * 0.87);
    };
    for (const auto & p : value.pages)
        for (const auto & item : p.blocks)
            if (candidate(item, p))
                occurrences[{ margin_signature(item.text), item.bbox.y1 <= p.height * 0.08 }].insert(p.index);
    for (auto & p : value.pages) {
        for (auto & item : p.blocks) {
            if (candidate(item, p)) {
                const bool top = item.bbox.y1 <= p.height * 0.08;
                const size_t repetitions = occurrences[{ margin_signature(item.text), top }].size();
                const bool page_number = !top && numeric(trim(item.text));
                const bool explicit_label = item.type == "PageHeader" || item.type == "PageFooter";
                if (repetitions >= 2 || page_number || explicit_label) {
                    item.type = top ? "PageHeader" : "PageFooter";
                    item.ignored = !opts.include_page_headers;
                    if (repetitions >= 2)
                        warn(item.warnings, "Repeated marginal text identified by position and normalized content.");
                }
            } else if (item.type == "PageHeader" || item.type == "PageFooter")
                item.ignored = !opts.include_page_headers;
        }
        std::stable_partition(p.blocks.begin(), p.blocks.end(),
                              [](const block & item) { return item.type == "PageHeader"; });
        std::stable_partition(p.blocks.begin(), p.blocks.end(),
                              [](const block & item) { return item.type != "Footnote" && item.type != "PageFooter"; });
    }
}

void section_hierarchy(document & value) {
    std::vector<double> sizes;
    for (const auto & p : value.pages)
        for (const auto & item : p.blocks)
            if (item.type == "SectionHeader" && !item.ignored && !trim(item.text).empty())
                sizes.push_back(block_font(item));
    std::sort(sizes.begin(), sizes.end(), std::greater<double>());
    std::vector<double> levels;
    for (double size : sizes)
        if (levels.empty() || size < levels.back() * 0.88)
            if (levels.size() < 6) levels.push_back(size);
    std::map<int, std::string> hierarchy;
    for (auto & p : value.pages) {
        walk_blocks(p.blocks, [&](block & item) {
            if (item.type == "SectionHeader" && !item.ignored) {
                if (trim(item.text).empty())
                    item.ignored = true;
                else {
                    if (item.heading_level < 1 || item.heading_level > 6) {
                        const double size = block_font(item);
                        size_t level = 0;
                        while (level + 1 < levels.size() && size < levels[level] * 0.88) ++level;
                        item.heading_level = static_cast<int>(std::min<size_t>(6, level + 1));
                    }
                    auto end = hierarchy.lower_bound(item.heading_level);
                    hierarchy.erase(end, hierarchy.end());
                    hierarchy[item.heading_level] = item.id;
                }
            }
            item.section_hierarchy = hierarchy;
        });
    }
}

void mark_continuations(document & value) {
    struct located {
        block * item;
        page * owner;
    };
    std::vector<located> flow;
    for (auto & p : value.pages)
        for (auto & item : p.blocks)
            if (!item.ignored && item.type != "PageHeader" && item.type != "PageFooter" && item.type != "Footnote")
                flow.push_back({ &item, &p });
    for (size_t i = 1; i < flow.size(); ++i) {
        auto & previous = *flow[i - 1].item;
        auto & next = *flow[i].item;
        const page &old_page = *flow[i - 1].owner, &new_page = *flow[i].owner;
        if (previous.type != "Text" || next.type != "Text" || previous.lines.size() < 2 || next.lines.empty()) continue;
        const bool page_break = old_page.index != new_page.index && next.bbox.x0 < new_page.width * 0.5 &&
                                next.bbox.y0 < new_page.height * 0.5 && previous.bbox.y1 > old_page.height * 0.65;
        const bool column_break = old_page.index == new_page.index &&
                                  next.bbox.x0 > previous.bbox.x1 + old_page.width * 0.02 &&
                                  next.bbox.y0 <= previous.bbox.y0 + 1;
        if (!page_break && !column_break) continue;
        const double first_x = next.lines.front().bbox.x0;
        double minimum_x = first_x;
        for (const auto & item : next.lines) minimum_x = std::min(minimum_x, item.bbox.x0);
        if (first_x > minimum_x + block_font(next) * 0.5) continue;
        const std::string tail = trim(previous.text);
        const bool punctuation =
            !tail.empty() && (tail.back() == '.' || tail.back() == '!' || tail.back() == '?' || tail.back() == ':');
        const bool full = previous.lines.back().bbox.x1 >= previous.bbox.x1 - block_font(previous) * 0.75;
        if (!punctuation && ascii_lower_start(next.text) && (hyphenated(tail) || full)) previous.continuation = true;
    }
}

} // namespace

void build_text_blocks(page & value) {
    if (!std::isfinite(value.width) || !std::isfinite(value.height) || value.width <= 0 || value.height <= 0)
        throw std::invalid_argument("Text-layout page dimensions must be finite and positive.");
    if (value.text_lines.size() > 100000) throw std::length_error("PDF page exceeds the 100000-line limit.");
    std::vector<line> clean;
    std::vector<double> font_sizes;
    for (const auto & item : value.text_lines) {
        if (!good_box(item.bbox)) {
            warn(value.warnings, "PDF line with invalid geometry omitted.");
            continue;
        }
        if (trim(text_of(item)).empty()) continue;
        clean.push_back(item);
        // Weight longer body lines more than short titles.
        const size_t copies = std::min<size_t>(8, 1 + text_of(item).size() / 20);
        for (size_t n = 0; n < copies; ++n) font_sizes.push_back(line_font(item));
    }
    const double body_font = median(font_sizes);
    std::vector<reading_unit> units;
    std::vector<bool> consumed(clean.size(), false);
    const auto rows = table_rows(clean);
    for (size_t begin = 0; begin < rows.size();) {
        size_t end = begin;
        bool numeric_cells = false;
        while (end < rows.size()) {
            const auto cells = row_cells(rows[end]);
            if (cells.size() < 2 || cells.size() > 64 ||
                (end > begin && rows[end].bbox.y0 - rows[end - 1].bbox.y1 > body_font * 1.8))
                break;
            for (const auto & cell : cells)
                if (numeric(cell.text)) numeric_cells = true;
            ++end;
        }
        // Bold column headings are common in ordinary two-column prose.
        // Automatic discovery requires numeric evidence; a model-provided
        // Table region still permits all-text grids through pdf_table_html.
        if (end - begin >= 3 && numeric_cells) {
            std::vector<line> subset(rows.begin() + begin, rows.begin() + end);
            const std::string html = make_pdf_table(subset);
            if (!html.empty()) {
                block item;
                item.type = "Table";
                item.raw_label = "aligned-pdf-grid";
                item.source = "pdftext-heuristic";
                item.lines = subset;
                item.html = html;
                for (const auto & row : subset) {
                    item.bbox = united(item.bbox, row.bbox);
                    item.text += (item.text.empty() ? "" : "\n") + text_of(row);
                }
                item.raw = item.text;
                item.polygon = rectangle(item.bbox);
                for (size_t i = 0; i < clean.size(); ++i)
                    if (intersection(clean[i].bbox, item.bbox) >= clean[i].bbox.width() * clean[i].bbox.height() * 0.9)
                        consumed[i] = true;
                units.push_back({ std::move(item), false });
            }
        }
        begin = end > begin ? end : begin + 1;
    }
    for (size_t i = 0; i < clean.size(); ++i) {
        if (consumed[i]) continue;
        for (const auto & piece : split_wide_gaps(clean[i], value.width)) {
            block item;
            const bool code_line = mostly_font(piece, false);
            item.text = code_line ? rtrim(text_of(piece)) : trim(text_of(piece));
            item.raw = item.text;
            item.lines = { piece };
            item.bbox = piece.bbox;
            item.polygon = rectangle(item.bbox);
            item.source = "pdftext-heuristic";
            const double size = line_font(piece);
            const auto marker = bullet(item.text);
            if (code_line)
                item.type = "Code";
            else if (caption_text(item.text))
                item.type = "Caption";
            else if (size < body_font * 0.88 && piece.bbox.y0 > value.height * 0.78 && !item.text.empty() &&
                     (std::isdigit(static_cast<unsigned char>(item.text.front())) || item.text.front() == '*' ||
                      item.text.front() == '['))
                item.type = "Footnote";
            else if (item.text.size() <= 180 &&
                     (size > body_font * 1.20 ||
                      (mostly_font(piece, true) && size >= body_font * 0.98 && item.text.size() <= 100 &&
                       !ends(item.text, ".") && !ends(item.text, ";"))))
                item.type = "SectionHeader";
            else if (marker.found)
                item.type = "ListItem";
            item.raw_label = item.type;
            units.push_back({ std::move(item), false });
        }
    }
    units = reading_order(std::move(units), value.width, value.height, body_font, value.warnings);
    value.blocks.clear();
    for (auto & unit : units) {
        auto & item = unit.value;
        bool merged = false;
        if (!value.blocks.empty() && !item.lines.empty()) {
            block & previous = value.blocks.back();
            const double gap = item.bbox.y0 - previous.bbox.y1;
            const bool close = gap >= -body_font * 0.3 && gap <= body_font * 0.9;
            const bool aligned = std::abs(item.bbox.x0 - previous.bbox.x0) <=
                                 body_font * (item.type == "Code" && previous.type == "Code" ? 12.0 : 1.2);
            const double font_ratio = block_font(item) / std::max(1.0, block_font(previous));
            const bool similar_font = font_ratio > 0.88 && font_ratio < 1.14;
            if (close && similar_font && aligned && item.type == previous.type &&
                (item.type == "Text" || item.type == "Code" ||
                 (item.type == "SectionHeader" && gap <= body_font * 0.4))) {
                merge_block(previous, item, item.type == "Code");
                merged = true;
            } else if (close && similar_font && item.type == "Text" && previous.type == "ListItem" &&
                       item.bbox.x0 > previous.bbox.x0 + body_font * 0.5 &&
                       item.bbox.x0 < previous.bbox.x0 + body_font * 4) {
                merge_block(previous, item, false);
                merged = true;
            }
        }
        if (!merged) value.blocks.push_back(std::move(item));
    }
    for (size_t i = 0; i < value.blocks.size(); ++i) value.blocks[i].order = static_cast<int>(i);
    value.extraction_method = "pdftext-heuristic";
}

void process_document(document & value, const options & opts) {
    if (opts.max_blocks < 1 || opts.max_pages < 1) throw std::invalid_argument("Document limits must be positive.");
    if (value.pages.size() > static_cast<size_t>(opts.max_pages))
        throw std::length_error("Document exceeds the page limit.");
    value.metadata["schema"] = "crispembed.marker/1";
    value.metadata["processor"] = "native-marker-inspired-v1";
    value.metadata["text_quality"] = "deterministic-unicode-glyph-and-repetition-v1";
    value.metadata["layout"] =
        opts.mode == 1 ? "PDF font/geometry heuristics" : "TeleOCR with PDF font/geometry fallback";
    value.metadata["upstream_marker"] = "2.1.0@71dd5fd7e51ab24f8624b6ba2fcb6014b4e40ed4";
    value.metadata["surya_parity"] = "not claimed; alternative TeleOCR and deterministic processing";
    value.metadata["headers"] =
        opts.include_page_headers ? "included" : "suppressed in rendered text; retained in JSON";
    for (auto & p : value.pages) {
        if (!std::isfinite(p.width) || !std::isfinite(p.height) || p.width <= 0 || p.height <= 0)
            throw std::invalid_argument("Document page dimensions must be finite and positive.");
        if (p.raw_layout.size() > text_limit) throw std::length_error("Raw layout exceeds the 4 MiB limit.");
        if (valid_utf8(p.raw_layout) != p.raw_layout)
            warn(p.warnings, "Malformed layout UTF-8 rendered with replacement characters; exact bytes retained in "
                             "raw_layout_bytes_hex.");
        if (p.blocks.size() > static_cast<size_t>(opts.max_blocks)) {
            warn(p.warnings, "Page exceeds the configured block limit; remaining blocks omitted.");
            p.blocks.resize(static_cast<size_t>(opts.max_blocks));
        }
        for (auto & item : p.blocks) normalize_block(item, p.width, p.height);
        group_lists(p);
        group_captions(p);
    }
    running_margins(value, opts);
    for (auto & p : value.pages) {
        int next = 0;
        for (size_t i = 0; i < p.blocks.size(); ++i) {
            p.blocks[i].order = static_cast<int>(i);
            assign_ids(p.blocks[i], p.index, next);
        }
    }
    section_hierarchy(value);
    mark_continuations(value);
}

namespace {

std::string markdown_escape(const std::string & input) {
    std::string out;
    for (unsigned char c : valid_utf8(input)) {
        if (c == '&')
            out += "&amp;";
        else if (c == '<')
            out += "&lt;";
        else if (c == '>')
            out += "&gt;";
        else {
            if (c == '\\' || c == '*' || c == '_' || c == '[' || c == ']' || c == char(96) || c == '#' || c == '!' ||
                c == '|' || c == '$')
                out += '\\';
            if (c >= 32 || c == '\n' || c == '\r' || c == '\t') out += static_cast<char>(c);
        }
    }
    return out;
}
bool safe_asset_name(const std::string & value) {
    if (value.empty() || value.front() == '/' || value.size() > 512 || value.find("..") != std::string::npos)
        return false;
    return std::all_of(value.begin(), value.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == '/'; });
}
bool safe_url(const std::string & value) {
    if (value.empty() || value.size() > 4096 ||
        std::any_of(value.begin(), value.end(), [](unsigned char c) { return c < 32 || c == 127; }))
        return false;
    const std::string scheme = lower(value);
    return starts(scheme, "https://") || starts(scheme, "http://") || starts(scheme, "mailto:") || starts(scheme, "#");
}
std::string inline_html(const block & value) {
    if (value.lines.size() != 1 || trim(text_of(value.lines.front())) != trim(value.text))
        return escape_html(value.text);
    std::string html;
    double last_x = 0;
    bool previous = false, trailing_space = false;
    for (const auto & item : value.lines.front().spans) {
        if (item.text.empty()) continue;
        if (previous && !trailing_space && !space(static_cast<unsigned char>(item.text.front())) &&
            item.bbox.x0 - last_x > std::max(0.5, item.font_size * 0.12))
            html += ' ';
        std::string text = escape_html(item.text);
        if (item.monospace) text = "<code>" + text + "</code>";
        if (item.italic) text = "<em>" + text + "</em>";
        if (item.bold) text = "<strong>" + text + "</strong>";
        if (safe_url(item.url)) text = "<a href=\"" + escape_html(item.url) + "\">" + text + "</a>";
        html += text;
        previous = true;
        trailing_space = space(static_cast<unsigned char>(item.text.back()));
        last_x = item.bbox.x1;
    }
    return trim(html);
}
std::string math_content(std::string value) {
    value = trim(value);
    if (starts(value, "$$") && ends(value, "$$") && value.size() >= 4)
        value = value.substr(2, value.size() - 4);
    else if (starts(value, "$") && ends(value, "$") && value.size() >= 2)
        value = value.substr(1, value.size() - 2);
    else if (((starts(value, "\\[") && ends(value, "\\]")) || (starts(value, "\\(") && ends(value, "\\)"))) &&
             value.size() >= 4)
        value = value.substr(2, value.size() - 4);
    return trim(value);
}
std::string markdown_math(const std::string & value) {
    // A malformed model result must not close its math delimiter and inject
    // raw HTML. Keep valid LaTeX escapes and express comparison signs in TeX.
    const std::string text = markdown_literal(math_content(value));
    std::string out;
    size_t backslashes = 0;
    for (char c : text) {
        if (c == '<')
            out += "\\lt{}";
        else if (c == '>')
            out += "\\gt{}";
        else if (c == '$' && backslashes % 2 == 0)
            out += "\\$";
        else
            out += c;
        backslashes = c == '\\' ? backslashes + 1 : 0;
    }
    return out;
}
std::string footnote_html(const std::string & value) {
    const std::string text = trim(value);
    size_t at = 0;
    if (at < text.size() && text[at] == '[') ++at;
    size_t digits = at;
    while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at])) && at - digits < 4) ++at;
    const std::string marker = text.substr(digits, at - digits);
    if (!marker.empty() && at < text.size() &&
        (text[at] == ']' || text[at] == '.' || space(static_cast<unsigned char>(text[at])))) {
        if (text[at] == ']' || text[at] == '.') ++at;
        return "<sup>" + marker + "</sup> " + escape_html(trim(text.substr(at)));
    }
    return escape_html(text);
}
bool ordered_list(const block & value) {
    if (value.children.empty()) return false;
    return value.children.front().raw_label == "ordered" || bullet(value.children.front().raw).ordered;
}

std::string block_html(const block & value, bool refs = false, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Renderer nesting limit exceeded.");
    if (value.ignored) return {};
    std::string children;
    for (const auto & child : value.children) {
        if (child.ignored) continue;
        children += refs ? "<content-ref src=\"" + escape_html(child.id) + "\"></content-ref>"
                         : block_html(child, false, depth + 1);
    }
    const std::string text = inline_html(value);
    if (value.type == "SectionHeader") {
        const std::string level = std::to_string(std::clamp(value.heading_level, 1, 6));
        return "<h" + level + ">" + text + "</h" + level + ">";
    }
    if (value.type == "Code") return "<pre><code>" + escape_html(value.text) + "</code></pre>";
    if (value.type == "Equation") return "<math display=\"block\">" + escape_html(math_content(value.text)) + "</math>";
    if (value.type == "ChemicalBlock") return "<pre class=\"chemical\">" + escape_html(value.text) + "</pre>";
    if (value.type == "ListGroup") {
        const std::string tag = ordered_list(value) ? "ol" : "ul";
        return "<" + tag + ">" + children + "</" + tag + ">";
    }
    if (value.type == "ListItem") return "<li>" + text + children + "</li>";
    if (value.type == "Table") {
        const std::string table = safe_table(value.html);
        return table.empty() ? "<pre class=\"table-fallback\">" + escape_html(value.text) + "</pre>" : table;
    }
    if (image_type(value.type)) {
        if (safe_asset_name(value.image_name))
            return "<figure><img src=\"" + escape_html(value.image_name) + "\" alt=\"" + escape_html(value.text) +
                   "\"></figure>";
        return "<p class=\"image-placeholder\">[" + escape_html(value.type) + (value.text.empty() ? "" : ": " + text) +
               "]</p>";
    }
    if (value.type == "Caption") return "<p class=\"caption\">" + text + "</p>";
    if (value.type == "Footnote") return "<aside class=\"footnote\">" + footnote_html(value.text) + "</aside>";
    if (value.type == "PageHeader") return "<header>" + text + "</header>";
    if (value.type == "PageFooter") return "<footer>" + text + "</footer>";
    if (!value.children.empty())
        return "<div data-block-type=\"" + escape_html(value.type) + "\">" + children + "</div>";
    return "<p" + std::string(value.continuation ? " class=\"has-continuation\"" : "") + ">" + text + "</p>";
}

std::string block_markdown(const block & value, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Renderer nesting limit exceeded.");
    if (value.ignored) return {};
    if (value.type == "SectionHeader")
        return std::string(static_cast<size_t>(std::clamp(value.heading_level, 1, 6)), '#') + " " +
               markdown_escape(value.text);
    if (value.type == "Code" || value.type == "ChemicalBlock") {
        size_t longest = 0, run = 0;
        for (char c : value.text) {
            run = c == char(96) ? run + 1 : 0;
            longest = std::max(longest, run);
        }
        const std::string fence(std::max<size_t>(3, longest + 1), char(96));
        return fence + (value.type == "ChemicalBlock" ? "chem" : "") + "\n" + markdown_literal(value.text) + "\n" +
               fence;
    }
    if (value.type == "Equation") return "$$\n" + markdown_math(value.text) + "\n$$";
    if (value.type == "Table") {
        const std::string table = safe_table(value.html);
        return table.empty() ? markdown_escape(value.text) : table;
    }
    if (value.type == "Footnote") return footnote_html(value.text);
    if (image_type(value.type)) {
        if (safe_asset_name(value.image_name))
            return "![" + markdown_escape(value.text) + "](" + value.image_name + ")";
        return "\\[" + markdown_escape(value.type + (value.text.empty() ? "" : ": " + value.text)) + "\\]";
    }
    if (value.type == "ListGroup") {
        std::string out;
        const bool ordered = ordered_list(value);
        size_t index = 1;
        for (const auto & child : value.children) {
            if (child.ignored) continue;
            if (!out.empty()) out += "\n";
            const std::string marker = ordered ? std::to_string(index++) + ". " : "- ";
            out += marker + markdown_escape(child.text);
            for (const auto & nested : child.children) {
                const std::string nested_text = block_markdown(nested, depth + 1);
                for (const auto & row : split_lines(nested_text)) out += "\n" + std::string(marker.size(), ' ') + row;
            }
        }
        return out;
    }
    if (!value.children.empty()) {
        std::string out;
        for (const auto & child : value.children) {
            const std::string content = block_markdown(child, depth + 1);
            if (!content.empty()) out += (out.empty() ? "" : "\n\n") + content;
        }
        return out;
    }
    return markdown_escape(value.text);
}

std::string warnings_json(const std::vector<std::string> & values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += quote(values[i]);
    }
    return out + "]";
}
std::string box_json(const box & value) {
    return "[" + number(value.x0) + "," + number(value.y0) + "," + number(value.x1) + "," + number(value.y1) + "]";
}
std::string polygon_json(const std::vector<point> & value) {
    std::string out = "[";
    for (size_t i = 0; i < value.size(); ++i) {
        if (i) out += ',';
        out += "[" + number(value[i].x) + "," + number(value[i].y) + "]";
    }
    return out + "]";
}
std::string hierarchy_json(const std::map<int, std::string> & value) {
    std::string out = "{";
    bool first = true;
    for (const auto & entry : value) {
        if (!first) out += ',';
        first = false;
        out += quote(std::to_string(entry.first)) + ":" + quote(entry.second);
    }
    return out + "}";
}
std::string strings_json(const std::map<std::string, std::string> & value) {
    std::string out = "{";
    bool first = true;
    for (const auto & entry : value) {
        if (!first) out += ',';
        first = false;
        out += quote(entry.first) + ":" + quote(entry.second);
    }
    return out + "}";
}
std::string lines_json(const std::vector<line> & values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += "{\"bbox\":" + box_json(values[i].bbox) + ",\"spans\":[";
        for (size_t j = 0; j < values[i].spans.size(); ++j) {
            if (j) out += ',';
            const span & s = values[i].spans[j];
            out += "{\"text\":" + quote(s.text) + ",\"bbox\":" + box_json(s.bbox) +
                   ",\"font_size\":" + number(s.font_size) + ",\"bold\":" + (s.bold ? "true" : "false") +
                   ",\"italic\":" + (s.italic ? "true" : "false") +
                   ",\"monospace\":" + (s.monospace ? "true" : "false") + ",\"url\":" + quote(s.url) + "}";
        }
        out += "]}";
    }
    return out + "]";
}
void collect_images(const block & value, std::map<std::string, std::string> & images, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Renderer nesting limit exceeded.");
    if (safe_asset_name(value.image_name)) images[value.image_name] = value.image_name;
    for (const auto & child : value.children) collect_images(child, images, depth + 1);
}
std::string block_json(const block & value, bool inline_children = false, int page_index = -1, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Renderer nesting limit exceeded.");
    std::map<std::string, std::string> images;
    collect_images(value, images);
    std::string out =
        "{\"id\":" + quote(value.id) + ",\"block_type\":" + quote(value.type) +
        ",\"html\":" + quote(block_html(value, !inline_children)) + ",\"polygon\":" + polygon_json(value.polygon) +
        ",\"bbox\":" + box_json(value.bbox) + ",\"section_hierarchy\":" + hierarchy_json(value.section_hierarchy) +
        ",\"images\":" + strings_json(images) + ",\"text\":" + quote(value.text) + ",\"raw\":" + quote(value.raw) +
        ",\"raw_label\":" + quote(value.raw_label) + ",\"source\":" + quote(value.source) +
        ",\"order\":" + std::to_string(value.order) + ",\"heading_level\":" + std::to_string(value.heading_level) +
        ",\"angle\":" + std::to_string(value.angle) + ",\"confidence\":" + number(value.confidence) +
        ",\"generated_tokens\":" + std::to_string(value.generated_tokens) +
        ",\"token_cap_reached\":" + (value.token_cap_reached ? "true" : "false") +
        ",\"ignored\":" + (value.ignored ? "true" : "false") +
        ",\"has_continuation\":" + (value.continuation ? "true" : "false") +
        ",\"warnings\":" + warnings_json(value.warnings) + ",\"lines\":" + lines_json(value.lines);
    out += malformed_bytes_field("raw", value.raw) + malformed_bytes_field("text", value.text);
    if (page_index >= 0) out += ",\"page\":" + std::to_string(page_index);
    out += ",\"children\":";
    if (value.children.empty())
        out += "null";
    else {
        out += "[";
        for (size_t i = 0; i < value.children.size(); ++i) {
            if (i) out += ',';
            out += block_json(value.children[i], inline_children, page_index, depth + 1);
        }
        out += "]";
    }
    return out + "}";
}
struct statistics {
    std::map<std::string, size_t> types;
    size_t tokens = 0, capped = 0, warnings = 0, blocks = 0;
};
void add_stats(statistics & result, const block & value, int depth = 0) {
    if (depth > recursion_limit) throw std::length_error("Renderer nesting limit exceeded.");
    ++result.types[value.type];
    ++result.blocks;
    result.tokens += static_cast<size_t>(std::max(0, value.generated_tokens));
    result.capped += value.token_cap_reached ? 1 : 0;
    result.warnings += value.warnings.size();
    for (const auto & child : value.children) add_stats(result, child, depth + 1);
}

} // namespace

std::string render_metadata(const document & value) {
    statistics total;
    size_t layout_tokens = 0, layout_capped_pages = 0;
    std::string pages = "[";
    for (size_t i = 0; i < value.pages.size(); ++i) {
        if (i) pages += ',';
        const page & p = value.pages[i];
        statistics stats;
        for (const auto & b : p.blocks) {
            add_stats(stats, b);
            add_stats(total, b);
        }
        layout_tokens += static_cast<size_t>(std::max(0, p.layout_generated_tokens));
        layout_capped_pages += p.layout_token_cap_reached ? 1 : 0;
        total.warnings += p.warnings.size();
        pages += "{\"page_id\":" + std::to_string(p.index) + ",\"width\":" + number(p.width) +
                 ",\"height\":" + number(p.height) + ",\"text_extraction_method\":" + quote(p.extraction_method) +
                 ",\"block_count\":" + std::to_string(stats.blocks) +
                 ",\"generated_tokens\":" + std::to_string(stats.tokens) +
                 ",\"layout_generated_tokens\":" + std::to_string(std::max(0, p.layout_generated_tokens)) +
                 ",\"layout_token_cap_reached\":" + (p.layout_token_cap_reached ? "true" : "false") +
                 ",\"capped_blocks\":" + std::to_string(stats.capped) + ",\"warnings\":" + warnings_json(p.warnings) +
                 ",\"block_types\":{";
        bool first = true;
        for (const auto & count : stats.types) {
            if (!first) pages += ',';
            first = false;
            pages += quote(count.first) + ":" + std::to_string(count.second);
        }
        pages += "}}";
    }
    pages += "]";
    std::string assets = "[";
    for (size_t i = 0; i < value.assets.size(); ++i) {
        if (i) assets += ',';
        assets += "{\"name\":" + quote(value.assets[i].name) + ",\"mime_type\":" + quote(value.assets[i].mime_type) +
                  ",\"bytes\":" + std::to_string(value.assets[i].bytes.size()) + "}";
    }
    assets += "]";
    return "{\"schema\":\"crispembed.marker/1\",\"source\":" + quote(value.source_name) +
           ",\"backend\":" + quote(value.backend) + ",\"page_count\":" + std::to_string(value.pages.size()) +
           ",\"generated_tokens\":" + std::to_string(total.tokens) +
           ",\"layout_generated_tokens\":" + std::to_string(layout_tokens) +
           ",\"layout_capped_pages\":" + std::to_string(layout_capped_pages) +
           ",\"capped_blocks\":" + std::to_string(total.capped) +
           ",\"warning_count\":" + std::to_string(total.warnings + value.warnings.size()) +
           ",\"warnings\":" + warnings_json(value.warnings) + ",\"page_stats\":" + pages + ",\"assets\":" + assets +
           ",\"configuration\":" + strings_json(value.metadata) + "}";
}

std::string render_document(const document & value, const std::string & format) {
    if (format == "markdown" || format == "md") {
        std::string out;
        bool continuation = false, dehyphenate = false;
        for (const auto & p : value.pages) {
            for (const auto & item : p.blocks) {
                const std::string content = block_markdown(item);
                if (content.empty()) continue;
                if (!out.empty()) {
                    if (continuation && item.type == "Text") {
                        if (dehyphenate && !out.empty() && out.back() == '-')
                            out.pop_back();
                        else
                            out += ' ';
                    } else
                        out += "\n\n";
                }
                out += content;
                continuation = item.continuation;
                dehyphenate = hyphenated(item.text);
            }
        }
        return out.empty() ? "" : out + "\n";
    }
    if (format == "html") {
        std::string out = "<article class=\"marker-document\" data-schema=\"crispembed.marker/1\">\n";
        for (const auto & p : value.pages) {
            out += "<section class=\"page\" data-page-id=\"" + std::to_string(p.index) + "\">\n";
            for (const auto & item : p.blocks) {
                const std::string content = block_html(item);
                if (!content.empty()) out += content + "\n";
            }
            out += "</section>\n";
        }
        return out + "</article>\n";
    }
    if (format == "json") {
        std::string out = "{\"block_type\":\"Document\",\"children\":[";
        for (size_t i = 0; i < value.pages.size(); ++i) {
            if (i) out += ',';
            const page & p = value.pages[i];
            const box bounds{ 0, 0, p.width, p.height };
            std::string refs;
            for (const auto & item : p.blocks)
                if (!item.ignored) refs += "<content-ref src=\"" + escape_html(item.id) + "\"></content-ref>";
            out += "{\"id\":" + quote("/page/" + std::to_string(p.index)) +
                   ",\"block_type\":\"Page\",\"html\":" + quote(refs) +
                   ",\"polygon\":" + polygon_json(rectangle(bounds)) + ",\"bbox\":" + box_json(bounds) +
                   ",\"text_extraction_method\":" + quote(p.extraction_method) +
                   ",\"raw_layout\":" + quote(p.raw_layout) +
                   ",\"layout_generated_tokens\":" + std::to_string(std::max(0, p.layout_generated_tokens)) +
                   ",\"layout_token_cap_reached\":" + (p.layout_token_cap_reached ? "true" : "false") +
                   ",\"warnings\":" + warnings_json(p.warnings) + malformed_bytes_field("raw_layout", p.raw_layout) +
                   ",\"children\":[";
            for (size_t j = 0; j < p.blocks.size(); ++j) {
                if (j) out += ',';
                out += block_json(p.blocks[j]);
            }
            out += "]}";
        }
        return out + "],\"metadata\":" + render_metadata(value) + "}\n";
    }
    if (format == "chunks") {
        std::string out = "{\"blocks\":[", pages = "{";
        bool first = true;
        for (size_t i = 0; i < value.pages.size(); ++i) {
            const page & p = value.pages[i];
            for (const auto & item : p.blocks) {
                if (item.ignored) continue;
                if (!first) out += ',';
                first = false;
                out += block_json(item, true, p.index);
            }
            if (i) pages += ',';
            pages += quote(std::to_string(p.index)) + ":{\"bbox\":[0,0," + number(p.width) + "," + number(p.height) +
                     "],\"text_extraction_method\":" + quote(p.extraction_method) +
                     ",\"raw_layout\":" + quote(p.raw_layout) +
                     ",\"layout_generated_tokens\":" + std::to_string(std::max(0, p.layout_generated_tokens)) +
                     ",\"layout_token_cap_reached\":" + (p.layout_token_cap_reached ? "true" : "false") +
                     malformed_bytes_field("raw_layout", p.raw_layout) + "}";
        }
        return out + "],\"page_info\":" + pages + "},\"metadata\":" + render_metadata(value) + "}\n";
    }
    throw std::invalid_argument("Unknown document format '" + format + "' (expected markdown, html, json, or chunks).");
}

} // namespace crisp_marker
