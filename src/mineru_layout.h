// MinerU2.5-Pro layout/crop/OTSL contracts, independent of model inference.
// Portions adapted from MinerU Team (2024-2026), Apache-2.0; see the
// third-party MinerU notice distributed with this implementation.
// Blueprint: opendatalab/mineru-vl-utils c9797cde8f54cc48306196fc6aebe73ea31dd4a3.
// The model-facing caller must enforce the MinerU opt-in gate. These weight-free
// helpers deliberately expose basic postprocessing separately from the Python
// package's full formula/image/table-image repair stack.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mineru {

struct block {
    std::string type;
    std::array<double, 4> bbox = { { 0, 0, 0, 0 } };
    int angle = -1; // Python None, otherwise 0, 90, 180, 270 (CCW crop correction).
    bool merge_prev = false;
    std::string content;
    bool has_content = false;
};

inline const char * blueprint_revision() {
    return "c9797cde8f54cc48306196fc6aebe73ea31dd4a3";
}

inline bool valid_type(const std::string & type) {
    static const std::vector<std::string> types = {
        "text",          "title",          "doc_title", "paragraph_title", "table",
        "equation",      "formula_number", "code",      "algorithm",       "aside_text",
        "ref_text",      "index",          "phonetic",  "list_item",       "table_caption",
        "image_caption", "code_caption",   "caption",   "table_footnote",  "image_footnote",
        "footnote",      "header",         "footer",    "page_number",     "page_footnote",
        "image",         "chart",          "list",      "image_block",     "equation_block",
        "unknown"
    };
    return std::find(types.begin(), types.end(), type) != types.end();
}

inline const char * prompt_for_type(const std::string & type) {
    if (type == "[layout]") return "\nLayout Detection:";
    if (type == "table") return "\nTable Recognition:";
    if (type == "equation") return "\nFormula Recognition:";
    if (type == "image" || type == "chart") return "\nImage Analysis:";
    return "\nText Recognition:";
}

inline double area(const std::array<double, 4> & b) {
    return std::max(0.0, b[2] - b[0]) * std::max(0.0, b[3] - b[1]);
}

inline double intersection(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    return std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0])) *
           std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
}

inline double cover_ratio(const block & inner, const block & outer) {
    const double a = area(inner.bbox);
    return a > 0 ? intersection(inner.bbox, outer.bbox) / a : 0.0;
}

inline bool covered_by_type(const std::vector<block> & blocks, size_t i, const std::vector<std::string> & types,
                            double threshold = 0.9) {
    for (size_t j = 0; j < blocks.size(); ++j) {
        if (i != j && std::find(types.begin(), types.end(), blocks[j].type) != types.end() &&
            cover_ratio(blocks[i], blocks[j]) >= threshold)
            return true;
    }
    return false;
}

inline std::vector<block> parse_layout(const std::string & output) {
    // Scan bounded block segments, then apply the blueprint's exact prefix
    // grammar. Avoid regex .* over an entire potentially long generated page.
    static const std::string start = "<|box_start|>";
    static const std::regex prefix(
        R"(^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(?:(<\|rotate_(?:up|right|down|left)\|>))?)");
    std::vector<block> blocks;
    for (size_t pos = output.find(start); pos != std::string::npos;) {
        const size_t next = output.find(start, pos + start.size());
        const std::string piece = output.substr(pos, next == std::string::npos ? next : next - pos);
        pos = next;
        std::smatch m;
        if (!std::regex_search(piece, m, prefix)) continue;
        block b;
        bool valid = true;
        for (int k = 0; k < 4; ++k) {
            const std::string s = m[k + 1].str();
            // Bound accumulation before conversion: Python accepts arbitrary
            // precision but discards every coordinate outside [0,1000].
            unsigned n = 0;
            for (char c : s) {
                n = n * 10 + static_cast<unsigned>(c - '0');
                if (n > 1000) {
                    valid = false;
                    break;
                }
            }
            if (!valid) break;
            b.bbox[k] = n / 1000.0;
        }
        if (!valid) continue;
        if (b.bbox[2] < b.bbox[0]) std::swap(b.bbox[0], b.bbox[2]);
        if (b.bbox[3] < b.bbox[1]) std::swap(b.bbox[1], b.bbox[3]);
        if (area(b.bbox) == 0) continue;
        b.type = m[5].str();
        for (char & c : b.type)
            if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        if (b.type == "unknown") b.type = "image";
        if (b.type == "inline_formula" || !valid_type(b.type)) continue;
        const std::string rotate = m[6].str();
        if (rotate == "<|rotate_up|>") b.angle = 0;
        if (rotate == "<|rotate_right|>") b.angle = 90;
        if (rotate == "<|rotate_down|>") b.angle = 180;
        if (rotate == "<|rotate_left|>") b.angle = 270;
        b.merge_prev = b.type == "text" && piece.substr(m.length()).find("txt_contd_tgt") != std::string::npos;
        bool duplicate = false;
        for (const block & kept : blocks) {
            if (kept.type != b.type || kept.angle != b.angle) continue;
            const double overlap = intersection(kept.bbox, b.bbox);
            const long double sum = static_cast<long double>(area(kept.bbox)) + area(b.bbox) - overlap;
            if (sum > 0 && overlap / static_cast<double>(sum) > 0.9) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) blocks.push_back(b);
    }
    std::vector<block> filtered;
    for (size_t i = 0; i < blocks.size(); ++i) {
        const std::string & t = blocks[i].type;
        if ((t == "text" || t == "equation" || t == "equation_block") && covered_by_type(blocks, i, { "table" }))
            continue;
        filtered.push_back(blocks[i]);
    }
    return filtered;
}

inline void filter_internal_captions(std::vector<block> & blocks) {
    std::vector<block> out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i].type == "image_caption" && covered_by_type(blocks, i, { "image", "chart", "image_block" }))
            continue;
        out.push_back(blocks[i]);
    }
    blocks.swap(out);
}

inline bool should_extract(const block & b, bool image_analysis = false) {
    if (b.type == "list" || b.type == "equation_block" || b.type == "image_block") return false;
    if (b.type == "image" || b.type == "chart") {
        if (!image_analysis) return false;
        const double w = b.bbox[2] - b.bbox[0], h = b.bbox[3] - b.bbox[1];
        return (w > 0.1 && h > 0.1) || w * h > 0.01;
    }
    return true;
}

inline bool should_extract(const std::vector<block> & blocks, size_t i, bool image_analysis = false) {
    if (!should_extract(blocks.at(i), image_analysis)) return false;
    const auto & b = blocks[i];
    if ((b.type == "image" || b.type == "chart") && covered_by_type(blocks, i, { "image_block" })) return false;
    return true;
}

// PIL's crop converts float bounds using round-to-nearest, ties-to-even. This
// must not use std::round (half away from zero) or depend on the FPU mode.
inline int round_even(double x) {
    if (!std::isfinite(x) || x < std::numeric_limits<int>::min() || x > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid MinerU pixel coordinate");
    const double lo = std::floor(x), frac = x - lo;
    const int i = static_cast<int>(lo);
    return i + ((frac > 0.5 || (frac == 0.5 && i % 2 != 0)) ? 1 : 0);
}

struct crop_rect {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int width() const { return x1 - x0; }
    int height() const { return y1 - y0; }
};

inline crop_rect crop_geometry(const block & b, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("invalid MinerU image dimensions");
    for (double c : b.bbox)
        if (!std::isfinite(c) || c < 0 || c > 1) throw std::invalid_argument("invalid MinerU normalized box");
    crop_rect r;
    r.x0 = round_even(b.bbox[0] * width);
    r.x1 = round_even(b.bbox[2] * width);
    r.y0 = round_even(b.bbox[1] * height);
    r.y1 = round_even(b.bbox[3] * height);
    if (r.width() <= 0 || r.height() <= 0) throw std::invalid_argument("empty MinerU crop");
    return r;
}

struct crop_transform {
    int rotated_width = 0, rotated_height = 0;
    int padded_width = 0, padded_height = 0;
    int paste_x = 0, paste_y = 0;
    int output_width = 0, output_height = 0;
};

inline crop_transform normalize_crop_geometry(int width, int height, int angle = -1, int min_edge = 28,
                                              double max_ratio = 50.0) {
    if (width <= 0 || height <= 0 || min_edge < 1 || !std::isfinite(max_ratio) || max_ratio < 1 ||
        (angle != -1 && angle != 0 && angle != 90 && angle != 180 && angle != 270))
        throw std::invalid_argument("invalid MinerU crop transform");
    if (angle == 90 || angle == 270) std::swap(width, height);
    crop_transform p;
    p.rotated_width = width;
    p.rotated_height = height;
    if (static_cast<double>(std::max(width, height)) / std::min(width, height) > max_ratio) {
        if (width > height)
            height = static_cast<int>(std::ceil(width / max_ratio));
        else
            width = static_cast<int>(std::ceil(height / max_ratio));
    }
    p.padded_width = width;
    p.padded_height = height;
    p.paste_x = (width - p.rotated_width) / 2;
    p.paste_y = (height - p.rotated_height) / 2;
    if (std::min(width, height) < min_edge) {
        const double scale = static_cast<double>(min_edge) / std::min(width, height);
        const double nw = std::ceil(width * scale), nh = std::ceil(height * scale);
        if (nw > std::numeric_limits<int>::max() || nh > std::numeric_limits<int>::max())
            throw std::invalid_argument("MinerU crop transform overflow");
        width = static_cast<int>(nw);
        height = static_cast<int>(nh);
    }
    p.output_width = width;
    p.output_height = height;
    return p;
}

inline bool python_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
           cp == 0x3000;
}

inline std::string trim(const std::string & s) {
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

inline std::string html_escape(const std::string & s) {
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

inline std::string otsl_to_html(const std::string & input) {
    if (input.compare(0, 6, "<table") == 0 && input.size() >= 8 && input.compare(input.size() - 8, 8, "</table>") == 0)
        return input;
    static const std::regex token_re(R"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)");
    std::vector<std::string> tokens, mixed;
    size_t last = 0;
    for (std::sregex_iterator it(input.begin(), input.end(), token_re), end; it != end; ++it) {
        const size_t pos = static_cast<size_t>(it->position());
        const std::string text = input.substr(last, pos - last);
        if (!trim(text).empty()) mixed.push_back(text);
        tokens.push_back(it->str());
        mixed.push_back(it->str());
        last = pos + static_cast<size_t>(it->length());
    }
    if (!trim(input.substr(last)).empty()) mixed.push_back(input.substr(last));
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
            if (t == "<fcel>" && i + 1 < rebuilt.size() && !otsl_token(rebuilt[i + 1])) v.text = trim(rebuilt[i + 1]);
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
            html << ">" << html_escape(v.text) << "</td>";
        }
        html << "</tr>";
    }
    html << "</table>";
    return html.str();
}

inline std::string json_quote(const std::string & input) {
    static const char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : input) {
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
                out += hex[c >> 4];
                out += hex[c & 15];
            } else
                out += static_cast<char>(c);
        }
    }
    out += '"';
    return out;
}

inline std::string blocks_json(const std::vector<block> & blocks) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(15) << '[';
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto & b = blocks[i];
        if (i) out << ',';
        out << "{\"type\":" << json_quote(b.type) << ",\"bbox\":[";
        for (int k = 0; k < 4; ++k) {
            if (k) out << ',';
            out << b.bbox[k];
        }
        out << "],\"angle\":";
        if (b.angle < 0)
            out << "null";
        else
            out << b.angle;
        out << ",\"content\":" << (b.has_content ? json_quote(b.content) : "null");
        if (b.type == "text") out << ",\"merge_prev\":" << (b.merge_prev ? "true" : "false");
        out << '}';
    }
    out << ']';
    return out.str();
}

inline bool has_cjk(const std::string & text) {
    for (size_t i = 0; i + 2 < text.size(); ++i) {
        const unsigned a = static_cast<unsigned char>(text[i]);
        if ((a & 0xf0) != 0xe0) continue;
        const unsigned b = static_cast<unsigned char>(text[i + 1]), c = static_cast<unsigned char>(text[i + 2]);
        if ((b & 0xc0) != 0x80 || (c & 0xc0) != 0x80) continue;
        const unsigned cp = ((a & 15) << 12) | ((b & 63) << 6) | (c & 63);
        if ((cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff)) return true;
    }
    return false;
}

inline std::string blocks_markdown(const std::vector<block> & blocks) {
    std::vector<std::string> parts;
    size_t last_text = std::numeric_limits<size_t>::max();
    for (const auto & b : blocks) {
        if (!b.has_content || b.content.empty()) continue;
        if (b.merge_prev && last_text != std::numeric_limits<size_t>::max()) {
            if (!has_cjk(b.content)) parts[last_text] += ' ';
            parts[last_text] += b.content;
        } else {
            parts.push_back(b.content);
            if (b.type == "text") last_text = parts.size() - 1;
        }
    }
    std::string out;
    for (const auto & p : parts) {
        if (!out.empty()) out += "\n\n";
        out += p;
    }
    return out;
}

inline void post_process_basic(std::vector<block> & blocks) {
    // Equivalent to the relevant image_analysis=False simple_process steps.
    // Does not claim the Python full LaTeX repair or table-image masking stack.
    std::vector<block> out;
    for (auto b : blocks) {
        if (b.type == "table" && b.has_content && !b.content.empty()) b.content = otsl_to_html(b.content);
        if (b.type == "list_item") {
            b.type = "text";
            b.merge_prev = false;
        }
        if (b.has_content && trim(b.content) == "[Non-Text]") {
            if (b.type == "header" || b.type == "footer")
                b.content.clear();
            else if (b.type != "image" && b.type != "chart" && b.type != "table" && b.type != "equation" &&
                     b.type != "list" && b.type != "image_block" && b.type != "equation_block")
                continue;
        }
        out.push_back(b);
    }
    blocks.swap(out);
}

} // namespace mineru
