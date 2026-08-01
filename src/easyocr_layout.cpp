#include "easyocr_layout.h"

#include <algorithm>
#include <cmath>

namespace easyocr_layout {

std::vector<region> group_lines(const std::vector<region> & input) {
    std::vector<region> sorted = input;
    std::sort(sorted.begin(), sorted.end(),
              [](const region & a, const region & b) { return a.y + a.h * 0.5f < b.y + b.h * 0.5f; });
    std::vector<region> lines;
    for (const auto & b : sorted) {
        const float cy = b.y + b.h * 0.5f;
        int match = -1;
        for (int i = 0; i < (int)lines.size(); ++i) {
            const float line_cy = lines[i].y + lines[i].h * 0.5f;
            if (std::fabs(cy - line_cy) <= 0.5f * std::max(lines[i].h, b.h)) {
                match = i;
                break;
            }
        }
        if (match < 0) {
            lines.push_back(b);
        } else {
            auto & line = lines[match];
            const float x1 = std::max(line.x + line.w, b.x + b.w);
            const float y1 = std::max(line.y + line.h, b.y + b.h);
            line.x = std::min(line.x, b.x);
            line.y = std::min(line.y, b.y);
            line.w = x1 - line.x;
            line.h = y1 - line.y;
            line.score = 0.5f * (line.score + b.score);
        }
    }
    std::sort(lines.begin(), lines.end(), [](const region & a, const region & b) { return a.y < b.y; });
    for (int i = 0; i < (int)lines.size(); ++i) lines[i].line = i;
    return lines;
}

std::vector<region> group_dbnet_lines(const std::vector<region> & input) {
    return group_lines(input);
}

std::vector<region> order_words(const std::vector<region> & input) {
    std::vector<std::vector<region>> groups;
    for (const auto & b : input) {
        const float cy = b.y + b.h * 0.5f;
        int match = -1;
        for (int i = 0; i < (int)groups.size(); ++i) {
            float y0 = groups[i].front().y, y1 = groups[i].front().y + groups[i].front().h;
            for (const auto & item : groups[i]) {
                y0 = std::min(y0, item.y);
                y1 = std::max(y1, item.y + item.h);
            }
            if (std::fabs(cy - (y0 + y1) * 0.5f) <= 0.5f * std::max(y1 - y0, b.h)) {
                match = i;
                break;
            }
        }
        if (match < 0)
            groups.push_back({ b });
        else
            groups[match].push_back(b);
    }
    std::sort(groups.begin(), groups.end(), [](const auto & a, const auto & b) { return a.front().y < b.front().y; });
    std::vector<region> ordered;
    for (int line = 0; line < (int)groups.size(); ++line) {
        auto & group = groups[line];
        std::sort(group.begin(), group.end(), [](const region & a, const region & b) { return a.x < b.x; });
        for (auto & item : group) item.line = line;
        ordered.insert(ordered.end(), group.begin(), group.end());
    }
    return ordered;
}

std::vector<region> order_regions(const std::vector<region> & regions, ordering_mode mode) {
    return mode == ordering_mode::lines ? group_lines(regions) : order_words(regions);
}

std::vector<word> reading_order(std::vector<word> words) {
    std::stable_sort(words.begin(), words.end(), [](const word & a, const word & b) {
        if (a.block != b.block) return a.block < b.block;
        if (a.line != b.line) return a.line < b.line;
        if (std::fabs(a.y - b.y) > 0.5f * std::max(a.h, b.h)) return a.y < b.y;
        return a.x < b.x;
    });
    return words;
}

std::vector<normalized_box> normalize_boxes(const std::vector<word> & words, int image_width, int image_height) {
    std::vector<normalized_box> out;
    out.reserve(words.size());
    for (const auto & w : words) {
        auto scale = [](float v, float size) { return std::clamp((int)std::lround(1000.0f * v / size), 0, 1000); };
        out.push_back({ scale(w.x, image_width), scale(w.y, image_height), scale(w.x + w.w, image_width),
                        scale(w.y + w.h, image_height) });
    }
    return out;
}

std::string join_lines(const std::vector<word> & words) {
    std::string out;
    int last_line = -1, last_block = -1;
    for (const auto & w : words) {
        if (!out.empty()) {
            if (w.block != last_block)
                out += "\n\n";
            else if (w.line != last_line)
                out += '\n';
            else
                out += ' ';
        }
        out += w.text;
        last_block = w.block;
        last_line = w.line;
    }
    return out;
}

} // namespace easyocr_layout
