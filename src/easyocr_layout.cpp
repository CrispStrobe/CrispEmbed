#include "easyocr_layout.h"

#include <algorithm>
#include <cmath>

namespace easyocr_layout {

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
