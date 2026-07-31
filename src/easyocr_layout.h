#pragma once

#include <string>
#include <vector>

namespace easyocr_layout {

struct word {
    std::string text;
    float x = 0, y = 0, w = 0, h = 0;
    float confidence = 0;
    int block = 0, line = 0, index = 0;
};

struct normalized_box {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

std::vector<word> reading_order(std::vector<word> words);
std::vector<normalized_box> normalize_boxes(const std::vector<word> & words, int image_width, int image_height);
std::string join_lines(const std::vector<word> & words);

} // namespace easyocr_layout
