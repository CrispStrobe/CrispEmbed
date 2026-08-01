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

// Detector geometry independent of a particular detector implementation.
struct region {
    float x = 0, y = 0, w = 0, h = 0;
    float score = 0;
    int line = 0;
};

enum class ordering_mode {
    lines,
    words,
};

// EasyOCR-style y grouping into line crops.
std::vector<region> group_lines(const std::vector<region> & regions);

// DBNet emits fragmented horizontal regions on the current IC15 artifact.
// This adapter deliberately groups by y-band without EasyOCR's horizontal-gap
// split, preserving a complete line crop for the downstream CRNN.
std::vector<region> group_dbnet_lines(const std::vector<region> & regions);

// Tesseract/LayoutLM-style y-band grouping with left-to-right word order.
std::vector<region> order_words(const std::vector<region> & regions);

// Select one of the two downstream-compatible ordering policies.
std::vector<region> order_regions(const std::vector<region> & regions, ordering_mode mode);

std::vector<word> reading_order(std::vector<word> words);
std::vector<normalized_box> normalize_boxes(const std::vector<word> & words, int image_width, int image_height);
std::string join_lines(const std::vector<word> & words);

} // namespace easyocr_layout
