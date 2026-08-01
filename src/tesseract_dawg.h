#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tesseract_dawg {

struct Dawg {
    uint32_t unicharset_size = 0;
    std::vector<uint64_t> edges;
};

// Parse one Tesseract SquishedDawg component. The parser is bounded and
// validates node references and terminated forward-edge runs.
bool parse(const std::vector<uint8_t> & bytes, Dawg & out, std::string * error = nullptr);

// Check a sequence of unicharset IDs against the graph. If complete is false,
// a valid prefix is accepted even when its final edge is not a word ending.
bool prefix_matches(const Dawg & dawg, const std::vector<int> & unichars, bool complete);

} // namespace tesseract_dawg
