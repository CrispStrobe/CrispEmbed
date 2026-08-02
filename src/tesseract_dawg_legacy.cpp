#include "tesseract_dawg.h"

#include <algorithm>

namespace tesseract_dawg {
namespace {

constexpr uint16_t kMagic = 42;
constexpr uint32_t kMaxEdges = 50000000;
constexpr uint64_t kMarkerFlag = 1;
constexpr uint64_t kDirectionFlag = 2;
constexpr uint64_t kWordEndFlag = 4;

void fail(std::string * error, const char * message) {
    if (error) *error = message;
}

uint16_t u16(const std::vector<uint8_t> & bytes, size_t & pos) {
    if (pos + 2 > bytes.size()) return 0;
    uint16_t value = (uint16_t)bytes[pos] | ((uint16_t)bytes[pos + 1] << 8);
    pos += 2;
    return value;
}

uint32_t u32(const std::vector<uint8_t> & bytes, size_t & pos) {
    if (pos + 4 > bytes.size()) return 0;
    uint32_t value = (uint32_t)bytes[pos] | ((uint32_t)bytes[pos + 1] << 8) | ((uint32_t)bytes[pos + 2] << 16) |
                     ((uint32_t)bytes[pos + 3] << 24);
    pos += 4;
    return value;
}

uint64_t u64(const std::vector<uint8_t> & bytes, size_t & pos) {
    if (pos + 8 > bytes.size()) return 0;
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= (uint64_t)bytes[pos + i] << (8 * i);
    pos += 8;
    return value;
}

int ceil_log2(uint32_t value) {
    int bits = 0;
    while (value > 0) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

} // namespace

bool parse(const std::vector<uint8_t> & bytes, Dawg & out, std::string * error) {
    out = {};
    if (error) error->clear();
    size_t pos = 0;
    const uint16_t magic = u16(bytes, pos);
    const uint32_t unicharset_size = u32(bytes, pos);
    const uint32_t num_edges = u32(bytes, pos);
    if (bytes.size() < 10) {
        fail(error, "truncated dawg header");
        return false;
    }
    if (magic != kMagic) {
        fail(error, "invalid dawg magic");
        return false;
    }
    if (unicharset_size == 0) {
        fail(error, "invalid dawg unicharset size");
        return false;
    }
    const int flag_start = ceil_log2(unicharset_size);
    const int next_start = flag_start + 3;
    if (flag_start <= 0 || next_start >= 64) {
        fail(error, "unsupported dawg unicharset size");
        return false;
    }
    if (num_edges == 0 || num_edges > kMaxEdges || num_edges > (bytes.size() - pos) / sizeof(uint64_t)) {
        fail(error, "invalid dawg edge count");
        return false;
    }
    out.unicharset_size = unicharset_size;
    out.edges.resize(num_edges);
    for (uint64_t & edge : out.edges) edge = u64(bytes, pos);
    const uint64_t next_mask = ~((1ull << next_start) - 1);
    const uint64_t direction = kDirectionFlag << flag_start;
    const uint64_t marker = kMarkerFlag << flag_start;
    for (size_t i = 0; i < out.edges.size(); ++i) {
        const uint64_t edge = out.edges[i];
        if (edge == next_mask) continue;
        if (((edge & next_mask) >> next_start) >= out.edges.size()) {
            out = {};
            fail(error, "dawg edge points outside edge array");
            return false;
        }
        if ((edge & direction) != 0) continue;
        bool terminated = false;
        for (size_t j = i; j < out.edges.size(); ++j) {
            const uint64_t run_edge = out.edges[j];
            if (run_edge == next_mask || (run_edge & direction) != 0) break;
            if (((run_edge & next_mask) >> next_start) >= out.edges.size()) {
                out = {};
                fail(error, "dawg forward edge points outside edge array");
                return false;
            }
            if ((run_edge & marker) != 0) {
                terminated = true;
                break;
            }
        }
        if (!terminated) {
            out = {};
            fail(error, "dawg forward edge run is unterminated");
            return false;
        }
    }
    return true;
}

bool prefix_matches(const Dawg & dawg, const std::vector<int> & unichars, bool complete) {
    if (unichars.empty()) return !complete;
    if (dawg.unicharset_size == 0 || dawg.edges.empty()) return false;
    const int flag_start = ceil_log2(dawg.unicharset_size);
    const int next_start = flag_start + 3;
    const uint64_t marker = kMarkerFlag << flag_start;
    const uint64_t direction = kDirectionFlag << flag_start;
    const uint64_t eow = kWordEndFlag << flag_start;
    const uint64_t next_mask = ~((1ull << next_start) - 1);
    const uint64_t letter_mask = (1ull << flag_start) - 1;
    size_t node = 0;
    for (size_t index = 0; index < unichars.size(); ++index) {
        if (node >= dawg.edges.size()) return false;
        bool found = false;
        for (size_t edge_index = node; edge_index < dawg.edges.size(); ++edge_index) {
            const uint64_t edge = dawg.edges[edge_index];
            if (edge == next_mask || (edge & direction) != 0) break;
            if ((int)(edge & letter_mask) == unichars[index]) {
                if (complete && index + 1 == unichars.size() && (edge & eow) == 0) return false;
                node = (size_t)((edge & next_mask) >> next_start);
                found = true;
                break;
            }
            if (edge & marker) break;
        }
        if (!found) return false;
        if (index + 1 < unichars.size() && node == 0) return false;
    }
    return true;
}

} // namespace tesseract_dawg
