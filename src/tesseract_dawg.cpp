#include "tesseract_dawg.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

void set_error(char * error, size_t error_size, const char * message) {
    if (error && error_size) {
        std::strncpy(error, message, error_size - 1);
        error[error_size - 1] = '\0';
    }
}

bool decode_base64(const char * input, std::vector<uint8_t> & output) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int value = 0;
    int bits = -8;
    for (const unsigned char * p = (const unsigned char *)input; *p; ++p) {
        if (*p == '=') break;
        const char * found = std::strchr(alphabet, *p);
        if (!found) return false;
        value = (value << 6) | (int)(found - alphabet);
        bits += 6;
        if (bits >= 0) {
            output.push_back((uint8_t)((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    return true;
}

uint16_t read_u16(const std::vector<uint8_t> & data, size_t offset) {
    return (uint16_t)data[offset] | ((uint16_t)data[offset + 1] << 8);
}

uint32_t read_u32(const std::vector<uint8_t> & data, size_t offset) {
    return (uint32_t)data[offset] | ((uint32_t)data[offset + 1] << 8) | ((uint32_t)data[offset + 2] << 16) |
           ((uint32_t)data[offset + 3] << 24);
}

uint64_t read_u64(const std::vector<uint8_t> & data, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= (uint64_t)data[offset + i] << (8 * i);
    return value;
}

int ceil_log2(uint32_t value) {
    int bits = 0;
    uint32_t limit = 1;
    while (limit < value) {
        limit <<= 1;
        ++bits;
    }
    return bits;
}

} // namespace

extern "C" int tesseract_dawg_validate_base64(const char * payload, char * error, size_t error_size) {
    set_error(error, error_size, "");
    if (!payload || !*payload) {
        set_error(error, error_size, "empty payload");
        return 0;
    }

    std::vector<uint8_t> data;
    if (!decode_base64(payload, data)) {
        set_error(error, error_size, "invalid base64");
        return 0;
    }
    if (data.size() < 10) {
        set_error(error, error_size, "truncated header");
        return 0;
    }
    if (read_u16(data, 0) != 42) {
        set_error(error, error_size, "bad magic");
        return 0;
    }
    const uint32_t unicharset_size = read_u32(data, 2);
    const uint32_t num_edges = read_u32(data, 6);
    if (unicharset_size == 0 || num_edges == 0 || num_edges > 50000000u) {
        set_error(error, error_size, "invalid DAWG dimensions");
        return 0;
    }
    if ((size_t)num_edges > (data.size() - 10) / sizeof(uint64_t)) {
        set_error(error, error_size, "edge array exceeds payload");
        return 0;
    }

    const int flag_start = ceil_log2(unicharset_size);
    const int next_start = flag_start + 3;
    if (next_start >= 64) {
        set_error(error, error_size, "invalid edge bit layout");
        return 0;
    }
    const uint64_t next_mask = ~0ull << next_start;
    const uint64_t marker = 1ull << flag_start;
    const uint64_t direction = 2ull << flag_start;
    const uint64_t empty = next_mask;

    for (uint32_t i = 0; i < num_edges; ++i) {
        const uint64_t edge = read_u64(data, 10 + (size_t)i * 8);
        if (edge == empty) continue;
        const uint64_t next = (edge & next_mask) >> next_start;
        if (next >= num_edges) {
            set_error(error, error_size, "edge next-node out of bounds");
            return 0;
        }
        if ((edge & direction) == 0) {
            bool terminated = false;
            for (uint32_t j = i; j < num_edges; ++j) {
                const uint64_t run_edge = read_u64(data, 10 + (size_t)j * 8);
                if (run_edge == empty || (run_edge & direction) != 0) {
                    set_error(error, error_size, "unterminated forward edge run");
                    return 0;
                }
                const uint64_t run_next = (run_edge & next_mask) >> next_start;
                if (run_next >= num_edges) {
                    set_error(error, error_size, "forward edge next-node out of bounds");
                    return 0;
                }
                if (run_edge & marker) {
                    terminated = true;
                    break;
                }
            }
            if (!terminated) {
                set_error(error, error_size, "forward edge run has no marker");
                return 0;
            }
        }
    }
    return 1;
}

extern "C" int tesseract_dawg_contains_base64(const char * payload, const int * unichar_ids, size_t count) {
    if (!payload || !unichar_ids || count == 0 || !tesseract_dawg_validate_base64(payload, nullptr, 0)) return 0;

    std::vector<uint8_t> data;
    if (!decode_base64(payload, data)) return 0;
    const uint32_t unicharset_size = read_u32(data, 2);
    const uint32_t num_edges = read_u32(data, 6);
    const int flag_start = ceil_log2(unicharset_size);
    const int next_start = flag_start + 3;
    const uint64_t next_mask = ~0ull << next_start;
    const uint64_t marker = 1ull << flag_start;
    const uint64_t direction = 2ull << flag_start;
    const uint64_t word_end = 4ull << flag_start;
    const uint64_t letter_mask = ~(~0ull << flag_start);

    uint32_t node = 0;
    for (size_t pos = 0; pos < count; ++pos) {
        bool found = false;
        uint64_t selected = 0;
        for (uint32_t edge_index = node; edge_index < num_edges; ++edge_index) {
            const uint64_t edge = read_u64(data, 10 + (size_t)edge_index * 8);
            if ((edge & next_mask) == next_mask || (edge & direction) != 0) break;
            if ((int)(edge & letter_mask) == unichar_ids[pos]) {
                selected = edge;
                found = true;
                break;
            }
            if (edge & marker) break;
        }
        if (!found) return 0;
        node = (uint32_t)((selected & next_mask) >> next_start);
        if (pos + 1 == count) return (selected & word_end) != 0;
    }
    return 0;
}
