#pragma once
#include "bpe.h"
#include "unicode_nfc_data.h"
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
namespace core_nfc {
inline unsigned combining_class(uint32_t cp) {
    const auto it = std::lower_bound(std::begin(classes), std::end(classes), cp,
                                     [](const combining & row, uint32_t value) { return row.cp < value; });
    return it != std::end(classes) && it->cp == cp ? it->value : 0;
}
inline void decompose(uint32_t cp, std::vector<uint32_t> & out) {
    if (cp >= 0xac00 && cp < 0xac00 + 11172) {
        const uint32_t index = cp - 0xac00;
        out.push_back(0x1100 + index / 588);
        out.push_back(0x1161 + (index % 588) / 28);
        if (index % 28) out.push_back(0x11a7 + index % 28);
        return;
    }
    const auto it = std::lower_bound(std::begin(decompositions), std::end(decompositions), cp,
                                     [](const decomposition & row, uint32_t value) { return row.cp < value; });
    if (it == std::end(decompositions) || it->cp != cp) {
        out.push_back(cp);
        return;
    }
    decompose(it->first, out);
    if (it->second) decompose(it->second, out);
}
inline uint32_t compose(uint32_t first, uint32_t second) {
    if (first >= 0x1100 && first < 0x1100 + 19 && second >= 0x1161 && second < 0x1161 + 21)
        return 0xac00 + ((first - 0x1100) * 21 + (second - 0x1161)) * 28;
    if (first >= 0xac00 && first < 0xac00 + 11172 && (first - 0xac00) % 28 == 0 && second > 0x11a7 &&
        second < 0x11a7 + 28)
        return first + second - 0x11a7;
    const uint64_t pair = (uint64_t(first) << 21) | second;
    const auto it = std::lower_bound(std::begin(compositions), std::end(compositions), pair,
                                     [](const composition & row, uint64_t value) { return row.pair < value; });
    return it != std::end(compositions) && it->pair == pair ? it->cp : 0;
}
// Canonical decomposition/order/composition, including algorithmic Hangul.
// Unicode9 matches tokenizers unicode-normalization-alignments 0.1.12.
// Only the opt-in Qwen3.5 frontend calls this helper; no shared tokenizer default changes.
inline std::string normalize(const std::string & input) {
    std::vector<uint32_t> parts;
    for (size_t i = 0; i < input.size();) {
        size_t length = 0;
        const uint32_t cp = core_bpe::utf8_decode_at(input, i, length);
        decompose(cp, parts);
        i += length;
    }
    if (parts.empty()) return {};
    size_t begin = 0;
    for (size_t i = 0; i <= parts.size(); ++i)
        if (i == parts.size() || combining_class(parts[i]) == 0) {
            std::stable_sort(parts.begin() + begin, parts.begin() + i,
                             [](uint32_t a, uint32_t b) { return combining_class(a) < combining_class(b); });
            begin = i + 1;
        }
    std::vector<uint32_t> result{ parts[0] };
    size_t starter = 0;
    unsigned previous = combining_class(parts[0]);
    for (size_t i = 1; i < parts.size(); ++i) {
        const auto cp = parts[i];
        const unsigned current = combining_class(cp);
        const uint32_t joined = (previous < current || previous == 0) ? compose(result[starter], cp) : 0;
        if (joined)
            result[starter] = joined;
        else {
            if (current == 0) starter = result.size();
            result.push_back(cp);
            previous = current;
        }
    }
    std::string output;
    for (uint32_t cp : result) core_bpe::utf8_encode(cp, output);
    return output;
}
} // namespace core_nfc
