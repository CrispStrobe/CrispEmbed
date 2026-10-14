#include "qwen35_positions.h"
#include "core/env_gate.h"
#include <algorithm>
#include <utility>

namespace qwen35_ocr {
bool image_positions(const std::vector<int> & modalities, const std::vector<int> & grid_thw, int spatial_merge,
                     position_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || modalities.empty() || modalities.size() > 4096 ||
        grid_thw.size() % 3 || spatial_merge < 1 || spatial_merge > 16)
        return false;
    const size_t tokens = modalities.size();
    position_outputs result;
    result.axes.resize(3 * tokens);
    size_t grid = 0;
    int current = 0, maximum = -1;
    for (size_t start = 0; start < tokens;) {
        size_t end = start + 1;
        while (end < tokens && modalities[end] == modalities[start]) ++end;
        if (modalities[start] == 0) {
            for (size_t i = start; i < end; ++i) {
                const int position = current + int(i - start);
                for (int axis = 0; axis < 3; ++axis) result.axes[axis * tokens + i] = float(position);
                maximum = std::max(maximum, position);
            }
            current += int(end - start);
        } else if (modalities[start] == 1) {
            if (grid + 3 > grid_thw.size()) return false;
            const int t = grid_thw[grid], h = grid_thw[grid + 1], w = grid_thw[grid + 2];
            if (t != 1 || h < spatial_merge || w < spatial_merge || h > 1024 || w > 1024 || h % spatial_merge ||
                w % spatial_merge)
                return false;
            const int rows = h / spatial_merge, cols = w / spatial_merge;
            if (size_t(rows) * cols != end - start) return false;
            for (size_t i = start; i < end; ++i) {
                result.axes[i] = float(current);
                result.axes[tokens + i] = float(current + int(i - start) / cols);
                result.axes[2 * tokens + i] = float(current + int(i - start) % cols);
            }
            maximum = std::max(maximum, current + std::max(rows, cols) - 1);
            current += std::max(rows, cols);
            grid += 3;
        } else {
            return false;
        }
        start = end;
    }
    if (grid != grid_thw.size()) return false;
    result.rope_delta = maximum + 1 - int(tokens);
    result.generation.resize(4 * tokens);
    for (size_t i = 0; i < tokens; ++i) result.generation[i] = float(i);
    std::copy(result.axes.begin(), result.axes.end(), result.generation.begin() + tokens);
    output = std::move(result);
    return true;
}

bool cached_positions(int cache_tokens, int tokens, int rope_delta, std::vector<float> & axes) {
    axes.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || cache_tokens < 1 || cache_tokens > 4096 || tokens < 1 ||
        tokens > 4096 - cache_tokens || rope_delta < -4096 || rope_delta > 4096 || cache_tokens + rope_delta < 0)
        return false;
    axes.resize(3 * size_t(tokens));
    for (int axis = 0; axis < 3; ++axis)
        for (int i = 0; i < tokens; ++i) axes[axis * size_t(tokens) + i] = float(cache_tokens + i + rope_delta);
    return true;
}
} // namespace qwen35_ocr
