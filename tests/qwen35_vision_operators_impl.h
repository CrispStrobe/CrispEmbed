#pragma once
#ifndef CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT
#error "Vision operator replay is validation-only"
#endif
// Include the actual implementation in this test translation unit. Rename only
// its externally linked entry points so the ordinary production implementation
// remains in the linked library with its original compilation context.
#include "qwen35_vision.h"
#include "qwen35_math.h"
#define vision_cpu vision_audit_copy_cpu
#define map_small_vision vision_audit_copy_map
#include "../src/qwen35_vision.cpp"
#undef map_small_vision
#undef vision_cpu
namespace qwen35_ocr {
static float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
static float gelu_erf(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}
bool vision_test_operator_cpu(const vision_weights & w, const vision_config & c, int index, const char * operation,
                              const std::vector<float> & input, const std::vector<int> & grid,
                              const std::vector<float> & cosine, const std::vector<float> & sine, int threads,
                              std::vector<float> & output) {
    output.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_QWEN35_VISION_PROBES") || !operation ||
        input.empty() || !finite(input) || grid.size() != 3 || grid[0] != 1 || grid[1] < 1 || grid[2] < 1 ||
        grid[1] > 1024 || grid[2] > 1024 || c.width < 1 || c.width > 2048 || c.heads < 1 || c.heads > 32 ||
        c.width % c.heads || (c.width / c.heads) % 4 || c.patch_width < 1 || c.patch_width > 4096 ||
        c.intermediate < 1 || c.intermediate > 8192 || c.merge < 1 || c.merge > 4 || threads < 1 || threads > 256)
        return false;
    const int tokens = grid[1] * grid[2], width = c.width, group = c.merge * c.merge;
    std::vector<int> rows, columns, offsets, heights, widths;
    if (tokens > 1024 || !geometry(c, tokens, grid, rows, columns, offsets, heights, widths)) return false;
    const std::string op(operation);
    int expected_width = width;
    if (op == "patch_embed") expected_width = c.patch_width;
    if (op == "pooled") expected_width = 3 * width;
    if (op == "activation" || op == "down") expected_width = c.intermediate;
    if (op == "merger.up" || op == "merger.activation" || op == "merger") expected_width = width;
    if (input.size() != size_t(tokens) * expected_width) return false;
    if (op == "patch_embed") return linear(w.patch, w.patch_bias, input, c.patch_width, tokens, threads, output);
    if (op == "merger.norm") return layer_norm(input, w.merger_norm, w.merger_norm_bias, width, c.eps, output);
    if (op == "merger.up")
        return linear(w.merger_up, w.merger_up_bias, input, width * group, tokens / group, threads, output);
    if (op == "merger.activation") {
        output = input;
        for (float & x : output) x = gelu_erf(x);
        return finite(output);
    }
    if (op == "merger")
        return linear(w.merger_down, w.merger_down_bias, input, width * group, tokens / group, threads, output);
    if (index < 0 || size_t(index) >= w.layers.size()) return false;
    const auto & layer = w.layers[index];
    if (op == "norm1") return layer_norm(input, layer.norm1, layer.norm1_bias, width, c.eps, output);
    if (op == "norm2") return layer_norm(input, layer.norm2, layer.norm2_bias, width, c.eps, output);
    if (op == "qkv") return linear(layer.qkv, layer.qkv_bias, input, width, tokens, threads, output);
    if (op == "attention") return linear(layer.out, layer.out_bias, input, width, tokens, threads, output);
    if (op == "up") return linear(layer.up, layer.up_bias, input, width, tokens, threads, output);
    if (op == "down") return linear(layer.down, layer.down_bias, input, c.intermediate, tokens, threads, output);
    if (op == "activation") {
        output = input;
        for (float & x : output) x = gelu_tanh(x);
        return finite(output);
    }
    if (op == "pooled") {
        if (cosine.size() != size_t(tokens) * (width / c.heads) || sine.size() != cosine.size() || !finite(cosine) ||
            !finite(sine))
            return false;
        output = attention(input, c, tokens, offsets, cosine, sine);
        return finite(output);
    }
    return false;
}
} // namespace qwen35_ocr
