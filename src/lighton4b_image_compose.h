#pragma once
#include "qwen35_frontend.h"
#include "qwen35_positions.h"
#include "qwen35_vision.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace qwen35_ocr {
inline bool lighton4b_image_enabled() {
    return core_env::on("CRISPEMBED_QWEN35_OCR") && core_env::on("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE");
}
// Still-image component for the pinned Qwen3VLProcessor, P16/T2/M2,
// min/max pixels 65536/16777216. Native execution remains bounded to1024
// patches; this does not admit the original processor's entire size range.
inline bool lighton4b_preprocess_rgb(const uint8_t * rgb, int height, int width, image_inputs & output) {
    output = {};
    return lighton4b_image_enabled() && preprocess_rgb(rgb, height, width, image_config{}, output);
}
// Use the config produced by map_lighton4b_vision. No payload ownership or
// checkpoint admission is inferred from a shape match. Caller owns weights.
inline bool lighton4b_image_vision_cpu(const vision_weights & weights, const vision_config & config,
                                       const image_inputs & image, int threads, vision_outputs & output) {
    output = {};
    if (!lighton4b_image_enabled() || !core_env::on("CRISPEMBED_LIGHTON4B_VISION_OWNER") || config.width != 1024 ||
        config.heads != 16 || config.intermediate != 4096 || config.output != 2560 || config.patch_width != 1536 ||
        config.merge != 2 || config.position_side != 48 || config.eps != 1e-6f || config.theta != 10000.0f ||
        weights.layers.size() != 24 || image.grid.size() != 3 || image.grid[0] != 1 || image.grid[1] < 2 ||
        image.grid[2] < 2 || image.grid[1] > 1024 || image.grid[2] > 1024 || image.grid[1] % 2 || image.grid[2] % 2 ||
        image.height != image.grid[1] * 16 || image.width != image.grid[2] * 16)
        return false;
    const size_t tokens = size_t(image.grid[1]) * image.grid[2];
    if (tokens > 1024 || image.patches.size() != tokens * 1536 ||
        !std::all_of(image.patches.begin(), image.patches.end(), [](float v) { return std::isfinite(v); }))
        return false;
    vision_outputs candidate;
    if (!vision_cpu(weights, config, image.patches.data(), int(tokens), image.grid, threads, candidate) ||
        candidate.merged.size() != tokens / 4 * 2560)
        return false;
    output = std::move(candidate);
    return true;
}
struct lighton4b_image_prefill {
    std::vector<float> hidden;
    position_outputs positions;
};
// Splice into the *native* token embedding result. Tokenizer/token-table
// provenance is the caller's responsibility: this helper accepts no reference
// hidden vectors or source-array files. Modalities must describe one still image.
inline bool lighton4b_image_prefill_cpu(const std::vector<float> & token_embeddings,
                                        const std::vector<int> & modalities, const std::vector<int> & grid,
                                        const std::vector<float> & image_embeddings, lighton4b_image_prefill & output) {
    output = {};
    if (!lighton4b_image_enabled() || modalities.empty() || modalities.size() > 4096 || grid.size() != 3 ||
        grid[0] != 1 || grid[1] < 2 || grid[2] < 2 || grid[1] > 1024 || grid[2] > 1024 || grid[1] % 2 || grid[2] % 2 ||
        size_t(grid[1]) * grid[2] > 1024 || token_embeddings.size() != modalities.size() * 2560 ||
        image_embeddings.size() != size_t(grid[1] / 2) * (grid[2] / 2) * 2560 ||
        !std::all_of(token_embeddings.begin(), token_embeddings.end(), [](float v) { return std::isfinite(v); }) ||
        !std::all_of(image_embeddings.begin(), image_embeddings.end(), [](float v) { return std::isfinite(v); }))
        return false;
    lighton4b_image_prefill candidate;
    if (!image_positions(modalities, grid, 2, candidate.positions)) return false;
    candidate.hidden = token_embeddings;
    size_t next = 0;
    for (size_t i = 0; i < modalities.size(); ++i)
        if (modalities[i] == 1) {
            if (next + 2560 > image_embeddings.size()) return false;
            std::copy_n(image_embeddings.data() + next, 2560, candidate.hidden.data() + i * 2560);
            next += 2560;
        }
    if (next != image_embeddings.size()) return false;
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
