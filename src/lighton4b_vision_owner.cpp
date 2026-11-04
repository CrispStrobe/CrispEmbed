#include "qwen35_vision.h"
#include "core/env_gate.h"
#include <string>
#include <utility>

namespace qwen35_ocr {
static bool shape(ggml_tensor * t, int columns, int rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
// Explicit ownership component only: format-5 frontend/recognition remains closed.
bool map_lighton4b_vision(const core_gguf::tensor_map & tensors, vision_weights & weights, vision_config & config) {
    weights = {};
    config = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_LIGHTON4B_VISION_OWNER")) return false;
    vision_weights candidate;
    auto get = [&](const std::string & name, int columns, int rows) -> ggml_tensor * {
        auto * t = core_gguf::try_get(tensors, name.c_str());
        return shape(t, columns, rows) && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) ? t : nullptr;
    };
    candidate.patch = get("v.patch_embed.proj.weight", 1536, 1024);
    candidate.patch_bias = get("v.patch_embed.proj.bias", 1024, 1);
    candidate.position = get("v.pos_embed.weight", 1024, 2304);
    candidate.merger_norm = get("v.merger.norm.weight", 1024, 1);
    candidate.merger_norm_bias = get("v.merger.norm.bias", 1024, 1);
    candidate.merger_up = get("v.merger.linear_fc1.weight", 4096, 4096);
    candidate.merger_up_bias = get("v.merger.linear_fc1.bias", 4096, 1);
    candidate.merger_down = get("v.merger.linear_fc2.weight", 4096, 2560);
    candidate.merger_down_bias = get("v.merger.linear_fc2.bias", 2560, 1);
    if (!candidate.patch || !candidate.patch_bias || !candidate.position || !candidate.merger_norm ||
        !candidate.merger_norm_bias || !candidate.merger_up || !candidate.merger_up_bias || !candidate.merger_down ||
        !candidate.merger_down_bias)
        return false;
    size_t count = 0;
    for (const auto & item : tensors)
        if (item.first.rfind("v.", 0) == 0) ++count;
    if (count != 297) return false;
    for (int i = 0; i < 24; ++i) {
        const auto prefix = "v.blocks." + std::to_string(i) + ".";
        auto field = [&](const char * name, int columns, int rows) { return get(prefix + name, columns, rows); };
        vision_layer_weights l{ field("norm1.weight", 1024, 1),
                                field("norm1.bias", 1024, 1),
                                field("norm2.weight", 1024, 1),
                                field("norm2.bias", 1024, 1),
                                field("attn.qkv.weight", 1024, 3072),
                                field("attn.qkv.bias", 3072, 1),
                                field("attn.proj.weight", 1024, 1024),
                                field("attn.proj.bias", 1024, 1),
                                field("mlp.linear_fc1.weight", 1024, 4096),
                                field("mlp.linear_fc1.bias", 4096, 1),
                                field("mlp.linear_fc2.weight", 4096, 1024),
                                field("mlp.linear_fc2.bias", 1024, 1) };
        if (!l.norm1 || !l.norm1_bias || !l.norm2 || !l.norm2_bias || !l.qkv || !l.qkv_bias || !l.out || !l.out_bias ||
            !l.up || !l.up_bias || !l.down || !l.down_bias)
            return false;
        candidate.layers.push_back(l);
    }
    config.width = 1024;
    config.heads = 16;
    config.intermediate = 4096;
    config.output = 2560;
    weights = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
