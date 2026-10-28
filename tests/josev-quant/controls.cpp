#include "josev2b_backbone.h"
#include "josev2b_packed_profile.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
namespace {
struct shape_record {
    const char * name;
    int64_t ne[4];
};
#include "backbone-shapes.h"
void require(bool value, const char * why) {
    if (!value) throw std::runtime_error(why);
}
void flag(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value)
        setenv(key, value, 1);
    else
        unsetenv(key);
#endif
}
struct fixture {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ ggml_init({ 2 * 1024 * 1024, nullptr, true }), ggml_free };
    core_gguf::tensor_map tensors;
    explicit fixture(bool quantized, bool bad_embedding = false) {
        require(bool(ctx), "fixture metadata allocation");
        for (const auto & row : SHAPES) {
            ggml_type type = quantized && josev2b_packed::projection(row.name) ? GGML_TYPE_Q4_K : GGML_TYPE_F32;
            if (bad_embedding && std::strcmp(row.name, "l.embed_tokens.weight") == 0) type = GGML_TYPE_Q8_0;
            auto * t = ggml_new_tensor_4d(ctx.get(), type, row.ne[0], row.ne[1], row.ne[2], row.ne[3]);
            ggml_set_name(t, row.name);
            tensors.emplace(row.name, t);
        }
    }
};
int run() {
    for (auto * key : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_JOSEV_BACKBONE", "CRISPEMBED_QWEN35_QUANT" })
        flag(key, "1");
    fixture f32(false), quantized(true), bad(true, true);
    qwen35_ocr::decoder_weights w;
    require(josev2b_backbone::map(f32.tensors, w), "legacy F32 map remains valid");
    require(!josev2b_backbone::map(quantized.tensors, w) && w.layers.empty(), "legacy map rejects Q4_K and clears");
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        flag("CRISPEMBED_JOSEV_QUANT", value);
        require(!josev2b_backbone::map_quantized(quantized.tensors, w) && w.layers.empty(), "value-parsed quant gate");
    }
    flag("CRISPEMBED_JOSEV_QUANT", "1");
    require(josev2b_backbone::map_quantized(quantized.tensors, w) && w.layers.size() == 24,
            "separate bounded mixed map");
    require(!josev2b_backbone::map_quantized(bad.tensors, w) && w.layers.empty(), "quantized vocabulary rejected");
    flag("CRISPEMBED_QWEN35_QUANT", "0");
    require(!josev2b_backbone::map_quantized(quantized.tensors, w), "primitive quant gate required");
    require(josev2b_backbone::map(f32.tensors, w), "legacy F32 remains independent of quant gates");
    for (const char * name : { "head.memory_proj.weight", "l.embed_tokens.weight", "l.layers.0.input_layernorm.weight",
                               "l.layers.0.linear_attn.conv1d.weight", "l.layers.0.linear_attn.in_proj_a.weight" })
        require(!josev2b_packed::projection(name), "critical projection policy");
    std::puts("PASS: 320 frozen shapes, legacy F32, mixed Q4_K gates, protected vocabulary/norm/head/conv/gates");
    return 0;
}
int entry() {
    try {
        return run();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
} // namespace
int main() {
    core_util::clean_exit(entry());
}
