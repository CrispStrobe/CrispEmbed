#include "josev2b_backbone.h"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
int main() {
    ggml_context * ctx = nullptr;
    try {
        ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
        require(ctx, "Metadata context missing");
        core_gguf::tensor_map tensors;
        auto add = [&](const std::string & name, int x, int y = 1, int z = 1) {
            tensors.emplace(name, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, x, y, z));
        };
        add("l.embed_tokens.weight", 2048, 248320);
        add("l.norm.weight", 2048);
        for (int i = 0; i < 24; ++i) {
            const auto p = "l.layers." + std::to_string(i) + ".";
            add(p + "input_layernorm.weight", 2048);
            add(p + "post_attention_layernorm.weight", 2048);
            add(p + "mlp.gate_proj.weight", 2048, 6144);
            add(p + "mlp.up_proj.weight", 2048, 6144);
            add(p + "mlp.down_proj.weight", 6144, 2048);
            if (i % 4 == 3) {
                add(p + "self_attn.q_proj.weight", 2048, 4096);
                add(p + "self_attn.k_proj.weight", 2048, 512);
                add(p + "self_attn.v_proj.weight", 2048, 512);
                add(p + "self_attn.o_proj.weight", 2048, 2048);
                add(p + "self_attn.q_norm.weight", 256);
                add(p + "self_attn.k_norm.weight", 256);
            } else {
                add(p + "linear_attn.in_proj_qkv.weight", 2048, 6144);
                add(p + "linear_attn.in_proj_z.weight", 2048, 2048);
                add(p + "linear_attn.in_proj_a.weight", 2048, 16);
                add(p + "linear_attn.in_proj_b.weight", 2048, 16);
                add(p + "linear_attn.out_proj.weight", 2048, 2048);
                add(p + "linear_attn.conv1d.weight", 4, 1, 6144);
                add(p + "linear_attn.A_log", 16);
                add(p + "linear_attn.dt_bias", 16);
                add(p + "linear_attn.norm.weight", 128);
            }
        }
        require(tensors.size() == 320, "Expected exact source-derived320tensor profile");
        gate("CRISPEMBED_QWEN35_OCR", "1");
        qwen35_ocr::decoder_weights w;
        for (const char * disabled : { "", "0", "false" }) {
            gate("CRISPEMBED_JOSEV_BACKBONE", disabled);
            require(!josev2b_backbone::map(tensors, w) && !w.embedding && w.layers.empty(),
                    "Disabled gate populated weights");
        }
        gate("CRISPEMBED_JOSEV_BACKBONE", "1");
        require(josev2b_backbone::map(tensors, w) && !w.head && w.layers.size() == 24,
                "Valid profile rejected or vocabulary head bound");
        qwen35_ocr::decoder_weights old;
        require(!qwen35_ocr::map_small_decoder(tensors, old), "Existing1024profile widened");
        auto * matrix = tensors.at("l.layers.0.mlp.up_proj.weight");
        --matrix->ne[1];
        require(!josev2b_backbone::map(tensors, w) && w.layers.empty(), "Malformed FFN accepted");
        ++matrix->ne[1];
        matrix->type = GGML_TYPE_F16;
        require(!josev2b_backbone::map(tensors, w), "Unqualified precision accepted");
        matrix->type = GGML_TYPE_F32;
        tensors.emplace("lm_head.weight", tensors.at("l.embed_tokens.weight"));
        require(!josev2b_backbone::map(tensors, w), "Unexpected vocabulary head accepted");
        tensors.erase("lm_head.weight");
        require(josev2b_backbone::map(tensors, w), "Metadata recovery failed");
        qwen35_ocr::decoder_outputs out;
        out.logits = { 42 };
        out.norm = { 42 };
        out.hidden = { { 42 } };
        require(!josev2b_backbone::run(w, {}, 2, out) && out.logits.empty() && out.norm.empty() && out.hidden.empty(),
                "Rejected input retained stale output");
        require(!josev2b_backbone::run(w, { 0 }, 0, out), "Invalid thread count accepted");
        gate("CRISPEMBED_JOSEV_BACKBONE", "0");
        require(!josev2b_backbone::run(w, { 0 }, 2, out), "Disabled forward entered unallocated metadata tensors");
        ggml_free(ctx);
        std::puts("PASS Josev metadata/gates/strictshape/nohead/output-recovery; no model data or numerical parity");
        return 0;
    } catch (const std::exception & error) {
        if (ctx) ggml_free(ctx);
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
