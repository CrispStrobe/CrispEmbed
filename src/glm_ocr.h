// glm_ocr.h — GLM-OCR inference engine (zai-org/GLM-OCR, 0.9B, MIT).
//
// Architecture:
//   Image → Conv3D patches → CogViT (24L, 1024d, RMSNorm+SwiGLU, Q/K norm)
//         → RMSNorm → Conv2D downsample (stride 2)
//         → merger (proj + SwiGLU + LayerNorm)
//         → splice into text tokens
//         → GLM-0.5B (16L, 1536d, GQA 16/8, post-norm, mRoPE, fused gate_up)
//         → autoregressive generation
//
// Key differences from InternVL2:
//   - Vision: RMSNorm (not LayerNorm), SwiGLU (not GELU), Q/K RMSNorm per head
//   - Downsample: learned Conv2D (not pixel unshuffle)
//   - LLM: 4 norms/layer (post-norm), Q upscale (1536→2048), mRoPE
//   - mRoPE sections [16,24,24] (same as Qwen2VL)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace glm_ocr {

struct vision_hparams {
    uint32_t depth = 24;
    uint32_t hidden_size = 1024;
    uint32_t intermediate_size = 4096;
    uint32_t num_heads = 16;
    uint32_t head_dim = 64;           // = hidden_size / num_heads
    uint32_t patch_size = 14;
    uint32_t image_size = 336;
    uint32_t temporal_patch_size = 2;
    uint32_t spatial_merge_size = 2;
    uint32_t out_hidden_size = 1536;
    float rms_norm_eps = 1e-5f;
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct llm_hparams {
    uint32_t vocab_size = 59392;
    uint32_t hidden_size = 1536;
    uint32_t intermediate_size = 4608;
    uint32_t num_hidden_layers = 16;
    uint32_t num_attention_heads = 16;
    uint32_t num_key_value_heads = 8;
    uint32_t head_dim = 128;
    uint32_t max_position_embeddings = 131072;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int rope_sections[3] = {16, 24, 24};
    uint32_t image_token_id = 59280;
    uint32_t eos_token_id = 59246;
};

// Vision block weights
struct vision_block {
    ggml_tensor *norm1_w = nullptr;
    ggml_tensor *norm2_w = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    ggml_tensor *q_norm_w = nullptr, *k_norm_w = nullptr;
    // SwiGLU FFN with bias
    ggml_tensor *ffn_gate_w = nullptr, *ffn_gate_b = nullptr;
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
};

struct vision_merger {
    ggml_tensor *proj_w = nullptr;
    ggml_tensor *gate_w = nullptr;
    ggml_tensor *up_w = nullptr;
    ggml_tensor *down_w = nullptr;
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
};

// LLM layer weights (post-norm with 4 norms)
struct llm_layer {
    ggml_tensor *input_layernorm_w = nullptr;
    ggml_tensor *post_self_attn_layernorm_w = nullptr;
    ggml_tensor *post_attention_layernorm_w = nullptr;
    ggml_tensor *post_mlp_layernorm_w = nullptr;
    ggml_tensor *q_w = nullptr;
    ggml_tensor *k_w = nullptr;
    ggml_tensor *v_w = nullptr;
    ggml_tensor *o_w = nullptr;
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

struct model {
    vision_hparams vhp;
    llm_hparams lhp;

    // Vision
    ggml_tensor *patch_embed_w = nullptr, *patch_embed_b = nullptr;
    std::vector<vision_block> vis_blocks;
    ggml_tensor *post_layernorm_w = nullptr;
    ggml_tensor *downsample_w = nullptr, *downsample_b = nullptr;
    vision_merger merger;

    // LLM
    ggml_tensor *embed_tokens = nullptr;
    std::vector<llm_layer> llm_layers;
    ggml_tensor *output_norm_w = nullptr;
    ggml_tensor *lm_head_w = nullptr;
};

struct kv_cache {
    ggml_tensor *k = nullptr, *v = nullptr;
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    int max_seq = 0;
    int n_past = 0;
    bool allocated = false;
};

struct tokenizer {
    std::vector<std::string> id_to_piece;
    int vocab_size = 0;
    int eos_id = 59246;
    std::string decode(const int32_t *ids, int n) const;
};

struct context {
    model m;
    ggml_context *model_ctx = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    kv_cache kvc;
    tokenizer tok;
    int n_threads = 4;
    int verbosity = 1;
    bool bench = false;
    std::string diff_ref_path;
};

bool load(context &ctx, const char *gguf_path, int n_threads = 4, int verbosity = 1);
void free_(context &ctx);

struct vision_result {
    float *hidden = nullptr;
    int n_tokens = 0;
    int hidden_dim = 0;
};

bool encode_vision(context &ctx, const float *pixels, vision_result &out);

struct llm_result {
    float *hidden = nullptr;
    float *logits = nullptr;
    int n_tokens = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
};

bool run_llm_forward(context &ctx, const int32_t *token_ids, int n_tokens,
                     llm_result &out);

struct generate_result {
    std::vector<int32_t> token_ids;
    std::string text;
    std::vector<float> token_confidences;
};

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *prompt_ids, int n_prompt,
              int max_new_tokens,
              generate_result &out);

}  // namespace glm_ocr

#ifdef __cplusplus
extern "C" {
#endif

typedef struct glm_ocr_context glm_ocr_context;
glm_ocr_context * glm_ocr_init(const char * model_path, int n_threads);
void glm_ocr_free(glm_ocr_context * ctx);
const char * glm_ocr_recognize_raw(glm_ocr_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len);
const char * glm_ocr_recognize(glm_ocr_context * ctx,
    const float * px, int w, int h, int * out_len);

const float * glm_ocr_confidences(const glm_ocr_context * ctx, int * n_tokens);
float glm_ocr_mean_confidence(const glm_ocr_context * ctx);

#ifdef __cplusplus
}
#endif
