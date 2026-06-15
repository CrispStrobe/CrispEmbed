// got_ocr.h — GOT-OCR2 inference engine (stepfun-ai/GOT-OCR2_0, 0.7B, Apache-2.0).
//
// Architecture:
//   Image → Conv2D patches → SAM ViT-B (12L, 768d, LayerNorm+GELU,
//           window+global attn, decomposed RPE)
//         → Neck (Conv→LN2d→Conv→LN2d)
//         → Downsample (Conv 256→512→1024, stride 2)
//         → Projector (Linear 1024→1024)
//         → splice into text tokens
//         → Qwen2-0.5B (24L, 1024d, MHA 16/16, RoPE, SiLU SwiGLU)
//         → autoregressive generation
//
// Key differences from GLM-OCR:
//   - Vision: LayerNorm (not RMSNorm), GELU (not SiLU), no Q/K norm
//   - Vision: SAM-style window attention with decomposed RPE
//   - LLM: standard pre-norm (2 norms/layer), MHA (not GQA), standard RoPE
//   - LLM: separate Q/K/V with bias, separate gate/up/down
//   - Tied word embeddings (no separate lm_head)

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <string>
#include <vector>

namespace got_ocr {

struct vision_hparams {
    uint32_t depth = 12;
    uint32_t hidden_size = 768;
    uint32_t intermediate_size = 3072;
    uint32_t num_heads = 12;
    uint32_t head_dim = 64;           // = hidden_size / num_heads
    uint32_t patch_size = 16;
    uint32_t image_size = 1024;
    uint32_t window_size = 14;
    uint32_t neck_out_channels = 256;
    std::vector<int> global_attn_indexes;  // [2, 5, 8, 11]
    float image_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float image_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};
};

struct llm_hparams {
    uint32_t vocab_size = 151860;
    uint32_t hidden_size = 1024;
    uint32_t intermediate_size = 2816;
    uint32_t num_hidden_layers = 24;
    uint32_t num_attention_heads = 16;
    uint32_t num_key_value_heads = 16;
    uint32_t head_dim = 64;
    uint32_t max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    uint32_t image_token_id = 151859;
    uint32_t image_start_token_id = 151857;
    uint32_t image_end_token_id = 151858;
    uint32_t image_token_len = 256;
    uint32_t eos_token_id = 151643;
};

// Vision block weights (SAM ViT-B)
struct vision_block {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    ggml_tensor *rel_pos_h = nullptr, *rel_pos_w = nullptr;
    // GELU MLP (not SwiGLU)
    ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
    bool is_global = false;
};

// LLM layer weights (standard Qwen2 pre-norm)
struct llm_layer {
    ggml_tensor *input_layernorm_w = nullptr;
    ggml_tensor *post_attention_layernorm_w = nullptr;
    // Separate Q/K/V with bias
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr;
    // SwiGLU FFN (separate gate/up/down)
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

struct model {
    vision_hparams vhp;
    llm_hparams lhp;

    // Vision: patch embed + pos embed
    ggml_tensor *patch_embed_w = nullptr, *patch_embed_b = nullptr;
    ggml_tensor *pos_embed = nullptr;
    std::vector<vision_block> vis_blocks;
    // Neck
    ggml_tensor *neck_conv1_w = nullptr;
    ggml_tensor *neck_ln1_w = nullptr, *neck_ln1_b = nullptr;
    ggml_tensor *neck_conv2_w = nullptr;
    ggml_tensor *neck_ln2_w = nullptr, *neck_ln2_b = nullptr;
    // Downsample
    ggml_tensor *net_2_w = nullptr;
    ggml_tensor *net_3_w = nullptr;
    // Projector
    ggml_tensor *projector_w = nullptr, *projector_b = nullptr;

    // LLM
    ggml_tensor *embed_tokens = nullptr;
    std::vector<llm_layer> llm_layers;
    ggml_tensor *output_norm_w = nullptr;
    ggml_tensor *lm_head_w = nullptr;  // nullptr if tied
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
    int eos_id = 151643;
    std::string decode(const int32_t *ids, int n) const;
};

struct context {
    model m;
    ggml_context *model_ctx = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;  // CPU fallback (sched requires CPU last)
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
    kv_cache kvc;
    tokenizer tok;
    int n_threads = 4;
    int verbosity = 1;
    std::string diff_ref_path;
    // Precomputed RPE tables per layer
    std::vector<std::vector<float>> rp_h_per_layer;
    std::vector<std::vector<float>> rp_w_per_layer;
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
};

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *prompt_ids, int n_prompt,
              int max_new_tokens,
              generate_result &out);

}  // namespace got_ocr

#ifdef __cplusplus
extern "C" {
#endif

typedef struct got_ocr_context got_ocr_context;
got_ocr_context * got_ocr_init(const char * model_path, int n_threads);
void got_ocr_free(got_ocr_context * ctx);
const char * got_ocr_recognize_raw(got_ocr_context * ctx,
    const uint8_t * px, int w, int h, int ch, int * out_len);
const char * got_ocr_recognize(got_ocr_context * ctx,
    const float * px, int w, int h, int * out_len);

#ifdef __cplusplus
}
#endif
