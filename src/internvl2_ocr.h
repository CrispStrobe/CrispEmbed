// internvl2_ocr.h — InternVL2/2.5 OCR inference engine.
//
// Loads an InternVL2 GGUF (vision encoder + projector + LLM decoder),
// processes an image + text prompt, and generates text output.
//
// Supports:
//   - InternVL2.5-2B (InternViT-300M + InternLM2.5-1.8B) — multilingual OCR
//   - InternVL2-1B (InternViT-300M + Qwen2-0.5B) — edge/WASM OCR
//
// Architecture:
//   Image → preprocess → InternViT (24L, 1024d, LayerNorm, GELU, LayerScale)
//         → remove CLS → pixel unshuffle (4:1, dim 1024→4096)
//         → MLP projector (LN + Linear + GELU + Linear, 4096→2048)
//         → splice into text token sequence
//         → InternLM2.5 decoder (24L, 2048d, GQA 16/8, SwiGLU, RMSNorm)
//         → autoregressive text generation
//
// Key differences from Qwen2.5-VL (qwen2vl_ocr):
//   - Standard ViT with LayerNorm (not RMSNorm), GELU (not SwiGLU), LayerScale
//   - CLS token + learnable position embeddings (not RoPE in vision)
//   - Pixel unshuffle (spatial reshape) instead of spatial merger conv
//   - Standard 1D RoPE in LLM (not mRoPE)
//   - GQA 16/8 (not 16/2)
//   - No windowed attention in vision encoder

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace internvl2_ocr {

// ── Hyperparameters ──────────────────────────────────────────────────

struct vision_hparams {
    uint32_t num_hidden_layers = 24;
    uint32_t hidden_size = 1024;
    uint32_t intermediate_size = 4096;
    uint32_t num_attention_heads = 16;
    uint32_t head_dim = 64;            // = hidden_size / num_attention_heads
    uint32_t patch_size = 14;
    uint32_t image_size = 448;
    float layer_norm_eps = 1e-6f;
    bool qkv_bias = true;

    // Derived
    uint32_t n_patches_per_side = 32;  // = image_size / patch_size
    uint32_t n_patches = 1024;         // = n_patches_per_side^2
    uint32_t n_positions = 1025;       // = n_patches + 1 (CLS)

    // Image preprocessor
    float image_mean[3] = {0.485f, 0.456f, 0.406f};
    float image_std[3]  = {0.229f, 0.224f, 0.225f};
};

struct projector_hparams {
    float downsample_ratio = 0.5f;
    uint32_t n_merged_tokens = 256;    // = n_patches * downsample_ratio^2
    uint32_t merge_dim = 4096;         // = hidden_size / downsample_ratio^2
    uint32_t output_dim = 2048;        // = llm hidden_size
};

struct llm_hparams {
    uint32_t vocab_size = 92553;
    uint32_t hidden_size = 2048;
    uint32_t intermediate_size = 8192;
    uint32_t num_hidden_layers = 24;
    uint32_t num_attention_heads = 16;
    uint32_t num_key_value_heads = 8;
    uint32_t head_dim = 128;           // = hidden_size / num_attention_heads
    uint32_t max_position_embeddings = 32768;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 1000000.0f;
    float rope_scaling_factor = 2.0f;
    bool tie_word_embeddings = false;

    // Dynamic resolution
    uint32_t max_dynamic_patch = 12;
    uint32_t min_dynamic_patch = 1;
    bool use_thumbnail = true;

    // Special tokens
    uint32_t bos_token_id = 1;
    uint32_t eos_token_id = 2;
    uint32_t image_token_id = 0;       // <IMG_CONTEXT> placeholder
};

// ── Weight structures ────────────────────────────────────────────────

struct vision_block {
    // LayerNorm (with bias)
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    // LayerScale
    ggml_tensor *ls1 = nullptr;
    ggml_tensor *ls2 = nullptr;
    // Fused QKV attention (with bias)
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    // GELU MLP (with bias)
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct vision_projector {
    // LayerNorm (on pixel-unshuffled input)
    ggml_tensor *norm_w = nullptr, *norm_b = nullptr;
    // Linear 4096→2048
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    // Linear 2048→2048
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct llm_layer {
    ggml_tensor *attn_norm_w = nullptr;  // RMSNorm (no bias)
    ggml_tensor *ffn_norm_w = nullptr;
    // Separate Q/K/V (InternLM2: no bias; Qwen2: with bias)
    ggml_tensor *q_w = nullptr, *q_b = nullptr;
    ggml_tensor *k_w = nullptr, *k_b = nullptr;
    ggml_tensor *v_w = nullptr, *v_b = nullptr;
    ggml_tensor *o_w = nullptr;
    // SwiGLU FFN (no bias)
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

struct model {
    vision_hparams vhp;
    projector_hparams php;
    llm_hparams lhp;

    // Vision encoder
    ggml_tensor *patch_embed_w = nullptr, *patch_embed_b = nullptr;
    ggml_tensor *class_embedding = nullptr;    // [1, 1, 1024]
    ggml_tensor *position_embedding = nullptr; // [1, 1025, 1024]
    std::vector<vision_block> vis_blocks;

    // MLP projector (mlp1)
    vision_projector proj;

    // LLM decoder
    ggml_tensor *embed_tokens = nullptr;
    std::vector<llm_layer> llm_layers;
    ggml_tensor *output_norm_w = nullptr;
    ggml_tensor *lm_head_w = nullptr;
};

// ── Context ──────────────────────────────────────────────────────────

// ── Tokenizer (decode only) ──────────────────────────────────────────

struct tokenizer {
    std::vector<std::string> id_to_piece;  // vocab: id → string
    int vocab_size = 0;
    int bos_id = 1;
    int eos_id = 2;
    int im_end_id = -1;  // <|im_end|> for chat stop
    int image_token_id = 0;

    // Decode token IDs to UTF-8 text.
    // SentencePiece convention: ▁ → space, <0xNN> → raw byte.
    std::string decode(const std::vector<int32_t> &ids) const;
    std::string decode(const int32_t *ids, int n) const;
};

// ── KV cache ─────────────────────────────────────────────────────────

struct kv_cache {
    // Persistent KV cache: (hd, max_seq, n_kv_heads, n_layers) F16
    ggml_tensor *k = nullptr;  // all layers K
    ggml_tensor *v = nullptr;  // all layers V
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    int max_seq = 0;      // allocated capacity
    int n_past = 0;       // tokens already in cache
    bool allocated = false;
};

struct context {
    model m;
    ggml_context *model_ctx = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    // Tokenizer for decode
    tokenizer tok;

    // KV cache for autoregressive generation
    kv_cache kvc;

    int n_threads = 4;
    int verbosity = 1;

    // Optional diff harness path
    std::string diff_ref_path;
};

// ── API ──────────────────────────────────────────────────────────────

// Load model from GGUF file.
bool load(context &ctx, const char *gguf_path, int n_threads = 4, int verbosity = 1);

// Free model resources.
void free_(context &ctx);

// Run vision encoder on a single 448×448 tile.
// pixels: (3, 448, 448) normalized float, planar RGB
// Returns (n_positions, hidden_size) hidden states.
struct vision_result {
    float *hidden = nullptr;     // malloc'd, caller frees
    int n_tokens = 0;
    int hidden_dim = 0;
};

bool encode_vision_tile(context &ctx,
                        const float *pixels,  // [3, 448, 448] normalized
                        vision_result &out);

// Run pixel unshuffle + MLP projector on vision hidden states.
// input: (n_patches, vis_hidden) from encode_vision_tile (CLS removed)
// Returns (n_merged, llm_hidden) projected embeddings.
struct project_result {
    float *embeds = nullptr;     // malloc'd, caller frees
    int n_tokens = 0;
    int embed_dim = 0;
};

bool project_vision(context &ctx,
                    const float *vis_hidden, int n_patches,
                    project_result &out);

// Run full vision pipeline: tiles → ViT → unshuffle → project → concat
// For multi-tile input (dynamic resolution).
struct vision_pipeline_result {
    float *image_embeds = nullptr;   // malloc'd, (total_tokens, llm_hidden)
    int n_image_tokens = 0;
    int embed_dim = 0;
};

bool encode_vision(context &ctx,
                   const float *tiles,  // [n_tiles, 3, 448, 448]
                   int n_tiles,
                   vision_pipeline_result &out);

// Run LLM decoder forward pass (for parity testing).
struct llm_result {
    float *hidden = nullptr;
    float *logits = nullptr;
    int n_tokens = 0;
    int hidden_dim = 0;
    int vocab_size = 0;
};

struct image_input {
    const float *image_embeds = nullptr;
    int n_image_tokens = 0;
};

bool run_llm_forward(context &ctx,
                     const int32_t *token_ids, int n_tokens,
                     llm_result &out,
                     const image_input *img = nullptr);

// Generate text from image + prompt.
struct generate_result {
    std::vector<int32_t> token_ids;
    std::string text;
};

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *prompt_token_ids, int n_prompt_tokens,
              int max_new_tokens,
              generate_result &out);

}  // namespace internvl2_ocr

// ── C ABI (for crispembed.cpp dispatch) ──────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

typedef struct internvl2_ocr_context internvl2_ocr_context;

internvl2_ocr_context * internvl2_ocr_init(const char * model_path, int n_threads);
void internvl2_ocr_free(internvl2_ocr_context * ctx);

void internvl2_ocr_set_prompt(internvl2_ocr_context * ctx, const char * prompt);
void internvl2_ocr_set_max_tokens(internvl2_ocr_context * ctx, int max_tokens);

const char * internvl2_ocr_recognize_raw(
    internvl2_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width, int height, int channels,
    int * out_len);

const char * internvl2_ocr_recognize(
    internvl2_ocr_context * ctx,
    const float * pixels,
    int width, int height,
    int * out_len);

#ifdef __cplusplus
}
#endif
