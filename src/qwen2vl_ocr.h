// qwen2vl_ocr.h — Qwen2.5-VL OCR inference engine.
//
// Loads a Qwen2.5-VL GGUF (vision encoder + LLM decoder), processes an
// image + text prompt, and generates text output. Primary use case:
// German business document OCR via Keyven/german-ocr fine-tunes.
//
// Architecture:
//   Image → preprocess → ViT (32 layers, 1280d, RoPE, windowed attn)
//         → spatial merger (4:1 merge, 2048d)
//         → splice into text token sequence
//         → Qwen2.5 LLM decoder (36 layers, 2048d, GQA, mRoPE)
//         → autoregressive text generation
//
// The vision encoder reuses patterns from bidirlm_vision.cpp (same Qwen
// family), with key differences:
//   - RMSNorm instead of LayerNorm (no bias in norms)
//   - SwiGLU FFN (gate + up + down) instead of GELU (fc1 + fc2)
//   - Windowed attention (window_size=112) with full attn at specific layers
//   - No DeepStack
//
// Integration with crispembed_diff.h for per-layer parity validation.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#include <string>
#include <vector>

namespace qwen2vl_ocr {

// ── Hyperparameters ──────────────────────────────────────────────────

struct vision_hparams {
  uint32_t depth = 32;
  uint32_t hidden_size = 1280;
  uint32_t intermediate_size = 3420;
  uint32_t num_heads = 16;
  uint32_t in_channels = 3;
  uint32_t spatial_patch_size = 14;
  uint32_t spatial_merge_size = 2;
  uint32_t temporal_patch_size = 2;
  uint32_t out_hidden_size = 2048;
  uint32_t window_size = 112; // windowed attention
  std::vector<int> fullatt_block_indexes = {7, 15, 23,
                                            31}; // full attention layers
  std::vector<int> deepstack_indexes; // Qwen3-VL: multi-layer feature concat
                                      // (e.g. [5,11,17])

  // Image preprocessor
  float image_mean[3] = {0.4815f, 0.4578f, 0.4082f};
  float image_std[3] = {0.2686f, 0.2613f, 0.2758f};
  uint32_t min_pixels = 3136;
  uint32_t max_pixels = 12845056;

  // Vision variant detection (auto-detected from weights):
  //   false = Qwen2.5-VL (RMSNorm, fused QKV, SwiGLU FFN)
  //   true  = Qwen2-VL   (LayerNorm with bias, fused QKV, GELU fc1/fc2 FFN)
  bool is_qwen2_vl = false;

  // PaddleOCR-VL: learned position embeddings + post_layernorm
  bool has_position_embed = false;
  float layer_norm_eps = 1e-6f;
};

struct llm_hparams {
  uint32_t vocab_size = 151936;
  uint32_t hidden_size = 2048;
  uint32_t intermediate_size = 11008;
  uint32_t num_hidden_layers = 36;
  uint32_t num_attention_heads = 16;
  uint32_t num_key_value_heads = 2;
  uint32_t max_position_embeddings = 128000;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  int rope_sections[4] = {16, 24, 24, 0}; // mRoPE sections
  bool mrope_interleaved = false;         // Qwen3-VL: interleaved mRoPE
  bool has_qk_norm = false;               // Qwen3-VL: RMSNorm on Q/K per head
  bool tie_word_embeddings = true;

  // Special token IDs
  uint32_t image_token_id = 0;
  uint32_t video_token_id = 0;
  uint32_t vision_start_token_id = 0;
  uint32_t vision_end_token_id = 0;
};

// ── Weight structures ────────────────────────────────────────────────

struct vision_block {
  // Norms: RMSNorm (Qwen2.5-VL, no bias) or LayerNorm (Qwen2-VL, with bias)
  ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
  ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
  // Attention: fused QKV (CrispEmbed) or separate Q/K/V (llama.cpp mmproj)
  ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr; // fused
  ggml_tensor *q_w = nullptr, *q_b = nullptr;     // separate (mmproj)
  ggml_tensor *k_w = nullptr, *k_b = nullptr;
  ggml_tensor *v_w = nullptr, *v_b = nullptr;
  ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
  // FFN: SwiGLU (Qwen2.5-VL: gate+up+down) or GELU (Qwen2-VL: fc1+fc2)
  // SwiGLU fields:
  ggml_tensor *ffn_gate_w = nullptr, *ffn_gate_b = nullptr;
  ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
  ggml_tensor *ffn_down_w = nullptr, *ffn_down_b = nullptr;
  // GELU fc1/fc2 fields (Qwen2-VL):
  ggml_tensor *ffn_fc1_w = nullptr, *ffn_fc1_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr, *ffn_fc2_b = nullptr;
};

struct vision_merger {
  ggml_tensor *norm_w = nullptr, *norm_b = nullptr; // LayerNorm
  ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
  ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
};

struct llm_layer {
  ggml_tensor *attn_norm_w = nullptr; // RMSNorm
  ggml_tensor *ffn_norm_w = nullptr;
  // Attention: Qwen2-VL has Q/K/V/O bias; Qwen2.5-VL generally omits them.
  ggml_tensor *q_w = nullptr, *q_b = nullptr;
  ggml_tensor *k_w = nullptr, *k_b = nullptr;
  ggml_tensor *v_w = nullptr, *v_b = nullptr;
  ggml_tensor *o_w = nullptr, *o_b = nullptr;
  // Qwen3-VL: per-head QK RMSNorm
  ggml_tensor *q_norm_w = nullptr;
  ggml_tensor *k_norm_w = nullptr;
  // SwiGLU FFN (no biases)
  ggml_tensor *ffn_gate_w = nullptr;
  ggml_tensor *ffn_up_w = nullptr;
  ggml_tensor *ffn_down_w = nullptr;
};

struct model {
  vision_hparams vhp;
  llm_hparams lhp;

  // Vision encoder
  ggml_tensor *patch_embed_w = nullptr, *patch_embed_b = nullptr;
  ggml_tensor *position_embed_w = nullptr;    // PaddleOCR-VL: learned pos embed
  ggml_tensor *packing_pos_embed_w = nullptr; // PaddleOCR-VL: packing pos embed
  ggml_tensor *post_layernorm_w =
      nullptr; // PaddleOCR-VL: post-encoder layernorm
  ggml_tensor *post_layernorm_b = nullptr;
  std::vector<vision_block> vis_blocks;
  vision_merger merger;
  std::vector<vision_merger>
      deepstack_mergers; // Qwen3-VL: mergers at deepstack_indexes

  // LLM decoder
  ggml_tensor *embed_tokens = nullptr;
  std::vector<llm_layer> llm_layers;
  ggml_tensor *output_norm_w = nullptr;
  ggml_tensor *lm_head_w = nullptr; // may be nullptr if tied
};

// ── Context ──────────────────────────────────────────────────────────

struct context {
  model m;
  ggml_context *model_ctx = nullptr;
  ggml_backend_buffer_t model_buf = nullptr;
  // Second buffer for mmproj (split GGUF loading)
  ggml_context *mmproj_ctx = nullptr;
  ggml_backend_buffer_t mmproj_buf = nullptr;

  ggml_backend_t backend = nullptr;
  ggml_backend_t backend_cpu = nullptr;
  ggml_backend_sched_t sched = nullptr;
  std::vector<uint8_t> compute_meta;

  int n_threads = 4;
  int verbosity = 1;
  bool bench = false;

  // Optional diff harness path (set before encode to enable comparison)
  std::string diff_ref_path;

  // Temporary: deepstack embeds for passing through generate → run_llm_forward
  const float *const *deepstack_embeds_tmp = nullptr;
  int n_deepstack_tmp = 0;

  // Qwen3-VL position embeddings may be quantized. Keep the dequantized
  // table once per loaded model instead of expanding it for every image.
  std::vector<float> position_embed_cache;

  struct kv_cache {
    ggml_context *ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor *k = nullptr;
    ggml_tensor *v = nullptr;
    int max_seq = 0;
    bool allocated = false;
  } kvc;

  // Persistent decode graph — built once per generate() call, reused every
  // step.
  struct decode_cache {
    ggml_context *gctx = nullptr;
    std::vector<uint8_t> meta;
    ggml_cgraph *gf = nullptr;
    bool allocated = false;
  } dkvc;
};

// ── API ──────────────────────────────────────────────────────────────

// Load model from GGUF file. For llama.cpp split format (Keyven etc.),
// pass the vision encoder mmproj GGUF as the second path.
bool load(context &ctx, const char *gguf_path, int n_threads = 4,
          int verbosity = 1, const char *mmproj_path = nullptr);

// Free model resources.
void free_(context &ctx);

// Run vision encoder on preprocessed patches.
// patches: (n_patches, C * T * P * P) row-major float
// grid_thw: (t, h, w) as int32_t[3]
// Returns merged image embeddings (n_merged, out_hidden_size).
struct vision_result {
  float *image_embeds = nullptr; // malloc'd, caller frees
  int n_merged = 0;
  int embed_dim = 0;
  // Qwen3-VL deepstack: per-layer merged features for LLM injection
  std::vector<float *> deepstack_embeds; // each malloc'd, caller frees
};

bool encode_vision(context &ctx, const float *patches, int n_patches,
                   const int32_t *grid_thw, vision_result &out);

void vision_result_free(vision_result &r);

// Run LLM decoder forward pass (text-only, no KV cache).
// For parity testing. Returns final hidden states.
struct llm_result {
  float *hidden = nullptr; // (T, D) final hidden states (malloc'd)
  float *logits = nullptr; // (n_logits, V) logits (if lm_head available)
  int n_tokens = 0;
  int n_logits = 0;
  int hidden_dim = 0;
  int vocab_size = 0;
  int rope_delta = 0; // mRoPE delta for cached decode after image prefill
  // Graph pointer for KV cache extraction (valid until next call)
  struct ggml_cgraph *kv_graph = nullptr;
  struct ggml_context *kv_graph_ctx = nullptr; // owns the graph memory
};

// image_input: optional, for vision-text splicing
struct image_input {
  const float *image_embeds = nullptr; // (n_image_tokens, D)
  int n_image_tokens = 0;
  const int32_t *grid_thw = nullptr; // (3,) per image
  int n_images = 0;
  // Qwen3-VL deepstack: per-layer features for injection into LLM layers 0..N-1
  const float *const *deepstack_embeds =
      nullptr; // array of (n_image_tokens, D) pointers
  int n_deepstack = 0;
};

bool run_llm_forward(context &ctx, const int32_t *token_ids, int n_tokens,
                     llm_result &out, const image_input *img = nullptr,
                     bool logits_last_only = false,
                     bool materialize_hidden = true);

// Generate text from image + prompt.
// Returns generated token IDs and decoded text.
struct generate_result {
  std::vector<int32_t> token_ids;
  std::string text;
  std::vector<float> token_confidences;
};

bool generate(context &ctx, const float *image_embeds, int n_image_tokens,
              int embed_dim,
              const int32_t *grid_thw, // actual image grid (t,h,w) for mRoPE
              const int32_t *prompt_token_ids, int n_prompt_tokens,
              int max_new_tokens, generate_result &out);

} // namespace qwen2vl_ocr

// ── C ABI (for crispembed.cpp dispatch) ──────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen2vl_ocr_context qwen2vl_ocr_context;

qwen2vl_ocr_context *qwen2vl_ocr_init(const char *model_path, int n_threads);
qwen2vl_ocr_context *qwen2vl_ocr_init_split(const char *llm_path,
                                            const char *mmproj_path,
                                            int n_threads);
void qwen2vl_ocr_free(qwen2vl_ocr_context *ctx);

// Set the text prompt for generation (default: "Describe this image.")
void qwen2vl_ocr_set_prompt(qwen2vl_ocr_context *ctx, const char *prompt);

// Set max generation tokens (default: 512)
void qwen2vl_ocr_set_max_tokens(qwen2vl_ocr_context *ctx, int max_tokens);

// Run OCR on raw pixel bytes (RGB/RGBA/grayscale).
// Returns generated text (owned by ctx, valid until next call or free).
const char *qwen2vl_ocr_recognize_raw(qwen2vl_ocr_context *ctx,
                                      const uint8_t *pixel_bytes, int width,
                                      int height, int channels, int *out_len);

// Run OCR on grayscale float pixels [0..1].
const char *qwen2vl_ocr_recognize(qwen2vl_ocr_context *ctx, const float *pixels,
                                  int width, int height, int *out_len);

const float *qwen2vl_ocr_confidences(const qwen2vl_ocr_context *ctx,
                                     int *n_tokens);
float qwen2vl_ocr_mean_confidence(const qwen2vl_ocr_context *ctx);

#ifdef __cplusplus
}
#endif
