#pragma once

#include "core/gguf_loader.h"
#include "tokenizer.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

// LoRA adapter data for runtime hot-swap
struct lora_pair {
    std::vector<float> A;  // [rank, in_dim] row-major F32
    std::vector<float> B;  // [out_dim, rank] row-major F32
    int rank    = 0;
    int in_dim  = 0;
    int out_dim = 0;
    bool empty() const { return A.empty(); }
};

struct lora_layer {
    lora_pair q, k, v, o, gate, up, down;
};

struct lora_adapter {
    std::string name;
    float alpha = 0.0f;
    int rank    = 0;
    std::vector<lora_layer> layers;
};

struct dec_layer {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * q_w = nullptr, * q_b = nullptr;
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * q_norm_w = nullptr;
    ggml_tensor * k_norm_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * gate_w = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * down_w = nullptr;
    // Gemma3 extra norms
    ggml_tensor * post_attn_norm_w = nullptr; // post_attention_layernorm (Gemma3: applied to attn out before residual)
    ggml_tensor * pre_ffn_norm_w = nullptr;   // pre_feedforward_layernorm
    ggml_tensor * post_ffn_norm_w = nullptr;  // post_feedforward_layernorm
};

struct dec_model {
    int n_vocab = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int n_layer = 0;
    int n_intermediate = 0;
    int n_max_pos = 8192;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    float rope_theta_local = 0.0f;  // Gemma3: sliding-window layers use shorter theta (0 = same as rope_theta)
    int global_attn_every_n = 0;    // Gemma3: period between global attention layers (0 = all global)
    bool is_bidirectional = false;  // true for EuroBERT-style encoder models
    int pooling_method = 2;  // 1=mean (BidirLM-style), 2=last-token (Qwen3/Gemma3)
    int activation = 0;  // 0=silu (SwiGLU), 1=gelu (GeGLU), 2=gelu_pytorch_tanh
    int head_dim = 0;    // explicit head_dim (Gemma3 uses 256 != hidden/heads)
    float attn_scale = 0.0f;  // query_pre_attn_scalar (0 = use 1/sqrt(head_dim))
    float embed_scale = 1.0f; // Gemma3: sqrt(hidden_size)
    bool gemma_norm = false;  // Gemma3: RMSNorm uses (1 + weight) instead of weight

    // Multimodal — BidirLM-Omni. mrope_section sums to head_dim/2 when
    // present; [0,0,0] (or absent) means standard RoPE (text-only path).
    int mrope_section[3] = {0, 0, 0};
    int vision_start_token_id = -1;
    int vision_end_token_id = -1;
    int image_token_id = -1;
    int spatial_merge_size = 1;

    ggml_tensor * token_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    std::vector<dec_layer> layers;

    // Post-pooling Dense projection layers (SentenceTransformer-style).
    // Each entry is the weight matrix data in row-major layout [out, in].
    // Applied after pooling but before L2 normalization (no bias, no activation).
    struct DenseLayer { int in_dim; int out_dim; std::vector<float> weight; };
    std::vector<DenseLayer> dense_proj;

    // LoRA adapter state for runtime hot-swap
    std::vector<lora_adapter> lora_adapters;
    std::string active_lora;  // "" = base weights only
    // Lazy F32 snapshot of base weights for LoRA-augmented projections.
    // Key = GGUF tensor name (e.g. "blk.0.attn_q.weight").
    std::map<std::string, std::vector<float>> base_weights_f32;
};

bool load_decoder_model(dec_model & m, core_gguf::WeightLoad & wl,
                         const char * path, ggml_backend_t backend);

// Optional multimodal extension: when `image_embeds` and friends are
// supplied, the decoder uses 3D MRoPE position ids and replaces token
// embeddings at `image_token_id` positions with `image_embeds` rows; at
// each of the first `n_deepstack` layers it adds `deepstack[k]` rows
// at the same positions. Pass nullptr for text-only.
struct dec_image_input {
    const float * image_embeds = nullptr;     // (n_image_tokens, n_embd)
    const float * deepstack    = nullptr;     // (n_deepstack, n_image_tokens, n_embd) flat
    int n_image_tokens = 0;                    // total across all images
    int n_deepstack = 0;
    // Per-image grid (t, h_patches, w_patches), flattened.
    const int32_t * grid_thw = nullptr;       // (n_images, 3)
    int n_images = 0;
};

std::vector<float> decoder_encode_tokens(
    const dec_model & m, ggml_backend_t backend,
    const embed_tokens & tokens, int n_threads,
    ggml_backend_sched_t sched = nullptr,
    std::vector<uint8_t> * compute_meta = nullptr,
    const dec_image_input * img = nullptr);

// Batched decoder encoding: encode multiple texts in a single graph compute.
// Returns one embedding vector per input text.
std::vector<std::vector<float>> decoder_encode_tokens_batch(
    const dec_model & m, ggml_backend_t backend,
    const std::vector<embed_tokens> & batch, int n_threads,
    ggml_backend_sched_t sched = nullptr,
    std::vector<uint8_t> * compute_meta = nullptr);

// LoRA adapter hot-swap. Merges adapter weights into base weights on CPU
// and writes the merged result back to the backend buffer. Pass "" or
// nullptr to unmerge (restore base weights). Returns true on success.
// NOT thread-safe with respect to decoder_encode_tokens().
bool decoder_set_lora(dec_model & m, ggml_backend_t backend,
                      const std::string & adapter_name);
