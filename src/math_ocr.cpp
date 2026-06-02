// math_ocr.cpp — pix2tex math OCR inference via ggml.
//
// Architecture: hybrid CNN+ViT encoder → transformer decoder → LaTeX.
// Same pattern as CrispASR's encoder-decoder (crispasr.cpp) adapted
// for vision input and LaTeX output.
//
// Build: add this file to the CrispEmbed CMakeLists.txt target.
//        Link against ggml (already a dependency).

#include "math_ocr.h"
#include "image_preprocess.h"  // CrispEmbed's image loading utilities

#include <ggml/ggml.h>
#include <ggml/ggml-alloc.h>
#include <ggml/ggml-backend.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <memory>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

struct math_ocr_layer_enc {
    // Self-attention
    struct ggml_tensor * attn_ln_w;
    struct ggml_tensor * attn_ln_b;
    struct ggml_tensor * attn_qkv_w;  // combined Q, K, V projection
    struct ggml_tensor * attn_out_w;
    struct ggml_tensor * attn_out_b;

    // Feed-forward (GLU variant)
    struct ggml_tensor * ff_ln_w;
    struct ggml_tensor * ff_ln_b;
    struct ggml_tensor * ff_w1;  // gate
    struct ggml_tensor * ff_w2;  // up
    struct ggml_tensor * ff_out_w;
    struct ggml_tensor * ff_out_b;
};

struct math_ocr_layer_dec {
    // Self-attention
    struct ggml_tensor * self_attn_ln_w;
    struct ggml_tensor * self_attn_ln_b;
    struct ggml_tensor * self_attn_qkv_w;
    struct ggml_tensor * self_attn_out_w;
    struct ggml_tensor * self_attn_out_b;

    // Cross-attention
    struct ggml_tensor * cross_attn_ln_w;
    struct ggml_tensor * cross_attn_ln_b;
    struct ggml_tensor * cross_attn_q_w;
    struct ggml_tensor * cross_attn_kv_w;  // K, V from encoder output
    struct ggml_tensor * cross_attn_out_w;
    struct ggml_tensor * cross_attn_out_b;

    // Feed-forward (GLU)
    struct ggml_tensor * ff_ln_w;
    struct ggml_tensor * ff_ln_b;
    struct ggml_tensor * ff_w1;
    struct ggml_tensor * ff_w2;
    struct ggml_tensor * ff_out_w;
    struct ggml_tensor * ff_out_b;
};

struct math_ocr_context {
    math_ocr_hparams hparams;

    // CNN backbone (ResNet-like)
    struct ggml_tensor * cnn_conv1_w;
    struct ggml_tensor * cnn_conv1_b;
    struct ggml_tensor * cnn_bn1_w;
    struct ggml_tensor * cnn_bn1_b;
    struct ggml_tensor * cnn_bn1_mean;
    struct ggml_tensor * cnn_bn1_var;
    // ... more ResNet layers loaded dynamically

    // ViT patch embedding (after CNN features)
    struct ggml_tensor * patch_embed_w;
    struct ggml_tensor * patch_embed_b;
    struct ggml_tensor * pos_embed;

    // Encoder transformer layers
    std::vector<math_ocr_layer_enc> enc_layers;

    // Encoder output layernorm
    struct ggml_tensor * enc_ln_w;
    struct ggml_tensor * enc_ln_b;

    // Decoder token embedding
    struct ggml_tensor * token_embed;

    // Decoder positional embedding
    struct ggml_tensor * dec_pos_embed;

    // Decoder transformer layers
    std::vector<math_ocr_layer_dec> dec_layers;

    // Decoder output layernorm + projection to vocab
    struct ggml_tensor * dec_ln_w;
    struct ggml_tensor * dec_ln_b;
    struct ggml_tensor * lm_head_w;
    struct ggml_tensor * lm_head_b;

    // Tokenizer vocabulary (from GGUF metadata)
    std::vector<std::string> vocab;

    // ggml context for weights
    struct ggml_context * ctx_w;
    ggml_backend_buffer_t buf_w;

    // Inference state
    int n_threads;
    std::string result_buf;  // holds the last recognize() result

    // KV cache for decoder (reused across decoding steps)
    // TODO: implement KV cache for efficient autoregressive decoding
};

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

math_ocr_context * math_ocr_init(const char * model_path, int n_threads) {
    auto ctx = std::make_unique<math_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    // Open the GGUF file
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx->ctx_w,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, params);
    if (!gguf_ctx) {
        fprintf(stderr, "math_ocr: failed to open %s\n", model_path);
        return nullptr;
    }

    // Read hyperparameters from GGUF metadata
    auto read_u32 = [&](const char * key, int32_t def) -> int32_t {
        int idx = gguf_find_key(gguf_ctx, key);
        return idx >= 0 ? (int32_t)gguf_get_val_u32(gguf_ctx, idx) : def;
    };

    ctx->hparams.encoder_layers = read_u32("pix2tex.encoder_layers", 4);
    ctx->hparams.decoder_layers = read_u32("pix2tex.decoder_layers", 4);
    ctx->hparams.dim            = read_u32("pix2tex.dim", 256);
    ctx->hparams.heads          = read_u32("pix2tex.heads", 8);
    ctx->hparams.vocab_size     = read_u32("pix2tex.vocab_size", 8000);
    ctx->hparams.max_seq_len    = read_u32("pix2tex.max_seq_len", 512);
    ctx->hparams.patch_size     = read_u32("pix2tex.patch_size", 16);
    ctx->hparams.max_height     = read_u32("pix2tex.max_height", 192);
    ctx->hparams.max_width      = read_u32("pix2tex.max_width", 672);
    ctx->hparams.channels       = read_u32("pix2tex.channels", 1);
    ctx->hparams.bos_token      = read_u32("pix2tex.bos_token", 1);
    ctx->hparams.eos_token      = read_u32("pix2tex.eos_token", 2);
    ctx->hparams.pad_token      = read_u32("pix2tex.pad_token", 0);

    // Read tokenizer vocab from metadata
    int tok_idx = gguf_find_key(gguf_ctx, "tokenizer.tokens");
    if (tok_idx >= 0) {
        int n = gguf_get_arr_n(gguf_ctx, tok_idx);
        ctx->vocab.resize(n);
        for (int i = 0; i < n; i++) {
            ctx->vocab[i] = gguf_get_arr_str(gguf_ctx, tok_idx, i);
        }
    }

    // Load tensors from the ggml context.
    // The tensor names match the converter's output:
    //   encoder_transformer_layers_N_... for encoder
    //   decoder_net_layers_N_...          for decoder
    //
    // TODO: map tensor names to the struct fields above.
    // This is model-specific wiring that depends on the exact
    // pix2tex state_dict key naming. The converter normalizes
    // dots to underscores, so "encoder.transformer.layers.0.0.fn.to_qkv.weight"
    // becomes "encoder_transformer_layers_0_0_fn_to_qkv_weight".

    // For now, store the total tensor count for verification.
    int n_tensors = ggml_ctx_ntensors(ctx->ctx_w);
    fprintf(stderr, "math_ocr: loaded %d tensors from %s\n", n_tensors, model_path);
    fprintf(stderr, "math_ocr: enc=%d dec=%d dim=%d heads=%d vocab=%d\n",
            ctx->hparams.encoder_layers, ctx->hparams.decoder_layers,
            ctx->hparams.dim, ctx->hparams.heads, ctx->hparams.vocab_size);

    gguf_free(gguf_ctx);
    return ctx.release();
}

void math_ocr_free(math_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->ctx_w) ggml_free(ctx->ctx_w);
    delete ctx;
}

const math_ocr_hparams * math_ocr_get_hparams(const math_ocr_context * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------

static std::vector<float> preprocess_image(
    const float * pixels, int w, int h,
    int target_w, int target_h
) {
    // Resize to target dimensions (bilinear interpolation).
    std::vector<float> resized(target_w * target_h);

    float scale_x = (float)w / target_w;
    float scale_y = (float)h / target_h;

    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            float src_x = (x + 0.5f) * scale_x - 0.5f;
            float src_y = (y + 0.5f) * scale_y - 0.5f;

            int x0 = (int)std::floor(src_x);
            int y0 = (int)std::floor(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float fx = src_x - x0;
            float fy = src_y - y0;

            x0 = std::max(0, std::min(x0, w - 1));
            x1 = std::max(0, std::min(x1, w - 1));
            y0 = std::max(0, std::min(y0, h - 1));
            y1 = std::max(0, std::min(y1, h - 1));

            float v00 = pixels[y0 * w + x0];
            float v10 = pixels[y0 * w + x1];
            float v01 = pixels[y1 * w + x0];
            float v11 = pixels[y1 * w + x1];

            resized[y * target_w + x] =
                v00 * (1 - fx) * (1 - fy) +
                v10 * fx * (1 - fy) +
                v01 * (1 - fx) * fy +
                v11 * fx * fy;
        }
    }

    // Normalize to [-1, 1] (pix2tex convention: ImageNet-like).
    for (auto & v : resized) {
        v = (v - 0.5f) / 0.5f;
    }

    return resized;
}

// ---------------------------------------------------------------------------
// Encoder forward pass
// ---------------------------------------------------------------------------

// TODO: implement the hybrid CNN + ViT encoder forward pass.
// The CNN backbone (ResNet) processes the image into feature maps,
// then the ViT transformer operates on the flattened patch embeddings.
// The output is a sequence of encoder hidden states that the decoder
// cross-attends to.
//
// This is the same structure as CrispEmbed's SigLIP encoder
// (bidirlm_vision.cpp) but with a ResNet front-end instead of a
// linear patch embedding.

// ---------------------------------------------------------------------------
// Decoder forward pass (autoregressive)
// ---------------------------------------------------------------------------

// TODO: implement the transformer decoder with:
//   1. Token embedding lookup
//   2. Positional embedding addition
//   3. For each layer:
//      a. Self-attention (causal mask)
//      b. Cross-attention to encoder output
//      c. Feed-forward (GLU)
//   4. Final layernorm + linear projection to vocab logits
//
// The decoder uses a KV-cache for efficient autoregressive generation
// (only the last token's Q vector is computed on each step; K and V
// for previous tokens are cached). Same pattern as CrispASR's
// decoder (crispasr.cpp lines ~2300-2600).

// ---------------------------------------------------------------------------
// Greedy decoding
// ---------------------------------------------------------------------------

static std::string greedy_decode(
    math_ocr_context * ctx,
    const float * encoder_output,  // [n_patches, dim]
    int n_patches
) {
    const auto & hp = ctx->hparams;
    std::vector<int> tokens = {hp.bos_token};
    std::string result;

    for (int step = 0; step < hp.max_seq_len; step++) {
        // TODO: run decoder forward pass on current token sequence
        // to get logits[vocab_size].
        //
        // For now, return a placeholder.
        break;
    }

    // Detokenize
    for (size_t i = 1; i < tokens.size(); i++) {
        int tok = tokens[i];
        if (tok == hp.eos_token) break;
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            result += ctx->vocab[tok];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const char * math_ocr_recognize(
    math_ocr_context * ctx,
    const float * pixels,
    int width,
    int height,
    int * out_len
) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    const auto & hp = ctx->hparams;

    // 1. Preprocess: resize to model's expected dimensions.
    // pix2tex uses variable-size input; clamp to max.
    int target_w = std::min(width, hp.max_width);
    int target_h = std::min(height, hp.max_height);
    // Round to patch_size multiples.
    target_w = (target_w / hp.patch_size) * hp.patch_size;
    target_h = (target_h / hp.patch_size) * hp.patch_size;
    if (target_w == 0) target_w = hp.patch_size;
    if (target_h == 0) target_h = hp.patch_size;

    auto preprocessed = preprocess_image(pixels, width, height, target_w, target_h);

    // 2. Run encoder.
    int n_patches = (target_w / hp.patch_size) * (target_h / hp.patch_size);
    // TODO: actual encoder forward pass.
    // For now, allocate a zero encoder output.
    std::vector<float> encoder_output(n_patches * hp.dim, 0.0f);

    // 3. Run decoder (greedy).
    ctx->result_buf = greedy_decode(ctx, encoder_output.data(), n_patches);

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * math_ocr_recognize_file(
    math_ocr_context * ctx,
    const char * image_path,
    int * out_len
) {
    if (!ctx || !image_path) return nullptr;

    // Use CrispEmbed's image loading utility.
    // TODO: load image from file, convert to grayscale float.
    // For now, return nullptr (not implemented).
    (void)out_len;
    fprintf(stderr, "math_ocr: recognize_file not yet implemented\n");
    return nullptr;
}

const char * math_ocr_recognize_raw(
    math_ocr_context * ctx,
    const uint8_t * pixel_bytes,
    int width,
    int height,
    int channels,
    int * out_len
) {
    if (!ctx || !pixel_bytes || width <= 0 || height <= 0) return nullptr;

    // Convert to grayscale float [0, 1].
    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        if (channels == 1) {
            gray[i] = pixel_bytes[i] / 255.0f;
        } else if (channels == 3) {
            // Luminance: 0.299R + 0.587G + 0.114B
            int base = i * 3;
            gray[i] = (0.299f * pixel_bytes[base] +
                        0.587f * pixel_bytes[base + 1] +
                        0.114f * pixel_bytes[base + 2]) / 255.0f;
        } else if (channels == 4) {
            int base = i * 4;
            gray[i] = (0.299f * pixel_bytes[base] +
                        0.587f * pixel_bytes[base + 1] +
                        0.114f * pixel_bytes[base + 2]) / 255.0f;
        }
    }

    return math_ocr_recognize(ctx, gray.data(), width, height, out_len);
}
