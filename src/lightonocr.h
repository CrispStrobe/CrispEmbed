// lightonocr.h — LightOnOCR-2-1B: Pixtral ViT + Qwen3 decoder.
//
// Architecture:
//   Vision:     Pixtral ViT (24L, 1024d, 2D RoPE, SiLU FFN)
//   Projection: patch_merger (2×2 spatial merge) + 2-layer MLP + RMSNorm
//   Decoder:    Qwen3 (28L, 1024d, GQA 16/8, SwiGLU, RoPE, QK norm)
//
// Usage:
//   lightonocr::context ctx;
//   lightonocr::load(ctx, "lightonocr-1b-f16.gguf", 4);
//   auto text = lightonocr::recognize_file(ctx, "document.png");
//   lightonocr::free_(ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lightonocr {

struct context;

bool load(context &ctx, const char *gguf_path, int n_threads = 4);
void free_(context &ctx);

// Recognize text from a raw RGB image (uint8, row-major).
std::string recognize_raw(context &ctx,
                           const uint8_t *pixels, int width, int height, int channels,
                           int max_tokens = 2048);

// Recognize text from an image file (JPG/PNG/BMP).
std::string recognize_file(context &ctx, const char *image_path,
                            int max_tokens = 2048);

} // namespace lightonocr

// ── C ABI ──
#ifdef __cplusplus
extern "C" {
#endif

typedef struct lightonocr_context lightonocr_context;

lightonocr_context * lightonocr_init(const char * model_path, int n_threads);
void lightonocr_free(lightonocr_context * ctx);

void lightonocr_set_max_tokens(lightonocr_context * ctx, int max_tokens);

const char * lightonocr_recognize_raw(
    lightonocr_context * ctx,
    const uint8_t * pixels, int width, int height, int channels,
    int * out_len);

const char * lightonocr_recognize_file(
    lightonocr_context * ctx,
    const char * image_path,
    int * out_len);

const float * lightonocr_confidences(const lightonocr_context * ctx, int * n_tokens);
float lightonocr_mean_confidence(const lightonocr_context * ctx);

#ifdef __cplusplus
}
#endif
