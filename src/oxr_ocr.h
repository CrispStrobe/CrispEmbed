// OXR-1.0: SigLIP2-NaViT visual tokenizer + Qwen3 native GGML inference.
// The explicit oxr GGUF profile and CRISPEMBED_OXR=1 are required.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oxr_ocr_context oxr_ocr_context;
oxr_ocr_context * oxr_ocr_init(const char * model_path, int n_threads);
void oxr_ocr_free(oxr_ocr_context * ctx);
int oxr_ocr_set_prompt(oxr_ocr_context * ctx, const char * prompt);
const char * oxr_ocr_get_prompt(const oxr_ocr_context * ctx);
// Zero resets the fixed 2048-token default. Positive values remain explicit;
// prompt+completion over the native 32768-token limit fails without clamping.
void oxr_ocr_set_max_tokens(oxr_ocr_context * ctx, int max_tokens);
int oxr_ocr_get_max_tokens(const oxr_ocr_context * ctx);
int oxr_ocr_set_repetition_penalty(oxr_ocr_context * ctx, float penalty);
float oxr_ocr_get_repetition_penalty(const oxr_ocr_context * ctx);
// Default = published vLLM profile (prompt and output history). The original
// HF inputs_embeds path has output-only history; select it explicitly for replay.
int oxr_ocr_set_hf_generation(oxr_ocr_context * ctx, int hf_output_history_only);
int oxr_ocr_get_hf_generation(const oxr_ocr_context * ctx);
// Default pixel limits match the published vLLM plugin: 200704..921600.
// Original HF preprocess_inputs defaults to 200704..2408448 instead.
int oxr_ocr_set_pixel_limits(oxr_ocr_context * ctx, int min_pixels, int max_pixels);
int oxr_ocr_get_min_pixels(const oxr_ocr_context * ctx);
int oxr_ocr_get_max_pixels(const oxr_ocr_context * ctx);

// Passing NULL pixels and zero dimensions produces a text-only ChatML request.
// Output storage belongs to ctx and lasts until the next request or destruction.
const char * oxr_ocr_recognize_raw(oxr_ocr_context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                   int * out_len);
const char * oxr_ocr_recognize(oxr_ocr_context * ctx, const float * grayscale, int width, int height, int * out_len);
const float * oxr_ocr_confidences(const oxr_ocr_context * ctx, int * count);
float oxr_ocr_mean_confidence(const oxr_ocr_context * ctx);
int oxr_ocr_last_ok(const oxr_ocr_context * ctx);
const char * oxr_ocr_last_error(const oxr_ocr_context * ctx);
const char * oxr_ocr_raw_text(const oxr_ocr_context * ctx);
const int32_t * oxr_ocr_prompt_ids(const oxr_ocr_context * ctx, int * count);
const int32_t * oxr_ocr_completion_ids(const oxr_ocr_context * ctx, int * count);
const int32_t * oxr_ocr_grid_thw(const oxr_ocr_context * ctx);
int oxr_ocr_stopped_eos(const oxr_ocr_context * ctx);
int oxr_ocr_capped(const oxr_ocr_context * ctx);
int oxr_ocr_effective_max_tokens(const oxr_ocr_context * ctx);

// Untimed diagnostic capture only. Float rows use reference/PyTorch row-major
// shape, e.g. [tokens,hidden]. Setting the observer retains graph intermediates;
// leave it NULL for ordinary execution and measured inference.
typedef void (*oxr_ocr_stage_observer)(void * user, const char * stage, const float * values, size_t count,
                                       const int64_t * shape, int ndim);
void oxr_ocr_set_stage_observer(oxr_ocr_context * ctx, oxr_ocr_stage_observer observer, void * user);

#ifdef __cplusplus
}
#endif
