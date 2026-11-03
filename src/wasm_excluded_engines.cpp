// wasm/ browser builds only: stand-ins for the large VLM/document OCR engines.
//
// The Emscripten build (CRISPEMBED_WASM / CRISPEMBED_WASM_EMBED) leaves the
// multi-GB vision-language and document engines out of the `crispembed`
// library (see CRISPEMBED_WASM_EXCLUDED_SOURCES in CMakeLists.txt): a browser
// cannot load their weights, and their code pushed crispembed_ocr.wasm past
// its 4 MiB budget. The unified OCR dispatcher and the orchestrator still name
// these entry points, so this file supplies them. Every *_init reports that
// the engine is not part of the WASM build and returns NULL, so callers take
// their ordinary load-failure path; the remaining functions are unreachable
// without a context and return neutral values.
//
// Native builds never compile this file.

#ifndef __EMSCRIPTEN__
#error "wasm_excluded_engines.cpp is only for Emscripten builds"
#endif

#include <cstdint>
#include <cstdio>

#include "crispembed.h"
#include "deepseek_ocr2.h"
#include "glm_ocr.h"
#include "got_ocr.h"
#include "granite_vision_ocr.h"
#include "internvl2_ocr.h"
#include "lfm2_vl_ocr.h"
#include "lightonocr.h"
#include "minicpm_ocr.h"
#include "oxr_ocr.h"
#include "pix2struct.h"
#include "qwen2vl_ocr.h"
#include "qwen35_ocr.h"
#include "smoldocling_ocr.h"
#include "unlimited_ocr.h"


namespace {
void * wasm_unavailable(const char * engine, const char * path) {
    fprintf(stderr, "crispembed: %s OCR engine is not available in the WASM build (model '%s')\n", engine,
            path ? path : "(null)");
    return nullptr;
}
const char k_unavailable[] = "engine not available in the WASM build";
} // namespace


const float * deepseek_ocr2_confidences(const deepseek_ocr2_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void deepseek_ocr2_free(deepseek_ocr2_context * ctx) {
    (void)ctx;
}

deepseek_ocr2_context * deepseek_ocr2_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (deepseek_ocr2_context *)wasm_unavailable("deepseek_ocr2", model_path);
}

float deepseek_ocr2_mean_confidence(const deepseek_ocr2_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * deepseek_ocr2_recognize(deepseek_ocr2_context * ctx, const float * pixels, int width, int height,
                                     int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)out_len;
    return nullptr;
}

const char * deepseek_ocr2_recognize_raw(deepseek_ocr2_context * ctx, const uint8_t * pixels, int width, int height,
                                         int channels, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void deepseek_ocr2_set_max_tokens(deepseek_ocr2_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * glm_ocr_confidences(const glm_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void glm_ocr_free(glm_ocr_context * ctx) {
    (void)ctx;
}

glm_ocr_context * glm_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (glm_ocr_context *)wasm_unavailable("glm_ocr", model_path);
}

float glm_ocr_mean_confidence(const glm_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * glm_ocr_recognize(glm_ocr_context * ctx, const float * px, int w, int h, int * out_len) {
    (void)ctx;
    (void)px;
    (void)w;
    (void)h;
    (void)out_len;
    return nullptr;
}

const char * glm_ocr_recognize_raw(glm_ocr_context * ctx, const uint8_t * px, int w, int h, int ch, int * out_len) {
    (void)ctx;
    (void)px;
    (void)w;
    (void)h;
    (void)ch;
    (void)out_len;
    return nullptr;
}

void glm_ocr_set_max_tokens(glm_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * got_ocr_confidences(const got_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void got_ocr_free(got_ocr_context * ctx) {
    (void)ctx;
}

got_ocr_context * got_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (got_ocr_context *)wasm_unavailable("got_ocr", model_path);
}

float got_ocr_mean_confidence(const got_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * got_ocr_recognize(got_ocr_context * ctx, const float * px, int w, int h, int * out_len) {
    (void)ctx;
    (void)px;
    (void)w;
    (void)h;
    (void)out_len;
    return nullptr;
}

const char * got_ocr_recognize_raw(got_ocr_context * ctx, const uint8_t * px, int w, int h, int ch, int * out_len) {
    (void)ctx;
    (void)px;
    (void)w;
    (void)h;
    (void)ch;
    (void)out_len;
    return nullptr;
}

void got_ocr_set_max_tokens(got_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * granite_vision_confidences(const granite_vision_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void granite_vision_free(granite_vision_context * ctx) {
    (void)ctx;
}

granite_vision_context * granite_vision_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (granite_vision_context *)wasm_unavailable("granite_vision", model_path);
}

float granite_vision_mean_confidence(const granite_vision_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * granite_vision_recognize(granite_vision_context * ctx, const uint8_t * pixels, int width, int height,
                                      int channels, const char * prompt, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)prompt;
    (void)out_len;
    return nullptr;
}

void granite_vision_set_max_tokens(granite_vision_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * internvl2_ocr_confidences(const internvl2_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void internvl2_ocr_free(internvl2_ocr_context * ctx) {
    (void)ctx;
}

const char * internvl2_ocr_get_prompt(const internvl2_ocr_context * ctx) {
    (void)ctx;
    return nullptr;
}

internvl2_ocr_context * internvl2_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (internvl2_ocr_context *)wasm_unavailable("internvl2_ocr", model_path);
}

float internvl2_ocr_mean_confidence(const internvl2_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * internvl2_ocr_recognize(internvl2_ocr_context * ctx, const float * pixels, int width, int height,
                                     int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)out_len;
    return nullptr;
}

const char * internvl2_ocr_recognize_raw(internvl2_ocr_context * ctx, const uint8_t * pixel_bytes, int width,
                                         int height, int channels, int * out_len) {
    (void)ctx;
    (void)pixel_bytes;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void internvl2_ocr_set_max_tokens(internvl2_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

void internvl2_ocr_set_prompt(internvl2_ocr_context * ctx, const char * prompt) {
    (void)ctx;
    (void)prompt;
}

const float * lfm2_vl_ocr_confidences(const lfm2_vl_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void lfm2_vl_ocr_free(lfm2_vl_ocr_context * ctx) {
    (void)ctx;
}

const char * lfm2_vl_ocr_get_prompt(const lfm2_vl_ocr_context * ctx) {
    (void)ctx;
    return nullptr;
}

lfm2_vl_ocr_context * lfm2_vl_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (lfm2_vl_ocr_context *)wasm_unavailable("lfm2_vl_ocr", model_path);
}

float lfm2_vl_ocr_mean_confidence(const lfm2_vl_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * lfm2_vl_ocr_recognize_raw(lfm2_vl_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void lfm2_vl_ocr_set_max_tokens(lfm2_vl_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

void lfm2_vl_ocr_set_prompt(lfm2_vl_ocr_context * ctx, const char * prompt) {
    (void)ctx;
    (void)prompt;
}

const float * lightonocr_confidences(const lightonocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void lightonocr_free(lightonocr_context * ctx) {
    (void)ctx;
}

lightonocr_context * lightonocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (lightonocr_context *)wasm_unavailable("lightonocr", model_path);
}

float lightonocr_mean_confidence(const lightonocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * lightonocr_recognize_raw(lightonocr_context * ctx, const uint8_t * pixels, int width, int height,
                                      int channels, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void lightonocr_set_max_tokens(lightonocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * minicpm_ocr_confidences(const minicpm_ocr_context * ctx, int * count) {
    (void)ctx;
    (void)count;
    return nullptr;
}

void minicpm_ocr_free(minicpm_ocr_context * ctx) {
    (void)ctx;
}

int minicpm_ocr_get_max_tokens(const minicpm_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

minicpm_ocr_context * minicpm_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (minicpm_ocr_context *)wasm_unavailable("minicpm_ocr", model_path);
}

const char * minicpm_ocr_last_error(const minicpm_ocr_context * ctx) {
    (void)ctx;
    return k_unavailable;
}

const char * minicpm_ocr_recognize_raw(minicpm_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

int minicpm_ocr_set_max_tokens(minicpm_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
    return -1;
}

int minicpm_ocr_set_prompt(minicpm_ocr_context * ctx, const char * prompt) {
    (void)ctx;
    (void)prompt;
    return -1;
}

int oxr_ocr_capped(const oxr_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

const float * oxr_ocr_confidences(const oxr_ocr_context * ctx, int * count) {
    (void)ctx;
    (void)count;
    return nullptr;
}

void oxr_ocr_free(oxr_ocr_context * ctx) {
    (void)ctx;
}

int oxr_ocr_get_max_tokens(const oxr_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

const char * oxr_ocr_get_prompt(const oxr_ocr_context * ctx) {
    (void)ctx;
    return nullptr;
}

oxr_ocr_context * oxr_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (oxr_ocr_context *)wasm_unavailable("oxr_ocr", model_path);
}

const char * oxr_ocr_last_error(const oxr_ocr_context * ctx) {
    (void)ctx;
    return k_unavailable;
}

int oxr_ocr_last_ok(const oxr_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

float oxr_ocr_mean_confidence(const oxr_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * oxr_ocr_recognize(oxr_ocr_context * ctx, const float * grayscale, int width, int height, int * out_len) {
    (void)ctx;
    (void)grayscale;
    (void)width;
    (void)height;
    (void)out_len;
    return nullptr;
}

const char * oxr_ocr_recognize_raw(oxr_ocr_context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                   int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void oxr_ocr_set_max_tokens(oxr_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

int oxr_ocr_set_prompt(oxr_ocr_context * ctx, const char * prompt) {
    (void)ctx;
    (void)prompt;
    return -1;
}

int oxr_ocr_set_repetition_penalty(oxr_ocr_context * ctx, float penalty) {
    (void)ctx;
    (void)penalty;
    return -1;
}

int oxr_ocr_stopped_eos(const oxr_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

const float * pix2struct_confidences(const pix2struct_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

const float * pix2struct_encode_patches(pix2struct_context * ctx, const float * patches, int n_patches, int * out_dim) {
    (void)ctx;
    (void)patches;
    (void)n_patches;
    (void)out_dim;
    return nullptr;
}

void pix2struct_free(pix2struct_context * ctx) {
    (void)ctx;
}

void pix2struct_free_text(const char * text) {
    (void)text;
}

const char * pix2struct_generate(pix2struct_context * ctx, const uint8_t * image, int width, int height,
                                 int max_tokens) {
    (void)ctx;
    (void)image;
    (void)width;
    (void)height;
    (void)max_tokens;
    return nullptr;
}

pix2struct_context * pix2struct_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (pix2struct_context *)wasm_unavailable("pix2struct", model_path);
}

float pix2struct_mean_confidence(const pix2struct_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const float * qwen2vl_ocr_confidences(const qwen2vl_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void qwen2vl_ocr_free(qwen2vl_ocr_context * ctx) {
    (void)ctx;
}

int qwen2vl_ocr_get_generation_params(const qwen2vl_ocr_context * ctx, float * repetition_penalty,
                                      float * presence_penalty, float * frequency_penalty, int * no_repeat_ngram) {
    (void)ctx;
    (void)repetition_penalty;
    (void)presence_penalty;
    (void)frequency_penalty;
    (void)no_repeat_ngram;
    return 0;
}

int qwen2vl_ocr_get_max_tokens(const qwen2vl_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

const char * qwen2vl_ocr_get_prompt(const qwen2vl_ocr_context * ctx) {
    (void)ctx;
    return nullptr;
}

qwen2vl_ocr_context * qwen2vl_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (qwen2vl_ocr_context *)wasm_unavailable("qwen2vl_ocr", model_path);
}

float qwen2vl_ocr_mean_confidence(const qwen2vl_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * qwen2vl_ocr_recognize(qwen2vl_ocr_context * ctx, const float * pixels, int width, int height,
                                   int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)out_len;
    return nullptr;
}

const char * qwen2vl_ocr_recognize_raw(qwen2vl_ocr_context * ctx, const uint8_t * pixel_bytes, int width, int height,
                                       int channels, int * out_len) {
    (void)ctx;
    (void)pixel_bytes;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

int qwen2vl_ocr_set_generation_params(qwen2vl_ocr_context * ctx, float repetition_penalty, float presence_penalty,
                                      float frequency_penalty, int no_repeat_ngram) {
    (void)ctx;
    (void)repetition_penalty;
    (void)presence_penalty;
    (void)frequency_penalty;
    (void)no_repeat_ngram;
    return -1;
}

void qwen2vl_ocr_set_max_tokens(qwen2vl_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

int qwen2vl_ocr_set_prompt(qwen2vl_ocr_context * ctx, const char * prompt) {
    (void)ctx;
    (void)prompt;
    return -1;
}

const float * qwen35_ocr_confidences(const qwen35_ocr_context * ctx, int * count) {
    (void)ctx;
    (void)count;
    return nullptr;
}

void qwen35_ocr_free(qwen35_ocr_context * ctx) {
    (void)ctx;
}

int qwen35_ocr_get_max_tokens(const qwen35_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

qwen35_ocr_context * qwen35_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (qwen35_ocr_context *)wasm_unavailable("qwen35_ocr", model_path);
}

int qwen35_ocr_is_ovisocr2(const qwen35_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

int qwen35_ocr_is_ovisocr3(const qwen35_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

float qwen35_ocr_mean_confidence(const qwen35_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * qwen35_ocr_recognize_raw(qwen35_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                      int channels, const char * prompt, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)prompt;
    (void)out_len;
    return nullptr;
}

int qwen35_ocr_set_max_tokens(qwen35_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
    return -1;
}

void smoldocling_free(smoldocling_context * ctx) {
    (void)ctx;
}

smoldocling_context * smoldocling_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (smoldocling_context *)wasm_unavailable("smoldocling", model_path);
}

const char * smoldocling_recognize_raw(smoldocling_context * ctx, const uint8_t * pixels, int w, int h, int ch,
                                       int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)w;
    (void)h;
    (void)ch;
    (void)out_len;
    return nullptr;
}

void smoldocling_set_max_tokens(smoldocling_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}

const float * unlimited_ocr_confidences(const unlimited_ocr_context * ctx, int * n_tokens) {
    (void)ctx;
    (void)n_tokens;
    return nullptr;
}

void unlimited_ocr_free(unlimited_ocr_context * ctx) {
    (void)ctx;
}

unlimited_ocr_context * unlimited_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (unlimited_ocr_context *)wasm_unavailable("unlimited_ocr", model_path);
}

float unlimited_ocr_mean_confidence(const unlimited_ocr_context * ctx) {
    (void)ctx;
    return 0.0f;
}

const char * unlimited_ocr_recognize_raw(unlimited_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                         int channels, int * out_len) {
    (void)ctx;
    (void)pixels;
    (void)width;
    (void)height;
    (void)channels;
    (void)out_len;
    return nullptr;
}

void unlimited_ocr_set_max_tokens(unlimited_ocr_context * ctx, int max_tokens) {
    (void)ctx;
    (void)max_tokens;
}


// Engine-local max-token accessors declared by the unified dispatcher.

extern "C" int deepseek_ocr2_get_max_tokens(const deepseek_ocr2_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int glm_ocr_get_max_tokens(const glm_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int got_ocr_get_max_tokens(const got_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int granite_vision_get_max_tokens(const granite_vision_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int internvl2_ocr_get_max_tokens(const internvl2_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int lfm2_vl_ocr_get_max_tokens(const lfm2_vl_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int lightonocr_get_max_tokens(const lightonocr_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int smoldocling_get_max_tokens(const smoldocling_context * ctx) {
    (void)ctx;
    return 0;
}

extern "C" int unlimited_ocr_get_max_tokens(const unlimited_ocr_context * ctx) {
    (void)ctx;
    return 0;
}

// MinerU2.5-Pro public C API (the orchestrator's mineru route calls it).
crispembed_mineru_context * crispembed_mineru_init(const char * model_path, int n_threads) {
    (void)n_threads;
    return (crispembed_mineru_context *)wasm_unavailable("mineru25_pro", model_path);
}

void crispembed_mineru_free(crispembed_mineru_context * ctx) {
    (void)ctx;
}

const char * crispembed_mineru_predict(crispembed_mineru_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, const char * prompt, int max_new_tokens) {
    (void)ctx, (void)pixels, (void)width, (void)height, (void)channels, (void)prompt, (void)max_new_tokens;
    return nullptr;
}

const char * crispembed_mineru_extract(crispembed_mineru_context * ctx, const uint8_t * pixels, int width, int height,
                                       int channels, int image_analysis, int max_new_tokens) {
    (void)ctx, (void)pixels, (void)width, (void)height, (void)channels, (void)image_analysis, (void)max_new_tokens;
    return nullptr;
}

const char * crispembed_mineru_extract_document(crispembed_mineru_context * ctx, const uint8_t * pixels, int width,
                                                int height, int channels, const char * options_json,
                                                int max_new_tokens) {
    (void)ctx, (void)pixels, (void)width, (void)height, (void)channels, (void)options_json, (void)max_new_tokens;
    return nullptr;
}

const char * crispembed_mineru_last_document(crispembed_mineru_context * ctx, const char * format) {
    (void)ctx, (void)format;
    return nullptr;
}

const char * crispembed_mineru_last_markdown(const crispembed_mineru_context * ctx) {
    (void)ctx;
    return nullptr;
}

const char * crispembed_mineru_last_error(const crispembed_mineru_context * ctx) {
    (void)ctx;
    return k_unavailable;
}
