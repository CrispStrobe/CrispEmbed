// wasm/ocr_wrapper.c — Emscripten entry point for CrispEmbed OCR.
//
// This is the "main" for the WASM build. It links against the static
// crispembed library and exposes a thin JS-friendly API via Emscripten's
// EXPORTED_FUNCTIONS. The JS side calls these via ccall/cwrap.
//
// Model loading flow:
//   1. JS fetches the GGUF file via fetch() and writes it to Emscripten MEMFS
//      using FS.writeFile('/model.gguf', data)
//   2. JS calls wasm_ocr_init('/model.gguf', n_threads) which delegates to
//      crispembed_ocr_model_init — the C++ code opens the MEMFS file via
//      fopen/fread (the mmap path is disabled under __EMSCRIPTEN__).

#include "crispembed.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WASM_EXPORT
#endif

// Version string for the JS loading banner.
WASM_EXPORT
const char * wasm_ocr_version(void) {
    return "crispembed-ocr-wasm-0.2.0";
}

// Initialize an OCR context from a GGUF file already in MEMFS.
// Returns an opaque pointer (passed back to recognize/free), or NULL on failure.
WASM_EXPORT
void * wasm_ocr_init(const char * model_path, int n_threads) {
    return crispembed_ocr_model_init(model_path, n_threads);
}

// Recognize from grayscale float pixels [0..1].
// Returns a pointer to a null-terminated string (owned by ctx,
// valid until the next call). Returns NULL on failure.
// *out_len receives the string length.
WASM_EXPORT
const char * wasm_ocr_recognize_gray(void * ctx, const float * pixels,
                                     int width, int height, int * out_len) {
    return crispembed_ocr_model_recognize_gray(ctx, pixels, width, height, out_len);
}

// Recognize from raw pixel bytes (RGB/RGBA/grayscale).
// Returns a context-internal pointer valid until the next call.
WASM_EXPORT
const char * wasm_ocr_recognize(void * ctx, const uint8_t * pixel_bytes,
                                int width, int height, int channels,
                                int * out_len) {
    return crispembed_ocr_model_recognize(ctx, pixel_bytes, width, height,
                                         channels, out_len);
}

// Recognize from raw pixel bytes, returning a malloc'd copy of the result.
// The caller owns the returned string and MUST call free() on it.
// This is the preferred API for JS — avoids lifetime issues with the
// context-internal pointer from wasm_ocr_recognize.
// *out_len receives the string length (excluding null terminator).
// Returns NULL on failure.
WASM_EXPORT
char * wasm_ocr_recognize_copy(void * ctx, const uint8_t * pixel_bytes,
                               int width, int height, int channels,
                               int * out_len) {
    int len = 0;
    const char * result = crispembed_ocr_model_recognize(
        ctx, pixel_bytes, width, height, channels, &len);
    if (!result || len <= 0) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = len;
    char * copy = (char *)malloc(len + 1);
    if (!copy) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    memcpy(copy, result, len + 1);
    return copy;
}

// Return per-token confidence scores. The returned pointer is owned by ctx
// and valid until the next recognize call. *n_tokens receives the count.
WASM_EXPORT
const float * wasm_ocr_confidences(void * ctx, int * n_tokens) {
    return crispembed_ocr_model_confidences(ctx, n_tokens);
}

// Return the mean confidence of the last recognition.
WASM_EXPORT
float wasm_ocr_mean_confidence(void * ctx) {
    return crispembed_ocr_model_mean_confidence(ctx);
}

// Set maximum decode tokens (controls output length budget).
WASM_EXPORT
void wasm_ocr_set_max_tokens(void * ctx, int max_tokens) {
    crispembed_ocr_model_set_max_tokens(ctx, max_tokens);
}

// Free the OCR context.
WASM_EXPORT
void wasm_ocr_free(void * ctx) {
    crispembed_ocr_model_free(ctx);
}

// Emscripten requires a main() for executables.
int main(void) {
    return 0;
}
