// wasm/ocr_wrapper.c — Minimal Emscripten entry point for CrispEmbed math OCR.
//
// This is the "main" for the WASM build. It links against the static
// crispembed library and exposes a thin JS-friendly API via Emscripten's
// EXPORTED_FUNCTIONS. The JS side (Dart interop) calls these via ccall/cwrap.
//
// Model loading flow:
//   1. JS fetches the GGUF file via fetch() and writes it to Emscripten MEMFS
//      using FS.writeFile('/model.gguf', data)
//   2. JS calls wasm_ocr_init('/model.gguf', n_threads) which delegates to
//      crispembed_math_ocr_init — the C++ code opens the MEMFS file via
//      fopen/fread (the mmap path is disabled under __EMSCRIPTEN__).

#include "crispembed.h"
#include <stdint.h>
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
    return "crispembed-ocr-wasm-0.1.0";
}

// Initialize a math OCR context from a GGUF file already in MEMFS.
// Returns an opaque pointer (passed back to recognize/free), or NULL on failure.
WASM_EXPORT
void * wasm_ocr_init(const char * model_path, int n_threads) {
    return crispembed_math_ocr_init(model_path, n_threads);
}

// Recognize math from grayscale float pixels [0..1].
// Returns a pointer to a null-terminated LaTeX string (owned by ctx,
// valid until the next call). Returns NULL on failure.
// *out_len receives the string length.
WASM_EXPORT
const char * wasm_ocr_recognize_gray(void * ctx, const float * pixels,
                                     int width, int height, int * out_len) {
    return crispembed_math_ocr_recognize_gray(ctx, pixels, width, height, out_len);
}

// Recognize math from raw pixel bytes (RGB/RGBA/grayscale).
WASM_EXPORT
const char * wasm_ocr_recognize(void * ctx, const uint8_t * pixel_bytes,
                                int width, int height, int channels,
                                int * out_len) {
    return crispembed_math_ocr_recognize(ctx, pixel_bytes, width, height,
                                         channels, out_len);
}

// Free the OCR context.
WASM_EXPORT
void wasm_ocr_free(void * ctx) {
    crispembed_math_ocr_free(ctx);
}

// Emscripten requires a main() for executables.
int main(void) {
    return 0;
}
