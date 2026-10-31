// Test-only Xiaomi-OCR-0 driver: run one GGUF over owned page images and print
// ids, raw text and wall time. No reference arrays are injected and no custom
// arithmetic is used; this only exercises the production qwen35 OCR context.
#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static int run(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf THREADS IMAGE [IMAGE ...]\n", argv[0]);
        return 2;
    }
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_XIAOMI_OCR_FRONTEND", "1");
    gate("CRISPEMBED_XIAOMI_OCR_INFERENCE", "1");
    qwen35_ocr_context * ctx = qwen35_ocr_init(argv[1], std::atoi(argv[2]));
    if (!ctx) {
        std::fprintf(stderr, "ERROR init failed: %s\n", qwen35_ocr_last_error(ctx));
        return 1;
    }
    int failures = 0;
    for (int i = 3; i < argc; ++i) {
        int length = 0;
        const auto started = std::chrono::steady_clock::now();
        const char * text = qwen35_ocr_recognize_file(ctx, argv[i], "", &length);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        int ids = 0;
        const int * tokens = qwen35_ocr_token_ids(ctx, &ids);
        if (!text) {
            std::printf("CASE %s STATUS error stop=%d message=%s\n", argv[i], int(qwen35_ocr_last_stop(ctx)),
                        qwen35_ocr_last_error(ctx));
            ++failures;
            continue;
        }
        std::printf("CASE %s STATUS ok seconds=%.3f ids=%d bytes=%d stop=%d confidence=%.6f\n", argv[i], seconds, ids,
                    length, int(qwen35_ocr_last_stop(ctx)), qwen35_ocr_mean_confidence(ctx));
        std::printf("IDS");
        for (int t = 0; t < ids; ++t) std::printf(" %d", tokens ? tokens[t] : -1);
        std::printf("\nTEXT_BEGIN\n%.*s\nTEXT_END\n", length, text);
        std::fflush(stdout);
    }
    qwen35_ocr_free(ctx);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
