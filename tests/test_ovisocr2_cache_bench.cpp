// Test-only paired cold/warm process runner. No reference data enters the library.
#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>

static int run(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    const bool two = argc == 4 && std::strcmp(argv[3], "--two-contexts") == 0;
    if (argc == 4 && !two) return 2;
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    const auto started = std::chrono::steady_clock::now();
    owner a(qwen35_ocr_init(argv[1], 2), qwen35_ocr_free);
    owner b(two ? qwen35_ocr_init(argv[1], 2) : nullptr, qwen35_ocr_free);
    const double init = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!a || (two && !b) || !qwen35_ocr_is_ovisocr2(a.get())) return 2;
    if (qwen35_ocr_init("", 2)) return 1;
    auto recognition = [&](qwen35_ocr_context * c, const char * label) {
        if (!qwen35_ocr_set_max_tokens(c, 12)) return false;
        const auto begin = std::chrono::steady_clock::now();
        int length = 0;
        const char * text = qwen35_ocr_recognize_file(c, argv[2], "", &length);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        int n = 0, m = 0;
        const int * ids = qwen35_ocr_token_ids(c, &n);
        const float * probabilities = qwen35_ocr_confidences(c, &m);
        if (!text || length < 0 || !ids || !probabilities || n != m || n < 1) return false;
        std::printf(
            "CACHE_BENCH_RESULT "
            "{\"label\":\"%s\",\"contexts\":%d,\"init_seconds\":%.17g,\"recognize_seconds\":%.17g,\"token_ids\":[",
            label, two ? 2 : 1, init, seconds);
        for (int i = 0; i < n; ++i) std::printf("%s%d", i ? "," : "", ids[i]);
        std::printf("],\"confidences\":[");
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(probabilities[i]) || probabilities[i] < 0 || probabilities[i] > 1) return false;
            std::printf("%s%.9g", i ? "," : "", probabilities[i]);
        }
        std::printf("],\"raw_text_utf8_hex\":\"");
        for (int i = 0; i < length; ++i) std::printf("%02x", static_cast<unsigned char>(text[i]));
        std::printf("\",\"stop_kind\":%d,\"mean_confidence\":%.9g}\n", qwen35_ocr_last_stop(c),
                    qwen35_ocr_mean_confidence(c));
        return true;
    };
    if (!recognition(a.get(), "cold") || !recognition(a.get(), "warm")) return 1;
    int length = -1;
    if (qwen35_ocr_recognize_raw(a.get(), nullptr, 1, 1, 3, "", &length) || length != 0 ||
        qwen35_ocr_last_stop(a.get()) != QWEN35_OCR_ERROR)
        return 1;
    if (two) {
        if (!recognition(b.get(), "independent")) return 1;
        a.reset();
        if (!recognition(b.get(), "survivor")) return 1;
    }
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
