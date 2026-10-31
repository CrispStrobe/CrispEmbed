// Test-only Surya OCR 2.5 owning-context runner: preserves actual ids/stop/confidence evidence.
//   test-surya25-run MODEL IMAGE CAP [THREADS] [PROMPT]
// Requires CRISPEMBED_QWEN35_OCR=1, CRISPEMBED_SURYA25_FRONTEND=1, CRISPEMBED_SURYA25_INFERENCE=1.
#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#include "qwen35_profile.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

static bool number(const char * text, int low, int high, int & result) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(text, &end, 10);
    if (errno || !text[0] || !end || *end || value < low || value > high) return false;
    result = int(value);
    return true;
}
static int run(int argc, char ** argv) {
    int cap = 0, threads = 2;
    if (argc < 4 || argc > 6 || !number(argv[3], 1, 4096, cap) || (argc >= 5 && !number(argv[4], 1, 256, threads))) {
        std::fprintf(stderr, "Usage: test-surya25-run MODEL IMAGE CAP [THREADS] [PROMPT]\n");
        return 2;
    }
    const char * prompt = argc == 6 ? argv[5] : "";
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)> context(qwen35_ocr_init(argv[1], threads),
                                                                            qwen35_ocr_free);
    if (!context || !qwen35_ocr_is_surya25(context.get()) || !qwen35_ocr_set_max_tokens(context.get(), cap)) {
        std::fprintf(stderr, "Surya 2.5 context rejected model or gates\n");
        return 3;
    }
    const double init = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto recognition = std::chrono::steady_clock::now();
    int length = 0;
    qwen35_ocr::profile_record record;
    const char * text = nullptr;
    {
        qwen35_ocr::profile_capture capture(record); // CRISPEMBED_QWEN35_PROFILE=1 only
        text = qwen35_ocr_recognize_file(context.get(), argv[2], prompt, &length);
        if (capture.enabled()) qwen35_ocr::profile_print(record, true);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - recognition).count();
    int n = 0, m = 0;
    const int * ids = qwen35_ocr_token_ids(context.get(), &n);
    const float * confidence = qwen35_ocr_confidences(context.get(), &m);
    const auto stop = qwen35_ocr_last_stop(context.get());
    if (!text || length < 0 || !ids || !confidence || n < 1 || n > cap || n != m ||
        (stop != QWEN35_OCR_EOS && stop != QWEN35_OCR_MAX_TOKENS && stop != QWEN35_OCR_CONTEXT_LIMIT)) {
        std::fprintf(stderr, "Surya 2.5 OCR failed: %s\n", qwen35_ocr_last_error(context.get()));
        return 4;
    }
    for (int i = 0; i < n; ++i)
        if (ids[i] < 0 || ids[i] >= 88526 || !std::isfinite(confidence[i]) || confidence[i] < 0 || confidence[i] > 1)
            return 5;
    if ((stop == QWEN35_OCR_EOS) != (ids[n - 1] == 2)) return 6;
    std::printf("SURYA25_RESULT {\"raw_text_utf8_hex\":\"");
    for (int i = 0; i < length; ++i) std::printf("%02x", static_cast<unsigned char>(text[i]));
    std::printf("\",\"token_ids\":[");
    for (int i = 0; i < n; ++i) std::printf("%s%d", i ? "," : "", ids[i]);
    std::printf("],\"confidences\":[");
    for (int i = 0; i < n; ++i) std::printf("%s%.9g", i ? "," : "", confidence[i]);
    std::printf("],\"mean_confidence\":%.9g,\"stop_kind\":%d,\"cap\":%d,\"wall_seconds\":%.9g,\"init_seconds\":%.9g}\n",
                qwen35_ocr_mean_confidence(context.get()), int(stop), cap, seconds, init);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
