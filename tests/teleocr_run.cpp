// Resident-context TeleOCR runner for device A/B (CPU vs GPU) qualification.
// Loads the model once, runs the same image/prompt N times and prints one JSON
// object with every run's completion IDs, text, EOS flag and wall time, plus
// the process peak RSS. Repeat0 is cold; repeats 1..N-1 are warm. IDs from
// every run are retained so warm timing never hides drift or stale state.
//
//   test-teleocr-run MODEL.gguf IMAGE PROMPT [max_tokens=1024] [threads=4] [repeat=3]
//
// The device follows the production gates (CRISPEMBED_TELEOCR_GPU,
// QWEN2VL_OCR_FORCE_CPU); this runner adds no device selection of its own.
#include "qwen2vl_ocr.h"
#include "core/clean_exit.h"
#include "core/json.h"

#ifndef _WIN32
#include <sys/resource.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

static double peak_rss_mib() {
#ifdef _WIN32
    return 0.0; // diagnostic only; not measured on Windows
#else
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return double(usage.ru_maxrss) / (1024.0 * 1024.0); // bytes
#else
    return double(usage.ru_maxrss) / 1024.0; // KiB
#endif
#endif
}

static int run(int argc, char ** argv) {
    if (argc < 4 || argc > 7) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf IMAGE PROMPT [max_tokens=1024] [threads=4] [repeat=3]\n", argv[0]);
        return 2;
    }
    const int cap = argc > 4 ? std::atoi(argv[4]) : 1024;
    const int threads = argc > 5 ? std::atoi(argv[5]) : 4;
    const int repeat = argc > 6 ? std::atoi(argv[6]) : 3;
    if (cap < 1 || threads < 1 || repeat < 1 || repeat > 100) return 2;
    int width = 0, height = 0, channels = 0;
    unsigned char * pixels = stbi_load(argv[2], &width, &height, &channels, 3);
    if (!pixels) return 2;
    const auto load_start = std::chrono::steady_clock::now();
    qwen2vl_ocr_context * ctx = qwen2vl_ocr_init(argv[1], threads);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    if (!ctx) {
        stbi_image_free(pixels);
        return 2;
    }
    qwen2vl_ocr_set_max_tokens(ctx, cap);
    if (!qwen2vl_ocr_set_prompt(ctx, argv[3])) {
        qwen2vl_ocr_free(ctx);
        stbi_image_free(pixels);
        return 2;
    }
    std::string out = "{\"load_ms\":" + std::to_string(load_ms) + ",\"threads\":" + std::to_string(threads) +
                      ",\"max_tokens\":" + std::to_string(cap) + ",\"runs\":[";
    int status = 0;
    for (int r = 0; r < repeat; ++r) {
        const auto start = std::chrono::steady_clock::now();
        int text_length = 0;
        const char * result = qwen2vl_ocr_recognize_raw(ctx, pixels, width, height, 3, &text_length);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (!result) {
            status = 1;
            break;
        }
        int n = 0;
        const int32_t * ids = qwen2vl_ocr_completion_ids(ctx, &n);
        out += std::string(r ? "," : "") + "{\"ms\":" + std::to_string(ms) + ",\"ids\":[";
        for (int i = 0; i < n; ++i) out += (i ? "," : "") + std::to_string(ids[i]);
        out += "],\"eos\":" + std::string(qwen2vl_ocr_stopped_eos(ctx) ? "true" : "false") + ",\"text\":\"" +
               core_json::json_escape(std::string(result, (size_t)text_length)) + "\"}";
    }
    out += "],\"peak_rss_mib\":" + std::to_string(peak_rss_mib()) + "}";
    qwen2vl_ocr_free(ctx);
    stbi_image_free(pixels);
    std::printf("%s\n", out.c_str());
    return status;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
