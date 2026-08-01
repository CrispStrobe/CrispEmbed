#include "ocr_detect.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <reference.gguf> <repetitions>\n", argv[0]);
        return 2;
    }
    const int repetitions = std::atoi(argv[3]);
    if (repetitions <= 0) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 3;
    const auto input = ref.get_f32("input_image");
    const auto shape = ref.shape("input_image");
    if (!input.first || shape.size() != 3) return 4;
    const int threads =
        std::max(1, std::getenv("OCR_DETECT_THREADS") ? std::atoi(std::getenv("OCR_DETECT_THREADS")) : 1);
    ocr_detect::context * ctx = nullptr;
    if (!ocr_detect::load(&ctx, argv[1], threads)) return 5;
    const auto options = ocr_detect::rapid_defaults();
    auto run = [&]() {
        return ocr_detect::detect_preprocessed_ex(ctx, input.first, (int)shape[1], (int)shape[0], options).size();
    };
    const auto cold_start = std::chrono::steady_clock::now();
    const size_t cold_boxes = run();
    const double cold_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold_start).count();
    std::vector<double> warm_ms;
    warm_ms.reserve(repetitions);
    size_t warm_boxes = 0;
    for (int i = 0; i < repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        warm_boxes = run();
        warm_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    const double mean = std::accumulate(warm_ms.begin(), warm_ms.end(), 0.0) / warm_ms.size();
    std::printf("dbnet-benchmark threads=%d repetitions=%d cold_ms=%.3f warm_ms=%.3f cold_boxes=%zu warm_boxes=%zu\n",
                threads, repetitions, cold_ms, mean, cold_boxes, warm_boxes);
    ocr_detect::free(ctx);
    core_util::clean_exit(0);
    return 0;
}
