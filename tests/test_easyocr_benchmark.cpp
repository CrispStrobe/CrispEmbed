#include "easyocr_ocr.h"
#include "core/clean_exit.h"
#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: %s <easyocr.gguf> <image> <repetitions> [width]\n", argv[0]);
        return 2;
    }
    const int repetitions = std::atoi(argv[3]);
    if (repetitions <= 0) return 2;
    easyocr_ocr_context * ctx = easyocr_ocr_init(argv[1], 1);
    if (!ctx) return 3;
    if (argc == 5 && !easyocr_ocr_set_width(ctx, std::atoi(argv[4]))) {
        easyocr_ocr_free(ctx);
        return 3;
    }
    int w = 0, h = 0, ch = 0;
    unsigned char * pixels = stbi_load(argv[2], &w, &h, &ch, 0);
    if (!pixels) {
        easyocr_ocr_free(ctx);
        return 4;
    }
    std::vector<double> preprocess, graph, decode, total;
    preprocess.reserve(repetitions);
    graph.reserve(repetitions);
    decode.reserve(repetitions);
    total.reserve(repetitions);
    std::string text;
    for (int i = 0; i < repetitions; ++i) {
        int len = 0;
        const char * result = easyocr_ocr_recognize(ctx, pixels, w, h, ch, &len);
        easyocr_ocr_timing timing = {};
        if (!result || !easyocr_ocr_last_timing(ctx, &timing)) {
            stbi_image_free(pixels);
            easyocr_ocr_free(ctx);
            return 5;
        }
        text.assign(result, (size_t)len);
        preprocess.push_back(timing.preprocess_ms);
        graph.push_back(timing.graph_ms);
        decode.push_back(timing.decode_ms);
        total.push_back(timing.total_ms);
    }
    auto mean = [](const std::vector<double> & values) {
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };
    std::printf(
        "easyocr-benchmark repetitions=%d text=%s preprocess_ms=%.3f graph_ms=%.3f decode_ms=%.3f total_ms=%.3f\n",
        repetitions, text.c_str(), mean(preprocess), mean(graph), mean(decode), mean(total));
    stbi_image_free(pixels);
    easyocr_ocr_free(ctx);
    core_util::clean_exit(0);
    return 0;
}
