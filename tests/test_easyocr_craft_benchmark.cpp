#include "easyocr_craft.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <craft.gguf> <reference.gguf> <repetitions>\n", argv[0]);
        return 2;
    }
    const int repetitions = std::atoi(argv[3]);
    if (repetitions <= 0) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 3;
    const auto input = ref.get_f32("input_image");
    const auto shape = ref.shape("input_image");
    if (!input.first || shape.size() != 4) return 4;
    auto * ctx = easyocr_craft_init(argv[1], (int)shape[0], (int)shape[1]);
    if (!ctx) return 5;
    std::vector<double> graph;
    graph.reserve(repetitions);
    int boxes = -1;
    for (int i = 0; i < repetitions; ++i) {
        if (!easyocr_craft_forward(ctx, input.first, input.second)) {
            easyocr_craft_free(ctx);
            return 6;
        }
        easyocr_craft_timing timing = {};
        if (!easyocr_craft_last_timing(ctx, &timing)) {
            easyocr_craft_free(ctx);
            return 7;
        }
        graph.push_back(timing.graph_ms);
        boxes = easyocr_craft_box_count(ctx, 0.7f, 0.4f, 0.4f);
    }
    const double mean = std::accumulate(graph.begin(), graph.end(), 0.0) / graph.size();
    std::printf("easyocr-craft-benchmark repetitions=%d boxes=%d graph_ms=%.3f\n", repetitions, boxes, mean);
    easyocr_craft_free(ctx);
    core_util::clean_exit(0);
    return 0;
}
