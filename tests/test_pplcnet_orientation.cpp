#include "core/clean_exit.h"
#include "pplcnet_orientation.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: %s <pplcnet-orientation.gguf> [image]\n", argv[0]);
        return 2;
    }
    auto * ctx = pplcnet_orientation::init(argv[1], 1);
    if (!ctx) return 3;
    auto r = argc == 3 ? pplcnet_orientation::classify_file(ctx, argv[2])
                       : pplcnet_orientation::classify_raw(ctx, std::vector<uint8_t>((size_t)160 * 80 * 3, 255).data(),
                                                           160, 80, 3);
    std::printf("pplcnet-orientation angle=%d confidence=%.6f p0=%.6f p180=%.6f logit0=%.9g logit180=%.9g\n", r.angle,
                r.confidence, r.probabilities[0], r.probabilities[1], r.logits[0], r.logits[1]);
    const bool valid = (r.angle == 0 || r.angle == 180) && r.confidence >= 0.5f && r.probabilities[0] >= 0.0f &&
                       r.probabilities[1] >= 0.0f;
    pplcnet_orientation::free(ctx);
    bool parity = true;
    if (argc == 3 && std::getenv("PPLCNET_ORIENTATION_GRAPH_PARITY")) {
        auto * cpu = pplcnet_orientation::init(argv[1], 1);
        const auto cpu_result = pplcnet_orientation::classify_file(cpu, argv[2]);
        pplcnet_orientation::free(cpu);
        setenv("PPLCNET_ORIENTATION_GRAPH", "1", 1);
        setenv("PPLCNET_ORIENTATION_GRAPH_ACCEPT", "1", 1);
        auto * graph = pplcnet_orientation::init(argv[1], 1);
        const auto graph_result = pplcnet_orientation::classify_file(graph, argv[2]);
        pplcnet_orientation::free(graph);
        const float d0 = std::fabs(cpu_result.logits[0] - graph_result.logits[0]);
        const float d1 = std::fabs(cpu_result.logits[1] - graph_result.logits[1]);
        parity = cpu_result.angle == graph_result.angle && d0 < 0.1f && d1 < 0.1f;
        std::printf("pplcnet-orientation parity angle=%d/%d logit_delta=%.6f/%.6f %s\n", cpu_result.angle,
                    graph_result.angle, d0, d1, parity ? "PASS" : "FAIL");
    }
    core_util::clean_exit(valid && parity ? 0 : 4);
    return valid && parity ? 0 : 4;
}
