// Test-only normalization attribution, consuming retained inputs and weights.
#include "lighton4b_vision_bf16.cpp"
#include <fstream>
#include <iostream>
using namespace qwen35_ocr;
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string dir = argv[1];
    constexpr int tokens = 560, width = 1024;
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF || !finite(x)) throw std::runtime_error("invalid retained input");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("output failed");
    };
    for (const std::string norm : { "norm1", "norm2" }) {
        auto x = read(norm == "norm1" ? "source.embedded" : "source.residual1", tokens * width);
        auto w = read("weight." + norm + ".weight", width), b = read("weight." + norm + ".bias", width);
        for (const std::string mode : { "old4", "bf16_8", "source_off", "source_fast" }) {
            std::vector<float> stats(tokens * 2);
            if (mode.rfind("source_", 0) == 0)
                stats = read(mode + "." + norm + ".stats", tokens * 2);
            else
                for (int t = 0; t < tokens; ++t) {
                    auto p = mode == "old4" ? row(x.data() + t * width, width, 1e-6f)
                                            : row_bf16_avx2(x.data() + t * width, width, 1e-6f);
                    stats[t * 2] = p.first;
                    stats[t * 2 + 1] = p.second;
                }
            emit(mode + "." + norm + ".stats", stats);
            for (int fma_first = 0; fma_first < 2; ++fma_first)
                for (int fma_second = 0; fma_second < 2; ++fma_second) {
                    std::vector<float> y(x.size());
                    for (int t = 0; t < tokens; ++t) {
                        const float mean = stats[t * 2], scale = stats[t * 2 + 1], bias = -scale * mean;
                        for (int i = 0; i < width; ++i) {
                            size_t at = size_t(t) * width + i;
                            const float normalized =
                                fma_first ? std::fma(x[at], scale, bias) : multiply_separately(x[at], scale) + bias;
                            const float affine = fma_second ? std::fma(normalized, w[i], b[i])
                                                            : multiply_separately(normalized, w[i]) + b[i];
                            y[at] = lighton4b_bf16(affine);
                        }
                    }
                    emit(mode + "." + norm + ".fma" + std::to_string(fma_first) + std::to_string(fma_second), y);
                }
        }
    }
    return 0;
}
