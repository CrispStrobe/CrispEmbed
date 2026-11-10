// Test-only original PyTorch header-fed moments and vector affine control.
#include <ATen/native/cpu/moments_utils.h>
#include "core/clean_exit.h"
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
using at::BFloat16;
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const std::string dir = argv[1], mode = argv[2];
    constexpr int tokens = 560, width = 1024;
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f) throw std::runtime_error("input failed");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("output failed");
    };
    using V = at::vec::AVX2::Vectorized<float>;
    for (const std::string norm : { "norm1", "norm2" }) {
        auto input = read(norm == "norm1" ? "source.embedded" : "source.residual1", tokens * width),
             w = read("weight." + norm + ".weight", width), b = read("weight." + norm + ".bias", width);
        std::vector<BFloat16> x(input.begin(), input.end());
        std::vector<float> stats(tokens * 2), y(input.size());
        for (int t = 0; t < tokens; ++t) {
            auto p = at::native::AVX2::RowwiseMoments(x.data() + t * width, width);
            const float scale = 1.0f / std::sqrt(p.second + 1e-6f), bias = -scale * p.first;
            stats[t * 2] = p.first;
            stats[t * 2 + 1] = scale;
            for (int i = 0; i < width; i += V::size()) {
                auto out = (V::loadu(input.data() + t * width + i) * V(scale) + V(bias)) * V::loadu(w.data() + i) +
                           V::loadu(b.data() + i);
                out.store(y.data() + t * width + i);
            }
        }
        for (float & v : y) v = float(BFloat16(v));
        emit(mode + "." + norm + ".stats", stats);
        emit(mode + "." + norm + ".control", y);
    }
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
