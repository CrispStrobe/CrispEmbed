// Test-only selected BF16 QKV dot attribution; no model ownership or inference.
// Scalar adaptation of PyTorch v2.13 ReducedPrecisionFloatGemvFastPathKernel.
// PyTorch BSD notice: core/third_party/pytorch/LICENSE; ggml MIT notice: ggml/LICENSE.
#include "lighton4b_vision_bf16.h"
#include "core/clean_exit.h"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using qwen35_ocr::lighton4b_bf16;
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string dir = argv[1];
    constexpr int width = 1024, rows = 3072, tokens = 560;
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF) throw std::runtime_error("input length differs");
        return x;
    };
    auto x = read("source.qkv.input", tokens * width), w = read("weight.qkv.weight", rows * width),
         b = read("weight.qkv.bias", rows);
    std::ifstream ids(dir + "/selected.ids");
    std::vector<int> selected;
    int id;
    while (ids >> id) selected.push_back(id);
    if (selected.empty() || selected.size() > 8192) return 3;
    for (const std::string mode :
         { "register8_fma", "register8_muladd", "double_bias", "double_then_float_bias", "serial_fma" }) {
        std::vector<float> out;
        for (int index : selected) {
            if (index < 0 || index >= rows * tokens) return 4;
            const auto * a = x.data() + size_t(index / rows) * width;
            const auto * v = w.data() + size_t(index % rows) * width;
            float dot = 0;
            if (mode.rfind("register8_", 0) == 0) {
                std::array<std::array<float, 8>, 8> s{};
                for (int block = 0; block < width; block += 64)
                    for (int reg = 0; reg < 8; ++reg)
                        for (int lane = 0; lane < 8; ++lane) {
                            int k = block + reg * 8 + lane;
                            s[reg][lane] = mode == "register8_fma" ? std::fma(a[k], v[k], s[reg][lane])
                                                                   : s[reg][lane] + a[k] * v[k];
                        }
                for (int offset = 4; offset; offset /= 2)
                    for (int reg = 0; reg < offset; ++reg)
                        for (int lane = 0; lane < 8; ++lane) s[reg][lane] += s[reg + offset][lane];
                const auto & z = s[0];
                dot = ((z[0] + z[4]) + (z[2] + z[6])) + ((z[1] + z[5]) + (z[3] + z[7]));
                dot += 0.0f; // Source's empty vector-tail reduction is still added.
            } else if (mode == "serial_fma") {
                for (int k = 0; k < width; ++k) dot = std::fma(a[k], v[k], dot);
            } else {
                double exact = 0;
                for (int k = 0; k < width; ++k) exact += double(a[k]) * double(v[k]);
                if (mode == "double_bias") {
                    out.push_back(lighton4b_bf16(float(exact + double(b[index % rows]))));
                    continue;
                }
                dot = float(exact);
            }
            out.push_back(lighton4b_bf16(dot + b[index % rows]));
        }
        std::ofstream f(dir + "/" + mode + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(out.data()), out.size() * 4);
        if (!f) return 5;
    }
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
