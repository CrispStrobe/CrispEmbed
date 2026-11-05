// Test-only source-fed replay: never linked into production artifacts.
#include "lighton4b_native_context.h"
#include "lighton4b_vision_bf16.cpp"
#include "core/clean_exit.h"
#include <fstream>
#include <iostream>
using namespace qwen35_ocr;
static int run(int argc, char ** argv) {
    if (argc != 4) return 2; // immutable gguf, stage directory, export|replay
    auto context = lighton4b_native_context::open(argv[1]);
    if (!context) return 3;
    const auto & model = context->model();
    const auto & w = model.vision.layers[0];
    const auto & c = model.vision_parameters;
    const std::string dir = argv[2], mode = argv[3];
    auto emit = [&](const std::string & name, const std::vector<float> & value) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(float));
        if (!f) throw std::runtime_error("write failed: " + name);
    };
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> value(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(value.data()), n * sizeof(float));
        if (!f || f.peek() != EOF || !finite(value)) throw std::runtime_error("input differs: " + name);
        return value;
    };
    if (mode == "export") {
        for (auto entry : std::vector<std::pair<std::string, ggml_tensor *>>{ { "norm1.weight", w.norm1 },
                                                                              { "norm1.bias", w.norm1_bias },
                                                                              { "norm2.weight", w.norm2 },
                                                                              { "norm2.bias", w.norm2_bias },
                                                                              { "qkv.weight", w.qkv },
                                                                              { "qkv.bias", w.qkv_bias },
                                                                              { "out.weight", w.out },
                                                                              { "out.bias", w.out_bias },
                                                                              { "up.weight", w.up },
                                                                              { "up.bias", w.up_bias },
                                                                              { "down.weight", w.down },
                                                                              { "down.bias", w.down_bias } }) {
            std::vector<float> value(ggml_nelements(entry.second));
            if (!tensor_floats(entry.second, value.data())) return 4;
            emit("weight." + entry.first, value);
        }
        return 0;
    }
    if (mode != "replay") return 2;
    const int tokens = 560, width = 1024;
    auto input = read("source.embedded", tokens * width);
    std::vector<float> value;
    if (!layer_norm(input, w.norm1, w.norm1_bias, width, c.eps, value)) return 5;
    emit("native.norm1", value);
    auto project = [&](const std::string & name, ggml_tensor * weight, ggml_tensor * b, int columns) {
        auto x = read("source." + name + ".input", size_t(tokens) * columns);
        if (!linear(weight, b, x, columns, tokens, 4, value)) throw std::runtime_error("linear failed");
        emit("native." + name, value);
    };
    project("qkv", w.qkv, w.qkv_bias, width);
    auto qkv = read("source.qkv", tokens * 3 * width);
    auto cosine = read("source.cos", tokens * 64), sine = read("source.sin", tokens * 64);
    emit("native.pooled", attention(qkv, c, tokens, { 0, tokens }, cosine, sine));
    project("out", w.out, w.out_bias, width);
    input = read("source.residual1", tokens * width);
    if (!layer_norm(input, w.norm2, w.norm2_bias, width, c.eps, value)) return 6;
    emit("native.norm2", value);
    project("up", w.up, w.up_bias, width);
    value = read("source.up", tokens * 4096);
    const bool source_gelu =
        core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_GELU_AVX2") && source_avx2_softmax_supported() && finite(value);
    for (float & x : value)
        x = lighton4b_bf16(source_gelu
                               ? lighton4b_avx2_math::gelu(x)
                               : 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))));
    emit("native.activation", value);
    project("down", w.down, w.down_bias, 4096);
    return 0;
}
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
