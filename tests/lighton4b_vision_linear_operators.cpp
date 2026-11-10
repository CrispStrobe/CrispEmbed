// Actual private production linear helper against retained full source-fed operators.
#include "lighton4b_vision_bf16.cpp"
#include "core/clean_exit.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
using namespace qwen35_ocr;
static void env(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR", value, 1);
    else
        unsetenv("CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR");
#endif
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", "1");
#else
    setenv("CRISPEMBED_QWEN35_OCR", "1", 1);
#endif
    const std::string dir = argv[1];
    constexpr int tokens = 560;
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF) throw std::runtime_error("retained input differs");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("output failed");
    };
    for (const std::string name : { "qkv", "out", "up", "down" }) {
        const int width = name == "down" ? 4096 : 1024, outputs = name == "qkv" ? 3072 : name == "up" ? 4096 : 1024;
        auto x = read("source." + name + ".input", size_t(tokens) * width),
             w = read("weight." + name + ".weight", size_t(width) * outputs),
             b = read("weight." + name + ".bias", outputs);
        auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
        if (!ctx) return 3;
        auto *tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, outputs),
             *tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, outputs);
        auto * backend = ggml_backend_cpu_init();
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!backend || !buffer) return 4;
        ggml_backend_tensor_set(tw, w.data(), 0, w.size() * 4);
        ggml_backend_tensor_set(tb, b.data(), 0, b.size() * 4);
        for (const char * gate : { "unset", "empty", "0" }) {
            env(std::string(gate) == "unset" ? nullptr : std::string(gate) == "empty" ? "" : gate);
            std::vector<float> y;
            if (!linear(tw, tb, x, width, tokens, 4, y)) return 5;
            emit("old." + name + "." + gate, y);
        }
        env("1");
        for (int threads : { 1, 4 }) {
            std::vector<float> y;
            if (!linear(tw, tb, x, width, tokens, threads, y)) return 6;
            emit("candidate." + name + ".threads" + std::to_string(threads), y);
        }
        auto invalid = x;
        invalid[0] = 1.00390625f;
        std::vector<float> y;
        if (linear(tw, tb, invalid, width, tokens, 4, y)) return 7;
        invalid = x;
        invalid[0] = INFINITY;
        if (linear(tw, tb, invalid, width, tokens, 4, y)) return 8;
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    env(nullptr);
    std::cout << "LINEAR_OPERATOR_CONTROLS passed=1 non_bf16_rejected=1 nonfinite_rejected=1\n";
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
