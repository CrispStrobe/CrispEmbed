// Original Conv3D patch fed into actual dispatch and independent sum-order probes.
#include "lighton4b_vision_bf16.cpp"
#include "ggml-cpu.h"
#include <fstream>
#include <iostream>
using namespace qwen35_ocr;
static void set(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string dir = argv[1];
    constexpr int tokens = 560, width = 1536, outputs = 1024;
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF) throw std::runtime_error("input differs");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("output failed");
    };
    auto x = read("source.patches", tokens * width), w = read("weight.patch.weight", outputs * width),
         b = read("weight.patch.bias", outputs);
    round_bf16(x);
    set("CRISPEMBED_QWEN35_OCR", "1");
    set("CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR", "1");
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    auto * tw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, outputs);
    auto * tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, outputs);
    auto * backend = ggml_backend_cpu_init();
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!ctx || !backend || !buffer) return 3;
    ggml_backend_tensor_set(tw, w.data(), 0, w.size() * 4);
    ggml_backend_tensor_set(tb, b.data(), 0, b.size() * 4);
    for (const char * gate : { "unset", "empty", "0", "1" }) {
        set("CRISPEMBED_LIGHTON4B_VISION_BF16_PATCH_LINEAR", std::string(gate) == "unset"   ? nullptr
                                                             : std::string(gate) == "empty" ? ""
                                                                                            : gate);
        std::vector<float> y;
        if (!linear(tw, tb, x, width, tokens, 4, y, false)) return 4;
        emit(std::string("patch-linear.") + gate, y);
    }
    for (const std::string mode : { "partial4_fma", "partial4_muladd", "partial4_pairwise", "double" }) {
        std::vector<float> y(size_t(tokens) * outputs);
        for (int t = 0; t < tokens; ++t)
            for (int o = 0; o < outputs; ++o) {
                const float *a = x.data() + size_t(t) * width, *v = w.data() + size_t(o) * width;
                float dot = 0;
                if (mode == "double") {
                    double exact = 0;
                    for (int i = 0; i < width; ++i) exact += double(a[i]) * double(v[i]);
                    dot = float(exact);
                } else {
                    std::array<float, 4> s{};
                    for (int i = 0; i < width; i += 4)
                        for (int lane = 0; lane < 4; ++lane)
                            s[lane] = mode == "partial4_fma" ? std::fma(a[i + lane], v[i + lane], s[lane])
                                                             : s[lane] + a[i + lane] * v[i + lane];
                    dot = mode == "partial4_pairwise" ? (s[0] + s[1]) + (s[2] + s[3]) : ((s[0] + s[1]) + s[2]) + s[3];
                }
                y[size_t(t) * outputs + o] = lighton4b_bf16(dot + b[o]);
            }
        emit(mode, y);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
