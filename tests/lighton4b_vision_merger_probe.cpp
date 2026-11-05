// Test-only source-fed merger, never linked into the production target.
#include "lighton4b_vision_bf16.cpp"
#include "ggml-cpu.h"
#include <fstream>
#include <stdexcept>
using namespace qwen35_ocr;
#include "vision_merger_expression.inc"
// Source expression from installed PyTorch Vectorized<float>::erf, BSD notice
// retained in core/third_party/pytorch/LICENSE. Test-only until measured.
static float source_erf(float x) {
    const float t = 1.0f / std::fma(0.3275911f, std::abs(x), 1.0f);
    float r = std::fma(1.061405429f, t, -1.453152027f);
    r = std::fma(r, t, 1.421413741f);
    r = std::fma(r, t, -0.284496736f);
    r = std::fma(r, t, 0.254829592f);
    const float square = x * x;
    const float negative_exp = -source_avx2_exp(-square);
    const float partial = negative_exp * t;
    const float result = std::fma(partial, r, 1.0f);
    return std::signbit(x) ? -result : result;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    const std::string dir = argv[1];
    auto read = [&](const std::string & name, size_t n) {
        std::vector<float> x(n);
        std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.read(reinterpret_cast<char *>(x.data()), n * 4);
        if (!f || f.peek() != EOF) throw std::runtime_error("source length differs");
        return x;
    };
    auto emit = [&](const std::string & name, const std::vector<float> & x) {
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4);
        if (!f) throw std::runtime_error("write failed");
    };
    auto * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    if (!ctx) return 3;
    std::vector<std::pair<ggml_tensor *, std::vector<float>>> weights;
    auto tensor = [&](const std::string & name, int columns, int rows) {
        auto * t = rows == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, columns)
                             : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
        weights.emplace_back(t, read("weight." + name, size_t(columns) * rows));
        return t;
    };
    auto *norm_w = tensor("norm.weight", 1024, 1), *norm_b = tensor("norm.bias", 1024, 1);
    auto *up_w = tensor("up.weight", 4096, 4096), *up_b = tensor("up.bias", 4096, 1);
    auto *down_w = tensor("down.weight", 4096, 2560), *down_b = tensor("down.bias", 2560, 1);
    auto * backend = ggml_backend_cpu_init();
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!backend || !buffer) return 4;
    for (auto & e : weights) ggml_backend_tensor_set(e.first, e.second.data(), 0, e.second.size() * 4);
    for (const std::string label : { "source", "actual" }) {
        auto hidden = read(label + ".hidden", 560 * 1024);
        std::vector<float> norm, up, merged;
        if (!layer_norm(hidden, norm_w, norm_b, 1024, 1e-6f, norm) || !linear(up_w, up_b, norm, 4096, 140, 4, up))
            return 5;
        emit(label + ".native_norm", norm);
        emit(label + ".native_up", up);
        auto activation = up;
        production_merger_activation(activation);
        emit(label + ".native_activation", activation);
        if (!linear(down_w, down_b, activation, 4096, 140, 4, merged)) return 6;
        emit(label + ".native_merged", merged);
        for (float & x : up) x = lighton4b_bf16((x * 0.5f) * (1.0f + source_erf(x * 0.7071067811865475f)));
        emit(label + ".source_erf_activation", up);
        if (!linear(down_w, down_b, up, 4096, 140, 4, merged)) return 7;
        emit(label + ".source_erf_merged", merged);
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
