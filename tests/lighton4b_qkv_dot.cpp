#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_math.h"
#include "lighton4b_bf16_linear.h"
#include "ggml-cpu.h"
#include <atomic>
#ifdef CRISPEMBED_TRACE_SHARED
#include <dlfcn.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>
struct ggml_compute_params;
static const float *expected_a = nullptr, *expected_b = nullptr;
static int64_t expected_tokens = 0;
static std::atomic<bool> observing{ false };
static std::atomic<int> qkv_calls{ 0 }, qkv_used{ 0 };
static std::atomic<int> vec_calls{ 0 }, llama_calls{ 0 }, llama_used{ 0 }, operand_matches{ 0 };
extern "C" void __real_ggml_vec_dot_f32(int, float *, size_t, const float *, size_t, const float *, size_t, int);
#ifdef CRISPEMBED_TRACE_SHARED
#define TRACE_VEC ggml_vec_dot_f32
#else
#define TRACE_VEC __wrap_ggml_vec_dot_f32
#endif
extern "C" void TRACE_VEC(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by,
                          int nrc) {
    if (observing) ++vec_calls;
#ifdef CRISPEMBED_TRACE_SHARED
    static auto real = reinterpret_cast<decltype(&__real_ggml_vec_dot_f32)>(dlsym(RTLD_NEXT, "ggml_vec_dot_f32"));
    if (!real) std::abort();
    real(n, s, bs, x, bx, y, by, nrc);
#else
    __real_ggml_vec_dot_f32(n, s, bs, x, bx, y, by, nrc);
#endif
}
#ifdef CRISPEMBED_TRACE_LLAMAFILE
extern "C" bool __real_llamafile_sgemm(const ggml_compute_params *, int64_t, int64_t, int64_t, const void *, int64_t,
                                       const void *, int64_t, void *, int64_t, int, int, int);
#ifdef CRISPEMBED_TRACE_SHARED
#define TRACE_LLAMA llamafile_sgemm
#else
#define TRACE_LLAMA __wrap_llamafile_sgemm
#endif
extern "C" bool TRACE_LLAMA(const ggml_compute_params * p, int64_t m, int64_t n, int64_t k, const void * a, int64_t lda,
                            const void * b, int64_t ldb, void * c, int64_t ldc, int at, int bt, int ct) {
    const bool qkv = m == 8192 && n == expected_tokens && k == 2560;
    if (observing) {
        ++llama_calls;
        if (qkv) ++qkv_calls;
        if (qkv && lda == 2560 && ldb == 2560 && at == GGML_TYPE_F32 && bt == GGML_TYPE_F32 && ct == GGML_TYPE_F32 &&
            expected_a && expected_b && !std::memcmp(a, expected_a, size_t(8192) * 2560 * 4) &&
            !std::memcmp(b, expected_b, size_t(n) * 2560 * 4))
            ++operand_matches;
    }
#ifdef CRISPEMBED_TRACE_SHARED
    static auto real = reinterpret_cast<decltype(&__real_llamafile_sgemm)>(dlsym(RTLD_NEXT, "llamafile_sgemm"));
    if (!real) std::abort();
    const bool used = real(p, m, n, k, a, lda, b, ldb, c, ldc, at, bt, ct);
#else
    const bool used = __real_llamafile_sgemm(p, m, n, k, a, lda, b, ldb, c, ldc, at, bt, ct);
#endif
    if (observing && used) {
        ++llama_used;
        if (qkv) ++qkv_used;
    }
    return used;
}
#endif
static float add(float a, float b) {
    volatile float value = a + b;
    return value;
}
static float mul(float a, float b) {
    volatile float value = a * b;
    return value;
}
static float reduce8(const float * s) {
    const float a = add(s[0], s[4]), b = add(s[1], s[5]), c = add(s[2], s[6]), d = add(s[3], s[7]);
    return add(add(a, c), add(b, d));
}
static float lane8(const float * w, const float * x, bool fused) {
    float s[8] = {};
    for (int i = 0; i < 2560; ++i) s[i % 8] = fused ? std::fma(w[i], x[i], s[i % 8]) : add(s[i % 8], mul(w[i], x[i]));
    return reduce8(s);
}
static float lane32(const float * w, const float * x) {
    float s[32] = {}, v[8];
    for (int i = 0; i < 2560; ++i) s[i % 32] = std::fma(w[i], x[i], s[i % 32]);
    for (int i = 0; i < 8; ++i) v[i] = add(add(s[i], s[16 + i]), add(s[8 + i], s[24 + i]));
    return add(add(add(v[0], v[4]), add(v[1], v[5])), add(add(v[2], v[6]), add(v[3], v[7])));
}
static float lane64(const float * w, const float * x, bool fused) {
    float v[64]{};
    for (int i = 0; i < 2560; ++i)
        v[i % 64] = fused ? std::fma(w[i], x[i], v[i % 64]) : add(v[i % 64], mul(w[i], x[i]));
    for (int step = 4; step; step /= 2)
        for (int reg = 0; reg < step; ++reg)
            for (int lane = 0; lane < 8; ++lane) v[reg * 8 + lane] = add(v[reg * 8 + lane], v[(reg + step) * 8 + lane]);
    return reduce8(v);
}
static bool read(const char * name, std::vector<float> & v) {
    std::ifstream f(name, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(v.size() * 4)) return false;
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), v.size() * 4);
    return bool(f);
}
static int run(int argc, char ** argv) {
    if (argc != 7) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "lighton4b-real-qkv-dot", weights)) return 2;
    std::vector<ggml_tensor *> matrices;
    for (auto * suffix : { "qkv", "z", "a", "b" }) {
        auto * tensor =
            core_gguf::try_get(weights.tensors, (std::string("linear_attn.in_proj_") + suffix + ".weight").c_str());
        if (!tensor) return 2;
        matrices.push_back(tensor);
    }
    std::vector<float> input(154 * 2560), independent_weight(8192 * 2560);
    if (!read(argv[2], input) || !read(argv[3], independent_weight)) return 2;
    expected_a = independent_weight.data();
    observing = std::string(argv[5]) == "observe";
    std::vector<float> raw;
    for (int offset = 0; offset < 154; offset += 16) {
        expected_tokens = std::min(16, 154 - offset);
        expected_b = input.data() + offset * 2560;
        std::vector<std::vector<float>> projected;
        if (!qwen35_ocr::project_cpu(matrices, expected_b, 2560, int(expected_tokens), 2, projected)) return 3;
        raw.insert(raw.end(), projected[0].begin(), projected[0].end());
    }
    const int calls = qkv_calls.load(), used = qkv_used.load(), matches = operand_matches.load(),
              vec = vec_calls.load();
    if (observing && (!calls || used != calls || matches != calls)) return 4;
    bool ok = true;
    auto emit = [&](const std::string & name, const std::vector<float> & value) {
        std::ofstream f(std::string(argv[4]) + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(value.data()), value.size() * 4);
        ok &= bool(f);
    };
    emit("qkv.raw", raw);
    auto rounded = raw;
    qwen35_ocr::lighton4b_bf16_round(rounded);
    emit("qkv.bf16", rounded);
    if (std::string(argv[6]) != "-") {
        std::ifstream selected(argv[6]);
        std::vector<int> indices;
        int index;
        while (selected >> index) {
            if (index < 0 || index >= 154 * 8192 || indices.size() >= 3) return 2;
            indices.push_back(index);
        }
        if (indices.empty()) return 2;
        std::ofstream values(std::string(argv[4]) + "/selected.csv");
        values << "index,lane8_fma,lane8_separate,lane32_fma,lane64_fma,lane64_separate,f64_products_sum,rounded_"
                  "products_sum\n"
               << std::hexfloat;
        for (size_t which = 0; which < indices.size(); ++which) {
            index = indices[which];
            const auto * w = expected_a + size_t(index % 8192) * 2560;
            const auto * x = input.data() + size_t(index / 8192) * 2560;
            double exact = 0, rounded_sum = 0;
            std::vector<float> products(2560), row(w, w + 2560), token(x, x + 2560);
            std::ofstream trace(std::string(argv[4]) + "/trace-" + std::to_string(which) + ".csv");
            trace << "k,lane,weight,input,rounded_product,before,fused_after,separate_from_same_before\n"
                  << std::hexfloat;
            float accum[8]{};
            for (int k = 0; k < 2560; ++k) {
                products[k] = mul(w[k], x[k]);
                exact += double(w[k]) * double(x[k]);
                rounded_sum += products[k];
                const int lane = k % 8;
                const float before = accum[lane], fused = std::fma(w[k], x[k], before),
                            separate = add(before, products[k]);
                trace << k << ',' << lane << ',' << w[k] << ',' << x[k] << ',' << products[k] << ',' << before << ','
                      << fused << ',' << separate << '\n';
                accum[lane] = fused;
            }
            emit("products-" + std::to_string(which), products);
            emit("weight-" + std::to_string(which), row);
            emit("input-" + std::to_string(which), token);
            values << index << ',' << lane8(w, x, true) << ',' << lane8(w, x, false) << ',' << lane32(w, x) << ','
                   << lane64(w, x, true) << ',' << lane64(w, x, false) << ',' << float(exact) << ','
                   << float(rounded_sum) << '\n';
        }
        // Complete projection hypotheses, not just selected convenient rows.
        std::vector<float> fma8(raw.size()), fma64(raw.size()), separate64(raw.size()), f64(raw.size());
        for (int token = 0; token < 154; ++token)
            for (int row = 0; row < 8192; ++row) {
                const size_t at = size_t(token) * 8192 + row;
                const auto * w = expected_a + size_t(row) * 2560;
                const auto * x = input.data() + size_t(token) * 2560;
                fma8[at] = lane8(w, x, true);
                fma64[at] = lane64(w, x, true);
                separate64[at] = lane64(w, x, false);
                double sum = 0;
                for (int k = 0; k < 2560; ++k) sum += double(w[k]) * double(x[k]);
                f64[at] = float(sum);
            }
        emit("lane8_fma", fma8);
        emit("lane64_fma", fma64);
        emit("lane64_separate", separate64);
        emit("f64_products_sum", f64);
        ok &= bool(values);
    }
    std::printf("QKV_DOT observed=%d qkv_calls=%d qkv_used=%d exact_operand_calls=%d vec_calls=%d complete=%d\n",
                int(observing.load()), calls, used, matches, vec, ok);
    core_gguf::free_weights(weights);
    ggml_backend_free(backend);
    return ok ? 0 : 5;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
