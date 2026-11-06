#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_math.h"
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
static std::atomic<int> vec_calls{ 0 }, llama_calls{ 0 }, llama_used{ 0 }, operand_matches{ 0 };
extern "C" void __real_ggml_vec_dot_f32(int, float *, size_t, const float *, size_t, const float *, size_t, int);
#ifdef CRISPEMBED_TRACE_SHARED
#define TRACE_VEC ggml_vec_dot_f32
#else
#define TRACE_VEC __wrap_ggml_vec_dot_f32
#endif
extern "C" void TRACE_VEC(int n, float * s, size_t bs, const float * x, size_t bx, const float * y, size_t by,
                          int nrc) {
    ++vec_calls;
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
    ++llama_calls;
    if (m == 9216 && n == 2 && k == 2560 && lda == 2560 && ldb == 2560 && at == GGML_TYPE_F32 && bt == GGML_TYPE_F32 &&
        expected_a && expected_b && !std::memcmp(a, expected_a, size_t(9216) * 2560 * 4) &&
        !std::memcmp(b, expected_b, size_t(2) * 2560 * 4))
        ++operand_matches;
#ifdef CRISPEMBED_TRACE_SHARED
    static auto real = reinterpret_cast<decltype(&__real_llamafile_sgemm)>(dlsym(RTLD_NEXT, "llamafile_sgemm"));
    if (!real) std::abort();
    const bool used = real(p, m, n, k, a, lda, b, ldb, c, ldc, at, bt, ct);
#else
    const bool used = __real_llamafile_sgemm(p, m, n, k, a, lda, b, ldb, c, ldc, at, bt, ct);
#endif
    if (used) ++llama_used;
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
static int run(int argc, char ** argv) {
    if (argc != 4) return 2;
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", "1");
#else
    setenv("CRISPEMBED_QWEN35_OCR", "1", 1);
#endif
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "lighton4b-exact-dense-trace", loaded)) return 2;
    auto * weight = core_gguf::try_get(loaded.tensors, "gate.weight");
    if (!weight || weight->type != GGML_TYPE_F32 || weight->ne[0] != 2560 || weight->ne[1] != 9216) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    const auto input = ref.get_f32("input"), selected = ref.get_f32("selected_indices");
    if (!input.first || input.second != 5120 || !selected.first || selected.second != 2) return 2;
    expected_a = static_cast<const float *>(weight->data);
    expected_b = input.first;
    std::vector<std::vector<float>> outputs;
    if (!qwen35_ocr::project_cpu({ weight }, input.first, 2560, 2, 2, outputs)) return 1;
    const int project_vec = vec_calls.load(), project_llama = llama_calls.load(), project_used = llama_used.load(),
              project_matches = operand_matches.load();
    bool ok = true;
    auto emit = [&](const std::string & name, const std::vector<float> & values) {
        std::ofstream f(std::string(argv[3]) + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(values.data()), values.size() * 4);
        ok &= bool(f);
    };
    emit("project", outputs[0]);
    std::vector<float> fma8(18432), separate8(18432), fma32(18432), seqfma(18432), seqseparate(18432),
        double_products(18432), rounded_products_double(18432), direct(18432);
    for (int token = 0; token < 2; ++token)
        for (int row = 0; row < 9216; ++row) {
            const size_t at = size_t(token) * 9216 + row;
            const auto * w = expected_a + size_t(row) * 2560;
            const auto * x = input.first + token * 2560;
            fma8[at] = lane8(w, x, true);
            separate8[at] = lane8(w, x, false);
            fma32[at] = lane32(w, x);
            float sf = 0, ss = 0;
            double ds = 0, rs = 0;
            for (int k = 0; k < 2560; ++k) {
                sf = std::fma(w[k], x[k], sf);
                const float product = mul(w[k], x[k]);
                ss = add(ss, product);
                ds += double(w[k]) * double(x[k]);
                rs += double(product);
            }
            seqfma[at] = sf;
            seqseparate[at] = ss;
            double_products[at] = float(ds);
            rounded_products_double[at] = float(rs);
            ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot(2560, &direct[at], 0, w, 0, x, 0, 1);
        }
    emit("lane8_fma", fma8);
    emit("lane8_separate", separate8);
    emit("lane32_fma", fma32);
    emit("sequential_fma", seqfma);
    emit("sequential_separate", seqseparate);
    emit("f64_products_sum", double_products);
    emit("rounded_f32_products_f64_sum", rounded_products_double);
    emit("direct_vecdot", direct);
    for (int which = 0; which < 2; ++which) {
        const int index = int(selected.first[which]);
        if (index < 0 || index >= 18432) return 2;
        const auto * w = expected_a + size_t(index % 9216) * 2560;
        const auto * x = input.first + (index / 9216) * 2560;
        std::ofstream csv(std::string(argv[3]) + "/trace-" + std::to_string(which) + ".csv");
        csv << "k,lane,weight,input,rounded_product,before,fused_after,separate_from_same_before\n" << std::hexfloat;
        float s[8] = {};
        std::vector<float> products(2560);
        for (int k = 0; k < 2560; ++k) {
            const int lane = k % 8;
            const float before = s[lane], product = mul(w[k], x[k]), fused = std::fma(w[k], x[k], before),
                        separate = add(before, product);
            csv << k << ',' << lane << ',' << w[k] << ',' << x[k] << ',' << product << ',' << before << ',' << fused
                << ',' << separate << '\n';
            s[lane] = fused;
            products[k] = product;
        }
        emit("products-" + std::to_string(which), products);
        ok &= bool(csv);
    }
    std::printf("DENSE_TRACE complete=%d project_vec_calls=%d project_llama_calls=%d project_llama_used=%d "
                "exact_operand_calls=%d decoder_repeated=0\n",
                ok, project_vec, project_llama, project_used, project_matches);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
