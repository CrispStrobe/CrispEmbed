#include "lighton4b_physical_q6_tiled.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
namespace qwen35_ocr {
bool lighton4b_physical_q6_tiled_available() {
#if !defined(__EMSCRIPTEN__) && defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#else
    return false;
#endif
}
namespace {
// GGUF K-quant storage layouts; assertions protect the public converter ABI.
struct q6_block {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    ggml_fp16_t d;
};
struct q8_block {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(q6_block) == 210 && offsetof(q6_block, d) == 208, "Q6_K layout");
static_assert(sizeof(q8_block) == 292 && offsetof(q8_block, qs) == 4 && offsetof(q8_block, bsums) == 260,
              "Q8_K layout");
#if !defined(__EMSCRIPTEN__) && defined(__linux__) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma,f16c")))
#endif
void compute_row(const uint8_t * packed, const q8_block * activations, int blocks, int rows, int tokens,
                 int row, float * output) {
    std::array<float, 144> acc{};
    for (int b = 0; b < blocks; ++b) {
        q6_block weight;
        std::memcpy(&weight, packed + (size_t(row) * blocks + b) * sizeof(weight), sizeof(weight));
        std::array<int16_t, 256> coefficients;
        for (int u = 0; u < 256; ++u) {
            const int half = u / 128, local = u % 128, group = local / 32, lane = local % 32;
            const int low = (weight.ql[half * 64 + (group % 2) * 32 + lane] >> (group >= 2 ? 4 : 0)) & 15;
            const int high = (weight.qh[half * 32 + lane] >> (2 * group)) & 3;
            coefficients[u] = int16_t(((low | (high << 4)) - 32) * int(weight.scales[u / 16]));
        }
        const float dw = ggml_fp16_to_fp32(weight.d);
        for (int token = 0; token < tokens; ++token) {
            const auto & activation = activations[size_t(token) * blocks + b];
            int32_t sum = 0;
            for (int u = 0; u < 256; ++u) sum += int32_t(coefficients[u]) * int32_t(activation.qs[u]);
            // Source tiled AVX2 sums integers first, then applies weight scale
            // and FMA activation scale once per256-element block, in K order.
            const float scaled = dw * float(sum);
            acc[token] = std::fma(scaled, activation.d, acc[token]);
        }
    }
    for (int token = 0; token < tokens; ++token) output[size_t(token) * rows + row] = acc[token];
}
struct join_threads {
    std::vector<std::thread> & threads;
    ~join_threads() {
        for (auto & thread : threads)
            if (thread.joinable()) thread.join();
    }
};
} // namespace
bool lighton4b_physical_q6_tiled_cpu(const void * packed, size_t bytes, const float * input, int width, int rows,
                                     int tokens, int threads, std::vector<float> & output) {
    output.clear();
#if defined(__EMSCRIPTEN__)
    (void)packed;
    (void)bytes;
    (void)input;
    (void)width;
    (void)rows;
    (void)tokens;
    (void)threads;
    return false;
#else
    if (!lighton4b_physical_q6_tiled_available() || !packed || !input || width < 256 || width > 32768 || width % 256 ||
        rows < 1 || rows > 300000 || tokens < 8 || tokens > 144 || threads < 1 || threads > 16 ||
        size_t(rows) * tokens > 16ull * 1024 * 1024 || bytes > 1ull * 1024 * 1024 * 1024 ||
        bytes != size_t(rows) * (width / 256) * sizeof(q6_block) ||
        size_t(tokens) * (width / 256) * sizeof(q8_block) > 64ull * 1024 * 1024 ||
        !std::all_of(input, input + size_t(width) * tokens, [](float v) { return std::isfinite(v); }))
        return false;
    const auto * trait = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    if (!trait || !trait->from_float || ggml_type_size(GGML_TYPE_Q8_K) != sizeof(q8_block)) return false;
    try {
        const int blocks = width / 256;
        std::vector<q8_block> activations(size_t(tokens) * blocks);
        for (int token = 0; token < tokens; ++token)
            trait->from_float(input + size_t(token) * width, activations.data() + size_t(token) * blocks, width);
        std::vector<float> complete(size_t(rows) * tokens);
        std::atomic<int> next{ 0 };
        auto run = [&]() {
            for (int row; (row = next.fetch_add(1, std::memory_order_relaxed)) < rows;)
                compute_row(static_cast<const uint8_t *>(packed), activations.data(), blocks, rows, tokens, row,
                            complete.data());
        };
        {
            std::vector<std::thread> workers;
            join_threads guard{ workers };
            for (int t = 1; t < std::min(threads, rows); ++t) workers.emplace_back(run);
            run();
        }
        if (!std::all_of(complete.begin(), complete.end(), [](float v) { return std::isfinite(v); })) return false;
        output = std::move(complete);
        return true;
    } catch (...) {
        output.clear();
        return false;
    }
#endif
}
} // namespace qwen35_ocr
