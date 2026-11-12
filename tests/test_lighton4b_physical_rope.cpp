// Model-free source-node, explicit-position and chunk controls. No weights.
#include "lighton4b_physical_rope.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
// Fail exactly the next C++ allocation to exercise helper exception containment.
static std::atomic_bool fail_next_allocation{ false };
void * operator new(std::size_t bytes) {
    if (fail_next_allocation.exchange(false)) throw std::bad_alloc();
    if (void * p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }
using namespace qwen35_ocr;
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
static void need(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static bool bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
static int run() try {
    const char * flag = "CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE";
    gate(flag, nullptr);
    need(!lighton4b_physical_rope_mode(true), "default off");
    for (const char * value : { "", "0", "false", "FaLsE", "off", "OFF", "no" }) {
        gate(flag, value);
        need(!lighton4b_physical_rope_mode(true), "false aliases");
    }
    gate(flag, "1");
    need(!lighton4b_physical_rope_mode(false), "nonphysical isolation");
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1");
    for (const char * key : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                              "CRISPEMBED_LIGHTON4B_HEAD_BF16" }) gate(key, "0");
    if (!lighton4b_physical_f16_attention_available()) {
        need(!lighton4b_physical_rope_valid(true), "unsupported rejects");
        std::puts("PHYSICAL_ROPE_UNSUPPORTED_GUARD PASS"); return 0;
    }
    attention_config c; c.query_heads = 16; c.kv_heads = 4;
    for (int heads : { 4, 16 }) for (int threads : { 1, 4 }) {
        const int n = 16; // admitted maximum; explicit4+11+1 chunk continuation
        std::vector<float> input(size_t(n) * heads * 256), positions(3 * n), actual;
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * .013f) * 3;
        for (int t = 0; t < n; ++t) {
            positions[t] = float(t + 130);
            positions[n + t] = float(300 + t / 3);
            positions[2 * n + t] = float(42 + t % 3);
        }
        positions[0] = 0; positions[n] = 262143; // admitted coordinate boundaries
        need(lighton4b_physical_rope_cpu(input, c, positions.data(), n, heads, threads, actual), "operator");
        const auto original_input = input;
        need(!lighton4b_physical_rope_cpu(input, c, positions.data(), n, heads, threads, input) &&
             bits(input, original_input), "alias rejects without mutation");
        std::vector<float> overlapping_positions(positions.size() + 1, 123);
        std::copy(positions.begin(), positions.end(), overlapping_positions.begin() + 1);
        const auto original_positions = overlapping_positions;
        need(!lighton4b_physical_rope_cpu(input, c, overlapping_positions.data() + 1,
             n, heads, threads, overlapping_positions) && bits(overlapping_positions, original_positions) &&
             bits(input, original_input), "positions/output overlap rejects without mutation");
        std::vector<float> failure_output{ 123 };
        fail_next_allocation = true;
        need(!lighton4b_physical_rope_cpu(input, c, positions.data(), n, heads, threads, failure_output) &&
             !fail_next_allocation && failure_output.empty() && bits(input, original_input),
             "allocation exception contained and input preserved");
        failure_output = { 123 };
        need(!lighton4b_physical_rope_cpu(input, c, positions.data(), 17, heads, threads, failure_output) &&
             failure_output.empty(), "oversized chunk rejected before access");
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 4ull << 20, nullptr, false }), ggml_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        need(ctx && backend, "independent backend"); ggml_backend_cpu_set_n_threads(backend.get(), threads);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, heads, n);
        auto * p = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4 * n);
        std::memcpy(x->data, input.data(), input.size() * 4);
        std::fill_n(static_cast<int32_t *>(p->data), 4 * n, 0);
        for (int i = 0; i < 3 * n; ++i) static_cast<int32_t *>(p->data)[i] = int32_t(positions[i]);
        int sections[4] = { 11, 11, 10, 0 };
        auto * y = ggml_rope_multi(ctx.get(), x, p, nullptr, 64, sections, GGML_ROPE_TYPE_IMROPE,
                                  262144, 10000000.f, 1.f, 0.f, 1.f, 32.f, 1.f);
        auto * graph = ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph, y);
        need(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "independent node");
        need(!std::memcmp(actual.data(), y->data, actual.size() * 4), "source node exact");
        std::vector<float> joined;
        int start = 0;
        for (int count : { 4, 11, 1 }) {
            std::vector<float> part(input.begin() + size_t(start) * heads * 256,
                                    input.begin() + size_t(start + count) * heads * 256), pos(3 * count), result;
            for (int axis = 0; axis < 3; ++axis)
                std::copy_n(positions.data() + axis * n + start, count, pos.data() + axis * count);
            need(lighton4b_physical_rope_cpu(part, c, pos.data(), count, heads, threads, result), "chunk");
            joined.insert(joined.end(), result.begin(), result.end()); start += count;
        }
        need(bits(actual, joined), "chunk exact explicit positions");
        // Separate invocations own their graph/backend/axes; no mutable shared cache.
        std::vector<float> parallel_a, parallel_b;
        bool accepted_a = false, accepted_b = false;
        std::thread worker_a([&] { accepted_a = lighton4b_physical_rope_cpu(
            input, c, positions.data(), n, heads, threads, parallel_a); });
        std::thread worker_b([&] { accepted_b = lighton4b_physical_rope_cpu(
            input, c, positions.data(), n, heads, threads, parallel_b); });
        worker_a.join(); worker_b.join();
        need(accepted_a && accepted_b && bits(actual, parallel_a) && bits(actual, parallel_b),
             "concurrent independent calls exact");
        for (int t = 0; t < n; ++t) for (int h = 0; h < heads; ++h)
            need(!std::memcmp(input.data() + (t * heads + h) * 256 + 64,
                              actual.data() + (t * heads + h) * 256 + 64, 192 * 4), "unrotated tail exact");
        auto rejected = [&] { std::vector<float> out{ 1 }; need(!lighton4b_physical_rope_cpu(
            input, c, positions.data(), n, heads, threads, out) && out.empty(), "invalid clears"); };
        for (float value : { -.5f, .5f, 262144.f, std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN() }) { positions[0] = value; rejected(); }
        positions[0] = 0;
        c.theta = 10000; rejected(); c.theta = 10000000;
        c.mrope_section[1] = 10; rejected(); c.mrope_section[1] = 11;
        input[0] = std::numeric_limits<float>::quiet_NaN(); rejected(); input[0] = 0;
        for (const char * key : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                                  "CRISPEMBED_LIGHTON4B_HEAD_BF16" }) {
            gate(key, "1"); rejected(); gate(key, "0");
        }
        gate(flag, "off"); rejected(); gate(flag, "1");
    }
    std::puts("PHYSICAL_ROPE source-node/chunk/tail/position/profile controls PASS"); return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
int main() { core_util::clean_exit(run()); }
