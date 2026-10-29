// Test-only controls and live graph replay. No model loader or saved norm input.
#include "core/clean_exit.h"
#include "core/wasm_native_order_norm.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORTED EMSCRIPTEN_KEEPALIVE
#else
#define EXPORTED
#endif
static void require(bool v, const char * s) {
    if (!v) throw std::runtime_error(s);
}
struct owner {
    ggml_context * c = nullptr;
    ggml_backend_t b = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~owner() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (c) ggml_free(c);
        if (b) ggml_backend_free(b);
    }
};
extern "C" EXPORTED int norm_candidate_replay(const float * x, float * y, int count, int threads) {
    if (!x || !y || count != 384 * 12 || threads < 1 || threads > 2) return -1;
    for (int i = 0; i < count; ++i)
        if (!std::isfinite(x[i])) return -2;
    owner o;
    o.c = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    o.b = ggml_backend_cpu_init();
    if (!o.c || !o.b) return -3;
    ggml_backend_cpu_set_n_threads(o.b, threads);
    auto * input = ggml_new_tensor_2d(o.c, GGML_TYPE_F32, 384, 12);
    const wasm_native_order_norm::parameters config;
    auto * result = wasm_native_order_norm::make(o.c, input, &config);
    if (!result || result->src[0] != input) return -4;
    auto * graph = ggml_new_graph(o.c);
    ggml_build_forward_expand(graph, result);
    o.buffer = ggml_backend_alloc_ctx_tensors(o.c, o.b);
    if (!o.buffer) return -5;
    ggml_backend_tensor_set(input, x, 0, size_t(count) * 4);
    if (ggml_backend_graph_compute(o.b, graph) != GGML_STATUS_SUCCESS) return -6;
    std::vector<float> pending(count);
    ggml_backend_tensor_get(result, pending.data(), 0, size_t(count) * 4);
    for (float v : pending)
        if (!std::isfinite(v)) return -7;
    std::copy(pending.begin(), pending.end(), y);
    return count;
}
extern "C" EXPORTED int norm_candidate_controls() {
    using namespace wasm_native_order_norm;
    float x[384], y[384];
    for (int i = 0; i < 384; ++i) {
        x[i] = float(i % 37 - 18) / 7.f;
        y[i] = 123.f;
    }
    if (!eligible(true, 384, 6, 12, 1536, false, 1e-12f) || eligible(false, 384, 6, 12, 1536, false, 1e-12f) ||
        eligible(true, 383, 6, 12, 1536, false, 1e-12f) || eligible(true, 384, 5, 12, 1536, false, 1e-12f) ||
        eligible(true, 384, 6, 8, 1536, false, 1e-12f) || eligible(true, 384, 6, 12, 1024, false, 1e-12f) ||
        eligible(true, 384, 6, 12, 1536, true, 1e-12f) || eligible(true, 384, 6, 12, 1536, false, 1e-5f))
        return -1;
    if (compute_row(x, y, 383, 1e-12f) || y[0] != 123.f || compute_row(x, y, 384, 1e-5f) || y[0] != 123.f) return -2;
    x[17] = std::numeric_limits<float>::quiet_NaN();
    if (compute_row(x, y, 384, 1e-12f) || y[0] != 123.f) return -3;
    x[17] = std::numeric_limits<float>::infinity();
    if (compute_row(x, y, 384, 1e-12f) || y[0] != 123.f) return -4;
    x[17] = 0.f;
    if (!compute_row(x, y, 384, 1e-12f)) return -5;
    for (float v : y)
        if (!std::isfinite(v)) return -6;
    std::fill(std::begin(x), std::end(x), 0.f);
    if (!compute_row(x, y, 384, 1e-12f)) return -7;
    for (float v : y)
        if (v != 0.f) return -8;
    // Geometry rejection occurs before backend allocation and keeps outputs intact.
    owner o;
    o.c = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    if (!o.c) return -9;
    const parameters config;
    auto * bad = ggml_new_tensor_2d(o.c, GGML_TYPE_F32, 383, 2);
    if (make(o.c, bad, &config)) return -10;
    bad = ggml_new_tensor_2d(o.c, GGML_TYPE_F16, 384, 2);
    if (make(o.c, bad, &config)) return -11;
    return 0;
}
static int body(int argc, char ** argv) {
    try {
        require(norm_candidate_controls() == 0, "Authored geometry/finite/recovery controls failed");
        if (argc == 1) {
            std::puts("PASS geometry/nonfinite/zero/recovery");
            return 0;
        }
        require(argc == 4, "Expected input.f32 output.f32 threads");
        const int threads = std::stoi(argv[3]);
        require(threads == 1 || threads == 2, "Threads1/2 required");
        std::vector<float> input(384 * 12), output(input.size()), repeat(input.size());
        std::ifstream f(argv[1], std::ios::binary);
        f.read(reinterpret_cast<char *>(input.data()), input.size() * 4);
        require(bool(f) && f.peek() == std::char_traits<char>::eof(), "Input axes/extent differ");
        auto before = input;
        require(norm_candidate_replay(input.data(), output.data(), input.size(), threads) == int(input.size()),
                "Connected live norm failed");
        require(norm_candidate_replay(input.data(), repeat.data(), input.size(), threads) == int(input.size()),
                "Repeat failed");
        require(std::memcmp(output.data(), repeat.data(), output.size() * 4) == 0 &&
                    std::memcmp(input.data(), before.data(), input.size() * 4) == 0,
                "Repeat/input neutrality failed");
        std::ofstream out(argv[2], std::ios::binary);
        out.write(reinterpret_cast<char *>(output.data()), output.size() * 4);
        require(bool(out), "Output write failed");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Norm controls: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
