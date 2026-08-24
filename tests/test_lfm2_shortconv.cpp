// test_lfm2_shortconv.cpp — hermetic guard for the LFM2 ShortConv decode step.
//
// The single-token decode path must reproduce, bit-for-bit in intent, what the
// prefill path computes with ggml_conv_1d_dw over a left-padded sequence. Both
// produce a [D] vector from a [D, K] window and a [K, D] kernel, so the shapes
// permit any reduction over the wrong axis — this test pins the axis.
//
// Two independent checks per shape:
//   1. vs a scalar reference: out[d] = sum_k window(d, k) * kern(k, d)
//   2. vs the PREFILL path itself: ggml_conv_1d_dw over a left-padded sequence,
//      last output column. This is the invariant that actually matters — it is
//      what makes KV-cached decode agree with full recompute.
//
// No weights, no model. Runs in milliseconds.

#include "core/clean_exit.h"
#include "lfm2_shortconv.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char * what) {
    printf("  %s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_failures++;
}

// Deterministic pseudo-random fill; no dependence on the platform RNG.
float rnd(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return (float)((int32_t)(s >> 8) % 2000 - 1000) / 1000.0f;
}

// ── The scalar reference for a single decode step ────────────────────────────
// window is [D, K] column-major: element (d, k) at flat k * D + d.
// kern   is [K, D] column-major: element (k, d) at flat d * K + k.
std::vector<float> scalar_step(const std::vector<float> & window, const std::vector<float> & kern, int D, int K) {
    std::vector<float> out((size_t)D, 0.0f);
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += window[(size_t)k * D + d] * kern[(size_t)d * K + k];
        }
        out[(size_t)d] = acc;
    }
    return out;
}

// ── The prefill path: causal depthwise conv1d over a left-padded sequence ────
// Returns the LAST output column, which is what the decode step must match.
std::vector<float> prefill_last_column(ggml_backend_t backend, const std::vector<float> & seq,
                                       const std::vector<float> & kern, int D, int K, int T) {
    const int pad = K - 1;

    size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(g);

    // Bx over the whole sequence, left-padded by K-1 zero columns: [D, pad + T]
    ggml_tensor * bx_padded = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, pad + T);
    ggml_set_name(bx_padded, "bx_padded");
    ggml_set_input(bx_padded);

    ggml_tensor * kw = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, D);
    ggml_set_name(kw, "kern");
    ggml_set_input(kw);

    // Exactly what build_prefill_graph does.
    ggml_tensor * bx_t = ggml_cont(g, ggml_transpose(g, bx_padded)); // [pad+T, D]
    ggml_tensor * conv_w = ggml_cast(g, kw, GGML_TYPE_F16);
    conv_w = ggml_reshape_3d(g, conv_w, conv_w->ne[0], 1, D);
    ggml_tensor * co = ggml_conv_1d_dw(g, conv_w, bx_t, 1, 0, 1); // [T, D]
    if (co->ne[0] > T) co = ggml_view_2d(g, co, T, D, co->nb[1], 0);
    co = ggml_cont(g, ggml_transpose(g, co)); // [D, T]

    ggml_set_name(co, "out");
    ggml_set_output(co);
    ggml_build_forward_expand(gf, co);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    std::vector<float> padded((size_t)D * (pad + T), 0.0f);
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            padded[(size_t)(t + pad) * D + d] = seq[(size_t)t * D + d];
        }
    }
    ggml_backend_tensor_set(bx_padded, padded.data(), 0, padded.size() * sizeof(float));
    ggml_backend_tensor_set(kw, kern.data(), 0, kern.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> all((size_t)D * T);
    ggml_backend_tensor_get(co, all.data(), 0, all.size() * sizeof(float));

    std::vector<float> last((size_t)D);
    for (int d = 0; d < D; d++) last[(size_t)d] = all[(size_t)(T - 1) * D + d];

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return last;
}

// ── The decode path under test ───────────────────────────────────────────────
std::vector<float> decode_step(ggml_backend_t backend, const std::vector<float> & window,
                               const std::vector<float> & kern, int D, int K) {
    size_t mem = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params ip = { mem, nullptr, true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(g);

    ggml_tensor * win = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, K);
    ggml_set_name(win, "window");
    ggml_set_input(win);

    ggml_tensor * kw = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, D);
    ggml_set_name(kw, "kern");
    ggml_set_input(kw);

    ggml_tensor * out = lfm2_shortconv::step(g, win, kw, D, K);
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(win, window.data(), 0, window.size() * sizeof(float));
    ggml_backend_tensor_set(kw, kern.data(), 0, kern.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res((size_t)D);
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(g);
    return res;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

void run_shape(ggml_backend_t backend, int D, int K, uint32_t seed) {
    printf("D=%d K=%d\n", D, K);

    // The decode window is the tail of a longer sequence, so build the sequence
    // first and slice the last K columns out of it. T > K so the prefill path
    // has real history rather than padding at the position we compare.
    const int T = K + 5;
    uint32_t s = seed;
    std::vector<float> seq((size_t)D * T);
    for (auto & v : seq) v = rnd(s);
    std::vector<float> kern((size_t)K * D);
    for (auto & v : kern) v = rnd(s);

    std::vector<float> window((size_t)D * K);
    for (int k = 0; k < K; k++) {
        int t = T - K + k; // oldest first, current token last
        for (int d = 0; d < D; d++) {
            window[(size_t)k * D + d] = seq[(size_t)t * D + d];
        }
    }

    std::vector<float> got = decode_step(backend, window, kern, D, K);
    std::vector<float> ref = scalar_step(window, kern, D, K);
    std::vector<float> pre = prefill_last_column(backend, seq, kern, D, K, T);

    double d_scalar = max_abs_diff(got, ref);
    double d_prefill = max_abs_diff(got, pre);

    // The prefill path rounds the kernel to F16, so allow a little slack there;
    // the scalar comparison is pure F32 and must be tight.
    char msg[160];
    snprintf(msg, sizeof(msg), "decode step vs scalar reference (max_abs=%.3e)", d_scalar);
    check(d_scalar < 1e-5, msg);
    snprintf(msg, sizeof(msg), "decode step vs prefill conv_1d_dw last column (max_abs=%.3e)", d_prefill);
    check(d_prefill < 2e-2, msg);
}

int test_main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 2);

    // A tiny shape where every channel is inspectable, and the real LFM2.5-VL
    // shape (hidden_size=2048, conv_kernel=3).
    run_shape(backend, 4, 3, 12345u);
    run_shape(backend, 2048, 3, 987u);

    ggml_backend_free(backend);

    printf("%s\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    core_util::clean_exit(test_main());
}
