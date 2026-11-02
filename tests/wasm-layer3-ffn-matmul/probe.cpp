// Layer3 FFN mul_mat same-input probe (test-only, not a production dependency).
// Built twice from one source: native (x86 AVX2/FMA, llamafile ON = production
// libcrispembed profile) and WASM (simd128, pthread = production embed profile).
//
//   probe selftest <outdir>                 weight-free authored IQ4_NL/IQ4_XS controls
//   probe replay <model.gguf> <outdir> <in1.f32> ...   frozen layer3 fc1/fc2 weights
//
// For every case it writes: the production ggml graph mul_mat output, the
// activation bytes from the production quantizer, an emulation of the native
// accumulation order ("native_order") and of the generic scalar order
// ("generic_order"). Python decides every comparison; this program decides none.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

extern "C" {
// Production activation quantizers of this build (ggml-cpu static library).
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void quantize_row_q8_K(const float * x, void * y, int64_t k);
#ifdef __EMSCRIPTEN__
void quantize_row_q8_0_wasm_nearest(const float * x, void * y, int64_t k);
#endif
}

static std::string host(const std::string & p) {
#ifdef __EMSCRIPTEN__
    return "/host" + p;
#else
    return p;
#endif
}
static void write_file(const std::string & path, const void * data, size_t n) {
    FILE * f = std::fopen(host(path).c_str(), "wb");
    if (!f || std::fwrite(data, 1, n, f) != n || std::fclose(f) != 0) { std::fprintf(stderr, "write failed %s\n", path.c_str()); std::exit(3); }
}
static std::vector<float> read_f32(const std::string & path, size_t n) {
    std::vector<float> v(n);
    FILE * f = std::fopen(host(path).c_str(), "rb");
    if (!f || std::fread(v.data(), 4, n, f) != n || std::fgetc(f) != EOF) { std::fprintf(stderr, "read failed %s\n", path.c_str()); std::exit(3); }
    std::fclose(f);
    return v;
}

// --- activation quantization exactly as this build's mul_mat does it --------
static std::vector<uint8_t> quantize_rows(ggml_type vec_dot_type, const float * x, int k, int rows) {
    const size_t row = ggml_row_size(vec_dot_type, k);
    std::vector<uint8_t> q(row * rows);
    for (int r = 0; r < rows; ++r) {
        if (vec_dot_type == GGML_TYPE_Q8_0) {
#ifdef __EMSCRIPTEN__
            quantize_row_q8_0_wasm_nearest(x + (size_t) r * k, q.data() + r * row, k);  // held CRISPEMBED_WASM_Q8_0_ROUND=1
#else
            quantize_row_q8_0(x + (size_t) r * k, q.data() + r * row, k);
#endif
        } else {
            quantize_row_q8_K(x + (size_t) r * k, q.data() + r * row, k);
        }
    }
    return q;
}

// AVX2 hsum_float_8 and llamafile hsum(__m256): ((x0+x4)+(x2+x6)) + ((x1+x5)+(x3+x7)).
static float hsum8(const float v[8]) {
    const float r0 = v[4] + v[0], r1 = v[5] + v[1], r2 = v[6] + v[2], r3 = v[7] + v[3];
    const float a = r0 + r2, b = r1 + r3;
    return a + b;
}
static int iq4(const uint8_t * qs, int e) { return kvalues_iq4nl[e < 16 ? (qs[e] & 0xf) : (qs[e - 16] >> 4)]; }

// IQ4_NL x Q8_0, native order = llamafile tinyBLAS_Q0_AVX (one 8-lane FMA accumulator,
// lane L = exact int sum of elements 4L..4L+3 of each 32-block, sequential blocks).
static float nl_native(const block_iq4_nl * x, const block_q8_0 * y, int nb) {
    float acc[8] = {0};
    for (int b = 0; b < nb; ++b) {
        const float d = ggml_fp16_to_fp32(x[b].d) * ggml_fp16_to_fp32(y[b].d);
        for (int L = 0; L < 8; ++L) {
            int s = 0;
            for (int u = 0; u < 4; ++u) s += iq4(x[b].qs, 4 * L + u) * y[b].qs[4 * L + u];
            acc[L] = std::fmaf(d, (float) s, acc[L]);
        }
    }
    return hsum8(acc);
}
// IQ4_NL x Q8_0, generic scalar order (ggml_vec_dot_iq4_nl_q8_0_generic).
static float nl_generic(const block_iq4_nl * x, const block_q8_0 * y, int nb) {
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        const float d = ggml_fp16_to_fp32(y[ib].d) * ggml_fp16_to_fp32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL / 2; ++j) {
            sumi1 += y[ib].qs[j + 0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK4_NL / 2] * kvalues_iq4nl[x[ib].qs[j] >> 4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    return sumf;
}
// IQ4_XS x Q8_K, native order = AVX2 ggml_vec_dot_iq4_xs_q8_K (exact int lanes with
// integer scales per superblock, one FMA per superblock per lane, hsum_float_8).
static float xs_native(const block_iq4_xs * x, const block_q8_K * y, int nb) {
    float acc[8] = {0};
    for (int ibl = 0; ibl < nb; ++ibl) {
        int lane[8] = {0};
        uint16_t sh = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const int ls1 = ((x[ibl].scales_l[ib / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib / 2] >> 4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            for (int g = 0; g < 2; ++g) {
                const int ls = g ? ls2 : ls1;
                for (int L = 0; L < 8; ++L) {
                    int s = 0;
                    for (int u = 0; u < 4; ++u) s += iq4(qs, 4 * L + u) * q8[4 * L + u];
                    lane[L] += ls * s;
                }
                qs += 16;
                q8 += 32;
            }
        }
        const float dd = ggml_fp16_to_fp32(x[ibl].d) * y[ibl].d;
        for (int L = 0; L < 8; ++L) acc[L] = std::fmaf(dd, (float) lane[L], acc[L]);
    }
    return hsum8(acc);
}
// IQ4_XS x Q8_K, generic scalar order (ggml_vec_dot_iq4_xs_q8_K_generic).
static float xs_generic(const block_iq4_xs * x, const block_q8_K * y, int nb) {
    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = ggml_fp16_to_fp32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K / 32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib / 2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib / 2] >> 4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8 * (ls1 - 32);
            const float d2 = d4d8 * (ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j + 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j + 16] * kvalues_iq4nl[qs[j] >> 4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    return sumf;
}

static std::vector<float> emulate(ggml_type wt, const uint8_t * w, int k, int out_rows, const std::vector<uint8_t> & q, int tokens, bool native) {
    std::vector<float> y((size_t) out_rows * tokens);
    const size_t wrow = ggml_row_size(wt, k);
    const ggml_type vt = wt == GGML_TYPE_IQ4_NL ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
    const size_t qrow = ggml_row_size(vt, k);
    for (int t = 0; t < tokens; ++t)
        for (int o = 0; o < out_rows; ++o) {
            const void * x = w + o * wrow;
            const void * a = q.data() + t * qrow;
            float v;
            if (wt == GGML_TYPE_IQ4_NL) {
                v = native ? nl_native((const block_iq4_nl *) x, (const block_q8_0 *) a, k / QK4_NL)
                           : nl_generic((const block_iq4_nl *) x, (const block_q8_0 *) a, k / QK4_NL);
            } else {
                v = native ? xs_native((const block_iq4_xs *) x, (const block_q8_K *) a, k / QK_K)
                           : xs_generic((const block_iq4_xs *) x, (const block_q8_K *) a, k / QK_K);
            }
            y[(size_t) t * out_rows + o] = v;  // ggml dst layout [out_rows, tokens]
        }
    return y;
}

// Production path: actual ggml graph mul_mat on the CPU backend, weights in a plain CPU buffer.
static std::vector<float> graph_mul_mat(ggml_backend_t backend, ggml_type wt, const void * w, int k, int out_rows,
                                        const float * x, int tokens, std::string & buffer_name) {
    ggml_init_params wp = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * W = ggml_new_tensor_2d(wctx, wt, k, out_rows);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    ggml_backend_tensor_set(W, w, 0, ggml_nbytes(W));
    buffer_name = ggml_backend_buffer_name(wbuf);
    ggml_init_params gp = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * X = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, k, tokens);
    ggml_tensor * Y = ggml_mul_mat(gctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, Y);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "alloc failed\n"); std::exit(4); }
    ggml_backend_tensor_set(X, x, 0, ggml_nbytes(X));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); std::exit(4); }
    std::vector<float> y((size_t) out_rows * tokens);
    ggml_backend_tensor_get(Y, y.data(), 0, ggml_nbytes(Y));
    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return y;
}

static ggml_backend_t make_backend() {
    ggml_backend_t b = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(b, 1);
#ifdef __EMSCRIPTEN__
    ggml_backend_cpu_set_wasm_q8_0_rounding(b, true);  // production WASM holds CRISPEMBED_WASM_Q8_0_ROUND=1
#endif
    return b;
}

static void run_case(ggml_backend_t backend, const std::string & out, const std::string & label, ggml_type wt,
                     const void * w, int k, int out_rows, const std::vector<float> & x, int tokens, bool & first) {
    std::string buffer;
    const auto prod = graph_mul_mat(backend, wt, w, k, out_rows, x.data(), tokens, buffer);
    const ggml_type vt = wt == GGML_TYPE_IQ4_NL ? GGML_TYPE_Q8_0 : GGML_TYPE_Q8_K;
    const auto q = quantize_rows(vt, x.data(), k, tokens);
    const auto nat = emulate(wt, (const uint8_t *) w, k, out_rows, q, tokens, true);
    const auto gen = emulate(wt, (const uint8_t *) w, k, out_rows, q, tokens, false);
    write_file(out + "/" + label + ".production.f32", prod.data(), prod.size() * 4);
    write_file(out + "/" + label + ".native_order.f32", nat.data(), nat.size() * 4);
    write_file(out + "/" + label + ".generic_order.f32", gen.data(), gen.size() * 4);
    write_file(out + "/" + label + ".activation.bin", q.data(), q.size());
    std::printf("%s{\"label\":\"%s\",\"weight_type\":\"%s\",\"activation_type\":\"%s\",\"k\":%d,\"out_rows\":%d,\"tokens\":%d,\"weight_buffer\":\"%s\"}",
                first ? "" : ",", label.c_str(), ggml_type_name(wt), ggml_type_name(vt), k, out_rows, tokens, buffer.c_str());
    first = false;
}

static uint32_t lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s >> 8; }

static int selftest(const std::string & out) {
    ggml_backend_t backend = make_backend();
    std::printf("{\"mode\":\"selftest\",\"cases\":[");
    bool first = true;
    uint32_t s = 12345;
    for (int pass = 0; pass < 2; ++pass) {
        const ggml_type wt = pass == 0 ? GGML_TYPE_IQ4_NL : GGML_TYPE_IQ4_XS;
        const int k = pass == 0 ? 384 : 1536, out_rows = 64, tokens = 12;
        std::vector<uint8_t> w(ggml_row_size(wt, k) * out_rows);
        for (auto & b : w) b = (uint8_t) lcg(s);
        // Valid small positive/negative fp16 block scales at the documented offsets.
        if (wt == GGML_TYPE_IQ4_NL) {
            for (size_t b = 0; b < w.size() / sizeof(block_iq4_nl); ++b) {
                auto * blk = (block_iq4_nl *) w.data() + b;
                blk->d = ggml_fp32_to_fp16(((int) (lcg(s) % 2001) - 1000) / 50000.0f);
            }
        } else {
            for (size_t b = 0; b < w.size() / sizeof(block_iq4_xs); ++b) {
                auto * blk = (block_iq4_xs *) w.data() + b;
                blk->d = ggml_fp32_to_fp16(((int) (lcg(s) % 2001) - 1000) / 200000.0f);
            }
        }
        std::vector<float> x((size_t) k * tokens);
        for (auto & v : x) v = ((int) (lcg(s) % 200001) - 100000) / 25000.0f;
        run_case(backend, out, pass == 0 ? "authored_iq4_nl" : "authored_iq4_xs", wt, w.data(), k, out_rows, x, tokens, first);
    }
    std::printf("]}\n");
    ggml_backend_free(backend);
    return 0;
}

static int replay(const std::string & model, const std::string & out, int argc, char ** argv) {
    ggml_context * mctx = nullptr;
    gguf_init_params gp = { false, &mctx };
    gguf_context * g = gguf_init_from_file(host(model).c_str(), gp);
    if (!g) { std::fprintf(stderr, "gguf load failed\n"); return 3; }
    auto find = [&](const char * a, const char * b) {
        ggml_tensor * t = ggml_get_tensor(mctx, a);
        return t ? t : ggml_get_tensor(mctx, b);
    };
    ggml_tensor * fc1 = find("blk.3.ffn_up.weight", "encoder.layer.3.ffn.fc1.weight");
    ggml_tensor * fc2 = find("blk.3.ffn_down.weight", "encoder.layer.3.ffn.fc2.weight");
    if (!fc1 || !fc2) { std::fprintf(stderr, "layer3 fc1/fc2 absent\n"); return 3; }
    ggml_backend_t backend = make_backend();
    std::printf("{\"mode\":\"replay\",\"fc1\":{\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld]},\"fc2\":{\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld]},\"cases\":[",
                ggml_get_name(fc1), ggml_type_name(fc1->type), (long long) fc1->ne[0], (long long) fc1->ne[1],
                ggml_get_name(fc2), ggml_type_name(fc2->type), (long long) fc2->ne[0], (long long) fc2->ne[1]);
    bool first = true;
    for (int i = 0; i < argc; i += 2) {  // pairs: <fc1|fc2>:<label> <input.f32>
        const std::string spec = argv[i];
        const bool is1 = spec.rfind("fc1:", 0) == 0;
        ggml_tensor * W = is1 ? fc1 : fc2;
        const int k = (int) W->ne[0], out_rows = (int) W->ne[1], tokens = 12;
        const auto x = read_f32(argv[i + 1], (size_t) k * tokens);
        if (W->type != GGML_TYPE_IQ4_NL && W->type != GGML_TYPE_IQ4_XS) { std::fprintf(stderr, "unexpected weight type %s\n", ggml_type_name(W->type)); return 5; }
        run_case(backend, out, spec.substr(4), W->type, W->data, k, out_rows, x, tokens, first);
    }
    std::printf("]}\n");
    ggml_backend_free(backend);
    gguf_free(g);
    ggml_free(mctx);
    return 0;
}

int main(int argc, char ** argv) {
#ifdef __EMSCRIPTEN__
    EM_ASM({ FS.mkdir('/host'); FS.mount(NODEFS, { root: '/' }, '/host'); });
#endif
    if (argc >= 3 && std::string(argv[1]) == "selftest") return selftest(argv[2]);
    if (argc >= 4 && std::string(argv[1]) == "replay" && (argc - 4) % 2 == 0) return replay(argv[2], argv[3], argc - 4, argv + 4);
    std::fprintf(stderr, "usage: probe selftest <out> | probe replay <model> <out> (<fc1|fc2>:<label> <in.f32>)...\n");
    return 2;
}
