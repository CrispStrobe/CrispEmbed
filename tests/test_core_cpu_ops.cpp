// tests/test_core_cpu_ops.cpp — Unit tests for src/core/cpu_ops.h
//
// Pure CPU tests with known-answer inputs. No GGUF model files needed.
// Tests every function in core_cpu namespace.
//
// Usage: ./build/test-core-cpu-ops
// Exit 0 = all pass, non-zero = failure.

#include "core/cpu_ops.h"
#include "core/clean_exit.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace core_cpu;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);                                          \
            g_fail++;                                                                                                  \
        } else {                                                                                                       \
            g_pass++;                                                                                                  \
        }                                                                                                              \
    } while (0)

#define CHECK_CLOSE(a, b, tol, msg) CHECK(fabsf((a) - (b)) < (tol), msg)

// ---------------------------------------------------------------------------
// to_f32 — test with a real ggml F32 tensor via a CPU backend buffer
// ---------------------------------------------------------------------------
// Helper: create a backend-allocated tensor, set its data, return it.
// Caller owns the buffer (returned via buf_out) and must free it.
static ggml_tensor * make_tensor(ggml_context * ctx, ggml_backend_t backend, ggml_type type, int n, const void * data,
                                 size_t data_bytes, ggml_backend_buffer_t * buf_out) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, type, n);
    *buf_out = ggml_backend_alloc_buffer(backend, ggml_nbytes(t) + 64);
    struct ggml_tallocr alloc = ggml_tallocr_new(*buf_out);
    ggml_tallocr_alloc(&alloc, t);
    ggml_backend_tensor_set(t, data, 0, data_bytes);
    return t;
}

static void test_to_f32() {
    printf("test_to_f32...\n");

    // no_alloc=true so tensors don't get context-buffer data pointers
    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);

    // --- F32 tensor ---
    {
        float data[] = { 1.0f, -2.5f, 3.14f, 0.0f };
        ggml_backend_buffer_t buf;
        ggml_tensor * t = make_tensor(ctx, backend, GGML_TYPE_F32, 4, data, sizeof(data), &buf);

        auto out = to_f32(t);
        CHECK(out.size() == 4, "to_f32 F32 size");
        CHECK_CLOSE(out[0], 1.0f, 1e-6f, "to_f32 F32 [0]");
        CHECK_CLOSE(out[1], -2.5f, 1e-6f, "to_f32 F32 [1]");
        CHECK_CLOSE(out[2], 3.14f, 1e-4f, "to_f32 F32 [2]");
        CHECK_CLOSE(out[3], 0.0f, 1e-6f, "to_f32 F32 [3]");

        ggml_backend_buffer_free(buf);
    }

    // --- F16 tensor ---
    {
        ggml_fp16_t fp16_data[3];
        fp16_data[0] = ggml_fp32_to_fp16(1.0f);
        fp16_data[1] = ggml_fp32_to_fp16(-0.5f);
        fp16_data[2] = ggml_fp32_to_fp16(2.25f);

        ggml_backend_buffer_t buf;
        ggml_tensor * t = make_tensor(ctx, backend, GGML_TYPE_F16, 3, fp16_data, sizeof(fp16_data), &buf);

        auto out = to_f32(t);
        CHECK(out.size() == 3, "to_f32 F16 size");
        CHECK_CLOSE(out[0], 1.0f, 1e-3f, "to_f32 F16 [0]");
        CHECK_CLOSE(out[1], -0.5f, 1e-3f, "to_f32 F16 [1]");
        CHECK_CLOSE(out[2], 2.25f, 1e-3f, "to_f32 F16 [2]");

        ggml_backend_buffer_free(buf);
    }

    // --- nullptr ---
    {
        auto out = to_f32(nullptr);
        CHECK(out.empty(), "to_f32 nullptr returns empty");
    }

    ggml_backend_free(backend);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// layernorm_cpu — raw float pointer version
// ---------------------------------------------------------------------------
static void test_layernorm_cpu() {
    printf("test_layernorm_cpu...\n");

    // Input: [1, 2, 3, 4], mean=2.5, var=1.25
    // With w=[1,1,1,1], b=[0,0,0,0], eps=0:
    //   inv_std = 1/sqrt(1.25) ≈ 0.894427
    //   out[i] = (x[i]-2.5) * inv_std
    float in[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float w[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float b[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float out[4];

    layernorm_cpu(in, out, 4, w, b, 0.0f);

    float inv_std = 1.0f / sqrtf(1.25f);
    CHECK_CLOSE(out[0], -1.5f * inv_std, 1e-5f, "layernorm [0]");
    CHECK_CLOSE(out[1], -0.5f * inv_std, 1e-5f, "layernorm [1]");
    CHECK_CLOSE(out[2], 0.5f * inv_std, 1e-5f, "layernorm [2]");
    CHECK_CLOSE(out[3], 1.5f * inv_std, 1e-5f, "layernorm [3]");

    // With scale and bias
    float w2[] = { 2.0f, 2.0f, 2.0f, 2.0f };
    float b2[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    layernorm_cpu(in, out, 4, w2, b2, 1e-5f);
    CHECK_CLOSE(out[2], 0.5f / sqrtf(1.25f + 1e-5f) * 2.0f + 1.0f, 1e-4f, "layernorm w/bias [2]");

    // With nullptr w and b (identity scale, zero bias)
    layernorm_cpu(in, out, 4, (const float *)nullptr, (const float *)nullptr, 0.0f);
    CHECK_CLOSE(out[0], -1.5f * inv_std, 1e-5f, "layernorm nullptr w/b [0]");

    // In-place (in == out)
    float inplace[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    layernorm_cpu(inplace, inplace, 4, w, b, 0.0f);
    CHECK_CLOSE(inplace[0], -1.5f * inv_std, 1e-5f, "layernorm in-place [0]");
}

// ---------------------------------------------------------------------------
// layernorm2d_cpu
// ---------------------------------------------------------------------------
static void test_layernorm2d_cpu() {
    printf("test_layernorm2d_cpu...\n");

    // C=2, H=1, W=2 — normalize over C at each spatial position
    // Position (0,0): values [1, 3] over channels → mean=2, var=1
    // Position (0,1): values [2, 4] over channels → mean=3, var=1
    float in[] = {
        1.0f, 2.0f, // channel 0: [1, 2]
        3.0f, 4.0f  // channel 1: [3, 4]
    };
    float w[] = { 1.0f, 1.0f };
    float b[] = { 0.0f, 0.0f };
    float out[4];

    layernorm2d_cpu(in, out, 2, 1, 2, w, b, 0.0f);

    // Position (0,0): (1-2)/1 = -1, (3-2)/1 = 1
    CHECK_CLOSE(out[0], -1.0f, 1e-5f, "layernorm2d c0 (0,0)");
    CHECK_CLOSE(out[2], 1.0f, 1e-5f, "layernorm2d c1 (0,0)");
    // Position (0,1): (2-3)/1 = -1, (4-3)/1 = 1
    CHECK_CLOSE(out[1], -1.0f, 1e-5f, "layernorm2d c0 (0,1)");
    CHECK_CLOSE(out[3], 1.0f, 1e-5f, "layernorm2d c1 (0,1)");
}

// ---------------------------------------------------------------------------
// rmsnorm_cpu
// ---------------------------------------------------------------------------
static void test_rmsnorm_cpu() {
    printf("test_rmsnorm_cpu...\n");

    // Input: [3, 4], rms = sqrt((9+16)/2) = sqrt(12.5)
    // inv_rms = 1/sqrt(12.5)
    // With w=[1,1], out[i] = in[i] / sqrt(12.5)
    float in[] = { 3.0f, 4.0f };
    float w[] = { 1.0f, 1.0f };
    float out[2];

    rmsnorm_cpu(in, out, 2, w, 0.0f);

    float inv_rms = 1.0f / sqrtf(12.5f);
    CHECK_CLOSE(out[0], 3.0f * inv_rms, 1e-5f, "rmsnorm [0]");
    CHECK_CLOSE(out[1], 4.0f * inv_rms, 1e-5f, "rmsnorm [1]");

    // With scale weights
    float w2[] = { 2.0f, 0.5f };
    rmsnorm_cpu(in, out, 2, w2, 1e-6f);
    float inv_rms2 = 1.0f / sqrtf(12.5f + 1e-6f);
    CHECK_CLOSE(out[0], 3.0f * inv_rms2 * 2.0f, 1e-5f, "rmsnorm w/ scale [0]");
    CHECK_CLOSE(out[1], 4.0f * inv_rms2 * 0.5f, 1e-5f, "rmsnorm w/ scale [1]");
}

// ---------------------------------------------------------------------------
// linear_cpu — raw float pointer version
// ---------------------------------------------------------------------------
static void test_linear_cpu() {
    printf("test_linear_cpu...\n");

    // in=[1, 2], w=[[1, 3], [2, 4]] (row-major: w[o*in+i])
    // out[0] = 1*1 + 2*3 = 7
    // out[1] = 1*2 + 2*4 = 10
    float in[] = { 1.0f, 2.0f };
    float w[] = { 1.0f, 3.0f, 2.0f, 4.0f }; // [out_dim=2, in_dim=2]
    float b[] = { 0.5f, -0.5f };
    float out[2];

    linear_cpu(in, out, 2, 2, w, b);
    CHECK_CLOSE(out[0], 7.5f, 1e-5f, "linear [0] with bias");
    CHECK_CLOSE(out[1], 9.5f, 1e-5f, "linear [1] with bias");

    // Without bias
    linear_cpu(in, out, 2, 2, w, nullptr);
    CHECK_CLOSE(out[0], 7.0f, 1e-5f, "linear [0] no bias");
    CHECK_CLOSE(out[1], 10.0f, 1e-5f, "linear [1] no bias");

    // Rectangular: in_dim=3, out_dim=2
    float in3[] = { 1.0f, 2.0f, 3.0f };
    float w32[] = { 1.0f, 0.0f, -1.0f,  // row 0: 1*1 + 0*2 + (-1)*3 = -2
                    0.0f, 1.0f, 1.0f }; // row 1: 0*1 + 1*2 + 1*3 = 5
    linear_cpu(in3, out, 3, 2, w32, nullptr);
    CHECK_CLOSE(out[0], -2.0f, 1e-5f, "linear rect [0]");
    CHECK_CLOSE(out[1], 5.0f, 1e-5f, "linear rect [1]");
}

// ---------------------------------------------------------------------------
// conv2d_cpu — standard and grouped convolution
// ---------------------------------------------------------------------------
static void test_conv2d_cpu() {
    printf("test_conv2d_cpu...\n");

    // 1x1 conv, 1 channel in, 1 channel out, 3x3 input, no padding, stride=1
    // This is effectively linear per-pixel with a scalar weight
    {
        float in[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // [1, 3, 3]
        float w[1] = { 2.0f };                       // [1, 1, 1, 1]
        float b[1] = { 1.0f };
        float out[9];

        conv2d_cpu(in, out, w, b, 1, 1, 3, 3, 1, 1, 1, 0);
        CHECK_CLOSE(out[0], 3.0f, 1e-5f, "conv2d 1x1 [0]");  // 2*1 + 1
        CHECK_CLOSE(out[4], 11.0f, 1e-5f, "conv2d 1x1 [4]"); // 2*5 + 1
        CHECK_CLOSE(out[8], 19.0f, 1e-5f, "conv2d 1x1 [8]"); // 2*9 + 1
    }

    // 3x3 conv, 1 channel, 1 filter, 3x3 input, no padding, stride=1
    // Output: 1x1
    {
        float in[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        float w[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 }; // diagonal filter
        float b[1] = { 0.0f };
        float out[1];

        conv2d_cpu(in, out, w, b, 1, 1, 3, 3, 3, 3, 1, 0);
        // sum of diagonal: 1 + 5 + 9 = 15
        CHECK_CLOSE(out[0], 15.0f, 1e-5f, "conv2d 3x3 diagonal");
    }

    // 3x3 conv with padding=1, stride=1 — output same size as input
    {
        float in[4] = { 1, 2, 3, 4 }; // [1, 2, 2]
        // All-ones 3x3 kernel
        float w[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        float b[1] = { 0.0f };
        float out[4];

        conv2d_cpu(in, out, w, b, 1, 1, 2, 2, 3, 3, 1, 1);
        // Output (0,0): in-range pixels are (0,0),(0,1),(1,0),(1,1) = 1+2+3+4 = 10
        CHECK_CLOSE(out[0], 10.0f, 1e-5f, "conv2d padded [0]");
    }

    // Stride=2 test
    {
        float in[16]; // [1, 4, 4]
        for (int i = 0; i < 16; i++) in[i] = (float)(i + 1);
        float w[1] = { 1.0f }; // 1x1 conv
        float out[4];          // [1, 2, 2]

        conv2d_cpu(in, out, w, nullptr, 1, 1, 4, 4, 1, 1, 2, 0);
        CHECK_CLOSE(out[0], 1.0f, 1e-5f, "conv2d stride2 [0]");
        CHECK_CLOSE(out[1], 3.0f, 1e-5f, "conv2d stride2 [1]");
        CHECK_CLOSE(out[2], 9.0f, 1e-5f, "conv2d stride2 [2]");
        CHECK_CLOSE(out[3], 11.0f, 1e-5f, "conv2d stride2 [3]");
    }

    // Depthwise (groups=channels) test
    {
        // 2 channels, 2 groups (depthwise), 2x2 input, 1x1 kernel
        float in[8] = { 1, 2, 3, 4,   // ch0: [[1,2],[3,4]]
                        5, 6, 7, 8 }; // ch1: [[5,6],[7,8]]
        float w[2] = { 2.0f, 3.0f };  // [2, 1, 1, 1] — one scalar per channel
        float b[2] = { 0.0f, 0.0f };
        float out[8];

        conv2d_cpu(in, out, w, b, 2, 2, 2, 2, 1, 1, 1, 0, 2);
        // ch0 scaled by 2, ch1 scaled by 3
        CHECK_CLOSE(out[0], 2.0f, 1e-5f, "conv2d depthwise ch0[0]");
        CHECK_CLOSE(out[3], 8.0f, 1e-5f, "conv2d depthwise ch0[3]");
        CHECK_CLOSE(out[4], 15.0f, 1e-5f, "conv2d depthwise ch1[0]");
        CHECK_CLOSE(out[7], 24.0f, 1e-5f, "conv2d depthwise ch1[3]");
    }

    // No bias (nullptr)
    {
        float in[4] = { 1, 2, 3, 4 };
        float w[1] = { 1.0f };
        float out[4];
        conv2d_cpu(in, out, w, nullptr, 1, 1, 2, 2, 1, 1, 1, 0);
        CHECK_CLOSE(out[0], 1.0f, 1e-5f, "conv2d no bias [0]");
    }
}

// ---------------------------------------------------------------------------
// conv2d_1x1_cpu — must agree with the generic conv2d_cpu path exactly
// ---------------------------------------------------------------------------
// The 1x1 fast path is selected inside conv2d_cpu by a read-once static env
// check (CRISPEMBED_CONV1X1_FAST), so it cannot be A/B'd through that entry
// point twice in one process. Calling conv2d_1x1_cpu directly compares the two
// implementations here instead of relying on a whole-engine decoded-output
// check to notice a shape it happens not to exercise.
//
// The shapes below are chosen to hit the parts that a "looks right" reading
// misses: an output-channel count that is not a multiple of the 4-wide unroll,
// a plane that straddles the 8192-element tile boundary so a short tail runs,
// grouped/depthwise layouts where the input and output channel offsets differ,
// and a null bias.
//
// This is a TOLERANCE check, not an equality one, and the reason matters: the
// generic path sums each output element through dot_product, which on aarch64
// runs eight FMA lanes and a horizontal add, and on AVX2 sixteen. The axpy form
// accumulates over input channels in order. Same arithmetic, different
// association, so the last ulp legitimately differs -- an exact assertion here
// would be a test that fails for a correct implementation. The tolerance is
// scaled by the magnitude actually produced, because an absolute epsilon on an
// output that happens to be large is a tolerance wider than the defect.
static void test_conv2d_1x1_equivalence() {
    printf("test_conv2d_1x1_equivalence...\n");

    struct shape {
        int in_ch, out_ch, H, W, groups;
        bool bias;
        const char * name;
    };
    const shape shapes[] = {
        { 8, 8, 4, 4, 1, true, "small square" },
        { 8, 7, 5, 5, 1, true, "out_ch not multiple of 4" },
        { 3, 6, 1, 1, 1, true, "single pixel" },
        { 6, 6, 64, 130, 1, true, "plane 8320 straddles tile boundary" },
        { 6, 6, 64, 128, 1, true, "plane exactly one tile" },
        { 12, 6, 9, 9, 3, true, "grouped, 3 groups" },
        { 5, 5, 7, 7, 5, true, "depthwise" },
        { 9, 4, 6, 6, 1, false, "null bias" },
        { 16, 33, 3, 3, 1, true, "out_ch 33, tail of 1" },
    };

    // Deterministic pseudo-random values; a constant-filled tensor would hide
    // an index mix-up between the two paths.
    uint32_t seed = 0x5eed1234u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    };

    for (const shape & s : shapes) {
        const size_t plane = (size_t)s.H * s.W;
        const int cin = s.in_ch / s.groups;
        std::vector<float> in((size_t)s.in_ch * plane);
        std::vector<float> w((size_t)s.out_ch * cin);
        std::vector<float> b(s.out_ch);
        for (float & v : in) v = next();
        for (float & v : w) v = next();
        for (float & v : b) v = next();

        std::vector<float> ref((size_t)s.out_ch * plane, 0.0f);
        std::vector<float> fast((size_t)s.out_ch * plane, 0.0f);
        const float * bias = s.bias ? b.data() : nullptr;

        conv2d_cpu(in.data(), ref.data(), w.data(), bias, s.in_ch, s.out_ch, s.H, s.W, 1, 1, 1, 0, s.groups);
        conv2d_1x1_cpu(in.data(), fast.data(), w.data(), bias, s.in_ch, s.out_ch, s.H, s.W, s.groups);

        float scale = 1e-6f;
        for (float v : ref) scale = fmaxf(scale, fabsf(v));
        const float tol = 1e-5f * scale;

        size_t mismatches = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = fabsf(ref[i] - fast[i]);
            if (d > worst) worst = d;
            if (d > tol) mismatches++;
        }
        if (mismatches)
            fprintf(stderr, "  %s: %zu/%zu over tol=%g, max_abs_diff=%g (scale=%g)\n", s.name, mismatches, ref.size(),
                    tol, worst, scale);
        CHECK(mismatches == 0, s.name);
    }
}

// ---------------------------------------------------------------------------
// conv2d_depthwise_cpu — must agree with the generic conv2d_cpu path
// ---------------------------------------------------------------------------
// Same rationale and same tolerance argument as the 1x1 guard above.
//
// The depthwise kernel replaces the generic path's per-pixel boundary test with
// a per-tap column range computed in closed form, so the shapes that matter are
// the ones where that range is not simply [0, out_W): odd padding, stride > 1
// with padding (where the first valid column is not 0), and a kernel wider than
// the padded input, which drives the range empty. That last case is why the
// bounds use explicit floor/ceil -- C division truncates toward zero, so a
// numerator of -1 over stride 2 would round UP to 0 and admit a column that
// reads past the end of the input row.
static void test_conv2d_depthwise_equivalence() {
    printf("test_conv2d_depthwise_equivalence...\n");

    struct shape {
        int channels, H, W, kh, kw, stride, pad;
        bool bias;
        const char * name;
    };
    const shape shapes[] = {
        { 4, 8, 8, 3, 3, 1, 1, true, "3x3 pad 1" },
        { 4, 8, 8, 3, 3, 1, 0, true, "3x3 no pad" },
        { 3, 9, 7, 5, 5, 1, 2, true, "5x5 pad 2" },
        { 96, 24, 18, 7, 7, 1, 3, true, "7x7 pad 3, detector shape" },
        { 4, 9, 9, 3, 3, 2, 1, true, "stride 2 with pad" },
        { 4, 8, 8, 3, 3, 2, 0, true, "stride 2 no pad" },
        { 2, 4, 4, 5, 5, 2, 2, true, "kernel wider than input, stride 2" },
        { 2, 3, 3, 5, 5, 1, 0, false, "kernel wider than input, no pad, null bias" },
        // Drives hi_num negative while the output stays non-empty, which needs
        // W + pad < kw <= W + 2*pad. With stride 2 this is the exact case where
        // C's truncating division turns floor(-1/2) = -1 into 0 and admits a
        // column one past the end of the input row.
        { 2, 4, 2, 3, 5, 2, 2, true, "tap range empty, stride 2 (floor vs trunc)" },
        { 5, 6, 6, 1, 3, 1, 0, true, "asymmetric 1x3" },
        { 5, 6, 6, 3, 1, 1, 1, true, "asymmetric 3x1 with pad" },
    };

    uint32_t seed = 0xc0ffee11u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    };

    for (const shape & s : shapes) {
        const int out_H = (s.H + 2 * s.pad - s.kh) / s.stride + 1;
        const int out_W = (s.W + 2 * s.pad - s.kw) / s.stride + 1;
        if (out_H <= 0 || out_W <= 0) continue;

        std::vector<float> in((size_t)s.channels * s.H * s.W);
        std::vector<float> w((size_t)s.channels * s.kh * s.kw);
        std::vector<float> b(s.channels);
        for (float & v : in) v = next();
        for (float & v : w) v = next();
        for (float & v : b) v = next();

        std::vector<float> ref((size_t)s.channels * out_H * out_W, 0.0f);
        std::vector<float> fast((size_t)s.channels * out_H * out_W, 0.0f);
        const float * bias = s.bias ? b.data() : nullptr;

        conv2d_cpu(in.data(), ref.data(), w.data(), bias, s.channels, s.channels, s.H, s.W, s.kh, s.kw, s.stride, s.pad,
                   s.channels);
        conv2d_depthwise_cpu(in.data(), fast.data(), w.data(), bias, s.channels, s.H, s.W, s.kh, s.kw, s.stride, s.pad);

        float scale = 1e-6f;
        for (float v : ref) scale = fmaxf(scale, fabsf(v));
        const float tol = 1e-5f * scale;

        size_t mismatches = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            const float d = fabsf(ref[i] - fast[i]);
            if (d > worst) worst = d;
            if (d > tol) mismatches++;
        }
        if (mismatches)
            fprintf(stderr, "  %s: %zu/%zu over tol=%g, max_abs_diff=%g (scale=%g)\n", s.name, mismatches, ref.size(),
                    tol, worst, scale);
        CHECK(mismatches == 0, s.name);
    }
}

// ---------------------------------------------------------------------------
// conv2d_im2col_cpu — must be BITWISE identical to the generic conv2d_cpu path
// ---------------------------------------------------------------------------
// Unlike the 1x1 and depthwise guards above, this is an EXACT comparison, not
// a tolerance: the im2col path's contract (see its comment in cpu_ops.h) is
// that it gathers the same patch in the same [ic, ky, kx] order and computes
// every output element through the same dot_product call, so the last ulp may
// not differ — at any thread count. A tolerance here would silently license a
// future edit that changes the accumulation order, which is exactly the edit
// the contract forbids without a per-engine decoded-output A/B.
//
// Shapes target the machinery the tolerance tests don't have: the position
// tiling (a plane larger than one 256-position tile, and one that ends in a
// short tail), the tile-length clamp (K > 8192 floats forces the 16-position
// floor), the fork-join split (n_threads > n_tiles, and a thread-count that
// does not divide the tile count), grouped layouts, boundary gathers under
// stride and padding, and a kernel wider than the padded input.
static void test_conv2d_im2col_equivalence() {
    printf("test_conv2d_im2col_equivalence...\n");

    struct shape {
        int in_ch, out_ch, H, W, kh, kw, stride, pad, groups;
        bool bias;
        const char * name;
    };
    const shape shapes[] = {
        { 8, 8, 8, 8, 3, 3, 1, 1, 1, true, "3x3 pad 1, single tile" },
        { 8, 6, 40, 40, 3, 3, 1, 1, 1, true, "1600 positions, 7 tiles with tail" },
        { 4, 5, 33, 31, 3, 3, 2, 1, 1, true, "stride 2, odd plane" },
        { 12, 9, 16, 16, 3, 3, 1, 1, 3, true, "grouped, 3 groups" },
        { 6, 6, 20, 20, 5, 5, 1, 2, 1, true, "5x5 pad 2 boundary gathers" },
        { 2, 3, 4, 4, 5, 5, 1, 0, 1, false, "kernel wider than input, null bias" },
        { 3, 4, 1, 1, 1, 1, 1, 0, 1, true, "single position" },
        { 1030, 3, 6, 6, 3, 3, 1, 1, 1, true, "K=9270 forces 16-position tile floor" },
        { 7, 11, 30, 30, 1, 1, 1, 0, 1, true, "1x1 through the generic entry" },
    };

    uint32_t seed = 0xd06f00du;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    };

    for (const shape & s : shapes) {
        const int out_H = (s.H + 2 * s.pad - s.kh) / s.stride + 1;
        const int out_W = (s.W + 2 * s.pad - s.kw) / s.stride + 1;
        if (out_H <= 0 || out_W <= 0) continue;
        const int cin = s.in_ch / s.groups;

        std::vector<float> in((size_t)s.in_ch * s.H * s.W);
        std::vector<float> w((size_t)s.out_ch * cin * s.kh * s.kw);
        std::vector<float> b(s.out_ch);
        for (float & v : in) v = next();
        for (float & v : w) v = next();
        for (float & v : b) v = next();
        const float * bias = s.bias ? b.data() : nullptr;

        const size_t n_out = (size_t)s.out_ch * out_H * out_W;
        std::vector<float> ref(n_out, 0.0f), one(n_out, 0.0f), four(n_out, 0.0f);

        conv2d_cpu(in.data(), ref.data(), w.data(), bias, s.in_ch, s.out_ch, s.H, s.W, s.kh, s.kw, s.stride, s.pad,
                   s.groups);
        conv2d_im2col_cpu(in.data(), one.data(), w.data(), bias, s.in_ch, s.out_ch, s.H, s.W, s.kh, s.kw, s.stride,
                          s.pad, s.groups, 1);
        conv2d_im2col_cpu(in.data(), four.data(), w.data(), bias, s.in_ch, s.out_ch, s.H, s.W, s.kh, s.kw, s.stride,
                          s.pad, s.groups, 4);

        const bool eq1 = memcmp(ref.data(), one.data(), n_out * sizeof(float)) == 0;
        const bool eq4 = memcmp(ref.data(), four.data(), n_out * sizeof(float)) == 0;
        if (!eq1 || !eq4) {
            size_t first = 0;
            const float * bad = !eq1 ? one.data() : four.data();
            while (first < n_out && ref[first] == bad[first]) first++;
            fprintf(stderr, "  %s: first mismatch (%s) at %zu: ref=%.9g got=%.9g\n", s.name, !eq1 ? "nt=1" : "nt=4",
                    first, first < n_out ? ref[first] : 0.0f, first < n_out ? bad[first] : 0.0f);
        }
        CHECK(eq1 && eq4, s.name);
    }
}

// ---------------------------------------------------------------------------
// dot_product_wide — four accumulators instead of two
// ---------------------------------------------------------------------------
// Both forms are equally valid summations of the same products; neither is the
// exact real sum, so this compares them within a magnitude-scaled tolerance
// rather than asserting equality. What it is really guarding is the TAIL
// handling: the wide form steps 32 (AVX2) or 16 (NEON) at a time and then falls
// through a narrower loop and a scalar remainder, so an off-by-one in any of
// those drops or double-counts terms. The lengths below straddle every one of
// those boundaries.
//
// A dot product also has an exact reference the SIMD paths cannot fake: with
// b[] all ones, the result is the sum of a[], and with a[] = 1/n and b[] = 1
// it is 1. Those catch a dropped tail that a same-shape comparison against an
// equally-wrong sibling would miss.
static void test_dot_product_wide() {
    printf("test_dot_product_wide...\n");

    const int lengths[] = { 0,  1,  2,  3,  4,  5,  7,  8,  9,   15,  16,  17,  23,  24,
                            31, 32, 33, 47, 48, 63, 64, 65, 100, 127, 128, 129, 1000 };

    uint32_t seed = 0x1234abcdu;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    };

    for (int n : lengths) {
        std::vector<float> a(n > 0 ? n : 1), b(n > 0 ? n : 1);
        for (int i = 0; i < n; i++) {
            a[i] = next();
            b[i] = next();
        }
        const float ref = dot_product(a.data(), b.data(), n);
        const float wide = dot_product_wide(a.data(), b.data(), n);

        float scale = 1e-6f;
        for (int i = 0; i < n; i++) scale = fmaxf(scale, fabsf(a[i] * b[i]));
        scale *= (n > 0 ? n : 1);
        char msg[64];
        snprintf(msg, sizeof(msg), "dot_product_wide agrees, n=%d", n);
        CHECK(fabsf(ref - wide) <= 1e-5f * scale, msg);

        // Exact-invariant check: sum of a[] when b[] is all ones.
        std::vector<float> ones(n > 0 ? n : 1, 1.0f);
        double exact = 0.0;
        for (int i = 0; i < n; i++) exact += a[i];
        const float got = dot_product_wide(a.data(), ones.data(), n);
        snprintf(msg, sizeof(msg), "dot_product_wide sums a[] exactly enough, n=%d", n);
        CHECK(fabsf(got - (float)exact) <= 1e-5f * (fabsf((float)exact) + (float)n * 1e-3f), msg);
    }
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------
static void test_gelu() {
    printf("test_gelu...\n");

    CHECK_CLOSE(gelu(0.0f), 0.0f, 1e-6f, "gelu(0)");
    // gelu(1) ≈ 0.8412 (tanh approx)
    CHECK_CLOSE(gelu(1.0f), 0.8412f, 1e-3f, "gelu(1)");
    // gelu(-1) ≈ -0.1588
    CHECK_CLOSE(gelu(-1.0f), -0.1588f, 1e-3f, "gelu(-1)");
    // Large positive: gelu(x) ≈ x
    CHECK_CLOSE(gelu(5.0f), 5.0f, 1e-3f, "gelu(5)");
    // Large negative: gelu(x) ≈ 0
    CHECK_CLOSE(gelu(-5.0f), 0.0f, 1e-3f, "gelu(-5)");
}

static void test_gelu_erf() {
    printf("test_gelu_erf...\n");

    CHECK_CLOSE(gelu_erf(0.0f), 0.0f, 1e-6f, "gelu_erf(0)");
    CHECK_CLOSE(gelu_erf(1.0f), 0.8413f, 1e-3f, "gelu_erf(1)");
    CHECK_CLOSE(gelu_erf(-1.0f), -0.1587f, 1e-3f, "gelu_erf(-1)");
}

static void test_silu() {
    printf("test_silu...\n");

    CHECK_CLOSE(silu(0.0f), 0.0f, 1e-6f, "silu(0)");
    // silu(1) = 1/(1+e^-1) ≈ 0.7311
    CHECK_CLOSE(silu(1.0f), 0.7311f, 1e-3f, "silu(1)");
    // silu(-1) = -1/(1+e) ≈ -0.2689
    CHECK_CLOSE(silu(-1.0f), -0.2689f, 1e-3f, "silu(-1)");

    // In-place version
    float data[] = { 0.0f, 1.0f, -1.0f };
    silu_inplace(data, 3);
    CHECK_CLOSE(data[0], 0.0f, 1e-6f, "silu_inplace [0]");
    CHECK_CLOSE(data[1], 0.7311f, 1e-3f, "silu_inplace [1]");
    CHECK_CLOSE(data[2], -0.2689f, 1e-3f, "silu_inplace [2]");
}

static void test_softmax() {
    printf("test_softmax...\n");

    float data[] = { 1.0f, 2.0f, 3.0f };
    softmax(data, 3);

    // Check sums to 1
    float sum = data[0] + data[1] + data[2];
    CHECK_CLOSE(sum, 1.0f, 1e-5f, "softmax sums to 1");

    // Check ordering preserved
    CHECK(data[0] < data[1], "softmax ordering 0<1");
    CHECK(data[1] < data[2], "softmax ordering 1<2");

    // Check known value: softmax([1,2,3])[2] = e^3/(e^1+e^2+e^3)
    float e1 = expf(1.0f), e2 = expf(2.0f), e3 = expf(3.0f);
    float expected = e3 / (e1 + e2 + e3);
    CHECK_CLOSE(data[2], expected, 1e-5f, "softmax [2] value");

    // Single element
    float single[] = { 42.0f };
    softmax(single, 1);
    CHECK_CLOSE(single[0], 1.0f, 1e-5f, "softmax single");
}

static void test_hardswish() {
    printf("test_hardswish...\n");

    float data[] = { -4.0f, -3.0f, 0.0f, 3.0f, 5.0f };
    hardswish_inplace(data, 5);

    CHECK_CLOSE(data[0], 0.0f, 1e-5f, "hardswish(-4) = 0");
    CHECK_CLOSE(data[1], 0.0f, 1e-5f, "hardswish(-3) = 0");
    CHECK_CLOSE(data[2], 0.0f, 1e-5f, "hardswish(0) = 0");
    CHECK_CLOSE(data[3], 3.0f, 1e-5f, "hardswish(3) = 3");
    CHECK_CLOSE(data[4], 5.0f, 1e-5f, "hardswish(5) = 5");

    // Middle range: hardswish(1) = 1*(1+3)/6 = 4/6 = 0.6667
    float mid[] = { 1.0f };
    hardswish_inplace(mid, 1);
    CHECK_CLOSE(mid[0], 4.0f / 6.0f, 1e-4f, "hardswish(1) = 2/3");
}

static void test_relu6() {
    printf("test_relu6...\n");

    float data[] = { -2.0f, 0.0f, 3.0f, 6.0f, 10.0f };
    relu6_inplace(data, 5);

    CHECK_CLOSE(data[0], 0.0f, 1e-5f, "relu6(-2) = 0");
    CHECK_CLOSE(data[1], 0.0f, 1e-5f, "relu6(0) = 0");
    CHECK_CLOSE(data[2], 3.0f, 1e-5f, "relu6(3) = 3");
    CHECK_CLOSE(data[3], 6.0f, 1e-5f, "relu6(6) = 6");
    CHECK_CLOSE(data[4], 6.0f, 1e-5f, "relu6(10) = 6");
}

static void test_relu() {
    printf("test_relu...\n");

    float data[] = { -2.0f, 0.0f, 3.0f, 100.0f };
    relu_inplace(data, 4);

    CHECK_CLOSE(data[0], 0.0f, 1e-5f, "relu(-2) = 0");
    CHECK_CLOSE(data[1], 0.0f, 1e-5f, "relu(0) = 0");
    CHECK_CLOSE(data[2], 3.0f, 1e-5f, "relu(3) = 3");
    CHECK_CLOSE(data[3], 100.0f, 1e-5f, "relu(100) = 100");
}

// ---------------------------------------------------------------------------
// mha_1q_cpu
// ---------------------------------------------------------------------------
static void test_mha_1q_cpu() {
    printf("test_mha_1q_cpu...\n");

    // Simple: D=2, n_heads=1, n_kv=1
    // q=[1, 0], k=[1, 0], v=[3, 4]
    // score = (1*1 + 0*0) / sqrt(2) = 1/sqrt(2)
    // softmax of single score = 1.0
    // out = 1.0 * [3, 4] = [3, 4]
    {
        float q[] = { 1.0f, 0.0f };
        float k[] = { 1.0f, 0.0f };
        float v[] = { 3.0f, 4.0f };
        float out[2];

        mha_1q_cpu(q, k, v, out, 1, 2, 1);
        CHECK_CLOSE(out[0], 3.0f, 1e-5f, "mha single kv [0]");
        CHECK_CLOSE(out[1], 4.0f, 1e-5f, "mha single kv [1]");
    }

    // Two KV pairs, D=2, n_heads=1
    // q=[1, 0], k=[[1,0],[0,1]], v=[[10,0],[0,10]]
    // scores: [1/sqrt(2), 0/sqrt(2)] = [0.707, 0]
    // After softmax: some distribution favoring first KV
    {
        float q[] = { 1.0f, 0.0f };
        float k[] = { 1.0f, 0.0f, 0.0f, 1.0f }; // 2 KV pairs
        float v[] = { 10.0f, 0.0f, 0.0f, 10.0f };
        float out[2];

        mha_1q_cpu(q, k, v, out, 2, 2, 1);
        // out[0] should be > 5 (weighted toward first KV's v[0]=10)
        CHECK(out[0] > 5.0f, "mha two kv: out[0] > 5");
        // out[1] should be < 5
        CHECK(out[1] < 5.0f, "mha two kv: out[1] < 5");
        // Should still sum well (weighted average)
        CHECK_CLOSE(out[0] + out[1], 10.0f, 1e-4f, "mha two kv: sum = 10");
    }

    // Multi-head: D=4, n_heads=2, n_kv=1
    {
        float q[] = { 1, 0, 0, 1 }; // head0=[1,0], head1=[0,1]
        float k[] = { 1, 0, 0, 1 }; // head0=[1,0], head1=[0,1]
        float v[] = { 5, 6, 7, 8 }; // head0=[5,6], head1=[7,8]
        float out[4];

        mha_1q_cpu(q, k, v, out, 1, 4, 2);
        // Single KV → softmax([score])=[1.0] → out = v
        CHECK_CLOSE(out[0], 5.0f, 1e-5f, "mha multihead [0]");
        CHECK_CLOSE(out[1], 6.0f, 1e-5f, "mha multihead [1]");
        CHECK_CLOSE(out[2], 7.0f, 1e-5f, "mha multihead [2]");
        CHECK_CLOSE(out[3], 8.0f, 1e-5f, "mha multihead [3]");
    }
}

// ---------------------------------------------------------------------------
// dot_product — SIMD-accelerated dot product
// ---------------------------------------------------------------------------
static void test_dot_product() {
    printf("test_dot_product...\n");

    // Small: 4 elements (scalar tail only on AVX2)
    {
        float a[] = { 1, 2, 3, 4 };
        float b[] = { 5, 6, 7, 8 };
        float r = dot_product(a, b, 4);
        CHECK_CLOSE(r, 70.0f, 1e-5f, "dot4");
    }

    // 8 elements (one AVX2 iteration, one NEON pair)
    {
        float a[8], b[8];
        for (int i = 0; i < 8; i++) {
            a[i] = (float)(i + 1);
            b[i] = (float)(i + 1);
        }
        float r = dot_product(a, b, 8);
        CHECK_CLOSE(r, 204.0f, 1e-4f, "dot8"); // sum(i^2, i=1..8) = 204
    }

    // 17 elements (tests unrolled + tail)
    {
        float a[17], b[17];
        float expected = 0;
        for (int i = 0; i < 17; i++) {
            a[i] = (float)i * 0.1f;
            b[i] = (float)(16 - i) * 0.1f;
            expected += a[i] * b[i];
        }
        float r = dot_product(a, b, 17);
        CHECK_CLOSE(r, expected, 1e-4f, "dot17");
    }

    // 256 elements (typical hidden dim)
    {
        std::vector<float> a(256), b(256);
        float expected = 0;
        for (int i = 0; i < 256; i++) {
            a[i] = sinf((float)i * 0.01f);
            b[i] = cosf((float)i * 0.01f);
            expected += a[i] * b[i];
        }
        float r = dot_product(a.data(), b.data(), 256);
        CHECK_CLOSE(r, expected, 1e-3f, "dot256");
    }

    // 1024 elements (typical embedding dim)
    {
        std::vector<float> a(1024), b(1024);
        double expected = 0;
        for (int i = 0; i < 1024; i++) {
            a[i] = (float)i / 1024.0f;
            b[i] = 1.0f - (float)i / 1024.0f;
            expected += (double)a[i] * b[i];
        }
        float r = dot_product(a.data(), b.data(), 1024);
        CHECK_CLOSE(r, (float)expected, 1e-2f, "dot1024");
    }
}

// ---------------------------------------------------------------------------
// DequantCache
// ---------------------------------------------------------------------------
static void test_dequant_cache() {
    printf("test_dequant_cache...\n");

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();

    float data[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    ggml_backend_buffer_t buf;
    ggml_tensor * t = make_tensor(ctx, backend, GGML_TYPE_F32, 4, data, sizeof(data), &buf);

    DequantCache cache;

    // First access dequantizes
    const float * p1 = cache.get(t);
    CHECK(p1 != nullptr, "cache first access non-null");
    CHECK_CLOSE(p1[0], 1.0f, 1e-6f, "cache val[0]");
    CHECK_CLOSE(p1[3], 4.0f, 1e-6f, "cache val[3]");

    // Second access returns same pointer (cached)
    const float * p2 = cache.get(t);
    CHECK(p1 == p2, "cache returns same pointer");

    // nullptr returns nullptr
    const float * p3 = cache.get(nullptr);
    CHECK(p3 == nullptr, "cache nullptr -> nullptr");

    // clear invalidates
    cache.clear();
    const float * p4 = cache.get(t);
    CHECK(p4 != nullptr, "cache after clear non-null");
    // pointer may differ since it's a new vector

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static int crispembed_test_main() {
    printf("=== core_cpu unit tests ===\n\n");

    test_to_f32();
    test_layernorm_cpu();
    test_layernorm2d_cpu();
    test_rmsnorm_cpu();
    test_linear_cpu();
    test_conv2d_cpu();
    test_conv2d_1x1_equivalence();
    test_conv2d_depthwise_equivalence();
    test_conv2d_im2col_equivalence();
    test_dot_product_wide();
    test_gelu();
    test_gelu_erf();
    test_silu();
    test_softmax();
    test_hardswish();
    test_relu6();
    test_relu();
    test_mha_1q_cpu();
    test_dot_product();
    test_dequant_cache();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
