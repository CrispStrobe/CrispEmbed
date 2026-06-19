// tests/test_core_cpu_ops.cpp — Unit tests for src/core/cpu_ops.h
//
// Pure CPU tests with known-answer inputs. No GGUF model files needed.
// Tests every function in core_cpu namespace.
//
// Usage: ./build/test-core-cpu-ops
// Exit 0 = all pass, non-zero = failure.

#include "core/cpu_ops.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace core_cpu;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) CHECK(fabsf((a) - (b)) < (tol), msg)

// ---------------------------------------------------------------------------
// to_f32 — test with a real ggml F32 tensor via a CPU backend buffer
// ---------------------------------------------------------------------------
// Helper: create a backend-allocated tensor, set its data, return it.
// Caller owns the buffer (returned via buf_out) and must free it.
static ggml_tensor* make_tensor(ggml_context* ctx, ggml_backend_t backend,
                                ggml_type type, int n,
                                const void* data, size_t data_bytes,
                                ggml_backend_buffer_t* buf_out) {
    ggml_tensor* t = ggml_new_tensor_1d(ctx, type, n);
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
    struct ggml_context* ctx = ggml_init(params);

    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);

    // --- F32 tensor ---
    {
        float data[] = {1.0f, -2.5f, 3.14f, 0.0f};
        ggml_backend_buffer_t buf;
        ggml_tensor* t = make_tensor(ctx, backend, GGML_TYPE_F32, 4,
                                     data, sizeof(data), &buf);

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
        ggml_tensor* t = make_tensor(ctx, backend, GGML_TYPE_F16, 3,
                                     fp16_data, sizeof(fp16_data), &buf);

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
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    float b[]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];

    layernorm_cpu(in, out, 4, w, b, 0.0f);

    float inv_std = 1.0f / sqrtf(1.25f);
    CHECK_CLOSE(out[0], -1.5f * inv_std, 1e-5f, "layernorm [0]");
    CHECK_CLOSE(out[1], -0.5f * inv_std, 1e-5f, "layernorm [1]");
    CHECK_CLOSE(out[2],  0.5f * inv_std, 1e-5f, "layernorm [2]");
    CHECK_CLOSE(out[3],  1.5f * inv_std, 1e-5f, "layernorm [3]");

    // With scale and bias
    float w2[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float b2[] = {1.0f, 1.0f, 1.0f, 1.0f};
    layernorm_cpu(in, out, 4, w2, b2, 1e-5f);
    CHECK_CLOSE(out[2], 0.5f / sqrtf(1.25f + 1e-5f) * 2.0f + 1.0f, 1e-4f, "layernorm w/bias [2]");

    // With nullptr w and b (identity scale, zero bias)
    layernorm_cpu(in, out, 4, (const float*)nullptr, (const float*)nullptr, 0.0f);
    CHECK_CLOSE(out[0], -1.5f * inv_std, 1e-5f, "layernorm nullptr w/b [0]");

    // In-place (in == out)
    float inplace[] = {1.0f, 2.0f, 3.0f, 4.0f};
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
        1.0f, 2.0f,   // channel 0: [1, 2]
        3.0f, 4.0f    // channel 1: [3, 4]
    };
    float w[] = {1.0f, 1.0f};
    float b[] = {0.0f, 0.0f};
    float out[4];

    layernorm2d_cpu(in, out, 2, 1, 2, w, b, 0.0f);

    // Position (0,0): (1-2)/1 = -1, (3-2)/1 = 1
    CHECK_CLOSE(out[0], -1.0f, 1e-5f, "layernorm2d c0 (0,0)");
    CHECK_CLOSE(out[2],  1.0f, 1e-5f, "layernorm2d c1 (0,0)");
    // Position (0,1): (2-3)/1 = -1, (4-3)/1 = 1
    CHECK_CLOSE(out[1], -1.0f, 1e-5f, "layernorm2d c0 (0,1)");
    CHECK_CLOSE(out[3],  1.0f, 1e-5f, "layernorm2d c1 (0,1)");
}

// ---------------------------------------------------------------------------
// rmsnorm_cpu
// ---------------------------------------------------------------------------
static void test_rmsnorm_cpu() {
    printf("test_rmsnorm_cpu...\n");

    // Input: [3, 4], rms = sqrt((9+16)/2) = sqrt(12.5)
    // inv_rms = 1/sqrt(12.5)
    // With w=[1,1], out[i] = in[i] / sqrt(12.5)
    float in[] = {3.0f, 4.0f};
    float w[]  = {1.0f, 1.0f};
    float out[2];

    rmsnorm_cpu(in, out, 2, w, 0.0f);

    float inv_rms = 1.0f / sqrtf(12.5f);
    CHECK_CLOSE(out[0], 3.0f * inv_rms, 1e-5f, "rmsnorm [0]");
    CHECK_CLOSE(out[1], 4.0f * inv_rms, 1e-5f, "rmsnorm [1]");

    // With scale weights
    float w2[] = {2.0f, 0.5f};
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
    float in[] = {1.0f, 2.0f};
    float w[]  = {1.0f, 3.0f, 2.0f, 4.0f};  // [out_dim=2, in_dim=2]
    float b[]  = {0.5f, -0.5f};
    float out[2];

    linear_cpu(in, out, 2, 2, w, b);
    CHECK_CLOSE(out[0], 7.5f, 1e-5f, "linear [0] with bias");
    CHECK_CLOSE(out[1], 9.5f, 1e-5f, "linear [1] with bias");

    // Without bias
    linear_cpu(in, out, 2, 2, w, nullptr);
    CHECK_CLOSE(out[0], 7.0f, 1e-5f, "linear [0] no bias");
    CHECK_CLOSE(out[1], 10.0f, 1e-5f, "linear [1] no bias");

    // Rectangular: in_dim=3, out_dim=2
    float in3[] = {1.0f, 2.0f, 3.0f};
    float w32[] = {1.0f, 0.0f, -1.0f,   // row 0: 1*1 + 0*2 + (-1)*3 = -2
                   0.0f, 1.0f,  1.0f};  // row 1: 0*1 + 1*2 + 1*3 = 5
    linear_cpu(in3, out, 3, 2, w32, nullptr);
    CHECK_CLOSE(out[0], -2.0f, 1e-5f, "linear rect [0]");
    CHECK_CLOSE(out[1],  5.0f, 1e-5f, "linear rect [1]");
}

// ---------------------------------------------------------------------------
// conv2d_cpu — standard and grouped convolution
// ---------------------------------------------------------------------------
static void test_conv2d_cpu() {
    printf("test_conv2d_cpu...\n");

    // 1x1 conv, 1 channel in, 1 channel out, 3x3 input, no padding, stride=1
    // This is effectively linear per-pixel with a scalar weight
    {
        float in[9] = {1,2,3, 4,5,6, 7,8,9};  // [1, 3, 3]
        float w[1] = {2.0f};  // [1, 1, 1, 1]
        float b[1] = {1.0f};
        float out[9];

        conv2d_cpu(in, out, w, b, 1, 1, 3, 3, 1, 1, 1, 0);
        CHECK_CLOSE(out[0], 3.0f, 1e-5f, "conv2d 1x1 [0]");  // 2*1 + 1
        CHECK_CLOSE(out[4], 11.0f, 1e-5f, "conv2d 1x1 [4]"); // 2*5 + 1
        CHECK_CLOSE(out[8], 19.0f, 1e-5f, "conv2d 1x1 [8]"); // 2*9 + 1
    }

    // 3x3 conv, 1 channel, 1 filter, 3x3 input, no padding, stride=1
    // Output: 1x1
    {
        float in[9] = {1,2,3, 4,5,6, 7,8,9};
        float w[9] = {1,0,0, 0,1,0, 0,0,1};  // diagonal filter
        float b[1] = {0.0f};
        float out[1];

        conv2d_cpu(in, out, w, b, 1, 1, 3, 3, 3, 3, 1, 0);
        // sum of diagonal: 1 + 5 + 9 = 15
        CHECK_CLOSE(out[0], 15.0f, 1e-5f, "conv2d 3x3 diagonal");
    }

    // 3x3 conv with padding=1, stride=1 — output same size as input
    {
        float in[4] = {1, 2, 3, 4};  // [1, 2, 2]
        // All-ones 3x3 kernel
        float w[9] = {1,1,1, 1,1,1, 1,1,1};
        float b[1] = {0.0f};
        float out[4];

        conv2d_cpu(in, out, w, b, 1, 1, 2, 2, 3, 3, 1, 1);
        // Output (0,0): in-range pixels are (0,0),(0,1),(1,0),(1,1) = 1+2+3+4 = 10
        CHECK_CLOSE(out[0], 10.0f, 1e-5f, "conv2d padded [0]");
    }

    // Stride=2 test
    {
        float in[16];  // [1, 4, 4]
        for (int i = 0; i < 16; i++) in[i] = (float)(i + 1);
        float w[1] = {1.0f};  // 1x1 conv
        float out[4];  // [1, 2, 2]

        conv2d_cpu(in, out, w, nullptr, 1, 1, 4, 4, 1, 1, 2, 0);
        CHECK_CLOSE(out[0], 1.0f, 1e-5f, "conv2d stride2 [0]");
        CHECK_CLOSE(out[1], 3.0f, 1e-5f, "conv2d stride2 [1]");
        CHECK_CLOSE(out[2], 9.0f, 1e-5f, "conv2d stride2 [2]");
        CHECK_CLOSE(out[3], 11.0f, 1e-5f, "conv2d stride2 [3]");
    }

    // Depthwise (groups=channels) test
    {
        // 2 channels, 2 groups (depthwise), 2x2 input, 1x1 kernel
        float in[8] = {1, 2, 3, 4,   // ch0: [[1,2],[3,4]]
                       5, 6, 7, 8};   // ch1: [[5,6],[7,8]]
        float w[2] = {2.0f, 3.0f};   // [2, 1, 1, 1] — one scalar per channel
        float b[2] = {0.0f, 0.0f};
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
        float in[4] = {1, 2, 3, 4};
        float w[1] = {1.0f};
        float out[4];
        conv2d_cpu(in, out, w, nullptr, 1, 1, 2, 2, 1, 1, 1, 0);
        CHECK_CLOSE(out[0], 1.0f, 1e-5f, "conv2d no bias [0]");
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
    float data[] = {0.0f, 1.0f, -1.0f};
    silu_inplace(data, 3);
    CHECK_CLOSE(data[0], 0.0f, 1e-6f, "silu_inplace [0]");
    CHECK_CLOSE(data[1], 0.7311f, 1e-3f, "silu_inplace [1]");
    CHECK_CLOSE(data[2], -0.2689f, 1e-3f, "silu_inplace [2]");
}

static void test_softmax() {
    printf("test_softmax...\n");

    float data[] = {1.0f, 2.0f, 3.0f};
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
    float single[] = {42.0f};
    softmax(single, 1);
    CHECK_CLOSE(single[0], 1.0f, 1e-5f, "softmax single");
}

static void test_hardswish() {
    printf("test_hardswish...\n");

    float data[] = {-4.0f, -3.0f, 0.0f, 3.0f, 5.0f};
    hardswish_inplace(data, 5);

    CHECK_CLOSE(data[0], 0.0f, 1e-5f, "hardswish(-4) = 0");
    CHECK_CLOSE(data[1], 0.0f, 1e-5f, "hardswish(-3) = 0");
    CHECK_CLOSE(data[2], 0.0f, 1e-5f, "hardswish(0) = 0");
    CHECK_CLOSE(data[3], 3.0f, 1e-5f, "hardswish(3) = 3");
    CHECK_CLOSE(data[4], 5.0f, 1e-5f, "hardswish(5) = 5");

    // Middle range: hardswish(1) = 1*(1+3)/6 = 4/6 = 0.6667
    float mid[] = {1.0f};
    hardswish_inplace(mid, 1);
    CHECK_CLOSE(mid[0], 4.0f / 6.0f, 1e-4f, "hardswish(1) = 2/3");
}

static void test_relu6() {
    printf("test_relu6...\n");

    float data[] = {-2.0f, 0.0f, 3.0f, 6.0f, 10.0f};
    relu6_inplace(data, 5);

    CHECK_CLOSE(data[0], 0.0f, 1e-5f, "relu6(-2) = 0");
    CHECK_CLOSE(data[1], 0.0f, 1e-5f, "relu6(0) = 0");
    CHECK_CLOSE(data[2], 3.0f, 1e-5f, "relu6(3) = 3");
    CHECK_CLOSE(data[3], 6.0f, 1e-5f, "relu6(6) = 6");
    CHECK_CLOSE(data[4], 6.0f, 1e-5f, "relu6(10) = 6");
}

static void test_relu() {
    printf("test_relu...\n");

    float data[] = {-2.0f, 0.0f, 3.0f, 100.0f};
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
        float q[] = {1.0f, 0.0f};
        float k[] = {1.0f, 0.0f};
        float v[] = {3.0f, 4.0f};
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
        float q[] = {1.0f, 0.0f};
        float k[] = {1.0f, 0.0f, 0.0f, 1.0f};  // 2 KV pairs
        float v[] = {10.0f, 0.0f, 0.0f, 10.0f};
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
        float q[] = {1, 0, 0, 1};       // head0=[1,0], head1=[0,1]
        float k[] = {1, 0, 0, 1};       // head0=[1,0], head1=[0,1]
        float v[] = {5, 6, 7, 8};       // head0=[5,6], head1=[7,8]
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
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== core_cpu unit tests ===\n\n");

    test_to_f32();
    test_layernorm_cpu();
    test_layernorm2d_cpu();
    test_rmsnorm_cpu();
    test_linear_cpu();
    test_conv2d_cpu();
    test_gelu();
    test_gelu_erf();
    test_silu();
    test_softmax();
    test_hardswish();
    test_relu6();
    test_relu();
    test_mha_1q_cpu();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
