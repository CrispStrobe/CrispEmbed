// tests/test_core_vlm_attention.cpp — Unit tests for src/core/vlm_attention.h
//
// Pure CPU tests with known-answer inputs. No GGUF model files needed.
// Tests every function in core_vlm namespace.
//
// Usage: ./build/test-core-vlm-attention
// Exit 0 = all pass, non-zero = failure.

#include "core/vlm_attention.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace core_vlm;

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

// ===========================================================================
// apply_rope — NEGHALF style (SmolLM2 / GPT-NeoX)
// ===========================================================================
static void test_rope_neghalf_identity() {
    printf("test_rope_neghalf_identity...\n");

    // position=0 → angle=0 for all freqs → cos=1, sin=0 → no rotation
    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};  // 1 head, head_dim=4
    apply_rope(qk, 1, 4, 0, 10000.0f, RoPEStyle::NEGHALF);
    CHECK_CLOSE(qk[0], 1.0f, 1e-6f, "neghalf pos=0 [0]");
    CHECK_CLOSE(qk[1], 2.0f, 1e-6f, "neghalf pos=0 [1]");
    CHECK_CLOSE(qk[2], 3.0f, 1e-6f, "neghalf pos=0 [2]");
    CHECK_CLOSE(qk[3], 4.0f, 1e-6f, "neghalf pos=0 [3]");
}

static void test_rope_neghalf_known_angle() {
    printf("test_rope_neghalf_known_angle...\n");

    // 1 head, head_dim=4, half=2
    // position=1, theta=1.0 (so freq = 1/theta^(2d/D))
    // d=0: freq = 1/1^0 = 1.0, angle = 1*1 = 1.0
    // d=1: freq = 1/1^0.5 = 1.0, angle = 1*1 = 1.0
    // With theta=1.0, all freqs = 1.0, angle = 1.0 rad
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    //
    // NEGHALF pairs: (0,2) and (1,3)
    // qk[0] = lo*cos - hi*sin = 1*0.5403 - 3*0.8415
    // qk[2] = hi*cos + lo*sin = 3*0.5403 + 1*0.8415

    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(qk, 1, 4, 1, 1.0f, RoPEStyle::NEGHALF);

    float c = cosf(1.0f), s = sinf(1.0f);
    CHECK_CLOSE(qk[0], 1.0f * c - 3.0f * s, 1e-5f, "neghalf angle [0]");
    CHECK_CLOSE(qk[1], 2.0f * c - 4.0f * s, 1e-5f, "neghalf angle [1]");
    CHECK_CLOSE(qk[2], 3.0f * c + 1.0f * s, 1e-5f, "neghalf angle [2]");
    CHECK_CLOSE(qk[3], 4.0f * c + 2.0f * s, 1e-5f, "neghalf angle [3]");
}

static void test_rope_neghalf_multihead() {
    printf("test_rope_neghalf_multihead...\n");

    // 2 heads, head_dim=2, half=1
    // head 0: [1, 2], head 1: [3, 4]
    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(qk, 2, 2, 1, 1.0f, RoPEStyle::NEGHALF);

    // d=0: freq = 1/1^0 = 1.0, angle = 1.0
    float c = cosf(1.0f), s = sinf(1.0f);
    // head 0: pair (0, 1) → lo=1, hi=2
    CHECK_CLOSE(qk[0], 1.0f * c - 2.0f * s, 1e-5f, "neghalf mh h0[0]");
    CHECK_CLOSE(qk[1], 2.0f * c + 1.0f * s, 1e-5f, "neghalf mh h0[1]");
    // head 1: pair (0, 1) → lo=3, hi=4
    CHECK_CLOSE(qk[2], 3.0f * c - 4.0f * s, 1e-5f, "neghalf mh h1[0]");
    CHECK_CLOSE(qk[3], 4.0f * c + 3.0f * s, 1e-5f, "neghalf mh h1[1]");
}

static void test_rope_neghalf_large_position() {
    printf("test_rope_neghalf_large_position...\n");

    // Test at position=100 with realistic theta=10000, head_dim=64
    // Verify no NaN/inf and that rotation is applied
    std::vector<float> qk(2 * 64, 1.0f);  // 2 heads, dim=64
    apply_rope(qk.data(), 2, 64, 100, 10000.0f, RoPEStyle::NEGHALF);

    bool any_nan = false;
    bool any_changed = false;
    for (int i = 0; i < 128; i++) {
        if (std::isnan(qk[i]) || std::isinf(qk[i])) any_nan = true;
        if (fabsf(qk[i] - 1.0f) > 1e-6f) any_changed = true;
    }
    CHECK(!any_nan, "neghalf large pos no NaN");
    CHECK(any_changed, "neghalf large pos values changed");
}

// ===========================================================================
// apply_rope — INTERLEAVED style (Llama / Granite)
// ===========================================================================
static void test_rope_interleaved_identity() {
    printf("test_rope_interleaved_identity...\n");

    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(qk, 1, 4, 0, 10000.0f, RoPEStyle::INTERLEAVED);
    CHECK_CLOSE(qk[0], 1.0f, 1e-6f, "interleaved pos=0 [0]");
    CHECK_CLOSE(qk[1], 2.0f, 1e-6f, "interleaved pos=0 [1]");
    CHECK_CLOSE(qk[2], 3.0f, 1e-6f, "interleaved pos=0 [2]");
    CHECK_CLOSE(qk[3], 4.0f, 1e-6f, "interleaved pos=0 [3]");
}

static void test_rope_interleaved_known_angle() {
    printf("test_rope_interleaved_known_angle...\n");

    // 1 head, head_dim=4
    // INTERLEAVED pairs: (0,1) and (2,3)
    // d=0: freq = 1/theta^(0/4) = 1.0, angle = 1.0
    // d=1: freq = 1/theta^(2/4) = 1/1^0.5 = 1.0, angle = 1.0
    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(qk, 1, 4, 1, 1.0f, RoPEStyle::INTERLEAVED);

    float c = cosf(1.0f), s = sinf(1.0f);
    // pair (0,1): v0=1, v1=2
    CHECK_CLOSE(qk[0], 1.0f * c - 2.0f * s, 1e-5f, "interleaved angle [0]");
    CHECK_CLOSE(qk[1], 1.0f * s + 2.0f * c, 1e-5f, "interleaved angle [1]");
    // pair (2,3): v0=3, v1=4
    CHECK_CLOSE(qk[2], 3.0f * c - 4.0f * s, 1e-5f, "interleaved angle [2]");
    CHECK_CLOSE(qk[3], 3.0f * s + 4.0f * c, 1e-5f, "interleaved angle [3]");
}

static void test_rope_interleaved_multihead() {
    printf("test_rope_interleaved_multihead...\n");

    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};  // 2 heads, head_dim=2
    apply_rope(qk, 2, 2, 1, 1.0f, RoPEStyle::INTERLEAVED);

    float c = cosf(1.0f), s = sinf(1.0f);
    // head 0, pair (0,1): v0=1, v1=2
    CHECK_CLOSE(qk[0], 1.0f * c - 2.0f * s, 1e-5f, "interleaved mh h0[0]");
    CHECK_CLOSE(qk[1], 1.0f * s + 2.0f * c, 1e-5f, "interleaved mh h0[1]");
    // head 1, pair (0,1): v0=3, v1=4
    CHECK_CLOSE(qk[2], 3.0f * c - 4.0f * s, 1e-5f, "interleaved mh h1[0]");
    CHECK_CLOSE(qk[3], 3.0f * s + 4.0f * c, 1e-5f, "interleaved mh h1[1]");
}

static void test_rope_interleaved_different_freqs() {
    printf("test_rope_interleaved_different_freqs...\n");

    // head_dim=4, theta=100, position=5
    // d=0: freq = 1/100^(0/4) = 1.0, angle = 5.0
    // d=1: freq = 1/100^(2/4) = 1/10 = 0.1, angle = 0.5
    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(qk, 1, 4, 5, 100.0f, RoPEStyle::INTERLEAVED);

    float c0 = cosf(5.0f), s0 = sinf(5.0f);
    float c1 = cosf(0.5f), s1 = sinf(0.5f);
    CHECK_CLOSE(qk[0], 1.0f * c0 - 2.0f * s0, 1e-5f, "interleaved freqs [0]");
    CHECK_CLOSE(qk[1], 1.0f * s0 + 2.0f * c0, 1e-5f, "interleaved freqs [1]");
    CHECK_CLOSE(qk[2], 3.0f * c1 - 4.0f * s1, 1e-5f, "interleaved freqs [2]");
    CHECK_CLOSE(qk[3], 3.0f * s1 + 4.0f * c1, 1e-5f, "interleaved freqs [3]");
}

// ===========================================================================
// RoPE: NEGHALF vs INTERLEAVED equivalence at position=0
// ===========================================================================
static void test_rope_styles_identity_at_zero() {
    printf("test_rope_styles_identity_at_zero...\n");

    // Both styles should be identity at position=0
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    apply_rope(a, 2, 4, 0, 10000.0f, RoPEStyle::NEGHALF);
    apply_rope(b, 2, 4, 0, 10000.0f, RoPEStyle::INTERLEAVED);

    for (int i = 0; i < 8; i++) {
        CHECK_CLOSE(a[i], b[i], 1e-6f, "both styles identity at pos=0");
    }
}

// ===========================================================================
// KV cache — alloc, offset helpers
// ===========================================================================
static void test_kv_cache_alloc() {
    printf("test_kv_cache_alloc...\n");

    auto cache = alloc_kv_cache(2, 10, 4, 8);  // 2 layers, 10 seq, 4 heads, dim 8
    int kv_dim = 4 * 8;  // 32
    size_t expected = 2 * 2 * 10 * 32;  // 1280
    CHECK(cache.size() == expected, "kv cache size");

    // Should be zero-initialized
    bool all_zero = true;
    for (size_t i = 0; i < cache.size(); i++)
        if (cache[i] != 0.0f) { all_zero = false; break; }
    CHECK(all_zero, "kv cache zero-init");
}

static void test_kv_offsets() {
    printf("test_kv_offsets...\n");

    // Layout: [layer][K_or_V][seq][kv_dim]
    // layer 0, K: offset 0
    // layer 0, V: offset max_seq * kv_dim
    // layer 1, K: offset 2 * max_seq * kv_dim
    int kv_dim = 32, max_seq = 10, n_layers = 2;

    CHECK(kv_k_offset(0, 0, kv_dim, max_seq, n_layers) == 0,
          "k_offset layer0 seq0");
    CHECK(kv_k_offset(0, 5, kv_dim, max_seq, n_layers) == 5u * 32,
          "k_offset layer0 seq5");
    CHECK(kv_v_offset(0, 0, kv_dim, max_seq, n_layers) == 10u * 32,
          "v_offset layer0 seq0");
    CHECK(kv_v_offset(0, 5, kv_dim, max_seq, n_layers) == 10u * 32 + 5u * 32,
          "v_offset layer0 seq5");

    // Layer 1
    CHECK(kv_k_offset(1, 0, kv_dim, max_seq, n_layers) == 2u * 10 * 32,
          "k_offset layer1 seq0");
    CHECK(kv_v_offset(1, 0, kv_dim, max_seq, n_layers) == 2u * 10 * 32 + 10u * 32,
          "v_offset layer1 seq0");
}

static void test_kv_offset_write_read_roundtrip() {
    printf("test_kv_offset_write_read_roundtrip...\n");

    int n_layers = 2, max_seq = 4, n_kv_heads = 2, head_dim = 3;
    int kv_dim = n_kv_heads * head_dim;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv_heads, head_dim);

    // Write sentinel values at specific positions
    float k_val[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float v_val[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    // Layer 1, position 2
    size_t ko = kv_k_offset(1, 2, kv_dim, max_seq, n_layers);
    size_t vo = kv_v_offset(1, 2, kv_dim, max_seq, n_layers);
    memcpy(cache.data() + ko, k_val, kv_dim * sizeof(float));
    memcpy(cache.data() + vo, v_val, kv_dim * sizeof(float));

    // Read back
    for (int i = 0; i < kv_dim; i++) {
        CHECK_CLOSE(cache[ko + i], k_val[i], 1e-6f, "kv roundtrip K");
        CHECK_CLOSE(cache[vo + i], v_val[i], 1e-6f, "kv roundtrip V");
    }

    // Verify other positions are still zero
    size_t ko_other = kv_k_offset(0, 0, kv_dim, max_seq, n_layers);
    CHECK_CLOSE(cache[ko_other], 0.0f, 1e-6f, "kv other pos zero");
}

// ===========================================================================
// gqa_attn_step — single head, no GQA repeat
// ===========================================================================
static void test_gqa_attn_single_head() {
    printf("test_gqa_attn_single_head...\n");

    // 1 head, head_dim=2, 1 layer, max_seq=4
    // First step: n_past=0 → only attends to itself
    int n_heads = 1, n_kv = 1, d = 2, max_seq = 4, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    float q[] = {1.0f, 0.0f};
    float k[] = {1.0f, 0.0f};
    float v[] = {3.0f, 4.0f};
    float out[2] = {0};

    gqa_attn_step(q, k, v, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // With only 1 position, softmax = 1.0, so output = v
    CHECK_CLOSE(out[0], 3.0f, 1e-5f, "single head attn out[0]");
    CHECK_CLOSE(out[1], 4.0f, 1e-5f, "single head attn out[1]");
}

static void test_gqa_attn_two_positions() {
    printf("test_gqa_attn_two_positions...\n");

    // 1 head, head_dim=2, seq has 2 positions
    int n_heads = 1, n_kv = 1, d = 2, max_seq = 4, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    // Step 0: write first K,V
    float q0[] = {1.0f, 0.0f};
    float k0[] = {1.0f, 0.0f};
    float v0[] = {10.0f, 0.0f};
    float out[2];
    gqa_attn_step(q0, k0, v0, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // Step 1: Q aligns perfectly with K0 but not K1
    float q1[] = {1.0f, 0.0f};
    float k1[] = {0.0f, 1.0f};  // orthogonal to q
    float v1[] = {0.0f, 20.0f};
    gqa_attn_step(q1, k1, v1, cache.data(), n_heads, n_kv, d, max_seq, 1, 0, n_layers, out);

    // Q·K0 = 1*1 + 0*0 = 1.0, scaled by 1/sqrt(2) ≈ 0.7071
    // Q·K1 = 1*0 + 0*1 = 0.0, scaled = 0.0
    // softmax([0.7071, 0.0]) → e^0.7071 / (e^0.7071 + e^0) and e^0 / (e^0.7071 + e^0)
    float s0_raw = 0.7071067f, s1_raw = 0.0f;
    float e0 = expf(s0_raw), e1 = expf(s1_raw);
    float sum = e0 + e1;
    float w0 = e0 / sum, w1 = e1 / sum;

    float expected0 = w0 * 10.0f + w1 * 0.0f;
    float expected1 = w0 * 0.0f + w1 * 20.0f;
    CHECK_CLOSE(out[0], expected0, 1e-4f, "two pos attn out[0]");
    CHECK_CLOSE(out[1], expected1, 1e-4f, "two pos attn out[1]");

    // Sanity: output[0] should be larger since Q aligns with K0
    CHECK(out[0] > out[1], "two pos: Q aligned with K0 gives more weight to V0");
}

static void test_gqa_attn_multihead() {
    printf("test_gqa_attn_multihead...\n");

    // 2 heads (MHA, not GQA), head_dim=2
    int n_heads = 2, n_kv = 2, d = 2, max_seq = 4, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    // Q: head0=[1,0], head1=[0,1]
    float q[] = {1.0f, 0.0f, 0.0f, 1.0f};
    // K: kv_head0=[1,0], kv_head1=[0,1]
    float k[] = {1.0f, 0.0f, 0.0f, 1.0f};
    // V: kv_head0=[5,6], kv_head1=[7,8]
    float v[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4];

    gqa_attn_step(q, k, v, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // Single position → softmax = 1.0 → out = V
    CHECK_CLOSE(out[0], 5.0f, 1e-5f, "multihead h0[0]");
    CHECK_CLOSE(out[1], 6.0f, 1e-5f, "multihead h0[1]");
    CHECK_CLOSE(out[2], 7.0f, 1e-5f, "multihead h1[0]");
    CHECK_CLOSE(out[3], 8.0f, 1e-5f, "multihead h1[1]");
}

static void test_gqa_attn_gqa_repeat() {
    printf("test_gqa_attn_gqa_repeat...\n");

    // 4 heads, 2 KV heads → kv_repeat=2
    // heads 0,1 share kv_head 0; heads 2,3 share kv_head 1
    int n_heads = 4, n_kv = 2, d = 2, max_seq = 4, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    // All Q heads = [1, 0]
    float q[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    float k[] = {1.0f, 0.0f, 0.0f, 1.0f};  // kv0=[1,0], kv1=[0,1]
    float v[] = {10.0f, 20.0f, 30.0f, 40.0f};  // kv0=[10,20], kv1=[30,40]
    float out[8];

    gqa_attn_step(q, k, v, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // Single position, softmax=1. Heads 0,1 use kv_head 0; heads 2,3 use kv_head 1
    CHECK_CLOSE(out[0], 10.0f, 1e-5f, "gqa h0[0] shares kv0");
    CHECK_CLOSE(out[1], 20.0f, 1e-5f, "gqa h0[1] shares kv0");
    CHECK_CLOSE(out[2], 10.0f, 1e-5f, "gqa h1[0] shares kv0");
    CHECK_CLOSE(out[3], 20.0f, 1e-5f, "gqa h1[1] shares kv0");
    CHECK_CLOSE(out[4], 30.0f, 1e-5f, "gqa h2[0] shares kv1");
    CHECK_CLOSE(out[5], 40.0f, 1e-5f, "gqa h2[1] shares kv1");
    CHECK_CLOSE(out[6], 30.0f, 1e-5f, "gqa h3[0] shares kv1");
    CHECK_CLOSE(out[7], 40.0f, 1e-5f, "gqa h3[1] shares kv1");
}

static void test_gqa_attn_multilayer() {
    printf("test_gqa_attn_multilayer...\n");

    // 2 layers, verify they don't interfere
    int n_heads = 1, n_kv = 1, d = 2, max_seq = 4, n_layers = 2;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    float q[] = {1.0f, 0.0f};
    float k[] = {1.0f, 0.0f};
    float out[2];

    // Layer 0: V = [10, 20]
    float v0[] = {10.0f, 20.0f};
    gqa_attn_step(q, k, v0, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);
    CHECK_CLOSE(out[0], 10.0f, 1e-5f, "multilayer L0 out[0]");
    CHECK_CLOSE(out[1], 20.0f, 1e-5f, "multilayer L0 out[1]");

    // Layer 1: V = [30, 40]
    float v1[] = {30.0f, 40.0f};
    gqa_attn_step(q, k, v1, cache.data(), n_heads, n_kv, d, max_seq, 0, 1, n_layers, out);
    CHECK_CLOSE(out[0], 30.0f, 1e-5f, "multilayer L1 out[0]");
    CHECK_CLOSE(out[1], 40.0f, 1e-5f, "multilayer L1 out[1]");

    // Verify layer 0 cache wasn't corrupted
    size_t ko = kv_k_offset(0, 0, n_kv * d, max_seq, n_layers);
    size_t vo = kv_v_offset(0, 0, n_kv * d, max_seq, n_layers);
    CHECK_CLOSE(cache[ko], 1.0f, 1e-6f, "L0 K preserved");
    CHECK_CLOSE(cache[vo], 10.0f, 1e-6f, "L0 V preserved");
}

static void test_gqa_attn_uniform_values() {
    printf("test_gqa_attn_uniform_values...\n");

    // When all values are the same, attention output = that value
    // regardless of Q/K alignment
    int n_heads = 2, n_kv = 1, d = 2, max_seq = 4, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    // Step 0
    float q0[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k0[] = {0.5f, 0.5f};
    float v0[] = {7.0f, 7.0f};
    float out[4];
    gqa_attn_step(q0, k0, v0, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // Step 1: different K but same V
    float q1[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k1[] = {-0.5f, 0.5f};
    float v1[] = {7.0f, 7.0f};
    gqa_attn_step(q1, k1, v1, cache.data(), n_heads, n_kv, d, max_seq, 1, 0, n_layers, out);

    // All outputs should be 7.0 since all V entries are 7.0
    for (int i = 0; i < 4; i++)
        CHECK_CLOSE(out[i], 7.0f, 1e-5f, "uniform V → uniform output");
}

// ===========================================================================
// swiglu_ffn
// ===========================================================================
static void test_swiglu_ffn_basic() {
    printf("test_swiglu_ffn_basic...\n");

    // hidden=2, intermediate=3
    // x = [1, 2]
    // gate_w = identity-ish 3x2, up_w = identity-ish 3x2, down_w = identity-ish 2x3
    //
    // Compute manually:
    //   gate = gate_w @ x = [g0, g1, g2]
    //   up   = up_w   @ x = [u0, u1, u2]
    //   mid  = silu(gate) * up
    //   out  = down_w @ mid

    float x[] = {1.0f, 2.0f};

    // gate_w: [3, 2] → each row is [1, 0], [0, 1], [1, 1]
    float gate_w[] = {1.0f, 0.0f,
                      0.0f, 1.0f,
                      1.0f, 1.0f};
    // up_w: all ones [3, 2]
    float up_w[] = {1.0f, 1.0f,
                    1.0f, 1.0f,
                    1.0f, 1.0f};
    // down_w: [2, 3]
    float down_w[] = {1.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f};

    float out[2];
    swiglu_ffn(x, out, 2, 3, gate_w, up_w, down_w);

    // gate = [1*1+0*2, 0*1+1*2, 1*1+1*2] = [1, 2, 3]
    // up   = [1+2, 1+2, 1+2] = [3, 3, 3]
    // silu(1) = 1/(1+e^-1) ≈ 0.7311
    // silu(2) = 2/(1+e^-2) ≈ 1.7616
    // silu(3) = 3/(1+e^-3) ≈ 2.8577
    // mid = [0.7311*3, 1.7616*3, 2.8577*3] = [2.1933, 5.2847, 8.5731]
    // out[0] = 1*2.1933 + 0*5.2847 + 0*8.5731 = 2.1933
    // out[1] = 0*2.1933 + 1*5.2847 + 0*8.5731 = 5.2847

    float silu1 = 1.0f / (1.0f + expf(-1.0f));
    float silu2 = 2.0f / (1.0f + expf(-2.0f));
    CHECK_CLOSE(out[0], silu1 * 3.0f, 1e-4f, "swiglu out[0]");
    CHECK_CLOSE(out[1], silu2 * 3.0f, 1e-4f, "swiglu out[1]");
}

static void test_swiglu_ffn_zero_input() {
    printf("test_swiglu_ffn_zero_input...\n");

    // x = 0 → gate = 0, up = 0, silu(0) = 0, out = 0
    float x[] = {0.0f, 0.0f};
    float gate_w[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float up_w[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float down_w[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float out[2];

    swiglu_ffn(x, out, 2, 2, gate_w, up_w, down_w);
    CHECK_CLOSE(out[0], 0.0f, 1e-6f, "swiglu zero [0]");
    CHECK_CLOSE(out[1], 0.0f, 1e-6f, "swiglu zero [1]");
}

static void test_swiglu_ffn_silu_gating() {
    printf("test_swiglu_ffn_silu_gating...\n");

    // Verify the SiLU gating property: large negative gate → output near 0
    // hidden=1, intermediate=1
    // x = [1]
    // gate_w = [-10] → gate = -10, silu(-10) ≈ 0.0000454
    // up_w = [1] → up = 1
    // down_w = [1]
    // out ≈ silu(-10) * 1 ≈ 0

    float x[] = {1.0f};
    float gate_w[] = {-10.0f};
    float up_w[] = {1.0f};
    float down_w[] = {1.0f};
    float out[1];

    swiglu_ffn(x, out, 1, 1, gate_w, up_w, down_w);
    CHECK(fabsf(out[0]) < 0.001f, "swiglu large neg gate → near zero");
}

static void test_swiglu_ffn_asymmetric() {
    printf("test_swiglu_ffn_asymmetric...\n");

    // hidden=2, intermediate=4 (typical: ffn_dim > hidden_dim)
    float x[] = {1.0f, -1.0f};
    // gate_w [4,2]: identity repeated
    float gate_w[] = {1.0f, 0.0f,
                      0.0f, 1.0f,
                      1.0f, 0.0f,
                      0.0f, 1.0f};
    float up_w[] = {1.0f, 0.0f,
                    0.0f, 1.0f,
                    1.0f, 0.0f,
                    0.0f, 1.0f};
    // down_w [2,4]: sum pairs
    float down_w[] = {1.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 1.0f};
    float out[2];

    swiglu_ffn(x, out, 2, 4, gate_w, up_w, down_w);

    // gate = [1, -1, 1, -1], up = [1, -1, 1, -1]
    // mid = [silu(1)*1, silu(-1)*(-1), silu(1)*1, silu(-1)*(-1)]
    float s1 = 1.0f / (1.0f + expf(-1.0f));   // silu(1) ≈ 0.7311
    float sm1 = -1.0f / (1.0f + expf(1.0f));  // silu(-1) ≈ -0.2689
    float mid0 = s1 * 1.0f;      // 0.7311
    float mid1 = sm1 * (-1.0f);  // 0.2689
    // out[0] = mid0 + mid2 = 2*mid0
    // out[1] = mid1 + mid3 = 2*mid1
    CHECK_CLOSE(out[0], 2.0f * mid0, 1e-4f, "swiglu asymmetric [0]");
    CHECK_CLOSE(out[1], 2.0f * mid1, 1e-4f, "swiglu asymmetric [1]");
}

// ===========================================================================
// RoPEFreqTable — precomputed frequency table, same results as apply_rope()
// ===========================================================================
static void test_rope_freq_table_identity() {
    printf("test_rope_freq_table_identity...\n");

    // position=0 → all angles = 0 → cos=1, sin=0 → identity for any style
    RoPEFreqTable ft;
    ft.precompute(4, 10000.0f);
    CHECK(ft.freqs.size() == 2, "freq_table: freqs size == head_dim/2");
    CHECK(ft.head_dim == 4, "freq_table: head_dim stored");

    float qk[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ft.apply(qk, 1, 0, RoPEStyle::NEGHALF);

    for (int i = 0; i < 4; i++)
        CHECK_CLOSE(qk[i], (float)(i + 1), 1e-6f, "freq_table: identity at pos=0");
}

static void test_rope_freq_table_matches_apply_rope_neghalf() {
    printf("test_rope_freq_table_matches_apply_rope_neghalf...\n");

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    RoPEFreqTable ft;
    ft.precompute(4, 10000.0f);
    ft.apply(a, 2, 7, RoPEStyle::NEGHALF);
    apply_rope(b, 2, 4, 7, 10000.0f, RoPEStyle::NEGHALF);

    for (int i = 0; i < 8; i++)
        CHECK_CLOSE(a[i], b[i], 1e-5f, "freq_table NEGHALF matches apply_rope");
}

static void test_rope_freq_table_matches_apply_rope_interleaved() {
    printf("test_rope_freq_table_matches_apply_rope_interleaved...\n");

    float a[] = {1.5f, -0.5f, 2.0f, 3.0f, -1.0f, 4.0f, 0.5f, -2.0f};
    float b[] = {1.5f, -0.5f, 2.0f, 3.0f, -1.0f, 4.0f, 0.5f, -2.0f};

    RoPEFreqTable ft;
    ft.precompute(4, 500.0f);
    ft.apply(a, 2, 13, RoPEStyle::INTERLEAVED);
    apply_rope(b, 2, 4, 13, 500.0f, RoPEStyle::INTERLEAVED);

    for (int i = 0; i < 8; i++)
        CHECK_CLOSE(a[i], b[i], 1e-5f, "freq_table INTERLEAVED matches apply_rope");
}

static void test_rope_freq_table_reuse() {
    printf("test_rope_freq_table_reuse...\n");

    // Same table used at different positions must produce same result as apply_rope
    RoPEFreqTable ft;
    ft.precompute(4, 10000.0f);

    for (int pos = 0; pos < 5; pos++) {
        float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
        ft.apply(a, 1, pos, RoPEStyle::INTERLEAVED);
        apply_rope(b, 1, 4, pos, 10000.0f, RoPEStyle::INTERLEAVED);
        for (int i = 0; i < 4; i++)
            CHECK_CLOSE(a[i], b[i], 1e-5f, "freq_table reuse matches apply_rope");
    }
}

// ===========================================================================
// Integration-style: RoPE + GQA attention together
// ===========================================================================
static void test_rope_then_attn() {
    printf("test_rope_then_attn...\n");

    // Simulate a mini decode step: apply RoPE to Q and K, then attention
    int n_heads = 1, n_kv = 1, d = 4, max_seq = 8, n_layers = 1;
    auto cache = alloc_kv_cache(n_layers, max_seq, n_kv, d);

    float theta = 10000.0f;

    // Position 0
    float q0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k0[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    apply_rope(q0, 1, d, 0, theta, RoPEStyle::NEGHALF);
    apply_rope(k0, 1, d, 0, theta, RoPEStyle::NEGHALF);
    float out[4];
    gqa_attn_step(q0, k0, v0, cache.data(), n_heads, n_kv, d, max_seq, 0, 0, n_layers, out);

    // Single position → output = V0
    CHECK_CLOSE(out[0], 1.0f, 1e-5f, "rope+attn pos0 [0]");
    CHECK_CLOSE(out[3], 4.0f, 1e-5f, "rope+attn pos0 [3]");

    // Position 1 — Q and K start identical, but RoPE rotates differently at pos 1
    float q1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k1[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v1[] = {5.0f, 6.0f, 7.0f, 8.0f};
    apply_rope(q1, 1, d, 1, theta, RoPEStyle::NEGHALF);
    apply_rope(k1, 1, d, 1, theta, RoPEStyle::NEGHALF);
    gqa_attn_step(q1, k1, v1, cache.data(), n_heads, n_kv, d, max_seq, 1, 0, n_layers, out);

    // Output should be a weighted blend of V0 and V1
    // K0 was at pos 0, K1 at pos 1 — Q1 at pos 1 aligns more with K1
    bool no_nan = true;
    for (int i = 0; i < 4; i++)
        if (std::isnan(out[i])) no_nan = false;
    CHECK(no_nan, "rope+attn pos1 no NaN");
    // V1[0]=5 should dominate since Q1 aligns better with K1 (same position)
    CHECK(out[0] > 3.0f, "rope+attn pos1 V1 dominates");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    printf("=== core_vlm unit tests ===\n\n");

    // RoPE NEGHALF
    test_rope_neghalf_identity();
    test_rope_neghalf_known_angle();
    test_rope_neghalf_multihead();
    test_rope_neghalf_large_position();

    // RoPE INTERLEAVED
    test_rope_interleaved_identity();
    test_rope_interleaved_known_angle();
    test_rope_interleaved_multihead();
    test_rope_interleaved_different_freqs();

    // RoPE cross-style
    test_rope_styles_identity_at_zero();

    // KV cache
    test_kv_cache_alloc();
    test_kv_offsets();
    test_kv_offset_write_read_roundtrip();

    // GQA attention
    test_gqa_attn_single_head();
    test_gqa_attn_two_positions();
    test_gqa_attn_multihead();
    test_gqa_attn_gqa_repeat();
    test_gqa_attn_multilayer();
    test_gqa_attn_uniform_values();

    // SwiGLU
    test_swiglu_ffn_basic();
    test_swiglu_ffn_zero_input();
    test_swiglu_ffn_silu_gating();
    test_swiglu_ffn_asymmetric();

    // RoPEFreqTable
    test_rope_freq_table_identity();
    test_rope_freq_table_matches_apply_rope_neghalf();
    test_rope_freq_table_matches_apply_rope_interleaved();
    test_rope_freq_table_reuse();

    // Integration
    test_rope_then_attn();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
