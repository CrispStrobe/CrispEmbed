// tests/test_litemla_attn.cpp — Unit test for the LiteMLA linear attention formula.
//
// Verifies that the ggml graph implementation of linear attention produces the
// same result as the scalar reference. Does NOT need a GGUF model file.
//
// LiteMLA computes:
//   KTV[d, dv, g]   = sum_hw K[hw, d, g] * V[hw, dv, g]   (K^T @ V per group)
//   K_sum[d, g]     = sum_hw K[hw, d, g]                   (denominator piece)
//   out[dv, hw, g]  = sum_d KTV[d, dv, g] * Q[hw, d, g]   (Q @ KTV per group)
//   norm[hw, g]     = sum_d K_sum[d, g] * Q[hw, d, g]      (Q · K_sum per group)
//   result[dv,hw,g] = out[dv, hw, g] / max(norm[hw,g], eps)
//
// Usage: ./build/test-litemla-attn
// Exit 0 = all pass.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        g_fail++; \
    } else { g_pass++; } \
} while (0)

#define CHECK_CLOSE(a, b, tol, msg) CHECK(fabsf((a) - (b)) < (tol), msg)

// ---------------------------------------------------------------------------
// Scalar reference: linear attention for one group.
// Q, K, V: each [HW, head_dim] (row-major).
// Returns out: [HW, head_dim].
// ---------------------------------------------------------------------------
static std::vector<float> linear_attn_scalar(
    const float* Q, const float* K, const float* V,
    int HW, int D, float eps = 1e-5f)
{
    // KTV[d, dv] = sum_hw K[hw,d] * V[hw,dv]
    std::vector<float> KTV(D * D, 0.0f);
    for (int d = 0; d < D; d++)
        for (int dv = 0; dv < D; dv++)
            for (int hw = 0; hw < HW; hw++)
                KTV[d * D + dv] += K[hw * D + d] * V[hw * D + dv];

    // K_sum[d] = sum_hw K[hw, d]
    std::vector<float> K_sum(D, 0.0f);
    for (int hw = 0; hw < HW; hw++)
        for (int d = 0; d < D; d++)
            K_sum[d] += K[hw * D + d];

    // out[hw, dv] = sum_d Q[hw,d] * KTV[d,dv]
    // norm[hw]    = sum_d Q[hw,d] * K_sum[d]
    std::vector<float> out(HW * D, 0.0f);
    for (int hw = 0; hw < HW; hw++) {
        float norm = 0.0f;
        for (int d = 0; d < D; d++)
            norm += Q[hw * D + d] * K_sum[d];
        norm = std::max(norm, eps);

        for (int dv = 0; dv < D; dv++) {
            float val = 0.0f;
            for (int d = 0; d < D; d++)
                val += Q[hw * D + d] * KTV[d * D + dv];
            out[hw * D + dv] = val / norm;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// ggml graph: replicate the same computation as a ggml graph.
// Inputs Q, K, V are [HW, D, n_groups] contiguous F32 tensors.
// Returns result: [D, HW, n_groups] (transpose from scalar [HW, D, n_groups]).
// ---------------------------------------------------------------------------
static std::vector<float> linear_attn_ggml(
    const float* Qdata, const float* Kdata, const float* Vdata,
    int HW, int D, int n_groups)
{
    ggml_backend_t backend = ggml_backend_cpu_init();

    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);

    // Input tensors: [HW, D, n_groups] (ne[0]=HW, ne[1]=D, ne[2]=n_groups)
    ggml_tensor* Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HW, D, n_groups);
    ggml_tensor* K = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HW, D, n_groups);
    ggml_tensor* V = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HW, D, n_groups);
    ggml_set_name(Q, "Q"); ggml_set_input(Q);
    ggml_set_name(K, "K"); ggml_set_input(K);
    ggml_set_name(V, "V"); ggml_set_input(V);

    // KTV = ggml_mul_mat(K, V): K=[HW=K, D=M, n_g], V=[HW=K, D=N, n_g] → [D, D, n_g]
    ggml_tensor* KTV = ggml_mul_mat(ctx, K, V);

    // K_sum = ggml_sum_rows(K): sums ne[0]=HW → [1, D, n_g]
    ggml_tensor* K_sum = ggml_sum_rows(ctx, K);

    // Q_T = permute Q [HW, D, n_g] → [D, HW, n_g] (swap ne[0] and ne[1])
    ggml_tensor* Q_T = ggml_cont(ctx, ggml_permute(ctx, Q, 1, 0, 2, 3));

    // out_unnorm = ggml_mul_mat(KTV, Q_T):
    //   KTV=[D=K, D=M, n_g], Q_T=[D=K, HW=N, n_g] → [D, HW, n_g]
    ggml_tensor* out_unnorm = ggml_mul_mat(ctx, KTV, Q_T);

    // K_sum_T = permute K_sum [1, D, n_g] → [D, 1, n_g]
    ggml_tensor* K_sum_T = ggml_cont(ctx, ggml_permute(ctx, K_sum, 1, 0, 2, 3));

    // norm = ggml_mul_mat(K_sum_T, Q_T):
    //   K_sum_T=[D=K, 1=M, n_g], Q_T=[D=K, HW=N, n_g] → [1, HW, n_g]
    ggml_tensor* norm = ggml_mul_mat(ctx, K_sum_T, Q_T);

    // Clamp norm, broadcast-divide
    ggml_tensor* norm_c   = ggml_clamp(ctx, norm, 1e-5f, 1e30f);
    ggml_tensor* norm_rep = ggml_repeat(ctx, norm_c, out_unnorm);
    ggml_tensor* result   = ggml_div(ctx, out_unnorm, norm_rep);
    ggml_set_name(result, "result");
    ggml_set_output(result);

    // Build and run
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    // Set inputs
    ggml_tensor* Q_t = ggml_graph_get_tensor(gf, "Q");
    ggml_tensor* K_t = ggml_graph_get_tensor(gf, "K");
    ggml_tensor* V_t = ggml_graph_get_tensor(gf, "V");
    size_t nb = (size_t)(HW * D * n_groups) * sizeof(float);
    ggml_backend_tensor_set(Q_t, Qdata, 0, nb);
    ggml_backend_tensor_set(K_t, Kdata, 0, nb);
    ggml_backend_tensor_set(V_t, Vdata, 0, nb);

    ggml_backend_graph_compute(backend, gf);

    // Read result [D, HW, n_groups]
    ggml_tensor* res_t = ggml_graph_get_tensor(gf, "result");
    std::vector<float> out(D * HW * n_groups);
    ggml_backend_tensor_get(res_t, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_linear_attn_tiny() {
    printf("test_linear_attn_tiny...\n");

    // Tiny: HW=3, D=2, n_groups=2. All values non-negative (ReLU-like inputs).
    int HW = 3, D = 2, n_groups = 2;

    // Per-group [HW, D] data stored as [g][hw][d]
    float Q_raw[2][3][2] = {
        {{0.5f, 0.3f}, {0.8f, 0.1f}, {0.2f, 0.9f}},
        {{0.4f, 0.6f}, {0.7f, 0.2f}, {0.1f, 0.5f}},
    };
    float K_raw[2][3][2] = {
        {{0.6f, 0.2f}, {0.3f, 0.7f}, {0.1f, 0.4f}},
        {{0.5f, 0.3f}, {0.2f, 0.8f}, {0.6f, 0.1f}},
    };
    float V_raw[2][3][2] = {
        {{0.1f, 0.9f}, {0.4f, 0.6f}, {0.7f, 0.3f}},
        {{0.2f, 0.8f}, {0.5f, 0.5f}, {0.3f, 0.7f}},
    };

    // Compute scalar reference for each group: out_scalar[g][hw][dv]
    float sc[2][3][2] = {};
    for (int g = 0; g < n_groups; g++) {
        auto r = linear_attn_scalar(&Q_raw[g][0][0], &K_raw[g][0][0], &V_raw[g][0][0], HW, D);
        for (int hw = 0; hw < HW; hw++)
            for (int dv = 0; dv < D; dv++)
                sc[g][hw][dv] = r[hw * D + dv];
    }

    // Build ggml input as [HW, D, n_groups]: element (hw, d, g) = hw + HW*d + HW*D*g
    std::vector<float> Qg(HW*D*n_groups), Kg(HW*D*n_groups), Vg(HW*D*n_groups);
    for (int g = 0; g < n_groups; g++)
        for (int hw = 0; hw < HW; hw++)
            for (int d = 0; d < D; d++) {
                int dst = hw + HW * d + HW * D * g;
                Qg[dst] = Q_raw[g][hw][d];
                Kg[dst] = K_raw[g][hw][d];
                Vg[dst] = V_raw[g][hw][d];
            }

    // ggml result: [D, HW, n_groups], element (dv, hw, g) = dv + D*hw + D*HW*g
    auto gg = linear_attn_ggml(Qg.data(), Kg.data(), Vg.data(), HW, D, n_groups);

    float max_diff = 0.0f;
    for (int g = 0; g < n_groups; g++)
        for (int hw = 0; hw < HW; hw++)
            for (int dv = 0; dv < D; dv++) {
                float diff = fabsf(sc[g][hw][dv] - gg[dv + D * hw + D * HW * g]);
                if (diff > max_diff) max_diff = diff;
            }

    printf("  max_diff scalar vs ggml: %.2e\n", max_diff);
    CHECK(max_diff < 1e-5f, "linear_attn_tiny: ggml matches scalar (tol=1e-5)");
}

static void test_linear_attn_single_pixel() {
    printf("test_linear_attn_single_pixel...\n");

    // Degenerate: HW=1. Linear attention collapses to trivial case:
    // out = Q * (K * V) / (Q * K) = V (assuming same Q/K/V weight sums to 1)
    // More precisely: KTV = K[0]^T * V[0] (outer product), norm = Q[0] dot K[0]
    // out = (Q[0] dot KTV) / norm
    int HW = 1, D = 2, n_groups = 1;
    float Q1[2] = {1.0f, 0.0f};  // e_0
    float K1[2] = {1.0f, 0.0f};  // e_0
    float V1[2] = {3.0f, 7.0f};

    auto sc = linear_attn_scalar(Q1, K1, V1, HW, D);
    // KTV[d, dv] = K[0,d] * V[0,dv]:
    //   d=0,dv=0: 1*3=3; d=0,dv=1: 1*7=7; d=1,dv=0: 0; d=1,dv=1: 0
    // K_sum = K[0] = [1, 0]
    // norm = Q[0] dot K_sum = 1*1 + 0*0 = 1
    // out[0, dv] = Q[0] dot KTV[:, dv] / 1 = KTV[0, dv] = V[0, dv]
    CHECK_CLOSE(sc[0], 3.0f, 1e-6f, "single_pixel: out[0] == V[0]");
    CHECK_CLOSE(sc[1], 7.0f, 1e-6f, "single_pixel: out[1] == V[1]");

    // Also verify ggml path
    std::vector<float> Qg(2), Kg(2), Vg(2);
    Qg[0] = Q1[0]; Qg[1] = Q1[1];
    Kg[0] = K1[0]; Kg[1] = K1[1];
    Vg[0] = V1[0]; Vg[1] = V1[1];
    auto gg = linear_attn_ggml(Qg.data(), Kg.data(), Vg.data(), HW, D, n_groups);
    // ggml result [D, HW, n_groups] = [2, 1, 1], element (dv, hw=0, g=0) = dv
    CHECK_CLOSE(gg[0], 3.0f, 1e-5f, "single_pixel ggml: out[0] == V[0]");
    CHECK_CLOSE(gg[1], 7.0f, 1e-5f, "single_pixel ggml: out[1] == V[1]");
}

static void test_linear_attn_uniform() {
    printf("test_linear_attn_uniform...\n");

    // Uniform K: all K[hw, d] = 1/(HW*D) → K_sum[d] = 1/D for all d.
    // Uniform Q: Q[hw, d] = 1. norm[hw] = D * (1/D) = 1.
    // KTV[d, dv] = sum_hw K[hw,d] * V[hw,dv] = (1/(HW*D)) * sum_hw V[hw,dv]
    // out[hw, dv] = sum_d 1 * KTV[d, dv] = D * KTV[0, dv]
    //             = D * (1/(HW*D)) * sum_hw V[hw,dv] = mean(V[:, dv])
    // So with uniform Q and K, output = mean of V for each feature.

    int HW = 4, D = 2, n_groups = 1;
    std::vector<float> Q(HW * D), K(HW * D), V(HW * D);
    float kval = 1.0f / (HW * D);
    for (int hw = 0; hw < HW; hw++)
        for (int d = 0; d < D; d++) {
            Q[hw * D + d] = 1.0f;
            K[hw * D + d] = kval;
            V[hw * D + d] = (float)(hw * D + d);  // distinct values
        }

    auto sc = linear_attn_scalar(Q.data(), K.data(), V.data(), HW, D);

    // Expected: out[hw, dv] = mean(V[:, dv])
    for (int dv = 0; dv < D; dv++) {
        float mean_v = 0.0f;
        for (int hw = 0; hw < HW; hw++) mean_v += V[hw * D + dv];
        mean_v /= HW;
        for (int hw = 0; hw < HW; hw++) {
            float sc_val = sc[hw * D + dv];
            CHECK_CLOSE(sc_val, mean_v, 1e-5f, "uniform: out[hw,dv] == mean(V[:,dv])");
        }
    }
}

int main() {
    printf("=== LiteMLA Linear Attention Unit Tests ===\n\n");
    test_linear_attn_single_pixel();
    test_linear_attn_uniform();
    test_linear_attn_tiny();
    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
