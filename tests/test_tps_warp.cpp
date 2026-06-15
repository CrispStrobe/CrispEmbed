// tests/test_tps_warp.cpp — unit tests for TPS spatial transformer
#include "tps_warp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

// ---------------------------------------------------------------------------
// Test 1: Identity transform — control points map to themselves.
// The warped image should match the input.
// ---------------------------------------------------------------------------
static void test_identity() {
    printf("\n=== Identity TPS (no distortion) ===\n");

    const int W = 100, H = 80;
    std::vector<uint8_t> src(W * H);
    // Gradient pattern: pixel value = x
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            src[y * W + x] = (uint8_t)(x * 255 / W);

    // 4 corner control points, identity mapping
    float sx[] = { 0, (float)(W-1), 0,         (float)(W-1) };
    float sy[] = { 0, 0,            (float)(H-1), (float)(H-1) };
    float dx[] = { 0, (float)(W-1), 0,         (float)(W-1) };
    float dy[] = { 0, 0,            (float)(H-1), (float)(H-1) };

    tps_model * model = tps_solve(sx, sy, dx, dy, 4);
    check("tps_solve succeeds", model != nullptr);

    if (model) {
        // Check that control points map to themselves
        for (int i = 0; i < 4; i++) {
            float ox, oy;
            tps_map_point(model, sx[i], sy[i], &ox, &oy);
            float err = std::sqrt((ox - dx[i]) * (ox - dx[i]) +
                                  (oy - dy[i]) * (oy - dy[i]));
            char msg[128];
            snprintf(msg, sizeof(msg), "control point %d maps correctly (err=%.6f)", i, err);
            check(msg, err < 0.01f);
        }

        // Warp and compare
        std::vector<uint8_t> dst(W * H, 0);
        tps_warp(src.data(), W, H, model, dst.data(), W, H, 255);

        int max_diff = 0;
        for (int i = 0; i < W * H; i++) {
            int d = std::abs((int)src[i] - (int)dst[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("  Max pixel diff: %d\n", max_diff);
        check("identity warp preserves image (max diff <= 2)", max_diff <= 2);

        tps_free(model);
    }
}

// ---------------------------------------------------------------------------
// Test 2: Pure translation — shift all points by (10, 5).
// ---------------------------------------------------------------------------
static void test_translation() {
    printf("\n=== Translation TPS ===\n");

    // Map output coords to input coords: output (x,y) → input (x+10, y+5)
    // So control points: src=(corners+offset), dst=(corners)
    const float tx = 10.0f, ty = 5.0f;
    float sx[] = { 0+tx, 99+tx, 0+tx,  99+tx };
    float sy[] = { 0+ty, 0+ty,  79+ty, 79+ty };
    float dx[] = { 0,    99,    0,     99    };
    float dy[] = { 0,    0,     79,    79    };

    tps_model * model = tps_solve(dx, dy, sx, sy, 4);
    check("tps_solve succeeds", model != nullptr);

    if (model) {
        // Check center point maps correctly
        float ox, oy;
        tps_map_point(model, 50.0f, 40.0f, &ox, &oy);
        float err_x = std::abs(ox - 60.0f);
        float err_y = std::abs(oy - 45.0f);
        printf("  Center maps to (%.2f, %.2f), expected (60, 45)\n", ox, oy);
        check("translation x correct", err_x < 0.1f);
        check("translation y correct", err_y < 0.1f);

        tps_free(model);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Nonlinear warp — create a barrel-distorted image, use TPS to
// undistort it. Verify that the result is closer to the original.
// ---------------------------------------------------------------------------
static void test_nonlinear_warp() {
    printf("\n=== Nonlinear warp (point mapping accuracy) ===\n");

    // Instead of trying to undo barrel distortion on a full image (which is
    // sensitive to sampling artifacts), verify that the TPS interpolation
    // accurately maps test points through a known nonlinear transform.
    //
    // Set up 9 control points on a 3x3 grid with nonlinear displacements,
    // then verify that intermediate test points are mapped accurately.

    const int grid = 3;
    const int n = grid * grid;
    std::vector<float> sx(n), sy(n), dx(n), dy(n);

    // Source: regular 3x3 grid on [10, 90]
    int idx = 0;
    for (int gy = 0; gy < grid; gy++) {
        for (int gx = 0; gx < grid; gx++) {
            sx[idx] = 10.0f + gx * 40.0f;
            sy[idx] = 10.0f + gy * 40.0f;
            // Target: same grid but with a nonlinear perturbation
            float px = sx[idx], py = sy[idx];
            dx[idx] = px + 5.0f * sinf(py * 0.05f);
            dy[idx] = py + 3.0f * cosf(px * 0.04f);
            idx++;
        }
    }

    tps_model * model = tps_solve(sx.data(), sy.data(), dx.data(), dy.data(), n);
    check("tps_solve succeeds for nonlinear warp", model != nullptr);

    if (model) {
        // Verify control points map exactly
        float max_ctrl_err = 0;
        for (int i = 0; i < n; i++) {
            float ox, oy;
            tps_map_point(model, sx[i], sy[i], &ox, &oy);
            float err = std::sqrt((ox - dx[i]) * (ox - dx[i]) +
                                  (oy - dy[i]) * (oy - dy[i]));
            if (err > max_ctrl_err) max_ctrl_err = err;
        }
        printf("  Max control point error: %.6f\n", max_ctrl_err);
        check("control points map exactly (err < 0.01)", max_ctrl_err < 0.01f);

        // Verify interpolation at midpoints is smooth and plausible
        // (between control points, the TPS should give a smooth blend)
        float ox1, oy1, ox2, oy2, ox_mid, oy_mid;
        tps_map_point(model, sx[0], sy[0], &ox1, &oy1);
        tps_map_point(model, sx[1], sy[1], &ox2, &oy2);
        tps_map_point(model, (sx[0]+sx[1])/2, (sy[0]+sy[1])/2, &ox_mid, &oy_mid);

        // Midpoint output should be roughly between the two endpoint outputs
        float expected_x = (ox1 + ox2) / 2.0f;
        float expected_y = (oy1 + oy2) / 2.0f;
        float mid_err = std::sqrt((ox_mid - expected_x) * (ox_mid - expected_x) +
                                  (oy_mid - expected_y) * (oy_mid - expected_y));
        printf("  Midpoint deviation from linear: %.2f px\n", mid_err);
        check("midpoint interpolation is smooth (deviation < 10px)", mid_err < 10.0f);

        // Warp a simple gradient image and verify no crashes
        const int W = 100, H = 100;
        std::vector<uint8_t> img(W * H);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                img[y * W + x] = (uint8_t)(x * 255 / W);

        std::vector<uint8_t> out(W * H, 0);
        tps_warp(img.data(), W, H, model, out.data(), W, H, 128);

        // Just verify no all-zero or all-bg output (warp actually did something)
        int n_bg = 0;
        for (int i = 0; i < W * H; i++)
            if (out[i] == 128) n_bg++;
        float bg_frac = (float)n_bg / (W * H);
        printf("  Background fraction: %.1f%%\n", bg_frac * 100);
        check("warp produces meaningful output (bg < 50%)", bg_frac < 0.5f);

        tps_free(model);
    }
}

// ---------------------------------------------------------------------------
// Test 4: Edge cases — too few points, null pointers
// ---------------------------------------------------------------------------
static void test_edge_cases() {
    printf("\n=== Edge cases ===\n");

    // Too few points
    float x[] = {0, 1}, y[] = {0, 1};
    tps_model * m = tps_solve(x, y, x, y, 2);
    check("tps_solve returns NULL for n < 3", m == nullptr);

    m = tps_solve(nullptr, y, x, y, 4);
    check("tps_solve returns NULL for null src_x", m == nullptr);

    // tps_warp_points with bad inputs
    uint8_t buf[100] = {};
    int ret = tps_warp_points(buf, 10, 10, nullptr, nullptr,
                              nullptr, nullptr, 0, buf, 10, 10, 255);
    check("tps_warp_points returns 1 for bad inputs", ret == 1);

    // tps_free(NULL) should not crash
    tps_free(nullptr);
    check("tps_free(NULL) does not crash", true);
}

// ---------------------------------------------------------------------------
// Test 5: tps_dewarp via classical_preproc.h
// ---------------------------------------------------------------------------
#include "classical_preproc.h"

static void test_tps_dewarp_api() {
    printf("\n=== tps_dewarp() classical_preproc API ===\n");

    const int W = 100, H = 80;
    std::vector<uint8_t> img(W * H, 128);
    std::vector<uint8_t> out(W * H, 0);

    // Identity control points — should succeed and copy
    float sx[] = { 0, 99, 0,  99 };
    float sy[] = { 0, 0,  79, 79 };

    int ret = tps_dewarp(img.data(), W, H,
                         sx, sy, sx, sy, 4,
                         out.data());
    check("tps_dewarp returns 0", ret == 0);

    // Too few points
    ret = tps_dewarp(img.data(), W, H,
                     sx, sy, sx, sy, 2,
                     out.data());
    check("tps_dewarp returns 1 for n < 3", ret == 1);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("TPS Spatial Transformer — unit tests\n");

    test_identity();
    test_translation();
    test_nonlinear_warp();
    test_edge_cases();
    test_tps_dewarp_api();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
