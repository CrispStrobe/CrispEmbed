// tests/test_tps_parity.cpp — parity test: C++ TPS localization vs Python reference.
//
// Loads the GGUF model and reference activations, runs the same forward pass,
// compares per-stage with crispembed_diff. Requires:
//   1. /mnt/storage/gguf-models/tps-loc-f32.gguf (model weights)
//   2. /tmp/tps-ref.gguf (reference from tools/dump_tps_reference.py)
//
// Usage:
//   ./test-tps-parity /mnt/storage/gguf-models/tps-loc-f32.gguf /tmp/tps-ref.gguf

#include "tps_warp.h"
#include "crispembed_diff.h"
#include <chrono>
#include <cmath>
#include <cstdio>
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

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path = argv[2];

    printf("TPS Localization — parity test\n");
    printf("  Model: %s\n", model_path);
    printf("  Ref:   %s\n", ref_path);

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "Failed to load reference GGUF\n");
        return 1;
    }
    printf("  Reference loaded\n\n");

    // Load model
    tps_locnet * net = tps_locnet_load(model_path);
    check("model loads", net != nullptr);
    if (!net) return 1;

    int F = tps_locnet_num_fiducial(net);
    printf("  num_fiducial: %d\n", F);

    // Create the same synthetic test image as the Python reference
    const int W = 200, H = 64;
    std::vector<uint8_t> gray(W * H, 230);
    for (int line = 0; line < 3; line++) {
        int base_y = 12 + line * 18;
        for (int x = 10; x < W - 10; x++) {
            int curve = (int)(4.0f * sinf(3.14159f * x / W));
            for (int dy = 0; dy < 5; dy++) {
                int y = base_y + curve + dy;
                if (y >= 0 && y < H) gray[y * W + x] = 30;
            }
        }
    }

    // Run C++ prediction
    std::vector<float> px(F), py(F);
    auto t0 = std::chrono::high_resolution_clock::now();
    int n = tps_locnet_predict(net, gray.data(), W, H, px.data(), py.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Predicted %d points in %.1f ms\n\n", n, ms);
    check("predict returns correct count", n == F);

    // Compare final control points against reference
    printf("=== Control point comparison (pixel space) ===\n");
    {
        // Build C++ points_pixel vector: [x0, y0, x1, y1, ...]
        std::vector<float> cpp_pts(F * 2);
        for (int i = 0; i < F; i++) {
            cpp_pts[i * 2 + 0] = px[i];
            cpp_pts[i * 2 + 1] = py[i];
        }

        auto r = ref.compare("points_pixel", cpp_pts.data(), F * 2);
        printf("  points_pixel: cos=%.6f max_abs=%.4f  %s\n",
               r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
        check("points_pixel cos >= 0.999", r.is_pass(0.999f));

        // Also report per-point max error
        float max_err = 0;
        for (int i = 0; i < F * 2; i++) {
            float ref_val = 0;
            // Read reference value
            // (we already have the cosine — just report max_abs)
            if (r.max_abs > max_err) max_err = r.max_abs;
        }
        printf("  Max absolute point error: %.4f pixels\n", r.max_abs);
        check("max point error < 1.0 pixel", r.max_abs < 1.0f);
    }

    // Compare fc2_out (raw model output before pixel conversion)
    printf("\n=== Raw model output (fc2_out) ===\n");
    {
        // We need the raw fc2 output. Since tps_locnet_predict converts to pixel
        // space, we reverse the conversion to get raw coords.
        std::vector<float> raw_pts(F * 2);
        for (int i = 0; i < F; i++) {
            raw_pts[i * 2 + 0] = px[i] / (0.5f * (W - 1)) - 1.0f;
            raw_pts[i * 2 + 1] = py[i] / (0.5f * (H - 1)) - 1.0f;
        }
        auto r = ref.compare("fc2_out", raw_pts.data(), F * 2);
        printf("  fc2_out: cos=%.6f max_abs=%.6f  %s\n",
               r.cos_min, r.max_abs, r.is_pass(0.999f) ? "PASS" : "FAIL");
        check("fc2_out cos >= 0.999", r.is_pass(0.999f));
    }

    // Print predicted points for visual comparison
    printf("\n=== Predicted control points ===\n");
    printf("  C++ vs Python reference:\n");
    for (int i = 0; i < F && i < 5; i++) {
        printf("  [%2d] C++: (%.1f, %.1f)\n", i, px[i], py[i]);
    }
    if (F > 5) printf("  ... (%d more)\n", F - 5);

    // Run full auto-dewarp pipeline and time it
    printf("\n=== Auto-dewarp pipeline ===\n");
    std::vector<uint8_t> out(W * H, 0);
    auto t2 = std::chrono::high_resolution_clock::now();
    int ret = tps_auto_dewarp(gray.data(), W, H, model_path, out.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    printf("  tps_auto_dewarp: ret=%d, %.1f ms\n", ret, ms2);
    check("auto_dewarp succeeds", ret == 0);

    int diff_count = 0;
    for (int i = 0; i < W * H; i++)
        if (gray[i] != out[i]) diff_count++;
    printf("  Pixels changed: %d / %d (%.1f%%)\n",
           diff_count, W * H, 100.0f * diff_count / (W * H));
    check("warp modifies pixels", diff_count > 0);

    tps_locnet_free(net);

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
