// tests/test_surya_det_diff.cpp — parity diff between C++ surya detector
// and Python reference activations.
//
// Usage:
//   # 1. Generate reference activations from Python:
//   python tools/dump_surya_det_reference.py --image test.png --output ref.gguf
//
//   # 2. Run parity test (feeds reference input through C++ engine):
//   SURYA_DET_DUMP=1 ./test-surya-det-diff surya-det-f32.gguf ref.gguf

#include "surya_det.h"
#include "crispembed_diff.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <surya-det.gguf> <reference.gguf>\n", argv[0]);
        fprintf(stderr, "\nGenerate reference with:\n");
        fprintf(stderr, "  python tools/dump_surya_det_reference.py --image test.png --output ref.gguf\n");
        return 1;
    }

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;
    printf("Reference: %zu tensors\n", ref.tensor_names().size());

    // Get the preprocessed input from reference
    auto [input_data, input_n] = ref.get_f32("input_image");
    if (!input_data || input_n == 0) {
        fprintf(stderr, "Reference has no 'input_image' tensor\n");
        return 1;
    }
    auto input_shape = ref.shape("input_image");
    // GGUF stores the raw bytes contiguously. The Python reference dumper
    // writes numpy arrays in C-contiguous (CHW) order. The GGUF reader
    // preserves this byte order. So input_data is already [C, H, W].
    int C = 3, H = 1200, W = 1200;
    if (input_n != (size_t)(C * H * W)) {
        fprintf(stderr, "Unexpected input size: %zu (expected %d)\n",
                input_n, C * H * W);
        return 1;
    }
    printf("Input: [%d, %d, %d] (%zu floats)\n", C, H, W, input_n);

    // Verify input matches reference stats
    {
        float mn = input_data[0], mx = input_data[0];
        double sum = 0;
        for (size_t i = 0; i < input_n; i++) {
            if (input_data[i] < mn) mn = input_data[i];
            if (input_data[i] > mx) mx = input_data[i];
            sum += input_data[i];
        }
        printf("  Input stats: min=%.4f max=%.4f mean=%.4f\n", mn, mx, (float)(sum/input_n));
        // Channel 0 starts at offset 0, should be ~2.249 for white background
        printf("  First values: [%.4f, %.4f, %.4f] (expect ~2.25, ~2.43, ~2.64 for white)\n",
               input_data[0], input_data[H*W], input_data[2*H*W]);
    }

    // Load model
    surya_det_context* ctx = surya_det_init(argv[1], 4);
    if (!ctx) return 1;

    // Run forward pass with reference input (bypasses preprocessing)
    printf("\nRunning C++ forward pass with reference input...\n");
    printf("(This takes ~15 min with CPU-scalar. Set SURYA_DET_DUMP=1 for progress.)\n\n");

    int out_h = 0, out_w = 0;
    const float* heatmap = surya_det_detect_raw(ctx, input_data, H, W, &out_h, &out_w);
    if (!heatmap) {
        fprintf(stderr, "Forward pass failed\n");
        surya_det_free(ctx);
        return 1;
    }

    printf("C++ heatmap: [2, %d, %d]\n\n", out_h, out_w);

    // Compare heatmap against reference
    printf("=== Heatmap parity ===\n");
    auto r = ref.compare("heatmap", heatmap, 2 * out_h * out_w);
    if (r.found) {
        printf("  cos_min=%.6f max_abs=%.4e mean_abs=%.4e  %s\n",
               r.cos_min, r.max_abs, r.mean_abs,
               r.is_pass(0.99f) ? "PASS" : "FAIL");
    } else {
        printf("  (reference 'heatmap' not found)\n");
    }

    // Also compare logits
    auto [logits_ref, logits_n] = ref.get_f32("logits");
    if (logits_ref && logits_n > 0) {
        // Our heatmap is after sigmoid; logits is before sigmoid in reference
        // We can't directly compare — but check if reference logits match shape
        printf("\n  Reference logits: %zu elements\n", logits_n);
    }

    // Print channel stats for both
    printf("\n=== Channel comparison ===\n");
    auto [ref_hm, ref_hm_n] = ref.get_f32("heatmap");
    for (int c = 0; c < 2; c++) {
        float cpp_min = heatmap[c * out_h * out_w], cpp_max = cpp_min;
        double cpp_sum = 0;
        float ref_min = ref_hm ? ref_hm[c * out_h * out_w] : 0;
        float ref_max = ref_min;
        double ref_sum = 0;
        int n = out_h * out_w;

        for (int i = 0; i < n; i++) {
            float cv = heatmap[c * n + i];
            if (cv < cpp_min) cpp_min = cv;
            if (cv > cpp_max) cpp_max = cv;
            cpp_sum += cv;

            if (ref_hm) {
                float rv = ref_hm[c * n + i];
                if (rv < ref_min) ref_min = rv;
                if (rv > ref_max) ref_max = rv;
                ref_sum += rv;
            }
        }
        printf("  Ch%d C++: min=%.4f max=%.4f mean=%.4f\n",
               c, cpp_min, cpp_max, (float)(cpp_sum / n));
        if (ref_hm) {
            printf("  Ch%d Ref: min=%.4f max=%.4f mean=%.4f\n",
                   c, ref_min, ref_max, (float)(ref_sum / n));
        }
    }

    surya_det_free(ctx);
    printf("\nDone.\n");
    return 0;
}
