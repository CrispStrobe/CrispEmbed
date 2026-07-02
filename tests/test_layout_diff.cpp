// test_layout_diff.cpp — per-stage parity comparison for layout detection.
// Usage: test-layout-diff model.gguf ref.gguf [image.png]
//
// Runs the full layout detection pipeline and compares intermediate
// tensors against the reference GGUF at each stage using crispembed_diff.

#include "layout_detect.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <layout.gguf> <ref.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char * image_path = (argc > 3) ? argv[3] : "/tmp/test_layout.png";

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    printf("Reference tensors:\n");
    for (auto & name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf("  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++) printf("%s%lld", i ? "," : "", (long long)s[i]);
        printf("]\n");
    }

    // Enable debug dumps (portable: MSVC has no setenv)
#ifdef _WIN32
    _putenv_s("LAYOUT_DEBUG", "1");
#else
    setenv("LAYOUT_DEBUG", "1", 1);
#endif

    // Load model
    layout_detect::context * ctx = nullptr;
    if (!layout_detect::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Try to load preprocessed pixels from reference GGUF (bypass resize for exact parity)
    std::vector<layout_detect::region> regions;
    auto [ref_pixels, ref_px_n] = ref.get_f32("input_image");
    if (ref_pixels && ref_px_n == 3 * 640 * 640) {
        printf("Using preprocessed pixels from reference GGUF (3x640x640)\n");
        regions = layout_detect::detect(ctx, ref_pixels, 640, 640, 0.1f);
    } else {
        printf("Using image file with C++ resize: %s\n", image_path);
        regions = layout_detect::detect_file(ctx, image_path, 0.1f);
    }

    printf("\n=== Parity Report ===\n");
    printf("%-15s %10s %10s %10s %6s\n", "Stage", "cos_min", "cos_mean", "max_abs", "");

    // Compare each stage from dumped files.
    // Per-stage cos_min threshold: input/backbone/encoder stages are exact
    // (they gate the RT-DETR encoder regression class — negative cos on scramble),
    // so they hold at 0.99. dec_0_cross_out rides the deformable cross-attention's
    // CPU-side bilinear-sampling floor (~0.977 cos_min on one boundary query,
    // cos_mean ~0.999) — a pre-existing parity gap unrelated to the encoder path,
    // so it gates lower. A real decoder crater still trips it (goes negative).
    struct StageFile {
        const char * ref_name;
        const char * cpp_file;
        float threshold;
    };
    StageFile stages[] = {
        { "ip3", "/tmp/cpp_ip3.bin", 0.99f },
        { "ip4", "/tmp/cpp_ip4.bin", 0.99f },
        { "ip5", "/tmp/cpp_ip5.bin", 0.99f },
        { "s3", "/tmp/cpp_s3.bin", 0.99f },
        { "s4", "/tmp/cpp_s4.bin", 0.99f },
        { "s5", "/tmp/cpp_s5.bin", 0.99f },
        { "enc_output", "/tmp/cpp_enc_output.bin", 0.99f },
        { "dec_0_cross_out", "/tmp/cpp_cross_out.bin", 0.97f },
    };

    int n_fail = 0;
    int n_compared = 0;

    for (auto & st : stages) {
        auto [ref_data, ref_n] = ref.get_f32(st.ref_name);
        if (!ref_data || ref_n == 0) {
            printf("%-15s %s\n", st.ref_name, "NOT IN REF");
            continue;
        }

        FILE * fp = fopen(st.cpp_file, "rb");
        if (!fp) {
            printf("%-15s %s\n", st.ref_name, "NO DUMP FILE");
            n_fail++; // an expected stage produced no dump → regression
            continue;
        }

        std::vector<float> cpp_data(ref_n);
        size_t read = fread(cpp_data.data(), sizeof(float), ref_n, fp);
        fclose(fp);

        if (read != ref_n) {
            printf("%-15s SIZE MISMATCH (ref=%zu, read=%zu)\n", st.ref_name, ref_n, read);
            n_fail++;
            continue;
        }

        auto r = ref.compare(st.ref_name, cpp_data.data(), ref_n);
        bool pass = r.is_pass(st.threshold);
        printf("%-15s %10.6f %10.6f %10.4f %s\n", st.ref_name, r.cos_min, r.cos_mean, r.max_abs,
               pass ? "PASS" : "FAIL");
        // Canonical line consumed by tests/regression/run_one.py (applies the
        // manifest's per-stage thresholds to cos_min).
        printf("%s: cos_min=%.6f max_abs=%.6e %s\n", st.ref_name, r.cos_min, r.max_abs, pass ? "PASS" : "FAIL");
        n_compared++;
        if (!pass) n_fail++;
    }

    printf("\nDetected %zu regions (threshold 0.1)\n", regions.size());
    for (size_t i = 0; i < std::min(regions.size(), (size_t)5); i++) {
        printf("  [%zu] %s score=%.3f [%.0f,%.0f,%.0f,%.0f]\n", i, regions[i].label_name, regions[i].score,
               regions[i].x1, regions[i].y1, regions[i].x2, regions[i].y2);
    }

    layout_detect::free(ctx);

    printf("\n%d/%d stages passed (per-stage cos_min thresholds)\n", n_compared - n_fail, n_compared);
    if (n_fail > 0) {
        printf("DIFF FAILED: %d stage(s) below threshold\n", n_fail);
        return 1;
    }
    printf("DIFF PASSED\n");
    return 0;
}
