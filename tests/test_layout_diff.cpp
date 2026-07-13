// test_layout_diff.cpp — per-stage parity comparison for layout detection.
// Usage: test-layout-diff model.gguf ref.gguf [image.png]
//
// Runs the full layout detection pipeline and compares intermediate
// tensors against the reference GGUF at each stage using crispembed_diff.

#include "layout_detect.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

// Permutation-tolerant per-query cosine. Data is query-major (flat = q*qdim + d);
// nq queries of length qdim. For each reference query, find its best-cosine match
// among all C++ queries and report the min / mean over those best matches. This is
// invariant to the top-K query reordering (see StageFile comment) but still trips
// on a genuine scramble (which leaves no good match for any query).
static void perm_tolerant_cos(const float * cpp, const float * ref, size_t nq, size_t qdim, float & cos_min,
                              float & cos_mean) {
    std::vector<double> cn(nq), rn(nq);
    for (size_t q = 0; q < nq; q++) {
        double sc = 0, sr = 0;
        const float * a = cpp + q * qdim;
        const float * b = ref + q * qdim;
        for (size_t d = 0; d < qdim; d++) {
            sc += (double)a[d] * a[d];
            sr += (double)b[d] * b[d];
        }
        cn[q] = std::sqrt(sc);
        rn[q] = std::sqrt(sr);
    }
    double sum = 0;
    float worst = 2.0f;
    for (size_t j = 0; j < nq; j++) {
        const float * b = ref + j * qdim;
        float best = -2.0f;
        for (size_t i = 0; i < nq; i++) {
            const float * a = cpp + i * qdim;
            double dot = 0;
            for (size_t d = 0; d < qdim; d++) dot += (double)a[d] * b[d];
            float c = (cn[i] > 1e-9 && rn[j] > 1e-9) ? (float)(dot / (cn[i] * rn[j])) : 0.0f;
            if (c > best) best = c;
        }
        sum += best;
        if (best < worst) worst = best;
    }
    cos_min = worst;
    cos_mean = (float)(sum / (double)nq);
}

static int crispembed_test_main(int argc, char ** argv) {
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
    // so they hold at 0.99.
    //
    // dec_0_cross_out is compared PERMUTATION-TOLERANTLY (perm_tolerant=true).
    // The 300 decoder queries are picked by a partial_sort over ~8400 near-tie
    // encoder proposals (layout_detect.cpp:1318). A tiny backend FP difference in
    // enc_output (Metal/CUDA vs the CPU/Python reference — max_abs ~0.02, cos
    // 0.99999) reshuffles the near-tie ranks, so "query i" in our output is a
    // *different physical proposal* than "query i" in the reference. The cross_out
    // VALUES are correct — index-aligned cos craters (mean ~0.79, min negative on
    // Metal) purely from this reordering; the final boxes are unaffected (they go
    // through score-sort + NMS). Matching each reference query to its best-cosine
    // partner recovers the true parity (cos_mean ~0.999, min ~0.95 = the one
    // deformable-sampling boundary query). A real decoder scramble destroys the
    // whole SET → no vector matches any → best-match collapses toward 0, tripping
    // the 0.85 gate. The encoder-scramble class is additionally caught directly by
    // the strict 0.99 s3..enc_output stages above.
    struct StageFile {
        const char * ref_name;
        const char * cpp_file;
        float threshold;
        bool perm_tolerant;
    };
    StageFile stages[] = {
        { "ip3", "/tmp/cpp_ip3.bin", 0.99f, false },
        { "ip4", "/tmp/cpp_ip4.bin", 0.99f, false },
        { "ip5", "/tmp/cpp_ip5.bin", 0.99f, false },
        { "s3", "/tmp/cpp_s3.bin", 0.99f, false },
        { "s4", "/tmp/cpp_s4.bin", 0.99f, false },
        { "s5", "/tmp/cpp_s5.bin", 0.99f, false },
        { "enc_output", "/tmp/cpp_enc_output.bin", 0.99f, false },
        { "dec_0_cross_out", "/tmp/cpp_cross_out.bin", 0.85f, true },
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
        if (st.perm_tolerant) {
            // Query axis = last shape dim (ggml ne1); query vector length = ne0.
            auto [ref_data2, ref_n2] = ref.get_f32(st.ref_name);
            (void)ref_n2;
            size_t nq = r.shape.empty() ? 1 : (size_t)r.shape.back();
            size_t qdim = (nq > 0) ? ref_n / nq : ref_n;
            if (nq > 0 && qdim > 0) perm_tolerant_cos(cpp_data.data(), ref_data2, nq, qdim, r.cos_min, r.cos_mean);
        }
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

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
