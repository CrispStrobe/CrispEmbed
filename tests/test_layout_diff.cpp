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

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <layout.gguf> <ref.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char* image_path = (argc > 3) ? argv[3] : "/tmp/test_layout.png";

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    printf("Reference tensors:\n");
    for (auto& name : ref.tensor_names()) {
        auto s = ref.shape(name);
        printf("  %s [", name.c_str());
        for (size_t i = 0; i < s.size(); i++) printf("%s%lld", i?",":"", (long long)s[i]);
        printf("]\n");
    }

    // Enable debug dumps (portable: MSVC has no setenv)
#ifdef _WIN32
    _putenv_s("LAYOUT_DEBUG", "1");
#else
    setenv("LAYOUT_DEBUG", "1", 1);
#endif

    // Load model and run
    layout_detect::context* ctx = nullptr;
    if (!layout_detect::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    auto regions = layout_detect::detect_file(ctx, image_path, 0.1f);

    printf("\n=== Parity Report ===\n");
    printf("%-15s %10s %10s %10s %6s\n", "Stage", "cos_min", "cos_mean", "max_abs", "");

    // Compare each stage from dumped files
    struct StageFile { const char* ref_name; const char* cpp_file; };
    StageFile stages[] = {
        {"ip3", "/tmp/cpp_ip3.bin"},
        {"ip4", "/tmp/cpp_ip4.bin"},
        {"ip5", "/tmp/cpp_ip5.bin"},
        {"s3",  "/tmp/cpp_s3.bin"},
        {"s4",  "/tmp/cpp_s4.bin"},
        {"s5",  "/tmp/cpp_s5.bin"},
        {"enc_output", "/tmp/cpp_enc_output.bin"},
    };

    for (auto& st : stages) {
        auto [ref_data, ref_n] = ref.get_f32(st.ref_name);
        if (!ref_data || ref_n == 0) {
            printf("%-15s %s\n", st.ref_name, "NOT IN REF");
            continue;
        }

        FILE* fp = fopen(st.cpp_file, "rb");
        if (!fp) {
            printf("%-15s %s\n", st.ref_name, "NO DUMP FILE");
            continue;
        }

        std::vector<float> cpp_data(ref_n);
        size_t read = fread(cpp_data.data(), sizeof(float), ref_n, fp);
        fclose(fp);

        if (read != ref_n) {
            printf("%-15s SIZE MISMATCH (ref=%zu, read=%zu)\n", st.ref_name, ref_n, read);
            continue;
        }

        auto r = ref.compare(st.ref_name, cpp_data.data(), ref_n);
        printf("%-15s %10.6f %10.6f %10.4f %s\n",
               st.ref_name, r.cos_min, r.cos_mean, r.max_abs,
               r.is_pass(0.99f) ? "PASS" : "FAIL");
    }

    printf("\nDetected %zu regions (threshold 0.1)\n", regions.size());
    for (size_t i = 0; i < std::min(regions.size(), (size_t)5); i++) {
        printf("  [%zu] %s score=%.3f [%.0f,%.0f,%.0f,%.0f]\n",
               i, regions[i].label_name, regions[i].score,
               regions[i].x1, regions[i].y1, regions[i].x2, regions[i].y2);
    }

    layout_detect::free(ctx);
    return 0;
}
