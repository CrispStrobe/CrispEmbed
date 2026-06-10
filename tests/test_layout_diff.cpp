// test_layout_diff.cpp — per-stage parity comparison for layout detection.
// Usage: test-layout-diff model.gguf ref.gguf [image.png]
//
// Compares C++ encoder intermediates against HF reference at each stage:
//   ip3/ip4/ip5: backbone output after encoder_input_proj
//   s3/s4/s5: encoder output after FPN/PAN
//   enc_output: decoder input after enc_proj + layernorm

#include "layout_detect.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

// The layout_detect namespace exposes detect_file but not intermediate tensors.
// We'll run detection and compare the outputs dumped via LAYOUT_DEBUG env var.
// For proper diff, we need to read the binary dumps.

static float cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <layout.gguf> <ref.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char* image_path = (argc > 3) ? argv[3] : "/tmp/test_layout.png";

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) {
        fprintf(stderr, "Failed to load reference: %s\n", argv[2]);
        return 1;
    }

    // Print available reference tensors
    printf("Reference tensors:\n");
    for (auto& name : ref.tensor_names()) {
        auto [data, n] = ref.get_f32(name);
        auto shape = ref.shape(name);
        printf("  %s: n=%zu shape=[", name.c_str(), n);
        for (size_t i = 0; i < shape.size(); i++) printf("%s%lld", i?",":"", (long long)shape[i]);
        printf("]\n");
    }

    // Load model and run detection (with debug dumps enabled)
    setenv("LAYOUT_DEBUG", "1", 1);

    layout_detect::context* ctx = nullptr;
    if (!layout_detect::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        return 1;
    }

    auto regions = layout_detect::detect_file(ctx, image_path, 0.1f);
    printf("\nDetected %zu regions\n", regions.size());
    for (size_t i = 0; i < std::min(regions.size(), (size_t)5); i++) {
        printf("  [%zu] %s score=%.3f\n", i, regions[i].label_name, regions[i].score);
    }

    // Compare encoder stages using dumped files
    struct StageCheck {
        const char* name;
        const char* dump_path;
        int n_elem;
        bool need_transpose;  // true if C++ dumps in different layout than ref
    };

    // The C++ dumps are done via LAYOUT_DEBUG env var in layout_detect.cpp
    // They dump to /tmp/cpp_*.bin files
    // The reference stores in ggml format

    // Compare enc_output
    {
        auto [ref_data, ref_n] = ref.get_f32("enc_output");
        if (ref_data && ref_n > 0) {
            FILE* fp = fopen("/tmp/cpp_enc_output.bin", "rb");
            if (fp) {
                std::vector<float> cpp_data(ref_n);
                fread(cpp_data.data(), sizeof(float), ref_n, fp);
                fclose(fp);

                auto r = ref.compare("enc_output", cpp_data.data(), ref_n);
                printf("\nenc_output: cos_min=%.6f cos_mean=%.6f max_abs=%.4f %s\n",
                       r.cos_min, r.cos_mean, r.max_abs,
                       r.cos_min >= 0.99f ? "PASS" : "FAIL");
            }
        }
    }

    // Compare s3/s4/s5 and ip3/ip4/ip5
    // The C++ encoder graph outputs are in ggml [W, H, C] layout
    // The reference stores in gguf which uses the same byte order as numpy
    // We need to compare after converting to the same layout

    // For now, use the ggml graph outputs which are read by the debug code
    // The s3/s4/s5 comparison would require hooking into the ggml graph
    // outputs, which is done via the debug range prints but not as binary dumps.

    // Instead, compare the raw memory (which includes all 3 scales flattened)
    {
        auto [ref_data, ref_n] = ref.get_f32("enc_output");
        if (ref_data) {
            // enc_output from ref is [256, 8400] in ggml ne order
            // The C++ dumps enc_output as [8400, 256] row-major
            // Let's compare directly
            FILE* fp = fopen("/tmp/cpp_enc_output.bin", "rb");
            if (fp) {
                std::vector<float> cpp(ref_n);
                fread(cpp.data(), sizeof(float), ref_n, fp);
                fclose(fp);

                float cos = cosine_sim(ref_data, cpp.data(), ref_n);
                printf("enc_output (flat cos): %.6f\n", cos);

                // Print first mismatches
                int mismatches = 0;
                for (size_t i = 0; i < std::min(ref_n, (size_t)20); i++) {
                    float diff = fabs(ref_data[i] - cpp[i]);
                    if (diff > 0.1f && mismatches < 5) {
                        printf("  [%zu] ref=%.4f cpp=%.4f diff=%.4f\n",
                               i, ref_data[i], cpp[i], diff);
                        mismatches++;
                    }
                }
            }
        }
    }

    layout_detect::free(ctx);
    return 0;
}
