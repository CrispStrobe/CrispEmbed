// tests/test_tesseract_lstm_diff.cpp — parity diff between C++ Tesseract LSTM
// engine and Python reference activations.
//
// Usage:
//   # 1. Generate reference:
//   python tools/dump_tesseract_reference.py \
//       --model eng.traineddata --image test.png --output ref.gguf
//
//   # 2. Convert model:
//   python models/convert-tesseract-to-gguf.py \
//       --model eng.traineddata --output model.gguf
//
//   # 3. Run diff:
//   ./test-tesseract-lstm-diff model.gguf ref.gguf test.png

#include "tesseract_lstm.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// stb_image for loading test image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model.gguf> <reference.gguf> <image.png>\n\n"
            "Generate reference with:\n"
            "  python tools/dump_tesseract_reference.py \\\n"
            "      --model eng.traineddata --image test.png --output ref.gguf\n",
            argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * ref_path   = argv[2];
    const char * image_path = argv[3];

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) return 1;

    auto ref_names = ref.tensor_names();
    printf("Reference: %zu tensors\n", ref_names.size());
    for (auto & n : ref_names) {
        auto [d, sz] = ref.get_f32(n);
        printf("  %-20s %zu elements\n", n.c_str(), sz);
    }

    // Load model
    tesseract_lstm_context * ctx = tesseract_lstm_init(model_path, 1);
    if (!ctx) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    // Enable dump mode
    tesseract_lstm_set_dump(ctx, 1);

    // Load image
    int w, h, channels;
    unsigned char * img = stbi_load(image_path, &w, &h, &channels, 1);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        tesseract_lstm_free(ctx);
        return 1;
    }
    printf("\nImage: %dx%d\n", w, h);

    // Run C++ forward pass
    int out_len = 0;
    const char * text = tesseract_lstm_recognize(ctx, img, w, h, &out_len);
    printf("C++ result: '%s'\n", text);

    // Check Python result from reference metadata
    std::string py_text = ref.meta("tesseract_lstm_ref.decoded_text");
    if (!py_text.empty())
        printf("Py  result: '%s'\n", py_text.c_str());

    // Compare each stage
    printf("\n=== Per-stage parity ===\n");

    const char * stages[] = {
        "input_image",
        "after_convolve",
        "after_conv_fc",
        "after_maxpool",
        "after_lstm_0",
        "after_lstm_1",
        "after_lstm_2",
        "after_lstm_3",
        "logits",
    };

    int n_pass = 0, n_fail = 0, n_skip = 0;

    for (const char * stage : stages) {
        // Get C++ capture
        int cpp_n = 0;
        const float * cpp_data = tesseract_lstm_get_capture(ctx, stage, &cpp_n);

        if (!cpp_data || cpp_n == 0) {
            // C++ didn't produce this stage (e.g. "after_convolve" not captured)
            if (ref.has(stage)) {
                printf("  %-20s  C++ missing, ref has %zu elem  SKIP\n",
                       stage, ref.get_f32(stage).second);
                n_skip++;
            }
            continue;
        }

        if (!ref.has(stage)) {
            printf("  %-20s  C++ has %d elem, ref missing  SKIP\n", stage, cpp_n);
            n_skip++;
            continue;
        }

        auto r = ref.compare(stage, cpp_data, cpp_n);

        const char * verdict;
        if (r.cos_min >= 0.9999f) verdict = "PASS";
        else if (r.cos_min >= 0.999f) verdict = "PASS (ok)";
        else if (r.cos_min >= 0.99f) verdict = "WARN";
        else { verdict = "FAIL"; }

        bool pass = r.cos_min >= 0.999f;
        if (pass) n_pass++; else n_fail++;

        printf("  %-20s  cos_min=%.6f  max_abs=%.2e  mean_abs=%.2e  %s\n",
               stage, r.cos_min, r.max_abs, r.mean_abs, verdict);
    }

    printf("\n=== Summary: %d PASS, %d FAIL, %d SKIP ===\n",
           n_pass, n_fail, n_skip);

    stbi_image_free(img);
    tesseract_lstm_free(ctx);

    return n_fail > 0 ? 1 : 0;
}
