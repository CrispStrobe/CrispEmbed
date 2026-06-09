// test_ocr_detect_diff.cpp — compare DBNet C++ output against Python reference.
//
// Usage: test-ocr-detect-diff model.gguf ref.gguf [image.png]
//
// Loads the DBNet model and the reference GGUF (from dump_dbnet_reference.py),
// runs the same preprocessed input through the C++ forward pass, and compares
// the probability map using the crispembed_diff harness.

#include "ocr_detect.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dbnet.gguf> <ref.gguf> [image.png]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* ref_path = argv[2];

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "Failed to load reference: %s\n", ref_path);
        return 1;
    }

    // Print available reference tensors
    printf("Reference tensors:\n");
    auto names = ref.tensor_names();
    for (auto& n : names) {
        auto shape = ref.shape(n);
        printf("  %s: [", n.c_str());
        for (size_t i = 0; i < shape.size(); i++) {
            if (i > 0) printf(", ");
            printf("%lld", (long long)shape[i]);
        }
        printf("]\n");
    }

    // Get the preprocessed input from reference
    auto [input_data, input_n] = ref.get_f32("input_image");
    if (!input_data || input_n == 0) {
        fprintf(stderr, "No input_image in reference GGUF\n");
        return 1;
    }

    auto input_shape = ref.shape("input_image");
    if (input_shape.size() < 3) {
        fprintf(stderr, "input_image has unexpected shape\n");
        return 1;
    }

    // GGUF stores dims as ne[0]=W, ne[1]=H, ne[2]=C (reversed from NumPy CHW)
    int W = (int)input_shape[0];
    int H = (int)input_shape[1];
    int C = (int)input_shape[2];
    printf("\nInput: C=%d H=%d W=%d (%zu elements)\n", C, H, W, input_n);

    // Load model
    ocr_detect::context* ctx = nullptr;
    if (!ocr_detect::load(&ctx, model_path, 4)) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    // Run forward pass with the reference input
    // The detect() function expects CHW float32 (already preprocessed)
    auto boxes = ocr_detect::detect(ctx, input_data, H, W);
    printf("Detected %zu text regions\n", boxes.size());

    // Get the probability map
    int prob_h, prob_w;
    const float* prob = ocr_detect::get_prob_map(ctx, &prob_h, &prob_w);
    if (!prob) {
        fprintf(stderr, "No probability map available\n");
        ocr_detect::free(ctx);
        return 1;
    }
    printf("Prob map: %dx%d\n", prob_w, prob_h);

    // Compare against reference prob_map_sigmoid
    printf("\n--- Probability map comparison ---\n");
    auto r = ref.compare("prob_map_sigmoid", prob, prob_h * prob_w);
    if (r.found) {
        printf("  cos_min=%.6f cos_mean=%.6f max_abs=%.2e rms=%.2e  %s\n",
               r.cos_min, r.cos_mean, r.max_abs, r.rms,
               r.is_pass(0.95f) ? "PASS" : "FAIL");
    } else {
        printf("  prob_map_sigmoid not found in reference\n");
    }

    // Also compare pre-sigmoid prob map
    // (We'd need to expose the pre-sigmoid map too — for now just check sigmoid)

    // Print stats
    float cpp_min = 1, cpp_max = 0, cpp_sum = 0;
    for (int i = 0; i < prob_h * prob_w; i++) {
        if (prob[i] < cpp_min) cpp_min = prob[i];
        if (prob[i] > cpp_max) cpp_max = prob[i];
        cpp_sum += prob[i];
    }
    float cpp_mean = cpp_sum / (prob_h * prob_w);

    auto [ref_prob, ref_n] = ref.get_f32("prob_map_sigmoid");
    if (ref_prob && ref_n > 0) {
        float ref_min = 1, ref_max = 0, ref_sum = 0;
        for (size_t i = 0; i < ref_n; i++) {
            if (ref_prob[i] < ref_min) ref_min = ref_prob[i];
            if (ref_prob[i] > ref_max) ref_max = ref_prob[i];
            ref_sum += ref_prob[i];
        }
        float ref_mean = ref_sum / ref_n;

        printf("\n  C++:    min=%.6f max=%.6f mean=%.6f (%d elements)\n",
               cpp_min, cpp_max, cpp_mean, prob_h * prob_w);
        printf("  Python: min=%.6f max=%.6f mean=%.6f (%zu elements)\n",
               ref_min, ref_max, ref_mean, ref_n);
    }

    ocr_detect::free(ctx);
    return r.found && r.is_pass(0.95f) ? 0 : 1;
}
