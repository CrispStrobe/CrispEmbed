// tests/test_lilt_kie.cpp — LiLT KIE integration test.
//
// Usage:
//   ./test-lilt-kie /path/to/lilt-funsd-f32.gguf

#include "lilt_kie.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <lilt-model.gguf>\n", argv[0]);
        return 1;
    }

    printf("Loading LiLT model: %s\n", argv[1]);
    lilt_kie::context* ctx = nullptr;
    if (!lilt_kie::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "ERROR: failed to load\n");
        return 1;
    }

    printf("Labels: %d\n", lilt_kie::num_labels(ctx));

    // Test input: "Date: 2026-06-15 Total: $48.60"
    // Token IDs from RoBERTa tokenizer (pre-computed)
    int32_t ids[] = {0, 10566, 35, 291, 2481, 12, 4124, 12, 996, 5480, 35, 68, 3818, 4, 2466, 2};
    int T = 16;

    // Bounding boxes: [x0, y0, x1, y1] per token
    // BOS/EOS get [0,0,0,0], words get spread positions
    int32_t bbox[] = {
        0,0,0,0,       // <s>
        10,50,90,80,    // Date
        90,50,110,80,   // :
        120,50,170,80,  // 20
        170,50,210,80,  // 26
        210,50,230,80,  // -
        230,50,270,80,  // 06
        270,50,290,80,  // -
        290,50,320,80,  // 15
        350,50,430,80,  // Total
        430,50,450,80,  // :
        460,50,490,80,  // $
        490,50,530,80,  // 48
        530,50,540,80,  // .
        540,50,570,80,  // 60
        0,0,0,0,        // </s>
    };

    printf("\nRunning inference (%d tokens)...\n", T);
    auto results = lilt_kie::classify(ctx, ids, bbox, T);

    printf("\nResults:\n");
    for (int i = 0; i < (int)results.size(); i++) {
        printf("  token=%5d  label=%-15s  score=%.3f\n",
               results[i].token_id, results[i].label.c_str(), results[i].score);
    }

    lilt_kie::free(ctx);
    printf("\nDone.\n");
    return 0;
}
