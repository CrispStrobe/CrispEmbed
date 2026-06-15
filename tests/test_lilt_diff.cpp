// tests/test_lilt_diff.cpp — LiLT parity test: C++ vs Python per-layer.
//
// Usage:
//   ./test-lilt-diff lilt-funsd-f32.gguf /tmp/lilt-funsd-ref-full.gguf

#include "lilt_kie.h"
#include "crispembed_diff.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { return 1; }

    printf("Reference tensors:\n");
    for (auto& name : ref.tensor_names()) {
        auto [ptr, n] = ref.get_f32(name);
        printf("  %s: %zu elements\n", name.c_str(), n);
    }

    // Load model
    lilt_kie::context* ctx = nullptr;
    if (!lilt_kie::load(&ctx, argv[1], 4)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Read input_ids and bbox from the reference GGUF
    auto [ref_ids_f, ref_ids_n] = ref.get_f32("input_ids");
    auto [ref_bbox_f, ref_bbox_n] = ref.get_f32("bbox");
    if (!ref_ids_f || ref_ids_n == 0) {
        fprintf(stderr, "Reference missing input_ids\n");
        return 1;
    }
    int T = (int)ref_ids_n;
    std::vector<int32_t> ids(T);
    for (int i = 0; i < T; i++) ids[i] = (int32_t)ref_ids_f[i];

    // bbox is stored as [4, T] in the reference (ggml col-major), read as flat [T*4]
    std::vector<int32_t> bbox(T * 4, 0);
    if (ref_bbox_f && ref_bbox_n >= (size_t)(T * 4)) {
        // ref bbox shape is [4, T] in ggml convention = [T, 4] row-major
        for (int i = 0; i < T * 4; i++)
            bbox[i] = (int32_t)ref_bbox_f[i];
    }

    printf("T=%d, ids[0..4]=[%d,%d,%d,%d,%d]\n", T, ids[0], ids[1], ids[2], ids[3], ids[4]);
    printf("bbox[1]=[%d,%d,%d,%d]\n", bbox[4], bbox[5], bbox[6], bbox[7]);

    printf("\nRunning C++ inference with dump...\n");
    auto dumps = lilt_kie::classify_dump(ctx, ids.data(), bbox.data(), T);

    printf("\n%-25s %8s %10s %10s %s\n", "Stage", "Elements", "cos_min", "max_abs", "Status");
    printf("%-25s %8s %10s %10s %s\n", "-----", "--------", "-------", "-------", "------");

    int n_pass = 0, n_fail = 0, n_skip = 0;
    for (auto& dt : dumps) {
        if (!ref.has(dt.name)) {
            printf("%-25s %8d %10s %10s SKIP (no ref)\n", dt.name.c_str(), dt.n_elem, "-", "-");
            n_skip++;
            continue;
        }

        auto r = ref.compare(dt.name, dt.data.data(), (size_t)dt.n_elem);
        bool pass = r.is_pass(0.999f);
        printf("%-25s %8zu %10.6f %10.2e %s\n",
               dt.name.c_str(), r.n_elem, r.cos_min, r.max_abs,
               pass ? "PASS" : "FAIL");
        if (pass) n_pass++; else n_fail++;
    }

    // Also compare logits if present
    if (ref.has("logits")) {
        // Run regular classify to get logits
        auto results = lilt_kie::classify(ctx, ids.data(), bbox.data(), T);
        // We don't have raw logits from classify, but we can compare predictions
        printf("\n%-25s\n", "--- Label predictions ---");
        auto [ref_logits, ref_n] = ref.get_f32("logits");
        int n_labels = (int)(ref_n / T);
        int matches = 0;
        for (int i = 0; i < T && i < (int)results.size(); i++) {
            // Get ref argmax
            const float* row = ref_logits + i * n_labels;
            int ref_best = 0;
            for (int j = 1; j < n_labels; j++)
                if (row[j] > row[ref_best]) ref_best = j;
            bool match = results[i].label_id == ref_best;
            if (match) matches++;
            printf("  tok=%5d  ref_label=%d  cpp_label=%d  %s\n",
                   ids[i], ref_best, results[i].label_id, match ? "OK" : "MISMATCH");
        }
        printf("Label match: %d/%d (%.0f%%)\n", matches, T, 100.0f * matches / T);
    }

    printf("\n%d PASS, %d FAIL, %d SKIP\n", n_pass, n_fail, n_skip);

    lilt_kie::free(ctx);
    return n_fail > 0 ? 1 : 0;
}
