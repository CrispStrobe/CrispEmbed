// tests/test_gliner_diff.cpp — Per-layer parity test for GLiNER.
//
// Compares C++ intermediates against Python reference dump.
//
// Usage:
//   python tools/dump_gliner_reference.py --output /mnt/volume1/gliner-ref.gguf
//   ./test-gliner-diff /mnt/storage/gguf-models/gliner-lfm-f32.gguf \
//       /mnt/volume1/gliner-ref.gguf "Barack Obama was born in Hawaii"

#include "crispembed.h"
#include "crispembed_diff.h"
#include "gliner_ner.h"

// We need access to the internals for layer-by-layer comparison.
// For now, just compare the final NER output and overall pipeline.
// Detailed per-layer comparison requires exposing intermediate dumps
// from gliner_ner.cpp (controlled by GLINER_DEBUG env var).

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf> [text]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * ref_path   = argv[2];
    const char * text = argc > 3 ? argv[3]
        : "Barack Obama was born in Hawaii";

    // Load reference
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) {
        fprintf(stderr, "ERROR: failed to load reference: %s\n", ref_path);
        return 1;
    }
    printf("Reference loaded: %zu tensors\n", ref.tensors.size());
    for (auto & [name, td] : ref.tensors) {
        printf("  %s: %zu elements\n", name.c_str(), td.data.size());
    }

    // Load model
    printf("\nLoading model: %s\n", model_path);
    void * ctx = crispembed_ner_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    // Run NER
    const char * labels[] = {"person", "organization", "location", "email"};
    crispembed_ner_entity * entities = nullptr;
    int n = crispembed_ner_extract(ctx, text, labels, 4, 0.3f, &entities);

    printf("\nC++ result (%d entities):\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d-%d] \"%s\" => %s (%.3f)\n",
               entities[i].start_char, entities[i].end_char,
               entities[i].text, entities[i].label, entities[i].score);
    }

    // TODO: Compare per-layer intermediates when the C++ runtime
    // exposes them via a stage callback or dump file.
    // For now, check that the reference tensors exist and are reasonable.
    printf("\nReference tensor stats:\n");
    for (auto & [name, td] : ref.tensors) {
        if (td.data.empty()) continue;
        float min_v = td.data[0], max_v = td.data[0], sum = 0;
        for (float v : td.data) {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += v;
        }
        float mean = sum / td.data.size();
        printf("  %s: min=%.4f max=%.4f mean=%.6f\n",
               name.c_str(), min_v, max_v, mean);
    }

    crispembed_ner_free(ctx);
    printf("\nDone.\n");
    return 0;
}
