// tests/test_gliner_ner.cpp — GLiNER NER integration test.
//
// Usage:
//   ./test-gliner-ner /path/to/gliner-lfm-f32.gguf
//
// Expected output: detects entities like "Maria Schmidt" (person),
// "Siemens" (organization), "München" (location).

#include "crispembed.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [text]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const char * text = argc > 2 ? argv[2]
        : "Maria Schmidt arbeitet bei Siemens in München, E-Mail: maria.schmidt@siemens.com";

    printf("Loading model: %s\n", model_path);
    void * ctx = crispembed_ner_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    const char * labels[] = {"person", "organization", "location", "email"};
    int n_labels = 4;
    float threshold = 0.5f;

    printf("\nText: %s\n", text);
    printf("Labels: person, organization, location, email\n");
    printf("Threshold: %.2f\n\n", threshold);

    crispembed_ner_entity * entities = nullptr;
    int n = crispembed_ner_extract(ctx, text, labels, n_labels, threshold, &entities);

    printf("Found %d entities:\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d-%d] \"%s\" => %s (%.3f)\n",
               entities[i].start_char, entities[i].end_char,
               entities[i].text, entities[i].label, entities[i].score);
    }

    crispembed_ner_free(ctx);
    printf("\nDone.\n");
    return 0;
}
