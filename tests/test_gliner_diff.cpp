// tests/test_gliner_diff.cpp — Per-layer parity test for GLiNER.
//
// The actual per-layer comparison runs inside gliner_ner_extract() when
// the GLINER_DIFF_REF environment variable points to a reference GGUF.
//
// Usage:
//   export GLINER_DEBUG=1
//   export GLINER_DIFF_REF=/mnt/volume1/gliner-ref.gguf
//   ./test-gliner-diff <model.gguf> [text]

#include "gliner_ner.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [text]\n", argv[0]);
        fprintf(stderr, "  Set GLINER_DIFF_REF=<ref.gguf> for per-layer comparison\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * text = argc > 2 ? argv[2]
        : "Barack Obama was born in Hawaii";

    // Enable debug output (portable: MSVC has no setenv)
#ifdef _WIN32
    _putenv_s("GLINER_DEBUG", "1");
#else
    setenv("GLINER_DEBUG", "1", 0);
#endif

    printf("Loading model: %s\n", model_path);
    void * ctx = gliner_ner_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    const char * labels[] = {"person", "organization", "location"};
    gliner_ner_entity * entities = nullptr;
    int n = gliner_ner_extract(ctx, text, labels, 3, 0.3f, &entities);

    printf("\nResult (%d entities):\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d-%d] \"%s\" => %s (%.3f)\n",
               entities[i].start_char, entities[i].end_char,
               entities[i].text, entities[i].label, entities[i].score);
    }

    // Independent output-check guardrail (used when the default text is run).
    // The per-stage ref path (GLINER_DIFF_REF) can be misleading if the *reference*
    // dumper is broken, so assert the ENGINE extracts the expected entities directly.
    int rc = 0;
    if (argc <= 2) {  // default text = "Barack Obama was born in Hawaii"
        bool person = false, hawaii = false;
        for (int i = 0; i < n; i++) {
            std::string t = entities[i].text ? entities[i].text : "";
            std::string l = entities[i].label ? entities[i].label : "";
            if (l == "person" && t.find("Obama") != std::string::npos) person = true;
            if (l == "location" && t.find("Hawaii") != std::string::npos) hawaii = true;
        }
        printf("[gliner-output-check] person(Obama)=%s location(Hawaii)=%s => %s\n",
               person ? "yes" : "no", hawaii ? "yes" : "no",
               (person && hawaii) ? "PASS" : "FAIL");
        if (!(person && hawaii)) rc = 1;
    }

    gliner_ner_free(ctx);
    printf("\nDone.\n");
    return rc;
}
