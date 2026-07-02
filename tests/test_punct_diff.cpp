// tests/test_punct_diff.cpp — FireRedPunc / PCS punctuation-restoration guardrail.
// Usage: ./test-punct-diff model.gguf "<input text>" ["<expected output>"]
//
// The punct C API exposes only the restored TEXT (no hidden/logits accessor), so the
// regression signal is a golden text match: a graph-scramble regression (the June-2026
// wave failure mode) turns correct punctuation into garbage / wrong output. With a golden
// (argv[3]) this exits 0 on exact match, 1 otherwise — wired as a manifest `run_check`.
// Without a golden it just prints the output (used to capture/verify the golden).

#include "crispembed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.gguf [\"input\"] [\"expected\"]\n", argv[0]);
        return 2;
    }
    const char * model = argv[1];
    const char * input = argc > 2 ? argv[2]
        : "hello world how are you today i am fine thanks";
    const char * expected = argc > 3 ? argv[3] : nullptr;

    void * ctx = crispembed_punct_init(model, 0);
    if (!ctx) { fprintf(stderr, "punct init failed: %s\n", model); return 1; }

    // Owned by ctx (valid until next call / free) — do NOT free.
    const char * out = crispembed_punct_process(ctx, input);
    if (!out) { fprintf(stderr, "punct process returned null\n"); crispembed_punct_free(ctx); return 1; }

    printf("input:    %s\n", input);
    printf("output:   %s\n", out);
    int rc = 0;
    if (expected) {
        printf("expected: %s\n", expected);
        rc = (std::strcmp(out, expected) == 0) ? 0 : 1;
        printf("%s\n", rc == 0 ? "MATCH" : "MISMATCH");
    }
    crispembed_punct_free(ctx);
    return rc;
}
