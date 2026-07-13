// tests/test_flova_diff.cpp — Flova/omr_transformer per-stage parity vs the
// Python reference GGUF (tools/dump_flova_reference.py).
#include "flova_ocr.h"
#include "core/clean_exit.h"
#include <cstdio>

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <flova-model.gguf> <flova_ref.gguf>\n", argv[0]);
        return 2;
    }
    flova_ocr_context * ctx = flova_ocr_init(argv[1], 4);
    if (!ctx) return 1;
    int rc = flova_ocr_run_diff(ctx, argv[2]);
    flova_ocr_free(ctx);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
