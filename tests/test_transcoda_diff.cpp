// tests/test_transcoda_diff.cpp — Transcoda per-stage parity vs the Python ref GGUF.
#include "transcoda_ocr.h"
#include "core/clean_exit.h"
#include <cstdio>

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <transcoda-model.gguf> <transcoda_ref.gguf>\n", argv[0]);
        return 2;
    }
    transcoda_ocr_context * ctx = transcoda_ocr_init(argv[1], 4);
    if (!ctx) return 1;
    int rc = transcoda_ocr_run_diff(ctx, argv[2]);
    transcoda_ocr_free(ctx);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
