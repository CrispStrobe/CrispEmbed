// tests/test_tromr_diff.cpp — TrOMR per-stage parity vs the Python reference GGUF.
#include "tromr_ocr.h"
#include "core/clean_exit.h"
#include <cstdio>

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <tromr-model.gguf> <tromr_ref.gguf>\n", argv[0]);
        return 2;
    }
    tromr_ocr_context * ctx = tromr_ocr_init(argv[1], 4);
    if (!ctx) return 1;
    int rc = tromr_ocr_run_diff(ctx, argv[2]);
    tromr_ocr_free(ctx);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
