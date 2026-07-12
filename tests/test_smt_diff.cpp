// tests/test_smt_diff.cpp — SMT per-stage parity vs the Python reference GGUF.
#include "smt_ocr.h"
#include "core/clean_exit.h"
#include <cstdio>

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <smt-model.gguf> <smt_ref.gguf>\n", argv[0]);
        return 2;
    }
    smt_ocr_context * ctx = smt_ocr_init(argv[1], 4);
    if (!ctx) return 1;
    int rc = smt_ocr_run_diff(ctx, argv[2]);
    smt_ocr_free(ctx);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
