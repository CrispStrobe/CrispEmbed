#include "smt_ocr.h"
#include "core/clean_exit.h"
#include <cstdio>
static int run(int c, char ** v) {
    if (c < 3) return 2;
    auto * ctx = smt_ocr_init(v[1], 4);
    if (!ctx) return 1;
    int len = 0;
    const char * s = smt_ocr_recognize_file(ctx, v[2], &len);
    printf("DECODED (%d):\n%s\n", len, s ? s : "(null)");
    smt_ocr_free(ctx);
    return 0;
}
int main(int c, char ** v) {
    core_util::clean_exit(run(c, v));
}
