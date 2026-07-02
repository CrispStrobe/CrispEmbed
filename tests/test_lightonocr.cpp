// tests/test_lightonocr.cpp — LightOnOCR-2-1B smoke test
#include "lightonocr.h"
#include "core/clean_exit.h"
#include <cstdio>
static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image>\n", argv[0]);
        return 1;
    }
    lightonocr_context * ctx = lightonocr_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load\n");
        return 1;
    }

    int len = 0;
    const char * text = lightonocr_recognize_file(ctx, argv[2], &len);
    if (text)
        printf("Result: %s\n", text);
    else
        printf("Failed\n");

    lightonocr_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
