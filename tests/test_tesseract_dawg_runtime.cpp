#include "tesseract_lstm.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>

static int test_main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dawg-name> <utf8-text>\n", argv[0]);
        return 2;
    }
    setenv("CRISPEMBED_TESSERACT_DAWG_LOAD", "1", 1);
    setenv("CRISPEMBED_TESSERACT_FORCE_CPU", "1", 1);
    auto * ctx = tesseract_lstm_init(argv[1], 1);
    if (!ctx) return 1;
    const int count = tesseract_lstm_dawg_count(ctx);
    const int complete = tesseract_lstm_dawg_matches_utf8(ctx, argv[2], argv[3], 1);
    const int prefix = tesseract_lstm_dawg_matches_utf8(ctx, argv[2], argv[3], 0);
    const int empty_prefix = tesseract_lstm_dawg_matches_utf8(ctx, argv[2], "", 0);
    const int missing = tesseract_lstm_dawg_matches_utf8(ctx, "missing-dawg", argv[3], 0);
    std::printf("dawgs=%d complete=%d prefix=%d empty_prefix=%d missing=%d\n", count, complete, prefix, empty_prefix,
                missing);
    tesseract_lstm_free(ctx);
    return count > 0 && complete >= 0 && prefix >= 0 && empty_prefix == 1 && missing == -1 ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(test_main(argc, argv));
}
