#include "tesseract_lstm.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
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
    std::printf("dawgs=%d complete=%d prefix=%d\n", count, complete, prefix);
    tesseract_lstm_free(ctx);
    return count > 0 && complete >= 0 && prefix >= 0 ? 0 : 1;
}
