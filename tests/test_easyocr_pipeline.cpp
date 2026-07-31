#include "easyocr_pipeline.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstring>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <easyocr.gguf> <image> <lines|words>\n", argv[0]);
        return 2;
    }
    easyocr_pipeline::context * ctx = nullptr;
    if (!easyocr_pipeline::load(&ctx, argv[1], argv[2], 1)) return 3;
    const auto mode = std::strcmp(argv[4], "words") == 0 ? easyocr_layout::ordering_mode::words
                                                         : easyocr_layout::ordering_mode::lines;
    easyocr_pipeline::set_ordering_mode(ctx, mode);
    const auto results = easyocr_pipeline::run_file(ctx, argv[3]);
    std::printf("easyocr-pipeline mode=%s results=%zu\n",
                mode == easyocr_layout::ordering_mode::words ? "words" : "lines", results.size());
    if (results.empty()) {
        easyocr_pipeline::free(ctx);
        return 4;
    }
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & item = results[i];
        std::printf("result=%zu line=%d box=%.1f,%.1f %.1fx%.1f det_conf=%.4f rec_conf=%.4f norm=%d,%d,%d,%d text=%s\n",
                    i, item.word.line, item.word.x, item.word.y, item.word.w, item.word.h, item.detector_confidence,
                    item.word.confidence, item.normalized.x0, item.normalized.y0, item.normalized.x1,
                    item.normalized.y1, item.word.text.c_str());
    }
    easyocr_pipeline::free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
