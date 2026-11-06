#include "qwen35_ocr.h"
#include "gguf.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
static void gate(const char * n, const char * v) {
#ifdef _WIN32
    _putenv_s(n, v);
#else
    setenv(n, v, 1);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    auto * metadata = gguf_init_empty();
    if (!metadata) return 2;
    gguf_set_val_str(metadata, "general.architecture", "qwen35_ocr");
    gguf_set_val_u32(metadata, "qwen35_ocr.format_version", 5);
    gguf_set_val_str(metadata, "qwen35_ocr.profile", "lightonocr3-4b");
    gguf_set_val_bool(metadata, "qwen35_ocr.tie_word_embeddings", true);
    gguf_set_val_bool(metadata, "qwen35_ocr.zero_centered_text_norm", true);
    if (!gguf_write_to_file(metadata, argv[1], true)) return 2;
    gguf_free(metadata);
    gate("CRISPEMBED_QWEN35_OCR", "1");
    int checks = 0;
    for (const char * value : { "0", "1" }) {
        for (const char * name :
             { "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_FRONTEND", "CRISPEMBED_LIGHTON4B_QUANT",
               "CRISPEMBED_LIGHTON4B_Q6_K", "CRISPEMBED_LIGHTON4B_Q4_K_M", "CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD",
               "CRISPEMBED_QWEN35_QUANT" })
            gate(name, value);
        auto * context = qwen35_ocr_init(argv[1], 1);
        if (context) {
            qwen35_ocr_free(context);
            return 1;
        }
        ++checks;
    }
    std::printf("LIGHTON4B_RELEASE_FENCE passed=1 metadata_only_public_rejections=%d format5_public_admission=0 "
                "model_payload=0\n",
                checks);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
