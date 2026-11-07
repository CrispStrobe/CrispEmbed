#include "lighton4b_native_context.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 1 && argc != 2) return 2;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "failed control %d\n", checks);
            std::exit(1);
        }
    };
    for (const char * value : { "", "0" }) {
        gate("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", value);
        check(!qwen35_ocr::lighton4b_native_context::open("missing.gguf"));
    }
    gate("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "1");
    gate("CRISPEMBED_LIGHTON4B_QUANT", "1");
    gate("CRISPEMBED_QWEN35_QUANT", "1");
    auto * ctx = ggml_init({ 16384, nullptr, true });
    for (const auto type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K }) {
        auto * t = ggml_new_tensor_2d(ctx, type, 2560, 32);
        check(qwen35_ocr::lighton4b_matrix_type(t) == (type != GGML_TYPE_Q6_K));
        gate("CRISPEMBED_LIGHTON4B_QUANT", "0");
        check(!qwen35_ocr::lighton4b_matrix_type(t));
        gate("CRISPEMBED_LIGHTON4B_QUANT", "1");
    }
    ggml_free(ctx);
    for (auto * name :
         { "CRISPEMBED_LIGHTON4B_GGUF_INSPECTION", "CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER",
           "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", "CRISPEMBED_LIGHTON4B_VISION_OWNER", "CRISPEMBED_QWEN35_OCR" })
        gate(name, "1");
    check(!qwen35_ocr::lighton4b_native_context::open("missing.gguf"));
    if (argc == 2) {
        auto owned = qwen35_ocr::lighton4b_native_context::open(argv[1]);
        check(bool(owned));
        check(owned->model().text.decoder.layers.size() == 32);
    }
    std::printf("native-context controls=%d passed=1 model_payload=%d\n", checks, argc == 2);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
