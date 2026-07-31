#include "ocr_pipeline_pool.h"

#include <cstdio>
#include <cstdlib>

static bool expect(bool value, const char * label) {
    if (!value) std::fprintf(stderr, "FAIL: %s\n", label);
    return value;
}

int main() {
    unsetenv("CRISPEMBED_DEBUG_ALLOW_OCR_Q4");
    if (!expect(ocr_pipeline::is_dangerous_q4_recognizer_path("trocr-small-printed-q4_k.gguf"),
                "detect dangerous TrOCR Q4 path"))
        return 1;
    if (!expect(!ocr_pipeline::dangerous_q4_override_enabled(), "Q4 override disabled by default")) return 1;
    ocr_pipeline::context * rejected = nullptr;
    if (!expect(!ocr_pipeline::load(&rejected, "/no/such/det.gguf", "trocr-small-printed-q4_k.gguf", 2),
                "reject dangerous Q4 before loading"))
        return 1;
    if (!expect(rejected == nullptr, "rejected Q4 leaves null context")) return 1;

    setenv("CRISPEMBED_DEBUG_ALLOW_OCR_Q4", "1", 1);
    if (!expect(ocr_pipeline::dangerous_q4_override_enabled(), "Q4 override enabled explicitly")) return 1;

    ocr_pipeline_pool::context * ctx = nullptr;
    if (ocr_pipeline_pool::load(&ctx, "/no/such/det.gguf", "/no/such/rec.gguf", 2, 1)) {
        std::fprintf(stderr, "pool unexpectedly loaded missing models\n");
        ocr_pipeline_pool::free(ctx);
        return 1;
    }
    if (ctx != nullptr) {
        std::fprintf(stderr, "failed pool load left a context\n");
        ocr_pipeline_pool::free(ctx);
        return 1;
    }
    return 0;
}
