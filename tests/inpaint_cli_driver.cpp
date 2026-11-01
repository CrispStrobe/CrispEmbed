// Test-only implementations exercise the real CLI's file/mask/batch handling.
// This target does not execute a neural model and is never installed or shipped.
#include "inpaint_cli.h"
#include "inpaint.h"
#include "ocr_detect.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace inpaint {
struct context {
    int frames = 0;
};

context * init(const char *, int, std::string & error, backend engine) {
    if (std::getenv("INPAINT_TEST_FAIL_INIT")) {
        error = "test model initialization failure";
        return nullptr;
    }
    std::fprintf(stderr, "INPAINT_TEST_LOAD\n");
    std::fprintf(stderr, "INPAINT_TEST_BACKEND=%s\n", engine == backend::lama ? "lama" : "migan");
    return new context;
}

void free(context * ctx) {
    if (ctx) std::fprintf(stderr, "INPAINT_TEST_FREE frames=%d\n", ctx->frames);
    delete ctx;
}

bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    ++ctx->frames;
    const char * fail_at = std::getenv("INPAINT_TEST_FAIL_AT");
    if (fail_at && ctx->frames == std::atoi(fail_at)) {
        error = "test inference failure";
        return false;
    }
    if (rgb_bytes != size_t(width) * height * 3 || mask_bytes != size_t(width) * height || output_bytes != rgb_bytes) {
        error = "invalid test image buffers";
        return false;
    }
    std::memcpy(output, rgb, rgb_bytes);
    for (size_t i = 0; i < mask_bytes; ++i) {
        if (mask[i] != 0 && mask[i] != 255) {
            error = "CLI did not binarize its mask";
            return false;
        }
        if (mask[i]) {
            output[3 * i] = 241;
            output[3 * i + 1] = 31;
            output[3 * i + 2] = 151;
        }
    }
    return true;
}
} // namespace inpaint

namespace ocr_detect {
struct context {};
bool load(context ** ctx, const char *, int) {
    *ctx = new context;
    return true;
}
void free(context * ctx) {
    delete ctx;
}
detect_options rapid_defaults() {
    return {};
}
bool detect_rgb_checked(context *, const uint8_t *, int, int, int, const detect_options &,
                        std::vector<text_box> & boxes, std::string & error) {
    boxes.clear();
    if (std::getenv("INPAINT_TEST_DETECT_FAIL")) {
        error = "test detector failure";
        return false;
    }
    if (std::getenv("INPAINT_TEST_NO_TEXT")) return true;
    text_box box{};
    box.x = 2;
    box.y = 3;
    box.w = 3;
    box.h = 2;
    box.score = 0.9f;
    box.qx[0] = box.qx[3] = 2;
    box.qx[1] = box.qx[2] = 5;
    box.qy[0] = box.qy[1] = 3;
    box.qy[2] = box.qy[3] = 5;
    if (std::getenv("INPAINT_TEST_DETECT_NAN")) box.qx[0] = std::numeric_limits<float>::quiet_NaN();
    boxes.push_back(box);
    return true;
}
} // namespace ocr_detect

int main(int argc, char ** argv) {
    core_util::clean_exit(inpaint_cli::run(argc, argv));
}
