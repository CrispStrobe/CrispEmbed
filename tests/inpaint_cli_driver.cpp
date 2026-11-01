// Test-only implementations exercise the real CLI's file/mask/batch handling.
// This target does not execute MI-GAN and is never installed or shipped.
#include "inpaint_cli.h"
#include "inpaint.h"
#include "lama.h"
#include "latent_inpaint.h"
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

context * init(const char *, int, std::string & error) {
    if (std::getenv("INPAINT_TEST_FAIL_INIT")) {
        error = "test model initialization failure";
        return nullptr;
    }
    std::fprintf(stderr, "INPAINT_TEST_LOAD\n");
    return new context;
}

context * init_variant(const char * path, const char *, const char *, int resolution, int threads,
                       std::string & error) {
    std::fprintf(stderr, "INPAINT_TEST_VARIANT resolution=%d\n", resolution);
    return init(path, threads, error);
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

namespace latent_inpaint {
struct context {
    inpaint::context * inner = nullptr;
};
context * init(const Options & options, std::string & error) {
    std::fprintf(stderr, "INPAINT_TEST_LATENT model=%s steps=%d seed=%llu tiles=%d guidance=%.9g noise_offset=%.9g\n",
                 options.model.c_str(), options.steps, static_cast<unsigned long long>(options.seed),
                 int(options.tiled), double(options.guidance), double(options.noise_offset));
    auto * inner = inpaint::init(options.denoiser_path.c_str(), options.threads, error);
    return inner ? new context{ inner } : nullptr;
}
void free(context * ctx) {
    if (ctx) inpaint::free(ctx->inner);
    delete ctx;
}
bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    return inpaint::process_rgb8(ctx->inner, rgb, rgb_bytes, mask, mask_bytes, width, height, output, output_bytes,
                                 error);
}
} // namespace latent_inpaint

namespace lama {
struct context {
    inpaint::context * inner = nullptr;
};
context * init(const std::string & path, const std::string &, int threads, std::string & error) {
    std::fprintf(stderr, "INPAINT_TEST_LAMA\n");
    auto * inner = inpaint::init(path.c_str(), threads, error);
    return inner ? new context{ inner } : nullptr;
}
void free(context * ctx) {
    if (ctx) inpaint::free(ctx->inner);
    delete ctx;
}
bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    return inpaint::process_rgb8(ctx->inner, rgb, rgb_bytes, mask, mask_bytes, width, height, output, output_bytes,
                                 error);
}
} // namespace lama

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
