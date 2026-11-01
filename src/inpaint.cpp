// Still-image geometry around the MI-GAN CPU/F32 graph (reviewed256, experimental512).
// Overlap composition is a new wrapper, not original ONNX-wrapper parity.
#include "inpaint.h"
#include "migan.h"
#include "core/env_gate.h"
#include "core/sha256_accelerated.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <vector>

namespace inpaint {

struct context {
    migan::context * model = nullptr;
    int resolution;
    std::vector<float> input;
    std::vector<float> raw;
    explicit context(int side) : resolution(side), input(size_t(side) * side * 4), raw(size_t(side) * side * 3) {}
    ~context() { migan::free(model); }
};

namespace {

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool enabled(int resolution, std::string & error) {
    if (!core_env::on("CRISPEMBED_MIGAN")) return fail(error, "MI-GAN requires CRISPEMBED_MIGAN=1");
    return resolution != 512 || core_env::on("CRISPEMBED_MIGAN512") ||
           fail(error, "MI-GAN512 additionally requires CRISPEMBED_MIGAN512=1");
}

bool valid_digest(const char * value) {
    if (!value || std::strlen(value) != 64) return false;
    return std::all_of(value, value + 64, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::vector<int> starts(int extent, int tile_size, int tile_stride) {
    const int last = std::max(0, extent - tile_size);
    std::vector<int> result;
    for (int at = 0;; at = std::min(at + tile_stride, last)) {
        result.push_back(at);
        if (at == last) return result;
    }
}

bool run_tile(void * opaque, const uint8_t * rgb, const uint8_t * mask, uint8_t * output, std::string & error) {
    auto * ctx = static_cast<context *>(opaque);
    const size_t pixels = size_t(ctx->resolution) * ctx->resolution;
    return migan::preprocess_rgb8_variant(ctx->resolution, rgb, 3 * pixels, mask, pixels, ctx->resolution,
                                          ctx->resolution, ctx->input.data(), ctx->input.size(), error) &&
           migan::run(ctx->model, ctx->input.data(), ctx->input.size(), ctx->raw.data(), ctx->raw.size(), error) &&
           migan::composite_rgb8_variant(ctx->resolution, ctx->raw.data(), ctx->raw.size(), rgb, 3 * pixels, mask,
                                         pixels, output, 3 * pixels, error);
}

struct weighted_pixel {
    uint32_t rgb[3] = {};
    uint32_t weight = 0;
};

} // namespace

context * init(const char * path, int n_threads, std::string & error) {
    return init_variant(path, checkpoint_sha256, model_sha256, tile_size, n_threads, error);
}

context * init_variant(const char * path, const char * expected_checkpoint_sha256, const char * expected_gguf_sha256,
                       int resolution, int n_threads, std::string & error) {
    error.clear();
    if (resolution != 256 && resolution != 512) {
        fail(error, "MI-GAN supports only fixed256 or fixed512");
        return nullptr;
    }
    if (!enabled(resolution, error)) return nullptr;
    if (!path || !*path || n_threads < 1 || n_threads > 64) {
        fail(error, "inpainting requires a model path and 1..64 CPU threads");
        return nullptr;
    }
    if (!valid_digest(expected_checkpoint_sha256) || !valid_digest(expected_gguf_sha256)) {
        fail(error, "MI-GAN requires independently reviewed lowercase checkpoint and GGUF SHA-256 values");
        return nullptr;
    }
    if (resolution == 256 && (std::strcmp(expected_checkpoint_sha256, checkpoint_sha256) != 0 ||
                              std::strcmp(expected_gguf_sha256, model_sha256) != 0)) {
        fail(error, "MI-GAN256 retains the exact reviewed checkpoint and GGUF pins");
        return nullptr;
    }
    try {
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(path, ec);
        if (ec) {
            fail(error, "cannot read inpainting model file");
            return nullptr;
        }
        const bool profile = core_env::on("CRISPEMBED_INPAINT_PROFILE");
        core_sha256_accel::Stats hash_stats;
        const bool size_ok = (resolution != 256 || size == model_bytes) && size > 0 && size <= size_t(64) * 1024 * 1024;
        const std::string observed =
            size_ok ? core_sha256_accel::file(path, core_env::on("CRISPEMBED_INPAINT_SHA256_ACCEL"),
                                              profile ? &hash_stats : nullptr)
                    : std::string();
        if (!size_ok || observed != expected_gguf_sha256) {
            fail(error, "inpainting model differs from the selected MI-GAN F32 GGUF (size/SHA-256 mismatch)");
            return nullptr;
        }
        if (profile) core_sha256_accel::profile_verified("migan", hash_stats, observed);
        std::unique_ptr<context> ctx(new context(resolution));
        ctx->model = migan::init_variant(path, expected_checkpoint_sha256, resolution, n_threads, error);
        if (!ctx->model) return nullptr;
        return ctx.release();
    } catch (const std::exception & exception) {
        fail(error, std::string("inpainting model initialization failed: ") + exception.what());
        return nullptr;
    }
}

int image_resolution(const context * ctx) {
    return ctx ? ctx->resolution : 0;
}

void free(context * ctx) {
    delete ctx;
}

bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error) {
    error.clear();
    if (!enabled(ctx ? ctx->resolution : 256, error)) return false;
    if (!ctx || !ctx->model) return fail(error, "inpainting requires an initialized model context");
    return detail::process_tiles_rgb8_variant(ctx->resolution, run_tile, ctx, rgb, rgb_bytes, fill_mask, mask_bytes,
                                              width, height, output_rgb, output_bytes, error);
}

namespace detail {

bool process_tiles_rgb8(tile_processor process_tile, void * opaque, const uint8_t * rgb, size_t rgb_bytes,
                        const uint8_t * fill_mask, size_t mask_bytes, int width, int height, uint8_t * output_rgb,
                        size_t output_bytes, std::string & error) {
    return process_tiles_rgb8_variant(tile_size, process_tile, opaque, rgb, rgb_bytes, fill_mask, mask_bytes, width,
                                      height, output_rgb, output_bytes, error);
}

bool process_tiles_rgb8_variant(int resolution, tile_processor process_tile, void * opaque, const uint8_t * rgb,
                                size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes, int width, int height,
                                uint8_t * output_rgb, size_t output_bytes, std::string & error) {
    error.clear();
    if (resolution != 256 && resolution != 512) return fail(error, "MI-GAN supports only fixed256 or fixed512");
    const int tile_size = resolution, tile_stride = resolution * 3 / 4;
    const size_t tile_pixels = size_t(tile_size) * tile_size;
    if (width < 1 || height < 1 || width > max_dimension || height > max_dimension ||
        size_t(width) > max_pixels / size_t(height))
        return fail(error, "inpainting dimensions must be positive, at most 32768 per side and 64M pixels");
    const size_t pixels = size_t(width) * size_t(height);
    if (!process_tile || !rgb || !fill_mask || !output_rgb || rgb_bytes != pixels * 3 || mask_bytes != pixels ||
        output_bytes != pixels * 3)
        return fail(error, "inpainting requires exact RGB8 and same-size mask/output byte counts");

    bool has_fill = false;
    for (size_t i = 0; i < pixels; ++i) {
        if (fill_mask[i] != 0 && fill_mask[i] != 255)
            return fail(error, "inpainting mask must be binary (0=keep, 255=fill)");
        has_fill |= fill_mask[i] != 0;
    }
    if (!has_fill) {
        std::memmove(output_rgb, rgb, rgb_bytes);
        return true;
    }

    try {
        // Transactional result also makes in-place calls safe: all tiles always
        // see the original frame, including when another tile has filled it.
        std::vector<uint8_t> result(rgb, rgb + rgb_bytes);
        std::vector<uint8_t> tile_rgb(3 * tile_pixels), tile_mask(tile_pixels);
        std::vector<uint8_t> tile_output(3 * tile_pixels);
        const std::vector<int> xs = starts(width, tile_size, tile_stride), ys = starts(height, tile_size, tile_stride);
        const int band_height = std::min(height, tile_size);
        std::vector<weighted_pixel> band(size_t(width) * band_height);
        int flushed = 0;

        for (size_t yi = 0; yi < ys.size(); ++yi) {
            const int top = ys[yi], actual_height = std::min(tile_size, height - top);
            for (int left : xs) {
                const int actual_width = std::min(tile_size, width - left);
                bool tile_has_fill = false;
                for (int y = 0; y < actual_height && !tile_has_fill; ++y)
                    for (int x = 0; x < actual_width; ++x)
                        if (fill_mask[size_t(top + y) * width + left + x]) {
                            tile_has_fill = true;
                            break;
                        }
                if (!tile_has_fill) continue;

                // Clamp padding only when an image dimension is below the model resolution.
                // Every generated tile reads original pixels, never previous fills.
                for (int y = 0; y < tile_size; ++y)
                    for (int x = 0; x < tile_size; ++x) {
                        const size_t source =
                            size_t(std::min(top + y, height - 1)) * width + std::min(left + x, width - 1);
                        const size_t target = size_t(y) * tile_size + x;
                        std::memcpy(tile_rgb.data() + 3 * target, rgb + 3 * source, 3);
                        tile_mask[target] = fill_mask[source];
                    }
                if (!process_tile(opaque, tile_rgb.data(), tile_mask.data(), tile_output.data(), error)) {
                    const std::string detail = error.empty() ? "tile processor failed" : error;
                    return fail(error, "inpainting tile (" + std::to_string(left) + "," + std::to_string(top) +
                                           "): " + detail);
                }

                for (int y = 0; y < actual_height; ++y) {
                    const uint32_t wy = uint32_t(std::min(y + 1, tile_size - y));
                    for (int x = 0; x < actual_width; ++x) {
                        const size_t source = size_t(top + y) * width + left + x;
                        if (!fill_mask[source]) continue;
                        const uint32_t weight = wy * uint32_t(std::min(x + 1, tile_size - x));
                        weighted_pixel & sum = band[size_t((top + y) % band_height) * width + left + x];
                        const size_t local = size_t(y) * tile_size + x;
                        // At most3x3 overlapping tiles, each weight <=256^2:
                        // every RGB sum <=9*256^2*255, safely within uint32.
                        for (int c = 0; c < 3; ++c) sum.rgb[c] += weight * tile_output[3 * local + c];
                        sum.weight += weight;
                    }
                }
            }

            // No later tile touches rows before the next origin. Recycle them
            // in a tile-height ring rather than allocating frame-sized float planes.
            const int flush_end = yi + 1 < ys.size() ? ys[yi + 1] : height;
            for (; flushed < flush_end; ++flushed) {
                weighted_pixel * row = band.data() + size_t(flushed % band_height) * width;
                for (int x = 0; x < width; ++x) {
                    const size_t pixel = size_t(flushed) * width + x;
                    const weighted_pixel & sum = row[x];
                    if (fill_mask[pixel]) {
                        if (!sum.weight) return fail(error, "inpainting tile coverage left an unfilled mask pixel");
                        for (int c = 0; c < 3; ++c)
                            result[3 * pixel + c] = uint8_t((sum.rgb[c] + sum.weight / 2) / sum.weight);
                    }
                    row[x] = weighted_pixel{};
                }
            }
        }
        std::memmove(output_rgb, result.data(), result.size());
        return true;
    } catch (const std::exception & exception) {
        return fail(error, std::string("inpainting image processing failed: ") + exception.what());
    }
}

} // namespace detail
} // namespace inpaint
