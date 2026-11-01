// Still-image geometry around the unchanged, reviewed MI-GAN CPU/F32 graph.
// Overlap composition is a new wrapper, not original ONNX-wrapper parity.
#include "inpaint.h"
#include "migan.h"
#include "core/env_gate.h"
#include "core/sha256.h"
#include "core/gpu_backend_pref.h"

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
    std::vector<float> input = std::vector<float>(migan::input_elements);
    std::vector<float> raw = std::vector<float>(migan::rgb_elements);
    ~context() { migan::free(model); }
};

namespace detail {
namespace {
struct batch_adapter {
    tile_processor single;
    void * opaque;
};
bool adapt_single(void * opaque, size_t count, const uint8_t * rgb, const uint8_t * mask, uint8_t * output,
                  std::string & error) {
    auto * adapter = static_cast<batch_adapter *>(opaque);
    for (size_t i = 0; i < count; ++i)
        if (!adapter->single(adapter->opaque, rgb + i * migan::rgb_elements, mask + i * migan::mask_elements,
                             output + i * migan::rgb_elements, error))
            return false;
    return true;
}
} // namespace
} // namespace detail

namespace {

bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}

bool enabled(std::string & error) {
    return core_env::on("CRISPEMBED_MIGAN") || fail(error, "MI-GAN requires CRISPEMBED_MIGAN=1");
}

std::vector<int> starts(int extent) {
    const int last = std::max(0, extent - tile_size);
    std::vector<int> result;
    for (int at = 0;; at = std::min(at + tile_stride, last)) {
        result.push_back(at);
        if (at == last) return result;
    }
}

bool run_tile(void * opaque, const uint8_t * rgb, const uint8_t * mask, uint8_t * output, std::string & error) {
    auto * ctx = static_cast<context *>(opaque);
    return migan::preprocess_rgb8(rgb, migan::rgb_elements, mask, migan::mask_elements, tile_size, tile_size,
                                  ctx->input.data(), ctx->input.size(), error) &&
           migan::run(ctx->model, ctx->input.data(), ctx->input.size(), ctx->raw.data(), ctx->raw.size(), error) &&
           migan::composite_rgb8(ctx->raw.data(), ctx->raw.size(), rgb, migan::rgb_elements, mask, migan::mask_elements,
                                 output, migan::rgb_elements, error);
}

// Several tiles per graph execution (GPU). Preprocess/composite are the same
// pure helpers as run_tile; only the generator call is batched.
bool run_tiles(void * opaque, size_t count, const uint8_t * rgb, const uint8_t * mask, uint8_t * output,
               std::string & error) {
    auto * ctx = static_cast<context *>(opaque);
    if (count == 1) return run_tile(opaque, rgb, mask, output, error);
    ctx->input.resize(count * migan::input_elements);
    ctx->raw.resize(count * migan::rgb_elements);
    for (size_t i = 0; i < count; ++i)
        if (!migan::preprocess_rgb8(rgb + i * migan::rgb_elements, migan::rgb_elements, mask + i * migan::mask_elements,
                                    migan::mask_elements, tile_size, tile_size,
                                    ctx->input.data() + i * migan::input_elements, migan::input_elements, error))
            return false;
    if (!migan::run_batch(ctx->model, ctx->input.data(), count, ctx->raw.data(), error)) return false;
    for (size_t i = 0; i < count; ++i)
        if (!migan::composite_rgb8(ctx->raw.data() + i * migan::rgb_elements, migan::rgb_elements,
                                   rgb + i * migan::rgb_elements, migan::rgb_elements, mask + i * migan::mask_elements,
                                   migan::mask_elements, output + i * migan::rgb_elements, migan::rgb_elements, error))
            return false;
    return true;
}

struct weighted_pixel {
    uint32_t rgb[3] = {};
    uint32_t weight = 0;
};

} // namespace

context * init(const char * path, int n_threads, std::string & error) {
    error.clear();
    if (!enabled(error)) return nullptr;
    if (!path || !*path || n_threads < 1 || n_threads > 64) {
        fail(error, "inpainting requires a model path and 1..64 CPU threads");
        return nullptr;
    }
    try {
        std::error_code ec;
        const uintmax_t size = std::filesystem::file_size(path, ec);
        if (ec) {
            fail(error, "cannot read inpainting model file");
            return nullptr;
        }
        const bool reviewed = size == model_bytes && core_sha256::file(path) == model_sha256;
        // A derived F16/Q8_0/Q4_K container (CRISPEMBED_MIGAN_QUANT=1) is not
        // byte-pinned here: migan validates its policy, every tensor type/
        // shape and its recorded source = the reviewed F32 GGUF SHA.
        if (!reviewed && !core_env::on("CRISPEMBED_MIGAN_QUANT")) {
            fail(error, "inpainting model differs from the reviewed MI-GAN fixed256 F32 GGUF (SHA-256 mismatch)");
            return nullptr;
        }
        std::unique_ptr<context> ctx(new context);
        ctx->model =
            migan::init_ex(path, checkpoint_sha256, n_threads, migan::device_choice::automatic, model_sha256, error);
        if (!ctx->model) return nullptr;
        return ctx.release();
    } catch (const std::exception & exception) {
        fail(error, std::string("inpainting model initialization failed: ") + exception.what());
        return nullptr;
    }
}

void free(context * ctx) {
    delete ctx;
}

bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                  int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error) {
    error.clear();
    if (!enabled(error)) return false;
    if (!ctx || !ctx->model) return fail(error, "inpainting requires an initialized model context");
    const int batch = migan::max_batch(ctx->model);
    if (batch <= 1)
        return detail::process_tiles_rgb8(run_tile, ctx, rgb, rgb_bytes, fill_mask, mask_bytes, width, height,
                                          output_rgb, output_bytes, error);
    return detail::process_tiles_rgb8_batched(run_tiles, ctx, size_t(batch), rgb, rgb_bytes, fill_mask, mask_bytes,
                                              width, height, output_rgb, output_bytes, error);
}

void set_gpu_backend(const char * name) {
    crispasr_set_gpu_backend_pref(name);
}

const char * backend_name(const context * ctx) {
    return ctx ? migan::backend_name(ctx->model) : "";
}

const char * weight_precision(const context * ctx) {
    return ctx ? migan::weight_precision(ctx->model) : "";
}

int max_batch(const context * ctx) {
    return ctx ? migan::max_batch(ctx->model) : 0;
}

namespace detail {

bool process_tiles_rgb8(tile_processor process_tile, void * opaque, const uint8_t * rgb, size_t rgb_bytes,
                        const uint8_t * fill_mask, size_t mask_bytes, int width, int height, uint8_t * output_rgb,
                        size_t output_bytes, std::string & error) {
    batch_adapter adapter{ process_tile, opaque };
    return process_tiles_rgb8_batched(process_tile ? adapt_single : nullptr, &adapter, 1, rgb, rgb_bytes, fill_mask,
                                      mask_bytes, width, height, output_rgb, output_bytes, error);
}

bool process_tiles_rgb8_batched(tile_batch_processor process_tiles, void * opaque, size_t max_batch,
                                const uint8_t * rgb, size_t rgb_bytes, const uint8_t * fill_mask, size_t mask_bytes,
                                int width, int height, uint8_t * output_rgb, size_t output_bytes, std::string & error) {
    error.clear();
    if (max_batch < 1 || max_batch > 64) return fail(error, "inpainting batch must be 1..64 tiles");
    tile_batch_processor process_tile = process_tiles;
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
        std::vector<uint8_t> tile_rgb(max_batch * migan::rgb_elements), tile_mask(max_batch * migan::mask_elements);
        std::vector<uint8_t> tile_output(max_batch * migan::rgb_elements);
        const std::vector<int> xs = starts(width), ys = starts(height);
        const int band_height = std::min(height, tile_size);
        std::vector<weighted_pixel> band(size_t(width) * band_height);
        int flushed = 0;

        for (size_t yi = 0; yi < ys.size(); ++yi) {
            const int top = ys[yi], actual_height = std::min(tile_size, height - top);
            std::vector<int> lefts;
            for (int left : xs) {
                const int actual_width = std::min(tile_size, width - left);
                bool tile_has_fill = false;
                for (int y = 0; y < actual_height && !tile_has_fill; ++y)
                    for (int x = 0; x < actual_width; ++x)
                        if (fill_mask[size_t(top + y) * width + left + x]) {
                            tile_has_fill = true;
                            break;
                        }
                if (tile_has_fill) lefts.push_back(left);
            }

            // Tiles of one band row are independent (all read the original
            // frame); integer weighted sums make accumulation order-neutral.
            for (size_t first = 0; first < lefts.size(); first += max_batch) {
                const size_t count = std::min(max_batch, lefts.size() - first);
                for (size_t k = 0; k < count; ++k) {
                    const int left = lefts[first + k];
                    uint8_t * trgb = tile_rgb.data() + k * migan::rgb_elements;
                    uint8_t * tmask = tile_mask.data() + k * migan::mask_elements;
                    // Clamp padding only when an image dimension is below256.
                    // Every generated tile reads original pixels, never previous fills.
                    for (int y = 0; y < tile_size; ++y)
                        for (int x = 0; x < tile_size; ++x) {
                            const size_t source =
                                size_t(std::min(top + y, height - 1)) * width + std::min(left + x, width - 1);
                            const size_t target = size_t(y) * tile_size + x;
                            std::memcpy(trgb + 3 * target, rgb + 3 * source, 3);
                            tmask[target] = fill_mask[source];
                        }
                }
                if (!process_tile(opaque, count, tile_rgb.data(), tile_mask.data(), tile_output.data(), error)) {
                    const std::string detail = error.empty() ? "tile processor failed" : error;
                    return fail(error, "inpainting tile (" + std::to_string(lefts[first]) + "," + std::to_string(top) +
                                           "): " + detail);
                }

                for (size_t k = 0; k < count; ++k) {
                    const int left = lefts[first + k];
                    const int actual_width = std::min(tile_size, width - left);
                    const uint8_t * tout = tile_output.data() + k * migan::rgb_elements;
                    for (int y = 0; y < actual_height; ++y) {
                        const uint32_t wy = uint32_t(std::min(y + 1, tile_size - y));
                        for (int x = 0; x < actual_width; ++x) {
                            const size_t source = size_t(top + y) * width + left + x;
                            if (!fill_mask[source]) continue;
                            const uint32_t weight = wy * uint32_t(std::min(x + 1, tile_size - x));
                            weighted_pixel & sum = band[size_t((top + y) % band_height) * width + left + x];
                            const size_t local = size_t(y) * tile_size + x;
                            // At most3x3 overlapping tiles, each weight <=128^2:
                            // every RGB sum <=9*128^2*255, safely within uint32.
                            for (int c = 0; c < 3; ++c) sum.rgb[c] += weight * tout[3 * local + c];
                            sum.weight += weight;
                        }
                    }
                }
            }

            // No later tile touches rows before the next origin. Recycle them
            // in a256-row ring rather than allocating frame-sized float planes.
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
