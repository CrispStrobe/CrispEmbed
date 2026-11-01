#include "lama.h"
#include "lama_fft.h"
#include "lama_schema.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace lama {
using latent_inpaint::Engine;
using latent_inpaint::Tensor;
namespace {
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error("Big-LaMa: " + message);
}
void enabled() {
    require(core_env::on("CRISPEMBED_LAMA"), "experimental CPU/F32 inference requires CRISPEMBED_LAMA=1");
}
bool digest_valid(const std::string & digest) {
    return digest.size() == 64 && std::all_of(digest.begin(), digest.end(),
                                              [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void dimensions(int width, int height) {
    require(width >= min_dimension && height >= min_dimension && width <= max_dimension && height <= max_dimension &&
                size_t(width) <= max_pixels / size_t(height),
            "image dimensions outside the supported9..1024 profile");
}
void buffers(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width, int height) {
    dimensions(width, height);
    const size_t pixels = size_t(width) * height;
    require(rgb && mask && rgb_bytes == pixels * 3 && mask_bytes == pixels, "RGB/mask buffer size differs");
    require(std::all_of(mask, mask + mask_bytes, [](uint8_t value) { return value == 0 || value == 255; }),
            "mask must be binary0/255, white=fill");
}
int rounded_eight(int value) {
    return (value + 7) / 8 * 8;
}
int symmetric_index(int index, int size) {
    const int phase = index % (2 * size);
    return phase < size ? phase : 2 * size - 1 - phase;
}
int reflect_index(int index, int size) {
    const int period = 2 * (size - 1);
    int phase = index % period;
    if (phase < 0) phase += period;
    return phase < size ? phase : period - phase;
}
Tensor relu(Tensor value) {
    for (float & scalar : value.v)
        if (scalar < 0) scalar = 0;
    return value;
}

void validate_model(const std::string & path, const std::string & expected_sha256) {
    require(digest_valid(expected_sha256), "a lowercase GGUF SHA-256 from the converter receipt is required");
    require(std::filesystem::is_regular_file(path), "model must be a regular GGUF file");
    const uintmax_t bytes = std::filesystem::file_size(path);
    require(bytes >= 32 && bytes <= UINT64_C(512) * 1024 * 1024, "model exceeds the bounded512MiB weight profile");
    const auto observed = core_sha256::file(path);
    require(!observed.empty() && observed == expected_sha256, "GGUF digest differs from supplied converter receipt");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path.c_str()),
                                                                            &core_gguf::free_metadata);
    require(bool(meta), "cannot read GGUF metadata");
    auto integer = [&](const char * key, uint32_t expected) {
        const int64_t id = gguf_find_key(meta.get(), key);
        return id >= 0 && gguf_get_kv_type(meta.get(), id) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(meta.get(), id) == expected;
    };
    auto string = [&](const char * key, const char * expected) {
        return core_gguf::kv_str(meta.get(), key, "") == expected;
    };
    const int64_t lfu = gguf_find_key(meta.get(), "lama.lfu");
    require(string("general.architecture", "big_lama") && integer("general.file_type", 0) &&
                string("general.license", "Apache-2.0") && integer("lama.schema", 1) &&
                string("lama.source.commit", detail::source_commit()) &&
                string("lama.contract.sha256", detail::source_contract_sha256()) &&
                string("lama.checkpoint.sha256", detail::checkpoint_sha256()) &&
                string("lama.checkpoint.revision", detail::checkpoint_revision()) &&
                string("lama.checkpoint.container", detail::checkpoint_container()) &&
                string("lama.checkpoint.member", detail::checkpoint_member()) &&
                string("lama.input.contract", detail::input_contract()) &&
                integer("lama.image.min_dimension", min_dimension) &&
                integer("lama.image.max_dimension", max_dimension) &&
                string("lama.weights.license.status", detail::weight_license_status()) &&
                string("lama.tensor.layout", detail::tensor_layout()) && integer("lama.generator.blocks", 18) &&
                string("lama.fft.norm", "ortho") && lfu >= 0 && gguf_get_kv_type(meta.get(), lfu) == GGUF_TYPE_BOOL &&
                !gguf_get_val_bool(meta.get(), lfu),
            "source, checkpoint, licence or generator contract differs");
}
} // namespace

struct context {
    std::unique_ptr<Engine> engine;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
#ifdef CRISPEMBED_LAMA_TESTING
    Observer observer = nullptr;
    void * observer_opaque = nullptr;
#endif
    void mark(const std::string & name, const Tensor & value) const {
#ifdef CRISPEMBED_LAMA_TESTING
        if (observer) observer(observer_opaque, name, value);
#else
        (void)name;
        (void)value;
#endif
    }
};

namespace detail {
Tensor reflect_pad(const Tensor & input, int padding) {
    require(latent_inpaint::finite(input) && padding >= 0 && padding <= 3 && input.h > padding && input.w > padding,
            "invalid reflection-padding input or extent");
    if (padding == 0) return input;
    Tensor result(input.c, input.h + 2 * padding, input.w + 2 * padding);
    for (int ch = 0; ch < input.c; ++ch)
        for (int y = 0; y < result.h; ++y)
            for (int x = 0; x < result.w; ++x)
                result.at(ch, y, x) =
                    input.at(ch, reflect_index(y - padding, input.h), reflect_index(x - padding, input.w));
    return result;
}

Tensor preprocess_rgb8(const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes, int width,
                       int height) {
    buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
    Tensor result(4, rounded_eight(height), rounded_eight(width));
    for (int y = 0; y < result.h; ++y)
        for (int x = 0; x < result.w; ++x) {
            const size_t from = size_t(symmetric_index(y, height)) * width + symmetric_index(x, width);
            const float fill = mask[from] == 255 ? 1.0f : 0.0f;
            for (int ch = 0; ch < 3; ++ch) result.at(ch, y, x) = (float(rgb[from * 3 + ch]) / 255.0f) * (1.0f - fill);
            result.at(3, y, x) = fill;
        }
    return result;
}

void composite_rgb8(const Tensor & prediction, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask,
                    size_t mask_bytes, int width, int height, uint8_t * output, size_t output_bytes) {
    buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
    require(output && output_bytes == rgb_bytes && prediction.c == 3 && prediction.h == rounded_eight(height) &&
                prediction.w == rounded_eight(width) && latent_inpaint::finite(prediction),
            "invalid predicted RGB tensor or output buffer");
    std::vector<uint8_t> result(rgb, rgb + rgb_bytes);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t pixel = size_t(y) * width + x;
            if (mask[pixel])
                for (int ch = 0; ch < 3; ++ch)
                    result[pixel * 3 + ch] = uint8_t(std::clamp(prediction.at(ch, y, x), 0.0f, 1.0f) * 255.0f);
        }
    std::memcpy(output, result.data(), result.size());
}
} // namespace detail

namespace {
struct Features {
    Tensor local, global;
};

Tensor reflected_conv(Engine & engine, const std::string & name, const Tensor & input, int stride, int padding) {
    return engine.conv(name, padding ? detail::reflect_pad(input, padding) : input, stride, 0);
}

Tensor spectral(context & model, const std::string & name, const Tensor & input) {
    Engine & engine = *model.engine;
    Tensor x = relu(engine.batch_norm(name + ".conv1.1", engine.conv(name + ".conv1.0", input)));
    Tensor packed = detail::rfft2_ortho(x);
    model.mark(name + ".fu.rfft", packed);
    packed = relu(engine.batch_norm(name + ".fu.bn", engine.conv(name + ".fu.conv_layer", packed)));
    model.mark(name + ".fu.spectral", packed);
    Tensor frequency = detail::irfft2_ortho(packed, x.w);
    model.mark(name + ".fu.irfft", frequency);
    // The pinned Big-LaMa config disables LFU and all spatial rescaling.
    return engine.conv(name + ".conv2", latent_inpaint::add(x, frequency));
}

Features ffc(context & model, const std::string & name, const Features & input, int stride, int padding) {
    Engine & engine = *model.engine;
    const std::string prefix = name + ".ffc";
    Features result;
    result.local = reflected_conv(engine, prefix + ".convl2l", input.local, stride, padding);
    if (input.global.c) {
        result.local = latent_inpaint::add(result.local,
                                           reflected_conv(engine, prefix + ".convg2l", input.global, stride, padding));
    }
    if (engine.has(prefix + ".convl2g.weight")) {
        result.global = reflected_conv(engine, prefix + ".convl2g", input.local, stride, padding);
        if (input.global.c) {
            // Only residual blocks contain global inputs in this fixed graph;
            // their stride is one, so SpectralTransform has no AvgPool branch.
            require(stride == 1, "unsupported global downsampling outside the pinned architecture");
            result.global = latent_inpaint::add(result.global, spectral(model, prefix + ".convg2g", input.global));
        }
    }
    result.local = relu(engine.batch_norm(name + ".bn_l", result.local));
    if (result.global.c) result.global = relu(engine.batch_norm(name + ".bn_g", result.global));
    model.mark(name + ".local", result.local);
    if (result.global.c) model.mark(name + ".global", result.global);
    return result;
}

Tensor forward(context & model, const Tensor & input) {
    enabled();
    require(input.c == 4 && input.h >= 16 && input.w >= 16 && input.h <= max_dimension && input.w <= max_dimension &&
                input.h % 8 == 0 && input.w % 8 == 0 && latent_inpaint::finite(input),
            "invalid padded generator input");
    for (size_t p = 0; p < input.spatial(); ++p) {
        const float mask = input.v[3 * input.spatial() + p];
        require(mask == 0 || mask == 1, "generator mask must be binary0/1");
        for (int ch = 0; ch < 3; ++ch) {
            const float value = input.v[size_t(ch) * input.spatial() + p];
            require(value >= 0 && value <= 1 && (!mask || value == 0), "generator requires masked RGB in[0,1]");
        }
    }
    model.mark("input", input);
    Features features = ffc(model, "generator.model.1", { detail::reflect_pad(input, 3), {} }, 1, 0);
    for (int block = 2; block <= 4; ++block)
        features = ffc(model, "generator.model." + std::to_string(block), features, 2, 1);
    for (int block = 5; block <= 22; ++block) {
        const std::string name = "generator.model." + std::to_string(block);
        Features residual = ffc(model, name + ".conv1", features, 1, 1);
        residual = ffc(model, name + ".conv2", residual, 1, 1);
        features.local = latent_inpaint::add(features.local, residual.local);
        features.global = latent_inpaint::add(features.global, residual.global);
        model.mark(name + ".local", features.local);
        model.mark(name + ".global", features.global);
    }
    Tensor x = latent_inpaint::concat(features.local, features.global);
    model.mark("generator.model.23", x);
    for (int block : { 24, 27, 30 }) {
        x = model.engine->conv_transpose("generator.model." + std::to_string(block), x, 2, 1, 1);
        x = relu(model.engine->batch_norm("generator.model." + std::to_string(block + 1), x));
        model.mark("generator.model." + std::to_string(block + 2), x);
    }
    x = model.engine->conv("generator.model.34", detail::reflect_pad(x, 3));
    for (float & value : x.v) value = 1.0f / (1.0f + std::exp(-value));
    require(latent_inpaint::finite(x), "nonfinite generated RGB");
    model.mark("prediction", x);
    return x;
}

struct Busy {
    context & model;
    explicit Busy(context & value) : model(value) {
        require(!model.busy.test_and_set(), "one context cannot process concurrent requests");
    }
    ~Busy() { model.busy.clear(); }
};
} // namespace

context * init(const std::string & path, const std::string & expected_gguf_sha256, int threads, std::string & error) {
    error.clear();
    try {
        enabled();
        require(threads >= 1 && threads <= 64, "thread count must be in[1,64]");
        validate_model(path, expected_gguf_sha256);
        auto model = std::make_unique<context>();
        model->engine = std::make_unique<Engine>(path.c_str(), threads, detail::schema());
        return model.release();
    } catch (const std::exception & exception) {
        error = exception.what();
        return nullptr;
    }
}
void free(context * model) {
    delete model;
}

bool process_rgb8(context * model, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    error.clear();
    try {
        require(model != nullptr, "null context");
        enabled();
        Busy lock(*model);
        buffers(rgb, rgb_bytes, mask, mask_bytes, width, height);
        require(output && output_bytes == rgb_bytes, "output byte count differs");
        if (std::none_of(mask, mask + mask_bytes, [](uint8_t value) { return value != 0; })) {
            std::memmove(output, rgb, rgb_bytes);
            return true;
        }
        Tensor input = detail::preprocess_rgb8(rgb, rgb_bytes, mask, mask_bytes, width, height);
        Tensor prediction = forward(*model, input);
        detail::composite_rgb8(prediction, rgb, rgb_bytes, mask, mask_bytes, width, height, output, output_bytes);
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}

#ifdef CRISPEMBED_LAMA_TESTING
void set_observer(context * model, Observer callback, void * opaque) {
    require(model != nullptr, "null reference context");
    model->observer = callback;
    model->observer_opaque = opaque;
}
bool run_reference(context * model, const Tensor & input, Tensor & prediction, std::string & error) {
    error.clear();
    try {
        require(model != nullptr, "null reference context");
        Busy lock(*model);
        Tensor result = forward(*model, input);
        prediction = std::move(result);
        return true;
    } catch (const std::exception & exception) {
        error = exception.what();
        return false;
    }
}
#endif
} // namespace lama
