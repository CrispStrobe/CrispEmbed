#include "latent_inpaint.h"
#include "latent_inpaint_schema.h"
#include "latent_inpaint_unet.h"
#include "latent_inpaint_vae.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/sha256.h"
#include "inpaint.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace latent_inpaint {
namespace {
bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error("latent inpaint: " + message);
}
bool digest_valid(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(),
                                             [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void enabled(const std::string & model) {
    require(model == "moebius" || model == "pixelhacker", "unknown model profile");
    const char * gate = model == "moebius" ? "CRISPEMBED_MOEBIUS" : "CRISPEMBED_PIXELHACKER";
    require(core_env::on(gate), std::string("experimental CPU/F32 inference requires ") + gate + "=1");
}
void validate_model(const std::string & path, const std::string & sha256, const std::string & model) {
    require(digest_valid(sha256), "a lowercase SHA-256 from the converter receipt is required for " + model);
    require(std::filesystem::is_regular_file(path), model + " must be a regular GGUF file");
    const uintmax_t bytes = std::filesystem::file_size(path);
    require(bytes >= 32 && bytes <= UINT64_C(4) * 1024 * 1024 * 1024, model + " file exceeds the bounded4GiB profile");
    const std::string observed = core_sha256::file(path);
    require(!observed.empty(), "cannot hash " + model + " GGUF");
    require(observed == sha256, model + " GGUF digest differs from supplied converter receipt");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path.c_str()),
                                                                            &core_gguf::free_metadata);
    require(bool(meta), "cannot open model metadata");
    const std::string component = model == "sdxl_vae" ? "vae" : "denoiser";
    auto u32_is = [&](const char * key, uint32_t expected) {
        const int64_t index = gguf_find_key(meta.get(), key);
        return index >= 0 && gguf_get_kv_type(meta.get(), index) == GGUF_TYPE_UINT32 &&
               gguf_get_val_u32(meta.get(), index) == expected;
    };
    require(core_gguf::kv_str(meta.get(), "general.architecture", "") == model && u32_is("general.file_type", 0) &&
                core_gguf::kv_str(meta.get(), "general.license", "") == license(model) &&
                u32_is("latent_inpaint.schema", 1) &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.component", "") == component &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.model", "") == model &&
                u32_is("latent_inpaint.image_size", 512) &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.contract.sha256", "") == source_contract_sha256() &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.source.commit", "") == source_commit(model) &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.checkpoint.sha256", "") == checkpoint_sha256(model) &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.checkpoint.revision", "") == checkpoint_revision(model) &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.input.contract", "") == input_contract() &&
                core_gguf::kv_str(meta.get(), "latent_inpaint.tensor.layout", "") == tensor_layout(),
            model + " source, checkpoint, licence or layout contract differs");
}

// SplitMix64 plus Box-Muller is specified here rather than relying on the
// implementation-defined sequence of std::normal_distribution. It is an
// independent native seed stream; parity replays use recorded Torch noise.
class NoiseGenerator {
    uint64_t state_;
    bool spare_valid_ = false;
    float spare_ = 0;
    double uniform() {
        uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        z ^= z >> 31;
        return (double(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

public:
    explicit NoiseGenerator(uint64_t seed) : state_(seed) {}
    float normal() {
        if (spare_valid_) {
            spare_valid_ = false;
            return spare_;
        }
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double theta = 6.2831853071795864769 * uniform();
        spare_ = float(radius * std::sin(theta));
        spare_valid_ = true;
        return float(radius * std::cos(theta));
    }
    Tensor tensor(int c, int h, int w) {
        Tensor result(c, h, w);
        for (float & value : result.v) value = normal();
        return result;
    }
};
struct Noise {
    Tensor posterior_image, posterior_masked, latent_noise, offset_noise;
};
Noise make_noise(uint64_t seed) {
    NoiseGenerator rng(seed);
    return { rng.tensor(4, 64, 64), rng.tensor(4, 64, 64), rng.tensor(4, 64, 64), rng.tensor(4, 1, 1) };
}
} // namespace

struct context {
    Options options;
    Variant variant = Variant::moebius;
    std::unique_ptr<Engine> denoiser, vae;
    std::vector<float> alphas;
    std::vector<int> timesteps;
    std::atomic_flag busy = ATOMIC_FLAG_INIT;
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
    Observer observer = nullptr;
    void * observer_opaque = nullptr;
    std::string stage_prefix;
#endif
    void mark(const std::string & name, const Tensor & tensor) {
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        if (observer) observer(observer_opaque, name, tensor);
#else
        (void)name;
        (void)tensor;
#endif
    }
};

namespace detail {
std::vector<int> ddim_timesteps(int steps, const std::string & model) {
    require(steps >= 2 && steps <= 100, "sampling steps must be in[2,100]");
    require(model == "moebius" || model == "pixelhacker", "unknown schedule profile");
    const double strength = model == "moebius" ? 0.99 : 0.999;
    const int retained = int(double(steps) * strength), first = steps - retained;
    const int ratio = 1000 / steps;
    std::vector<int> result;
    for (int i = first; i < steps; ++i) result.push_back((steps - 1 - i) * ratio);
    return result;
}
std::vector<float> ddim_alphas() {
    const float start = float(std::sqrt(0.00085)), end = float(std::sqrt(0.012));
    const float delta = (end - start) / 999.0f;
    std::vector<float> result(1000);
    double cumulative = 1;
    for (int i = 0; i < 1000; ++i) {
        // CPU torch.linspace uses the nearer endpoint to limit rounding error.
        const float root = i < 500 ? start + delta * float(i) : end - delta * float(999 - i);
        const float beta = root * root;
        cumulative *= double(1.0f - beta);
        result[i] = float(cumulative);
    }
    return result;
}
Tensor ddim_step(const Tensor & latent, const Tensor & noise, int timestep, int previous,
                 const std::vector<float> & alphas) {
    require(latent.c == noise.c && latent.h == noise.h && latent.w == noise.w && latent.v.size() == noise.v.size() &&
                finite(latent) && finite(noise),
            "DDIM shape or value mismatch");
    require(alphas.size() == 1000 && timestep >= 0 && timestep < 1000 && previous < timestep && previous >= -1000,
            "invalid DDIM timestep");
    const float alpha = alphas[size_t(timestep)], prior = previous < 0 ? 1.0f : alphas[size_t(previous)];
    require(alpha > 0 && alpha <= 1 && prior > 0 && prior <= 1, "invalid DDIM alpha");
    const float sqrt_alpha = std::sqrt(alpha), sqrt_beta = std::sqrt(1.0f - alpha);
    const float sqrt_prior = std::sqrt(prior), direction = std::sqrt(1.0f - prior);
    Tensor result(latent.c, latent.h, latent.w);
    for (size_t i = 0; i < latent.v.size(); ++i) {
        const float predicted_original = (latent.v[i] - sqrt_beta * noise.v[i]) / sqrt_alpha;
        // epsilon prediction, eta=0, no threshold/clip and final alpha=1.
        result.v[i] = sqrt_prior * predicted_original + direction * noise.v[i];
    }
    return result;
}
Tensor preprocess_rgb8(const uint8_t * rgb, size_t bytes) {
    require(rgb && bytes == size_t(512) * 512 * 3, "RGB512 input byte count differs");
    Tensor result(3, 512, 512);
    for (size_t p = 0; p < result.spatial(); ++p)
        for (int ch = 0; ch < 3; ++ch)
            result.v[size_t(ch) * result.spatial() + p] = float(double(rgb[p * 3 + ch]) / 255.0 * 2.0 - 1.0);
    return result;
}
void composite_rgb8(const Tensor & decoded, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask,
                    size_t mask_bytes, uint8_t * output, size_t output_bytes) {
    require(decoded.c == 3 && finite(decoded) && rgb && mask && output && rgb_bytes == decoded.spatial() * 3 &&
                mask_bytes == decoded.spatial() && output_bytes == rgb_bytes,
            "invalid decoded RGB or composition buffer");
    require(std::all_of(mask, mask + mask_bytes, [](uint8_t v) { return v == 0 || v == 255; }),
            "composition mask must be binary0/255");
    // Transactional and alias-safe even if output overlaps RGB or mask.
    std::vector<uint8_t> result(rgb, rgb + rgb_bytes);
    for (size_t p = 0; p < mask_bytes; ++p)
        if (mask[p])
            for (int ch = 0; ch < 3; ++ch) {
                const float pixel = std::clamp((decoded.v[size_t(ch) * mask_bytes + p] + 1.0f) / 2.0f, 0.0f, 1.0f);
                result[p * 3 + ch] = uint8_t(pixel * 255.0f);
            }
    std::memcpy(output, result.data(), result.size());
}
} // namespace detail

namespace {
struct Busy {
    context & model;
    explicit Busy(context & model) : model(model) {
        require(!model.busy.test_and_set(), "one context cannot process concurrent requests");
    }
    ~Busy() { model.busy.clear(); }
};

Tensor forward(context & ctx, const Tensor & image, const Tensor & mask, const Noise & noise) {
    enabled(ctx.options.model);
    Tensor alpha_table(1000, 1, 1);
    alpha_table.v = ctx.alphas;
    ctx.mark("sampler.alphas_cumprod", alpha_table);
    require(image.c == 3 && image.h == 512 && image.w == 512 && image.v.size() == size_t(3) * 512 * 512 &&
                mask.c == 1 && mask.h == 512 && mask.w == 512 && mask.v.size() == size_t(512) * 512 && finite(image) &&
                finite(mask),
            "pipeline requires finite RGB/mask512 tensors");
    require(std::all_of(mask.v.begin(), mask.v.end(), [](float v) { return v == 0 || v == 1; }),
            "pipeline mask must be binary white=fill");
    auto valid_noise = [](const Tensor & value, int height, int width) {
        return value.c == 4 && value.h == height && value.w == width && value.v.size() == size_t(4) * height * width &&
               finite(value);
    };
    require(valid_noise(noise.posterior_image, 64, 64) && valid_noise(noise.posterior_masked, 64, 64) &&
                valid_noise(noise.latent_noise, 64, 64) && valid_noise(noise.offset_noise, 1, 1),
            "noise shape or value differs");
    Tensor masked(image.c, image.h, image.w);
    for (int ch = 0; ch < 3; ++ch)
        for (size_t p = 0; p < image.spatial(); ++p)
            masked.v[size_t(ch) * image.spatial() + p] = image.v[size_t(ch) * image.spatial() + p] * (1.0f - mask.v[p]);
    Tensor moments = vae_encode_moments(*ctx.vae, image);
    ctx.mark("vae.image.moments", moments);
    Tensor image_latents = vae_sample(moments, noise.posterior_image);
    ctx.mark("vae.image.latents", image_latents);
    moments = vae_encode_moments(*ctx.vae, masked);
    ctx.mark("vae.masked.moments", moments);
    Tensor masked_latents = vae_sample(moments, noise.posterior_masked);
    ctx.mark("vae.masked.latents", masked_latents);
    Tensor latent_mask = resize_nearest(mask, 64, 64);
    Tensor latent(4, 64, 64);
    const float alpha = ctx.alphas[size_t(ctx.timesteps.front())];
    for (size_t i = 0; i < latent.v.size(); ++i) {
        const float added_noise =
            noise.latent_noise.v[i] + ctx.options.noise_offset * noise.offset_noise.v[i / latent.spatial()];
        latent.v[i] = std::sqrt(alpha) * image_latents.v[i] + std::sqrt(1.0f - alpha) * added_noise;
    }
    ctx.mark("latent.initial", latent);
    for (size_t step = 0; step < ctx.timesteps.size(); ++step) {
        const int t = ctx.timesteps[step];
        Tensor input = concat(concat(latent, latent_mask), masked_latents);
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        Engine::Observer observe = [](void * opaque, const std::string & name, const Tensor & value) {
            auto & model = *static_cast<context *>(opaque);
            model.mark(model.stage_prefix + name, value);
        };
        ctx.denoiser->set_observer(step == 0 && ctx.observer ? observe : nullptr, &ctx);
        ctx.stage_prefix = "unet.step.0.unconditional.";
#endif
        Tensor uncond = denoise(*ctx.denoiser, ctx.variant, input, t, false);
        ctx.mark("unet.step" + std::to_string(step) + ".unconditional", uncond);
#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
        ctx.stage_prefix = "unet.step.0.conditional.";
#endif
        Tensor conditional = denoise(*ctx.denoiser, ctx.variant, input, t, true);
        ctx.mark("unet.step" + std::to_string(step) + ".conditional", conditional);
        require(uncond.c == 4 && uncond.h == 64 && uncond.w == 64 && conditional.v.size() == uncond.v.size(),
                "denoiser output shape differs");
        for (size_t i = 0; i < uncond.v.size(); ++i)
            uncond.v[i] += ctx.options.guidance * (conditional.v[i] - uncond.v[i]);
        ctx.mark("sampler.noise." + std::to_string(step), uncond);
        latent = detail::ddim_step(latent, uncond, t, t - 1000 / ctx.options.steps, ctx.alphas);
        require(finite(latent), "nonfinite sampled latent");
        ctx.mark("sampler.step." + std::to_string(step), latent);
    }
    for (float & value : latent.v) value /= vae_scaling_factor;
    Tensor decoded = vae_decode(*ctx.vae, latent);
    require(finite(decoded), "nonfinite decoded RGB");
    ctx.mark("vae.decoded", decoded);
    return decoded;
}

struct TileContext {
    context * model;
    uint64_t index = 0;
};
bool process_tile(void * opaque, const uint8_t * rgb, const uint8_t * mask, uint8_t * output, std::string & error) {
    try {
        auto & tile = *static_cast<TileContext *>(opaque);
        Tensor image = detail::preprocess_rgb8(rgb, size_t(3) * 512 * 512), fill(1, 512, 512);
        for (size_t p = 0; p < fill.v.size(); ++p) fill.v[p] = mask[p] == 255 ? 1.0f : 0.0f;
        const uint64_t seed = tile.model->options.seed + tile.index++ * UINT64_C(0x9e3779b97f4a7c15);
        Tensor decoded = forward(*tile.model, image, fill, make_noise(seed));
        detail::composite_rgb8(decoded, rgb, size_t(3) * 512 * 512, mask, size_t(512) * 512, output,
                               size_t(3) * 512 * 512);
        return true;
    } catch (const std::exception & exc) {
        return fail(error, exc.what());
    }
}
} // namespace

context * init(const Options & options, std::string & error) {
    error.clear();
    try {
        enabled(options.model);
        require(options.threads >= 1 && options.threads <= 64 && std::isfinite(options.guidance) &&
                    options.guidance >= 0 && options.guidance <= 30 && std::isfinite(options.noise_offset) &&
                    options.noise_offset >= 0 && options.noise_offset <= 1,
                "invalid inference settings");
        auto ctx = std::make_unique<context>();
        ctx->options = options;
        ctx->timesteps = detail::ddim_timesteps(options.steps, options.model);
        ctx->alphas = detail::ddim_alphas();
        ctx->variant = options.model == "moebius" ? Variant::moebius : Variant::pixelhacker;
        validate_model(options.denoiser_path, options.denoiser_sha256, options.model);
        validate_model(options.vae_path, options.vae_sha256, "sdxl_vae");
        ctx->denoiser =
            std::make_unique<Engine>(options.denoiser_path.c_str(), options.threads, denoiser_schema(options.model));
        ctx->vae = std::make_unique<Engine>(options.vae_path.c_str(), options.threads, vae_schema());
        return ctx.release();
    } catch (const std::exception & exc) {
        error = exc.what();
        return nullptr;
    }
}
void free(context * ctx) {
    delete ctx;
}

bool process_rgb8(context * ctx, const uint8_t * rgb, size_t rgb_bytes, const uint8_t * mask, size_t mask_bytes,
                  int width, int height, uint8_t * output, size_t output_bytes, std::string & error) {
    error.clear();
    try {
        require(ctx && rgb && mask && output, "null context or image buffer");
        enabled(ctx->options.model);
        Busy lock(*ctx);
        require(width > 0 && height > 0 && width <= inpaint::max_dimension && height <= inpaint::max_dimension &&
                    size_t(width) <= inpaint::max_pixels / size_t(height),
                "image dimensions exceed limits");
        const size_t pixels = size_t(width) * height;
        require(rgb_bytes == pixels * 3 && mask_bytes == pixels && output_bytes == rgb_bytes,
                "image buffer size differs");
        require(ctx->options.tiled || (width == 512 && height == 512),
                "latent models require512x512; explicitly enable --inpaint-tiles for independent512 tiles");
        bool any = false;
        for (size_t i = 0; i < pixels; ++i) {
            require(mask[i] == 0 || mask[i] == 255, "mask must be binary0/255");
            any = any || mask[i] != 0;
        }
        if (!any) {
            std::memmove(output, rgb, rgb_bytes);
            return true;
        }
        TileContext tile{ ctx, 0 };
        if (ctx->options.tiled)
            return inpaint::detail::process_tiles_rgb8_variant(512, process_tile, &tile, rgb, rgb_bytes, mask,
                                                               mask_bytes, width, height, output, output_bytes, error);
        std::vector<uint8_t> result(rgb_bytes);
        if (!process_tile(&tile, rgb, mask, result.data(), error)) return false;
        std::memcpy(output, result.data(), result.size());
        return true;
    } catch (const std::exception & exc) {
        return fail(error, exc.what());
    }
}

#ifdef CRISPEMBED_LATENT_INPAINT_TESTING
void set_observer(context * ctx, Observer callback, void * opaque) {
    require(ctx != nullptr, "null reference context");
    ctx->observer = callback;
    ctx->observer_opaque = opaque;
}
bool run_reference(context * ctx, const Tensor & image, const Tensor & mask, const ReferenceNoise & noise,
                   Tensor & decoded, std::string & error) {
    error.clear();
    try {
        require(ctx != nullptr, "null reference context");
        Busy lock(*ctx);
        Tensor result =
            forward(*ctx, image, mask,
                    { noise.posterior_image, noise.posterior_masked, noise.latent_noise, noise.offset_noise });
        decoded = std::move(result);
        return true;
    } catch (const std::exception & exc) {
        return fail(error, exc.what());
    }
}
#endif
} // namespace latent_inpaint
