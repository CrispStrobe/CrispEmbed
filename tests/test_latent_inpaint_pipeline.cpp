// Model-free tests of the production latent sampler, pixel boundary and loader
// guards. Run on hosted CI; these do not load model weights or perform inference.
#include "latent_inpaint.h"
#include "latent_inpaint_schema.h"
#include "latent_inpaint_vae.h"
#include "core/clean_exit.h"
#include "core/sha256.h"
#include "core/temp_file.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using latent_inpaint::Options;
using latent_inpaint::Tensor;
int checks = 0, failures = 0;

void check(bool condition, const std::string & description) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description.c_str());
    }
}

template <class Function> void rejects(Function function, const std::string & description) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, description);
}

bool near(float actual, double expected, double tolerance = 2e-6) {
    return std::isfinite(actual) && std::abs(double(actual) - expected) <= tolerance;
}

struct Gate {
    std::string name, previous;
    bool was_set;
    explicit Gate(const char * key) : name(key), was_set(std::getenv(key) != nullptr) {
        if (was_set) previous = std::getenv(key);
    }
    void set(const char * value) const {
#ifdef _WIN32
        _putenv_s(name.c_str(), value ? value : "");
#else
        if (value)
            setenv(name.c_str(), value, 1);
        else
            unsetenv(name.c_str());
#endif
    }
    ~Gate() { set(was_set ? previous.c_str() : nullptr); }
};

void schedule_contracts() {
    // Both upstream strengths retain 19 of the 20 requested leading timesteps.
    // The non-divisor case distinguishes integer leading spacing from linspace.
    const std::vector<int> twenty{ 900, 850, 800, 750, 700, 650, 600, 550, 500, 450,
                                   400, 350, 300, 250, 200, 150, 100, 50,  0 };
    for (const char * model : { "moebius", "pixelhacker" }) {
        check(latent_inpaint::detail::ddim_timesteps(20, model) == twenty,
              std::string(model) + " default strength and leading timesteps");
        check(latent_inpaint::detail::ddim_timesteps(3, model) == std::vector<int>({ 333, 0 }),
              std::string(model) + " non-divisor schedule uses floor(1000/steps)");
        check(latent_inpaint::detail::ddim_timesteps(2, model) == std::vector<int>({ 0 }),
              std::string(model) + " lower bounded schedule still executes the terminal step");
        const auto largest = latent_inpaint::detail::ddim_timesteps(100, model);
        check(largest.size() == 99 && largest.front() == 980 && largest.back() == 0,
              std::string(model) + " upper bounded schedule retains strength truncation");
    }
    for (int steps : { -1, 0, 1, 101, std::numeric_limits<int>::max() })
        rejects([&] { latent_inpaint::detail::ddim_timesteps(steps, "moebius"); }, "invalid step count rejected");
    rejects([] { latent_inpaint::detail::ddim_timesteps(20, "unknown"); }, "unknown schedule rejected");

    const auto alphas = latent_inpaint::detail::ddim_alphas();
    check(alphas.size() == 1000, "DDIM contains all 1000 training alphas");
    bool decreasing = alphas.size() == 1000;
    for (size_t i = 0; i < alphas.size(); ++i)
        decreasing &= std::isfinite(alphas[i]) && alphas[i] > 0 && alphas[i] < 1 && (!i || alphas[i] < alphas[i - 1]);
    check(decreasing, "scaled-linear alphas are positive and strictly decreasing");
    // Source-derived numerical anchors, not claimed original-Torch captures.
    // Exact alpha-table parity is checked separately by the original replay.
    if (alphas.size() == 1000) {
        check(near(alphas[0], 0.9991499781608582, 6e-8), "first scaled-linear alpha");
        check(near(alphas[499], 0.27766942977905273, 6e-8), "alpha before linspace midpoint");
        check(near(alphas[500], 0.27633246779441833, 6e-8), "alpha after linspace midpoint");
        check(near(alphas[999], 0.00466009508818388, 2e-9), "terminal scaled-linear alpha");
    }

    Tensor latent(1, 1, 3), noise(1, 1, 3);
    latent.v = { 1, -2, 3 };
    noise.v = { 0.5f, 1, -1 };
    std::vector<float> simple(1000, 0.5f);
    simple[10] = 0.25f;
    simple[5] = 0.81f;
    const auto stepped = latent_inpaint::detail::ddim_step(latent, noise, 10, 5, simple);
    bool interior = stepped.v.size() == 3;
    for (size_t i = 0; i < 3; ++i) {
        const double original = 2.0 * latent.v[i] - std::sqrt(3.0) * noise.v[i];
        interior &= near(stepped.v[i], 0.9 * original + std::sqrt(0.19) * noise.v[i]);
    }
    check(interior, "DDIM eta=0 agrees with a closed-form epsilon-prediction step");
    const auto terminal = latent_inpaint::detail::ddim_step(latent, noise, 10, -1, simple);
    bool final = true;
    for (size_t i = 0; i < 3; ++i) final &= near(terminal.v[i], 2.0 * latent.v[i] - std::sqrt(3.0) * noise.v[i]);
    check(final && terminal.v[2] > 1, "terminal alpha is one and predicted image is not clipped");
    check(latent.v == std::vector<float>({ 1, -2, 3 }) && noise.v == std::vector<float>({ 0.5f, 1, -1 }),
          "sampler leaves its input tensors unchanged");
    for (const auto & pair : { std::pair<int, int>{ -1, -2 }, { 1000, 999 }, { 10, 10 }, { 10, 11 }, { 10, -1001 } })
        rejects([&] { latent_inpaint::detail::ddim_step(latent, noise, pair.first, pair.second, simple); },
                "invalid sampler timestep rejected");
    auto short_table = simple;
    short_table.pop_back();
    rejects([&] { latent_inpaint::detail::ddim_step(latent, noise, 10, 5, short_table); },
            "incomplete alpha table rejected");
    for (float bad : { 0.0f, -0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN() }) {
        auto changed = simple;
        changed[10] = bad;
        rejects([&] { latent_inpaint::detail::ddim_step(latent, noise, 10, 5, changed); },
                "invalid current alpha rejected");
        changed = simple;
        changed[5] = bad;
        rejects([&] { latent_inpaint::detail::ddim_step(latent, noise, 10, 5, changed); },
                "invalid previous alpha rejected");
    }
    for (int size : { 0, 2, 4 }) {
        auto malformed_latent = latent, malformed_noise = noise;
        malformed_latent.v.resize(size);
        malformed_noise.v.resize(size);
        rejects([&] { latent_inpaint::detail::ddim_step(malformed_latent, malformed_noise, 10, 5, simple); },
                "matching malformed storage lengths rejected before sampler indexing");
    }
    noise.v[0] = std::numeric_limits<float>::infinity();
    rejects([&] { latent_inpaint::detail::ddim_step(latent, noise, 10, 5, simple); }, "nonfinite noise rejected");
}

void posterior_contracts() {
    Tensor moments(8, 1, 2), noise(4, 1, 2);
    const std::array<float, 8> means{ 0, 1, -2, 3, 4, -5, 6, -7 };
    std::copy(means.begin(), means.end(), moments.v.begin());
    noise.v = { 1, -1, 0.5f, -0.5f, 2, 0, -2, 1 };
    const auto sampled = latent_inpaint::vae_sample(moments, noise);
    bool expected = sampled.c == 4 && sampled.h == 1 && sampled.w == 2;
    for (size_t i = 0; i < means.size(); ++i) expected &= near(sampled.v[i], (means[i] + noise.v[i]) * 0.13025, 2e-7);
    check(expected, "VAE posterior uses separate means/logvars, supplied noise, and SDXL scale");

    // Clamping both tails is material: unconstrained exp(logvar/2) overflows
    // for the large positive case and suppresses noise for the negative case.
    std::fill(moments.v.begin(), moments.v.end(), 0.0f);
    std::fill(noise.v.begin(), noise.v.end(), 1.0f);
    moments.v[8] = -1000;
    moments.v[9] = 1000;
    const auto clamped = latent_inpaint::vae_sample(moments, noise);
    check(near(clamped.v[0], std::exp(-15.0) * 0.13025, 2e-14) && near(clamped.v[1], std::exp(10.0) * 0.13025, 5e-4),
          "VAE posterior log variance clamps to [-30,20]");
    for (int size : { 0, 15, 17 }) {
        auto broken = moments;
        broken.v.resize(size);
        rejects([&] { latent_inpaint::vae_sample(broken, noise); }, "malformed moment storage rejected");
    }
    auto broken_noise = noise;
    broken_noise.v.pop_back();
    rejects([&] { latent_inpaint::vae_sample(moments, broken_noise); }, "malformed posterior noise storage rejected");
    broken_noise = noise;
    broken_noise.w = 1;
    rejects([&] { latent_inpaint::vae_sample(moments, broken_noise); }, "posterior spatial mismatch rejected");
    broken_noise = noise;
    broken_noise.v[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { latent_inpaint::vae_sample(moments, broken_noise); }, "nonfinite posterior noise rejected");
    check(!latent_inpaint::finite(Tensor{}), "empty default Tensor is not a valid finite activation");
}

void pixel_contracts() {
    constexpr size_t pixels = size_t(512) * 512;
    std::vector<uint8_t> rgb(3 * pixels), mask(pixels), output(rgb.size(), 213);
    for (size_t p = 0; p < pixels; ++p) {
        rgb[p * 3] = uint8_t(p % 256);
        rgb[p * 3 + 1] = uint8_t(255 - p % 256);
        rgb[p * 3 + 2] = 128;
    }
    const auto image = latent_inpaint::detail::preprocess_rgb8(rgb.data(), rgb.size());
    check(image.c == 3 && image.h == 512 && image.w == 512 && image.v.size() == rgb.size(),
          "preprocessing produces RGB planar CHW512");
    check(image.v[0] == -1 && image.v[255] == 1 && image.v[pixels] == 1 && image.v[pixels + 255] == -1 &&
              near(image.v[2 * pixels], 1.0 / 255.0, 1e-9),
          "RGB channel order, endpoints and midpoint normalization");
    rejects([&] { latent_inpaint::detail::preprocess_rgb8(nullptr, rgb.size()); }, "null RGB rejected");
    rejects([&] { latent_inpaint::detail::preprocess_rgb8(rgb.data(), rgb.size() - 1); }, "short RGB rejected");
    rejects([&] { latent_inpaint::detail::preprocess_rgb8(rgb.data(), rgb.size() + 1); }, "oversized RGB rejected");

    Tensor decoded(3, 512, 512);
    decoded.v[0] = -2;
    decoded.v[pixels] = 0;
    decoded.v[2 * pixels] = 2;
    decoded.v[1] = 0.5f;
    decoded.v[pixels + 1] = -0.5f;
    decoded.v[2 * pixels + 1] = 0.25f;
    mask[0] = mask[1] = 255;
    auto compose = [&] {
        latent_inpaint::detail::composite_rgb8(decoded, rgb.data(), rgb.size(), mask.data(), mask.size(), output.data(),
                                               output.size());
    };
    compose();
    check(output[0] == 0 && output[1] == 127 && output[2] == 255 && output[3] == 191 && output[4] == 63 &&
              output[5] == 159,
          "filled RGB clamps and truncates the decoded float value");
    check(std::equal(output.begin() + 6, output.end(), rgb.begin() + 6),
          "every known RGB byte is preserved across all 256 byte values");
    auto alias = rgb;
    latent_inpaint::detail::composite_rgb8(decoded, alias.data(), alias.size(), mask.data(), mask.size(), alias.data(),
                                           alias.size());
    check(alias == output, "known-pixel composition is safe with input/output aliasing");

    auto rejected_without_write = [&](const std::function<void()> & call, const std::string & description) {
        std::fill(output.begin(), output.end(), 213);
        rejects(call, description);
        check(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 213; }),
              description + " leaves output untouched");
    };
    mask.back() = 1;
    rejected_without_write(compose, "nonbinary mask rejected before any output write");
    mask.back() = 0;
    decoded.v.back() = std::numeric_limits<float>::quiet_NaN();
    rejected_without_write(compose, "nonfinite decoded value rejected even at a known pixel");
    decoded.v.back() = 0;
    decoded.v.pop_back();
    rejected_without_write(compose, "short decoded storage rejected");
    decoded.v.push_back(0);
    rejected_without_write(
        [&] {
            latent_inpaint::detail::composite_rgb8(decoded, rgb.data(), rgb.size(), mask.data(), mask.size() - 1,
                                                   output.data(), output.size());
        },
        "short mask storage rejected");
    std::fill(mask.begin(), mask.end(), 0);
    compose();
    check(output == rgb, "empty mask preserves complete input byte for byte");

    std::string error;
    std::fill(output.begin(), output.end(), 213);
    check(!latent_inpaint::process_rgb8(nullptr, rgb.data(), rgb.size(), mask.data(), mask.size(), 512, 512,
                                        output.data(), output.size(), error) &&
              !error.empty(),
          "public entry rejects a null context");
    check(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 213; }),
          "public null-context rejection is transactional");
}

void init_rejects(const Options & options, const std::string & message, const std::string & description) {
    std::string error = "stale error";
    auto * ctx = latent_inpaint::init(options, error);
    const bool rejected = ctx == nullptr && error.find(message) != std::string::npos;
    if (!rejected) std::fprintf(stderr, "  observed init error: %s\n", error.c_str());
    check(rejected, description);
    latent_inpaint::free(ctx);
}

struct MetadataFile {
    std::string path = core_tmp::make_private(".gguf");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta{ gguf_init_empty(), &gguf_free };
    explicit MetadataFile(const std::string & model) {
        if (path.empty() || !meta) throw std::runtime_error("cannot create temporary metadata fixture");
        gguf_set_val_str(meta.get(), "general.architecture", model.c_str());
        gguf_set_val_u32(meta.get(), "general.file_type", 0);
        gguf_set_val_str(meta.get(), "general.license", latent_inpaint::license(model));
        gguf_set_val_u32(meta.get(), "latent_inpaint.schema", 1);
        gguf_set_val_str(meta.get(), "latent_inpaint.component", model == "sdxl_vae" ? "vae" : "denoiser");
        gguf_set_val_str(meta.get(), "latent_inpaint.model", model.c_str());
        gguf_set_val_u32(meta.get(), "latent_inpaint.image_size", 512);
        gguf_set_val_str(meta.get(), "latent_inpaint.contract.sha256", latent_inpaint::source_contract_sha256());
        gguf_set_val_str(meta.get(), "latent_inpaint.source.commit", latent_inpaint::source_commit(model));
        gguf_set_val_str(meta.get(), "latent_inpaint.checkpoint.sha256", latent_inpaint::checkpoint_sha256(model));
        gguf_set_val_str(meta.get(), "latent_inpaint.checkpoint.revision", latent_inpaint::checkpoint_revision(model));
        gguf_set_val_str(meta.get(), "latent_inpaint.input.contract", latent_inpaint::input_contract());
        gguf_set_val_str(meta.get(), "latent_inpaint.tensor.layout", latent_inpaint::tensor_layout());
    }
    ~MetadataFile() {
        if (!path.empty()) std::remove(path.c_str());
    }
    std::string write() {
        if (!gguf_write_to_file(meta.get(), path.c_str(), true))
            throw std::runtime_error("cannot write metadata fixture");
        const auto digest = core_sha256::file(path);
        if (digest.size() != 64) throw std::runtime_error("cannot hash metadata fixture");
        return digest;
    }
};

void model_boundaries() {
    Gate moebius("CRISPEMBED_MOEBIUS"), pixelhacker("CRISPEMBED_PIXELHACKER");
    for (const char * model : { "moebius", "pixelhacker" }) {
        Options options;
        options.model = model;
        const Gate & gate = options.model == "moebius" ? moebius : pixelhacker;
        const Gate & other = options.model == "moebius" ? pixelhacker : moebius;
        other.set("1");
        for (const char * disabled : std::array<const char *, 3>{ nullptr, "", "0" }) {
            gate.set(disabled);
            init_rejects(options, gate.name, std::string(model) + " unset/empty/zero gate fails independently");
        }
        gate.set("1");
        const std::vector<std::function<void(Options &)>> bad_settings{
            [](Options & o) { o.threads = 0; },
            [](Options & o) { o.threads = 65; },
            [](Options & o) { o.guidance = -1; },
            [](Options & o) { o.guidance = 31; },
            [](Options & o) { o.guidance = std::numeric_limits<float>::quiet_NaN(); },
            [](Options & o) { o.noise_offset = -1; },
            [](Options & o) { o.noise_offset = 2; },
            [](Options & o) { o.noise_offset = std::numeric_limits<float>::infinity(); },
        };
        for (const auto & mutate : bad_settings) {
            auto bad = options;
            mutate(bad);
            init_rejects(bad, "invalid inference settings",
                         std::string(model) + " invalid inference option rejected before file access");
        }
        auto bad_steps = options;
        bad_steps.steps = 1;
        init_rejects(bad_steps, "sampling steps",
                     std::string(model) + " invalid sampling steps rejected before file access");
        for (const std::string & digest :
             { std::string(), std::string(63, 'a'), std::string(64, 'A'), std::string(64, 'g') }) {
            auto bad = options;
            bad.denoiser_sha256 = digest;
            init_rejects(bad, "lowercase SHA-256", std::string(model) + " malformed receipt digest rejected");
        }

        MetadataFile denoiser(model), vae("sdxl_vae");
        options.denoiser_path = denoiser.path;
        options.vae_path = vae.path;
        options.denoiser_sha256 = denoiser.write();
        options.vae_sha256 = vae.write();
        init_rejects(options, "checkpoint tensor count differs",
                     std::string(model) + " valid metadata reaches tensor-schema guard without loading weights");
        auto wrong_digest = options;
        wrong_digest.denoiser_sha256[0] = wrong_digest.denoiser_sha256[0] == '0' ? '1' : '0';
        init_rejects(wrong_digest, "GGUF digest differs", std::string(model) + " receipt digest mismatch rejected");

        const std::array<const char *, 10> strings{ "general.architecture",
                                                    "general.license",
                                                    "latent_inpaint.component",
                                                    "latent_inpaint.model",
                                                    "latent_inpaint.contract.sha256",
                                                    "latent_inpaint.source.commit",
                                                    "latent_inpaint.checkpoint.sha256",
                                                    "latent_inpaint.checkpoint.revision",
                                                    "latent_inpaint.input.contract",
                                                    "latent_inpaint.tensor.layout" };
        const std::array<std::pair<const char *, uint32_t>, 3> integers{ std::pair<const char *, uint32_t>{
                                                                             "general.file_type", 0 },
                                                                         { "latent_inpaint.schema", 1 },
                                                                         { "latent_inpaint.image_size", 512 } };
        // Exercise both components independently. Each mutation is a legitimate
        // tiny GGUF with its newly computed digest, so a metadata guard must be
        // what rejects it, rather than a stale file hash or missing weight file.
        for (bool change_vae : { false, true }) {
            const std::string profile = change_vae ? "sdxl_vae" : model;
            auto reject_meta = [&](const std::function<void(gguf_context *)> & mutate, const std::string & label) {
                MetadataFile changed(profile);
                mutate(changed.meta.get());
                auto bad = options;
                if (change_vae) {
                    bad.vae_path = changed.path;
                    bad.vae_sha256 = changed.write();
                } else {
                    bad.denoiser_path = changed.path;
                    bad.denoiser_sha256 = changed.write();
                }
                init_rejects(bad, "contract differs", std::string(model) + "/" + profile + " " + label);
            };
            for (const char * key : strings) {
                reject_meta([&](gguf_context * m) { gguf_set_val_str(m, key, "wrong"); },
                            std::string(key) + " mismatch");
                reject_meta([&](gguf_context * m) { gguf_remove_key(m, key); }, std::string(key) + " missing");
            }
            for (const auto & item : integers) {
                reject_meta([&](gguf_context * m) { gguf_set_val_u32(m, item.first, item.second + 1); },
                            std::string(item.first) + " mismatch");
                reject_meta(
                    [&](gguf_context * m) { gguf_set_val_u64(m, item.first, (UINT64_C(1) << 32) + item.second); },
                    std::string(item.first) + " cannot pass by uint64 truncation");
                reject_meta(
                    [&](gguf_context * m) { gguf_set_val_str(m, item.first, std::to_string(item.second).c_str()); },
                    std::string(item.first) + " wrong scalar type");
            }
        }
    }
    Options unknown;
    unknown.model = "sdxl_vae";
    init_rejects(unknown, "unknown model profile", "VAE component cannot be selected as an inpainting model");
}

int run() {
    try {
        schedule_contracts();
        posterior_contracts();
        pixel_contracts();
        model_boundaries();
    } catch (const std::exception & error) {
        check(false, std::string("unexpected test exception: ") + error.what());
    }
    std::printf("latent pipeline: %d checks, %d failures (model-free; original-model parity is separate)\n", checks,
                failures);
    return failures ? 1 : 0;
}
} // namespace

int main() {
    core_util::clean_exit(run());
}
