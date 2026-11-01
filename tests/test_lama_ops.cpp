// Bounded model-free tests of production LaMa FFT/padding/pixel primitives.
// The direct DFT below is an independent tiny-input oracle, never a runtime
// fallback. Original-checkpoint generator parity is a separate hosted gate.
#include "lama.h"
#include "lama_fft.h"
#include "lama_schema.h"
#include "core/clean_exit.h"
#include "core/sha256.h"
#include "core/temp_file.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using latent_inpaint::Tensor;
using Complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
int checks = 0, failures = 0;

void check(bool ok, const char * description) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description);
    }
}
template <class Function> void rejects(Function function, const char * description) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, description);
}
Complex direct_frequency(const Tensor & image, int channel, int ky, int kx) {
    Complex result = 0;
    for (int y = 0; y < image.h; ++y)
        for (int x = 0; x < image.w; ++x) {
            const double phase = -2 * pi * (double(ky) * y / image.h + double(kx) * x / image.w);
            result += double(image.at(channel, y, x)) * std::polar(1.0, phase);
        }
    return result / std::sqrt(double(image.spatial()));
}

void fft_contracts() {
    for (const auto & shape : { std::pair<int, int>{ 1, 1 }, { 8, 8 }, { 3, 5 }, { 7, 10 }, { 16, 9 } }) {
        Tensor image(2, shape.first, shape.second);
        for (size_t i = 0; i < image.v.size(); ++i) image.v[i] = float(int(i * 37 % 29) - 14) / 11.0f;
        const Tensor packed = lama::detail::rfft2_ortho(image);
        check(packed.c == 4 && packed.h == image.h && packed.w == image.w / 2 + 1,
              "rfft2 keeps the official real/imag channel packing");
        double maximum = 0;
        for (int ch = 0; ch < image.c; ++ch)
            for (int y = 0; y < packed.h; ++y)
                for (int x = 0; x < packed.w; ++x) {
                    const auto expected = direct_frequency(image, ch, y, x);
                    maximum = std::max(maximum, std::abs(double(packed.at(2 * ch, y, x)) - expected.real()));
                    maximum = std::max(maximum, std::abs(double(packed.at(2 * ch + 1, y, x)) - expected.imag()));
                }
        check(maximum <= 2e-6, "radix-2/Bluestein rfft2 agrees with independent direct DFT");
        const Tensor restored = lama::detail::irfft2_ortho(packed, image.w);
        maximum = 0;
        for (size_t i = 0; i < image.v.size(); ++i)
            maximum = std::max(maximum, std::abs(double(image.v[i]) - restored.v[i]));
        check(maximum <= 2e-6, "orthonormal FFT round trip covers odd, even and prime dimensions");
    }

    Tensor constant(1, 5, 7);
    std::fill(constant.v.begin(), constant.v.end(), 1.0f);
    auto packed = lama::detail::rfft2_ortho(constant);
    check(std::abs(packed.v[0] - std::sqrt(35.0)) <= 1e-6, "orthonormal DC of a constant is sqrt(pixel count)");
    packed.v[0] = 0;
    check(std::all_of(packed.v.begin(), packed.v.end(), [](float v) { return std::abs(v) < 1e-6f; }),
          "constant has zero non-DC spectrum");

    // The learned spectral convolution need not preserve Hermitian symmetry
    // on the DC/Nyquist columns. Compare irfft's real projection independently.
    Tensor learned(2, 3, 3);
    for (size_t i = 0; i < learned.v.size(); ++i) learned.v[i] = float(int(i * 7 % 19) - 9) / 5.0f;
    const int width = 5;
    const auto restored = lama::detail::irfft2_ortho(learned, width);
    double maximum = 0;
    for (int y = 0; y < restored.h; ++y)
        for (int x = 0; x < width; ++x) {
            Complex sum = 0;
            for (int ky = 0; ky < restored.h; ++ky)
                for (int kx = 0; kx < width; ++kx) {
                    Complex spectrum;
                    if (kx < learned.w)
                        spectrum = Complex(learned.at(0, ky, kx), learned.at(1, ky, kx));
                    else {
                        const int mirror_y = (restored.h - ky) % restored.h;
                        spectrum = std::conj(
                            Complex(learned.at(0, mirror_y, width - kx), learned.at(1, mirror_y, width - kx)));
                    }
                    sum += spectrum * std::polar(1.0, 2 * pi * (double(ky) * y / restored.h + double(kx) * x / width));
                }
            maximum =
                std::max(maximum, std::abs(restored.at(0, y, x) - sum.real() / std::sqrt(double(restored.spatial()))));
        }
    check(maximum < 2e-6, "irfft real projection matches direct inverse DFT for learned spectra");
    Tensor imaginary_boundaries(2, 1, 3);
    imaginary_boundaries.at(1, 0, 0) = 100;
    imaginary_boundaries.at(1, 0, 2) = 200;
    const auto real_boundary = lama::detail::irfft2_ortho(imaginary_boundaries, 4);
    check(std::all_of(real_boundary.v.begin(), real_boundary.v.end(), [](float v) { return std::abs(v) < 1e-10f; }),
          "imaginary DC and Nyquist components do not contribute to real output");

    rejects([] { lama::detail::rfft2_ortho(Tensor{}); }, "empty FFT input rejected");
    auto malformed = constant;
    malformed.v.pop_back();
    rejects([&] { lama::detail::rfft2_ortho(malformed); }, "malformed FFT storage rejected");
    malformed = constant;
    malformed.v[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { lama::detail::rfft2_ortho(malformed); }, "nonfinite FFT input rejected");
    rejects([&] { lama::detail::irfft2_ortho(learned, 7); }, "inverse FFT cannot infer a mismatched real width");
    rejects([] { lama::detail::rfft2_ortho(Tensor(1, 257, 1)); }, "FFT workspace dimension cap enforced");
}

void pixel_contracts() {
    Tensor tiny(1, 2, 3);
    tiny.v = { 0, 1, 2, 3, 4, 5 };
    const auto padded = lama::detail::reflect_pad(tiny, 1);
    const std::vector<float> expected{ 4, 3, 4, 5, 4, 1, 0, 1, 2, 1, 4, 3, 4, 5, 4, 1, 0, 1, 2, 1 };
    check(padded.c == 1 && padded.h == 4 && padded.w == 5 && padded.v == expected,
          "convolution reflection excludes edge repetition exactly as PyTorch");
    check(lama::detail::reflect_pad(Tensor(1, 1, 1), 0).v == std::vector<float>{ 0 },
          "zero reflection padding is identity");
    rejects([&] { lama::detail::reflect_pad(tiny, 2); }, "reflection extent must be smaller than both axes");

    constexpr int width = 9, height = 11;
    std::vector<uint8_t> rgb(size_t(width) * height * 3), mask(size_t(width) * height), output(rgb.size(), 213);
    for (size_t p = 0; p < mask.size(); ++p) {
        rgb[p * 3] = uint8_t(p * 3 % 256);
        rgb[p * 3 + 1] = uint8_t(p % 256);
        rgb[p * 3 + 2] = 255;
    }
    mask[1] = 255;
    const auto input = lama::detail::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height);
    check(input.c == 4 && input.h == 16 && input.w == 16, "LaMa pads variable dimensions to multiples of eight");
    check(input.at(0, 0, 1) == 0 && input.at(1, 0, 1) == 0 && input.at(2, 0, 1) == 0 && input.at(3, 0, 1) == 1,
          "white mask zeros RGB and becomes the fourth input channel");
    check(input.at(2, 0, 0) == 1 && input.at(3, 0, 0) == 0, "known input RGB is normalized to [0,1]");
    check(input.at(1, 11, 9) == float(rgb[(size_t(10) * width + 8) * 3 + 1]) / 255.0f &&
              input.at(1, 15, 15) == float(rgb[(size_t(6) * width + 2) * 3 + 1]) / 255.0f,
          "input modulo padding is symmetric with repeated edges, distinct from convolution reflection");

    Tensor prediction(3, 16, 16);
    prediction.at(0, 0, 1) = -1;
    prediction.at(1, 0, 1) = 0.5f;
    prediction.at(2, 0, 1) = 2;
    auto compose = [&] {
        lama::detail::composite_rgb8(prediction, rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                                     output.data(), output.size());
    };
    compose();
    check(output[3] == 0 && output[4] == 127 && output[5] == 255, "LaMa fill output clamps and truncates uint8");
    bool known_exact = true;
    for (size_t i = 0; i < rgb.size(); ++i)
        if (!mask[i / 3]) known_exact &= rgb[i] == output[i];
    check(known_exact, "all known RGB bytes are preserved exactly after crop");
    auto alias = rgb;
    lama::detail::composite_rgb8(prediction, alias.data(), alias.size(), mask.data(), mask.size(), width, height,
                                 alias.data(), alias.size());
    check(alias == output, "LaMa composition supports input/output aliasing");
    std::fill(output.begin(), output.end(), 213);
    mask.back() = 1;
    rejects(compose, "nonbinary mask rejected");
    check(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 213; }),
          "late mask failure is transactional");
    mask.back() = 0;
    prediction.v.back() = std::numeric_limits<float>::quiet_NaN();
    rejects(compose, "nonfinite predicted RGB rejected even outside crop");
    check(std::all_of(output.begin(), output.end(), [](uint8_t v) { return v == 213; }),
          "nonfinite failure is transactional");
    rejects([&] { lama::detail::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 8, height); },
            "input too small for official bottleneck reflection rejected");
    rejects([&] { lama::detail::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 1025, height); },
            "input dimension cap enforced before buffer reads");
    std::string error;
    check(!lama::process_rgb8(nullptr, rgb.data(), rgb.size(), mask.data(), mask.size(), width, height, output.data(),
                              output.size(), error) &&
              !error.empty(),
          "null LaMa context rejected");
}

void metadata_contracts() {
    const std::vector<std::pair<const char *, const char *>> strings{
        { "general.architecture", "big_lama" },
        { "general.license", "Apache-2.0" },
        { "lama.source.commit", lama::detail::source_commit() },
        { "lama.contract.sha256", lama::detail::source_contract_sha256() },
        { "lama.checkpoint.sha256", lama::detail::checkpoint_sha256() },
        { "lama.checkpoint.revision", lama::detail::checkpoint_revision() },
        { "lama.checkpoint.container", lama::detail::checkpoint_container() },
        { "lama.checkpoint.member", lama::detail::checkpoint_member() },
        { "lama.tensor.layout", lama::detail::tensor_layout() },
        { "lama.input.contract", lama::detail::input_contract() },
        { "lama.weights.license.status", lama::detail::weight_license_status() },
        { "lama.fft.norm", "ortho" }
    };
    const std::vector<std::pair<const char *, uint32_t>> integers{ { "general.file_type", 0 },
                                                                   { "lama.schema", 1 },
                                                                   { "lama.generator.blocks", 18 },
                                                                   { "lama.image.min_dimension", 9 },
                                                                   { "lama.image.max_dimension", 1024 } };
    const std::string path = core_tmp::make_private(".gguf");
    if (path.empty()) throw std::runtime_error("cannot create private metadata fixture");
    struct Cleanup {
        const std::string & path;
        ~Cleanup() { std::remove(path.c_str()); }
    } cleanup{ path };
    auto probe = [&](const std::function<void(gguf_context *)> & mutate, const char * expected) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), &gguf_free);
        if (!metadata) throw std::runtime_error("cannot allocate metadata fixture");
        for (const auto & item : strings) gguf_set_val_str(metadata.get(), item.first, item.second);
        for (const auto & item : integers) gguf_set_val_u32(metadata.get(), item.first, item.second);
        gguf_set_val_bool(metadata.get(), "lama.lfu", false);
        mutate(metadata.get());
        if (!gguf_write_to_file(metadata.get(), path.c_str(), true))
            throw std::runtime_error("cannot write metadata fixture");
        std::string error;
        std::unique_ptr<lama::context, decltype(&lama::free)> model(lama::init(path, core_sha256::file(path), 1, error),
                                                                    &lama::free);
        if (model || error.find(expected) == std::string::npos)
            std::fprintf(stderr, "metadata error: %s\n", error.c_str());
        return !model && error.find(expected) != std::string::npos;
    };
    check(probe([](gguf_context *) {}, "checkpoint tensor count differs"),
          "correct metadata reaches tensor schema before any weight allocation");
    for (const auto & item : strings) {
        check(probe([&](gguf_context * meta) { gguf_set_val_str(meta, item.first, "wrong"); }, "contract differs"),
              "each semantic string mismatch rejected with a recomputed valid file digest");
        check(probe([&](gguf_context * meta) { gguf_remove_key(meta, item.first); }, "contract differs"),
              "each missing semantic string rejected");
    }
    for (const auto & item : integers) {
        check(probe([&](gguf_context * meta) { gguf_set_val_u32(meta, item.first, item.second + 1); },
                    "contract differs"),
              "each semantic integer mismatch rejected");
        check(probe([&](gguf_context * meta) { gguf_set_val_u64(meta, item.first, (UINT64_C(1) << 32) + item.second); },
                    "contract differs"),
              "semantic integers cannot pass through uint64 truncation");
    }
    check(probe([](gguf_context * meta) { gguf_set_val_bool(meta, "lama.lfu", true); }, "contract differs"),
          "unsupported LFU rejected");
    check(probe([](gguf_context * meta) { gguf_set_val_u32(meta, "lama.lfu", 0); }, "contract differs"),
          "LFU requires exact bool metadata type");
}

void gates() {
    const char * prior = std::getenv("CRISPEMBED_LAMA");
    const bool existed = prior != nullptr;
    const std::string saved = prior ? prior : "";
    auto set = [](const char * value) {
#ifdef _WIN32
        _putenv_s("CRISPEMBED_LAMA", value ? value : "");
#else
        if (value)
            setenv("CRISPEMBED_LAMA", value, 1);
        else
            unsetenv("CRISPEMBED_LAMA");
#endif
    };
    for (const char * off : std::array<const char *, 3>{ nullptr, "", "0" }) {
        set(off);
        std::string error;
        auto * model = lama::init("unused.gguf", std::string(64, '0'), 1, error);
        check(!model && error.find("CRISPEMBED_LAMA") != std::string::npos,
              "unset/empty/zero LaMa gate rejects before file access");
        lama::free(model);
    }
    set("1");
    std::string error;
    auto * model = lama::init("unused.gguf", std::string(64, '0'), 0, error);
    check(!model && error.find("thread count") != std::string::npos, "invalid LaMa thread count rejected");
    lama::free(model);
    model = lama::init("unused.gguf", std::string(64, 'A'), 1, error);
    check(!model && error.find("lowercase") != std::string::npos, "invalid LaMa receipt digest rejected");
    lama::free(model);
    metadata_contracts();
    set(existed ? saved.c_str() : nullptr);
}

int run() {
    try {
        fft_contracts();
        pixel_contracts();
        gates();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "unexpected error: %s\n", error.what());
        ++failures;
    }
    std::printf("LaMa operators: %d checks, %d failures (model-free; no original-checkpoint qualification)\n", checks,
                failures);
    return failures ? 1 : 0;
}
} // namespace
int main() {
    core_util::clean_exit(run());
}
