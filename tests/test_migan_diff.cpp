// Original-module stage + decoded RGB8 comparison for the experimental CPU
// prototype. This executable and observer variant are never linked into the
// production shared library. Model workloads run only in hosted CI.
#include "migan.h"
#include "migan_test_observer.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

void set_gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MIGAN", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_MIGAN", value, 1);
    else
        unsetenv("CRISPEMBED_MIGAN");
#endif
}

bool bytes(const char * path, std::vector<uint8_t> & data, size_t expected) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != std::streamoff(expected)) return false;
    data.resize(expected);
    stream.seekg(0);
    return bool(stream.read(reinterpret_cast<char *>(data.data()), expected));
}

bool dump(const std::filesystem::path & path, const void * data, size_t count) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(data), count);
    return bool(stream);
}

bool contract_controls() {
    std::string error;
    std::vector<uint8_t> rgb(migan::rgb_elements, 17), mask(migan::mask_elements, 0), result(migan::rgb_elements);
    std::vector<float> input(migan::input_elements), raw(migan::rgb_elements, 0.0f);
    bool pass = migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(),
                                       input.size(), error);
    pass &= input[0] == 0.5f;
    mask[0] = 255;
    pass &= migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(),
                                   input.size(), error);
    pass &= input[0] == -0.5f && input[migan::mask_elements] == 0.0f;
    pass &= migan::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                                  result.data(), result.size(), error);
    pass &= result[0] == 127 && result[1] == 127 && result[2] == 127 && result[3] == 17;
    mask[0] = 127;
    pass &= !migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(),
                                    input.size(), error);
    pass &= !migan::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                                   result.data(), result.size(), error);
    mask[0] = 0;
    pass &= !migan::preprocess_rgb8(rgb.data(), rgb.size() - 1, mask.data(), mask.size(), 256, 256, input.data(),
                                    input.size(), error);
    pass &= !migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size() - 1, 256, 256, input.data(),
                                    input.size(), error);
    pass &= !migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 512, 256, input.data(),
                                    input.size(), error);
    pass &= !migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(),
                                    input.size() - 1, error);
    pass &= !migan::composite_rgb8(raw.data(), raw.size() - 1, rgb.data(), rgb.size(), mask.data(), mask.size(),
                                   result.data(), result.size(), error);
    raw[0] = std::numeric_limits<float>::quiet_NaN();
    pass &= !migan::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(),
                                   result.data(), result.size(), error);
    std::printf("MIGAN_CONTRACT {\"passed\":%s}\n", pass ? "true" : "false");
    return pass;
}

bool gate_controls(const char * path, const char * digest) {
    std::string error;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        set_gate(value);
        std::unique_ptr<migan::context, decltype(&migan::free)> ctx(migan::init(path, digest, 2, error), migan::free);
        if (ctx || error != "MI-GAN requires CRISPEMBED_MIGAN=1") return false;
        if (migan::run(nullptr, nullptr, 0, nullptr, 0, error) || error != "MI-GAN requires CRISPEMBED_MIGAN=1")
            return false;
    }
    set_gate("1");
    std::printf("MIGAN_GATES {\"unset_rejected\":true,\"empty_rejected\":true,\"literal_zero_rejected\":true}\n");
    return true;
}

bool image_metrics(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & reference,
                   const std::vector<uint8_t> & input, const std::vector<uint8_t> & fill) {
    size_t count = 0, known_errors = 0;
    double sum = 0, sum_squared = 0;
    int max_abs = 0;
    for (size_t pixel = 0; pixel < fill.size(); ++pixel)
        for (size_t channel = 0; channel < 3; ++channel) {
            const size_t i = 3 * pixel + channel;
            if (fill[pixel] == 0) {
                if (actual[i] != input[i] || reference[i] != input[i]) ++known_errors;
            } else {
                const int delta = std::abs(int(actual[i]) - int(reference[i]));
                max_abs = std::max(max_abs, delta);
                sum += delta;
                sum_squared += delta * delta;
                ++count;
            }
        }
    const double mae = count ? sum / count : 0;
    const double mse = count ? sum_squared / count : 0;
    const double psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : std::numeric_limits<double>::infinity();
    const bool pass = known_errors == 0 && (!count || (max_abs <= 1 && mae <= 0.1 && psnr >= 50.0));
    if (!count) {
        std::printf("MIGAN_IMAGE "
                    "{\"fill_channels\":0,\"max_abs\":null,\"mae\":null,\"mse\":null,\"psnr_db\":null,\"known_errors\":"
                    "%zu,\"passed\":%s}\n",
                    known_errors, pass ? "true" : "false");
    } else {
        std::printf("MIGAN_IMAGE {\"fill_channels\":%zu,\"max_abs\":%d,\"mae\":%.17g,\"mse\":%.17g,\"psnr_db\":", count,
                    max_abs, mae, mse);
        if (std::isfinite(psnr))
            std::printf("%.17g", psnr);
        else
            std::printf("null");
        std::printf(",\"exact_fill\":%s,\"known_errors\":%zu,\"passed\":%s}\n", mse == 0 ? "true" : "false",
                    known_errors, pass ? "true" : "false");
    }
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--contracts") == 0) return contract_controls() ? 0 : 1;
    if (argc == 4 && std::strcmp(argv[1], "--reject") == 0) {
        set_gate("1");
        std::string error;
        std::unique_ptr<migan::context, decltype(&migan::free)> ctx(migan::init(argv[2], argv[3], 2, error),
                                                                    migan::free);
        std::printf("MIGAN_REJECT {\"rejected\":%s}\n", ctx ? "false" : "true");
        std::fprintf(stderr, "%s\n", error.c_str());
        return ctx ? 1 : 0;
    }
    if (argc != 8) {
        std::fprintf(stderr, "Usage: test-migan-diff MODEL.gguf REFERENCE.gguf CHECKPOINT_SHA256 RGB.raw FILL_MASK.raw "
                             "ORIGINAL_COMPOSITED.raw DUMP_DIR\n");
        return 2;
    }
    if (!contract_controls() || !gate_controls(argv[1], argv[3])) return 2;
    crispembed_diff::Ref reference;
    if (!reference.load(argv[2])) return 2;
    std::vector<uint8_t> rgb, mask, original_image;
    if (!bytes(argv[4], rgb, migan::rgb_elements) || !bytes(argv[5], mask, migan::mask_elements) ||
        !bytes(argv[6], original_image, migan::rgb_elements))
        return 2;
    std::filesystem::path directory(argv[7]);
    std::filesystem::create_directories(directory);
    std::string error;
    std::unique_ptr<migan::context, decltype(&migan::free)> ctx(migan::init(argv[1], argv[3], 2, error), migan::free);
    if (!ctx) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    std::vector<float> input(migan::input_elements), known(migan::mask_elements), raw(migan::rgb_elements),
        observed(raw.size());
    if (!migan::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), 256, 256, input.data(), input.size(),
                                error))
        return 2;
    for (size_t i = 0; i < known.size(); ++i) known[i] = mask[i] == 0 ? 1.0f : 0.0f;
    // Runtime input negatives reject before any graph allocation.
    if (migan::run(ctx.get(), input.data(), input.size() - 1, raw.data(), raw.size(), error)) return 2;
    const float saved = input[0];
    input[0] = std::numeric_limits<float>::infinity();
    if (migan::run(ctx.get(), input.data(), input.size(), raw.data(), raw.size(), error)) return 2;
    input[0] = saved;
    set_gate("0");
    if (migan::run(ctx.get(), input.data(), input.size(), raw.data(), raw.size(), error)) return 2;
    set_gate("1");

    std::set<std::string> compared;
    size_t failures = 0;
    std::string first_failure;
    auto compare = [&](const std::string & stage, const float * data, size_t count,
                       const std::array<int64_t, 4> & shape) {
        if (compared.count(stage)) return true;
        const auto expected = reference.get_f32(stage);
        if (!expected.first || expected.second != count) {
            std::fprintf(stderr, "Missing/count mismatch original stage: %s\n", stage.c_str());
            return false;
        }
        const auto expected_shape = reference.shape(stage);
        bool shape_match = expected_shape.size() <= 4;
        for (size_t axis = 0; axis < 4; ++axis)
            shape_match = shape_match && shape[axis] == (axis < expected_shape.size() ? expected_shape[axis] : 1);
        const bool finite = std::all_of(data, data + count, [](float value) { return std::isfinite(value); });
        const auto report = reference.compare(stage, data, count);
        // Direct frozen gates; CRISPEMBED_DIFF_COS_THRESHOLD cannot override.
        const bool pass = shape_match && finite && report.found && report.cos_min >= 0.99999f &&
                          report.cos_global >= 0.99999f && report.max_abs <= 0.002f;
        if (!pass) {
            ++failures;
            if (first_failure.empty()) first_failure = stage;
        }
        std::printf(
            "MIGAN_DIFF "
            "{\"stage\":\"%s\",\"count\":%zu,\"max_abs\":%.9g,\"mae\":%.9g,\"rms\":%.9g,\"cos_min\":%.9g,\"cos_"
            "global\":%.9g,\"native_l2\":%.9g,\"original_l2\":%.9g,\"finite\":%s,\"shape_match\":%s,\"passed\":%s}\n",
            stage.c_str(), count, report.max_abs, report.mean_abs, report.rms, report.cos_min, report.cos_global,
            report.mine_norm, report.ref_norm, finite ? "true" : "false", shape_match ? "true" : "false",
            pass ? "true" : "false");
        compared.insert(stage);
        return dump(directory / (stage + ".f32"), data, count * sizeof(float));
    };
    if (!compare("generator.input", input.data(), input.size(), { 256, 256, 4, 1 }) ||
        !compare("known_mask", known.data(), known.size(), { 256, 256, 1, 1 }))
        return 2;
    const auto start = std::chrono::steady_clock::now();
    if (!migan::run(ctx.get(), input.data(), input.size(), raw.data(), raw.size(), error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    size_t buffer_bytes = 0;
    bool neutrality = true;
    const bool full = reference.has("encoder.b256.fromrgb.output");
    if (full) {
        if (!migan::test::run_observed(ctx.get(), input.data(), input.size(), observed.data(), observed.size(), compare,
                                       buffer_bytes, error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 2;
        }
        neutrality = std::memcmp(raw.data(), observed.data(), raw.size() * sizeof(float)) == 0;
    } else if (!compare("generator.output", raw.data(), raw.size(), { 256, 256, 3, 1 }))
        return 2;
    for (const auto & stage : reference.tensor_names())
        if (!compared.count(stage)) {
            std::fprintf(stderr, "Original stage not compared: %s\n", stage.c_str());
            return 2;
        }
    std::vector<uint8_t> image(migan::rgb_elements), generated(image.size()), full_mask(mask.size(), 255);
    if (!migan::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(), image.data(),
                               image.size(), error) ||
        !migan::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), full_mask.data(), full_mask.size(),
                               generated.data(), generated.size(), error))
        return 2;
    if (!dump(directory / "composited.rgb8", image.data(), image.size()) ||
        !dump(directory / "generated.rgb8", generated.data(), generated.size()))
        return 2;
    const bool images = image_metrics(image, original_image, rgb, mask);
    std::printf("MIGAN_SUMMARY "
                "{\"compared\":%zu,\"expected\":%zu,\"failed_float_stages\":%zu,\"first_failure\":\"%s\",\"observer_"
                "neutrality_bit_exact\":%s,\"observed_graph_buffer_bytes\":%zu,\"generator_seconds\":%.9g,\"image_"
                "passed\":%s,\"passed\":%s}\n",
                compared.size(), reference.tensor_names().size(), failures, first_failure.c_str(),
                neutrality ? "true" : "false", buffer_bytes, seconds, images ? "true" : "false",
                failures == 0 && neutrality && images ? "true" : "false");
    return failures == 0 && neutrality && images ? 0 : 1;
}
