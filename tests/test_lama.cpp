// Model-free Big-LaMa preprocessing, compositing, bounds, and gate contracts.
// These tests run in hosted CI; no checkpoint or original framework is needed.
#include "lama.h"
#include "core/clean_exit.h"
#include "core/temp_file.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {
int checks = 0, failures = 0;
void check(bool value, const char * description) {
    ++checks;
    if (!value) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description);
    }
}

void set_gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LAMA", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_LAMA", value, 1);
    else
        unsetenv("CRISPEMBED_LAMA");
#endif
}

void dimensions() {
    std::string error;
    int width = -1, height = -2;
    check(lama::padded_dimensions(37, 29, width, height, error) && width == 40 && height == 32 && error.empty(),
          "odd image pads to eight");
    check(lama::padded_dimensions(1920, 1080, width, height, error) && width == 1920 && height == 1080,
          "1080p full frame is within pixel bound");
    check(lama::padded_dimensions(2048, 1024, width, height, error), "maximum padded pixel count accepted");
    check(!lama::padded_dimensions(2048, 1025, width, height, error), "padded count over bound rejected");
    check(!lama::padded_dimensions(2049, 16, width, height, error), "axis beyond bound rejected");
    check(!lama::padded_dimensions(8, 16, width, height, error), "one-cell reflected bottleneck rejected");
    check(!lama::padded_dimensions(-1, 16, width, height, error), "negative dimension rejected");
    check(lama::padded_dimensions(9, 9, width, height, error) && width == 16 && height == 16,
          "smallest supported frame pads to two-cell bottleneck");
}

void preprocessing() {
    const int width = 9, height = 11, pw = 16, ph = 16;
    const size_t pixels = size_t(width) * height, padded = size_t(pw) * ph;
    std::vector<uint8_t> rgb(3 * pixels), mask(pixels);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 17 + 19) % 256);
    mask[size_t(10) * width + 8] = 255;
    std::vector<float> input(4 * padded, -99.0f);
    std::string error;
    check(lama::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height, input.data(),
                                input.size(), error),
          "preprocessing succeeds");
    bool equal = true;
    // The table is independent of the production expression. Symmetric padding
    // includes the edge; reflect padding would start with 7 and 9 instead.
    const std::array<int, 16> xs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2 };
    const std::array<int, 16> ys = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6 };
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
            const size_t source = size_t(ys[y]) * width + xs[x], target = size_t(y) * pw + x;
            equal &= input[3 * padded + target] == (mask[source] ? 1.0f : 0.0f);
            for (size_t channel = 0; channel < 3; ++channel)
                equal &= input[channel * padded + target] ==
                         (mask[source] ? 0.0f : float(rgb[3 * source + channel]) / 255.0f);
        }
    check(equal, "RGB channel order, unit normalization, white fill, and symmetric padding exactly match");
    const std::vector<float> before = input;
    mask[5] = 128;
    check(!lama::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height, input.data(),
                                 input.size(), error) &&
              input == before,
          "nonbinary mask fails before touching preprocessing output");
    mask[5] = 0;
    check(!lama::preprocess_rgb8(rgb.data(), rgb.size() - 1, mask.data(), mask.size(), width, height, input.data(),
                                 input.size(), error),
          "truncated RGB rejected");
    check(!lama::preprocess_rgb8(rgb.data(), rgb.size(), mask.data(), mask.size(), width, height, input.data(),
                                 input.size() - 1, error),
          "wrong padded input count rejected");
    check(!lama::preprocess_rgb8(nullptr, rgb.size(), mask.data(), mask.size(), width, height, input.data(),
                                 input.size(), error),
          "null image rejected");
}

void composition() {
    const int width = 13, height = 9, pw = 16, ph = 16;
    const size_t pixels = size_t(width) * height, padded = size_t(pw) * ph;
    std::vector<uint8_t> rgb(3 * pixels), mask(pixels), output(3 * pixels, 42);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 11 + 37) % 256);
    mask[width + 2] = 255;
    mask[pixels - 1] = 255;
    std::vector<float> raw(3 * padded);
    std::fill(raw.begin(), raw.begin() + padded, -0.2f);
    std::fill(raw.begin() + padded, raw.begin() + 2 * padded, 0.5f);
    std::fill(raw.begin() + 2 * padded, raw.end(), 1.5f);
    std::string error;
    check(lama::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                               output.data(), output.size(), error),
          "composition succeeds");
    bool correct = true;
    const std::array<uint8_t, 3> generated = { 0, 127, 255 };
    for (size_t pixel = 0; pixel < pixels; ++pixel)
        for (size_t channel = 0; channel < 3; ++channel)
            correct &= output[3 * pixel + channel] == (mask[pixel] ? generated[channel] : rgb[3 * pixel + channel]);
    check(correct, "masked pixels clamp and truncate; all known pixels remain exact");
    std::vector<uint8_t> alias = rgb;
    check(lama::composite_rgb8(raw.data(), raw.size(), alias.data(), alias.size(), mask.data(), mask.size(), width,
                               height, alias.data(), alias.size(), error) &&
              alias == output,
          "in-place image composition is safe");
    const std::vector<uint8_t> before = output;
    raw.back() = std::numeric_limits<float>::quiet_NaN();
    check(!lama::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                                output.data(), output.size(), error) &&
              output == before,
          "nonfinite padded output fails transactionally even outside crop");
    raw.back() = 1;
    mask[1] = 1;
    check(!lama::composite_rgb8(raw.data(), raw.size(), rgb.data(), rgb.size(), mask.data(), mask.size(), width, height,
                                output.data(), output.size(), error) &&
              output == before,
          "nonbinary mask fails composition transactionally");
    mask[1] = 0;
    check(!lama::composite_rgb8(raw.data(), raw.size() - 1, rgb.data(), rgb.size(), mask.data(), mask.size(), width,
                                height, output.data(), output.size(), error),
          "truncated raw output rejected");
}

void gates() {
    std::string error;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        set_gate(value);
        check(lama::init("unused.gguf", 1, error) == nullptr && error.find("CRISPEMBED_LAMA=1") != std::string::npos,
              "unset, empty, and zero gates disable before model access");
    }
    set_gate("1");
    check(lama::init(nullptr, 1, error) == nullptr && error.find("model path") != std::string::npos,
          "null path rejected");
    check(lama::init("unused.gguf", 0, error) == nullptr && error.find("threads") != std::string::npos,
          "zero threads rejected before model access");
    check(lama::init("unused.gguf", 65, error) == nullptr && error.find("threads") != std::string::npos,
          "too many threads rejected before model access");
    std::array<uint8_t, 16 * 16 * 3> rgb{}, out{};
    std::array<uint8_t, 16 * 16> mask{};
    std::fill(out.begin(), out.end(), 129);
    check(!lama::process_rgb8(nullptr, rgb.data(), rgb.size(), mask.data(), mask.size(), 16, 16, out.data(), out.size(),
                              error) &&
              std::all_of(out.begin(), out.end(), [](uint8_t x) { return x == 129; }),
          "null context never changes output");
    lama::free(nullptr);
    set_gate(nullptr);
}

void metadata() {
    set_gate("1");
    const std::string path = core_tmp::make_private(".gguf");
    check(!path.empty(), "private model fixture path created");
    if (path.empty()) return;
    gguf_context * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "lama");
    gguf_set_val_u32(meta, "general.file_type", 0);
    gguf_set_val_u32(meta, "lama.schema.version", 1);
    gguf_set_val_str(meta, "lama.variant", "big-lama");
    gguf_set_val_str(meta, "lama.profile", "big-lama-18-075-v1");
    gguf_set_val_str(meta, "lama.source.commit", "786f5936b27fb3dacd2b1ad799e4de968ea697e7");
    gguf_set_val_str(meta, "lama.source.license", "Apache-2.0");
    gguf_set_val_str(meta, "lama.fft.norm", "ortho");
    gguf_set_val_str(meta, "lama.input.contract", "RGB8;binary-mask8;255=fill;symmetric-pad8");
    gguf_set_val_str(meta, "lama.tensor.layout", "original contiguous OIHW/IOHW; GGUF dimensions reversed");
    const std::string digest(64, 'a');
    gguf_set_val_str(meta, "lama.checkpoint.sha256", digest.c_str());
    gguf_set_val_str(meta, "lama.config.sha256", digest.c_str());
    auto reject = [&](const char * expected, const char * description) {
        check(gguf_write_to_file(meta, path.c_str(), true), "metadata fixture written");
        std::string error;
        lama::context * ctx = lama::init(path.c_str(), 1, error);
        check(!ctx && error.find(expected) != std::string::npos, description);
        lama::free(ctx);
    };
    reject("837", "valid metadata with missing inference tensors fails before weight allocation");
    gguf_set_val_u32(meta, "general.file_type", 1);
    reject("metadata", "F16 model profile without the lama-v1 quantization policy rejected");
    gguf_set_val_str(meta, "lama.quantization.policy", "lama-v2");
    reject("metadata", "unknown quantization policy rejected");
    gguf_set_val_str(meta, "lama.quantization.policy", "lama-v1");
    reject("837", "lama-v1 quantized header accepted; tensor inventory still enforced");
    gguf_set_val_u32(meta, "general.file_type", 0);
    reject("837", "F32 profile with an inert policy key keeps the strict F32 inventory check");
    gguf_remove_key(meta, "lama.quantization.policy");
    gguf_set_val_u64(meta, "lama.schema.version", uint64_t(1) + (uint64_t(1) << 32));
    reject("metadata", "wide integer cannot wrap into an accepted schema version");
    gguf_set_val_u32(meta, "lama.schema.version", 1);
    gguf_set_val_str(meta, "lama.profile", "nine-block-variant");
    reject("metadata", "alternate architecture profile rejected");
    gguf_set_val_str(meta, "lama.profile", "big-lama-18-075-v1");
    gguf_set_val_str(meta, "lama.checkpoint.sha256", "unknown");
    reject("metadata", "checkpoint provenance digest must be explicit lowercase SHA-256");
    gguf_free(meta);
    std::remove(path.c_str());
    set_gate(nullptr);
}

int crispembed_test_main() {
    dimensions();
    preprocessing();
    composition();
    gates();
    metadata();
    std::fprintf(stderr, "Big-LaMa boundary contracts: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
} // namespace

int main() {
    core_util::clean_exit(crispembed_test_main());
}
