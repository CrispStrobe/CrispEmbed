#include "qwen2vl_ocr.h"
#include "image_preprocess.h"
#include "core/clean_exit.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int crispembed_test_main() {
    image_preproc::config config;
    config.patch_size = 14;
    config.temporal_patch_size = 2;
    config.merge_size = 2;
    config.min_pixels = 784;
    config.max_pixels = 12845056;
    for (int c = 0; c < 3; ++c) {
        config.mean[c] = image_preproc::kCLIPMean[c];
        config.std[c] = image_preproc::kCLIPStd[c];
    }
    int failures = 0;
    auto check = [&](bool pass, const char * name) {
        std::printf("%s %s\n", pass ? "PASS" : "FAIL", name);
        failures += !pass;
    };
    image_preproc::result out;
    std::vector<uint8_t> square(98 * 98 * 3, 127);
    check(qwen2vl_ocr::preprocess_teleocr(square.data(), 70, 70, 3, config, out) && out.resized_h == 56 &&
              out.resized_w == 56,
          "Python half-to-even rounds 2.5 to 2");
    check(qwen2vl_ocr::preprocess_teleocr(square.data(), 98, 98, 3, config, out) && out.resized_h == 112 &&
              out.resized_w == 112,
          "Python half-to-even rounds 3.5 to 4");

    constexpr int width = 56, height = 28;
    std::vector<uint8_t> rgb(width * height * 3), rgba(width * height * 4), gray(width * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int i = y * width + x;
            gray[i] = (uint8_t)(x + 3 * y);
            for (int c = 0; c < 3; ++c) rgba[i * 4 + c] = rgb[i * 3 + c] = (uint8_t)(x + 3 * y + 17 * c);
            rgba[i * 4 + 3] = (uint8_t)((i * 7) % 256);
        }
    check(qwen2vl_ocr::preprocess_teleocr(rgb.data(), height, width, 3, config, out) && out.grid_thw[0] == 1 &&
              out.grid_thw[1] == 2 && out.grid_thw[2] == 4 && out.n_patches == 8 && out.row_dim == 1176,
          "natural image grid and patch width");
    const auto expected = [&](int y, int x, int c) {
        return ((float)((double)rgb[(y * width + x) * 3 + c] / 255.0) - config.mean[c]) / config.std[c];
    };
    check(out.patches[0] == expected(0, 0, 0) && out.patches[1176] == expected(0, 14, 0) &&
              out.patches[2 * 1176] == expected(14, 0, 0) && out.patches[4 * 1176] == expected(0, 28, 0),
          "2 by 2 merge-block order");
    check(out.patches[196] == out.patches[0] && out.patches[392] == expected(0, 0, 1) &&
              out.patches[1175] == expected(13, 13, 2),
          "channel then repeated temporal frame then patch pixels");
    const auto original = out.patches;
    check(qwen2vl_ocr::preprocess_teleocr(rgba.data(), height, width, 4, config, out) && out.patches == original,
          "RGBA conversion drops alpha without altering RGB");
    check(qwen2vl_ocr::preprocess_teleocr(gray.data(), height, width, 1, config, out) &&
              out.patches[0] == (0.0f - config.mean[0]) / config.std[0] &&
              out.patches[392] == (0.0f - config.mean[1]) / config.std[1],
          "grayscale expands to RGB before channel normalization");

    // Independently frozen with Pillow 13.0.0 Image.resize(BICUBIC).
    // Hash the reconstructed uint8 patch stream so the regression covers
    // every resized pixel without depending on floating-point byte layout.
    struct resize_case {
        int height, width;
        uint64_t hash;
        const char * name;
    };
    const resize_case cases[] = {
        { 41, 53, UINT64_C(0xcf8f4c7476cc4ca1), "Pillow 13 vertical-first mixed resize" },
        { 50, 51, UINT64_C(0x049ef7e136a64405), "Pillow 13 horizontal-first upscale" },
        { 62, 59, UINT64_C(0x2c29c6b77617ba15), "Pillow 13 equal reduction threshold stays horizontal-first" },
        { 63, 59, UINT64_C(0x2d9493072a130bfd), "Pillow 13 above reduction threshold becomes vertical-first" },
    };
    for (const auto & sample : cases) {
        std::vector<uint8_t> pattern((size_t)sample.height * sample.width * 3);
        for (int y = 0; y < sample.height; ++y)
            for (int x = 0; x < sample.width; ++x)
                for (int c = 0; c < 3; ++c)
                    pattern[((size_t)y * sample.width + x) * 3 + c] =
                        (uint8_t)((x * 17 + y * 53 + c * 81 + ((x * y) % 47) * 7) % 256);
        const bool ok = qwen2vl_ocr::preprocess_teleocr(pattern.data(), sample.height, sample.width, 3, config, out);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < out.patches.size(); ++i) {
            const int channel = (int)(i % 1176) / 392;
            const auto value =
                (uint8_t)std::lround((out.patches[i] * config.std[channel] + config.mean[channel]) * 255);
            hash = (hash ^ value) * UINT64_C(1099511628211);
        }
        check(ok && hash == sample.hash, sample.name);
    }
    check(!qwen2vl_ocr::preprocess_teleocr(rgb.data(), 0, width, 3, config, out) &&
              !qwen2vl_ocr::preprocess_teleocr(rgb.data(), height, width, 2, config, out) &&
              !qwen2vl_ocr::preprocess_teleocr(rgb.data(), 1, 201, 3, config, out),
          "reject invalid dimensions, channels and extreme aspect ratio");
    config.max_pixels = 0;
    check(!qwen2vl_ocr::preprocess_teleocr(rgb.data(), height, width, 3, config, out), "reject invalid pixel budget");
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
