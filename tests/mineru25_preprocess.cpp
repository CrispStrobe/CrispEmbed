// Original-input MinerU preprocessing comparison. No model weights/inference.
// This closes the blind spot in the source-patch-fed graph diff: the actual
// decoder, outer PIL resize/padding, fast uint8 resize, normalization, patch
// ordering and grid must agree before a graph result can establish parity.
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "image_preprocess.h"
#include "mineru_layout.h"
#include "mineru_model_profile.h"
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr double max_absolute_error = 2e-6;

size_t count_pixels(int width, int height) {
    if (width <= 0 || height <= 0 || uint64_t(width) * uint64_t(height) > 100000000ULL)
        throw std::invalid_argument("image dimensions must be positive and at most 100 megapixels");
    return size_t(width) * size_t(height);
}

void pil_resize(std::vector<uint8_t> & pixels, int & width, int & height, int new_width, int new_height) {
    if (width == new_width && height == new_height) return;
    std::vector<float> resized(count_pixels(new_width, new_height) * 3);
    image_preproc::resize_bicubic_pil_u8_hwc(pixels.data(), height, width, resized.data(), new_height, new_width, 3);
    pixels.resize(resized.size());
    for (size_t i = 0; i < resized.size(); ++i) {
        if (!std::isfinite(resized[i]) || resized[i] < 0 || resized[i] > 255 || resized[i] != std::floor(resized[i]))
            throw std::runtime_error("PIL uint8 resize produced a non-integral or invalid pixel");
        pixels[i] = uint8_t(resized[i]);
    }
    width = new_width;
    height = new_height;
}

void prepare(std::vector<uint8_t> & pixels, int & width, int & height, const std::string & task) {
    if (task == "layout") {
        // prepare_for_layout does not preserve aspect ratio or pad first.
        pil_resize(pixels, width, height, 1036, 1036);
        return;
    }
    // A full-page-as-content request uses resize_by_need; page extraction
    // subsequently crops/rotates before this stage and has its own pipeline QA.
    const auto plan = mineru::normalize_crop_geometry(width, height);
    if (width != plan.padded_width || height != plan.padded_height) {
        std::vector<uint8_t> padded(count_pixels(plan.padded_width, plan.padded_height) * 3, 255);
        for (int y = 0; y < height; ++y) {
            const size_t target = (size_t(y + plan.paste_y) * plan.padded_width + plan.paste_x) * 3;
            std::memcpy(padded.data() + target, pixels.data() + size_t(y) * width * 3, size_t(width) * 3);
        }
        pixels.swap(padded);
        width = plan.padded_width;
        height = plan.padded_height;
    }
    pil_resize(pixels, width, height, plan.output_width, plan.output_height);
}

int execute(int argc, char ** argv) {
    std::string image, reference, task = "text";
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing option value");
        const std::string value = argv[++i];
        if (option == "--image")
            image = value;
        else if (option == "--reference")
            reference = value;
        else if (option == "--task")
            task = value;
        else
            throw std::invalid_argument("unknown option: " + option);
    }
    if (image.empty() || reference.empty() ||
        (task != "text" && task != "layout" && task != "table" && task != "formula" && task != "image")) {
        std::cerr << "Usage: test-mineru25-preprocess --image IMAGE --reference ORIGINAL_F32.gguf "
                     "--task text|layout|table|formula|image\n";
        return 2;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(reference)) throw std::runtime_error("cannot load original reference GGUF");
    if (ref.meta("mineru25.model") != "opendatalab/MinerU2.5-Pro-2604-1.2B" ||
        ref.meta("mineru25.model_revision") != "d3f5e08d073c21466bbabe21c71bb1e9c2e595da" ||
        ref.meta("mineru25.blueprint_revision") != "c9797cde8f54cc48306196fc6aebe73ea31dd4a3" ||
        ref.meta("mineru25.processor").find("Fast") == std::string::npos || ref.meta("mineru25.task") != task ||
        ref.meta("mineru25.dtype") != "f32")
        throw std::runtime_error(
            "reference model/processor/task/dtype identity differs; require the pinned F32 original");
    const auto expected_grid = ref.get_f32("grid_thw");
    if (!expected_grid.first || expected_grid.second != 3 || ref.shape("grid_thw") != std::vector<int64_t>{ 3 })
        throw std::runtime_error("missing or incorrectly shaped grid_thw tensor");
    const auto expected = ref.get_f32("input_patches");
    if (!expected.first || !expected.second) throw std::runtime_error("missing input_patches tensor");

    int width = 0, height = 0, channels = 0;
    uint8_t * raw = stbi_load(image.c_str(), &width, &height, &channels, 4);
    if (!raw) throw std::runtime_error("input image decode failed");
    std::vector<uint8_t> pixels;
    try {
        pixels.resize(count_pixels(width, height) * 3);
        for (size_t i = 0; i < pixels.size() / 3; ++i)
            for (int c = 0; c < 3; ++c) pixels[3 * i + c] = raw[4 * i + c]; // PIL RGBA->RGB drops alpha.
    } catch (...) {
        stbi_image_free(raw);
        throw;
    }
    stbi_image_free(raw);
    const int original_width = width, original_height = height;
    prepare(pixels, width, height, task);
    image_preproc::config config;
    config.patch_size = 14;
    config.temporal_patch_size = 2;
    config.merge_size = 2;
    config.min_pixels = 50176;
    config.max_pixels = 1605632;
    for (int c = 0; c < 3; ++c) {
        config.mean[c] = image_preproc::kCLIPMean[c];
        config.std[c] = image_preproc::kCLIPStd[c];
    }
    image_preproc::result native;
    if (!mineru25_profile::preprocess_rgb(pixels.data(), height, width, 3, config, native))
        throw std::runtime_error("native MinerU fast preprocessing failed");
    bool grid_equal = true;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(expected_grid.first[i]) || expected_grid.first[i] != native.grid_thw[i]) grid_equal = false;
    }
    const std::vector<int64_t> shape{ native.row_dim, native.n_patches }; // GGUF fastest-varying dimension first.
    const bool shape_equal = ref.shape("input_patches") == shape && expected.second == native.patches.size();
    bool finite = shape_equal;
    double max_abs = 0, squared = 0, mean_abs = 0;
    size_t first_mismatch = native.patches.size(), differing = 0;
    if (shape_equal)
        for (size_t i = 0; i < native.patches.size(); ++i) {
            if (!std::isfinite(native.patches[i]) || !std::isfinite(expected.first[i])) {
                finite = false;
                continue;
            }
            const double error = std::abs(double(native.patches[i]) - expected.first[i]);
            max_abs = std::max(max_abs, error);
            squared += error * error;
            mean_abs += error;
            if (error > max_absolute_error) {
                ++differing;
                if (first_mismatch == native.patches.size()) first_mismatch = i;
            }
        }
    const bool pass = grid_equal && shape_equal && finite && max_abs <= max_absolute_error;
    std::cout.imbue(std::locale::classic());
    std::cout << std::setprecision(12)
              << "{\"schema\":\"crispembed.mineru25.preprocess.v1\",\"task\":" << mineru::json_quote(task)
              << ",\"original_width\":" << original_width << ",\"original_height\":" << original_height
              << ",\"prepared_width\":" << width << ",\"prepared_height\":" << height
              << ",\"resized_width\":" << native.resized_w << ",\"resized_height\":" << native.resized_h
              << ",\"grid_thw\":[" << native.grid_thw[0] << ',' << native.grid_thw[1] << ',' << native.grid_thw[2]
              << ']' << ",\"grid_equal\":" << (grid_equal ? "true" : "false")
              << ",\"shape_equal\":" << (shape_equal ? "true" : "false")
              << ",\"finite\":" << (finite ? "true" : "false") << ",\"patches\":" << native.n_patches
              << ",\"elements\":" << native.patches.size() << ",\"max_abs\":" << max_abs
              << ",\"max_abs_budget\":" << max_absolute_error
              << ",\"rms\":" << std::sqrt(squared / std::max<size_t>(1, native.patches.size()))
              << ",\"mean_abs\":" << mean_abs / std::max<size_t>(1, native.patches.size())
              << ",\"elements_above_budget\":" << differing << ",\"first_mismatch_index\":";
    if (first_mismatch == native.patches.size())
        std::cout << "null";
    else
        std::cout << first_mismatch;
    std::cout << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
    return pass ? 0 : 1;
}
} // namespace

int main(int argc, char ** argv) {
    int code = 2;
    try {
        code = execute(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "MinerU preprocessing: " << error.what() << '\n';
    }
    core_util::clean_exit(code);
}
