// Hosted, weight-free comparison against actual pinned HF processor/tokenizer.
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "qwen35_frontend.h"
#include "qwen35_xiaomi0.h"
#include "crispembed_diff.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static std::string bytes(const crispembed_diff::Ref & ref, const std::string & key) {
    const auto data = ref.get_f32(key);
    std::string result;
    for (size_t i = 0; data.first && i < data.second; ++i) result.push_back(char(data.first[i]));
    return result;
}
static bool ints(const crispembed_diff::Ref & ref, const std::string & key, const std::vector<int> & actual) {
    const auto data = ref.get_f32(key);
    if (!data.first || actual.size() != data.second) return false;
    for (size_t i = 0; i < actual.size(); ++i)
        if (actual[i] != data.first[i]) return false;
    return true;
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    auto * meta = core_gguf::open_metadata(argv[1]);
    if (!meta) return 2;
    qwen35_ocr::frontend_tokenizer tokenizer;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "0");
    bool ok = !tokenizer.load(meta);
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "1");
    ok &= tokenizer.load(meta) && tokenizer.is_xiaomi0() && !tokenizer.is_ovisocr2() && !tokenizer.is_ovisocr3();
    ok &= tokenizer.is_stop(248046) && !tokenizer.is_stop(248044);
    // Existing combining-mark diagnostic must not change this exact profile.
    gate("CRISPEMBED_QWEN35_TOKENIZER_MARKS", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[1])) return 2;
    const int cases = int(core_gguf::kv_u32(meta, "test.xiaomi_cases", 0));
    ok &= cases == 4;
    for (int i = 0; i < cases; ++i) {
        const auto label = "case." + std::to_string(i);
        std::vector<int> ids, modalities;
        std::string rendered, decoded;
        const int image_tokens = int(core_gguf::kv_u32(meta, (label + ".image_tokens").c_str(), 0));
        bool match = tokenizer.image_prompt(bytes(ref, label + ".prompt"), image_tokens, ids, modalities, rendered) &&
                     ints(ref, label + ".ids", ids) && ints(ref, label + ".modalities", modalities) &&
                     rendered == bytes(ref, label + ".rendered");
        std::printf("%s XIAOMI0_FRONTEND task=%d tokens=%zu\n", match ? "PASS" : "FAIL", i, ids.size());
        ok &= match;
    }
    const int texts = int(core_gguf::kv_u32(meta, "test.xiaomi_texts", 0));
    ok &= texts == 6;
    for (int i = 0; i < texts; ++i) {
        const auto label = "text." + std::to_string(i);
        std::vector<int> ids;
        std::string decoded;
        ok &= tokenizer.encode(bytes(ref, label + ".input"), ids) && ints(ref, label + ".ids", ids) &&
              tokenizer.decode(ids, decoded) && decoded == bytes(ref, label + ".decoded");
    }
    const auto rgb = ref.get_f32("image.rgb"), patches = ref.get_f32("image.patches");
    const auto resized = ref.get_f32("image.resized");
    constexpr int image_height = 257, image_width = 359;
    std::vector<uint8_t> pixels;
    for (size_t i = 0; rgb.first && i < rgb.second; ++i) pixels.push_back(uint8_t(rgb.first[i]));
    qwen35_ocr::image_inputs image;
    bool image_ok = pixels.size() == image_height * image_width * 3 &&
                    qwen35_ocr::preprocess_rgb(pixels.data(), image_height, image_width, tokenizer.image_limits(), image);
    std::printf("XIAOMI0_IMAGE_GEOMETRY input_height=%d input_width=%d native_height=%d native_width=%d "
                "grid_match=%d patches=%zu expected_patches=%zu resized=%zu expected_resized=%zu\n",
                image_height, image_width, image.height, image.width, ints(ref, "image.grid", image.grid),
                image.patches.size(), patches.second, image.resized_rgb.size(), resized.second);
    image_ok &= ints(ref, "image.grid", image.grid) && patches.second == image.patches.size() &&
                resized.second == image.resized_rgb.size();
    if (image_ok) {
        double patch_max = 0, resized_max = 0;
        size_t patch_failures = 0, resized_changes = 0;
        for (size_t i = 0; i < patches.second; ++i) {
            const double error = std::abs(double(image.patches[i]) - patches.first[i]);
            patch_max = std::max(patch_max, error);
            patch_failures += error > 1.2e-7;
        }
        for (size_t i = 0; i < resized.second; ++i) {
            const double error = std::abs(double(image.resized_rgb[i]) - resized.first[i]);
            resized_max = std::max(resized_max, error);
            resized_changes += error != 0;
        }
        std::printf("XIAOMI0_IMAGE_DIFF resized_changes=%zu resized_max_abs=%.17g "
                    "patch_failures=%zu patch_max_abs=%.17g patch_budget=1.2e-7\n",
                    resized_changes, resized_max, patch_failures, patch_max);
        image_ok &= patch_failures == 0 && resized_changes == 0;
    }
    ok &= image_ok;
    std::vector<int> ids, modalities;
    std::string rendered;
    ok &= !tokenizer.image_prompt("invented task", 64, ids, modalities, rendered) && ids.empty() && rendered.empty();
    ok &= !tokenizer.image_prompt("", 0, ids, modalities, rendered);
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "0");
    ok &= !tokenizer.encode("text", ids) && !tokenizer.is_stop(248046);
    core_gguf::free_metadata(meta);
    std::printf("%s XIAOMI0_FRONTEND_COMPLETE image=%d\n", ok ? "PASS" : "FAIL", image_ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
