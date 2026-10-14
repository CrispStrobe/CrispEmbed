#include "core/clean_exit.h"
#include "qwen35_frontend.h"
#include "crispembed_diff.h"
#include "core/nfc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
static std::string bytes(const crispembed_diff::Ref & ref, gguf_context * metadata, const std::string & name) {
    const auto value = ref.get_f32(name);
    const size_t length = core_gguf::kv_u32(metadata, (name + ".length").c_str(), 0);
    if (!value.first || length > value.second) return {};
    std::string result;
    for (size_t i = 0; i < length; ++i) result.push_back(char(value.first[i]));
    return result;
}
static bool ints(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<int> & actual) {
    const auto value = ref.get_f32(name);
    if (!value.first || value.second != actual.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i)
        if (value.first[i] != actual[i]) return false;
    return true;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    gate("1");
    crispembed_diff::Ref ref;
    auto * meta = core_gguf::open_metadata(argv[1]);
    qwen35_ocr::frontend_tokenizer tokenizer;
    if (!ref.load(argv[1]) || !meta || !tokenizer.load(meta)) return 2;
    bool ok = true;
    const int image_count = int(core_gguf::kv_u32(meta, "test.image_count", 0));
    const int text_count = int(core_gguf::kv_u32(meta, "test.text_count", 0));
    if (image_count < 1 || image_count > 30 || text_count < 1 || text_count > 50) return 2;
    for (int i = 0; i < image_count; ++i) {
        const std::string label = "image." + std::to_string(i);
        const auto pixels = ref.get_f32(label + ".rgb"), expected = ref.get_f32(label + ".resized");
        const auto patches = ref.get_f32(label + ".patches"), limits = ref.get_f32(label + ".limits");
        const auto shape = ref.shape(label + ".rgb"), resized_shape = ref.shape(label + ".resized");
        if (!pixels.first || shape.size() != 3 || shape[0] != 3 || !limits.first || limits.second != 2 ||
            !expected.first || !patches.first || resized_shape.size() != 3)
            return 2;
        std::vector<uint8_t> rgb(pixels.second);
        for (size_t k = 0; k < rgb.size(); ++k) rgb[k] = uint8_t(pixels.first[k]);
        qwen35_ocr::image_inputs image;
        bool match = qwen35_ocr::preprocess_rgb(rgb.data(), int(shape[2]), int(shape[1]),
                                                { int(limits.first[0]), int(limits.first[1]) }, image);
        match &= image.resized_rgb.size() == expected.second && image.patches.size() == patches.second &&
                 ints(ref, label + ".grid", image.grid) && image.height == resized_shape[2] &&
                 image.width == resized_shape[1];
        size_t different = 0;
        double maximum = 0, native_norm = 0, reference_norm = 0;
        if (image.resized_rgb.size() == expected.second)
            for (size_t k = 0; k < expected.second; ++k) different += image.resized_rgb[k] != expected.first[k];
        if (image.patches.size() == patches.second)
            for (size_t k = 0; k < patches.second; ++k) {
                maximum = std::max(maximum, std::abs(double(image.patches[k]) - patches.first[k]));
                native_norm += double(image.patches[k]) * image.patches[k];
                reference_norm += double(patches.first[k]) * patches.first[k];
            }
        match &= different == 0 && maximum <= 1.2e-7;
        std::vector<int> ids, modalities;
        std::string rendered;
        match &= tokenizer.image_prompt(bytes(ref, meta, label + ".prompt"), int(image.patches.size() / 1536 / 4), ids,
                                        modalities, rendered) &&
                 ints(ref, label + ".ids", ids) && ints(ref, label + ".modalities", modalities) &&
                 rendered == bytes(ref, meta, label + ".rendered");
        std::printf("%s %s resized=%dx%d byte_differences=%zu patch_max_abs=%.9g norms=%.9g/%.9g "
                    "IDs/modalities/template_exact=%d\n",
                    match ? "PASS" : "FAIL", label.c_str(), image.height, image.width, different, maximum,
                    std::sqrt(native_norm), std::sqrt(reference_norm), match);
        ok &= match;
    }
    for (int i = 0; i < text_count; ++i) {
        const auto label = "text." + std::to_string(i);
        std::vector<int> ids;
        std::string decoded;
        bool match = tokenizer.encode(bytes(ref, meta, label + ".input"), ids) && ints(ref, label + ".ids", ids) &&
                     tokenizer.decode(ids, decoded) && decoded == bytes(ref, meta, label + ".decoded");
        std::printf("%s %s IDs/decode exact tokens=%zu\n", match ? "PASS" : "FAIL", label.c_str(), ids.size());
        ok &= match;
    }
    const int normalization_count = int(core_gguf::kv_u32(meta, "test.nfc_count", 0));
    if (normalization_count < 4000 || normalization_count > 10000) return 2;
    int normalization_errors = 0;
    for (int i = 0; i < normalization_count; ++i) {
        const auto label = "nfc." + std::to_string(i);
        if (core_nfc::normalize(bytes(ref, meta, label + ".input")) != bytes(ref, meta, label + ".expected")) {
            ++normalization_errors;
            if (normalization_errors <= 5) std::printf("FAIL NFC original Rust case %d\n", i);
        }
    }
    ok &= normalization_errors == 0;
    std::printf("%s NFC original Rust checks=%d errors=%d\n", normalization_errors == 0 ? "PASS" : "FAIL",
                normalization_count, normalization_errors);
    const auto partial = ref.get_f32("decode.partial.ids");
    std::vector<int> partial_ids;
    if (!partial.first) return 2;
    for (size_t i = 0; i < partial.second; ++i) partial_ids.push_back(int(partial.first[i]));
    std::string decoded;
    const bool partial_ok =
        tokenizer.decode(partial_ids, decoded) && decoded == bytes(ref, meta, "decode.partial.expected");
    ok &= partial_ok;
    std::printf("%s partial UTF8 token decode replacement exact\n", partial_ok ? "PASS" : "FAIL");
    qwen35_ocr::image_inputs image;
    uint8_t pixel[3] = { 0, 0, 0 };
    bool controls = !qwen35_ocr::preprocess_rgb(nullptr, 32, 32, {}, image) &&
                    !qwen35_ocr::preprocess_rgb(pixel, 0, 32, {}, image) &&
                    !qwen35_ocr::preprocess_rgb(pixel, 1, 201, {}, image) &&
                    !qwen35_ocr::preprocess_rgb(pixel, 256, 256, { 0, 1 }, image) &&
                    !qwen35_ocr::preprocess_rgb(pixel, 1024, 1024, {}, image) && image.patches.empty();
    std::vector<int> ids, types;
    std::string rendered;
    controls &= tokenizer.encode("", ids) && ids.empty() && !tokenizer.decode({ -1 }, rendered) && rendered.empty() &&
                !tokenizer.image_prompt("<|image_pad|>", 64, ids, types, rendered) && ids.empty() && types.empty();
    gate("0");
    controls &= !qwen35_ocr::preprocess_rgb(pixel, 32, 32, {}, image) && image.patches.empty() &&
                !tokenizer.encode("abc", ids) && ids.empty() && !tokenizer.decode({ 1 }, rendered) &&
                rendered.empty() && !tokenizer.load(meta);
    gate("1");
    std::printf("%s frontend bounds, missing input, marker injection, reset and master=0 controls\n",
                controls ? "PASS" : "FAIL");
    core_gguf::free_metadata(meta);
    return ok && controls ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
