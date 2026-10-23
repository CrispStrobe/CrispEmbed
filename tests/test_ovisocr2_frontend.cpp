#include "core/clean_exit.h"
#include "qwen35_frontend.h"
#include "crispembed_diff.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static bool exact(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<int> & actual) {
    const auto expected = ref.get_f32(name);
    if (!expected.first || expected.second != actual.size()) return false;
    for (size_t i = 0; i < actual.size(); ++i)
        if (expected.first[i] != actual[i]) return false;
    return true;
}
static std::string bytes(const crispembed_diff::Ref & ref, gguf_context * meta, const std::string & name) {
    const auto values = ref.get_f32(name);
    const size_t n = core_gguf::kv_u32(meta, (name + ".length").c_str(), 0);
    if (!values.first || n > values.second) return {};
    std::string text;
    for (size_t i = 0; i < n; ++i) text.push_back(char(values.first[i]));
    return text;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_OVISOCR2_FRONTEND", "1");
    crispembed_diff::Ref ref;
    auto * meta = core_gguf::open_metadata(argv[1]);
    qwen35_ocr::frontend_tokenizer tokenizer;
    if (!ref.load(argv[1]) || !meta || !tokenizer.load(meta)) return 2;
    bool ok = core_gguf::kv_u32(meta, "test.image_count", 0) == 3;
    for (int i = 0; i < 3; ++i) {
        const std::string label = "image." + std::to_string(i);
        const auto input = ref.get_f32(label + ".rgb"), expected = ref.get_f32(label + ".patches");
        const auto shape = ref.shape(label + ".rgb");
        if (!input.first || !expected.first || shape.size() != 3 || shape[0] != 3) return 2;
        std::vector<uint8_t> rgb(input.second);
        for (size_t j = 0; j < rgb.size(); ++j) rgb[j] = uint8_t(input.first[j]);
        qwen35_ocr::image_inputs image;
        bool match =
            qwen35_ocr::preprocess_rgb(rgb.data(), int(shape[2]), int(shape[1]), tokenizer.image_limits(), image);
        match &= exact(ref, label + ".grid", image.grid) && image.patches.size() == expected.second;
        double max_abs = 0, native_norm = 0, reference_norm = 0;
        if (image.patches.size() == expected.second)
            for (size_t j = 0; j < image.patches.size(); ++j) {
                max_abs = std::max(max_abs, std::abs(double(image.patches[j]) - expected.first[j]));
                native_norm += double(image.patches[j]) * image.patches[j];
                reference_norm += double(expected.first[j]) * expected.first[j];
            }
        match &= max_abs <= 1.2e-7;
        std::vector<int> ids, modalities;
        std::string rendered;
        match &= tokenizer.image_prompt("", int(image.patches.size() / 1536 / 4), ids, modalities, rendered) &&
                 exact(ref, label + ".ids", ids) && exact(ref, label + ".modalities", modalities) &&
                 exact(ref, label + ".mask", std::vector<int>(ids.size(), 1)) &&
                 rendered == bytes(ref, meta, label + ".rendered");
        std::printf("%s %s grid/IDs/masks/modalities/template_exact=%d patch_max_abs=%.9g norms=%.9g/%.9g\n",
                    match ? "PASS" : "FAIL", label.c_str(), match, max_abs, std::sqrt(native_norm),
                    std::sqrt(reference_norm));
        ok &= match;
    }
    std::vector<int> ids, types;
    std::string text;
    bool controls = tokenizer.eos() == 248046 && tokenizer.is_stop(248044) && tokenizer.is_stop(248046) &&
                    !tokenizer.is_stop(248045) && !tokenizer.is_stop(-1) &&
                    tokenizer.decode({ 248044, 248046 }, text) && text.empty() &&
                    !tokenizer.image_prompt("custom", 196, ids, types, text) && ids.empty() && types.empty() &&
                    text.empty() && !tokenizer.image_prompt("", 0, ids, types, text) &&
                    !tokenizer.image_prompt("", 1025, ids, types, text);
    // Wrong source/profile contracts must reset a previously loaded tokenizer.
    const std::vector<std::function<void(gguf_context *)>> corruptions = {
        [](auto * m) { gguf_set_val_str(m, "general.source.huggingface.repository", "unknown/model"); },
        [](auto * m) { gguf_set_val_str(m, "general.source.huggingface.revision", "wrong"); },
        [](auto * m) { gguf_set_val_u32(m, "qwen35_ocr.format_version", 1); },
        [](auto * m) { gguf_set_val_str(m, "qwen35_ocr.profile", "other"); },
        [](auto * m) { gguf_set_val_str(m, "qwen35_ocr.checkpoint_sha256", "wrong"); },
        [](auto * m) { gguf_set_val_str(m, "qwen35_ocr.profile.rendered_prompt", "wrong"); },
        [](auto * m) { gguf_set_val_bool(m, "qwen35_ocr.native_alias_qualified", true); },
        [](auto * m) { gguf_set_val_bool(m, "qwen35_ocr.image_max_qualified", true); },
        [](auto * m) { gguf_set_val_u32(m, "qwen35_ocr.image_min_pixels", 65536); },
        [](auto * m) { gguf_set_val_u32(m, "qwen35_ocr.image_max_pixels", 262144); },
        [](auto * m) { gguf_set_val_u32(m, "qwen35_ocr.generation_eos_id", 248046); },
        [](auto * m) { gguf_set_val_u32(m, "qwen35_ocr.primary_eos_id", 248044); },
        [](auto * m) { gguf_set_val_u32(m, "tokenizer.ggml.eos_token_id", 248044); },
        [](auto * m) {
            const int32_t id = 248046;
            gguf_set_arr_data(m, "qwen35_ocr.stop_token_ids", GGUF_TYPE_INT32, &id, 1);
        },
        [](auto * m) { gguf_remove_key(m, "qwen35_ocr.missing_source_files"); },
    };
    for (const auto & corrupt : corruptions) {
        auto * bad = gguf_init_empty();
        gguf_set_kv(bad, meta);
        corrupt(bad);
        controls &= tokenizer.load(meta) && !tokenizer.load(bad) && tokenizer.eos() == -1 &&
                    !tokenizer.encode("abc", ids) && ids.empty();
        gguf_free(bad);
    }
    controls &= tokenizer.load(meta);
    for (const char * name : { "CRISPEMBED_OVISOCR2_FRONTEND", "CRISPEMBED_QWEN35_OCR" }) {
        gate(name, "0");
        controls &= !tokenizer.encode("abc", ids) && ids.empty() && !tokenizer.decode({ 1 }, text) && text.empty() &&
                    !tokenizer.is_stop(248044) && !tokenizer.image_prompt("", 196, ids, types, text) && ids.empty() &&
                    types.empty() && text.empty() && !tokenizer.load(meta);
        gate(name, "1");
        controls &= tokenizer.load(meta);
    }
    qwen35_ocr::image_inputs output;
    const uint8_t dummy[3] = { 0, 0, 0 };
    controls &=
        !qwen35_ocr::preprocess_rgb(dummy, 1024, 1024, tokenizer.image_limits(), output) && output.patches.empty();
    std::printf("%s pinned profile mutations=%zu, dual stops, disabled/reset, custom prompt and resolution rejection\n",
                controls ? "PASS" : "FAIL", corruptions.size());
    core_gguf::free_metadata(meta);
    return ok && controls ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
