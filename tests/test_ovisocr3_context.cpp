#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "qwen35_ocr.h"
#include "crispembed_diff.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static bool cleared(qwen35_ocr_context * context) {
    int n = -1, m = -1;
    return !qwen35_ocr_token_ids(context, &n) && !qwen35_ocr_confidences(context, &m) && n == 0 && m == 0 &&
           qwen35_ocr_last_stop(context) == QWEN35_OCR_ERROR && qwen35_ocr_mean_confidence(context) == 0 &&
           *qwen35_ocr_last_error(context);
}
static bool compare(qwen35_ocr_context * context, const char * text, int length, const crispembed_diff::Ref & ref,
                    const char * label) {
    const auto expected = ref.get_f32("generated.ids"), decoded = ref.get_f32("input.decoded_utf8");
    const auto decoded_length = ref.get_f32("input.decoded_length");
    if (!text || length < 0 || !expected.first || !expected.second || !decoded.first || !decoded_length.first ||
        decoded_length.second != 1 || !std::isfinite(decoded_length.first[0]) ||
        std::floor(decoded_length.first[0]) != decoded_length.first[0] || decoded_length.first[0] < 0 ||
        decoded_length.first[0] > decoded.second)
        return false;
    int n = 0, m = 0;
    const auto * actual = qwen35_ocr_token_ids(context, &n);
    const auto * confidences = qwen35_ocr_confidences(context, &m);
    std::printf("IDS %s:", label);
    if (actual)
        for (int i = 0; i < n; ++i) std::printf(" %d", actual[i]);
    std::printf("\nUTF8_HEX %s:", label);
    for (int i = 0; i < length; ++i) std::printf("%02x", static_cast<unsigned char>(text[i]));
    std::printf("\n");
    bool ok = actual && confidences && n == int(expected.second) && m == n;
    double max_confidence_error = 0;
    if (ok)
        for (int i = 0; i < n; ++i) {
            ok &= actual[i] == int(expected.first[i]);
            const auto scores = ref.get_f32("generated.scores." + std::to_string(i));
            if (!scores.first || scores.second != 248320) return false;
            const float maximum = *std::max_element(scores.first, scores.first + scores.second);
            double total = 0;
            for (size_t j = 0; j < scores.second; ++j) total += std::exp(double(scores.first[j]) - maximum);
            max_confidence_error = std::max(max_confidence_error, std::abs(double(confidences[i]) - 1.0 / total));
            ok &= std::isfinite(confidences[i]) && confidences[i] >= 0 && confidences[i] <= 1;
        }
    std::string oracle;
    for (int i = 0; i < int(decoded_length.first[0]); ++i) oracle.push_back(char(decoded.first[i]));
    const int last = int(expected.first[expected.second - 1]);
    const bool eos = last == 248044 || last == 248046;
    ok &= std::string(text, size_t(length)) == oracle &&
          qwen35_ocr_last_stop(context) == (eos ? QWEN35_OCR_EOS : QWEN35_OCR_MAX_TOKENS) &&
          max_confidence_error <= 1e-6;
    std::printf("%s OVIS3_CONTEXT %s ids=%d decoded_bytes=%d confidence_max_abs=%.9g stop=%d\n", ok ? "PASS" : "FAIL",
                label, n, length, max_confidence_error, int(qwen35_ocr_last_stop(context)));
    std::printf("DECODED %s %s\n", label, text);
    return ok;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[2]) == "--source-only") {
        // The reviewed metadata alone does not qualify a native forward.
        const char * gates[] = { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR3_FRONTEND",
                               "CRISPEMBED_OVISOCR3_INFERENCE" };
        for (const char * name : gates) gate(name, "1");
        auto * meta = core_gguf::open_metadata(argv[1]);
        const bool source_only = meta && core_gguf::kv_u32(meta, "qwen35_ocr.format_version", 0) == 3 &&
                                 core_gguf::kv_str(meta, "general.source.huggingface.repository", "") == "ATH-MaaS/OvisOCR3" &&
                                 core_gguf::kv_str(meta, "general.source.huggingface.revision", "") ==
                                     "5f6ce6932efee9ec10c30746db55bd780214b7fd" &&
                                 core_gguf::kv_str(meta, "qwen35_ocr.profile", "") == "ovisocr3" &&
                                 !core_gguf::kv_bool(meta, "qwen35_ocr.effective_stops_qualified", true) &&
                                 !core_gguf::kv_bool(meta, "qwen35_ocr.native_alias_qualified", true);
        if (meta) core_gguf::free_metadata(meta);
        auto * unexpected = source_only ? qwen35_ocr_init(argv[1], 2) : nullptr;
        const bool rejected = source_only && !unexpected &&
                              !qwen35_ocr_is_ovisocr3(nullptr) && !qwen35_ocr_is_ovisocr2(nullptr);
        if (unexpected) qwen35_ocr_free(unexpected);
        std::printf("%s OVIS3_SOURCE_ONLY_REJECTION\n", rejected ? "PASS" : "FAIL");
        return rejected ? 0 : 1;
    }
    // F16 default exercises two full contexts; F32 may select one for bounded hosted memory.
    if (argc != 4 && argc != 5) return 2;
    const bool single = argc == 5 && std::string(argv[4]) == "--one-context";
    if (argc == 5 && !single) return 2;
    const char * gates[] = { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR3_FRONTEND", "CRISPEMBED_OVISOCR3_INFERENCE" };
    for (const char * name : gates) gate(name, "1");
    bool ok = true;
    for (const char * name : gates) {
        gate(name, "0");
        ok &= qwen35_ocr_init(argv[1], 2) == nullptr;
        gate(name, "1");
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[3])) return 2;
    const auto expected = ref.get_f32("generated.ids");
    if (!expected.first || expected.second < 1 || expected.second > 64) return 2;
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    owner a(qwen35_ocr_init(argv[1], 2), qwen35_ocr_free);
    owner b(single ? nullptr : qwen35_ocr_init(argv[1], 2), qwen35_ocr_free);
    if (!a || (!single && !b)) return 2;
    ok &= qwen35_ocr_is_ovisocr3(a.get()) == 1 && qwen35_ocr_is_ovisocr3(nullptr) == 0 && qwen35_ocr_is_ovisocr2(a.get()) == 0;
    if (b) ok &= qwen35_ocr_is_ovisocr3(b.get()) == 1 && qwen35_ocr_is_ovisocr2(b.get()) == 0;
    int w = 0, h = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(stbi_load(argv[2], &w, &h, &channels, 3),
                                                                      stbi_image_free);
    if (!pixels) return 2;
    ok &= qwen35_ocr_set_max_tokens(a.get(), int(expected.second)) && !qwen35_ocr_set_max_tokens(a.get(), 0);
    int length = -1;
    auto run = [&](qwen35_ocr_context * context, const char * label) {
        const char * text = qwen35_ocr_recognize_raw(context, pixels.get(), w, h, 3, nullptr, &length);
        return compare(context, text, length, ref, label);
    };
    ok &= run(a.get(), "raw");
    ok &= run(a.get(), "repeat");
    if (b) {
        ok &= qwen35_ocr_set_max_tokens(b.get(), int(expected.second)) && run(b.get(), "independent-context");
    }
    auto reject = [&](const char * prompt) {
        length = -1;
        return !qwen35_ocr_recognize_raw(a.get(), pixels.get(), w, h, 3, prompt, &length) && length == 0 &&
               cleared(a.get());
    };
    ok &= reject("unqualified custom prompt");
    for (const char * name : gates) {
        gate(name, "0");
        ok &= reject(nullptr);
        gate(name, "1");
        ok &= run(a.get(), "gate-recovery");
    }
    if (b) {
        a.reset();
        int retained = 0;
        const auto * retained_ids = qwen35_ocr_token_ids(b.get(), &retained);
        ok &= retained_ids && retained == int(expected.second);
        if (retained_ids && retained == int(expected.second))
            for (int i = 0; i < retained; ++i) ok &= retained_ids[i] == int(expected.first[i]);
        ok &= run(b.get(), "survivor-after-free");
    }
    std::printf("%s OvisOCR3 native API IDs/raw text/stops/confidence/repeat/gates/recovery contexts=%d\n",
                ok ? "PASS" : "FAIL", single ? 1 : 2);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
