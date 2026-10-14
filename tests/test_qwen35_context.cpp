#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#include "crispembed_diff.h"
// The standalone experimental target supplies the image decoder normally
// provided by image_preprocess.cpp in the main library. No oracle inputs here.
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
extern "C" int qwen35_context_c_header_smoke(void);
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
static bool empty(qwen35_ocr_context * c) {
    int n = 99, m = 99;
    return !qwen35_ocr_token_ids(c, &n) && !qwen35_ocr_confidences(c, &m) && n == 0 && m == 0 &&
           qwen35_ocr_mean_confidence(c) == 0 && qwen35_ocr_last_stop(c) == QWEN35_OCR_ERROR &&
           *qwen35_ocr_last_error(c);
}
static bool compare(const char * label, qwen35_ocr_context * c, const char * text, int len,
                    const crispembed_diff::Ref & ref, int expected_count = -1) {
    auto expected = ref.get_f32("generated.ids");
    auto bytes = ref.get_f32("input.decoded_utf8");
    if (!expected.first || !bytes.first || !text || len < 0) return false;
    const int count = expected_count < 0 ? int(expected.second) : expected_count;
    int n = 0, m = 0;
    const int * ids = qwen35_ocr_token_ids(c, &n);
    const float * probabilities = qwen35_ocr_confidences(c, &m);
    bool ok = ids && probabilities && n == count && m == count;
    double maximum = 0, mean = 0, reference_mean = 0;
    if (ok)
        for (int i = 0; i < count; ++i) {
            ok &= ids[i] == int(expected.first[i]);
            auto scores = ref.get_f32("generated.scores." + std::to_string(i));
            if (!scores.first || scores.second != 248320) return false;
            const float highest = *std::max_element(scores.first, scores.first + scores.second);
            double total = 0;
            for (size_t j = 0; j < scores.second; ++j) total += std::exp(double(scores.first[j]) - highest);
            const double reference = 1.0 / total;
            maximum = std::max(maximum, std::abs(double(probabilities[i]) - reference));
            mean += probabilities[i];
            reference_mean += reference;
            ok &= std::isfinite(probabilities[i]) && probabilities[i] >= 0 && probabilities[i] <= 1;
        }
    if (expected_count < 0) {
        std::string oracle;
        for (size_t i = 0; i < bytes.second; ++i) oracle.push_back(char(bytes.first[i]));
        ok &= std::string(text, size_t(len)) == oracle && qwen35_ocr_last_stop(c) == QWEN35_OCR_EOS;
    } else {
        ok &= qwen35_ocr_last_stop(c) == QWEN35_OCR_MAX_TOKENS;
        // Frozen original IDs9419,77796 decode to "Hello OCR"; no EOS in prefix.
        ok &= std::string(text, size_t(len)) == (count == 1 ? "Hello" : "Hello OCR");
    }
    ok &=
        maximum <= 1e-6 && std::abs(qwen35_ocr_mean_confidence(c) - mean / count) < 1e-7 && !*qwen35_ocr_last_error(c);
    std::printf("%s CONTEXT %s ids=%d text_bytes=%d confidence_max_abs=%.9g mean=%.9g/%.9g stop=%d\n",
                ok ? "PASS" : "FAIL", label, n, len, maximum, mean / count, reference_mean / count,
                int(qwen35_ocr_last_stop(c)));
    return ok;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 7) return 2;
    bool ok = qwen35_context_c_header_smoke();
    gate("0");
    ok &= !qwen35_ocr_init(argv[1], 4);
    gate("1");
    ok &= !qwen35_ocr_init(nullptr, 4) && !qwen35_ocr_init(argv[1], 0) && !qwen35_ocr_init(argv[1], 257) &&
          !qwen35_ocr_init(argv[3], 4) && !qwen35_ocr_init(argv[6], 4);
    qwen35_ocr_free(nullptr);
    crispembed_diff::Ref ref, resized;
    if (!ref.load(argv[3]) || !resized.load(argv[5])) return 2;
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    owner a(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free), b(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free);
    if (!a || !b) {
        std::printf("FAIL context load\n");
        return 1;
    }
    int w = 0, h = 0, ch = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> rgb(stbi_load(argv[2], &w, &h, &ch, 3), stbi_image_free);
    if (!rgb) return 2;
    ok &= qwen35_ocr_get_max_tokens(a.get()) == 2048 && qwen35_ocr_set_max_tokens(a.get(), 12) &&
          qwen35_ocr_set_max_tokens(b.get(), 12) && !qwen35_ocr_set_max_tokens(a.get(), 0) &&
          !qwen35_ocr_set_max_tokens(a.get(), 4097) && qwen35_ocr_get_max_tokens(a.get()) == 12;
    int len = 99;
    const char * text = qwen35_ocr_recognize_raw(a.get(), rgb.get(), w, h, 3, nullptr, &len);
    ok &= compare("raw", a.get(), text, len, ref);
    text = qwen35_ocr_recognize_file(b.get(), argv[4], "", &len);
    ok &= compare("independent-resized-file", b.get(), text, len, resized);
    text = qwen35_ocr_recognize_file(a.get(), argv[2], nullptr, &len);
    ok &= compare("repeat-file", a.get(), text, len, ref);
    int retained = 0;
    const auto * retained_ids = qwen35_ocr_token_ids(b.get(), &retained);
    ok &= retained == 9 && retained_ids && retained_ids[8] == 248046 && qwen35_ocr_last_stop(b.get()) == QWEN35_OCR_EOS;
    ok &= qwen35_ocr_set_max_tokens(a.get(), 2);
    text = qwen35_ocr_recognize_raw(a.get(), rgb.get(), w, h, 3, nullptr, &len);
    ok &= compare("truncated", a.get(), text, len, ref, 2);
    std::vector<uint8_t> gray(size_t(w) * h), rgba(size_t(w) * h * 4);
    for (size_t i = 0; i < gray.size(); ++i) {
        gray[i] = rgb.get()[i * 3];
        for (int k = 0; k < 3; ++k) rgba[i * 4 + k] = rgb.get()[i * 3 + k];
        rgba[i * 4 + 3] = uint8_t(i % 256);
    }
    ok &= qwen35_ocr_set_max_tokens(a.get(), 1);
    text = qwen35_ocr_recognize_raw(a.get(), gray.data(), w, h, 1, nullptr, &len);
    ok &= compare("grayscale", a.get(), text, len, ref, 1);
    text = qwen35_ocr_recognize_raw(a.get(), rgba.data(), w, h, 4, nullptr, &len);
    ok &= compare("rgba-discard-alpha", a.get(), text, len, ref, 1);
    // Each rejection must invalidate the preceding successful result, not leak it.
    auto reject = [&](const uint8_t * p, int ww, int hh, int cc, const char * prompt) {
        len = 99;
        return !qwen35_ocr_recognize_raw(a.get(), p, ww, hh, cc, prompt, &len) && len == 0 && empty(a.get());
    };
    ok &= reject(nullptr, w, h, 3, nullptr) && reject(rgb.get(), 0, h, 3, nullptr) &&
          reject(rgb.get(), w, h, 2, nullptr) && reject(rgb.get(), 1024, 1024, 3, nullptr) &&
          reject(rgb.get(), w, h, 3, "<|image_pad|>");
    std::string enormous(65537, 'x');
    ok &= reject(rgb.get(), w, h, 3, enormous.c_str());
    len = 99;
    ok &= !qwen35_ocr_recognize_file(a.get(), "", nullptr, &len) && len == 0 && empty(a.get());
    gate("0");
    ok &= reject(rgb.get(), w, h, 3, nullptr);
    gate("1");
    ok &= qwen35_ocr_set_max_tokens(a.get(), 12);
    text = qwen35_ocr_recognize_raw(a.get(), rgb.get(), w, h, 3, nullptr, &len);
    ok &= compare("recovered-after-errors", a.get(), text, len, ref);
    len = 99;
    ok &= !qwen35_ocr_recognize_raw(nullptr, nullptr, 0, 0, 0, nullptr, &len) && len == 0;
    // Releasing A must not destroy B's result or configuration.
    a.reset();
    ok &= qwen35_ocr_get_max_tokens(b.get()) == 12;
    text = qwen35_ocr_recognize_file(b.get(), argv[4], nullptr, &len);
    ok &= compare("survivor-after-free", b.get(), text, len, resized);
    std::printf(
        "%s context lifecycle, native RGB/file/gray/RGBA, EOS/truncation/confidence, reset/invalid/master=0 controls\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
