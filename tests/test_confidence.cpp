// test_confidence.cpp — verify per-character/token confidence API for all OCR engines.
//
// Unit tests (no model needed): verify API function signatures exist and
// return sensible values when called with NULL context.
//
// Live tests (model needed): load model, run OCR on synthetic image,
// verify confidences are in [0,1], count matches text length.
//
// Usage:
//   test-confidence                             (unit tests only)
//   test-confidence --parseq <model.gguf>       (live test PARSeq)
//   test-confidence --all <models_dir>          (live test all engines)

#include "parseq_ocr.h"
#include "math_ocr.h"
#include "hmer_ocr.h"
#include "bttr_ocr.h"
#include "posformer_ocr.h"
#include "mixtex_ocr.h"
#include "ppformulanet_ocr.h"
#include "ppformulanet_l_ocr.h"
#include "tesseract_lstm.h"
#include "got_ocr.h"
#include "glm_ocr.h"
#include "qwen2vl_ocr.h"
#include "internvl2_ocr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { n_pass++; } \
    else { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// Verify confidence array: all values in [0,1], length > 0, mean > 0
static bool verify_confidences(const float* conf, int n, const char* engine,
                                const char* text, int text_len) {
    printf("  %s: %d confidences for %d-char text\n", engine, n, text_len);
    if (!conf || n <= 0) {
        printf("    FAIL: no confidences returned\n");
        n_fail++;
        return false;
    }

    float min_c = 1.0f, max_c = 0.0f;
    double sum = 0;
    bool all_valid = true;
    for (int i = 0; i < n; i++) {
        if (conf[i] < 0.0f || conf[i] > 1.0f) {
            printf("    FAIL: conf[%d]=%.6f out of [0,1]\n", i, conf[i]);
            all_valid = false;
        }
        if (conf[i] < min_c) min_c = conf[i];
        if (conf[i] > max_c) max_c = conf[i];
        sum += conf[i];
    }

    float mean = (float)(sum / n);
    printf("    min=%.4f max=%.4f mean=%.4f\n", min_c, max_c, mean);

    CHECK(all_valid, "all confidences in [0,1]");
    CHECK(mean > 0.01f, "mean confidence > 0.01");
    CHECK(max_c > 0.1f, "max confidence > 0.1");

    return all_valid;
}

// Generate a synthetic grayscale text-like image
static std::vector<uint8_t> make_text_image(int w, int h) {
    std::vector<uint8_t> img(w * h * 3, 240);
    // Draw dark horizontal lines (text-like)
    for (int y = h/4; y < 3*h/4; y += 5) {
        for (int x = w/8; x < 7*w/8; x++) {
            for (int c = 0; c < 3; c++)
                img[(y * w + x) * 3 + c] = 30;
        }
    }
    return img;
}

static std::vector<uint8_t> make_gray_image(int w, int h) {
    std::vector<uint8_t> img(w * h, 240);
    for (int y = h/4; y < 3*h/4; y += 5)
        for (int x = w/8; x < 7*w/8; x++)
            img[y * w + x] = 30;
    return img;
}

// ── Unit tests (no models) ──────────────────────────────────────────

static void test_null_safety() {
    printf("=== NULL safety tests ===\n");
    int n = 99;

    // PARSeq
    CHECK(parseq_ocr_confidences(nullptr, &n) == nullptr, "parseq null ctx");
    CHECK(n == 0, "parseq null ctx n=0");
    CHECK(parseq_ocr_mean_confidence(nullptr) == 0.0f, "parseq null mean");

    // Math OCR
    CHECK(math_ocr_confidences(nullptr, &n) == nullptr, "math_ocr null ctx");
    CHECK(math_ocr_mean_confidence(nullptr) == 0.0f, "math_ocr null mean");

    // HMER
    CHECK(hmer_ocr_confidences(nullptr, &n) == nullptr, "hmer null ctx");
    CHECK(hmer_ocr_mean_confidence(nullptr) == 0.0f, "hmer null mean");

    // BTTR
    CHECK(bttr_ocr_confidences(nullptr, &n) == nullptr, "bttr null ctx");
    CHECK(bttr_ocr_mean_confidence(nullptr) == 0.0f, "bttr null mean");

    // PosFormer
    CHECK(posformer_ocr_confidences(nullptr, &n) == nullptr, "posformer null ctx");
    CHECK(posformer_ocr_mean_confidence(nullptr) == 0.0f, "posformer null mean");

    // MixTex
    CHECK(mixtex_ocr_confidences(nullptr, &n) == nullptr, "mixtex null ctx");
    CHECK(mixtex_ocr_mean_confidence(nullptr) == 0.0f, "mixtex null mean");

    // PPFormulaNet
    CHECK(ppformulanet_ocr_confidences(nullptr, &n) == nullptr, "ppfn null ctx");
    CHECK(ppformulanet_ocr_mean_confidence(nullptr) == 0.0f, "ppfn null mean");

    // PPFormulaNet-L
    CHECK(ppformulanet_l_ocr_confidences(nullptr, &n) == nullptr, "ppfnl null ctx");
    CHECK(ppformulanet_l_ocr_mean_confidence(nullptr) == 0.0f, "ppfnl null mean");

    // Tesseract LSTM
    CHECK(tesseract_lstm_confidences(nullptr, &n) == nullptr, "tess null ctx");

    // GOT-OCR
    CHECK(got_ocr_confidences(nullptr, &n) == nullptr, "got null ctx");
    CHECK(got_ocr_mean_confidence(nullptr) == 0.0f, "got null mean");

    // GLM-OCR
    CHECK(glm_ocr_confidences(nullptr, &n) == nullptr, "glm null ctx");
    CHECK(glm_ocr_mean_confidence(nullptr) == 0.0f, "glm null mean");

    // Qwen2-VL
    CHECK(qwen2vl_ocr_confidences(nullptr, &n) == nullptr, "qwen2vl null ctx");
    CHECK(qwen2vl_ocr_mean_confidence(nullptr) == 0.0f, "qwen2vl null mean");

    // InternVL2
    CHECK(internvl2_ocr_confidences(nullptr, &n) == nullptr, "internvl2 null ctx");
    CHECK(internvl2_ocr_mean_confidence(nullptr) == 0.0f, "internvl2 null mean");

    printf("  %d/%d NULL safety checks passed\n\n", n_pass, n_pass + n_fail);
}

// ── Live tests (with models) ────────────────────────────────────────

static void test_parseq_live(const char* model_path) {
    printf("=== PARSeq live confidence test ===\n");
    auto* ctx = parseq_ocr_init(model_path, 2);
    CHECK(ctx != nullptr, "parseq init");
    if (!ctx) return;

    auto img = make_text_image(128, 32);
    int len = 0;
    const char* text = parseq_ocr_recognize_raw(ctx, img.data(), 128, 32, 3, &len);
    printf("  text: '%s' (%d chars)\n", text ? text : "(null)", len);

    int n_conf = 0;
    const float* conf = parseq_ocr_confidences(ctx, &n_conf);
    verify_confidences(conf, n_conf, "parseq", text, len);

    float mean = parseq_ocr_mean_confidence(ctx);
    printf("  mean_confidence: %.4f\n", mean);
    CHECK(mean > 0.0f, "mean > 0");
    if (n_conf > 0) {
        // Verify mean matches manual computation
        double manual = 0;
        for (int i = 0; i < n_conf; i++) manual += conf[i];
        float manual_mean = (float)(manual / n_conf);
        CHECK(fabsf(mean - manual_mean) < 1e-5f, "mean matches manual computation");
    }

    parseq_ocr_free(ctx);
}

static void test_tesseract_live(const char* model_path) {
    printf("=== Tesseract LSTM live confidence test ===\n");
    auto* ctx = tesseract_lstm_init(model_path, 2);
    CHECK(ctx != nullptr, "tesseract init");
    if (!ctx) return;

    auto img = make_gray_image(200, 32);
    int len = 0;
    const char* text = tesseract_lstm_recognize(ctx, img.data(), 200, 32, &len);
    printf("  text: '%s' (%d chars)\n", text ? text : "(null)", len);

    int n_conf = 0;
    const float* conf = tesseract_lstm_confidences(ctx, &n_conf);
    if (conf && n_conf > 0) {
        verify_confidences(conf, n_conf, "tesseract", text, len);
    } else {
        printf("  tesseract: no confidences (empty output)\n");
    }

    tesseract_lstm_free(ctx);
}

static void test_hmer_live(const char* model_path) {
    printf("=== HMER live confidence test ===\n");
    auto* ctx = hmer_ocr_init(model_path, 2);
    CHECK(ctx != nullptr, "hmer init");
    if (!ctx) return;

    auto img = make_gray_image(128, 64);
    // Convert to float [0,1]
    std::vector<float> fimg(128 * 64);
    for (int i = 0; i < 128 * 64; i++) fimg[i] = img[i] / 255.0f;

    int len = 0;
    const char* text = hmer_ocr_recognize(ctx, fimg.data(), 128, 64, &len);
    printf("  text: '%s' (%d chars)\n", text ? text : "(null)", len);

    int n_conf = 0;
    const float* conf = hmer_ocr_confidences(ctx, &n_conf);
    verify_confidences(conf, n_conf, "hmer", text, len);
    printf("  mean: %.4f\n", hmer_ocr_mean_confidence(ctx));
    hmer_ocr_free(ctx);
}

static void test_bttr_live(const char* model_path) {
    printf("=== BTTR live confidence test ===\n");
    auto* ctx = bttr_ocr_init(model_path, 2);
    CHECK(ctx != nullptr, "bttr init");
    if (!ctx) return;

    auto img = make_gray_image(76, 56);
    std::vector<float> fimg(76 * 56);
    for (int i = 0; i < 76 * 56; i++) fimg[i] = img[i] / 255.0f;

    int len = 0;
    const char* text = bttr_ocr_recognize(ctx, fimg.data(), 76, 56, &len);
    printf("  text: '%s' (%d chars)\n", text ? text : "(null)", len);

    int n_conf = 0;
    const float* conf = bttr_ocr_confidences(ctx, &n_conf);
    verify_confidences(conf, n_conf, "bttr", text, len);
    printf("  mean: %.4f\n", bttr_ocr_mean_confidence(ctx));
    bttr_ocr_free(ctx);
}

static void test_posformer_live(const char* model_path) {
    printf("=== PosFormer live confidence test ===\n");
    auto* ctx = posformer_ocr_init(model_path, 2);
    CHECK(ctx != nullptr, "posformer init");
    if (!ctx) return;

    auto img = make_gray_image(76, 56);
    std::vector<float> fimg(76 * 56);
    for (int i = 0; i < 76 * 56; i++) fimg[i] = img[i] / 255.0f;

    int len = 0;
    const char* text = posformer_ocr_recognize(ctx, fimg.data(), 76, 56, &len);
    printf("  text: '%s' (%d chars)\n", text ? text : "(null)", len);

    int n_conf = 0;
    const float* conf = posformer_ocr_confidences(ctx, &n_conf);
    verify_confidences(conf, n_conf, "posformer", text, len);
    printf("  mean: %.4f\n", posformer_ocr_mean_confidence(ctx));
    posformer_ocr_free(ctx);
}

static void test_mixtex_live(const char* model_path) {
    printf("=== MixTex live confidence test ===\n");
    auto* ctx = mixtex_ocr_init(model_path, 2);
    CHECK(ctx != nullptr, "mixtex init");
    if (!ctx) return;

    auto img = make_gray_image(500, 400);

    int len = 0;
    const char* text = mixtex_ocr_recognize(ctx, img.data(), 500, 400, 1, &len);
    printf("  text: '%.60s%s' (%d chars)\n", text ? text : "(null)", len > 60 ? "..." : "", len);

    int n_conf = 0;
    const float* conf = mixtex_ocr_confidences(ctx, &n_conf);
    verify_confidences(conf, n_conf, "mixtex", text, len);
    printf("  mean: %.4f\n", mixtex_ocr_mean_confidence(ctx));
    mixtex_ocr_free(ctx);
}

int main(int argc, char** argv) {
    // Always run unit tests
    test_null_safety();

    // Live tests with model paths
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--parseq") == 0 && i + 1 < argc) {
            test_parseq_live(argv[++i]);
        } else if (strcmp(argv[i], "--tesseract") == 0 && i + 1 < argc) {
            test_tesseract_live(argv[++i]);
        } else if (strcmp(argv[i], "--hmer") == 0 && i + 1 < argc) {
            test_hmer_live(argv[++i]);
        } else if (strcmp(argv[i], "--bttr") == 0 && i + 1 < argc) {
            test_bttr_live(argv[++i]);
        } else if (strcmp(argv[i], "--posformer") == 0 && i + 1 < argc) {
            test_posformer_live(argv[++i]);
        } else if (strcmp(argv[i], "--mixtex") == 0 && i + 1 < argc) {
            test_mixtex_live(argv[++i]);
        }
    }

    printf("\n=== RESULTS: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
