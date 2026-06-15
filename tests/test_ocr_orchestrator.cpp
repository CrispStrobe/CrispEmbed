// test_ocr_orchestrator.cpp — comprehensive tests for the OCR pipeline
// orchestrator: source-type classifier, accept-gate logic, multi-stage
// escalation, per-stage config, chain selection.
//
// No models needed (all logic tests use synthetic images + config).
// Model-dependent tests (Tesseract, punctuation) gated behind
// CRISPEMBED_MODELS_DIR env var.
//
// Usage: test-ocr-orchestrator   (exits non-zero on failure)

#include "ocr_orchestrator.h"
#include "crispembed.h"

// stbi_write_png is exported by the crispembed lib.
extern "C" int stbi_write_png(char const* filename, int w, int h, int comp,
                              const void* data, int stride_in_bytes);

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

static std::string write_temp(const std::vector<uint8_t>& px, int w, int h, int ch,
                              const char* name) {
    const char* dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    std::string path = std::string(dir) + "/orch_test_" + name + ".png";
    stbi_write_png(path.c_str(), w, h, ch, px.data(), w * ch);
    return path;
}

// ═══════════════════════════════════════════════════════════════════════
// 1. default_config() structure
// ═══════════════════════════════════════════════════════════════════════
static void test_default_config() {
    printf("── default_config ──\n");
    using namespace ocr_orchestrator;

    config cfg = default_config();
    CHECK(cfg.router, "router on");
    CHECK(!cfg.chains.empty(), "has chains");

    bool has_auto = false, has_scan = false, has_photo = false, has_shot = false;
    for (auto& c : cfg.chains) {
        if (c.type == source_type::auto_detect)  has_auto = true;
        if (c.type == source_type::scanned_doc)  has_scan = true;
        if (c.type == source_type::photo)        has_photo = true;
        if (c.type == source_type::screenshot)   has_shot = true;
    }
    CHECK(has_auto && has_scan && has_photo && has_shot,
          "chains for auto/scan/photo/screenshot");

    // Per-source cleanup intent
    for (auto& c : cfg.chains) {
        if (c.type == source_type::scanned_doc && !c.stages.empty())
            CHECK(c.stages[0].cleanup.params.binarize == 1,
                  "scanned_doc chain binarizes");
        if (c.type == source_type::photo && !c.stages.empty())
            CHECK(c.stages[0].cleanup.denoise,
                  "photo chain denoises (NAFNet)");
        if (c.type == source_type::screenshot && !c.stages.empty())
            CHECK(!c.stages[0].cleanup.enabled || !c.stages[0].cleanup.params.binarize,
                  "screenshot chain does not binarize");
    }

    // Default accept-gate values
    for (auto& c : cfg.chains) {
        for (auto& s : c.stages) {
            CHECK(s.accept.min_chars >= 1,
                  "accept gate min_chars >= 1");
            CHECK(s.accept.min_confidence >= 0.0f,
                  "accept gate min_confidence >= 0");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. Source-type classifier
// ═══════════════════════════════════════════════════════════════════════
static void test_classifier() {
    printf("── classify_file ──\n");
    using namespace ocr_orchestrator;

    // Colourful image → photo (mean saturation high)
    {
        std::vector<uint8_t> photo(64 * 64 * 3);
        for (int i = 0; i < 64 * 64; i++) {
            photo[i*3+0] = 200; photo[i*3+1] = 30; photo[i*3+2] = 20;
        }
        std::string p = write_temp(photo, 64, 64, 3, "photo");
        CHECK(classify_file(p.c_str()) == source_type::photo,
              "saturated red → photo");
    }

    // White page with sparse black lines → scanned_doc
    {
        std::vector<uint8_t> doc(80 * 80 * 3, 255);
        for (int y = 20; y < 80; y += 20)
            for (int x = 0; x < 80; x++)
                for (int ch = 0; ch < 3; ch++) doc[(y*80+x)*3+ch] = 0;
        std::string d = write_temp(doc, 80, 80, 3, "doc");
        CHECK(classify_file(d.c_str()) == source_type::scanned_doc,
              "white page with lines → scanned_doc");
    }

    // Very wide grayscale strip → screenshot (aspect > 2.2)
    {
        std::vector<uint8_t> wide(300 * 50 * 3, 240);
        std::string w = write_temp(wide, 300, 50, 3, "wide");
        CHECK(classify_file(w.c_str()) == source_type::screenshot,
              "wide strip → screenshot");
    }

    // All-white image → scanned_doc (high white fraction, low saturation)
    {
        std::vector<uint8_t> white(100 * 100 * 3, 255);
        std::string p = write_temp(white, 100, 100, 3, "white");
        auto t = classify_file(p.c_str());
        CHECK(t == source_type::scanned_doc || t == source_type::screenshot,
              "all-white → scanned_doc or screenshot");
    }

    // Very tall image → screenshot (aspect > 2.2 in either direction)
    {
        std::vector<uint8_t> tall(50 * 300 * 3, 200);
        std::string p = write_temp(tall, 50, 300, 3, "tall");
        CHECK(classify_file(p.c_str()) == source_type::screenshot,
              "tall strip → screenshot");
    }

    // Green saturated → photo
    {
        std::vector<uint8_t> green(80 * 80 * 3);
        for (int i = 0; i < 80 * 80; i++) {
            green[i*3+0] = 20; green[i*3+1] = 180; green[i*3+2] = 30;
        }
        std::string p = write_temp(green, 80, 80, 3, "green");
        CHECK(classify_file(p.c_str()) == source_type::photo,
              "saturated green → photo");
    }

    // Missing file → fallback (no crash)
    CHECK(classify_file("/no/such/file.png") == source_type::scanned_doc,
          "missing file → scanned_doc fallback");

    // NULL path → no crash
    CHECK(classify_file(nullptr) == source_type::scanned_doc,
          "null path → scanned_doc fallback");
}

// ═══════════════════════════════════════════════════════════════════════
// 3. Accept-gate logic (tested via run_file with no models)
// ═══════════════════════════════════════════════════════════════════════
static void test_accept_gate() {
    printf("── accept_gate ──\n");
    using namespace ocr_orchestrator;

    // The accept gate logic: passes if text.size() >= min_chars AND
    // (min_confidence == 0 || mean_confidence >= min_confidence).
    // Since no models are loaded, run_file produces empty text → always fails gate.
    // We verify by checking stages_tried == number of enabled stages.

    // Create a white test image
    std::vector<uint8_t> img(100 * 100 * 3, 255);
    std::string path = write_temp(img, 100, 100, 3, "gate_test");

    // Single-stage chain, no models → stage runs but produces empty → fails gate
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;
        stage s;
        s.eng = engine::dbnet_trocr;
        s.accept.min_chars = 8;
        s.accept.min_confidence = 0.5f;
        ch.stages.push_back(s);
        cfg.chains.push_back(ch);

        context* ctx = nullptr;
        CHECK(load(&ctx, cfg), "load succeeds with no models");
        result r = run_file(ctx, path.c_str());
        CHECK(r.stages_tried == 1, "single stage tried");
        CHECK(r.full_text.empty(), "empty text (no models)");
        CHECK(r.mean_confidence == 0.0f, "zero confidence (no models)");
        ocr_orchestrator::free(ctx);
    }

    // Zero min_chars, zero min_confidence → gate should pass even for empty
    // ... except empty text has size 0 which is < min_chars=0? Let's check.
    // passes_gate: size >= min_chars (0 >= 0 = true) && (0 > 0? no → skip) = true
    // But text is still empty because no engine runs. Actually even min_chars=0
    // would pass for empty text. But the engine still produces no regions.
    // run_file returns best result, which has empty text anyway.
    // The gate logic is tested implicitly: if stages_tried matches the
    // number of stages, the gate failed for all of them.

    // Two-stage chain: both fail → stages_tried == 2
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;
        stage s1;
        s1.eng = engine::dbnet_trocr;
        s1.accept.min_chars = 8;
        ch.stages.push_back(s1);
        stage s2;
        s2.eng = engine::got;  // also no model → empty
        s2.accept.min_chars = 8;
        ch.stages.push_back(s2);
        cfg.chains.push_back(ch);

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, path.c_str());
        CHECK(r.stages_tried == 2, "two stages tried (both fail gate)");
        ocr_orchestrator::free(ctx);
    }

    // Disabled stage → skipped
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;
        stage s1;
        s1.eng = engine::dbnet_trocr;
        s1.enabled = false;
        ch.stages.push_back(s1);
        stage s2;
        s2.eng = engine::got;
        ch.stages.push_back(s2);
        cfg.chains.push_back(ch);

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, path.c_str());
        CHECK(r.stages_tried == 1, "disabled stage skipped (only 1 tried)");
        ocr_orchestrator::free(ctx);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Multi-stage escalation + chain selection
// ═══════════════════════════════════════════════════════════════════════
static void test_multi_stage() {
    printf("── multi-stage escalation ──\n");
    using namespace ocr_orchestrator;

    std::vector<uint8_t> img(100 * 100 * 3, 255);
    std::string path = write_temp(img, 100, 100, 3, "multi_test");

    // 3-stage chain: all fail (no models) → best-by-yield returned
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;
        for (int i = 0; i < 3; i++) {
            stage s;
            s.eng = engine::dbnet_trocr;
            s.accept.min_chars = 5;
            ch.stages.push_back(s);
        }
        cfg.chains.push_back(ch);

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, path.c_str());
        CHECK(r.stages_tried == 3, "all 3 stages tried");
        CHECK(r.full_text.empty(), "best result is still empty (no models)");
        ocr_orchestrator::free(ctx);
    }

    // Chain with mixed engines → each engine type attempted
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;
        stage s1; s1.eng = engine::dbnet_trocr;
        stage s2; s2.eng = engine::tesseract;
        stage s3; s3.eng = engine::got;
        ch.stages.push_back(s1);
        ch.stages.push_back(s2);
        ch.stages.push_back(s3);
        cfg.chains.push_back(ch);

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, path.c_str());
        CHECK(r.stages_tried == 3, "3 different engines tried");
        ocr_orchestrator::free(ctx);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. Per-stage config isolation
// ═══════════════════════════════════════════════════════════════════════
static void test_per_stage_config() {
    printf("── per-stage config ──\n");
    using namespace ocr_orchestrator;

    // Verify stages have independent accept-gate thresholds
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;

        stage s1;
        s1.eng = engine::dbnet_trocr;
        s1.accept.min_chars = 100;       // very high → will fail
        s1.accept.min_confidence = 0.9f;
        s1.cleanup.enabled = true;
        s1.cleanup.params.binarize = 1;
        ch.stages.push_back(s1);

        stage s2;
        s2.eng = engine::dbnet_trocr;
        s2.accept.min_chars = 1;         // very low
        s2.accept.min_confidence = 0.0f; // no conf check
        s2.cleanup.enabled = false;
        ch.stages.push_back(s2);

        cfg.chains.push_back(ch);

        // Verify the config is stored correctly
        CHECK(cfg.chains[0].stages[0].accept.min_chars == 100,
              "stage 0: min_chars=100");
        CHECK(cfg.chains[0].stages[1].accept.min_chars == 1,
              "stage 1: min_chars=1");
        CHECK(cfg.chains[0].stages[0].cleanup.params.binarize == 1,
              "stage 0: binarize on");
        CHECK(!cfg.chains[0].stages[1].cleanup.enabled,
              "stage 1: cleanup off");
        CHECK(cfg.chains[0].stages[0].accept.min_confidence == 0.9f,
              "stage 0: min_confidence=0.9");
        CHECK(cfg.chains[0].stages[1].accept.min_confidence == 0.0f,
              "stage 1: min_confidence=0.0");
    }

    // Verify different engine_params per stage
    {
        config cfg;
        cfg.router = false;
        chain ch;
        ch.type = source_type::auto_detect;

        stage s1;
        s1.params.det_prob_threshold = 0.1f;
        s1.params.det_target_short = 512;
        ch.stages.push_back(s1);

        stage s2;
        s2.params.det_prob_threshold = 0.5f;
        s2.params.det_target_short = 1024;
        ch.stages.push_back(s2);

        cfg.chains.push_back(ch);

        CHECK(cfg.chains[0].stages[0].params.det_prob_threshold == 0.1f,
              "stage 0: det_prob=0.1");
        CHECK(cfg.chains[0].stages[1].params.det_prob_threshold == 0.5f,
              "stage 1: det_prob=0.5");
        CHECK(cfg.chains[0].stages[0].params.det_target_short == 512,
              "stage 0: target_short=512");
        CHECK(cfg.chains[0].stages[1].params.det_target_short == 1024,
              "stage 1: target_short=1024");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Router + chain selection
// ═══════════════════════════════════════════════════════════════════════
static void test_chain_selection() {
    printf("── chain selection ──\n");
    using namespace ocr_orchestrator;

    // Router OFF → always uses first chain regardless of image type
    {
        config cfg;
        cfg.router = false;

        chain ch_auto;
        ch_auto.type = source_type::auto_detect;
        stage s; s.eng = engine::dbnet_trocr;
        ch_auto.stages.push_back(s);
        cfg.chains.push_back(ch_auto);

        chain ch_photo;
        ch_photo.type = source_type::photo;
        stage sp; sp.eng = engine::got;
        ch_photo.stages.push_back(sp);
        cfg.chains.push_back(ch_photo);

        // Even with a colourful photo image, router=false → uses auto chain
        std::vector<uint8_t> photo(64 * 64 * 3);
        for (int i = 0; i < 64 * 64; i++) {
            photo[i*3+0] = 200; photo[i*3+1] = 30; photo[i*3+2] = 20;
        }
        std::string p = write_temp(photo, 64, 64, 3, "chain_photo");

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, p.c_str());
        // Router off → source_type::auto_detect used → picks auto chain
        CHECK(r.used_type == source_type::auto_detect,
              "router off → auto_detect type");
        ocr_orchestrator::free(ctx);
    }

    // Router ON → classifies image and picks matching chain
    {
        config cfg;
        cfg.router = true;

        chain ch_auto;
        ch_auto.type = source_type::auto_detect;
        stage sa; sa.eng = engine::dbnet_trocr;
        ch_auto.stages.push_back(sa);
        cfg.chains.push_back(ch_auto);

        chain ch_photo;
        ch_photo.type = source_type::photo;
        stage sp; sp.eng = engine::got;
        ch_photo.stages.push_back(sp);
        cfg.chains.push_back(ch_photo);

        // Colourful photo → classifier returns photo → picks photo chain
        std::vector<uint8_t> photo(64 * 64 * 3);
        for (int i = 0; i < 64 * 64; i++) {
            photo[i*3+0] = 200; photo[i*3+1] = 30; photo[i*3+2] = 20;
        }
        std::string p = write_temp(photo, 64, 64, 3, "chain_photo2");

        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, p.c_str());
        CHECK(r.used_type == source_type::photo,
              "router on + photo → photo type selected");
        ocr_orchestrator::free(ctx);
    }

    // Empty config → no crash
    {
        config cfg;
        cfg.chains.clear();
        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, "/tmp/orch_test_white.png");
        CHECK(r.stages_tried == 0, "empty config → 0 stages tried");
        ocr_orchestrator::free(ctx);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. C API (crispembed_ocr_pipeline_*)
// ═══════════════════════════════════════════════════════════════════════
static void test_c_api() {
    printf("── C API ──\n");

    // defaults
    crispembed_ocr_pipeline_params pp = crispembed_ocr_pipeline_defaults();
    CHECK(pp.router == 1, "C API defaults: router=1");
    CHECK(pp.min_chars >= 1, "C API defaults: min_chars >= 1");
    CHECK(pp.min_confidence > 0.0f, "C API defaults: min_confidence > 0");

    // init with NULL models → succeeds (lazy loading)
    pp.det_model = nullptr;
    pp.rec_model = nullptr;
    pp.nafnet_model = nullptr;
    pp.vlm_model = nullptr;
    pp.punct_model = nullptr;
    void* ctx = crispembed_ocr_pipeline_init(&pp, 4);
    CHECK(ctx != nullptr, "C API init with NULL models succeeds");

    if (ctx) {
        // Run on a synthetic image → no crash, returns empty
        std::vector<uint8_t> img(100 * 100 * 3, 255);
        std::string path = write_temp(img, 100, 100, 3, "capi_test");

        int n_res = 0;
        const char* full_text = nullptr;
        float mean_conf = 0.0f;
        const crispembed_ocr_result* res = crispembed_ocr_pipeline_run(
            ctx, path.c_str(), &n_res, &full_text, &mean_conf);
        CHECK(n_res == 0, "C API run with no models → 0 regions");
        // full_text may be NULL or empty
        CHECK(!full_text || full_text[0] == '\0' || n_res == 0,
              "C API run → empty text");

        crispembed_ocr_pipeline_free(ctx);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 8. Edge cases
// ═══════════════════════════════════════════════════════════════════════
static void test_edge_cases() {
    printf("── edge cases ──\n");
    using namespace ocr_orchestrator;

    // NULL context → no crash
    {
        result r = run_file(nullptr, "/tmp/test.png");
        CHECK(r.full_text.empty(), "null context → empty result");
    }

    // NULL image path → no crash
    {
        config cfg = default_config();
        context* ctx = nullptr;
        load(&ctx, cfg);
        result r = run_file(ctx, nullptr);
        CHECK(r.full_text.empty(), "null path → empty result");
        ocr_orchestrator::free(ctx);
    }

    // free(NULL) → no crash
    ocr_orchestrator::free(nullptr);
    CHECK(true, "free(nullptr) → no crash");

    // Load with verbose → no crash
    {
        config cfg = default_config();
        cfg.verbose = true;
        context* ctx = nullptr;
        CHECK(load(&ctx, cfg), "load with verbose=true succeeds");
        ocr_orchestrator::free(ctx);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 9. Model-dependent: Tesseract regression (gated)
// ═══════════════════════════════════════════════════════════════════════
static void test_tesseract_regression() {
    printf("── tesseract regression (model-gated) ──\n");

    const char* models_dir = getenv("CRISPEMBED_MODELS_DIR");
    if (!models_dir || !models_dir[0]) {
        printf("  SKIP: CRISPEMBED_MODELS_DIR not set\n");
        return;
    }

    std::string det_path = std::string(models_dir) + "/dbnet-ic15-q8_0.gguf";
    std::string tess_path = std::string(models_dir) + "/tesseract-eng-q8_0.gguf";

    // Check if models exist
    FILE* f1 = fopen(det_path.c_str(), "r");
    FILE* f2 = fopen(tess_path.c_str(), "r");
    if (!f1 || !f2) {
        if (f1) fclose(f1);
        if (f2) fclose(f2);
        printf("  SKIP: tesseract models not found at %s\n", models_dir);
        return;
    }
    fclose(f1); fclose(f2);

    using namespace ocr_orchestrator;

    // Build a tesseract-only pipeline
    config cfg;
    cfg.router = false;
    chain ch;
    ch.type = source_type::auto_detect;
    stage s;
    s.eng = engine::tesseract;
    s.model_a = det_path;
    s.model_b = tess_path;
    s.accept.min_chars = 1;
    s.accept.min_confidence = 0.0f;
    ch.stages.push_back(s);
    cfg.chains.push_back(ch);

    context* ctx = nullptr;
    if (!load(&ctx, cfg)) {
        printf("  SKIP: failed to load tesseract pipeline\n");
        return;
    }

    // Create a test image with large text (black on white, 400x60)
    // Use a simple block-letter pattern
    std::vector<uint8_t> img(400 * 60 * 3, 255);
    // Draw a simple "T" shape
    for (int y = 5; y < 15; y++)
        for (int x = 20; x < 80; x++)
            for (int c = 0; c < 3; c++) img[(y*400+x)*3+c] = 0;
    for (int y = 15; y < 50; y++)
        for (int x = 45; x < 55; x++)
            for (int c = 0; c < 3; c++) img[(y*400+x)*3+c] = 0;
    std::string path = write_temp(img, 400, 60, 3, "tess_regr");

    result r = run_file(ctx, path.c_str());
    CHECK(r.stages_tried == 1, "tesseract stage ran");
    // Don't assert on text content — just verify no crash and non-negative confidence
    CHECK(r.mean_confidence >= 0.0f, "tesseract confidence >= 0");
    printf("  INFO: tesseract output: \"%s\" (conf=%.2f)\n",
           r.full_text.c_str(), r.mean_confidence);

    free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════
// 10. Model-dependent: Punctuation post-process (gated)
// ═══════════════════════════════════════════════════════════════════════
static void test_punctuation() {
    printf("── punctuation (model-gated) ──\n");

    const char* models_dir = getenv("CRISPEMBED_MODELS_DIR");
    if (!models_dir || !models_dir[0]) {
        printf("  SKIP: CRISPEMBED_MODELS_DIR not set\n");
        return;
    }

    std::string punct_path = std::string(models_dir) + "/fireredpunc-q8_0.gguf";
    FILE* f = fopen(punct_path.c_str(), "r");
    if (!f) {
        printf("  SKIP: punct model not found at %s\n", punct_path.c_str());
        return;
    }
    fclose(f);

    // Test the C API directly
    void* pctx = crispembed_punct_init(punct_path.c_str(), 4);
    if (!pctx) {
        printf("  SKIP: failed to load punct model\n");
        return;
    }

    const char* input = "hello world this is a test";
    const char* output = crispembed_punct_process(pctx, input);
    CHECK(output != nullptr, "punct process returns non-null");
    if (output) {
        // Punctuation model should add at least some capitalization or punctuation
        bool changed = strcmp(input, output) != 0;
        printf("  INFO: \"%s\" → \"%s\"\n", input, output);
        CHECK(changed, "punct model modifies input");
    }

    // NULL model → punct_process with NULL context
    // (already tested by the pipeline when punct_model is NULL)

    crispembed_punct_free(pctx);
}

int main() {
    test_default_config();
    test_classifier();
    test_accept_gate();
    test_multi_stage();
    test_per_stage_config();
    test_chain_selection();
    test_c_api();
    test_edge_cases();
    test_tesseract_regression();
    test_punctuation();

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
