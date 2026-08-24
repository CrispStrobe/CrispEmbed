// ocr_orchestrator.h — configurable multi-stage OCR pipeline.
//
// Composes the existing C++ building blocks into one "proper" pipeline:
//
//   classify source type  →  pick chain  →  for each stage in order:
//        cleanup(profile)  →  engine(detect+recognize)  →  accept-gate?
//        accept ─┘                                          └─ escalate
//
// Everything here is C++-primary (no Rust orchestration): cleanup is
// `scan_cleanup` (classical tier-1 + learned NAFNet tier-2), the engines are
// the ggml-native OCR contexts already in this repo (DBNet+TrOCR `ocr_pipeline`,
// Surya, Qwen2.5-VL, GOT-OCR2, ParSeq, GLM-OCR, InternVL2). The accept-gate uses
// the per-region `confidence` that `ocr_pipeline::ocr_result` already carries
// plus recognized-text yield, so a weak tier escalates to the next.
//
// Consumed via the flat `crispembed_ocr_pipeline_*` C API (crispembed.h), which
// the Rust (`CrispOcrPipeline`) and Dart bindings wrap. CrispSorter stays a thin
// caller: it builds a `config`, calls `run_file`, and renders the result.
//
// Usage:
//   ocr_orchestrator::config cfg = ocr_orchestrator::default_config();
//   ocr_orchestrator::context* ctx;
//   ocr_orchestrator::load(&ctx, cfg, /*n_threads=*/1);
//   auto res = ocr_orchestrator::run_file(ctx, "scan.png");
//   printf("%s\n", res.full_text.c_str());   // joined reading-order text
//   ocr_orchestrator::free(ctx);

#pragma once

#include "ocr_pipeline.h"      // ocr_pipeline::ocr_result (box + text + confidence)
#include "layout_detect.h"     // optional document layout regions
#include "ocr_region_router.h" // deterministic structured-region dispatch
#include "scan_cleanup.h"      // scan_cleanup_params
#include <string>
#include <vector>

namespace ocr_orchestrator {

// Which ggml-native engine runs a stage. Each maps to an existing context type
// in this repo; `dbnet_trocr` is the general `ocr_pipeline` (detect+recognize).
enum class engine {
    dbnet_trocr,       // ocr_pipeline.cpp  (DBNet detection + TrOCR recognition)
    ppocrv6,           // PP-OCRv6 detector + recognizer
    surya,             // surya_det.cpp + recognizer
    qwen2vl,           // qwen2vl_ocr.cpp   (VLM)
    got,               // got_ocr.cpp
    parseq,            // parseq_ocr.cpp
    glm,               // glm_ocr.cpp
    internvl2,         // internvl2_ocr.cpp
    tesseract,         // DBNet detection + Tesseract-LSTM line recognition
    tesseract_fraktur, // DBNet detection + grayscale crops + German Fraktur LSTM
    deepseek_ocr2,     // deepseek_ocr2.cpp (MoE VLM)
    pix2struct,        // pix2struct.cpp (document/chart understanding)
    granite_vision,    // granite_vision_ocr.cpp (LLaVA-Next, OCRBench 852)
    lightonocr,        // lightonocr.cpp (Pixtral ViT + Qwen3 decoder)
    qwen3vl,           // qwen2vl_ocr.cpp (Qwen3-VL, DeepStack + IMROPE)
    unlimited_ocr,     // unlimited_ocr.cpp (SAM + CLIP + MoE VLM)
    unified,           // metadata-dispatched crispembed_ocr_model_* GGUF
    easyocr,           // easyocr_pipeline.cpp (DBNet detection + EasyOCR CRNN)
    olmocr,            // qwen2vl_ocr.cpp (olmOCR document fine-tune of Qwen2.5-VL)
    lfm2_vl,           // lfm2_vl_ocr.cpp (SigLIP2 NaFlex multi-tile + LFM2 hybrid conv/attn)
};

// Image category used to pick a chain. `auto_detect` runs the classifier.
enum class source_type {
    auto_detect,
    screenshot,  // born-digital UI capture — no deskew/binarize
    scanned_doc, // flatbed/phone scan of a page — deskew + crop + binarize
    photo,       // photo containing text — NAFNet denoise, never binarize
};

// Per-stage cleanup recipe. `params` are the 10 classical knobs; `denoise`
// switches on the NAFNet learned tier-2 (uses config.nafnet_model).
struct cleanup_profile {
    bool enabled = false;
    scan_cleanup_params params = scan_cleanup_defaults();
    bool denoise = false; // NAFNet tier-2
};

// When is a stage's output "good enough" to stop (else escalate to next stage)?
struct accept_gate {
    int min_chars = 8;           // recognized-text yield floor
    float min_confidence = 0.5f; // mean region confidence floor (0 = ignore)
};

// Tunable engine parameters (per stage). Only the fields relevant to the
// stage's engine are used; the rest keep their defaults.
struct engine_params {
    // Tesseract engines: 0 = DBNet, 1 = classical page segmentation.
    int page_segmentation = 0;
    // Detection (DBNet / Surya), used by ocr_pipeline::run_file.
    float det_prob_threshold = 0.3f;
    float det_box_threshold = 0.5f;
    int det_target_short = 736;
    int det_max_side = 2000;
    int det_min_height = 30;
    float det_width_height_ratio = 8.0f;
    int det_max_candidates = 1000;
    int det_dilation = 1;
    ocr_detect::score_mode det_scoring = ocr_detect::score_mode::fast;
    // VLM generation (GOT / GLM / Qwen2.5-VL / InternVL2).
    int vlm_max_tokens = 0; // 0 = engine default
    std::string vlm_prompt; // empty = engine default prompt
};

// One engine stage with its own cleanup + acceptance criteria + model paths.
struct stage {
    engine eng = engine::dbnet_trocr;
    bool enabled = true;
    cleanup_profile cleanup;
    accept_gate accept;
    engine_params params;
    std::string model_a; // det / single model GGUF (resolved by caller)
    std::string model_b; // rec model GGUF (engines that need a pair)
    // Optional line-orientation model.  PP-OCRv6 uses PP-LCNet here; keeping
    // it separate from the recognizer lets the stage report an explicit
    // classifier fallback instead of silently treating orientation as part of
    // recognition.
    std::string model_c;
};

// Ordered stages for one source type. First passing stage wins; otherwise the
// best-by-yield result is returned.
struct chain {
    source_type type = source_type::auto_detect;
    std::vector<stage> stages;
};

struct config {
    bool router = true;          // classify + route; false → first chain
    std::string nafnet_model;    // shared NAFNet GGUF path ("" = no tier-2)
    std::string sr_model;        // text SR GGUF path ("" = disabled)
    int sr_target_dpi = 200;     // auto-trigger SR when estimated DPI < this
    std::string lid_model;       // text LID GGUF path ("" = no LID)
    std::string truecase_model;  // truecaser GGUF path ("" = no truecasing)
    std::string tess_model_dir;  // directory of tesseract-{lang}-q8_0.gguf files for auto-select
    std::string layout_model;    // optional RT-DETR layout GGUF ("" = disabled)
    bool route_tables = false;   // route table regions when layout is enabled
    bool route_formulas = false; // route formula regions when layout is enabled
    bool image_text_fallback = true;
    std::string table_model;   // optional Tesseract-LSTM GGUF for table cells
    std::string formula_model; // optional PP-FormulaNet GGUF
    std::vector<chain> chains; // one per source_type, or a single chain
    bool verbose = false;      // log stage transitions, gate decisions, failures
};

// Sensible defaults: router on; per-source chains with binarize for classical
// stages, denoise-only for VLM/detector stages; accept-gate {8 chars, 0.5}.
config default_config();

// Explicit German-Fraktur profile. The caller supplies model_a as the DBNet
// detector and model_b as tesseract-frk-{precision}.gguf. Unlike the generic
// scanned-document profile it deliberately preserves grayscale input, since
// binarization can erase thin Fraktur strokes and long-s forms.
stage tesseract_fraktur_stage();

struct result {
    int page_width = 0;
    int page_height = 0;
    std::vector<ocr_pipeline::ocr_result> regions; // reading-order regions
    std::vector<layout_detect::region> layout;     // optional structured regions
    ocr_region_router::routing_plan routing;       // deterministic dispatch plan
    struct table_output {
        int layout_index = -1;
        float confidence = 0.0f;
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        std::string html;
    };
    struct formula_output {
        int layout_index = -1;
        float confidence = 0.0f;
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        std::string latex;
    };
    std::vector<table_output> tables;
    std::vector<formula_output> formulas;
    std::vector<int> reading_order; // region indices in document order
    std::string full_text;          // regions joined in reading order
    std::string markdown;           // lightweight structured page export
    float mean_confidence = 0.0f;
    engine used_engine = engine::dbnet_trocr;
    source_type used_type = source_type::auto_detect;
    int stages_tried = 0;
    std::string detected_lang; // ISO 639-1 code from LID ("" if no LID)
    float lang_confidence = 0.0f;
    struct stage_metric {
        int index = 0;
        std::string engine;
        float elapsed_ms = 0.0f;
        bool cleanup_applied = false;
        bool accepted = false;
        int text_chars = 0;
        float mean_confidence = 0.0f;
    };
    std::vector<stage_metric> stage_metrics;
};

struct context;

struct capabilities {
    bool layout = false;
    bool tables = false;
    bool formulas = false;
    bool image_text_fallback = true;
};

// Build a pipeline context. Engines/models are lazily loaded on first use so an
// absent GGUF for one stage just skips that stage rather than failing load.
bool load(context ** ctx, const config & cfg, int n_threads = 1);

capabilities get_capabilities(const context * ctx);

// Run the full pipeline on an image file.
result run_file(context * ctx, const char * image_path);

// Standalone source-type classifier (cheap heuristics: aspect ratio, mean
// saturation, alpha presence, EXIF camera tag). Exposed for the router + tests.
source_type classify_file(const char * image_path);

void free(context * ctx);

} // namespace ocr_orchestrator
