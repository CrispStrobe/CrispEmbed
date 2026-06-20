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

#include "ocr_pipeline.h"   // ocr_pipeline::ocr_result (box + text + confidence)
#include "scan_cleanup.h"   // scan_cleanup_params
#include <string>
#include <vector>

namespace ocr_orchestrator {

// Which ggml-native engine runs a stage. Each maps to an existing context type
// in this repo; `dbnet_trocr` is the general `ocr_pipeline` (detect+recognize).
enum class engine {
    dbnet_trocr,   // ocr_pipeline.cpp  (DBNet detection + TrOCR recognition)
    surya,         // surya_det.cpp + recognizer
    qwen2vl,       // qwen2vl_ocr.cpp   (VLM)
    got,           // got_ocr.cpp
    parseq,        // parseq_ocr.cpp
    glm,           // glm_ocr.cpp
    internvl2,     // internvl2_ocr.cpp
    tesseract,     // DBNet detection + Tesseract-LSTM line recognition
    deepseek_ocr2, // deepseek_ocr2.cpp (MoE VLM)
    pix2struct,    // pix2struct.cpp (document/chart understanding)
    granite_vision,// granite_vision_ocr.cpp (LLaVA-Next, OCRBench 852)
    lightonocr,    // lightonocr.cpp (Pixtral ViT + Qwen3 decoder)
    qwen3vl,       // qwen2vl_ocr.cpp (Qwen3-VL, DeepStack + IMROPE)
};

// Image category used to pick a chain. `auto_detect` runs the classifier.
enum class source_type {
    auto_detect,
    screenshot,    // born-digital UI capture — no deskew/binarize
    scanned_doc,   // flatbed/phone scan of a page — deskew + crop + binarize
    photo,         // photo containing text — NAFNet denoise, never binarize
};

// Per-stage cleanup recipe. `params` are the 10 classical knobs; `denoise`
// switches on the NAFNet learned tier-2 (uses config.nafnet_model).
struct cleanup_profile {
    bool               enabled = false;
    scan_cleanup_params params  = scan_cleanup_defaults();
    bool               denoise = false;   // NAFNet tier-2
};

// When is a stage's output "good enough" to stop (else escalate to next stage)?
struct accept_gate {
    int   min_chars      = 8;     // recognized-text yield floor
    float min_confidence = 0.5f;  // mean region confidence floor (0 = ignore)
};

// Tunable engine parameters (per stage). Only the fields relevant to the
// stage's engine are used; the rest keep their defaults.
struct engine_params {
    // Detection (DBNet / Surya), used by ocr_pipeline::run_file.
    float det_prob_threshold = 0.3f;
    float det_box_threshold  = 0.5f;
    int   det_target_short   = 736;
    // VLM generation (GOT / GLM / Qwen2.5-VL / InternVL2).
    int         vlm_max_tokens = 0;   // 0 = engine default
    std::string vlm_prompt;           // empty = engine default prompt
};

// One engine stage with its own cleanup + acceptance criteria + model paths.
struct stage {
    engine          eng     = engine::dbnet_trocr;
    bool            enabled = true;
    cleanup_profile cleanup;
    accept_gate     accept;
    engine_params   params;
    std::string     model_a; // det / single model GGUF (resolved by caller)
    std::string     model_b; // rec model GGUF (engines that need a pair)
};

// Ordered stages for one source type. First passing stage wins; otherwise the
// best-by-yield result is returned.
struct chain {
    source_type        type   = source_type::auto_detect;
    std::vector<stage> stages;
};

struct config {
    bool               router = true;   // classify + route; false → first chain
    std::string        nafnet_model;    // shared NAFNet GGUF path ("" = no tier-2)
    std::string        sr_model;        // text SR GGUF path ("" = disabled)
    int                sr_target_dpi = 200;  // auto-trigger SR when estimated DPI < this
    std::string        lid_model;       // text LID GGUF path ("" = no LID)
    std::string        truecase_model;  // truecaser GGUF path ("" = no truecasing)
    std::string        tess_model_dir;  // directory of tesseract-{lang}-q8_0.gguf files for auto-select
    std::vector<chain> chains;          // one per source_type, or a single chain
    bool               verbose = false; // log stage transitions, gate decisions, failures
};

// Sensible defaults: router on; per-source chains with binarize for classical
// stages, denoise-only for VLM/detector stages; accept-gate {8 chars, 0.5}.
config default_config();

struct result {
    std::vector<ocr_pipeline::ocr_result> regions; // reading-order regions
    std::string full_text;        // regions joined in reading order
    float       mean_confidence = 0.0f;
    engine      used_engine = engine::dbnet_trocr;
    source_type used_type   = source_type::auto_detect;
    int         stages_tried = 0;
    std::string detected_lang;    // ISO 639-1 code from LID ("" if no LID)
    float       lang_confidence = 0.0f;
};

struct context;

// Build a pipeline context. Engines/models are lazily loaded on first use so an
// absent GGUF for one stage just skips that stage rather than failing load.
bool load(context** ctx, const config& cfg, int n_threads = 1);

// Run the full pipeline on an image file.
result run_file(context* ctx, const char* image_path);

// Standalone source-type classifier (cheap heuristics: aspect ratio, mean
// saturation, alpha presence, EXIF camera tag). Exposed for the router + tests.
source_type classify_file(const char* image_path);

void free(context* ctx);

} // namespace ocr_orchestrator
