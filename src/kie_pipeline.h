// kie_pipeline.h — Key Information Extraction: OCR + NER pipeline.
//
// Chains the OCR orchestrator (text detection + recognition) with GLiNER
// zero-shot NER to extract structured key-value fields from document images
// (receipts, invoices, forms, business cards).
//
// Usage:
//   kie_pipeline::config cfg;
//   cfg.ocr = ocr_orchestrator::default_config();
//   cfg.ner_model = "gliner-lfm-f32.gguf";
//   cfg.threshold = 0.5f;
//
//   kie_pipeline::context* ctx;
//   kie_pipeline::load(&ctx, cfg, 4);
//
//   const char* labels[] = {"total", "date", "vendor"};
//   auto result = kie_pipeline::extract(ctx, "receipt.png", labels, 3);
//   for (auto& f : result.fields) {
//       printf("%s = %s (%.2f)\n", f.label.c_str(), f.value.c_str(), f.score);
//   }
//   kie_pipeline::free(ctx);

#pragma once

#include "ocr_orchestrator.h"
#include <string>
#include <vector>

namespace kie_pipeline {

// A single extracted field: label + value + confidence + spatial position.
struct field {
    std::string label;       // entity type (e.g. "total", "date")
    std::string value;       // extracted text span
    float       score;       // NER confidence [0, 1]
    float       x, y, w, h; // bounding box in original image coordinates
};

struct result {
    std::vector<field> fields;           // extracted key-value pairs
    std::string        ocr_full_text;    // raw OCR text (for debugging)
    float              ocr_confidence;   // mean OCR confidence
    int                n_ocr_regions;    // number of OCR regions detected
};

struct config {
    ocr_orchestrator::config ocr;       // OCR pipeline configuration
    std::string              ner_model; // GLiNER GGUF model path (Phase 1)
    std::string              lilt_model; // LiLT GGUF model path (Phase 2, optional)
    float                    threshold; // NER confidence threshold (default 0.5)
};

struct context;

// Build a KIE context. Loads OCR pipeline + NER model.
// Returns false on failure (missing models, etc.).
bool load(context** ctx, const config& cfg, int n_threads = 4);

// Extract fields from a document image.
// labels: array of field names to extract (e.g. "total", "date", "vendor")
// n_labels: number of labels
// threshold: NER confidence threshold (0 = use config default)
result extract(context* ctx, const char* image_path,
               const char** labels, int n_labels,
               float threshold = 0.0f);

// Free all resources.
void free(context* ctx);

} // namespace kie_pipeline
