// kie_pipeline.cpp — Key Information Extraction: OCR + NER pipeline.

#include "kie_pipeline.h"
#include "gliner_ner.h"

#include <cstdio>
#include <cstring>

namespace kie_pipeline {

struct context {
    ocr_orchestrator::context* ocr_ctx = nullptr;
    void*                      ner_ctx = nullptr;
    float                      threshold = 0.5f;
};

bool load(context** out, const config& cfg, int n_threads) {
    if (!out) return false;

    auto* ctx = new context;
    ctx->threshold = cfg.threshold > 0.0f ? cfg.threshold : 0.5f;

    // Load OCR pipeline.
    if (!ocr_orchestrator::load(&ctx->ocr_ctx, cfg.ocr, n_threads)) {
        fprintf(stderr, "kie_pipeline: failed to load OCR pipeline\n");
        delete ctx;
        return false;
    }

    // Load NER model.
    ctx->ner_ctx = gliner_ner_init(cfg.ner_model.c_str(), n_threads);
    if (!ctx->ner_ctx) {
        fprintf(stderr, "kie_pipeline: failed to load NER model: %s\n",
                cfg.ner_model.c_str());
        ocr_orchestrator::free(ctx->ocr_ctx);
        delete ctx;
        return false;
    }

    *out = ctx;
    return true;
}

// Per-region character offset tracking for mapping NER spans back to regions.
struct region_span {
    int char_start;  // start offset in concatenated text
    int char_end;    // end offset (exclusive)
    int region_idx;  // index into ocr result regions
};

result extract(context* ctx, const char* image_path,
               const char** labels, int n_labels,
               float threshold) {
    result res;
    if (!ctx || !image_path || !labels || n_labels <= 0) return res;

    // Step 1: Run OCR to get text regions with bounding boxes.
    auto ocr_res = ocr_orchestrator::run_file(ctx->ocr_ctx, image_path);
    res.ocr_full_text  = ocr_res.full_text;
    res.ocr_confidence = ocr_res.mean_confidence;
    res.n_ocr_regions  = (int)ocr_res.regions.size();

    if (ocr_res.regions.empty()) return res;

    // Step 2: Build concatenated text with per-region offset tracking.
    // Regions are separated by newlines so NER sees natural boundaries.
    std::string combined;
    std::vector<region_span> spans;
    spans.reserve(ocr_res.regions.size());

    for (int i = 0; i < (int)ocr_res.regions.size(); i++) {
        const auto& r = ocr_res.regions[i];
        if (r.text.empty()) continue;

        region_span sp;
        sp.char_start = (int)combined.size();
        sp.region_idx = i;
        combined += r.text;
        sp.char_end = (int)combined.size();
        spans.push_back(sp);

        combined += '\n';  // separator between regions
    }

    if (combined.empty()) return res;

    // Step 3: Run NER on the concatenated text.
    const float thr = threshold > 0.0f ? threshold : ctx->threshold;
    gliner_ner_entity* entities = nullptr;
    int n_entities = gliner_ner_extract(
        ctx->ner_ctx, combined.c_str(),
        labels, n_labels,
        thr, &entities);

    if (n_entities <= 0 || !entities) return res;

    // Step 4: Map each NER entity back to its source OCR region.
    for (int i = 0; i < n_entities; i++) {
        const auto& ent = entities[i];

        // Find which region(s) this entity overlaps.
        // Use the midpoint of the entity span to pick the primary region.
        int ent_mid = (ent.start_char + ent.end_char) / 2;
        int best_region = -1;
        for (const auto& sp : spans) {
            if (ent_mid >= sp.char_start && ent_mid < sp.char_end) {
                best_region = sp.region_idx;
                break;
            }
        }

        field f;
        f.label = ent.label ? ent.label : "";
        f.value = ent.text  ? ent.text  : "";
        f.score = ent.score;

        if (best_region >= 0) {
            const auto& box = ocr_res.regions[best_region].box;
            f.x = box.x;
            f.y = box.y;
            f.w = box.w;
            f.h = box.h;
        } else {
            f.x = f.y = f.w = f.h = 0.0f;
        }

        res.fields.push_back(std::move(f));
    }

    return res;
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->ocr_ctx) ocr_orchestrator::free(ctx->ocr_ctx);
    if (ctx->ner_ctx) gliner_ner_free(ctx->ner_ctx);
    delete ctx;
}

} // namespace kie_pipeline
