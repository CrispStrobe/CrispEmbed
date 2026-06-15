// kie_pipeline.cpp — Key Information Extraction: OCR + NER pipeline.
//
// Phase 1: OCR → GLiNER NER (text-only entity extraction)
// Phase 2: OCR → LiLT (layout-aware token classification, when lilt_model set)

#include "kie_pipeline.h"
#include "gliner_ner.h"
#include "lilt_kie.h"

#include <cstdio>
#include <cstring>
#include <sstream>

namespace kie_pipeline {

struct context {
    ocr_orchestrator::context* ocr_ctx = nullptr;
    void*                      ner_ctx = nullptr;    // Phase 1: GLiNER
    lilt_kie::context*         lilt_ctx = nullptr;   // Phase 2: LiLT
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

    // Phase 2: try LiLT first (layout-aware, higher quality).
    if (!cfg.lilt_model.empty()) {
        if (lilt_kie::load(&ctx->lilt_ctx, cfg.lilt_model.c_str(), n_threads)) {
            fprintf(stderr, "kie_pipeline: using LiLT backend (%d labels)\n",
                    lilt_kie::num_labels(ctx->lilt_ctx));
        } else {
            fprintf(stderr, "kie_pipeline: failed to load LiLT model: %s (falling back to NER)\n",
                    cfg.lilt_model.c_str());
        }
    }

    // Phase 1: GLiNER NER (fallback or primary if no LiLT).
    if (!ctx->lilt_ctx && !cfg.ner_model.empty()) {
        ctx->ner_ctx = gliner_ner_init(cfg.ner_model.c_str(), n_threads);
        if (!ctx->ner_ctx) {
            fprintf(stderr, "kie_pipeline: failed to load NER model: %s\n",
                    cfg.ner_model.c_str());
            ocr_orchestrator::free(ctx->ocr_ctx);
            delete ctx;
            return false;
        }
    }

    if (!ctx->lilt_ctx && !ctx->ner_ctx) {
        fprintf(stderr, "kie_pipeline: no extraction backend (need --ner or --lilt model)\n");
        ocr_orchestrator::free(ctx->ocr_ctx);
        delete ctx;
        return false;
    }

    *out = ctx;
    return true;
}

// Per-region character offset tracking for mapping NER spans back to regions.
struct region_span {
    int char_start;
    int char_end;
    int region_idx;
};

// Phase 1: extract via GLiNER NER (text-only).
static void extract_ner(context* ctx, const ocr_orchestrator::result& ocr_res,
                         const char** labels, int n_labels, float threshold,
                         result& res) {
    // Build concatenated text with per-region offset tracking.
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
        combined += '\n';
    }

    if (combined.empty()) return;

    gliner_ner_entity* entities = nullptr;
    int n_entities = gliner_ner_extract(
        ctx->ner_ctx, combined.c_str(),
        labels, n_labels, threshold, &entities);

    if (n_entities <= 0 || !entities) return;

    for (int i = 0; i < n_entities; i++) {
        const auto& ent = entities[i];
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
            f.x = box.x; f.y = box.y; f.w = box.w; f.h = box.h;
        } else {
            f.x = f.y = f.w = f.h = 0.0f;
        }
        res.fields.push_back(std::move(f));
    }
}

// Phase 2: extract via LiLT (layout-aware token classification).
// LiLT takes pre-tokenized words + bboxes and outputs per-token labels.
// We use OCR regions as "words" and their bboxes as spatial input.
static void extract_lilt(context* ctx, const ocr_orchestrator::result& ocr_res,
                          result& res) {
    if (ocr_res.regions.empty()) return;

    // Build token sequence from OCR regions.
    // For LiLT, each OCR region is treated as a "word" (or we could split
    // by whitespace within regions for finer granularity).
    // We use simple BPE-free approach: each region is one token group.

    // For now, use a simple word-level approach:
    // Split each region's text into words, assign each word the region's bbox.
    struct word_info {
        std::string word;
        int region_idx;
        float x0, y0, x1, y1;
    };
    std::vector<word_info> words;

    for (int i = 0; i < (int)ocr_res.regions.size(); i++) {
        const auto& r = ocr_res.regions[i];
        if (r.text.empty()) continue;

        // Split region text into words
        std::istringstream iss(r.text);
        std::string w;
        int n_words = 0;
        // Count words first
        {
            std::istringstream c(r.text);
            std::string tmp;
            while (c >> tmp) n_words++;
        }
        if (n_words == 0) continue;

        // Distribute bbox evenly across words
        float word_w = r.box.w / std::max(n_words, 1);
        int wi = 0;
        while (iss >> w) {
            word_info winfo;
            winfo.word = w;
            winfo.region_idx = i;
            // Normalize bbox to [0, 1000] range (LiLT convention)
            // Assume image is ~1000x1000 for now (TODO: pass actual page dims)
            winfo.x0 = r.box.x + wi * word_w;
            winfo.y0 = r.box.y;
            winfo.x1 = r.box.x + (wi + 1) * word_w;
            winfo.y1 = r.box.y + r.box.h;
            words.push_back(winfo);
            wi++;
        }
    }

    if (words.empty()) return;

    // Build token IDs using a simple approach:
    // BOS + word tokens + EOS.
    // Since we don't have the full RoBERTa tokenizer in C++, we use the
    // word text directly as a rough token (the model will still produce
    // reasonable output thanks to the layout information).
    //
    // For proper tokenization, the caller should use the Python tokenizer
    // and call lilt_kie::classify directly.
    //
    // For now, build a simple BOS + UNK*N + EOS sequence with correct bboxes.
    int n_words = (int)words.size();
    int T = n_words + 2; // BOS + words + EOS

    std::vector<int32_t> ids(T);
    std::vector<int32_t> bbox(T * 4, 0);

    ids[0] = 0;        // BOS
    ids[T - 1] = 2;    // EOS
    for (int i = 0; i < n_words; i++) {
        ids[i + 1] = 3; // UNK token — placeholder since we lack tokenizer
        bbox[(i + 1) * 4 + 0] = (int)words[i].x0;
        bbox[(i + 1) * 4 + 1] = (int)words[i].y0;
        bbox[(i + 1) * 4 + 2] = (int)words[i].x1;
        bbox[(i + 1) * 4 + 3] = (int)words[i].y1;
    }

    auto results = lilt_kie::classify(ctx->lilt_ctx, ids.data(), bbox.data(), T);

    // Map LiLT token labels to fields.
    // Group consecutive B-/I- tokens into spans.
    std::string current_label;
    std::string current_value;
    float current_score = 0.0f;
    int current_region = -1;
    int span_count = 0;

    auto flush_span = [&]() {
        if (current_label.empty() || current_label == "O") return;
        // Strip B-/I- prefix
        std::string label = current_label;
        if (label.size() > 2 && (label[0] == 'B' || label[0] == 'I') && label[1] == '-')
            label = label.substr(2);

        field f;
        f.label = label;
        f.value = current_value;
        f.score = current_score / std::max(span_count, 1);
        if (current_region >= 0) {
            const auto& box = ocr_res.regions[current_region].box;
            f.x = box.x; f.y = box.y; f.w = box.w; f.h = box.h;
        }
        res.fields.push_back(std::move(f));
    };

    for (int i = 1; i < T - 1; i++) { // skip BOS/EOS
        const auto& tok = results[i];
        int word_idx = i - 1;

        bool is_begin = tok.label.substr(0, 2) == "B-";
        bool is_inside = tok.label.substr(0, 2) == "I-";
        std::string base_label = tok.label;
        if (base_label.size() > 2 && (base_label[0] == 'B' || base_label[0] == 'I') && base_label[1] == '-')
            base_label = base_label.substr(2);

        if (is_begin || (!is_inside && tok.label != "O")) {
            flush_span();
            current_label = tok.label;
            current_value = words[word_idx].word;
            current_score = tok.score;
            current_region = words[word_idx].region_idx;
            span_count = 1;
        } else if (is_inside && !current_label.empty()) {
            current_value += " " + words[word_idx].word;
            current_score += tok.score;
            span_count++;
        } else {
            flush_span();
            current_label.clear();
            current_value.clear();
            current_score = 0.0f;
            span_count = 0;
        }
    }
    flush_span();
}

result extract(context* ctx, const char* image_path,
               const char** labels, int n_labels,
               float threshold) {
    result res;
    if (!ctx || !image_path) return res;

    // Step 1: Run OCR to get text regions with bounding boxes.
    auto ocr_res = ocr_orchestrator::run_file(ctx->ocr_ctx, image_path);
    res.ocr_full_text  = ocr_res.full_text;
    res.ocr_confidence = ocr_res.mean_confidence;
    res.n_ocr_regions  = (int)ocr_res.regions.size();

    if (ocr_res.regions.empty()) return res;

    const float thr = threshold > 0.0f ? threshold : ctx->threshold;

    // Step 2: Extract fields using the available backend.
    if (ctx->lilt_ctx) {
        // Phase 2: LiLT layout-aware extraction (ignores labels param)
        extract_lilt(ctx, ocr_res, res);
    } else if (ctx->ner_ctx) {
        // Phase 1: GLiNER NER text-only extraction
        if (labels && n_labels > 0) {
            extract_ner(ctx, ocr_res, labels, n_labels, thr, res);
        }
    }

    return res;
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->ocr_ctx) ocr_orchestrator::free(ctx->ocr_ctx);
    if (ctx->ner_ctx) gliner_ner_free(ctx->ner_ctx);
    if (ctx->lilt_ctx) lilt_kie::free(ctx->lilt_ctx);
    delete ctx;
}

} // namespace kie_pipeline
