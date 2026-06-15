// ocr_orchestrator.cpp — see ocr_orchestrator.h.
//
// Slice A: source-type classifier + per-stage cleanup (classical + NAFNet) fed
// to the DBNet+TrOCR `ocr_pipeline` engine, with a text-yield + confidence
// accept-gate that escalates through the chain. Cleanup → engine handoff is via
// a transient temp PNG so it works uniformly with every path-based engine
// (slice B wires the remaining ggml engines into `run_engine`).

#include "ocr_orchestrator.h"

#include "scan_cleanup.h"
#include "ocr_pipeline.h"
// Single-shot VLM/document OCR engines (full image → text). C API.
#include "got_ocr.h"
#include "glm_ocr.h"
#include "qwen2vl_ocr.h"
#include "internvl2_ocr.h"
// Tesseract-LSTM line recognizer + DBNet detection (the tesseract engine pairs
// detection with per-line tesseract recognition).
#include "tesseract_lstm.h"
#include "ocr_detect.h"
// Text super-resolution (low-DPI upscale before OCR).
#include "text_sr.h"
#include "pan_sr.h"
#include "core/gguf_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ggml/examples/stb_image_write.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>   // _getpid
#else
#include <unistd.h>    // getpid
#endif

// stbi_load is exported (non-static) by image_preprocess.cpp's
// STB_IMAGE_IMPLEMENTATION; forward-declare what we use rather than re-include.
extern "C" {
unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file,
                         int desired_channels);
void stbi_image_free(void* retval_from_stbi_load);
}

namespace ocr_orchestrator {

struct context {
    config cfg;
    int    n_threads = 4;
    // Lazily-loaded engine + cleanup handles (loaded on first use).
    ocr_pipeline::context*   dbnet  = nullptr;   // DBNet detection + TrOCR recognition
    got_ocr_context*         got    = nullptr;   // GOT-OCR2 (single-shot VLM)
    glm_ocr_context*         glm    = nullptr;   // GLM-OCR (single-shot VLM)
    qwen2vl_ocr_context*     qwen   = nullptr;   // Qwen2.5-VL (single-shot VLM)
    internvl2_ocr_context*   intern = nullptr;   // InternVL2 (single-shot VLM)
    ocr_detect::context*     tess_det = nullptr; // DBNet detection for the tesseract engine
    tesseract_lstm_context*  tess   = nullptr;   // Tesseract-LSTM line recognizer
    scan_cleanup_ctx*        clean1 = nullptr;   // tier-1 classical (model = NULL)
    scan_cleanup_ctx*        clean2 = nullptr;   // tier-2 NAFNet (model = nafnet_model)
    text_sr_context*         sr     = nullptr;   // NAFNet-SR (low-DPI upscale)
    pan_sr_context*          pan    = nullptr;   // PAN 4x SR (alternative upscaler)
    enum { SR_NONE, SR_NAFNET, SR_PAN } sr_kind = SR_NONE;
};

// ── defaults ────────────────────────────────────────────────────────────────

static cleanup_profile classical_profile(bool binarize) {
    cleanup_profile p;
    p.enabled               = true;
    p.params                = scan_cleanup_defaults();   // deskew+crop+whiten on
    p.params.binarize       = binarize ? 1 : 0;
    p.denoise               = false;
    return p;
}

static cleanup_profile denoise_profile(bool deskew) {
    cleanup_profile p;
    p.enabled               = true;
    p.params                = scan_cleanup_defaults();
    p.params.binarize       = 0;            // never binarize for detector/VLM
    p.params.deskew         = deskew ? 1 : 0;
    p.denoise               = true;         // NAFNet
    return p;
}

static stage dbnet_stage(cleanup_profile cp) {
    stage s;
    s.eng     = engine::dbnet_trocr;
    s.enabled = true;
    s.cleanup = cp;
    // model_a/model_b left empty — the caller (CrispSorter / CLI) resolves the
    // DBNet + TrOCR GGUF paths and fills them before load().
    return s;
}

config default_config() {
    config cfg;
    cfg.router = true;
    // nafnet_model left empty by default; caller sets it to enable tier-2.
    chain screenshot{ source_type::screenshot,  { dbnet_stage(classical_profile(false)) } };
    chain scan      { source_type::scanned_doc, { dbnet_stage(classical_profile(true))  } };
    chain photo     { source_type::photo,       { dbnet_stage(denoise_profile(true))    } };
    chain any       { source_type::auto_detect, { dbnet_stage(classical_profile(false)) } };
    cfg.chains = { screenshot, scan, photo, any };
    return cfg;
}

// ── source-type classifier (cheap pixel heuristics) ──────────────────────────

source_type classify_file(const char* image_path) {
    if (!image_path) return source_type::scanned_doc;
    int w = 0, h = 0, c = 0;
    unsigned char* d = stbi_load(image_path, &w, &h, &c, 0);
    if (!d || w <= 0 || h <= 0) {
        if (d) stbi_image_free(d);
        return source_type::scanned_doc;
    }

    const bool   has_alpha = (c == 4);
    const double aspect    = (double)w / (double)h;

    // Mean saturation + fraction of near-white pixels over a stride sample.
    double  sat_sum    = 0.0;
    long    near_white = 0;
    long    n          = 0;
    const int stride   = (w * h > 400000) ? 7 : 1;   // subsample big images
    for (long i = 0; i < (long)w * h; i += stride) {
        const unsigned char* px = d + (size_t)i * c;
        int r = px[0], g = px[1], b = (c >= 3) ? px[2] : px[0];
        int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
        int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
        sat_sum += (mx == 0) ? 0.0 : (double)(mx - mn) / (double)mx;
        if (mn >= 200) near_white++;
        n++;
    }
    stbi_image_free(d);
    const double mean_sat   = n ? sat_sum / (double)n : 0.0;
    const double white_frac = n ? (double)near_white / (double)n : 0.0;

    // Heuristics:
    //  - lots of near-white background + low saturation → a page (scan) or UI.
    //  - alpha channel or very wide/tall aspect → screenshot (born-digital).
    //  - otherwise colourful / photographic → photo.
    if (mean_sat > 0.28) return source_type::photo;
    if (has_alpha || aspect > 2.2 || aspect < 0.45) return source_type::screenshot;
    if (white_frac > 0.45) return source_type::scanned_doc;
    return source_type::screenshot;
}

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string temp_png_path() {
    static std::atomic<unsigned> counter{0};
    const char* dir = std::getenv("TMPDIR");
    if (!dir || !*dir) dir = std::getenv("TEMP");
    if (!dir || !*dir) dir = "/tmp";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/crispembed_ocr_%u_%u.png",
                  (unsigned)
#ifdef _WIN32
                      _getpid()
#else
                      getpid()
#endif
                  ,
                  counter.fetch_add(1));
    return std::string(dir) + buf;
}

// Run scan cleanup on `src` and write the result to a temp PNG; returns the
// temp path (empty on failure → caller falls back to the original image).
static std::string clean_to_temp(context* ctx, const cleanup_profile& cp,
                                 const char* src) {
    if (!cp.enabled) return "";
    int w = 0, h = 0, c = 0;
    unsigned char* d = stbi_load(src, &w, &h, &c, 0);
    if (!d) return "";

    scan_cleanup_ctx** slot = cp.denoise ? &ctx->clean2 : &ctx->clean1;
    if (!*slot) {
        const char* model = (cp.denoise && !ctx->cfg.nafnet_model.empty())
                                ? ctx->cfg.nafnet_model.c_str()
                                : nullptr;   // NULL → tier-1 classical only
        *slot = scan_cleanup_init(model, ctx->n_threads);
    }

    std::string out_path;
    uint8_t* out = nullptr;
    int ow = 0, oh = 0;
    if (*slot &&
        scan_cleanup_process(*slot, d, w, h, c, cp.params, &out, &ow, &oh) == 0 &&
        out) {
        out_path = temp_png_path();
        if (stbi_write_png(out_path.c_str(), ow, oh, 3, out, ow * 3) == 0) {
            out_path.clear();   // write failed
        }
        scan_cleanup_free_image(out);
    }
    stbi_image_free(d);
    return out_path;
}

// Wrap a single-shot VLM engine's whole-image text as one region covering the
// full image. VLMs don't expose per-region confidence, so use 1.0.
static std::vector<ocr_pipeline::ocr_result> wrap_fulltext(const char* text,
                                                           int w, int h) {
    std::vector<ocr_pipeline::ocr_result> out;
    if (text && *text) {
        ocr_pipeline::ocr_result r;
        r.box.x = 0.0f; r.box.y = 0.0f;
        r.box.w = (float)w; r.box.h = (float)h;
        r.box.score = 1.0f;
        r.confidence = 1.0f;
        r.text = text;
        out.push_back(std::move(r));
    }
    return out;
}

// Run one engine on a (already-cleaned) image path. DBNet+TrOCR is a
// detect+recognize pipeline; the VLM engines are single-shot full-image OCR.
static std::vector<ocr_pipeline::ocr_result> run_engine(context* ctx,
                                                        const stage& st,
                                                        const char* path) {
    switch (st.eng) {
        case engine::dbnet_trocr:
        case engine::surya: {
            // DBNet/Surya detection + TrOCR recognition (model_a=det, model_b=rec).
            if (!ctx->dbnet) {
                if (st.model_a.empty() || st.model_b.empty()) {
                    fprintf(stderr,
                            "ocr_orchestrator: detect+recognize stage missing model "
                            "paths (model_a=det, model_b=rec)\n");
                    return {};
                }
                if (!ocr_pipeline::load(&ctx->dbnet, st.model_a.c_str(),
                                        st.model_b.c_str(), ctx->n_threads)) {
                    fprintf(stderr, "ocr_orchestrator: detect+recognize load failed\n");
                    ctx->dbnet = nullptr;
                    return {};
                }
            }
            return ocr_pipeline::run_file(ctx->dbnet, path,
                                          st.params.det_prob_threshold,
                                          st.params.det_box_threshold,
                                          st.params.det_target_short);
        }
        case engine::got: {
            if (!ctx->got) {
                if (st.model_a.empty()) { fprintf(stderr, "ocr_orchestrator: got stage missing model_a\n"); return {}; }
                ctx->got = got_ocr_init(st.model_a.c_str(), ctx->n_threads);
                if (!ctx->got) { fprintf(stderr, "ocr_orchestrator: got load failed\n"); return {}; }
            }
            int w = 0, h = 0, c = 0;
            unsigned char* px = stbi_load(path, &w, &h, &c, 3);
            if (!px) return {};
            int len = 0;
            const char* t = got_ocr_recognize_raw(ctx->got, px, w, h, 3, &len);
            auto out = wrap_fulltext(t, w, h);
            stbi_image_free(px);
            return out;
        }
        case engine::glm: {
            if (!ctx->glm) {
                if (st.model_a.empty()) { fprintf(stderr, "ocr_orchestrator: glm stage missing model_a\n"); return {}; }
                ctx->glm = glm_ocr_init(st.model_a.c_str(), ctx->n_threads);
                if (!ctx->glm) { fprintf(stderr, "ocr_orchestrator: glm load failed\n"); return {}; }
            }
            int w = 0, h = 0, c = 0;
            unsigned char* px = stbi_load(path, &w, &h, &c, 3);
            if (!px) return {};
            int len = 0;
            const char* t = glm_ocr_recognize_raw(ctx->glm, px, w, h, 3, &len);
            auto out = wrap_fulltext(t, w, h);
            stbi_image_free(px);
            return out;
        }
        case engine::qwen2vl: {
            if (!ctx->qwen) {
                if (st.model_a.empty()) { fprintf(stderr, "ocr_orchestrator: qwen2vl stage missing model_a\n"); return {}; }
                ctx->qwen = qwen2vl_ocr_init(st.model_a.c_str(), ctx->n_threads);
                if (!ctx->qwen) { fprintf(stderr, "ocr_orchestrator: qwen2vl load failed\n"); return {}; }
            }
            if (st.params.vlm_max_tokens > 0) qwen2vl_ocr_set_max_tokens(ctx->qwen, st.params.vlm_max_tokens);
            if (!st.params.vlm_prompt.empty()) qwen2vl_ocr_set_prompt(ctx->qwen, st.params.vlm_prompt.c_str());
            int w = 0, h = 0, c = 0;
            unsigned char* px = stbi_load(path, &w, &h, &c, 3);
            if (!px) return {};
            int len = 0;
            const char* t = qwen2vl_ocr_recognize_raw(ctx->qwen, px, w, h, 3, &len);
            auto out = wrap_fulltext(t, w, h);
            stbi_image_free(px);
            return out;
        }
        case engine::internvl2: {
            if (!ctx->intern) {
                if (st.model_a.empty()) { fprintf(stderr, "ocr_orchestrator: internvl2 stage missing model_a\n"); return {}; }
                ctx->intern = internvl2_ocr_init(st.model_a.c_str(), ctx->n_threads);
                if (!ctx->intern) { fprintf(stderr, "ocr_orchestrator: internvl2 load failed\n"); return {}; }
            }
            if (st.params.vlm_max_tokens > 0) internvl2_ocr_set_max_tokens(ctx->intern, st.params.vlm_max_tokens);
            if (!st.params.vlm_prompt.empty()) internvl2_ocr_set_prompt(ctx->intern, st.params.vlm_prompt.c_str());
            int w = 0, h = 0, c = 0;
            unsigned char* px = stbi_load(path, &w, &h, &c, 3);
            if (!px) return {};
            int len = 0;
            const char* t = internvl2_ocr_recognize_raw(ctx->intern, px, w, h, 3, &len);
            auto out = wrap_fulltext(t, w, h);
            stbi_image_free(px);
            return out;
        }
        case engine::tesseract: {
            // DBNet detection (model_a) + per-line Tesseract-LSTM recognition
            // (model_b). Tesseract-LSTM recognizes a single text line, so each
            // detected region is cropped (grayscale) and recognized in turn.
            if (!ctx->tess_det) {
                if (st.model_a.empty() || st.model_b.empty()) {
                    fprintf(stderr, "ocr_orchestrator: tesseract stage missing models "
                            "(model_a=det, model_b=tesseract)\n");
                    return {};
                }
                if (!ocr_detect::load(&ctx->tess_det, st.model_a.c_str(), ctx->n_threads)) {
                    fprintf(stderr, "ocr_orchestrator: tesseract detection load failed\n");
                    ctx->tess_det = nullptr;
                    return {};
                }
            }
            if (!ctx->tess) {
                ctx->tess = tesseract_lstm_init(st.model_b.c_str(), ctx->n_threads);
                if (!ctx->tess) { fprintf(stderr, "ocr_orchestrator: tesseract load failed\n"); return {}; }
            }
            auto boxes = ocr_detect::detect_file(ctx->tess_det, path,
                                                 st.params.det_prob_threshold,
                                                 st.params.det_box_threshold,
                                                 1.5f, st.params.det_target_short);
            if (boxes.empty()) return {};
            int w = 0, h = 0, c = 0;
            unsigned char* gray = stbi_load(path, &w, &h, &c, 1); // force 1-channel
            if (!gray) return {};
            std::vector<ocr_pipeline::ocr_result> results;
            results.reserve(boxes.size());
            const int pad = 2;
            for (auto& b : boxes) {
                int cx = std::max(0, (int)b.x - pad);
                int cy = std::max(0, (int)b.y - pad);
                int cw = std::min((int)b.w + 2 * pad, w - cx);
                int chh = std::min((int)b.h + 2 * pad, h - cy);
                if (cw <= 0 || chh <= 0) continue;
                std::vector<uint8_t> crop((size_t)cw * chh);
                for (int y = 0; y < chh; y++) {
                    std::memcpy(crop.data() + (size_t)y * cw,
                                gray + (size_t)(cy + y) * w + cx, (size_t)cw);
                }
                int len = 0;
                const char* t = tesseract_lstm_recognize(ctx->tess, crop.data(), cw, chh, &len);
                if (!t || len <= 0) continue;
                ocr_pipeline::ocr_result r;
                r.box = b;
                // Mean per-character confidence from the last recognition.
                int n_conf = 0;
                const float* conf = tesseract_lstm_confidences(ctx->tess, &n_conf);
                float mean = b.score;
                if (conf && n_conf > 0) {
                    double s = 0.0;
                    for (int k = 0; k < n_conf; k++) s += conf[k];
                    mean = (float)(s / n_conf);
                }
                r.confidence = mean;
                r.text = std::string(t, len);
                if (!r.text.empty()) results.push_back(std::move(r));
            }
            stbi_image_free(gray);
            return results;
        }
        case engine::parseq:
        default:
            // ParSeq is a scene-text recognizer (needs detection); not a
            // standalone document engine — skip in the orchestrator for now.
            fprintf(stderr, "ocr_orchestrator: engine %d not wired\n", (int)st.eng);
            return {};
    }
}

static result assemble(std::vector<ocr_pipeline::ocr_result> regions, engine eng,
                       source_type st) {
    result r;
    r.used_engine = eng;
    r.used_type   = st;
    double conf_sum = 0.0;
    std::string joined;
    for (auto& reg : regions) {
        if (!joined.empty()) joined += "\n";
        joined += reg.text;
        conf_sum += reg.confidence;
    }
    r.mean_confidence = regions.empty() ? 0.0f
                                        : (float)(conf_sum / (double)regions.size());
    r.full_text = std::move(joined);
    r.regions   = std::move(regions);
    return r;
}

static bool passes_gate(const result& r, const accept_gate& g) {
    if ((int)r.full_text.size() < g.min_chars) return false;
    if (g.min_confidence > 0.0f && r.mean_confidence < g.min_confidence) return false;
    return true;
}

// Estimate effective DPI from image dimensions. Documents are typically
// letter/A4 (~8.5x11 in). If the image is small enough to suggest low DPI,
// apply text super-resolution before OCR.
static int estimate_dpi(int w, int h) {
    // Assume the longer dimension corresponds to ~11 inches (letter/A4 long edge)
    int longer = std::max(w, h);
    return (int)(longer / 11.0f + 0.5f);
}

// Run text SR on the image if estimated DPI is below the threshold.
// Returns path to upscaled temp PNG (empty if SR was skipped or failed).
// Auto-detects PAN vs NAFNet-SR from GGUF architecture metadata.
static std::string maybe_sr(context* ctx, const char* src) {
    if (ctx->cfg.sr_model.empty()) return "";

    int w = 0, h = 0, c = 0;
    unsigned char* d = stbi_load(src, &w, &h, &c, 3);
    if (!d) return "";

    int dpi = estimate_dpi(w, h);
    if (dpi >= ctx->cfg.sr_target_dpi) {
        stbi_image_free(d);
        return "";
    }

    // Lazy-load SR model (auto-detect engine from GGUF architecture)
    if (ctx->sr_kind == context::SR_NONE) {
        // Detect architecture
        gguf_context* meta = core_gguf::open_metadata(ctx->cfg.sr_model.c_str());
        std::string arch;
        if (meta) {
            arch = core_gguf::kv_str(meta, "general.architecture", "text_sr");
            core_gguf::free_metadata(meta);
        }

        if (arch == "pan") {
            ctx->pan = pan_sr_init(ctx->cfg.sr_model.c_str(), ctx->n_threads);
            if (ctx->pan) ctx->sr_kind = context::SR_PAN;
            else fprintf(stderr, "ocr_orchestrator: pan_sr load failed\n");
        } else {
            ctx->sr = text_sr_init(ctx->cfg.sr_model.c_str(), ctx->n_threads);
            if (ctx->sr) ctx->sr_kind = context::SR_NAFNET;
            else fprintf(stderr, "ocr_orchestrator: text_sr load failed\n");
        }

        if (ctx->sr_kind == context::SR_NONE) {
            stbi_image_free(d);
            return "";
        }
    }

    uint8_t* out = nullptr;
    int ow = 0, oh = 0;
    int rc = -1;
    if (ctx->sr_kind == context::SR_PAN) {
        rc = pan_sr_process(ctx->pan, d, w, h, 0, 0, &out, &ow, &oh);
    } else {
        rc = text_sr_process(ctx->sr, d, w, h, 0, 0, &out, &ow, &oh);
    }
    if (rc != 0 || !out) {
        stbi_image_free(d);
        return "";
    }
    stbi_image_free(d);

    std::string out_path = temp_png_path();
    if (stbi_write_png(out_path.c_str(), ow, oh, 3, out, ow * 3) == 0) {
        if (ctx->sr_kind == context::SR_PAN) pan_sr_free_image(out);
        else text_sr_free_image(out);
        return "";
    }
    if (ctx->sr_kind == context::SR_PAN) pan_sr_free_image(out);
    else text_sr_free_image(out);

    if (ctx->cfg.verbose)
        fprintf(stderr, "ocr_orchestrator: SR(%s) %dx%d (%d DPI) -> %dx%d\n",
                ctx->sr_kind == context::SR_PAN ? "PAN" : "NAFNet",
                w, h, dpi, ow, oh);
    return out_path;
}

static const chain* pick_chain(const config& cfg, source_type st) {
    const chain* fallback = nullptr;
    for (auto& c : cfg.chains) {
        if (c.type == st) return &c;
        if (c.type == source_type::auto_detect) fallback = &c;
    }
    if (fallback) return fallback;
    return cfg.chains.empty() ? nullptr : &cfg.chains.front();
}

// ── public API ────────────────────────────────────────────────────────────────

bool load(context** out, const config& cfg, int n_threads) {
    if (!out) return false;
    auto* ctx       = new context();
    ctx->cfg        = cfg;
    ctx->n_threads  = n_threads;
    *out            = ctx;
    return true;
}

static const char * engine_name(engine e) {
    switch (e) {
        case engine::dbnet_trocr: return "dbnet_trocr";
        case engine::surya:       return "surya";
        case engine::got:         return "got";
        case engine::glm:         return "glm";
        case engine::qwen2vl:     return "qwen2vl";
        case engine::internvl2:   return "internvl2";
        case engine::tesseract:   return "tesseract";
        default: return "unknown";
    }
}

static const char * source_type_name(source_type t) {
    switch (t) {
        case source_type::auto_detect: return "auto";
        case source_type::screenshot:  return "screenshot";
        case source_type::scanned_doc: return "scanned_doc";
        case source_type::photo:       return "photo";
        default: return "unknown";
    }
}

result run_file(context* ctx, const char* image_path) {
    result best;
    if (!ctx || !image_path) return best;
    bool verbose = ctx->cfg.verbose;

    const source_type st =
        ctx->cfg.router ? classify_file(image_path) : source_type::auto_detect;
    if (verbose)
        fprintf(stderr, "ocr_orchestrator: source_type=%s for %s\n",
                source_type_name(st), image_path);

    const chain* ch = pick_chain(ctx->cfg, st);
    if (!ch) {
        if (verbose) fprintf(stderr, "ocr_orchestrator: no chain for source_type=%s\n",
                             source_type_name(st));
        return best;
    }

    // Text super-resolution: upscale low-DPI images before OCR
    std::string sr_path = maybe_sr(ctx, image_path);
    const char* effective_path = sr_path.empty() ? image_path : sr_path.c_str();

    int tried = 0;
    for (const stage& s : ch->stages) {
        if (!s.enabled) continue;
        tried++;

        if (verbose)
            fprintf(stderr, "ocr_orchestrator: stage %d engine=%s cleanup=%s\n",
                    tried, engine_name(s.eng),
                    s.cleanup.enabled ? "on" : "off");

        std::string tmp = clean_to_temp(ctx, s.cleanup, effective_path);
        const char* ocr_path = tmp.empty() ? effective_path : tmp.c_str();

        result r = assemble(run_engine(ctx, s, ocr_path), s.eng, st);
        r.used_type    = st;
        r.stages_tried = tried;

        if (!tmp.empty()) std::remove(tmp.c_str());

        bool passed = passes_gate(r, s.accept);
        if (verbose)
            fprintf(stderr, "ocr_orchestrator: stage %d → %d chars, conf=%.2f, gate=%s\n",
                    tried, (int)r.full_text.size(), r.mean_confidence,
                    passed ? "PASS" : "FAIL");

        if (passed) {
            if (!sr_path.empty()) std::remove(sr_path.c_str());
            return r;
        }
        if (r.full_text.size() > best.full_text.size()) best = std::move(r);
    }
    if (!sr_path.empty()) std::remove(sr_path.c_str());

    if (verbose)
        fprintf(stderr, "ocr_orchestrator: all %d stages failed gate, returning best (%d chars)\n",
                tried, (int)best.full_text.size());
    best.used_type    = st;
    best.stages_tried = tried;
    return best;
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->dbnet)  ocr_pipeline::free(ctx->dbnet);
    if (ctx->got)    got_ocr_free(ctx->got);
    if (ctx->glm)    glm_ocr_free(ctx->glm);
    if (ctx->qwen)   qwen2vl_ocr_free(ctx->qwen);
    if (ctx->intern) internvl2_ocr_free(ctx->intern);
    if (ctx->tess_det) ocr_detect::free(ctx->tess_det);
    if (ctx->tess)   tesseract_lstm_free(ctx->tess);
    if (ctx->clean1) scan_cleanup_free(ctx->clean1);
    if (ctx->clean2) scan_cleanup_free(ctx->clean2);
    if (ctx->sr)     text_sr_free(ctx->sr);
    if (ctx->pan)    pan_sr_free(ctx->pan);
    delete ctx;
}

} // namespace ocr_orchestrator
