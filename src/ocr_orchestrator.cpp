// ocr_orchestrator.cpp — see ocr_orchestrator.h.
//
// Slice A: source-type classifier + per-stage cleanup (classical + NAFNet) fed
// to the DBNet+TrOCR `ocr_pipeline` engine, with a text-yield + confidence
// accept-gate that escalates through the chain. Cleanup → engine handoff is via
// a transient temp PNG so it works uniformly with every path-based engine
// (slice B wires the remaining ggml engines into `run_engine`).

#include "ocr_orchestrator.h"
#include "crispembed.h"

#include "scan_cleanup.h"
#include "ocr_pipeline.h"
#include "ocr_crop.h"
#include "easyocr_layout.h"
#include "easyocr_pipeline.h"
// Single-shot VLM/document OCR engines (full image → text). C API.
#include "got_ocr.h"
#include "glm_ocr.h"
#include "qwen2vl_ocr.h"
#include "internvl2_ocr.h"
#include "deepseek_ocr2.h"
#include "pix2struct.h"
#include "granite_vision_ocr.h"
#include "lightonocr.h"
#include "lfm2_vl_ocr.h"
#include "unlimited_ocr.h"
// Tesseract-LSTM line recognizer + DBNet detection (the tesseract engine pairs
// detection with per-line tesseract recognition).
#include "tesseract_lstm.h"
#include "tesseract_pageseg.h"
#include "parseq_ocr.h"
#include "ocr_detect.h"
#include "ppocrv6_det.h"
#include "ppocrv6_ocr.h"
#include "pplcnet_orientation.h"
#include "table_parse.h"
#include "ppformulanet_ocr.h"
#include "ppformulanet_l_ocr.h"
// Text super-resolution (low-DPI upscale before OCR).
#include "text_sr.h"
#include "pan_sr.h"
#include "core/gguf_loader.h"
// Text LID for language-aware Tesseract model selection (optional).
#if __has_include("text_lid_dispatch.h")
#include "text_lid_dispatch.h"
#define CRISPEMBED_HAS_LID 1
#else
#define CRISPEMBED_HAS_LID 0
#endif
// Truecasing (optional, from crisp_truecase shared lib).
#if __has_include("crisp_truecase.h")
#include "crisp_truecase.h"
#define CRISPEMBED_HAS_TRUECASE 1
#else
#define CRISPEMBED_HAS_TRUECASE 0
#endif

// Implementation lives in core/image_out.cpp (single definition project-wide).
#include "../ggml/examples/stb_image_write.h"

#include "core/temp_file.h"
#include "core/env_gate.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <future>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h> // _getpid
#else
#include <unistd.h> // getpid
#endif

// setenv(name, value, /*overwrite=*/0) is POSIX-only; MSVC has _putenv_s, which
// always overwrites. Both callers here want "set a default the user can still
// override from the environment", so the presence check carries the semantics
// on every platform.
static void set_env_if_unset(const char * name, const char * value) {
    if (std::getenv(name)) return;
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/0);
#endif
}

// stbi_load is exported (non-static) by image_preprocess.cpp's
// STB_IMAGE_IMPLEMENTATION; forward-declare what we use rather than re-include.
extern "C" {
unsigned char * stbi_load(const char * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

// Otsu threshold over an 8-bit grey page. Shared by the coverage and column
// probes so they cannot disagree about what counts as ink.
static int pageseg_otsu(const uint8_t * gray, size_t n) {
    size_t hist[256] = { 0 };
    for (size_t i = 0; i < n; ++i) hist[gray[i]]++;
    double sum_all = 0.0;
    for (int t = 0; t < 256; ++t) sum_all += (double)t * (double)hist[t];
    double sum_bg = 0.0, w_bg = 0.0, best = -1.0;
    int thresh = 128;
    for (int t = 0; t < 256; ++t) {
        w_bg += (double)hist[t];
        if (w_bg == 0.0) continue;
        const double w_fg = (double)n - w_bg;
        if (w_fg <= 0.0) break;
        sum_bg += (double)t * (double)hist[t];
        const double var =
            w_bg * w_fg * (sum_bg / w_bg - (sum_all - sum_bg) / w_fg) * (sum_bg / w_bg - (sum_all - sum_bg) / w_fg);
        if (var > best) {
            best = var;
            thresh = t;
        }
    }
    return thresh;
}

// Fraction of the page's foreground ink that falls inside the proposed boxes.
//
// The segmentation router (H9) needs to judge a classical segmentation without
// running the detector it is trying to skip, so the test has to come from the
// page itself. Under-segmentation -- the way projection segmentation fails on
// dense scans -- leaves most of the text ink outside the returned boxes, which
// this measures directly. Region count cannot: a dense receipt collapsing to 2
// boxes and a clean page that genuinely has 2 boxes are identical by count.
//
// Otsu picks the ink/paper split rather than a fixed threshold, so the measure
// survives grey scans and inverted stock. Boxes are rasterised into a page mask
// first so overlapping proposals cannot count the same pixel twice (which would
// otherwise let a pile of overlapping boxes fake full coverage).
static double pageseg_ink_coverage(const uint8_t * gray, int w, int h,
                                   const std::vector<ocr_detect::text_box> & boxes) {
    if (!gray || w <= 0 || h <= 0) return 0.0;
    const size_t n = (size_t)w * (size_t)h;

    const int thresh = pageseg_otsu(gray, n);

    // Ink is the darker class. A page with no ink at all has nothing to cover;
    // report full coverage so a blank page is not routed to the detector.
    size_t total_ink = 0;
    for (size_t i = 0; i < n; ++i)
        if (gray[i] <= thresh) total_ink++;
    if (total_ink == 0) return 1.0;

    std::vector<uint8_t> mask(n, 0);
    for (const auto & b : boxes) {
        int x0 = (int)std::floor(b.x), y0 = (int)std::floor(b.y);
        int x1 = (int)std::ceil(b.x + b.w), y1 = (int)std::ceil(b.y + b.h);
        x0 = std::max(0, x0);
        y0 = std::max(0, y0);
        x1 = std::min(w, x1);
        y1 = std::min(h, y1);
        for (int y = y0; y < y1; ++y) std::memset(mask.data() + (size_t)y * w + x0, 1, (size_t)(x1 - x0));
    }
    size_t covered = 0;
    for (size_t i = 0; i < n; ++i)
        if (mask[i] && gray[i] <= thresh) covered++;
    return (double)covered / (double)total_ink;
}

// Number of text columns, detected as a gutter with text on BOTH sides.
//
// This is the signal the H9 router needs, and it took looking at the fixtures to
// find it. Projection segmentation groups rows by horizontal ink profile, so a
// two-column page merges the columns into single rows and the line crops become
// unusable -- that, not density or noise, is why commons_test_ocr_document.jpg
// drops from 1754 characters to 677 while single-column receipts and
// german_official_print do fine. Ink coverage and median box height both failed
// to separate these because they are page-level scalars and the property is
// structural.
//
// A plain "run of empty columns" is NOT enough, and measuring it that way was
// wrong: it called german_official_print 2 columns and receipt_historical 3,
// because a ragged right margin or a short receipt line leaves a tall empty
// band that looks identical to a gutter from the column totals alone. A real
// gutter has text to its LEFT and RIGHT on the same rows. Requiring that on a
// majority of text-bearing rows is what separates a column break from
// whitespace.
// Second accept test for a candidate column gutter, on top of the "most text
// rows have ink on both sides" one (pageseg_column_count below).
//
// That first test does not separate the cases at all -- MEASURED over the 24
// arms fixtures: true two-column commons_test scores 0.60 while the two
// FALSE positives score 0.62 and 0.82. It cannot: a single line of ragged
// text with one vertically-aligned word gap has ink on both sides of that gap
// on every row, exactly like two real columns do. The consequence was live --
// synth_02_clean/synth_03_clean (three centred lines each) were read as
// two-column and re-routed to DBNet, costing CER 0.0146/0.0439 against the
// 0.0000 classical achieves on both.
//
// What actually separates them is INK BALANCE across the gutter, which is
// what "two columns" MEANS rather than a fitted number: a page typeset in two
// columns carries comparable ink mass on each side (commons_test: 129280 vs
// 129555, balance 1.00), while a coincidental word gap at x~0.69w leaves most
// of the page's ink on one side (0.28 and 0.25). The 0.5 bound sits with ~2x
// margin on both sides of that gap.
//
// The width escape covers the known untested case: a genuine two-column page
// whose second column is mostly empty (end of an article) would fail the
// balance test, but its gutter is a real typeset gutter -- 0.055 of page width
// on commons_test versus 0.011/0.012 for the coincidences, a 5x separation.
//
// ⚠ SHIPS OPT-IN (`CRISPEMBED_TESSERACT_SEG_GUTTER_BALANCE=1`), because a more
// truthful column count did NOT translate into better output. Being an
// ADDITIONAL condition it can only move a page from DBNet to classical, and
// over the 20-render truthed corpus that is a WASH: synth_02_clean 0.0146 ->
// 0.0000 and synth_02_skew 0.0219 -> 0.0000, but synth_01_noise 0.0299 ->
// 0.0746 and synth_03_clean 0.0439 -> 0.0614 (mean 0.0189 -> 0.0202). The two
// regressions are pages the FALSE positive was accidentally routing to DBNet,
// which happens to win on them. So the column count is a proxy for "which
// segmenter wins", and making the proxy honest does not by itself improve the
// routing -- kept and gated per the standing rule rather than deleted, since
// the mis-classification is real and a better router will want this.
static bool pageseg_gutter_is_column_split(const uint8_t * gray, int w, int h, int thresh, int gl, int gr) {
    if (!core_env::on("CRISPEMBED_TESSERACT_SEG_GUTTER_BALANCE")) return true;
    if ((double)(gr - gl) >= 0.03 * (double)w) return true; // a real typeset gutter
    size_t ink_left = 0, ink_right = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t * r = gray + (size_t)y * w;
        for (int xx = 0; xx < gl; ++xx)
            if (r[xx] <= thresh) ink_left++;
        for (int xx = gr; xx < w; ++xx)
            if (r[xx] <= thresh) ink_right++;
    }
    const size_t lo_ink = std::min(ink_left, ink_right), hi_ink = std::max(ink_left, ink_right);
    if (hi_ink == 0) return false;
    return (double)lo_ink / (double)hi_ink >= 0.5;
}

static int pageseg_column_count(const uint8_t * gray, int w, int h) {
    if (!gray || w <= 0 || h <= 0) return 1;
    const int thresh = pageseg_otsu(gray, (size_t)w * (size_t)h);

    // Rows that carry text at all; blank leading/trailing rows would otherwise
    // dilute the majority test below.
    std::vector<int> row_ink(h, 0);
    std::vector<int> col(w, 0);
    for (int y = 0; y < h; ++y) {
        const uint8_t * r = gray + (size_t)y * w;
        for (int x = 0; x < w; ++x)
            if (r[x] <= thresh) {
                row_ink[y]++;
                col[x]++;
            }
    }
    int row_peak = 0;
    for (int y = 0; y < h; ++y) row_peak = std::max(row_peak, row_ink[y]);
    if (row_peak <= 0) return 1;
    const int row_min = std::max(1, (int)(0.05 * (double)row_peak));
    std::vector<int> text_rows;
    for (int y = 0; y < h; ++y)
        if (row_ink[y] >= row_min) text_rows.push_back(y);
    if (text_rows.size() < 8) return 1; // too little text to talk about columns

    int col_peak = 0;
    for (int x = 0; x < w; ++x) col_peak = std::max(col_peak, col[x]);
    const int empty_max = std::max(1, (int)(0.02 * (double)col_peak));
    const int min_gutter_w = std::max(3, (int)(0.012 * (double)w));
    const int lo = (int)(0.15 * (double)w), hi = (int)(0.85 * (double)w);

    int columns = 1, run_start = -1;
    for (int x = lo; x <= hi; ++x) {
        const bool empty = (x < hi) && col[x] <= empty_max;
        if (empty) {
            if (run_start < 0) run_start = x;
            continue;
        }
        if (run_start >= 0 && x - run_start >= min_gutter_w) {
            // Candidate gutter [run_start, x). Accept only if most text rows
            // carry ink on both sides of it.
            const int gl = run_start, gr = x;
            size_t both = 0;
            for (int y : text_rows) {
                const uint8_t * r = gray + (size_t)y * w;
                bool left = false, right = false;
                for (int xx = 0; xx < gl && !left; ++xx)
                    if (r[xx] <= thresh) left = true;
                for (int xx = gr; xx < w && !right; ++xx)
                    if (r[xx] <= thresh) right = true;
                if (left && right) both++;
            }
            if (both * 2 > text_rows.size() && pageseg_gutter_is_column_split(gray, w, h, thresh, gl, gr)) columns++;
        }
        run_start = -1;
    }
    return columns;
}

static bool load_gray_exact(const char * path, std::vector<uint8_t> & gray, int * out_w, int * out_h) {
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    gray.clear();
    int w = 0, h = 0, c = 0;
    unsigned char * rgb = stbi_load(path, &w, &h, &c, 3);
    if (!rgb || w <= 0 || h <= 0) {
        if (rgb) stbi_image_free(rgb);
        return false;
    }
    gray.resize((size_t)w * h);
    for (size_t i = 0; i < gray.size(); ++i) {
        const unsigned char r = rgb[3 * i + 0];
        const unsigned char g = rgb[3 * i + 1];
        const unsigned char b = rgb[3 * i + 2];
        gray[i] = (uint8_t)((77u * r + 150u * g + 29u * b) >> 8);
    }
    stbi_image_free(rgb);
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return true;
}

namespace ocr_orchestrator {

struct context {
    config cfg;
    int n_threads = 1;
    // Lazily-loaded engine + cleanup handles (loaded on first use).
    ocr_pipeline::context * dbnet = nullptr;        // DBNet detection + TrOCR recognition
    ppocrv6_det::context * ppdet = nullptr;         // PP-OCRv6 detector
    ppocrv6_ocr_context * pprec = nullptr;          // PP-OCRv6 recognizer
    pplcnet_orientation::context * ppori = nullptr; // optional PP-LCNet line orientation
    easyocr_pipeline::context * easy = nullptr;     // DBNet detection + EasyOCR CRNN recognition
    layout_detect::context * layout = nullptr;      // optional document layout
    table_parse_context * table = nullptr;
    ppformulanet_ocr_context * formula = nullptr;
    ppformulanet_l_ocr_context * formula_l = nullptr;
    got_ocr_context * got = nullptr;             // GOT-OCR2 (single-shot VLM)
    glm_ocr_context * glm = nullptr;             // GLM-OCR (single-shot VLM)
    qwen2vl_ocr_context * qwen = nullptr;        // Qwen2.5-VL (single-shot VLM)
    qwen2vl_ocr_context * qwen3 = nullptr;       // Qwen3-VL (DeepStack + IMROPE)
    internvl2_ocr_context * intern = nullptr;    // InternVL2 (single-shot VLM)
    deepseek_ocr2_context * dsocr2 = nullptr;    // DeepSeek-OCR-2 (MoE VLM)
    pix2struct_context * p2s = nullptr;          // Pix2Struct (doc/chart understanding)
    granite_vision_context * gv = nullptr;       // Granite Vision (LLaVA-Next)
    lightonocr_context * locr = nullptr;         // LightOnOCR (Pixtral ViT + Qwen3)
    lfm2_vl_ocr_context * lfm2vl = nullptr;      // LFM2.5-VL (SigLIP2 NaFlex multi-tile)
    unlimited_ocr_context * uocr = nullptr;      // Unlimited-OCR (SAM + CLIP + MoE)
    void * unified = nullptr;                    // metadata-dispatched OCR model
    ocr_detect::context * tess_det = nullptr;    // DBNet detection for the tesseract engine
    ppocrv6_det::context * tess_ppdet = nullptr; // PP-OCRv6 detection for the tesseract engine (CJK line boxes)
    tesseract_lstm_context * tess = nullptr;     // Tesseract-LSTM line recognizer
    std::vector<tesseract_lstm_context *> tess_workers;
    ocr_detect::context * parseq_det = nullptr; // DBNet detection for the parseq engine
    parseq_ocr_context * parseq = nullptr;      // PARSeq scene-text recognizer (per-char conf)
    scan_cleanup_ctx * clean1 = nullptr;        // tier-1 classical (model = NULL)
    scan_cleanup_ctx * clean2 = nullptr;        // tier-2 NAFNet (model = nafnet_model)
    text_sr_context * sr = nullptr;             // NAFNet-SR (low-DPI upscale)
    pan_sr_context * pan = nullptr;             // PAN 4x SR (alternative upscaler)
    enum { SR_NONE, SR_NAFNET, SR_PAN } sr_kind = SR_NONE;
#if CRISPEMBED_HAS_LID
    text_lid_context * lid = nullptr; // text LID for language routing
#endif
    std::string detected_lang; // cached LID result
    float lang_confidence = 0.0f;
    std::string tess_resolved_model; // LID-resolved tesseract model path
    bool bench = false;
#if CRISPEMBED_HAS_TRUECASE
    truecaser_lstm_context * tc = nullptr; // truecaser (BiLSTM)
#endif
};

// ── defaults ────────────────────────────────────────────────────────────────

static cleanup_profile classical_profile(bool binarize) {
    cleanup_profile p;
    p.enabled = true;
    p.params = scan_cleanup_defaults(); // deskew+crop+whiten on
    p.params.binarize = binarize ? 1 : 0;
    p.denoise = false;
    return p;
}

static cleanup_profile denoise_profile(bool deskew) {
    cleanup_profile p;
    p.enabled = true;
    p.params = scan_cleanup_defaults();
    p.params.binarize = 0; // never binarize for detector/VLM
    p.params.deskew = deskew ? 1 : 0;
    p.denoise = true; // NAFNet
    return p;
}

static stage dbnet_stage(cleanup_profile cp) {
    stage s;
    s.eng = engine::dbnet_trocr;
    s.enabled = true;
    s.cleanup = cp;
    // model_a/model_b left empty — the caller (CrispSorter / CLI) resolves the
    // DBNet + TrOCR GGUF paths and fills them before load().
    return s;
}

stage tesseract_fraktur_stage() {
    stage s;
    s.eng = engine::tesseract_fraktur;
    s.enabled = true;
    s.cleanup = classical_profile(false);
    // Historical Fraktur pages contain narrow glyphs and long text lines.
    s.params.det_min_height = 18;
    s.params.det_width_height_ratio = 20.0f;
    s.accept.min_chars = 4;
    s.accept.min_confidence = 0.25f;
    return s;
}

config default_config() {
    config cfg;
    cfg.router = true;
    // nafnet_model left empty by default; caller sets it to enable tier-2.
    chain screenshot{ source_type::screenshot, { dbnet_stage(classical_profile(false)) } };
    chain scan{ source_type::scanned_doc, { dbnet_stage(classical_profile(true)) } };
    chain photo{ source_type::photo, { dbnet_stage(denoise_profile(true)) } };
    chain any{ source_type::auto_detect, { dbnet_stage(classical_profile(false)) } };
    cfg.chains = { screenshot, scan, photo, any };
    return cfg;
}

// ── source-type classifier (cheap pixel heuristics) ──────────────────────────

source_type classify_file(const char * image_path) {
    if (!image_path) return source_type::scanned_doc;
    int w = 0, h = 0, c = 0;
    unsigned char * d = stbi_load(image_path, &w, &h, &c, 0);
    if (!d || w <= 0 || h <= 0) {
        if (d) stbi_image_free(d);
        return source_type::scanned_doc;
    }

    const bool has_alpha = (c == 4);
    const double aspect = (double)w / (double)h;

    // Mean saturation + fraction of near-white pixels over a stride sample.
    double sat_sum = 0.0;
    long near_white = 0;
    long n = 0;
    const int stride = (w * h > 400000) ? 7 : 1; // subsample big images
    for (long i = 0; i < (long)w * h; i += stride) {
        const unsigned char * px = d + (size_t)i * c;
        int r = px[0], g = px[1], b = (c >= 3) ? px[2] : px[0];
        int mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
        int mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
        sat_sum += (mx == 0) ? 0.0 : (double)(mx - mn) / (double)mx;
        if (mn >= 200) near_white++;
        n++;
    }
    stbi_image_free(d);
    const double mean_sat = n ? sat_sum / (double)n : 0.0;
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

// Create an empty, private temporary PNG and return its path.
//
// Delegates to core_tmp::make_private. This used to build a predictable name
// (/tmp/crispembed_ocr_<pid>_<counter>.png) and leave stbi_write_png to
// fopen(..., "wb") it — symlink-redirectable, world-readable, holding the
// user's scanned page. The server had the identical defect in its own copy;
// there is now one implementation for both.
static std::string temp_png_path() {
    return core_tmp::make_private(".png");
}

// Run scan cleanup on `src` and return an owned RGB buffer. The caller owns the
// vector, so raw-image engines can consume cleanup output without a disk round
// trip. `ow`/`oh` are set only on success.
static bool clean_to_pixels(context * ctx, const cleanup_profile & cp, const char * src, std::vector<uint8_t> & pixels,
                            int * ow, int * oh) {
    if (ow) *ow = 0;
    if (oh) *oh = 0;
    pixels.clear();
    if (!cp.enabled) return false;
    int w = 0, h = 0, c = 0;
    unsigned char * d = stbi_load(src, &w, &h, &c, 0);
    if (!d) return false;

    scan_cleanup_ctx ** slot = cp.denoise ? &ctx->clean2 : &ctx->clean1;
    if (!*slot) {
        const char * model = (cp.denoise && !ctx->cfg.nafnet_model.empty()) ? ctx->cfg.nafnet_model.c_str()
                                                                            : nullptr; // NULL → tier-1 classical only
        *slot = scan_cleanup_init(model, ctx->n_threads);
    }

    uint8_t * out = nullptr;
    int out_w = 0, out_h = 0;
    const bool ok = *slot && scan_cleanup_process(*slot, d, w, h, c, cp.params, &out, &out_w, &out_h) == 0 && out;
    if (ok) {
        pixels.assign(out, out + (size_t)out_w * out_h * 3);
        if (ow) *ow = out_w;
        if (oh) *oh = out_h;
        scan_cleanup_free_image(out);
    }
    stbi_image_free(d);
    return ok;
}

// Path-only engines still use the legacy temporary-file handoff. Raw-image
// engines should use clean_to_pixels instead.
static std::string clean_to_temp(context * ctx, const cleanup_profile & cp, const char * src) {
    std::vector<uint8_t> pixels;
    int w = 0, h = 0;
    if (!clean_to_pixels(ctx, cp, src, pixels, &w, &h)) return "";
    const std::string out_path = temp_png_path();
    if (stbi_write_png(out_path.c_str(), w, h, 3, pixels.data(), w * 3) == 0) return "";
    return out_path;
}

// Wrap a single-shot VLM engine's whole-image text as one region covering the
// full image. VLMs don't expose per-region confidence, so use 1.0.
static std::vector<ocr_pipeline::ocr_result> wrap_fulltext(const char * text, int w, int h,
                                                           const float * conf = nullptr, int n_conf = 0,
                                                           float mean = 0.0f) {
    std::vector<ocr_pipeline::ocr_result> out;
    if (text && *text) {
        ocr_pipeline::ocr_result r;
        r.box.x = 0.0f;
        r.box.y = 0.0f;
        r.box.w = (float)w;
        r.box.h = (float)h;
        r.box.score = 1.0f;
        // Recognition confidence (mean per-token softmax) when the engine
        // exposes it; per-token vector kept for the proofreading UI.
        r.rec_confidence = mean;
        r.confidence = (mean > 0.0f) ? mean : 1.0f;
        if (conf && n_conf > 0) r.char_conf.assign(conf, conf + n_conf);
        r.text = text;
        out.push_back(std::move(r));
    }
    return out;
}

// ISO 639-1 (LID output) → Tesseract ISO 639-3 code mapping.
static const char * lid_to_tesseract(const char * iso1) {
    if (!iso1) return nullptr;
    struct {
        const char * iso1;
        const char * tess;
    } map[] = {
        { "en", "eng" }, { "de", "deu" }, { "fr", "fra" }, { "es", "spa" }, { "it", "ita" }, { "pt", "por" },
        { "nl", "nld" }, { "ru", "rus" }, { "ar", "ara" }, { "ja", "jpn" }, { "ko", "kor" }, { "zh", "chi_sim" },
    };
    for (auto & m : map)
        if (strcmp(iso1, m.iso1) == 0) return m.tess;
    return nullptr;
}

// Resolve a Tesseract model path from LID language code.
static std::string resolve_tess_model(const config & cfg, const char * iso1) {
    const char * tess_code = lid_to_tesseract(iso1);
    if (!tess_code) return "";
    if (cfg.tess_model_dir.empty()) return "";
    std::string path = cfg.tess_model_dir + "/tesseract-" + std::string(tess_code) + "-q8_0.gguf";
    FILE * f = fopen(path.c_str(), "r");
    if (!f) return "";
    fclose(f);
    return path;
}

static ocr_detect::detect_options detector_options(const engine_params & p) {
    auto o = ocr_detect::rapid_defaults();
    o.prob_threshold = p.det_prob_threshold;
    o.box_threshold = p.det_box_threshold;
    o.target_short_side = p.det_target_short;
    o.max_side = p.det_max_side;
    o.min_height = p.det_min_height;
    o.width_height_ratio = p.det_width_height_ratio;
    o.max_candidates = p.det_max_candidates;
    o.dilation = p.det_dilation;
    o.scoring = p.det_scoring;
    return o;
}

// Run one engine on a (already-cleaned) image. VLM engines use pre-loaded
// pixels (px/pw/ph) when available to avoid redundant stbi_load from disk.
// Falls back to loading from `path` if px is null.
static std::vector<ocr_pipeline::ocr_result> run_engine(context * ctx, const stage & st, const char * path,
                                                        const unsigned char * px = nullptr, int pw = 0, int ph = 0) {
    const auto geometry = detector_options(st.params);
    switch (st.eng) {
    case engine::dbnet_trocr:
    case engine::surya: {
        // DBNet/Surya detection + TrOCR recognition (model_a=det, model_b=rec).
        if (!ctx->dbnet) {
            if (st.model_a.empty() || st.model_b.empty()) {
                fprintf(stderr, "ocr_orchestrator: detect+recognize stage missing model "
                                "paths (model_a=det, model_b=rec)\n");
                return {};
            }
            if (!ocr_pipeline::load(&ctx->dbnet, st.model_a.c_str(), st.model_b.c_str(), ctx->n_threads)) {
                fprintf(stderr, "ocr_orchestrator: detect+recognize load failed\n");
                ctx->dbnet = nullptr;
                return {};
            }
        }
        if (px && pw > 0 && ph > 0)
            return ocr_pipeline::run_raw(ctx->dbnet, px, pw, ph, 3, st.params.det_prob_threshold,
                                         st.params.det_box_threshold, st.params.det_target_short, &geometry);
        return ocr_pipeline::run_file(ctx->dbnet, path, st.params.det_prob_threshold, st.params.det_box_threshold,
                                      st.params.det_target_short, &geometry);
    }
    case engine::easyocr: {
        // EasyOCR's own detector is CRAFT; this stage pairs its CRNN
        // recognizer with the repo's DBNet detector, which is what
        // easyocr_pipeline already validates against the Python reference.
        // `model_a` is therefore the DBNet GGUF, not a CRAFT one.
        const bool easy_bench = core_env::on("CRISPEMBED_EASYOCR_BENCH");
        const auto easy_started = std::chrono::steady_clock::now();
        if (st.model_a.empty() || st.model_b.empty()) {
            fprintf(stderr, "ocr_orchestrator: easyocr stage missing models (model_a=det, model_b=rec)\n");
            return {};
        }
        if (!ctx->easy && !easyocr_pipeline::load(&ctx->easy, st.model_a.c_str(), st.model_b.c_str(), ctx->n_threads))
            return {};
        if (!ctx->easy) return {};
        // Split load from compute, the same way the tesseract and ppocrv6
        // stages do. That split is what exposed a wasted Metal init worth 12.5x
        // in one lane and 7-14% in another; without it a slow loader and a slow
        // engine are indistinguishable in the total.
        const auto easy_loaded = std::chrono::steady_clock::now();
        // Line mode matches EasyOCR's own `paragraph=False` line output; word
        // mode exists for the LayoutLM/Tesseract-style word handoff and would
        // fragment the text a CER comparison sees.
        easyocr_pipeline::set_ordering_mode(ctx->easy, easyocr_layout::ordering_mode::lines);
        const auto items =
            px ? easyocr_pipeline::run_raw(ctx->easy, px, pw, ph, 3) : easyocr_pipeline::run_file(ctx->easy, path);
        std::vector<ocr_pipeline::ocr_result> results;
        results.reserve(items.size());
        for (const auto & it : items) {
            if (it.word.text.empty()) continue;
            ocr_pipeline::ocr_result r;
            r.box = { it.word.x, it.word.y, it.word.w, it.word.h, it.detector_confidence };
            r.text = it.word.text;
            r.confidence = it.word.confidence;
            r.rec_confidence = it.word.confidence;
            results.push_back(std::move(r));
        }
        if (easy_bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            const auto easy_done = std::chrono::steady_clock::now();
            fprintf(stderr,
                    "[easyocr-stage-bench] load=%.1f ms detect+recognize=%.1f ms total=%.1f ms units=%zu results=%zu\n",
                    ms(easy_started, easy_loaded), ms(easy_loaded, easy_done), ms(easy_started, easy_done),
                    items.size(), results.size());
        }
        return results;
    }
    case engine::ppocrv6: {
        const bool ppocr_bench = core_env::on("CRISPEMBED_PPOCRV6_BENCH");
        const auto ppocr_started = std::chrono::steady_clock::now();
        if (st.model_a.empty() || st.model_b.empty()) {
            fprintf(stderr, "ocr_orchestrator: ppocrv6 stage missing models (model_a=det, model_b=rec)\n");
            return {};
        }
        if (!ctx->ppdet) ctx->ppdet = ppocrv6_det::init(st.model_a.c_str(), ctx->n_threads);
        if (!ctx->ppdet) return {};
        const auto ppocr_det_loaded = std::chrono::steady_clock::now();
        // PP-OCRv6's official predictor applies resize_long=960/max-side and
        // rounds dimensions to a 32-pixel grid before inference.  Do not use
        // detect_raw here: the routed C API has the original page pixels, and
        // bypassing this geometry produces a different probability map and
        // materially different region counts on large fixtures.
        //
        // Detection deliberately runs BEFORE the recognizer loads: the
        // detector is a cheap CPU load (~3 ms), and the box count is the one
        // number that decides whether the recognizer's ~1.1 s Metal init pays
        // for itself (T5). Measured one-shot: 3 boxes -> CPU wins (1.86 s vs
        // 2.04-2.17 s); 47 boxes -> Metal wins (8.1 s vs 14.0 s). Crossover
        // at ~175 ms/crop CPU-minus-Metal against the 1.1 s init: ~6-8 boxes.
        auto boxes = ppocrv6_det::detect_file(ctx->ppdet, path, std::min(st.params.det_prob_threshold, 0.2f));
        const auto ppocr_detect_done = std::chrono::steady_clock::now();
        if (!ctx->pprec && std::getenv("CRISPEMBED_PPOCRV6_ONESHOT") && !std::getenv("CRISPEMBED_PPOCRV6_FORCE_CPU")) {
            int cpu_max_regions = 8;
            if (const char * limit = std::getenv("CRISPEMBED_PPOCRV6_ONESHOT_CPU_MAX_REGIONS")) {
                const int parsed = std::atoi(limit);
                if (parsed >= 0) cpu_max_regions = parsed;
            }
            if ((int)boxes.size() <= cpu_max_regions) set_env_if_unset("CRISPEMBED_PPOCRV6_FORCE_CPU", "1");
        }
        if (!ctx->pprec) ctx->pprec = ppocrv6_ocr_init(st.model_b.c_str(), ctx->n_threads);
        if (!ctx->ppori && !st.model_c.empty())
            ctx->ppori = pplcnet_orientation::init(st.model_c.c_str(), ctx->n_threads);
        const auto ppocr_load_done = std::chrono::steady_clock::now();
        if (!ctx->pprec) return {};
        if (ppocr_bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[ppocrv6-load-bench] detector=%.1f ms recognizer+orientation=%.1f ms total=%.1f ms\n",
                    ms(ppocr_started, ppocr_det_loaded), ms(ppocr_detect_done, ppocr_load_done),
                    ms(ppocr_started, ppocr_det_loaded) + ms(ppocr_detect_done, ppocr_load_done));
        }
        if (boxes.empty()) return {};
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_ACCEPT")) {
            int max_graph_regions = 8;
            if (const char * limit = std::getenv("CRISPEMBED_PPOCRV6_GRAPH_MAX_REGIONS")) {
                const int parsed = std::atoi(limit);
                if (parsed >= 0) max_graph_regions = parsed;
            }
            // The opt-in recognizer graph is numerically sound per crop, but
            // Metal graph planning per line is currently too expensive for a
            // full page. Keep direct/small-crop diagnostics available while
            // falling back to the accepted CPU recognizer for large pages.
            ppocrv6_ocr_set_graph_accept(ctx->pprec, (int)boxes.size() <= max_graph_regions ? 1 : 0);
            if ((int)boxes.size() > max_graph_regions && core_env::on("CRISPEMBED_PPOCRV6_BENCH"))
                fprintf(stderr, "[ppocrv6-graph-budget] regions=%zu max=%d action=cpu-fallback\n", boxes.size(),
                        max_graph_regions);
        } else {
            ppocrv6_ocr_set_graph_accept(ctx->pprec, -1);
        }
        int w = pw, h = ph;
        std::vector<uint8_t> owned;
        const unsigned char * rgb = px;
        if (!rgb) {
            int c = 0;
            rgb = stbi_load(path, &w, &h, &c, 3);
            owned.assign(rgb, rgb ? rgb + (size_t)w * h * 3 : nullptr);
            if (rgb) stbi_image_free((void *)rgb);
            rgb = owned.data();
        }
        if (!rgb || w <= 0 || h <= 0) return {};
        std::vector<ocr_pipeline::ocr_result> results;
        std::vector<int> model_widths;
        std::vector<std::vector<uint8_t>> crops;
        std::vector<size_t> crop_box_indices;
        std::vector<int> crop_widths, crop_heights;
        std::vector<ocr_crop::orientation_info> crop_orientations;
        double crop_ms = 0.0, orientation_ms = 0.0, recognize_ms = 0.0;
        for (const auto & b : boxes) {
            const auto crop_started = std::chrono::steady_clock::now();
            int cw = 0, ch = 0;
            const bool has_quad = std::hypot(b.qx[1] - b.qx[0], b.qy[1] - b.qy[0]) > 1.0f;
            auto crop = has_quad ? ocr_crop::extract_quad(rgb, w, h, 3, b.qx, b.qy, 2, &cw, &ch)
                                 : ocr_crop::extract(rgb, w, h, 3, (int)b.x, (int)b.y, (int)b.w, (int)b.h, 2, &cw, &ch);
            crop_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - crop_started).count();
            if (crop.empty()) continue;
            int model_width = std::max(320, int(48.0f * std::max(320.0f / 48.0f, cw / float(std::max(1, ch)))));
            // Mirror the recognizer's default width bucketing (resize_normalize
            // in ppocrv6_ocr.cpp is the source of truth) so this diagnostic
            // reports the widths the graphs actually run at. Before this the
            // line printed the PRE-bucket estimate — "27 unique widths" on a
            // page whose real graph-shape count was ~8, which misled the O13b
            // profile until read against the recognizer code.
            {
                int bucket_step = 64;
                if (const char * be = std::getenv("CRISPEMBED_PPOCRV6_WIDTH_BUCKET")) bucket_step = std::atoi(be);
                if (!std::getenv("CRISPEMBED_PPOCRV6_FIXED_WIDTH") && bucket_step >= 8)
                    model_width = (model_width + bucket_step - 1) / bucket_step * bucket_step;
            }
            if (std::find(model_widths.begin(), model_widths.end(), model_width) == model_widths.end())
                model_widths.push_back(model_width);
            const auto orientation_started = std::chrono::steady_clock::now();
            ocr_crop::orientation_info orientation;
            if (ctx->ppori) {
                const auto classified = pplcnet_orientation::classify_raw(ctx->ppori, crop.data(), cw, ch, 3);
                orientation.angle = classified.angle;
                orientation.confidence = classified.confidence;
                if (classified.angle == 180) {
                    ocr_crop::rotate_180_rgb(crop, cw, ch);
                    orientation.corrected = true;
                }
            } else {
                orientation = ocr_crop::orient_180_rgb_info(crop, cw, ch);
            }
            orientation_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - orientation_started)
                    .count();
            crop_box_indices.push_back(&b - boxes.data());
            crop_widths.push_back(cw);
            crop_heights.push_back(ch);
            crop_orientations.push_back(orientation);
            crops.push_back(std::move(crop));
        }
        std::vector<const uint8_t *> crop_pixels;
        std::vector<std::string> batch_text(crops.size(), std::string(4096, '\0'));
        std::vector<char *> batch_outputs;
        std::vector<int> batch_capacities(crops.size(), 4096), batch_lengths(crops.size(), 0);
        crop_pixels.reserve(crops.size());
        batch_outputs.reserve(crops.size());
        for (size_t i = 0; i < crops.size(); ++i) {
            crop_pixels.push_back(crops[i].data());
            batch_outputs.push_back(batch_text[i].data());
        }
        const auto recognize_started = std::chrono::steady_clock::now();
        ppocrv6_ocr_recognize_raw_batch(ctx->pprec, crop_pixels.data(), crop_widths.data(), crop_heights.data(),
                                        std::vector<int>(crops.size(), 3).data(), (int)crops.size(),
                                        batch_outputs.data(), batch_capacities.data(), batch_lengths.data());
        recognize_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - recognize_started).count();
        for (size_t i = 0; i < crops.size(); ++i) {
            if (batch_lengths[i] <= 0) continue;
            ocr_pipeline::ocr_result r;
            const auto & b = boxes[crop_box_indices[i]];
            r.box = { b.x, b.y, b.w, b.h, b.score };
            r.text.assign(batch_text[i].data(), (size_t)batch_lengths[i]);
            r.confidence = b.score;
            r.rec_confidence = b.score;
            r.orientation_corrected = crop_orientations[i].corrected;
            r.orientation_angle = crop_orientations[i].angle;
            r.orientation_confidence = crop_orientations[i].confidence;
            results.push_back(std::move(r));
        }
        if (ppocr_bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            // detect/total exclude model-load time: the load cost is on the
            // [ppocrv6-load-bench] line above. The pre-2026-08 figures spanned
            // stage entry, silently folding a ~1.1 s one-shot Metal init into
            // "detect" — which inflated every load-excluded engine_ms
            // comparison built on this line. Detection now runs before the
            // recognizer loads (box-count backend choice), so total is the
            // detect span plus the post-load span.
            fprintf(
                stderr,
                "[ppocrv6-stage-bench] detect=%.1f ms crop=%.1f ms orientation=%.1f ms recognize=%.1f ms total=%.1f ms "
                "boxes=%zu results=%zu\n",
                ms(ppocr_det_loaded, ppocr_detect_done), crop_ms, orientation_ms, recognize_ms,
                ms(ppocr_det_loaded, ppocr_detect_done) + ms(ppocr_load_done, std::chrono::steady_clock::now()),
                boxes.size(), results.size());
            fprintf(stderr, "[ppocrv6-width-bench] crops=%zu unique_model_widths=%zu widths=", boxes.size(),
                    model_widths.size());
            for (size_t i = 0; i < model_widths.size(); ++i) fprintf(stderr, "%s%d", i ? "," : "", model_widths[i]);
            fprintf(stderr, "\n");
        }
        return results;
    }
    case engine::got: {
        if (!ctx->got) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: got stage missing model_a\n");
                return {};
            }
            ctx->got = got_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->got) {
                fprintf(stderr, "ocr_orchestrator: got load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) got_ocr_set_max_tokens(ctx->got, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = got_ocr_recognize_raw(ctx->got, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = got_ocr_confidences(ctx->got, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, got_ocr_mean_confidence(ctx->got));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::glm: {
        if (!ctx->glm) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: glm stage missing model_a\n");
                return {};
            }
            ctx->glm = glm_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->glm) {
                fprintf(stderr, "ocr_orchestrator: glm load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) glm_ocr_set_max_tokens(ctx->glm, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = glm_ocr_recognize_raw(ctx->glm, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = glm_ocr_confidences(ctx->glm, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, glm_ocr_mean_confidence(ctx->glm));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    // olmocr is the same runtime as qwen2vl — the fine-tune is auto-detected
    // inside qwen2vl_ocr_init from the model file, which switches the prompt
    // contract, token order, and output post-processing.
    case engine::olmocr:
    case engine::qwen2vl: {
        if (!ctx->qwen) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: qwen2vl stage missing model_a\n");
                return {};
            }
            ctx->qwen = qwen2vl_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->qwen) {
                fprintf(stderr, "ocr_orchestrator: qwen2vl load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) qwen2vl_ocr_set_max_tokens(ctx->qwen, st.params.vlm_max_tokens);
        if (!st.params.vlm_prompt.empty()) qwen2vl_ocr_set_prompt(ctx->qwen, st.params.vlm_prompt.c_str());
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = qwen2vl_ocr_recognize_raw(ctx->qwen, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = qwen2vl_ocr_confidences(ctx->qwen, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, qwen2vl_ocr_mean_confidence(ctx->qwen));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::qwen3vl: {
        if (!ctx->qwen3) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: qwen3vl stage missing model_a\n");
                return {};
            }
            ctx->qwen3 = qwen2vl_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->qwen3) {
                fprintf(stderr, "ocr_orchestrator: qwen3vl load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) qwen2vl_ocr_set_max_tokens(ctx->qwen3, st.params.vlm_max_tokens);
        if (!st.params.vlm_prompt.empty()) qwen2vl_ocr_set_prompt(ctx->qwen3, st.params.vlm_prompt.c_str());
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = qwen2vl_ocr_recognize_raw(ctx->qwen3, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = qwen2vl_ocr_confidences(ctx->qwen3, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, qwen2vl_ocr_mean_confidence(ctx->qwen3));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::internvl2: {
        if (!ctx->intern) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: internvl2 stage missing model_a\n");
                return {};
            }
            ctx->intern = internvl2_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->intern) {
                fprintf(stderr, "ocr_orchestrator: internvl2 load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) internvl2_ocr_set_max_tokens(ctx->intern, st.params.vlm_max_tokens);
        if (!st.params.vlm_prompt.empty()) internvl2_ocr_set_prompt(ctx->intern, st.params.vlm_prompt.c_str());
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = internvl2_ocr_recognize_raw(ctx->intern, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = internvl2_ocr_confidences(ctx->intern, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, internvl2_ocr_mean_confidence(ctx->intern));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::deepseek_ocr2: {
        if (!ctx->dsocr2) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: deepseek_ocr2 stage missing model_a\n");
                return {};
            }
            ctx->dsocr2 = deepseek_ocr2_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->dsocr2) {
                fprintf(stderr, "ocr_orchestrator: deepseek_ocr2 load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) deepseek_ocr2_set_max_tokens(ctx->dsocr2, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = deepseek_ocr2_recognize_raw(ctx->dsocr2, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = deepseek_ocr2_confidences(ctx->dsocr2, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, deepseek_ocr2_mean_confidence(ctx->dsocr2));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::tesseract:
    case engine::tesseract_fraktur: {
        // DBNet detection (model_a) + per-line Tesseract-LSTM recognition
        // (model_b). Tesseract-LSTM recognizes a single text line, so each
        // detected region is cropped (grayscale) and recognized in turn.
        //
        // Model load is timed separately from the stage bench below, because
        // for a one-shot CLI invocation it is not a rounding error: it was
        // measured at ~4.6 s against 512 ms of actual detect+recognize work,
        // i.e. 90% of what the user waits for. The stage bench starts after
        // both loads and therefore cannot see it.
        const auto tess_load_start = std::chrono::steady_clock::now();
        if (!ctx->tess_det && !ctx->tess_ppdet) {
            if (st.model_a.empty() || st.model_b.empty()) {
                fprintf(stderr, "ocr_orchestrator: tesseract stage missing models "
                                "(model_a=det, model_b=tesseract)\n");
                return {};
            }
            // model_a dispatches on GGUF metadata: a PP-OCRv6 detector hosts
            // the detection stage with line-level boxes. The IC15 DBNet
            // artifact fragments CJK pages into word-level boxes and the
            // classical segmenter fails on them independently, so this is the
            // page-level path for tesseract-jpn/kor/chi. A ppocrv6 det can
            // only arrive here by an explicit --ocr-det choice (the engine
            // default stays dbnet-det), so no existing invocation changes.
            auto * meta = core_gguf::open_metadata(st.model_a.c_str());
            const bool a_is_ppdet = meta && core_gguf::kv_str(meta, "ppocrv6.kind", "") == "det";
            if (meta) core_gguf::free_metadata(meta);
            if (a_is_ppdet) {
                ctx->tess_ppdet = ppocrv6_det::init(st.model_a.c_str(), ctx->n_threads);
                if (!ctx->tess_ppdet) {
                    fprintf(stderr, "ocr_orchestrator: tesseract ppocrv6 detection load failed\n");
                    return {};
                }
            } else if (!ocr_detect::load(&ctx->tess_det, st.model_a.c_str(), ctx->n_threads)) {
                fprintf(stderr, "ocr_orchestrator: tesseract detection load failed\n");
                ctx->tess_det = nullptr;
                return {};
            }
        }
        const auto tess_det_loaded = std::chrono::steady_clock::now();
        if (!ctx->tess) {
            std::string tess_model = st.model_b;
            // Auto-select: if model_b is "auto" and LID detected a language,
            // resolve to the matching tesseract-{lang} model.
            if (tess_model == "auto") {
#if CRISPEMBED_HAS_LID
                if (!ctx->detected_lang.empty()) {
                    std::string resolved = resolve_tess_model(ctx->cfg, ctx->detected_lang.c_str());
                    if (!resolved.empty()) {
                        tess_model = resolved;
                        if (ctx->cfg.verbose)
                            fprintf(stderr, "ocr_orchestrator: LID auto-select → %s\n", resolved.c_str());
                    }
                }
#endif
                if (tess_model == "auto") {
                    // Fallback to English if no LID or no matching model
                    std::string fallback = resolve_tess_model(ctx->cfg, "en");
                    tess_model = fallback.empty() ? st.model_b : fallback;
                }
            }
            if (tess_model == "auto") {
                fprintf(stderr, "ocr_orchestrator: tesseract model_b='auto' but no models found\n");
                return {};
            }
            ctx->tess = tesseract_lstm_init(tess_model.c_str(), ctx->n_threads);
            if (!ctx->tess) {
                fprintf(stderr, "ocr_orchestrator: tesseract load failed: %s\n", tess_model.c_str());
                return {};
            }
            ctx->tess_resolved_model = tess_model;
        }
        const auto tess_bench_start = std::chrono::steady_clock::now();
        if (core_env::on("CRISPEMBED_TESSERACT_BENCH") || ctx->bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[tesseract-load-bench] detector=%.1f ms recognizer=%.1f ms total=%.1f ms\n",
                    ms(tess_load_start, tess_det_loaded), ms(tess_det_loaded, tess_bench_start),
                    ms(tess_load_start, tess_bench_start));
        }
        std::vector<ocr_detect::text_box> boxes;
        // An EXPLICIT caller request for classical segmentation (the
        // page_segmentation param / CRISPEMBED_TESSERACT_PAGESEG) is separate
        // from the router merely choosing classical, and the two must not be
        // conflated: the router may override its own choice, never the
        // caller's. Measured 2026-08-07 -- with the router defaulted ON,
        // --tesseract-pageseg on the 2-column commons_test fixture still ended
        // up on DBNet (`path=dbnet(fallback)`), so the flag had silently
        // stopped forcing anything.
        const bool explicit_classical =
            st.params.page_segmentation != 0 || std::getenv("CRISPEMBED_TESSERACT_PAGESEG") != nullptr;
        bool classical_pageseg = explicit_classical;
        // PLAN.md H9 — segmentation router (CRISPEMBED_TESSERACT_SEG_ROUTER=1).
        //
        // Classical segmentation is ~4x faster than DBNet on this lane and
        // halves CER on clean single-column print, but loses 92-98% of the text
        // on dense scans. The two fail in opposite directions, so the useful
        // move is to choose per page rather than to pick one globally.
        //
        // The accept test cannot compare against DBNet -- running DBNet is
        // exactly what it is trying to avoid. Ink coverage can be computed from
        // the page alone: binarise, then ask what fraction of the foreground
        // falls inside the proposed boxes. Under-segmentation is precisely the
        // failure that leaves text ink outside them, which is why region count
        // does not work as a proxy (a dense receipt collapsing to 2 boxes and a
        // clean page legitimately having 2 boxes look identical by count, and
        // completely different by coverage).
        // Default ON since 2026-08-07 (post pageseg round 3): the classical
        // route now wins or ties the dbnet route on every truthed
        // single-column fixture measured (5/6 synth, Fraktur 0.218 vs 0.235,
        // simple_form by inspection; dbnet's only truthed win is lowdpi by
        // +0.008) at ~3.5x less stage time, and multi-column pages still
        // fall back to the detector structurally (columns > 1).
        // CRISPEMBED_TESSERACT_SEG_ROUTER=0 restores the dbnet-first
        // default. Value-parsed (the UOCR =0 lesson).
        const char * router_env = std::getenv("CRISPEMBED_TESSERACT_SEG_ROUTER");
        const bool seg_router = router_env ? (*router_env && std::strcmp(router_env, "0") != 0) : true;
        if (seg_router) classical_pageseg = true;
        // The router routes; it does not veto. When the caller asked for
        // classical explicitly, the column fallback below is suppressed --
        // otherwise the request is advisory, which is not what the flag says.
        // The degenerate empty-boxes fallback is NOT suppressed: that is a
        // failure guard, not a routing preference.
        const bool router_may_reroute = seg_router && !explicit_classical;
        bool router_fell_back = false;
        double router_coverage = -1.0;
        int router_columns = -1;
        bool ppdet_line_boxes = false;
        if (ctx->tess_ppdet) {
            // PP-OCRv6 detection replaces both the classical segmenter and
            // DBNet: the caller chose this detector explicitly, so the
            // segmentation router does not reroute it. Its boxes are already
            // line-level, so the DBNet fragment grouping is skipped too.
            // Threshold mirrors the ppocrv6 stage (min with 0.2).
            const auto ppboxes =
                ppocrv6_det::detect_file(ctx->tess_ppdet, path, std::min(st.params.det_prob_threshold, 0.2f));
            boxes.reserve(ppboxes.size());
            for (const auto & b : ppboxes) {
                ocr_detect::text_box tb{};
                tb.x = b.x;
                tb.y = b.y;
                tb.w = b.w;
                tb.h = b.h;
                tb.score = b.score;
                for (int corner = 0; corner < 4; ++corner) {
                    tb.qx[corner] = b.qx[corner];
                    tb.qy[corner] = b.qy[corner];
                }
                boxes.push_back(tb);
            }
            classical_pageseg = false;
            ppdet_line_boxes = true;
            fprintf(stderr, "[tesseract-det] path=ppocrv6 boxes=%zu\n", boxes.size());
        } else if (classical_pageseg) {
            std::vector<uint8_t> seg_gray;
            int sw = 0, sh = 0;
            if (load_gray_exact(path, seg_gray, &sw, &sh)) {
                if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_PROJECTION") ||
                    std::getenv("CRISPEMBED_TESSERACT_COMPONENT_PAGESEG")) {
                    // segment_gray() owns the projection path and the
                    // explicitly opt-in component prototype.
                    boxes = tesseract_pageseg::segment_gray(seg_gray.data(), sw, sh);
                } else {
                    // Legacy component grouping remains the default classical
                    // adapter and is separately gated from DBNet.
                    boxes = tesseract_pageseg::segment_gray_components(seg_gray.data(), sw, sh);
                }
                if (seg_router || std::getenv("CRISPEMBED_TESSERACT_SEG_COVERAGE")) {
                    router_coverage = pageseg_ink_coverage(seg_gray.data(), sw, sh, boxes);
                    router_columns = pageseg_column_count(seg_gray.data(), sw, sh);
                }
                if (seg_router) {
                    // Route on COLUMN COUNT, not on ink coverage. Coverage was
                    // implemented first and measured wrong: commons_test_ocr_document
                    // scores 1.0000 coverage while losing 91% of its text, because
                    // the boxes are at paragraph granularity rather than misplaced.
                    // Median box height overlapped outright. Column count is the
                    // structural property that actually distinguishes the cases --
                    // a horizontal row projection cannot separate side-by-side
                    // columns, so multi-column pages must go to the detector.
                    if (boxes.empty() || (router_may_reroute && router_columns > 1)) {
                        boxes = ocr_detect::detect_file_ex(ctx->tess_det, path, geometry);
                        classical_pageseg = false; // DBNet boxes need line grouping
                        router_fell_back = true;
                    }
                }
            } else if (seg_router) {
                boxes = ocr_detect::detect_file_ex(ctx->tess_det, path, geometry);
                classical_pageseg = false;
                router_fell_back = true;
            }
        } else {
            boxes = ocr_detect::detect_file_ex(ctx->tess_det, path, geometry);
        }
        if (router_coverage >= 0.0 && (seg_router || std::getenv("CRISPEMBED_TESSERACT_SEG_COVERAGE") || ctx->bench)) {
            fprintf(stderr, "[tesseract-seg-router] columns=%d ink_coverage=%.4f boxes=%zu path=%s\n", router_columns,
                    router_coverage, boxes.size(), router_fell_back ? "dbnet(fallback)" : "classical");
        }
        const auto tess_detect_done = std::chrono::steady_clock::now();
        if (boxes.empty()) return {};
        // The IC15 DBNet artifact returns fragmented word-like regions on
        // historical pages. Tesseract-LSTM is a line recognizer, so preserve
        // the detector boxes for geometry but merge same-baseline fragments
        // into complete line crops before recognition.
        std::vector<easyocr_layout::region> detected_regions;
        detected_regions.reserve(boxes.size());
        for (const auto & box : boxes) detected_regions.push_back({ box.x, box.y, box.w, box.h, box.score });
        const auto line_regions = (classical_pageseg || ppdet_line_boxes)
                                      ? detected_regions
                                      : easyocr_layout::group_dbnet_lines(detected_regions);
        const bool pageseg_debug = classical_pageseg && std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG");
        // Opt-in recognition-confidence floor (pageseg round 5): ornament,
        // seal, and noise-band crops decode as garbage at mean char
        // confidence 0.23-0.47 while real lines sit >= 0.70 on the Fraktur
        // fixture. Reject regions the recognizer itself disbelieves. Only
        // applies when the model exposes a real per-char confidence buffer;
        // unset or <= 0 disables (default).
        float min_rec_conf = 0.0f;
        if (const char * env = std::getenv("CRISPEMBED_TESSERACT_MIN_REC_CONFIDENCE"))
            min_rec_conf = strtof(env, nullptr);
        if (pageseg_debug) {
            fprintf(stderr, "ocr_orchestrator: pageseg candidates=%zu\n", line_regions.size());
            for (size_t i = 0; i < line_regions.size(); ++i) {
                const auto & line = line_regions[i];
                fprintf(stderr, "  candidate=%zu box=%.1f,%.1f %.1fx%.1f\n", i, line.x, line.y, line.w, line.h);
            }
        }
        const auto tess_group_done = std::chrono::steady_clock::now();
        int w = 0, h = 0;
        std::vector<uint8_t> gray;
        if (!load_gray_exact(path, gray, &w, &h)) return {};
        struct line_crop {
            ocr_detect::text_box box;
            std::vector<uint8_t> pixels;
            int width = 0;
            int height = 0;
            ocr_crop::orientation_info orientation{};
        };
        std::vector<line_crop> crops;
        crops.reserve(line_regions.size());
        int pad = 2;
        if (const char * pad_env = std::getenv("CRISPEMBED_TESSERACT_CROP_PAD")) {
            // Keep the production default unchanged while allowing controlled
            // parity experiments: Tesseract's line box often includes a small
            // amount of surrounding paper, but historical scans can need a
            // tighter or wider border.
            pad = std::clamp(std::atoi(pad_env), 0, 32);
        }
        for (const auto & line : line_regions) {
            ocr_detect::text_box b{};
            b.x = line.x;
            b.y = line.y;
            b.w = line.w;
            b.h = line.h;
            b.score = line.score;
            int cw = 0, chh = 0;
            auto crop = ocr_crop::extract(gray.data(), w, h, 1, (int)b.x, (int)b.y, (int)b.w, (int)b.h, pad, &cw, &chh);
            if (crop.empty()) continue;
            if (std::getenv("CRISPEMBED_TESSERACT_CROP_TRIM_INK")) {
                int threshold = 128;
                if (const char * threshold_env = std::getenv("CRISPEMBED_TESSERACT_CROP_TRIM_THRESHOLD"))
                    threshold = std::clamp(std::atoi(threshold_env), 1, 254);
                int first_ink = chh, last_ink = -1;
                for (int row = 0; row < chh; ++row) {
                    bool ink = false;
                    for (int col = 0; col < cw; ++col) {
                        if (crop[(size_t)row * cw + col] < threshold) {
                            ink = true;
                            break;
                        }
                    }
                    if (ink) {
                        first_ink = std::min(first_ink, row);
                        last_ink = row;
                    }
                }
                if (last_ink >= first_ink) {
                    first_ink = std::max(0, first_ink - 1);
                    last_ink = std::min(chh - 1, last_ink + 1);
                    const int trimmed_height = last_ink - first_ink + 1;
                    if (first_ink > 0 || trimmed_height < chh) {
                        std::vector<uint8_t> trimmed((size_t)cw * trimmed_height);
                        std::memcpy(trimmed.data(), crop.data() + (size_t)first_ink * cw, (size_t)cw * trimmed_height);
                        crop.swap(trimmed);
                        chh = trimmed_height;
                    }
                }
            }
            if (const char * dump_dir = std::getenv("CRISPEMBED_TESSERACT_CROP_DUMP_DIR")) {
                char crop_path[1024];
                std::snprintf(crop_path, sizeof(crop_path), "%s/crop-%02zu.png", dump_dir, crops.size());
                if (stbi_write_png(crop_path, cw, chh, 1, crop.data(), cw) == 0)
                    std::fprintf(stderr, "ocr_orchestrator: failed to dump crop %s\n", crop_path);
                char manifest_path[1024];
                std::snprintf(manifest_path, sizeof(manifest_path), "%s/crops.tsv", dump_dir);
                FILE * manifest = std::fopen(manifest_path, crops.empty() ? "w" : "a");
                if (manifest) {
                    if (crops.empty())
                        std::fprintf(manifest, "index\tbox_x\tbox_y\tbox_w\tbox_h\tcrop_w\tcrop_h\tmin\tmax\n");
                    uint8_t min_value = 255, max_value = 0;
                    for (const uint8_t value : crop) {
                        min_value = std::min(min_value, value);
                        max_value = std::max(max_value, value);
                    }
                    std::fprintf(manifest, "%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\t%u\t%u\n", crops.size(), b.x, b.y, b.w,
                                 b.h, cw, chh, (unsigned)min_value, (unsigned)max_value);
                    std::fclose(manifest);
                } else {
                    std::fprintf(stderr, "ocr_orchestrator: failed to open crop manifest %s\n", manifest_path);
                }
            }
            const auto orientation = ocr_crop::orient_180_gray_info(crop, cw, chh);
            crops.push_back({ b, std::move(crop), cw, chh, orientation });
        }
        if (crops.empty()) return {};
        const auto tess_crop_done = std::chrono::steady_clock::now();

        // The recognizer context is stateful (captures and confidences), so
        // use one independent context per worker. Detection and crop order
        // remain deterministic; only independent line inference is parallel.
        int worker_count = 1;
        if (const char * env = std::getenv("CRISPEMBED_TESSERACT_WORKERS")) worker_count = std::max(1, std::atoi(env));
        worker_count = std::min(worker_count, (int)crops.size());
        const int missing_workers = worker_count - 1 - (int)ctx->tess_workers.size();
        const bool parallel_load = std::getenv("CRISPEMBED_TESSERACT_PARALLEL_LOAD") != nullptr;
        if (parallel_load && missing_workers > 0) {
            std::vector<std::future<tesseract_lstm_context *>> loads;
            loads.reserve(missing_workers);
            for (int i = 0; i < missing_workers; ++i) {
                loads.push_back(
                    std::async(std::launch::async, [model = ctx->tess_resolved_model, threads = ctx->n_threads]() {
                        return tesseract_lstm_init(model.c_str(), threads);
                    }));
            }
            for (auto & load : loads) {
                auto * worker = load.get();
                if (worker) ctx->tess_workers.push_back(worker);
            }
        } else {
            while ((int)ctx->tess_workers.size() < worker_count - 1) {
                auto * worker = tesseract_lstm_init(ctx->tess_resolved_model.c_str(), ctx->n_threads);
                if (!worker) break;
                ctx->tess_workers.push_back(worker);
            }
        }
        worker_count = std::min(worker_count, (int)ctx->tess_workers.size() + 1);
        std::vector<ocr_pipeline::ocr_result> slots(crops.size());
        std::vector<char> valid(crops.size(), 0);
        auto recognize_worker = [&](int worker_index) {
            tesseract_lstm_context * tess = worker_index == 0 ? ctx->tess : ctx->tess_workers[worker_index - 1];
            for (size_t i = (size_t)worker_index; i < crops.size(); i += (size_t)worker_count) {
                auto & item = crops[i];
                const auto orientation = item.orientation;
                int len = 0;
                const char * t = tesseract_lstm_recognize(tess, item.pixels.data(), item.width, item.height, &len);
                if (pageseg_debug) {
                    fprintf(stderr, "  candidate=%zu crop=%dx%d decoded_len=%d text=%.*s\n", i, item.width, item.height,
                            len, len > 0 ? len : 0, t ? t : "");
                }
                if (!t || len <= 0) continue;
                auto & r = slots[i];
                r.box = item.box;
                r.orientation_corrected = orientation.corrected;
                r.orientation_angle = orientation.angle;
                r.orientation_confidence = orientation.confidence;
                int n_conf = 0;
                const float * conf = tesseract_lstm_confidences(tess, &n_conf);
                float mean = item.box.score;
                if (conf && n_conf > 0) {
                    double s = 0.0;
                    for (int k = 0; k < n_conf; k++) s += conf[k];
                    mean = (float)(s / n_conf);
                    r.char_conf.assign(conf, conf + n_conf);
                }
                // Some converted Tesseract models expose a confidence buffer
                // whose values are all zero even though recognition succeeded.
                // Preserve the detector/segmentation score in that case rather
                // than reporting a misleading page confidence of exactly zero.
                if (mean <= 0.0f) mean = item.box.score;
                if (conf == nullptr || n_conf == 0) {
                    const float sequence_mean = tesseract_lstm_mean_confidence(tess);
                    if (sequence_mean > 0.0f) mean = sequence_mean;
                }
                r.confidence = mean;
                r.rec_confidence = mean;
                if (pageseg_debug) {
                    float min_conf = 1.0f;
                    for (const float c : r.char_conf) min_conf = std::min(min_conf, c);
                    fprintf(stderr, "  candidate_conf=%zu mean=%.4f min=%.4f word=%.4f n=%d\n", i, mean,
                            r.char_conf.empty() ? -1.0f : min_conf, tesseract_lstm_word_confidence(tess),
                            (int)r.char_conf.size());
                }
                if (min_rec_conf > 0.0f && !r.char_conf.empty() && mean < min_rec_conf) {
                    if (pageseg_debug)
                        fprintf(stderr, "  candidate_rejected=%zu mean=%.4f floor=%.4f\n", i, mean, min_rec_conf);
                    continue;
                }
                r.text = std::string(t, len);
                valid[i] = !r.text.empty();
            }
        };
        std::vector<std::future<void>> jobs;
        for (int i = 1; i < worker_count; ++i) jobs.push_back(std::async(std::launch::async, recognize_worker, i));
        recognize_worker(0);
        for (auto & job : jobs) job.get();
        std::vector<ocr_pipeline::ocr_result> results;
        results.reserve(crops.size());
        for (size_t i = 0; i < slots.size(); ++i)
            if (valid[i]) results.push_back(std::move(slots[i]));
        if (ctx->bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr,
                    "[tesseract-stage-bench] detect=%.1f ms group=%.1f ms crop=%.1f ms recognize=%.1f ms total=%.1f ms "
                    "boxes=%zu lines=%zu\n",
                    ms(tess_bench_start, tess_detect_done), ms(tess_detect_done, tess_group_done),
                    ms(tess_group_done, tess_crop_done), ms(tess_crop_done, std::chrono::steady_clock::now()),
                    ms(tess_bench_start, std::chrono::steady_clock::now()), boxes.size(), line_regions.size());
        }
        return results;
    }
    case engine::parseq: {
        // DBNet detection (model_a) + per-region PARSeq recognition
        // (model_b). PARSeq is a scene-text recognizer for a single cropped
        // line/word, so each detected region is cropped (RGB) and recognized
        // in turn. PARSeq exposes per-character confidence (1:1 with chars).
        if (!ctx->parseq_det) {
            if (st.model_a.empty() || st.model_b.empty()) {
                fprintf(stderr, "ocr_orchestrator: parseq stage missing models "
                                "(model_a=det, model_b=parseq)\n");
                return {};
            }
            if (!ocr_detect::load(&ctx->parseq_det, st.model_a.c_str(), ctx->n_threads)) {
                fprintf(stderr, "ocr_orchestrator: parseq detection load failed\n");
                ctx->parseq_det = nullptr;
                return {};
            }
        }
        if (!ctx->parseq) {
            ctx->parseq = parseq_ocr_init(st.model_b.c_str(), ctx->n_threads);
            if (!ctx->parseq) {
                fprintf(stderr, "ocr_orchestrator: parseq load failed: %s\n", st.model_b.c_str());
                return {};
            }
        }
        auto boxes = ocr_detect::detect_file_ex(ctx->parseq_det, path, geometry);
        if (boxes.empty()) return {};
        int w = 0, h = 0, c = 0;
        unsigned char * rgb = stbi_load(path, &w, &h, &c, 3); // force RGB
        if (!rgb) return {};
        std::vector<ocr_pipeline::ocr_result> results;
        results.reserve(boxes.size());
        const int pad = 2;
        for (auto & b : boxes) {
            int cw = 0, chh = 0;
            auto crop = ocr_crop::extract(rgb, w, h, 3, (int)b.x, (int)b.y, (int)b.w, (int)b.h, pad, &cw, &chh);
            if (crop.empty()) continue;
            const auto orientation = ocr_crop::orient_180_rgb_info(crop, cw, chh);
            int len = 0;
            const char * t = parseq_ocr_recognize_raw(ctx->parseq, crop.data(), cw, chh, 3, &len);
            if (!t || len <= 0) continue;
            ocr_pipeline::ocr_result r;
            r.box = b;
            r.orientation_corrected = orientation.corrected;
            r.orientation_angle = orientation.angle;
            r.orientation_confidence = orientation.confidence;
            int n_conf = 0;
            const float * conf = parseq_ocr_confidences(ctx->parseq, &n_conf);
            float mean = parseq_ocr_mean_confidence(ctx->parseq);
            if (mean <= 0.0f) mean = b.score;
            if (conf && n_conf > 0) r.char_conf.assign(conf, conf + n_conf);
            r.confidence = mean;
            r.rec_confidence = mean;
            r.text = std::string(t, len);
            if (!r.text.empty()) results.push_back(std::move(r));
        }
        stbi_image_free(rgb);
        return results;
    }
    case engine::pix2struct: {
        if (!ctx->p2s) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: pix2struct stage missing model_a\n");
                return {};
            }
            ctx->p2s = pix2struct_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->p2s) {
                fprintf(stderr, "ocr_orchestrator: pix2struct load failed\n");
                return {};
            }
        }
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int max_tok = st.params.vlm_max_tokens > 0 ? st.params.vlm_max_tokens : 2048;
        const char * t = pix2struct_generate(ctx->p2s, img, w, h, max_tok);
        int nconf = 0;
        const float * conf = pix2struct_confidences(ctx->p2s, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, pix2struct_mean_confidence(ctx->p2s));
        if (t) pix2struct_free_text(t);
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::granite_vision: {
        if (!ctx->gv) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: granite_vision stage missing model_a\n");
                return {};
            }
            ctx->gv = granite_vision_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->gv) {
                fprintf(stderr, "ocr_orchestrator: granite_vision load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) granite_vision_set_max_tokens(ctx->gv, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * prompt = st.params.vlm_prompt.empty() ? nullptr : st.params.vlm_prompt.c_str();
        const char * t = granite_vision_recognize(ctx->gv, img, w, h, 3, prompt, &len);
        int nconf = 0;
        const float * conf = granite_vision_confidences(ctx->gv, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, granite_vision_mean_confidence(ctx->gv));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::lightonocr: {
        if (!ctx->locr) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: lightonocr stage missing model_a\n");
                return {};
            }
            ctx->locr = lightonocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->locr) {
                fprintf(stderr, "ocr_orchestrator: lightonocr load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) lightonocr_set_max_tokens(ctx->locr, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = lightonocr_recognize_raw(ctx->locr, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = lightonocr_confidences(ctx->locr, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, lightonocr_mean_confidence(ctx->locr));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::lfm2_vl: {
        if (!ctx->lfm2vl) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: lfm2_vl stage missing model_a\n");
                return {};
            }
            // lfm2_vl_ocr_init finds the companion mmproj by scanning the
            // model's own directory — a VL model here is two files, and the
            // registry downloads both.
            ctx->lfm2vl = lfm2_vl_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->lfm2vl) {
                fprintf(stderr, "ocr_orchestrator: lfm2_vl load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) lfm2_vl_ocr_set_max_tokens(ctx->lfm2vl, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = lfm2_vl_ocr_recognize_raw(ctx->lfm2vl, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = lfm2_vl_ocr_confidences(ctx->lfm2vl, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, lfm2_vl_ocr_mean_confidence(ctx->lfm2vl));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::unlimited_ocr: {
        if (!ctx->uocr) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: unlimited_ocr stage missing model_a\n");
                return {};
            }
            ctx->uocr = unlimited_ocr_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->uocr) {
                fprintf(stderr, "ocr_orchestrator: unlimited_ocr load failed\n");
                return {};
            }
        }
        if (st.params.vlm_max_tokens > 0) unlimited_ocr_set_max_tokens(ctx->uocr, st.params.vlm_max_tokens);
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = unlimited_ocr_recognize_raw(ctx->uocr, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = unlimited_ocr_confidences(ctx->uocr, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, unlimited_ocr_mean_confidence(ctx->uocr));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    case engine::unified: {
        if (!ctx->unified) {
            if (st.model_a.empty()) {
                fprintf(stderr, "ocr_orchestrator: unified OCR stage missing model_a\n");
                return {};
            }
            ctx->unified = crispembed_ocr_model_init(st.model_a.c_str(), ctx->n_threads);
            if (!ctx->unified) {
                fprintf(stderr, "ocr_orchestrator: unified OCR model load failed: %s\n", st.model_a.c_str());
                return {};
            }
        }
        int w = pw, h = ph;
        unsigned char * loaded = nullptr;
        const unsigned char * img = px;
        if (!img) {
            int c = 0;
            loaded = stbi_load(path, &w, &h, &c, 3);
            img = loaded;
        }
        if (!img) return {};
        int len = 0;
        const char * t = crispembed_ocr_model_recognize(ctx->unified, img, w, h, 3, &len);
        int nconf = 0;
        const float * conf = crispembed_ocr_model_confidences(ctx->unified, &nconf);
        auto out = wrap_fulltext(t, w, h, conf, nconf, crispembed_ocr_model_mean_confidence(ctx->unified));
        if (loaded) stbi_image_free(loaded);
        return out;
    }
    default:
        fprintf(stderr, "ocr_orchestrator: engine %d not wired\n", (int)st.eng);
        return {};
    }
}

static std::vector<layout_detect::region> detect_layout(context * ctx, const char * path) {
    if (!ctx || ctx->cfg.layout_model.empty() || !path) return {};
    if (!ctx->layout) {
        if (!layout_detect::load(&ctx->layout, ctx->cfg.layout_model.c_str(), ctx->n_threads)) {
            fprintf(stderr, "ocr_orchestrator: layout load failed: %s\n", ctx->cfg.layout_model.c_str());
            ctx->layout = nullptr;
            return {};
        }
    }
    return layout_detect::detect_file(ctx->layout, path);
}

static bool crop_region_rgb(const unsigned char * pixels, int width, int height, const layout_detect::region & region,
                            std::vector<uint8_t> & out, int * out_w, int * out_h) {
    if (!pixels || width <= 0 || height <= 0) return false;
    const int x0 = std::max(0, std::min(width - 1, (int)std::floor(region.x1)));
    const int y0 = std::max(0, std::min(height - 1, (int)std::floor(region.y1)));
    const int x1 = std::max(x0 + 1, std::min(width, (int)std::ceil(region.x2)));
    const int y1 = std::max(y0 + 1, std::min(height, (int)std::ceil(region.y2)));
    const int w = x1 - x0, h = y1 - y0;
    out.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        std::memcpy(out.data() + (size_t)y * w * 3, pixels + ((size_t)(y0 + y) * width + x0) * 3, (size_t)w * 3);
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return true;
}

static void run_specialized(context * ctx, result & out, const unsigned char * pixels, int width, int height) {
    if (!ctx || !pixels) return;

    if (ctx->cfg.route_tables && !ctx->cfg.table_model.empty() && !out.routing.table_layout_indices.empty()) {
        if (!ctx->table) ctx->table = table_parse_init(ctx->cfg.table_model.c_str(), ctx->n_threads);
        if (ctx->table) {
            std::vector<uint8_t> rgb;
            for (int li : out.routing.table_layout_indices) {
                if (li < 0 || li >= (int)out.layout.size()) continue;
                int w = 0, h = 0;
                if (!crop_region_rgb(pixels, width, height, out.layout[li], rgb, &w, &h)) continue;
                std::vector<uint8_t> gray((size_t)w * h);
                for (int i = 0; i < w * h; ++i)
                    gray[i] = (uint8_t)((299 * rgb[i * 3] + 587 * rgb[i * 3 + 1] + 114 * rgb[i * 3 + 2] + 500) / 1000);
                char * html = table_parse_to_html(ctx->table, gray.data(), w, h);
                if (!html) continue;
                result::table_output item;
                item.layout_index = li;
                item.confidence = out.layout[li].score;
                item.x1 = out.layout[li].x1;
                item.y1 = out.layout[li].y1;
                item.x2 = out.layout[li].x2;
                item.y2 = out.layout[li].y2;
                item.html = html;
                out.tables.push_back(std::move(item));
                table_parse_free_string(html);
            }
        }
    }

    if (ctx->cfg.route_formulas && !ctx->cfg.formula_model.empty() && !out.routing.formula_layout_indices.empty()) {
        if (!ctx->formula) ctx->formula = ppformulanet_ocr_init(ctx->cfg.formula_model.c_str(), ctx->n_threads);
        if (!ctx->formula && !ctx->formula_l)
            ctx->formula_l = ppformulanet_l_ocr_init(ctx->cfg.formula_model.c_str(), ctx->n_threads);
        if (ctx->formula || ctx->formula_l) {
            std::vector<uint8_t> crop;
            for (int li : out.routing.formula_layout_indices) {
                if (li < 0 || li >= (int)out.layout.size()) continue;
                int w = 0, h = 0;
                if (!crop_region_rgb(pixels, width, height, out.layout[li], crop, &w, &h)) continue;
                int len = 0;
                const char * latex = ctx->formula
                                         ? ppformulanet_ocr_recognize_raw(ctx->formula, crop.data(), w, h, 3, &len)
                                         : ppformulanet_l_ocr_recognize_raw(ctx->formula_l, crop.data(), w, h, 3, &len);
                if (!latex || len <= 0) continue;
                result::formula_output item;
                item.layout_index = li;
                item.confidence = ctx->formula ? ppformulanet_ocr_mean_confidence(ctx->formula)
                                               : ppformulanet_l_ocr_mean_confidence(ctx->formula_l);
                item.x1 = out.layout[li].x1;
                item.y1 = out.layout[li].y1;
                item.x2 = out.layout[li].x2;
                item.y2 = out.layout[li].y2;
                item.latex.assign(latex, len);
                out.formulas.push_back(std::move(item));
            }
        }
    }
}

static result assemble(std::vector<ocr_pipeline::ocr_result> regions, engine eng, source_type st) {
    double conf_sum = 0.0;
    result r;
    r.used_engine = eng;
    r.used_type = st;
    r.regions = std::move(regions);
    r.reading_order.resize(r.regions.size());
    std::iota(r.reading_order.begin(), r.reading_order.end(), 0);
    std::stable_sort(r.reading_order.begin(), r.reading_order.end(), [&](int a, int b) {
        const auto & x = r.regions[(size_t)a].box;
        const auto & y = r.regions[(size_t)b].box;
        const float row = std::max(8.0f, std::min(x.h, y.h) * 0.65f);
        if (std::fabs(x.y - y.y) > row) return x.y < y.y;
        return x.x < y.x;
    });
    std::string joined;
    for (int index : r.reading_order) {
        auto & reg = r.regions[(size_t)index];
        if (!joined.empty()) joined += "\n";
        joined += reg.text;
        conf_sum += reg.confidence;
    }
    r.mean_confidence = r.regions.empty() ? 0.0f : (float)(conf_sum / (double)r.regions.size());
    r.full_text = std::move(joined);
    return r;
}

static void build_markdown(result & r) {
    r.markdown.clear();
    if (!r.layout.empty()) {
        std::vector<int> layout_order(r.layout.size());
        std::iota(layout_order.begin(), layout_order.end(), 0);
        std::stable_sort(layout_order.begin(), layout_order.end(), [&](int a, int b) {
            const auto & x = r.layout[(size_t)a];
            const auto & y = r.layout[(size_t)b];
            if (std::fabs(x.y1 - y.y1) > 8.0f) return x.y1 < y.y1;
            return x.x1 < y.x1;
        });
        std::vector<uint8_t> used(r.regions.size(), 0);
        for (int li : layout_order) {
            const auto & box = r.layout[(size_t)li];
            for (const auto & table : r.tables) {
                if (table.layout_index == li && !table.html.empty()) r.markdown += table.html + "\n\n";
            }
            for (const auto & formula : r.formulas) {
                if (formula.layout_index == li && !formula.latex.empty())
                    r.markdown += "$$\n" + formula.latex + "\n$$\n\n";
            }
            std::vector<int> inside;
            for (int ri : r.reading_order) {
                const auto & text = r.regions[(size_t)ri].box;
                const float cx = text.x + text.w * 0.5f;
                const float cy = text.y + text.h * 0.5f;
                if (cx >= box.x1 && cx <= box.x2 && cy >= box.y1 && cy <= box.y2) inside.push_back(ri);
            }
            for (int ri : inside) {
                if (used[(size_t)ri]) continue;
                used[(size_t)ri] = 1;
                const auto & text = r.regions[(size_t)ri].text;
                if (text.empty()) continue;
                if (box.label == layout_detect::label_id::title)
                    r.markdown += "# " + text + "\n\n";
                else if (box.label == layout_detect::label_id::section_header)
                    r.markdown += "## " + text + "\n\n";
                else
                    r.markdown += text + "\n\n";
            }
        }
        for (int ri : r.reading_order) {
            if (ri >= 0 && ri < (int)used.size() && !used[(size_t)ri] && !r.regions[(size_t)ri].text.empty())
                r.markdown += r.regions[(size_t)ri].text + "\n\n";
        }
        return;
    }
    for (int index : r.reading_order) {
        if (index < 0 || index >= (int)r.regions.size()) continue;
        const auto & region = r.regions[(size_t)index];
        if (region.text.empty()) continue;
        r.markdown += region.text;
        r.markdown += "\n\n";
    }
    for (const auto & table : r.tables) {
        if (!table.html.empty()) r.markdown += table.html + "\n\n";
    }
    for (const auto & formula : r.formulas) {
        if (!formula.latex.empty()) r.markdown += "$$\n" + formula.latex + "\n$$\n\n";
    }
}

static bool passes_gate(const result & r, const accept_gate & g) {
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
static std::string maybe_sr(context * ctx, const char * src) {
    if (ctx->cfg.sr_model.empty()) return "";

    int w = 0, h = 0, c = 0;
    unsigned char * d = stbi_load(src, &w, &h, &c, 3);
    if (!d) return "";

    int dpi = estimate_dpi(w, h);
    if (dpi >= ctx->cfg.sr_target_dpi) {
        stbi_image_free(d);
        return "";
    }

    // Lazy-load SR model (auto-detect engine from GGUF architecture)
    if (ctx->sr_kind == context::SR_NONE) {
        // Detect architecture
        gguf_context * meta = core_gguf::open_metadata(ctx->cfg.sr_model.c_str());
        std::string arch;
        if (meta) {
            arch = core_gguf::kv_str(meta, "general.architecture", "text_sr");
            core_gguf::free_metadata(meta);
        }

        if (arch == "pan") {
            ctx->pan = pan_sr_init(ctx->cfg.sr_model.c_str(), ctx->n_threads);
            if (ctx->pan)
                ctx->sr_kind = context::SR_PAN;
            else
                fprintf(stderr, "ocr_orchestrator: pan_sr load failed\n");
        } else {
            ctx->sr = text_sr_init(ctx->cfg.sr_model.c_str(), ctx->n_threads);
            if (ctx->sr)
                ctx->sr_kind = context::SR_NAFNET;
            else
                fprintf(stderr, "ocr_orchestrator: text_sr load failed\n");
        }

        if (ctx->sr_kind == context::SR_NONE) {
            stbi_image_free(d);
            return "";
        }
    }

    uint8_t * out = nullptr;
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
        if (ctx->sr_kind == context::SR_PAN)
            pan_sr_free_image(out);
        else
            text_sr_free_image(out);
        return "";
    }
    if (ctx->sr_kind == context::SR_PAN)
        pan_sr_free_image(out);
    else
        text_sr_free_image(out);

    if (ctx->cfg.verbose)
        fprintf(stderr, "ocr_orchestrator: SR(%s) %dx%d (%d DPI) -> %dx%d\n",
                ctx->sr_kind == context::SR_PAN ? "PAN" : "NAFNet", w, h, dpi, ow, oh);
    return out_path;
}

static const chain * pick_chain(const config & cfg, source_type st) {
    const chain * fallback = nullptr;
    for (auto & c : cfg.chains) {
        if (c.type == st) return &c;
        if (c.type == source_type::auto_detect) fallback = &c;
    }
    if (fallback) return fallback;
    return cfg.chains.empty() ? nullptr : &cfg.chains.front();
}

// ── public API ────────────────────────────────────────────────────────────────

bool load(context ** out, const config & cfg, int n_threads) {
    if (!out) return false;
    if (cfg.route_tables && (cfg.layout_model.empty() || cfg.table_model.empty())) {
        fprintf(stderr, "ocr_orchestrator: route_tables requires layout_model and table_model\n");
        return false;
    }
    if (cfg.route_formulas && (cfg.layout_model.empty() || cfg.formula_model.empty())) {
        fprintf(stderr, "ocr_orchestrator: route_formulas requires layout_model and formula_model\n");
        return false;
    }
    auto * ctx = new context();
    ctx->cfg = cfg;
    ctx->n_threads = n_threads;
    ctx->bench = core_env::on("CRISPEMBED_OCR_ORCH_BENCH");
#if CRISPEMBED_HAS_LID
    if (!cfg.lid_model.empty()) {
        ctx->lid = text_lid_init_from_file(cfg.lid_model.c_str(), n_threads);
        if (ctx->lid) {
            if (cfg.verbose)
                fprintf(stderr, "ocr_orchestrator: LID loaded (%s, %d labels)\n", text_lid_backend(ctx->lid),
                        text_lid_n_labels(ctx->lid));
        } else {
            fprintf(stderr, "ocr_orchestrator: WARNING: failed to load LID model: %s\n", cfg.lid_model.c_str());
        }
    }
#endif
#if CRISPEMBED_HAS_TRUECASE
    if (!cfg.truecase_model.empty()) {
        ctx->tc = truecaser_lstm_init(cfg.truecase_model.c_str());
        if (ctx->tc) {
            if (cfg.verbose) fprintf(stderr, "ocr_orchestrator: truecaser loaded\n");
        } else {
            fprintf(stderr, "ocr_orchestrator: WARNING: failed to load truecaser: %s\n", cfg.truecase_model.c_str());
        }
    }
#endif
    *out = ctx;
    return true;
}

capabilities get_capabilities(const context * ctx) {
    capabilities out;
    if (!ctx) return out;
    out.layout = !ctx->cfg.layout_model.empty();
    out.tables = ctx->cfg.route_tables && !ctx->cfg.table_model.empty();
    out.formulas = ctx->cfg.route_formulas && !ctx->cfg.formula_model.empty();
    out.image_text_fallback = ctx->cfg.image_text_fallback;
    return out;
}

static const char * engine_name(engine e) {
    switch (e) {
    case engine::dbnet_trocr:
        return "dbnet_trocr";
    case engine::ppocrv6:
        return "ppocrv6";
    case engine::surya:
        return "surya";
    case engine::got:
        return "got";
    case engine::glm:
        return "glm";
    case engine::qwen2vl:
        return "qwen2vl";
    case engine::qwen3vl:
        return "qwen3vl";
    case engine::internvl2:
        return "internvl2";
    case engine::tesseract:
        return "tesseract";
    case engine::tesseract_fraktur:
        return "tesseract_fraktur";
    case engine::deepseek_ocr2:
        return "deepseek_ocr2";
    case engine::pix2struct:
        return "pix2struct";
    case engine::granite_vision:
        return "granite_vision";
    case engine::lightonocr:
        return "lightonocr";
    case engine::lfm2_vl:
        return "lfm2_vl";
    case engine::unlimited_ocr:
        return "unlimited_ocr";
    case engine::unified:
        return "unified";
    case engine::easyocr:
        return "easyocr";
    case engine::olmocr:
        return "olmocr";
    default:
        return "unknown";
    }
}

static const char * source_type_name(source_type t) {
    switch (t) {
    case source_type::auto_detect:
        return "auto";
    case source_type::screenshot:
        return "screenshot";
    case source_type::scanned_doc:
        return "scanned_doc";
    case source_type::photo:
        return "photo";
    default:
        return "unknown";
    }
}

static bool is_vlm_engine(engine e) {
    switch (e) {
    case engine::olmocr:
    case engine::qwen2vl:
    case engine::qwen3vl:
    case engine::got:
    case engine::glm:
    case engine::internvl2:
    case engine::deepseek_ocr2:
    case engine::pix2struct:
    case engine::granite_vision:
    case engine::lightonocr:
    case engine::unlimited_ocr:
    case engine::lfm2_vl:
        return true;
    default:
        return false;
    }
}

// Apply optional post-processing (truecasing) to OCR result.
static void postprocess(context * ctx, result & r) {
#if CRISPEMBED_HAS_TRUECASE
    if (ctx->tc && !r.full_text.empty()) {
        char * tc_text = truecaser_lstm_process(ctx->tc, r.full_text.c_str());
        if (tc_text && strcmp(tc_text, r.full_text.c_str()) != 0) {
            r.full_text = tc_text;
            if (ctx->cfg.verbose) fprintf(stderr, "ocr_orchestrator: truecaser applied\n");
        }
        if (tc_text) ::free(tc_text);
    }
#else
    (void)ctx;
    (void)r;
#endif
}

result run_file(context * ctx, const char * image_path) {
    result best;
    if (!ctx || !image_path) return best;
    bool verbose = ctx->cfg.verbose;
    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Classify source type
    auto t_classify = std::chrono::steady_clock::now();
    const source_type st = ctx->cfg.router ? classify_file(image_path) : source_type::auto_detect;
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_classify).count();
        fprintf(stderr, "[ocr_orch-bench] classify: %.1f ms (%s)\n", ms, source_type_name(st));
    }
    if (verbose) fprintf(stderr, "ocr_orchestrator: source_type=%s for %s\n", source_type_name(st), image_path);

    const chain * ch = pick_chain(ctx->cfg, st);
    if (!ch) {
        if (verbose) fprintf(stderr, "ocr_orchestrator: no chain for source_type=%s\n", source_type_name(st));
        return best;
    }

    // Text super-resolution: upscale low-DPI images before OCR
    auto t_sr = std::chrono::steady_clock::now();
    std::string sr_path = maybe_sr(ctx, image_path);
    if (bench && !sr_path.empty()) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sr).count();
        fprintf(stderr, "[ocr_orch-bench] SR: %.1f ms\n", ms);
    }
    const char * effective_path = sr_path.empty() ? image_path : sr_path.c_str();

    int page_width = 0, page_height = 0, page_channels = 0;
    unsigned char * page_pixels = stbi_load(effective_path, &page_width, &page_height, &page_channels, 3);
    if (page_pixels) stbi_image_free(page_pixels);
    const auto layout_regions = detect_layout(ctx, effective_path);
    best.page_width = page_width;
    best.page_height = page_height;
    best.layout = layout_regions;

    int tried = 0;
    std::vector<result::stage_metric> stage_metrics;
    for (const stage & s : ch->stages) {
        if (!s.enabled) continue;
        tried++;

        if (verbose)
            fprintf(stderr, "ocr_orchestrator: stage %d engine=%s cleanup=%s\n", tried, engine_name(s.eng),
                    s.cleanup.enabled ? "on" : "off");

        auto t_stage = std::chrono::steady_clock::now();
        const bool raw_stage = s.eng == engine::dbnet_trocr || s.eng == engine::surya;
        cleanup_profile stage_cleanup = s.cleanup;
        if ((s.eng == engine::tesseract || s.eng == engine::tesseract_fraktur) &&
            core_env::explicitly_off("CRISPEMBED_TESSERACT_PAGESEG_CLEANUP")) {
            // UNBUNDLED 2026-08-07 (pageseg round 4). The skip used to be
            // implied by the segmentation choice: setting page_segmentation /
            // CRISPEMBED_TESSERACT_PAGESEG also turned cleanup off, on the
            // theory that classical row-ink measurement wants the original
            // page geometry. Two decisions, one flag.
            //
            // That coupling is what made the H9 router look like it had better
            // stage parameters. It does not: the router simply never sets the
            // PAGESEG flag, so it kept cleanup. Measured on the Fraktur page --
            // forced-classical WITH cleanup is BYTE-IDENTICAL to the router
            // (CER 0.1988, 22 regions, sha 832a55e89039) while forced-classical
            // without it scores 0.2214/21 regions. Same segmenter both arms;
            // the whole difference was preprocessing. Confirmed across 24
            // fixtures: forced-classical+cleanup == router on 24/24.
            //
            // So cleanup is now its own knob and defaults to the stage profile
            // for every tesseract stage, whichever segmenter runs.
            // CRISPEMBED_TESSERACT_PAGESEG_CLEANUP=0 restores the historical
            // skip -- worth having: on the CLEAN synthetic corpus the skip is
            // still better (mean CER 0.0126 vs 0.0189 over 20 renders, 10 of
            // them regressions), while every real scan measured prefers
            // cleanup (receipt_historical 20 -> 776 chars, commons_example_
            // receipt 264 -> 362, commons_test 1671 -> 2060, Fraktur
            // 0.2214 -> 0.1988). Rendered-text callers should set it to 0.
            if (verbose) fprintf(stderr, "ocr_orchestrator: Tesseract cleanup skipped (PAGESEG_CLEANUP=0)\n");
            stage_cleanup.enabled = false;
            stage_cleanup.denoise = false;
        }
        if (is_vlm_engine(s.eng) && stage_cleanup.enabled) {
            // VLMs perform their own letterboxing/resizing. Classical deskew,
            // binarization, or denoise can destroy the visual distribution the
            // vision encoder expects, so cleanup is opt-in only for VLM stages.
            if (verbose) fprintf(stderr, "ocr_orchestrator: VLM stage skips destructive cleanup\n");
            stage_cleanup.enabled = false;
            stage_cleanup.denoise = false;
        }
        if (s.eng == engine::ppocrv6 && stage_cleanup.enabled && std::getenv("CRISPEMBED_PPOCRV6_CLEANUP") == nullptr) {
            // The official PP-OCR pipeline runs its detector on the raw page;
            // classical cleanup before it is a deviation, and despeckle /
            // blackfilter run even when deskew/crop/whiten are disabled,
            // eroding thin strokes on clean rendered type ($ -> S, I -> :,
            // hyphens vanish). Measured on the labelled CC0 fixtures, CER
            // cleanup-on -> cleanup-off: commons_example_receipt 0.0885 ->
            // 0.0025, simple_form 0.7368 -> 0.6154, while the scans cleanup
            // was meant to help move only at noise level
            // (german_official_print 0.0486 -> 0.0535, receipt_historical
            // 0.0260 -> 0.0273). CRISPEMBED_PPOCRV6_CLEANUP=1 restores the
            // old behaviour.
            if (verbose) fprintf(stderr, "ocr_orchestrator: ppocrv6 stage skips destructive cleanup\n");
            stage_cleanup.enabled = false;
            stage_cleanup.denoise = false;
        }
        std::vector<uint8_t> cleaned_pixels;
        int cleaned_w = 0, cleaned_h = 0;
        const bool cleaned_in_memory =
            raw_stage && clean_to_pixels(ctx, stage_cleanup, effective_path, cleaned_pixels, &cleaned_w, &cleaned_h);
        std::string tmp;
        if (!cleaned_in_memory) tmp = clean_to_temp(ctx, stage_cleanup, effective_path);
        const char * ocr_path = tmp.empty() ? effective_path : tmp.c_str();

        // Pre-load image once for VLM engines (avoids redundant stbi_load
        // inside each engine case when multiple stages use the same image)
        int img_w = 0, img_h = 0, img_c = 0;
        unsigned char * loaded_px = nullptr;
        const unsigned char * img_px = nullptr;
        if (cleaned_in_memory) {
            img_px = cleaned_pixels.data();
            img_w = cleaned_w;
            img_h = cleaned_h;
            img_c = 3;
        } else {
            loaded_px = stbi_load(ocr_path, &img_w, &img_h, &img_c, 3);
            img_px = loaded_px;
        }
        result r = assemble(run_engine(ctx, s, ocr_path, img_px, img_w, img_h), s.eng, st);
        r.page_width = page_width;
        r.page_height = page_height;
        r.layout = layout_regions;
        std::vector<ocr_detect::text_box> text_boxes;
        text_boxes.reserve(r.regions.size());
        for (const auto & region : r.regions) text_boxes.push_back(region.box);
        ocr_region_router::request_policy route_policy;
        route_policy.want_tables = ctx->cfg.route_tables && !ctx->cfg.table_model.empty();
        route_policy.want_formulas = ctx->cfg.route_formulas && !ctx->cfg.formula_model.empty();
        route_policy.image_text_fallback = ctx->cfg.image_text_fallback;
        ocr_region_router::build(r.layout, text_boxes, route_policy, r.routing);
        run_specialized(ctx, r, img_px, img_w, img_h);
        build_markdown(r);
        if (loaded_px) stbi_image_free(loaded_px);
        r.used_type = st;
        r.stages_tried = tried;

        if (!tmp.empty()) std::remove(tmp.c_str());

        const float stage_ms =
            (float)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_stage).count();
        bool passed = passes_gate(r, s.accept);
        stage_metrics.push_back({ tried, engine_name(s.eng), stage_ms, stage_cleanup.enabled, passed,
                                  (int)r.full_text.size(), r.mean_confidence });
        r.stage_metrics = stage_metrics;
        if (bench) {
            fprintf(stderr, "[ocr_orch-bench] stage %d (%s): %.1f ms, gate=%s\n", tried, engine_name(s.eng), stage_ms,
                    passed ? "PASS" : "FAIL");
        }
        if (verbose)
            fprintf(stderr, "ocr_orchestrator: stage %d → %d chars, conf=%.2f, gate=%s\n", tried,
                    (int)r.full_text.size(), r.mean_confidence, passed ? "PASS" : "FAIL");

        if (passed) {
            if (!sr_path.empty()) std::remove(sr_path.c_str());
#if CRISPEMBED_HAS_LID
            // Run LID on the recognized text to detect language.
            auto t_lid = std::chrono::steady_clock::now();
            if (ctx->lid && !r.full_text.empty()) {
                float conf = 0.0f;
                const char * lang = text_lid_predict(ctx->lid, r.full_text.c_str(), &conf);
                if (lang && conf > 0.3f) {
                    r.detected_lang = lang;
                    r.lang_confidence = conf;
                    ctx->detected_lang = lang;
                    ctx->lang_confidence = conf;
                    if (verbose) fprintf(stderr, "ocr_orchestrator: LID → %s (%.2f)\n", lang, conf);
                }
                if (bench) {
                    double ms =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lid).count();
                    fprintf(stderr, "[ocr_orch-bench] LID: %.1f ms\n", ms);
                }
            }
#endif
            auto t_post = std::chrono::steady_clock::now();
            postprocess(ctx, r);
            if (bench) {
                double ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_post).count();
                double total_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
                fprintf(stderr, "[ocr_orch-bench] postprocess: %.1f ms\n", ms);
                fprintf(stderr, "[ocr_orch-bench] total: %.1f ms\n", total_ms);
            }
            return r;
        }
        if (r.full_text.size() > best.full_text.size()) best = std::move(r);
    }
    if (!sr_path.empty()) std::remove(sr_path.c_str());
    best.stage_metrics = std::move(stage_metrics);

    if (verbose)
        fprintf(stderr, "ocr_orchestrator: all %d stages failed gate, returning best (%d chars)\n", tried,
                (int)best.full_text.size());
    best.used_type = st;
    best.stages_tried = tried;
#if CRISPEMBED_HAS_LID
    if (ctx->lid && !best.full_text.empty()) {
        auto t_lid = std::chrono::steady_clock::now();
        float conf = 0.0f;
        const char * lang = text_lid_predict(ctx->lid, best.full_text.c_str(), &conf);
        if (lang && conf > 0.3f) {
            best.detected_lang = lang;
            best.lang_confidence = conf;
        }
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lid).count();
            fprintf(stderr, "[ocr_orch-bench] LID: %.1f ms\n", ms);
        }
    }
#endif
    auto t_post = std::chrono::steady_clock::now();
    postprocess(ctx, best);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_post).count();
        double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[ocr_orch-bench] postprocess: %.1f ms\n", ms);
        fprintf(stderr, "[ocr_orch-bench] total: %.1f ms\n", total_ms);
    }
    return best;
}

void free(context * ctx) {
    if (!ctx) return;
    if (ctx->dbnet) ocr_pipeline::free(ctx->dbnet);
    if (ctx->ppdet) ppocrv6_det::free(ctx->ppdet);
    if (ctx->tess_ppdet) ppocrv6_det::free(ctx->tess_ppdet);
    if (ctx->pprec) ppocrv6_ocr_free(ctx->pprec);
    if (ctx->ppori) pplcnet_orientation::free(ctx->ppori);
    if (ctx->easy) easyocr_pipeline::free(ctx->easy);
    if (ctx->layout) layout_detect::free(ctx->layout);
    if (ctx->table) table_parse_free(ctx->table);
    if (ctx->formula) ppformulanet_ocr_free(ctx->formula);
    if (ctx->formula_l) ppformulanet_l_ocr_free(ctx->formula_l);
    if (ctx->got) got_ocr_free(ctx->got);
    if (ctx->glm) glm_ocr_free(ctx->glm);
    if (ctx->qwen) qwen2vl_ocr_free(ctx->qwen);
    if (ctx->qwen3) qwen2vl_ocr_free(ctx->qwen3);
    if (ctx->intern) internvl2_ocr_free(ctx->intern);
    if (ctx->dsocr2) deepseek_ocr2_free(ctx->dsocr2);
    if (ctx->p2s) pix2struct_free(ctx->p2s);
    if (ctx->gv) granite_vision_free(ctx->gv);
    if (ctx->locr) lightonocr_free(ctx->locr);
    if (ctx->lfm2vl) lfm2_vl_ocr_free(ctx->lfm2vl);
    if (ctx->uocr) unlimited_ocr_free(ctx->uocr);
    if (ctx->unified) crispembed_ocr_model_free(ctx->unified);
    if (ctx->tess_det) ocr_detect::free(ctx->tess_det);
    if (ctx->tess) tesseract_lstm_free(ctx->tess);
    for (auto * worker : ctx->tess_workers) tesseract_lstm_free(worker);
    if (ctx->parseq_det) ocr_detect::free(ctx->parseq_det);
    if (ctx->parseq) parseq_ocr_free(ctx->parseq);
    if (ctx->clean1) scan_cleanup_free(ctx->clean1);
    if (ctx->clean2) scan_cleanup_free(ctx->clean2);
    if (ctx->sr) text_sr_free(ctx->sr);
    if (ctx->pan) pan_sr_free(ctx->pan);
#if CRISPEMBED_HAS_LID
    if (ctx->lid) text_lid_free(ctx->lid);
#endif
#if CRISPEMBED_HAS_TRUECASE
    if (ctx->tc) truecaser_lstm_free(ctx->tc);
#endif
    delete ctx;
}

} // namespace ocr_orchestrator
