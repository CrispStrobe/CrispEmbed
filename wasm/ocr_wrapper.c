// wasm/ocr_wrapper.c — Emscripten entry point for CrispEmbed OCR.
//
// Exposes the full OCR pipeline to the browser:
//   - Single-model recognition (TrOCR / pix2tex)
//   - Full pipeline (DBNet detection + TrOCR recognition + reading order)
//   - Advanced pipeline (with cleanup, routing, accept-gates)
//   - Scan cleanup (deskew, binarize, denoise — model-free classical tier)
//   - OCR rendering (text, hOCR, ALTO XML)
//   - Text detection standalone (DBNet / Surya)
//   - Layout detection (RT-DETRv2, 17 document classes)
//
// All "run" functions serialize results as JSON strings (malloc'd copies),
// so JS gets all fields in a single call without re-running inference.

#include "crispembed.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define WASM_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define WASM_EXPORT
#endif

// ── Helpers ──────────────────────────────────────────────────────────────

// Dynamic string buffer for JSON serialization.
typedef struct {
    char * data;
    size_t len;
    size_t cap;
} strbuf;

static void sb_init(strbuf * sb) { sb->data = NULL; sb->len = 0; sb->cap = 0; }

static void sb_ensure(strbuf * sb, size_t extra) {
    size_t need = sb->len + extra + 1;
    if (need <= sb->cap) return;
    size_t newcap = sb->cap ? sb->cap * 2 : 256;
    while (newcap < need) newcap *= 2;
    sb->data = (char *)realloc(sb->data, newcap);
    sb->cap = newcap;
}

static void sb_append(strbuf * sb, const char * s) {
    size_t slen = strlen(s);
    sb_ensure(sb, slen);
    memcpy(sb->data + sb->len, s, slen);
    sb->len += slen;
    sb->data[sb->len] = '\0';
}

// Append a JSON-escaped string value (handles \n, \r, \t, \, ").
static void sb_append_json_str(strbuf * sb, const char * s) {
    sb_append(sb, "\"");
    if (s) {
        for (const char * p = s; *p; ++p) {
            switch (*p) {
                case '"':  sb_append(sb, "\\\""); break;
                case '\\': sb_append(sb, "\\\\"); break;
                case '\n': sb_append(sb, "\\n"); break;
                case '\r': sb_append(sb, "\\r"); break;
                case '\t': sb_append(sb, "\\t"); break;
                default: {
                    sb_ensure(sb, 1);
                    sb->data[sb->len++] = *p;
                    sb->data[sb->len] = '\0';
                }
            }
        }
    }
    sb_append(sb, "\"");
}

static void sb_appendf(strbuf * sb, const char * fmt, ...) {
    char tmp[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sb_append(sb, tmp);
}

// ═══════════════════════════════════════════════════════════════════════════
// Version
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
const char * wasm_ocr_version(void) {
    return "crispembed-ocr-wasm-0.3.0";
}

// ═══════════════════════════════════════════════════════════════════════════
// Single-model recognition (TrOCR / pix2tex)
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_ocr_init(const char * model_path, int n_threads) {
    return crispembed_ocr_model_init(model_path, n_threads);
}

WASM_EXPORT
const char * wasm_ocr_recognize_gray(void * ctx, const float * pixels,
                                     int width, int height, int * out_len) {
    return crispembed_ocr_model_recognize_gray(ctx, pixels, width, height, out_len);
}

WASM_EXPORT
const char * wasm_ocr_recognize(void * ctx, const uint8_t * pixel_bytes,
                                int width, int height, int channels,
                                int * out_len) {
    return crispembed_ocr_model_recognize(ctx, pixel_bytes, width, height,
                                         channels, out_len);
}

WASM_EXPORT
char * wasm_ocr_recognize_copy(void * ctx, const uint8_t * pixel_bytes,
                               int width, int height, int channels,
                               int * out_len) {
    int len = 0;
    const char * result = crispembed_ocr_model_recognize(
        ctx, pixel_bytes, width, height, channels, &len);
    if (!result || len <= 0) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = len;
    char * copy = (char *)malloc(len + 1);
    if (!copy) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    memcpy(copy, result, len + 1);
    return copy;
}

WASM_EXPORT
const float * wasm_ocr_confidences(void * ctx, int * n_tokens) {
    return crispembed_ocr_model_confidences(ctx, n_tokens);
}

WASM_EXPORT
float wasm_ocr_mean_confidence(void * ctx) {
    return crispembed_ocr_model_mean_confidence(ctx);
}

WASM_EXPORT
void wasm_ocr_set_max_tokens(void * ctx, int max_tokens) {
    crispembed_ocr_model_set_max_tokens(ctx, max_tokens);
}

WASM_EXPORT
void wasm_ocr_free(void * ctx) {
    crispembed_ocr_model_free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// Full OCR pipeline (detection + recognition)
//
// Returns JSON: [{"x":..,"y":..,"w":..,"h":..,"confidence":..,"text":".."},...]
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_ocr_pipeline_init(const char * det_model_path,
                              const char * rec_model_path,
                              int n_threads) {
    return crispembed_ocr_init(det_model_path, rec_model_path, n_threads);
}

// Run pipeline on an image in MEMFS. Returns malloc'd JSON string.
// Caller must free() the returned string.
WASM_EXPORT
char * wasm_ocr_pipeline_run(void * ctx, const char * image_path) {
    int n = 0;
    const crispembed_ocr_result * results = crispembed_ocr(ctx, image_path, &n);

    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "[");
    for (int i = 0; i < n; ++i) {
        if (i > 0) sb_append(&sb, ",");
        sb_append(&sb, "{");
        sb_appendf(&sb, "\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f,\"confidence\":%.4f,\"text\":",
                   results[i].x, results[i].y, results[i].w, results[i].h,
                   results[i].confidence);
        sb_append_json_str(&sb, results[i].text);
        sb_append(&sb, "}");
    }
    sb_append(&sb, "]");
    return sb.data;  // caller frees
}

WASM_EXPORT
void wasm_ocr_pipeline_free(void * ctx) {
    crispembed_ocr_free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// Advanced pipeline (with cleanup, routing, accept-gates)
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_ocr_pipeline_full_init(const char * det_model,
                                   const char * rec_model,
                                   const char * nafnet_model,
                                   const char * sr_model,
                                   int cleanup_enabled,
                                   int router,
                                   int n_threads) {
    crispembed_ocr_pipeline_params p = crispembed_ocr_pipeline_defaults();
    p.det_model = det_model;
    p.rec_model = rec_model;
    p.nafnet_model = nafnet_model;
    p.sr_model = sr_model;
    p.cleanup_enabled = cleanup_enabled;
    p.router = router;
    return crispembed_ocr_pipeline_init(&p, n_threads);
}

// Returns malloc'd JSON: {"text":"...","n_regions":N,"confidence":0.XX,"regions":[...]}
WASM_EXPORT
char * wasm_ocr_pipeline_full_run(void * ctx, const char * image_path) {
    const char * full_text = NULL;
    int n_results = 0;
    float mean_conf = 0.0f;
    const crispembed_ocr_result * results = crispembed_ocr_pipeline_run(
        ctx, image_path, &n_results, &full_text, &mean_conf);

    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "{\"text\":");
    sb_append_json_str(&sb, full_text ? full_text : "");
    sb_appendf(&sb, ",\"n_regions\":%d,\"confidence\":%.4f,\"regions\":[", n_results, mean_conf);

    for (int i = 0; i < n_results; ++i) {
        if (i > 0) sb_append(&sb, ",");
        float rc = crispembed_ocr_pipeline_region_rec_confidence(ctx, i);
        sb_append(&sb, "{");
        sb_appendf(&sb, "\"x\":%.1f,\"y\":%.1f,\"w\":%.1f,\"h\":%.1f,\"confidence\":%.4f,\"text\":",
                   results[i].x, results[i].y, results[i].w, results[i].h, rc);
        sb_append_json_str(&sb, results[i].text);
        sb_append(&sb, "}");
    }
    sb_append(&sb, "]}");
    return sb.data;
}

WASM_EXPORT
void wasm_ocr_pipeline_full_free(void * ctx) {
    crispembed_ocr_pipeline_free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// Scan cleanup (classical preprocessing — no model needed)
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_scan_cleanup_init(const char * model_path, int n_threads) {
    return crispembed_scan_cleanup_init(model_path, n_threads);
}

// Process a scan image. Returns malloc'd RGB output pixels.
// Caller must free via wasm_scan_cleanup_free_image().
WASM_EXPORT
uint8_t * wasm_scan_cleanup_process(void * ctx, const uint8_t * pixels,
                                    int width, int height, int channels,
                                    int deskew, int crop_borders,
                                    int whiten_background, int binarize,
                                    int * out_width, int * out_height) {
    uint8_t * out_pixels = NULL;
    int ow = 0, oh = 0;
    int rc = crispembed_scan_cleanup_process_simple(
        ctx, pixels, width, height, channels,
        deskew, crop_borders, whiten_background, binarize,
        &out_pixels, &ow, &oh);
    if (rc != 0) return NULL;
    if (out_width) *out_width = ow;
    if (out_height) *out_height = oh;
    return out_pixels;
}

WASM_EXPORT
void wasm_scan_cleanup_free_image(uint8_t * pixels) {
    crispembed_scan_cleanup_free_image(pixels);
}

WASM_EXPORT
void wasm_scan_cleanup_free(void * ctx) {
    crispembed_scan_cleanup_free(ctx);
}

WASM_EXPORT
int wasm_scan_cleanup_detect_page_split(const uint8_t * pixels,
                                        int width, int height, int channels) {
    return crispembed_scan_cleanup_detect_page_split(pixels, width, height, channels);
}

WASM_EXPORT
int wasm_scan_cleanup_content_bbox(const uint8_t * pixels,
                                   int width, int height, int channels,
                                   int * x0, int * y0, int * x1, int * y1) {
    return crispembed_scan_cleanup_content_bbox(pixels, width, height, channels,
                                                x0, y0, x1, y1);
}

// ═══════════════════════════════════════════════════════════════════════════
// OCR result rendering (text / hOCR / ALTO XML)
// ═══════════════════════════════════════════════════════════════════════════

// Render the last pipeline results to the given format.
// format: "text", "hocr", "alto"
// Uses the basic pipeline (crispembed_ocr). Caller frees the returned string.
WASM_EXPORT
char * wasm_ocr_render(void * pipeline_ctx, const char * image_path,
                       int page_width, int page_height,
                       const char * format) {
    int n = 0;
    const crispembed_ocr_result * results = crispembed_ocr(pipeline_ctx, image_path, &n);
    if (!results || n <= 0) return NULL;
    return crispembed_ocr_render(results, n, page_width, page_height, format);
}

// ═══════════════════════════════════════════════════════════════════════════
// Text detection standalone (DBNet / Surya)
// Returns JSON: [{"x0":..,"y0":..,"x1":..,"y1":..,"confidence":..},...]
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_text_det_init(const char * model_path, int n_threads) {
    return crispembed_text_det_init(model_path, n_threads);
}

WASM_EXPORT
char * wasm_text_det_run(void * ctx, const uint8_t * pixels,
                         int width, int height, int channels,
                         float text_threshold, float low_threshold) {
    int n = 0;
    const crispembed_text_det_result * results = crispembed_text_det(
        ctx, pixels, width, height, channels,
        text_threshold, low_threshold, &n);

    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "[");
    for (int i = 0; i < n; ++i) {
        if (i > 0) sb_append(&sb, ",");
        sb_appendf(&sb, "{\"x0\":%.1f,\"y0\":%.1f,\"x1\":%.1f,\"y1\":%.1f,\"confidence\":%.4f}",
                   results[i].x0, results[i].y0, results[i].x1, results[i].y1,
                   results[i].confidence);
    }
    sb_append(&sb, "]");
    return sb.data;
}

WASM_EXPORT
void wasm_text_det_free(void * ctx) {
    crispembed_text_det_free(ctx);
}

// ═══════════════════════════════════════════════════════════════════════════
// Layout detection (RT-DETRv2, 17 document classes)
// Returns JSON: [{"x1":..,"y1":..,"x2":..,"y2":..,"score":..,"label":N,"label_name":".."},...]
// ═══════════════════════════════════════════════════════════════════════════

WASM_EXPORT
void * wasm_layout_init(const char * model_path, int n_threads) {
    return crispembed_layout_init(model_path, n_threads);
}

WASM_EXPORT
char * wasm_layout_detect(void * ctx, const char * image_path,
                          float score_threshold) {
    int n = 0;
    const crispembed_layout_region * regions = crispembed_layout_detect(
        ctx, image_path, score_threshold, &n);

    strbuf sb;
    sb_init(&sb);
    sb_append(&sb, "[");
    for (int i = 0; i < n; ++i) {
        if (i > 0) sb_append(&sb, ",");
        sb_append(&sb, "{");
        sb_appendf(&sb, "\"x1\":%.1f,\"y1\":%.1f,\"x2\":%.1f,\"y2\":%.1f,\"score\":%.4f,\"label\":%d,\"label_name\":",
                   regions[i].x1, regions[i].y1, regions[i].x2, regions[i].y2,
                   regions[i].score, regions[i].label);
        sb_append_json_str(&sb, regions[i].label_name);
        sb_append(&sb, "}");
    }
    sb_append(&sb, "]");
    return sb.data;
}

WASM_EXPORT
void wasm_layout_free(void * ctx) {
    crispembed_layout_free(ctx);
}

#ifdef __EMSCRIPTEN_PTHREADS__
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <emscripten/em_asm.h>
#include <pthread.h>

// PROXY_TO_PTHREAD path — multithreaded, deadlock-free recognize in the browser.
// The deadlock rule: the thread owning the event loop (the "servicer" worker)
// must not block in pthread_join, but ggml's compute threads do exactly that.
// Under -sPROXY_TO_PTHREAD, main() runs on a dedicated "runtime" pthread (kept
// alive below). We proxy the blocking pipeline call onto that thread — the
// servicer never blocks, so ggml's compute threads run fine — and deliver the
// JSON result back to the servicer via a JS callback (Module.__ocrDeliver),
// keyed by request id. Mirrors CrispASR's ttsSynthesizeAsync.
static em_proxying_queue * g_ocr_pq = NULL;
static pthread_t           g_ocr_runtime_thread;
static int                 g_ocr_runtime_ready = 0;

typedef struct {
    void *    ctx;
    char *    image_path;
    int       req_id;
    int       full;      // 1 = wasm_ocr_pipeline_full_run, 0 = wasm_ocr_pipeline_run
    char *    result;    // malloc'd JSON; ownership passed to JS (frees via Module._free)
    pthread_t servicer;  // thread to deliver the result on
} ocr_async_job;

static void ocr_async_deliver(void * arg) {  // runs on the servicer thread
    ocr_async_job * j = (ocr_async_job *) arg;
    EM_ASM({ if (Module['__ocrDeliver']) Module['__ocrDeliver']($0, $1); }, j->req_id, (int) j->result);
    if (j->image_path) free(j->image_path);
    free(j);
}

static void ocr_async_run(void * arg) {  // runs on the runtime thread (pthread-0)
    ocr_async_job * j = (ocr_async_job *) arg;
    j->result = j->full ? wasm_ocr_pipeline_full_run(j->ctx, j->image_path)
                        : wasm_ocr_pipeline_run(j->ctx, j->image_path);
    emscripten_proxy_async(g_ocr_pq, j->servicer, ocr_async_deliver, j);
}

// Fire-and-forget recognize for the multithreaded (PROXY_TO_PTHREAD) build.
// Returns immediately; JS receives (req_id, resultPtr) via Module.__ocrDeliver.
WASM_EXPORT
void wasm_ocr_pipeline_run_async(void * ctx, const char * image_path, int req_id, int full) {
    ocr_async_job * j = (ocr_async_job *) malloc(sizeof(ocr_async_job));
    j->ctx        = ctx;
    j->image_path = image_path ? strdup(image_path) : NULL;
    j->req_id     = req_id;
    j->full       = full;
    j->result     = NULL;
    j->servicer   = pthread_self();
    if (g_ocr_runtime_ready && g_ocr_pq) {
        emscripten_proxy_async(g_ocr_pq, g_ocr_runtime_thread, ocr_async_run, j);
    } else {
        // No proxied runtime thread (shouldn't happen once main() has run) —
        // run inline as a fallback and deliver synchronously.
        j->result = j->full ? wasm_ocr_pipeline_full_run(j->ctx, j->image_path)
                            : wasm_ocr_pipeline_run(j->ctx, j->image_path);
        EM_ASM({ if (Module['__ocrDeliver']) Module['__ocrDeliver']($0, $1); }, j->req_id, (int) j->result);
        if (j->image_path) free(j->image_path);
        free(j);
    }
}
#endif

// Emscripten requires a main() for executables. Under -sPROXY_TO_PTHREAD this
// runs on the dedicated runtime pthread; record it + keep the runtime alive so
// it can service the proxying queue (see wasm_ocr_pipeline_run_async above).
int main(void) {
#ifdef __EMSCRIPTEN_PTHREADS__
    g_ocr_pq             = em_proxying_queue_create();
    g_ocr_runtime_thread = pthread_self();
    g_ocr_runtime_ready  = 1;
    emscripten_exit_with_live_runtime();
#endif
    return 0;
}
