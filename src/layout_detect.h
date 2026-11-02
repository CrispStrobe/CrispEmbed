// layout_detect.h — Document layout analysis (ggml): RT-DETRv2 and D-FINE.
//
// Two architectures share one module, selected by GGUF metadata
// `layout.arch` (absent = "rt_detr_v2"):
//
//   * rt_detr_v2 — docling-layout-heron (Apache 2.0): ResNet-50 backbone,
//     hybrid AIFI/CCFM encoder, 6-layer decoder with deformable cross-attention,
//     17 canonical document classes, 640x640 input.
//   * dfine — OXR-1.0's layout detector (SoMarkAI/OXR-1.0, OXR License:
//     Apache-2.0 with additional terms): HGNetv2-B2 backbone, the same hybrid
//     encoder topology with RepNCSPELAN4/SCDown blocks, 4-layer D-FINE decoder
//     (gated deformable attention, fine-grained distribution refinement box
//     head, LQE), 12 OXR classes, 960x960 letterboxed input. Loading it
//     requires CRISPEMBED_OXR=1 and CRISPEMBED_ACCEPT_OXR_LICENSE=1.
//
// Usage:
//   layout_detect::context *ctx;
//   layout_detect::load(&ctx, "layout-heron.gguf");
//   auto regions = layout_detect::detect_file(ctx, "page.png");
//   for (auto& r : regions) {
//       printf("%s at (%.0f,%.0f)-(%.0f,%.0f) score=%.2f\n",
//              r.label_name, r.x1, r.y1, r.x2, r.y2, r.score);
//   }
//   layout_detect::free(ctx);

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace layout_detect {

// 17 canonical document layout classes (docling-layout-heron order).
enum class label_id : int {
    caption = 0,
    footnote,
    formula,
    list_item,
    page_footer,
    page_header,
    picture,
    section_header,
    table,
    text,
    title,
    document_index,
    code,
    checkbox_selected,
    checkbox_unselected,
    form,
    key_value_region,
    NUM_CLASSES
};

const char * label_name(label_id id);

enum class arch_id : int { rt_detr_v2 = 0, dfine = 1 };

struct region {
    float x1, y1, x2, y2; // bbox in original image coordinates
    float score;
    label_id label;          // canonical class (see canonical_exact)
    const char * label_name; // canonical class name (static string)
    // The model's own class: index into its label set and its name. For
    // rt_detr_v2 these equal the canonical class; for dfine they are OXR's
    // class id (0..11) and name ("Caption", ..., "Chemical Structure").
    int raw_class = -1;
    const char * raw_label = nullptr; // static string, valid for the process lifetime
    // False when the raw class has no canonical equivalent and `label` is the
    // documented nearest class (OXR "Stamp" and "Chemical Structure" -> picture).
    bool canonical_exact = true;
};

// OXR D-FINE class table (official service order) and its canonical mapping.
constexpr int OXR_NUM_CLASSES = 12;
const char * oxr_label_name(int raw_class); // nullptr when out of range
label_id oxr_canonical_label(int raw_class, bool * exact = nullptr);

struct context;

// Load a layout GGUF (either architecture). Returns true on success.
bool load(context ** ctx, const char * path, int n_threads = 1);

arch_id arch(const context * ctx);
// Number of raw model classes and their names (heron: 17, dfine: 12).
int num_raw_classes(const context * ctx);
const char * raw_label_name(const context * ctx, int raw_class);
// Model-default score threshold for detect_file/detect_rgb when a negative
// threshold is passed: heron 0.3, dfine 0.5 (OXR LayoutAnalyzeModelConfig).
float default_score_threshold(const context * ctx);

// Detect layout regions from an image file.
//   rt_detr_v2: resize to 640x640, threshold, score-sorted (unchanged behaviour).
//   dfine: OXR service preprocessing and postprocessing, then OXR's client-side
//   selection (score >= threshold, clip to the page, class-wise NMS IoU 0.6),
//   in service candidate order. A negative threshold selects the model default.
std::vector<region> detect_file(context * ctx, const char * path, float score_threshold = -1.0f);

// Same as detect_file for an in-memory 8-bit image (1, 3 or 4 interleaved
// channels; gray is replicated, alpha is discarded like cv2.IMREAD_COLOR).
std::vector<region> detect_rgb(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                               float score_threshold = -1.0f);

// rt_detr_v2 only: detect from preprocessed pixels [3, 640, 640] CHW float32.
std::vector<region> detect(context * ctx, const float * pixels, int orig_h, int orig_w, float score_threshold = 0.3f);

// ---------------------------------------------------------------------------
// D-FINE (OXR) raw service boundary
// ---------------------------------------------------------------------------

// One /inference candidate: top-k over (query, class) sigmoid scores,
// letterbox-inverted and clipped exactly like DFINEPostProcessor.
struct candidate {
    int raw_class;
    float score;
    float box[4]; // x1, y1, x2, y2 in original pixels
};

struct timings {
    double preprocess_ms = 0, encoder_ms = 0, decoder_ms = 0, postprocess_ms = 0, total_ms = 0;
};

// Run the D-FINE model and return all num_queries candidates in the exact
// service order (no thresholding). Returns false on error (see last_error).
bool detect_candidates(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                       std::vector<candidate> & out);
// Same from an already preprocessed 960x960 CHW float input (parity tooling).
bool detect_candidates_preprocessed(context * ctx, const float * chw, int orig_width, int orig_height,
                                    std::vector<candidate> & out);

// OXR parse_layout_batch selection (oxr.pipeline.postprocess.layout_postprocess):
// keep score >= threshold, reject degenerate boxes, clip to the page,
// class-wise greedy NMS (iou_threshold < 0 disables), preserving input order.
std::vector<region> oxr_select(const std::vector<candidate> & candidates, int width, int height, float score_threshold,
                               float nms_iou_threshold);

const char * last_error(const context * ctx);
timings last_timings(const context * ctx);

// Parity tooling: when set, D-FINE inference copies every named stage (shapes
// in ggml order, ne0 fastest) into *capture. Pass nullptr to disable.
struct stage_tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;
};
using stage_map = std::map<std::string, stage_tensor>;
void set_stage_capture(context * ctx, stage_map * capture);

// Free resources.
void free(context * ctx);

} // namespace layout_detect
