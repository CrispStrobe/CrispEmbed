// oxr_document.h — native OXR-1.0 document route (everything the original
// CrispOXR.process_pages helper does except the D-FINE layout detector).
//
// Translated from SoMarkAI/OXR @ c1a1dcb312ee9f691404f127785510453c0d9350
// (oxr/pipeline/*, oxr/model/prompts.py, oxr/utils/{image,table}.py),
// Apache License 2.0 with the additional OXR terms in licenses/oxr-LICENSE.
// See licenses/oxr-NOTICE. Changes: C++ data structures; synchronous,
// deterministic request order (all reading-order requests, then blocks in
// page order) instead of asyncio scheduling; no 1 s sleep between retries;
// recognition is a callback (the resident native OXR model in production).
//
// Split between generic and OXR-specific steps:
//   generic: layout_region (boundary for any detector, incl. native D-FINE),
//            crop geometry (crop_rgb), cv_compat resize, xycut_plus_plus,
//            conversion to crisp_marker::document for the shared renderers;
//   OXR:     D-FINE response validation/threshold/NMS, prompt routing,
//            reading-order prompt/parse/repair, length recovery, content
//            postprocessing, page cleanup, exact upstream JSON + Markdown.
//
// Unsupported upstream options are rejected rather than approximated:
// element_formats.table="markdown", table_image_placeholders, asset_dir.
#pragma once

#include "core/json_value.h"
#include "marker_document.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace layout_detect {
struct region;
struct candidate;
struct context;
} // namespace layout_detect

namespace oxr_document {

// D-FINE class indices in the official response order (not alphabetical).
enum class block_type : int {
    caption = 0,
    footnote,
    formula,
    text,
    page_footer,
    page_header,
    picture,
    title,
    table,
    code_block,
    stamp,
    chemical_structure,
};
constexpr int k_num_layout_classes = 12;
const char * type_name(block_type type); // "Caption", ..., "Chemical Structure"
bool type_from_name(const std::string & name, block_type & out);

enum class content_format { text, html, latex, markdown };
const char * format_name(content_format format);

// Layout-provider boundary: one detected candidate in page pixel coordinates,
// before score filtering, clipping and class-wise NMS (exactly what a D-FINE
// /inference response carries for one box).
struct layout_region {
    int label = 3; // block_type index 0..11
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    double score = 1.0;
};
using page_layout = std::vector<layout_region>;

// Convert CrispEmbed layout_detect output whose label names are OXR class
// names (as a native D-FINE provider reports them). Throws on other labels.
page_layout from_layout_detect(const std::vector<layout_detect::region> & regions);
// Build the official batch response {"status","params","data":{labels,boxes,scores}}.
core_json_value::value layout_response(const std::vector<page_layout> & pages);
// Same response from native D-FINE candidates (layout_detect::detect_candidates,
// all queries in service order, no thresholding): what the OXR layout service
// returns and what the original pipeline hands to parse_layout_batch.
core_json_value::value layout_response(const std::vector<std::vector<layout_detect::candidate>> & pages);

struct rgb_page;
// Default native layout provider: run OXR D-FINE (layout_detect arch "dfine")
// on each RGB page and return the batch response. The document route then
// applies its own threshold, clipping and class-wise NMS (parse_layout_batch),
// exactly as for a supplied or service response. Throws on detector failure.
core_json_value::value detect_layout_response(layout_detect::context * dfine, const std::vector<rgb_page> & pages);

struct block {
    int idx = 0;
    block_type type = block_type::text;
    std::array<double, 4> bbox{};
    std::string content; // UTF-8
    content_format format = content_format::text;
    std::string text_before, text_after, generated_role;
    std::optional<double> layout_score;
    uint64_t uid = 0; // object identity (Python id()) for cleanup witnesses
};

struct rgb_page {
    int width = 0, height = 0;
    const uint8_t * rgb = nullptr; // tightly packed RGB
};

struct rec_request {
    const uint8_t * rgb = nullptr;
    int width = 0, height = 0;
    std::string prompt;
    int max_tokens = 0;
    double repetition_penalty = 1.0; // Python float; the native ABI receives float32
};
struct rec_result {
    std::string text; // processed native text (special tokens removed)
    bool eos = false, capped = false;
    std::string finish_reason; // "stop" | "length" | other
    std::string native_json;   // complete native result object (trace only)
};
using recognizer = std::function<rec_result(const rec_request &)>;

struct options {
    std::string file_name = "document.png";
    bool want_json = true, want_markdown = true;
    bool keep_header_footer = false;
    // model_options: temperature/top_p must stay 0/1 (native greedy).
    double temperature = 0.0, top_p = 1.0, repetition_penalty = 1.0;
    std::map<std::string, std::string> element_formats; // "cs": "image" supported
    double score_threshold = 0.5;
    std::optional<double> nms_iou_threshold = 0.6;
    bool provides_reading_order = false;
    std::string image_output_dir = "outputs";
    bool write_images = true; // upstream writes Picture crops as PNG files
    int max_new_tokens = 8192;
    bool allow_truncated_content = false;
    bool full_overlap_dedupe = false;
    bool table_image_placeholders = false; // rejected when true
    int retry_times = 2;
    std::string layout_source = "provided"; // metadata.crispembed.layout_source
};
// Parse/validate the options JSON accepted by the C API (see docs/oxr.md).
options parse_options(const std::string & json);

struct trace_entry {
    std::string kind; // "reading_order" | "block"
    int page = 0, block = -1;
    std::string prompt, rgb_sha256;
    int width = 0, height = 0, max_tokens = 0;
    double repetition_penalty = 1.0;
    std::string finish_reason, native_json;
};

struct result {
    std::vector<std::vector<block>> pages;
    std::vector<std::pair<int, int>> page_sizes; // (w, h)
    std::string markdown;
    std::string json; // upstream-shaped document result
    std::vector<trace_entry> trace;
    int native_requests = 0, unresolved = 0;
    std::vector<crisp_marker::asset> assets;        // Picture crops (PNG)
    std::map<uint64_t, std::string> picture_assets; // block uid -> asset name
};

// Full document route. layout_response is the official D-FINE batch object.
// Throws std::runtime_error/invalid_argument with upstream-equivalent messages.
// trace_out, when given, receives every native request even when processing fails.
result process_pages(const std::vector<rgb_page> & pages, const core_json_value::value & layout_response,
                     const options & opts, const recognizer & recognize,
                     std::vector<trace_entry> * trace_out = nullptr);

// Same blocks in CrispEmbed's shared document model (Markdown/HTML/JSON/chunk
// renderers in marker_document.cpp). Block.source is "oxr-1.0", raw_label the
// OXR class, order the final reading order and raw the exact OXR content.
crisp_marker::document to_marker_document(const result & value, const std::string & source_name);

// ---- individually testable helpers (exact upstream semantics) ----
namespace helpers {
std::vector<std::vector<block>> parse_layout_batch(const core_json_value::value & payload,
                                                   const std::vector<std::pair<int, int>> & image_sizes,
                                                   double score_threshold, std::optional<double> nms_iou_threshold);
std::string reading_order_prompt(const std::vector<block> & blocks, int width, int height);
bool parse_reading_order(const std::string & text, int count, std::vector<int> & order);
std::optional<std::vector<int>> repair_incomplete_reading_order(const std::string & text,
                                                                const std::vector<block> & blocks);
std::optional<std::vector<int>> supplement_incomplete_reading_order(const std::string & text,
                                                                    const std::vector<block> & blocks);
std::vector<int> xycut_plus_order(const std::vector<block> & blocks);
void postprocess(block & value, const std::string & raw, const std::string & table_format = "html");
std::string move_formula_numbering_outside_math(const std::string & text);
std::string space_markdown_math(const std::string & markdown);
std::string clean_formula_markdown_delimiters(const std::string & latex);
std::vector<block> dedupe_systematic_cross_category(const std::vector<block> & blocks);
std::vector<block> expand_formula_numbering(const std::vector<block> & blocks);
std::vector<block> dedupe_visual_parents(const std::vector<block> & raw, const std::vector<block> & blocks,
                                         const rgb_page & image, bool allow_mixed_children);
std::vector<block> dedupe_complete_text(const std::vector<block> & blocks);
std::vector<block> dedupe_exact_textual(const std::vector<block> & blocks);
std::vector<block> dedupe_full_overlaps(const std::vector<block> & blocks, const rgb_page * image);
std::string blocks_to_markdown(const std::vector<block> & blocks);
std::string serialize_page_json(const std::vector<block> & blocks, int page_num, int width, int height);
std::vector<uint8_t> crop_rgb(const rgb_page & image, const std::array<double, 4> & bbox, int & width, int & height);
} // namespace helpers

} // namespace oxr_document
