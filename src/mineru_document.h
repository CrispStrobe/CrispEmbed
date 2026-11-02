// mineru_document.h - MinerU2.5-Pro complete two-step document route built from
// CrispEmbed pipeline components:
//
//   1. layout provider: MinerU's own VLM layout output (upstream regex) becomes
//      layout_detect::region-compatible regions plus MinerU extras (raw type,
//      rotation angle, continuation flag, normalized bbox).
//   2. ocr_region_router plans per-region destinations; recognition runs with
//      MinerU's prompts through an injected predictor (native GGML in
//      production, recorded outputs in weight-free parity tests).
//   3. MinerU-specific steps, translated from mineru-vl-utils c9797cde
//      (Apache-2.0): crop/rotation/resize_by_need, table-image masking and
//      restoration, formula/text repairs, image/chart analysis, equation-block
//      merging, [Non-Text] cleanup (mineru_postprocess / mineru_table_image).
//   4. The result is written into the crisp_marker document model; the
//      "upstream" output profile renders byte-identical mineru-vl-utils JSON
//      (json.dumps(..., ensure_ascii=False, separators=(",", ":"))) and
//      json2md Markdown from that model, and the generic Marker renderers
//      (markdown/html/json/chunks) work on the same document.
//
// Gate: callers honour CRISPEMBED_MINERU25_PRO_FULL_DOC=0 (default on).
#pragma once

#include "layout_detect.h"
#include "marker_document.h"
#include "mineru_postprocess.h"
#include "mineru_table_image.h"
#include "ocr_region_router.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace mineru_doc {

const char * profile_name(); // "mineru-vl-utils-c9797cde-full-v1"

// Layout provider output: a layout_detect region plus MinerU's extras.
struct region {
    layout_detect::region base; // pixel bbox, score 1 (MinerU emits none), mapped label
    std::string raw_type;       // MinerU block type
    int angle = -1;             // None or 0/90/180/270
    bool merge_prev = false;
    std::array<double, 4> norm = { { 0, 0, 0, 0 } }; // exact normalized bbox
    int index = -1;                                  // position in the parsed block list
};

struct recognition {
    std::string text;
    bool eos = true, capped = false;
    double ms = 0;
    float confidence = 0;
    std::string json; // native per-request receipt (optional)
};
using predictor = std::function<recognition(const mineru_img::rgb_image & image, const std::string & prompt)>;

struct options {
    mineru_pp::options post;
    bool image_analysis = false;
    std::vector<std::string> not_extract;
    bool allow_truncated = false; // CrispMinerU(allow_truncated_content=...)
    bool has_seed = false;        // table-image token RNG (Python random.seed)
    uint64_t seed = 0;
};

struct crop_record {
    size_t block_index = 0;
    std::string type, prompt;
    int width = 0, height = 0;
    uint64_t fnv1a = 0;        // FNV-1a 64 of the RGB bytes sent to the model
    bool masked_table = false; // contains approximated token glyphs
    recognition result;
};

struct page_result {
    std::vector<region> regions;
    ocr_region_router::routing_plan plan;
    recognition layout;
    std::vector<crop_record> crops;
    mineru_pp::block_list blocks; // final upstream ExtractResult
    std::string json, markdown;   // upstream profile
    bool capped = false;
};

uint64_t fnv1a(const std::vector<uint8_t> & bytes);

// MinerUClientHelper.parse_layout_output (exact).
mineru_pp::block_list parse_layout_output(const std::string & text);
std::vector<region> layout_regions(const mineru_pp::block_list & blocks, int width, int height);
layout_detect::label_id label_for(const std::string & mineru_type);

// prepare_for_layout: Pillow bicubic resize to 1036x1036.
mineru_img::rgb_image prepare_for_layout(const mineru_img::rgb_image & page);
// MinerUClientHelper.resize_by_need.
mineru_img::rgb_image resize_by_need(const mineru_img::rgb_image & img, int min_edge = 28, double max_ratio = 50);

// One page: layout, prepare_for_extract, recognition, post_process.
page_result extract_page(const mineru_img::rgb_image & page, const predictor & predict, const options & opts,
                         mineru_img::python_random & rng);

// Document model integration.
void append_page(crisp_marker::document & doc, const page_result & page, int page_index, int width, int height);
std::string render_upstream_json(const crisp_marker::page & page);     // byte-identical to Python
std::string render_upstream_markdown(const crisp_marker::page & page); // json2md

// Weight-free test helpers.
std::string layout_blocks_json(const std::string & layout_text);
std::string replay_case_json(const std::string & case_json);
bool self_test(std::string & error);

// Options from a JSON object (C API); unknown keys are rejected.
options parse_options_json(const std::string & json);

} // namespace mineru_doc
