// ocr_region_router.h — model-free routing plan for structured OCR.
//
// Layout and text detection are intentionally supplied as already-computed
// results. This keeps routing cheap, deterministic, and independently
// testable; model dispatch is a later pipeline concern.

#pragma once

#include "layout_detect.h"
#include "ocr_detect.h"

#include <cstdint>
#include <vector>

namespace ocr_region_router {

enum class destination : uint8_t {
    text,
    table,
    formula,
    fallback,
};

struct request_policy {
    bool want_tables = false;
    bool want_formulas = false;
    bool image_text_fallback = true;
};

struct decision {
    int layout_index = -1;
    destination dest = destination::text;
    // A specialized recognizer may also emit text that should be merged into
    // the document text stream (currently reserved for future salvage rules).
    bool also_text = false;
};

struct routing_plan {
    std::vector<decision> layout_decisions;
    std::vector<int> text_indices;
    std::vector<int> text_to_layout;
    std::vector<int> table_layout_indices;
    std::vector<int> formula_layout_indices;
    std::vector<int> fallback_layout_indices;
    std::vector<uint8_t> suppress_text;

    void clear();
};

// Build a deterministic dispatch plan. Detection boxes are assigned to the
// first layout region containing their centroid. When no layout is available,
// every detection box remains on the text path.
void build(const std::vector<layout_detect::region> & layout, const std::vector<ocr_detect::text_box> & text_boxes,
           const request_policy & policy, routing_plan & out);

} // namespace ocr_region_router
