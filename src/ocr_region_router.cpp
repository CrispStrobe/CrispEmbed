#include "ocr_region_router.h"

namespace ocr_region_router {

void routing_plan::clear() {
    layout_decisions.clear();
    text_indices.clear();
    text_to_layout.clear();
    table_layout_indices.clear();
    formula_layout_indices.clear();
    fallback_layout_indices.clear();
    suppress_text.clear();
}

static bool contains_centroid(const layout_detect::region & r, const ocr_detect::text_box & b) {
    const float cx = b.x + b.w * 0.5f;
    const float cy = b.y + b.h * 0.5f;
    return cx >= r.x1 && cx <= r.x2 && cy >= r.y1 && cy <= r.y2;
}

static int owner_for(const std::vector<layout_detect::region> & layout, const ocr_detect::text_box & box) {
    for (int i = 0; i < static_cast<int>(layout.size()); ++i)
        if (contains_centroid(layout[i], box)) return i;
    return -1;
}

void build(const std::vector<layout_detect::region> & layout, const std::vector<ocr_detect::text_box> & text_boxes,
           const request_policy & policy, routing_plan & out) {
    out.clear();
    out.suppress_text.assign(text_boxes.size(), 0);

    if (layout.empty()) {
        out.text_indices.reserve(text_boxes.size());
        out.text_to_layout.assign(text_boxes.size(), -1);
        for (int i = 0; i < static_cast<int>(text_boxes.size()); ++i) out.text_indices.push_back(i);
        return;
    }

    out.layout_decisions.reserve(layout.size());
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto label = layout[i].label;
        destination dest = destination::text;
        if (label == layout_detect::label_id::table && policy.want_tables) {
            dest = destination::table;
            out.table_layout_indices.push_back(i);
        } else if (label == layout_detect::label_id::formula && policy.want_formulas) {
            dest = destination::formula;
            out.formula_layout_indices.push_back(i);
        } else if (label == layout_detect::label_id::picture && policy.image_text_fallback) {
            dest = destination::fallback;
            out.fallback_layout_indices.push_back(i);
        }
        out.layout_decisions.push_back({ i, dest, false });
    }

    out.text_indices.reserve(text_boxes.size());
    out.text_to_layout.reserve(text_boxes.size());
    for (int i = 0; i < static_cast<int>(text_boxes.size()); ++i) {
        const int owner = owner_for(layout, text_boxes[i]);
        out.text_to_layout.push_back(owner);
        const bool specialized = owner >= 0 && (out.layout_decisions[owner].dest == destination::table ||
                                                out.layout_decisions[owner].dest == destination::formula);
        if (specialized) {
            out.suppress_text[i] = 1;
            continue;
        }
        out.text_indices.push_back(i);
    }
}

} // namespace ocr_region_router
