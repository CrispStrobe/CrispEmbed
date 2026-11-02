#pragma once

// Deterministic top-to-bottom, left-to-right reading order for recognised
// regions (ocr_orchestrator assemble(), default since 2026-11-02).
//
// The previous comparator ("same row when |top_a - top_b| <= 0.65 * min h,
// then by x") is not transitive, which makes std::stable_sort's result
// implementation-defined; libc++ and libstdc++ produced different page text
// from identical regions. This version only ever sorts with strict weak
// orderings: regions are sorted by vertical centre (then x, then index), split
// greedily into rows -- a region joins the current row when its centre is
// within half the smaller height of the row's first region -- and each row is
// sorted by x (then index). Same input, same output, on every standard library.

#include <algorithm>
#include <numeric>
#include <vector>

namespace ocr_reading_order {

template <class Region> std::vector<int> rows(const std::vector<Region> & regions) {
    const int n = (int)regions.size();
    std::vector<int> order((size_t)n);
    std::iota(order.begin(), order.end(), 0);
    auto cy = [&](int i) { return regions[(size_t)i].box.y + 0.5f * regions[(size_t)i].box.h; };
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const float ya = cy(a), yb = cy(b);
        if (ya != yb) return ya < yb;
        if (regions[(size_t)a].box.x != regions[(size_t)b].box.x)
            return regions[(size_t)a].box.x < regions[(size_t)b].box.x;
        return a < b;
    });
    std::vector<int> out;
    out.reserve((size_t)n);
    size_t row_start = 0;
    auto flush = [&](size_t end) {
        std::sort(order.begin() + (long)row_start, order.begin() + (long)end, [&](int a, int b) {
            if (regions[(size_t)a].box.x != regions[(size_t)b].box.x)
                return regions[(size_t)a].box.x < regions[(size_t)b].box.x;
            return a < b;
        });
        out.insert(out.end(), order.begin() + (long)row_start, order.begin() + (long)end);
        row_start = end;
    };
    for (size_t k = 1; k < order.size(); ++k) {
        const int anchor = order[row_start];
        const int cur = order[k];
        const float tol = std::max(4.0f, 0.5f * std::min(regions[(size_t)anchor].box.h, regions[(size_t)cur].box.h));
        if (cy(cur) - cy(anchor) > tol) flush(k);
    }
    if (row_start < order.size()) flush(order.size());
    return out;
}

} // namespace ocr_reading_order
