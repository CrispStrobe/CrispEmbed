#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace ppocrv6_det {
namespace policy {

enum class shape_action { build, reuse, fallback, rebuild };

inline shape_action graph_shape(bool attempted, bool ready, int old_h, int old_w, int h, int w, bool rebuild) {
    if (!attempted) return shape_action::build;
    if (old_h == h && old_w == w) return ready ? shape_action::reuse : shape_action::fallback;
    return rebuild ? shape_action::rebuild : shape_action::fallback;
}

// Empty detections are only trustworthy when a complete, finite probability
// map was produced. Nonempty graph results preserve their existing behavior.
inline bool accept_graph(bool compare, bool accept_empty, size_t boxes, const std::vector<float> & probability, int h,
                         int w) {
    if (compare) return false;
    if (boxes) return true;
    if (!accept_empty || h <= 0 || w <= 0 || probability.size() != size_t(h) * size_t(w)) return false;
    for (float p : probability)
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f) return false;
    return true;
}

} // namespace policy
} // namespace ppocrv6_det
