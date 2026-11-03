// Temporally consistent inpainting of a frame sequence (issue 71: burned-in
// subtitles in video). Engine-agnostic: the engine is a fill callback.
//
// Frames are pushed in display order and come out in the same order, delayed by
// up to `window` frames of look-ahead. Per frame, for the whole masked area:
//   1. propagation: copy the real background from the nearest frame within
//      +-window that has no mask at all, if the ring around the mask is static
//      between the two frames (strict: half the static threshold);
//   2. static-span reuse: with the same mask as the previous frame and a static
//      ring relative to both the previous input and the last fresh fill's input,
//      reuse the previous output (no flicker, no engine call);
//   3. flat fill: where known pixels near the hole are flat (every trimmed
//      luma/R/G/B std <= 4, blending to the engine up to 8), a harmonic interpolation of
//      the surroundings; engines drift on flat tinted areas, and fully flat
//      frames need no engine call;
//   4. the engine fills the rest; with the same mask and only a moderate ring
//      change, its fill is blended with the previous output (less shimmer).
// "Static" compares 8x8 cell means of the ring (unmasked pixels within 24 px of
// the mask), guarded by half the per-pixel absolute residual so moving fine
// texture cannot cancel within a cell. Mean difference <= static_threshold and
// 90th percentile <= twice that; small noise can still reuse. Analysis is limited
// to the mask's bounding box plus that ring; frame I/O/copies are full-size.
// window = 0 disables 1 and 2.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace inpaint_temporal {

// Fill the 255 pixels of `mask` (W*H, 0/255) in `rgb` (W*H*3) into `out`; other
// pixels must be copied unchanged.
using fill_fn = std::function<bool(const uint8_t * rgb, const uint8_t * mask, uint8_t * out, std::string & error)>;

struct settings {
    int window = 15;               // frames of look-ahead and look-behind
    float static_threshold = 3.0f; // ring cell mean difference (levels)
    float blend_threshold = 12.0f; // blend engine fills up to this change (0 = off)
    bool flat_fill = true;
    // Conservative cap on owned frame payloads: input RGB/cropped masks,
    // previous/current/queued output, reuse anchor, and engine RGB/mask scratch.
    // Excludes region-local analysis/harmonic scratch, model memory, caller I/O
    // buffers and allocator overhead. The window is reduced to fit; a budget
    // below 18 * width * height bytes makes push/finish fail before allocation.
    size_t max_buffer_bytes = size_t(2) << 30;
};

struct stats {
    int64_t frames = 0, frames_masked = 0, frames_engine = 0;
    int64_t px_masked = 0, px_propagated = 0, px_reused = 0, px_flat = 0, px_engine = 0, px_blended = 0;
    int64_t harmonic_updates = 0; // SOR pixel updates, at most 20 million per processed frame
};

class stream {
public:
    stream(int width, int height, const settings & options, fill_fn fill);
    ~stream();
    stream(const stream &) = delete;
    stream & operator=(const stream &) = delete;

    // Next frame: RGB8 HWC and a W*H mask (>= 128 = fill; null = nothing to fill).
    // Both are copied. Drain pop() before the next push: pending ready output
    // returns false without consuming input; pop and retry is safe. A null RGB
    // pointer also rejects without consumption. False on invalid construction
    // or an engine error leaves the stream unusable.
    bool push(const uint8_t * rgb, const uint8_t * mask, std::string & error);
    // No more frames: the whole remaining tail becomes ready in one call.
    // Idempotent on success. Completed outputs stay poppable after an error.
    bool finish(std::string & error);
    // Writes the next finished frame (W*H*3) and returns true, or false if none is ready.
    bool pop(uint8_t * out);
    const stats & counters() const;
    // The window in use (settings.window, reduced to fit max_buffer_bytes).
    int window() const;

private:
    struct state;
    std::unique_ptr<state> s_;
};

} // namespace inpaint_temporal
