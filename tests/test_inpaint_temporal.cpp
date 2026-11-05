// Model-free stream contracts: temporal selection, preserved known pixels,
// bounded frame payloads and failures, without loading an inpainting model.
#include "core/clean_exit.h"
#include "inpaint_temporal.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {
int failures = 0, checks = 0;
void check(bool ok, const char * description) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description);
    }
}
constexpr int W = 192, H = 64;
using image = std::vector<uint8_t>;
using sequence = std::vector<image>;

struct engine {
    int width = W, height = H, calls = 0, fail_at = 0;
    bool fail = false, alternate = false, follow_input = false, save_masks = false, binary_masks = true;
    sequence masks;
    inpaint_temporal::fill_fn fn() {
        return [this](const uint8_t * rgb, const uint8_t * mask, uint8_t * out, std::string & error) {
            ++calls;
            if (fail || calls == fail_at) {
                error = "fake engine failure";
                return false;
            }
            if (save_masks) masks.emplace_back(mask, mask + size_t(width) * height);
            for (int i = 0; i < width * height; i++) {
                binary_masks = binary_masks && (mask[i] == 0 || mask[i] == 255);
                for (int c = 0; c < 3; c++) {
                    const uint8_t fill = follow_input ? rgb[c] : uint8_t(alternate && calls % 2 ? 100 : 200);
                    out[i * 3 + c] = mask[i] ? fill : rgb[i * 3 + c];
                }
            }
            return true;
        };
    }
};

image textured(int shift, int offset = 0) {
    image rgb(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                rgb[(y * W + x) * 3 + c] = uint8_t(((x + shift) / 12 + y / 12) % 2 * 100 +
                                                   ((x + shift) * 37 + y * 101 + c * 53) % 40 + 50 + offset);
    return rgb;
}
image flat() {
    image rgb(W * H * 3);
    for (int i = 0; i < W * H; i++) {
        rgb[i * 3] = 40;
        rgb[i * 3 + 1] = 140;
        rgb[i * 3 + 2] = 60;
    }
    return rgb;
}
image bar_mask() {
    image mask(W * H, 0);
    for (int y = 40; y < 52; y++)
        for (int x = 20; x < 76; x++) mask[y * W + x] = 255;
    return mask;
}
image burn(image rgb, const image & mask, uint8_t value = 255) {
    for (size_t i = 0; i < mask.size(); i++)
        if (mask[i] >= 128)
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = value;
    return rgb;
}
bool known_unchanged(const image & input, const image & mask, const image & out) {
    if (out.size() != input.size()) return false;
    for (size_t i = 0; i < mask.size(); i++)
        if (mask[i] < 128)
            for (int c = 0; c < 3; c++)
                if (out[i * 3 + c] != input[i * 3 + c]) return false;
    return true;
}
inpaint_temporal::settings with_window(int window) {
    inpaint_temporal::settings s;
    s.window = window;
    return s;
}
void drain(inpaint_temporal::stream & s, sequence & outputs, int width = W, int height = H) {
    image out(size_t(width) * height * 3);
    while (s.pop(out.data())) outputs.push_back(out);
}
void feed(inpaint_temporal::stream & s, const image & rgb, const image * mask, sequence & outputs, int width = W,
          int height = H) {
    std::string error;
    check(s.push(rgb.data(), mask ? mask->data() : nullptr, error), "push complete frame");
    drain(s, outputs, width, height);
}
void flush(inpaint_temporal::stream & s, sequence & outputs, int width = W, int height = H) {
    std::string error;
    check(s.finish(error), "finish stream");
    drain(s, outputs, width, height);
    const auto & st = s.counters();
    check(st.px_masked == st.px_propagated + st.px_reused + st.px_flat + st.px_engine,
          "every masked pixel has one fill category");
}

void order_and_delay() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(2), e.fn());
    sequence outputs;
    for (int t = 0; t < 2; t++) feed(s, textured(t), nullptr, outputs);
    check(outputs.empty(), "no output before the look-ahead is full");
    feed(s, textured(2), nullptr, outputs);
    check(outputs.size() == 1 && outputs[0] == textured(0), "first output after two look-ahead frames");
    flush(s, outputs);
    check(outputs == sequence{ textured(0), textured(1), textured(2) }, "distinct frames retain order");
    check(e.calls == 0 && s.counters().frames == 3, "empty masks need no engine");
}

void propagation_from_past_and_future() {
    const auto mask = bar_mask(), background = textured(0), masked = burn(background, mask);
    for (bool future : { false, true }) {
        engine e;
        inpaint_temporal::stream s(W, H, with_window(3), e.fn());
        sequence outputs;
        for (int t = 0; t < 4; t++) {
            const bool clean = t == (future ? 3 : 0);
            feed(s, clean ? background : masked, clean ? nullptr : &mask, outputs);
        }
        flush(s, outputs);
        check(outputs == sequence(4, background), "past/future donor reconstructs exact background");
        check(e.calls == 0 && s.counters().px_propagated == 3 * 12 * 56, "window-edge donor skips engine");
    }
}

void nearest_donor_and_window_boundary() {
    const auto mask = bar_mask(), background = textured(0), masked = burn(background, mask);
    const auto near = burn(background, mask, 101), far = burn(background, mask, 202);
    {
        engine e;
        inpaint_temporal::stream s(W, H, with_window(3), e.fn());
        sequence outputs;
        feed(s, masked, &mask, outputs);
        feed(s, near, nullptr, outputs);
        feed(s, far, nullptr, outputs);
        flush(s, outputs);
        check(outputs.size() == 3 && outputs[0] == near, "nearest future donor wins");
    }
    {
        engine e;
        inpaint_temporal::stream s(W, H, with_window(2), e.fn());
        sequence outputs;
        feed(s, near, nullptr, outputs);
        feed(s, masked, &mask, outputs);
        feed(s, far, nullptr, outputs);
        flush(s, outputs);
        check(outputs.size() == 3 && outputs[1] == near, "equidistant donors prefer the past");
    }
    {
        engine e;
        inpaint_temporal::stream s(W, H, with_window(1), e.fn());
        sequence outputs;
        feed(s, masked, &mask, outputs);
        feed(s, masked, &mask, outputs);
        feed(s, near, nullptr, outputs);
        flush(s, outputs);
        check(outputs.size() == 3 && outputs[0][(45 * W + 30) * 3] == 200 && outputs[1] == near,
              "donor outside the window cannot fill the first frame");
        check(e.calls == 1, "only frame without an in-window donor uses engine");
    }
}

void static_span_reuse_and_small_noise() {
    const auto mask = bar_mask();
    for (bool noisy : { false, true }) {
        engine e;
        inpaint_temporal::stream s(W, H, with_window(2), e.fn());
        sequence inputs, outputs;
        for (int t = 0; t < 8; t++) {
            auto rgb = textured(0);
            if (noisy)
                for (size_t i = 0; i < rgb.size(); i++) rgb[i] = uint8_t(rgb[i] + ((i + t) % 2 ? 2 : -2));
            inputs.push_back(burn(rgb, mask));
            feed(s, inputs.back(), &mask, outputs);
        }
        flush(s, outputs);
        check(outputs.size() == inputs.size() && e.calls == 1, "static/noisy span needs one engine fill");
        bool preserved = outputs.size() == inputs.size();
        for (size_t i = 0; i < outputs.size() && i < inputs.size(); i++)
            preserved =
                preserved && known_unchanged(inputs[i], mask, outputs[i]) && outputs[i][(45 * W + 30) * 3] == 200;
        check(preserved, "reuse preserves current known pixels");
        check(s.counters().px_reused == 7 * 12 * 56, "static reuse retains useful noise tolerance");
    }
}

void cumulative_drift_refreshes_the_fill() {
    engine e;
    e.follow_input = true;
    auto options = with_window(2);
    options.flat_fill = false;
    inpaint_temporal::stream s(W, H, options, e.fn());
    const auto mask = bar_mask();
    sequence inputs, outputs;
    for (int t = 0; t < 50; t++) {
        inputs.push_back(burn(textured(0, t), mask));
        feed(s, inputs.back(), &mask, outputs);
    }
    flush(s, outputs);
    check(outputs.size() == 50 && e.calls >= 10 && e.calls < 50,
          "gradual drift refreshes without freezing or disabling all reuse");
    check(outputs.size() == 50 && outputs.back()[(45 * W + 30) * 3] > outputs.front()[(45 * W + 30) * 3] + 30,
          "fill follows cumulative change beyond the temporal window");
    bool preserved = outputs.size() == inputs.size();
    for (size_t i = 0; i < outputs.size() && i < inputs.size(); i++)
        preserved = preserved && known_unchanged(inputs[i], mask, outputs[i]);
    check(preserved, "drift refresh preserves known pixels");
}

image stripes(int phase) {
    image rgb(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) rgb[(y * W + x) * 3 + c] = uint8_t(50 + 150 * ((x + phase) % 2));
    return rgb;
}
// textured(0) with deterministic per-frame grain of up to +-amplitude levels.
image grainy(int frame, int amplitude) {
    image rgb = textured(0);
    uint32_t state = 2654435761u * uint32_t(frame + 1);
    for (auto & v : rgb) {
        state = state * 1664525u + 1013904223u;
        v = uint8_t(std::clamp(int(v) + int(state >> 24) % (2 * amplitude + 1) - amplitude, 0, 255));
    }
    return rgb;
}

void grain_on_a_still_shot_still_propagates() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(3), e.fn());
    sequence outputs;
    const auto mask = bar_mask();
    const image donor = grainy(0, 5);
    feed(s, donor, nullptr, outputs);
    for (int t = 1; t < 4; t++) feed(s, burn(grainy(t, 5), mask), &mask, outputs);
    flush(s, outputs);
    bool from_donor = outputs.size() == 4;
    for (size_t i = 1; i < outputs.size(); i++)
        for (int y = 40; y < 52; y++)
            for (int x = 20; x < 76; x++)
                for (int c = 0; c < 3; c++)
                    from_donor = from_donor && outputs[i][(y * W + x) * 3 + c] == donor[(y * W + x) * 3 + c];
    check(from_donor && e.calls == 0 && s.counters().px_propagated == 3 * 12 * 56,
          "grain (+-5 levels) on a still shot propagates the real background");
}

void moving_texture_cannot_cancel_inside_cells() {
    const auto mask = bar_mask();
    for (bool clean_donor : { false, true }) {
        engine e;
        e.alternate = true;
        inpaint_temporal::stream s(W, H, with_window(1), e.fn());
        sequence outputs;
        feed(s, clean_donor ? stripes(0) : burn(stripes(0), mask), clean_donor ? nullptr : &mask, outputs);
        feed(s, burn(stripes(1), mask), &mask, outputs);
        flush(s, outputs);
        check(e.calls == (clean_donor ? 1 : 2) && s.counters().px_propagated == 0 && s.counters().px_reused == 0 &&
                  s.counters().px_blended == 0,
              "one-pixel texture motion rejects propagation, reuse and blend");
        check(outputs.size() == 2 && known_unchanged(stripes(1), mask, outputs[1]),
              "moving texture fallback preserves current known pixels");
    }
}

void moving_background_calls_the_engine() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(2), e.fn());
    const auto mask = bar_mask();
    sequence outputs;
    for (int t = 0; t < 4; t++) feed(s, burn(textured(4 * t), mask), &mask, outputs);
    flush(s, outputs);
    check(e.calls == 4, "every frame of a moving shot uses engine");
}

void flat_background_needs_no_engine() {
    const auto mask = bar_mask(), input = burn(flat(), mask);
    for (bool enabled : { false, true }) {
        engine e;
        auto options = with_window(0);
        options.flat_fill = enabled;
        inpaint_temporal::stream s(W, H, options, e.fn());
        sequence outputs;
        feed(s, input, &mask, outputs);
        flush(s, outputs);
        check(outputs.size() == 1 && e.calls == (enabled ? 0 : 1), "flat fill can bypass engine or be disabled");
        check(outputs.size() == 1 && known_unchanged(input, mask, outputs[0]),
              "flat/engine fill preserves known pixels");
        if (enabled) check(outputs.size() == 1 && outputs[0] == flat(), "flat RGB reconstructed exactly");
    }
}
void equal_luma_colour_texture_is_not_flat() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(0), e.fn());
    const auto mask = bar_mask();
    image background(W * H * 3);
    for (int i = 0; i < W * H; i++) {
        background[i * 3] = i % 2 ? 255 : 0;
        background[i * 3 + 1] = i % 2 ? 0 : 130;
    }
    sequence outputs;
    feed(s, burn(background, mask), &mask, outputs);
    flush(s, outputs);
    check(e.calls == 1 && s.counters().px_flat == 0, "equal-luma red/green texture reaches engine");
    check(outputs.size() == 1 && known_unchanged(background, mask, outputs[0]),
          "chroma fallback preserves known pixels");
}
void moderate_change_blends_with_the_previous_fill() {
    engine e;
    e.alternate = true;
    inpaint_temporal::stream s(W, H, with_window(1), e.fn());
    const auto mask = bar_mask();
    sequence outputs;
    for (int t = 0; t < 2; t++) feed(s, burn(textured(0, 5 * t), mask), &mask, outputs);
    flush(s, outputs);
    const int v = outputs.size() == 2 ? outputs[1][(45 * W + 30) * 3] : -1;
    // 200 blended with 100, weight 0.6 * (1 - (5-3)/(12-3)) = 0.467.
    check(e.calls == 2 && v >= 150 && v <= 156, "moderate change blends engine output with previous fill");
    check(s.counters().px_blended == 12 * 56, "all second-frame engine pixels are blended");
    check(outputs.size() == 2 && known_unchanged(textured(0, 5), mask, outputs[1]), "blend preserves known pixels");
}
void partly_flat_background_mixes_flat_fill_and_engine() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(0), e.fn());
    image background = textured(0), mask(W * H, 0);
    const auto plain = flat();
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W / 2; x++)
            for (int c = 0; c < 3; c++) background[(y * W + x) * 3 + c] = plain[(y * W + x) * 3 + c];
    for (int y = 40; y < 52; y++)
        for (int x = 8; x < 184; x++) mask[y * W + x] = 255;
    sequence outputs;
    feed(s, burn(background, mask), &mask, outputs);
    flush(s, outputs);
    check(outputs.size() == 1 && e.calls == 1, "partly flat image uses one engine call");
    if (outputs.empty()) return;
    const uint8_t * left = &outputs[0][(45 * W + 10) * 3];
    check(std::abs(left[0] - 40) <= 2 && std::abs(left[1] - 140) <= 2 && std::abs(left[2] - 60) <= 2,
          "flat strip interpolates RGB surroundings");
    check(outputs[0][(45 * W + 180) * 3] == 200, "textured strip retains engine output");
    check(s.counters().px_flat > 0 && s.counters().px_engine > 0, "partial fill counts both sources");
    check(known_unchanged(background, mask, outputs[0]), "partial composition preserves known pixels");
}

void changed_masks_and_thresholds() {
    engine e;
    e.save_masks = true;
    auto options = with_window(1);
    options.flat_fill = false;
    inpaint_temporal::stream s(W, H, options, e.fn());
    image first(W * H, 0), second(W * H, 0);
    const int at = 45 * W + 30;
    first[at - 1] = 127;
    first[at] = 128;
    first[at + 2] = 255;
    second = first;
    second[at + 1] = 255; // same bounding box, different contents
    sequence outputs;
    const auto input = textured(0);
    feed(s, burn(input, first), &first, outputs);
    feed(s, burn(input, second), &second, outputs);
    // Move the third mask away instead of offering a clean future donor.
    image shifted(W * H, 0);
    shifted[10 * W + 150] = 255;
    feed(s, burn(input, shifted), &shifted, outputs);
    flush(s, outputs);
    check(e.calls == 3 && e.binary_masks && e.masks.size() == 3, "changed masks get fresh binary engine masks");
    if (e.masks.size() == 3)
        check(e.masks[0][at - 1] == 0 && e.masks[0][at] == 255 && e.masks[0][at + 1] == 0 &&
                  e.masks[0][at + 2] == 255 && e.masks[1][at + 1] == 255 && e.masks[2][at] == 0 &&
                  e.masks[2][at + 2] == 0 && e.masks[2][10 * W + 150] == 255,
              "127 stays known, 128/255 fill, moved masks leave no stale engine pixels");
    check(outputs.size() == 3 && known_unchanged(input, first, outputs[0]) &&
              known_unchanged(input, second, outputs[1]) && known_unchanged(input, shifted, outputs[2]),
          "thresholds and changing bounds preserve known pixels");
}
void zero_window_is_independent() {
    engine e;
    auto options = with_window(0);
    options.flat_fill = false;
    inpaint_temporal::stream s(W, H, options, e.fn());
    const auto mask = bar_mask();
    sequence outputs;
    for (int i = 0; i < 4; i++) feed(s, burn(flat(), mask), &mask, outputs);
    const image empty(W * H, 0);
    feed(s, flat(), &empty, outputs);
    flush(s, outputs);
    const auto & st = s.counters();
    check(outputs.size() == 5 && outputs.back() == flat() && e.calls == 4 && st.px_reused == 0 &&
              st.px_propagated == 0 && st.px_flat == 0 && st.px_blended == 0,
          "zero window with flat fill off stays independent; empty mask passes through");
}

void window_and_backpressure_bound_frame_payloads() {
    engine e;
    auto options = with_window(120);
    options.max_buffer_bytes = size_t(W) * H * 42;
    inpaint_temporal::stream bounded(W, H, options, e.fn());
    check(bounded.window() == 3, "mask/output/anchor reservation reduces window to three");
    options.max_buffer_bytes--;
    inpaint_temporal::stream smaller(W, H, options, e.fn());
    check(smaller.window() == 2, "one byte below full reservation reduces window");

    options.max_buffer_bytes = size_t(W) * H * 18;
    inpaint_temporal::stream s(W, H, options, e.fn());
    const auto rgb = textured(0);
    const image mask(W * H, 255);
    std::string error;
    image out(W * H * 3);
    check(s.window() == 0 && s.push(rgb.data(), mask.data(), error), "minimum payload budget admits full mask");
    check(!s.push(rgb.data(), mask.data(), error) && error.find("drain ready output") != std::string::npos &&
              e.calls == 1,
          "backpressure consumes no input and retains engine count");
    check(s.pop(out.data()) && std::all_of(out.begin(), out.end(), [](uint8_t v) { return v == 200; }),
          "backpressure retains completed output");
    check(s.push(rgb.data(), mask.data(), error) && error.empty() && e.calls == 2,
          "drain/retry consumes blocked input exactly once");
    check(s.finish(error) && s.pop(out.data()) && !s.pop(out.data()) && s.counters().frames == 2,
          "bounded stream retains exact count");
    options.max_buffer_bytes--;
    inpaint_temporal::stream impossible(W, H, options, e.fn());
    check(!impossible.push(rgb.data(), mask.data(), error) && error.find("budget too small") != std::string::npos,
          "impossible budget rejects before processing");
    check(!impossible.finish(error) && e.calls == 2, "impossible budget stays failed without engine calls");
}

void lifecycle_and_short_clips() {
    engine e;
    std::string error;
    image out(W * H * 3);
    inpaint_temporal::stream empty(W, H, with_window(3), e.fn());
    check(empty.finish(error) && empty.finish(error) && !empty.pop(out.data()), "empty finish is idempotent");
    check(!empty.push(textured(0).data(), nullptr, error) && error == "stream already finished",
          "push after finish rejects");
    inpaint_temporal::stream short_clip(W, H, with_window(3), e.fn());
    sequence outputs;
    feed(short_clip, textured(0), nullptr, outputs);
    feed(short_clip, textured(1), nullptr, outputs);
    check(outputs.empty(), "short clip waits for EOF");
    flush(short_clip, outputs);
    check(outputs == sequence{ textured(0), textured(1) } && short_clip.finish(error), "EOF emits short clip once");
    inpaint_temporal::stream invalid(0, H, with_window(0), e.fn());
    check(!invalid.finish(error) && error == "invalid stream dimensions", "zero dimensions reject cleanly");
    inpaint_temporal::stream null_input(W, H, with_window(0), e.fn());
    check(!null_input.push(nullptr, nullptr, error) && error == "null stream RGB frame", "null input does not consume");
    check(null_input.push(textured(0).data(), nullptr, error) && null_input.pop(out.data()),
          "valid input can follow null rejection");
    // finish may be called while the first processed frame is still ready.
    inpaint_temporal::stream pending(W, H, with_window(1), e.fn());
    check(pending.push(textured(0).data(), nullptr, error) && pending.push(textured(1).data(), nullptr, error) &&
              pending.finish(error),
          "finish handles an already-ready output and the remaining tail");
    check(pending.pop(out.data()) && out == textured(0) && pending.pop(out.data()) && out == textured(1) &&
              !pending.pop(out.data()),
          "finish preserves pending output ordering");
}

void engine_errors_stop_the_stream_and_preserve_prefix() {
    const auto mask = bar_mask();
    std::string error;
    image out(W * H * 3);
    {
        engine e;
        e.fail = true;
        inpaint_temporal::stream s(W, H, with_window(0), e.fn());
        check(!s.push(burn(textured(0), mask).data(), mask.data(), error) && error == "fake engine failure",
              "push reports engine error");
        check(!s.push(textured(0).data(), nullptr, error) && !s.finish(error) && !s.pop(out.data()),
              "engine failure is terminal");
    }
    {
        engine e;
        e.fail_at = 2;
        inpaint_temporal::stream s(W, H, with_window(3), e.fn());
        sequence outputs;
        for (int t = 0; t < 3; t++) feed(s, burn(textured(4 * t), mask), &mask, outputs);
        check(outputs.empty(), "delayed failure has no output before EOF");
        check(!s.finish(error) && error == "fake engine failure", "finish reports delayed failure");
        check(s.pop(out.data()) && out[(45 * W + 30) * 3] == 200 && !s.pop(out.data()),
              "completed EOF prefix stays poppable after later failure");
        check(!s.finish(error) && !s.push(textured(0).data(), nullptr, error) && e.calls == 2,
              "later calls cannot retry failed engine frame");
    }
}

void tiny_border_and_full_masks() {
    const std::pair<int, int> cases[] = { { 1, 1 }, { 2, 3 }, { 7, 7 }, { W, H } };
    for (const auto & dims : cases) {
        for (bool full : { false, true }) {
            engine e;
            e.width = dims.first;
            e.height = dims.second;
            const size_t pixels = size_t(e.width) * e.height;
            image rgb(pixels * 3, 73), mask(pixels, full ? 255 : 0);
            mask[0] = 255;
            inpaint_temporal::stream s(e.width, e.height, with_window(1), e.fn());
            sequence outputs;
            feed(s, burn(rgb, mask), &mask, outputs, e.width, e.height);
            flush(s, outputs, e.width, e.height);
            check(outputs.size() == 1 && known_unchanged(rgb, mask, outputs[0]),
                  "corner/tiny/full masks preserve known bytes");
            if (full) check(e.calls == 1, "full mask without usable ring reaches engine");
        }
    }
}

void harmonic_iteration_budget_is_a_ceiling() {
    engine e;
    e.width = e.height = 1026;
    const size_t pixels = size_t(e.width) * e.height;
    image rgb(pixels * 3), mask(pixels, 0);
    for (int y = 0; y < e.height; y++)
        for (int x = 0; x < e.width; x++) {
            const size_t i = size_t(y) * e.width + x;
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = y < e.height / 2 ? 100 : 108;
            if (x > 0 && x + 1 < e.width && y > 0 && y + 1 < e.height) mask[i] = 255;
        }
    inpaint_temporal::stream s(e.width, e.height, with_window(0), e.fn());
    sequence outputs;
    feed(s, burn(rgb, mask), &mask, outputs, e.width, e.height);
    flush(s, outputs, e.width, e.height);
    check(s.counters().harmonic_updates > 0 && s.counters().harmonic_updates <= 20000000,
          "large mask cannot force twenty over-budget SOR sweeps");
    check(outputs.size() == 1 && known_unchanged(rgb, mask, outputs[0]), "budgeted fill preserves boundary pixels");
}
} // namespace

static int crispembed_test_main() {
    order_and_delay();
    propagation_from_past_and_future();
    nearest_donor_and_window_boundary();
    static_span_reuse_and_small_noise();
    cumulative_drift_refreshes_the_fill();
    moving_texture_cannot_cancel_inside_cells();
    grain_on_a_still_shot_still_propagates();
    moving_background_calls_the_engine();
    flat_background_needs_no_engine();
    equal_luma_colour_texture_is_not_flat();
    moderate_change_blends_with_the_previous_fill();
    partly_flat_background_mixes_flat_fill_and_engine();
    changed_masks_and_thresholds();
    zero_window_is_independent();
    window_and_backpressure_bound_frame_payloads();
    lifecycle_and_short_clips();
    engine_errors_stop_the_stream_and_preserve_prefix();
    tiny_border_and_full_masks();
    harmonic_iteration_budget_is_a_ceiling();
    std::printf("inpaint temporal contracts: %d/%d passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
