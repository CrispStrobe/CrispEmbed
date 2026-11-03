// Model-free contracts for src/inpaint_temporal: frame order and delay,
// propagation, static-span reuse, flat fill and engine errors, with a counting
// fake engine that paints the hole a fixed colour.
#include "inpaint_temporal.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
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

struct engine {
    int calls = 0;
    bool fail = false;
    bool alternate = false; // paint 100 and 200 on alternate calls (blend test)
    inpaint_temporal::fill_fn fn() {
        return [this](const uint8_t * rgb, const uint8_t * mask, uint8_t * out, std::string & error) {
            ++calls;
            if (fail) {
                error = "fake engine failure";
                return false;
            }
            for (int i = 0; i < W * H; i++)
                for (int c = 0; c < 3; c++)
                    out[i * 3 + c] = mask[i] ? uint8_t(alternate && calls % 2 ? 100 : 200) : rgb[i * 3 + c];
            return true;
        };
    }
};

// Textured background, shifted by `shift` px and brightened by `offset`: a 12 px checkerboard (coarse enough
// for the cell means of the static test to see motion) with fine noise on top.
std::vector<uint8_t> textured(int shift, int offset = 0) {
    std::vector<uint8_t> rgb(W * H * 3);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int c = 0; c < 3; c++)
                rgb[(y * W + x) * 3 + c] = uint8_t(((x + shift) / 12 + y / 12) % 2 * 100 +
                                                   ((x + shift) * 37 + y * 101 + c * 53) % 40 + 50 + offset);
    return rgb;
}

std::vector<uint8_t> flat() {
    std::vector<uint8_t> rgb(W * H * 3);
    for (int i = 0; i < W * H; i++) {
        rgb[i * 3] = 40;
        rgb[i * 3 + 1] = 140;
        rgb[i * 3 + 2] = 60;
    }
    return rgb;
}

// A subtitle bar: mask and white "text" burned into the frame.
std::vector<uint8_t> bar_mask() {
    std::vector<uint8_t> mask(W * H, 0);
    for (int y = 40; y < 52; y++)
        for (int x = 20; x < 76; x++) mask[y * W + x] = 255;
    return mask;
}

std::vector<uint8_t> burn(std::vector<uint8_t> rgb, const std::vector<uint8_t> & mask) {
    for (int i = 0; i < W * H; i++)
        if (mask[i])
            for (int c = 0; c < 3; c++) rgb[i * 3 + c] = 255;
    return rgb;
}

inpaint_temporal::settings with_window(int window) {
    inpaint_temporal::settings s;
    s.window = window;
    return s;
}

void order_and_delay() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(2), e.fn());
    std::string error;
    std::vector<uint8_t> out(W * H * 3);
    for (int t = 0; t < 2; t++) check(s.push(textured(t).data(), nullptr, error), "push");
    check(!s.pop(out.data()), "no output before the look-ahead is full");
    check(s.push(textured(2).data(), nullptr, error) && s.pop(out.data()) && out == textured(0),
          "first frame out after window frames of look-ahead");
    check(s.finish(error), "finish");
    for (int t = 1; t < 3; t++) check(s.pop(out.data()) && out == textured(t), "frames out in order");
    check(!s.pop(out.data()) && e.calls == 0 && s.counters().frames == 3, "no engine call without a mask");
}

void propagation_from_a_clean_frame() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(3), e.fn());
    std::string error;
    const auto mask = bar_mask();
    check(s.push(textured(0).data(), nullptr, error), "clean frame");
    for (int t = 0; t < 3; t++) check(s.push(burn(textured(0), mask).data(), mask.data(), error), "masked frame");
    check(s.finish(error), "finish");
    std::vector<uint8_t> out(W * H * 3);
    bool exact = s.pop(out.data());
    for (int t = 0; t < 3; t++) exact = exact && s.pop(out.data()) && out == textured(0);
    check(exact, "static background copied exactly from the frame without a subtitle");
    check(e.calls == 0 && s.counters().px_propagated == 3 * 12 * 56, "propagation needs no engine call");
}

void static_span_reuse() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(2), e.fn());
    std::string error;
    const auto mask = bar_mask();
    for (int t = 0; t < 5; t++) check(s.push(burn(textured(0), mask).data(), mask.data(), error), "masked frame");
    check(s.finish(error), "finish");
    std::vector<uint8_t> first(W * H * 3), out(W * H * 3);
    bool same = s.pop(first.data());
    for (int t = 1; t < 5; t++) same = same && s.pop(out.data()) && out == first;
    check(same && first[(45 * W + 30) * 3] == 200, "one engine fill reused for the whole static span");
    check(e.calls == 1 && s.counters().px_reused == 4 * 12 * 56, "reuse needs no further engine calls");
}

void moving_background_calls_the_engine() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(2), e.fn());
    std::string error;
    const auto mask = bar_mask();
    for (int t = 0; t < 4; t++) check(s.push(burn(textured(4 * t), mask).data(), mask.data(), error), "masked frame");
    check(s.finish(error) && e.calls == 4, "every frame of a moving shot is filled by the engine");
}

void flat_background_needs_no_engine() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(0), e.fn());
    std::string error;
    const auto mask = bar_mask();
    std::vector<uint8_t> out(W * H * 3);
    check(s.push(burn(flat(), mask).data(), mask.data(), error) && s.finish(error) && s.pop(out.data()), "flat frame");
    check(e.calls == 0 && out == flat(), "flat surroundings are interpolated exactly, without the engine");

    inpaint_temporal::settings off = with_window(0);
    off.flat_fill = false;
    inpaint_temporal::stream s2(W, H, off, e.fn());
    check(s2.push(burn(flat(), mask).data(), mask.data(), error) && s2.finish(error) && e.calls == 1,
          "flat fill can be switched off");
}

void moderate_change_blends_with_the_previous_fill() {
    engine e;
    e.alternate = true;
    inpaint_temporal::stream s(W, H, with_window(1), e.fn());
    std::string error;
    const auto mask = bar_mask();
    // +5 levels per frame: above the static threshold (3), below the blend threshold (12).
    for (int t = 0; t < 2; t++)
        check(s.push(burn(textured(0, 5 * t), mask).data(), mask.data(), error), "masked frame");
    check(s.finish(error), "finish");
    std::vector<uint8_t> first(W * H * 3), second(W * H * 3);
    check(s.pop(first.data()) && s.pop(second.data()), "two frames");
    const int v = second[(45 * W + 30) * 3];
    // Engine 200 blended with the previous 100: weight 0.6 * (1 - (5 - 3) / (12 - 3)) = 0.467 -> 153.
    check(e.calls == 2 && first[(45 * W + 30) * 3] == 100 && v >= 150 && v <= 156,
          "engine fill blended with the previous output on a moderate change");
    check(s.counters().px_blended == 12 * 56, "every engine pixel of the second frame is blended");
}

void partly_flat_background_mixes_flat_fill_and_engine() {
    engine e;
    inpaint_temporal::stream s(W, H, with_window(0), e.fn());
    std::string error;
    // Flat left half, textured right half, and a bar across both (three 64 px strips).
    std::vector<uint8_t> background = textured(0), mask(W * H, 0);
    const auto plain = flat();
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W / 2; x++)
            for (int c = 0; c < 3; c++) background[(y * W + x) * 3 + c] = plain[(y * W + x) * 3 + c];
    for (int y = 40; y < 52; y++)
        for (int x = 8; x < 184; x++) mask[y * W + x] = 255;
    std::vector<uint8_t> out(W * H * 3);
    check(s.push(burn(background, mask).data(), mask.data(), error) && s.finish(error) && s.pop(out.data()),
          "partly flat frame");
    const uint8_t * left = &out[(45 * W + 10) * 3];
    check(e.calls == 1 && std::abs(left[0] - 40) <= 2 && std::abs(left[1] - 140) <= 2 && std::abs(left[2] - 60) <= 2,
          "flat strip filled from its surroundings");
    check(out[(45 * W + 180) * 3] == 200, "textured strip filled by the engine");
    check(s.counters().px_flat > 0 && s.counters().px_engine > 0, "both counted");
}

void window_fits_the_memory_budget() {
    engine e;
    inpaint_temporal::settings options = with_window(15);
    options.max_buffer_bytes = size_t(W) * H * 3 * 7; // room for 7 frames: window 3
    inpaint_temporal::stream s(W, H, options, e.fn());
    check(s.window() == 3, "window reduced to fit the buffer budget");
}

void engine_errors_stop_the_stream() {
    engine e;
    e.fail = true;
    inpaint_temporal::stream s(W, H, with_window(0), e.fn());
    std::string error;
    const auto mask = bar_mask();
    check(!s.push(burn(textured(0), mask).data(), mask.data(), error) && error == "fake engine failure",
          "engine error is reported by push");
    check(!s.push(textured(0).data(), nullptr, error), "the stream stays failed");
}

} // namespace

int main() {
    order_and_delay();
    propagation_from_a_clean_frame();
    static_span_reuse();
    moving_background_calls_the_engine();
    flat_background_needs_no_engine();
    moderate_change_blends_with_the_previous_fill();
    partly_flat_background_mixes_flat_fill_and_engine();
    window_fits_the_memory_budget();
    engine_errors_stop_the_stream();
    std::printf("inpaint temporal contracts: %d/%d passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
