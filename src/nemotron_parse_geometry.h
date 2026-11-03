// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
//
// C++ adaptation of the image geometry and packed-patch contract in
// nvidia/Nemotron-3.5-Parse@4035ffb1d8f8bafcdb867dfaf6f7514f101464f9,
// hf_nemotron_parse_processor.py. See licenses/nemotron-parse-LICENSE and
// licenses/nemotron-parse-NOTICE for origin and modifications.
//
// Bounded prototype: input pixels MUST already have the size selected below.
// This file does not implement or claim Pillow bilinear resize equivalence,
// learned position interpolation, a neural backend, or OCR support.
#pragma once

#include "core/env_gate.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace nemotron_parse_geometry {

constexpr const char * gate = "CRISPEMBED_NEMOTRON_PARSE_GEOMETRY";
constexpr int max_height = 2048;
constexpr int max_width = 1664;
constexpr int patch_size = 16;
constexpr int width_alignment = 64;
constexpr size_t patch_elements = 3 * patch_size * patch_size;
constexpr size_t max_pixel_bytes = size_t(max_height) * max_width * 3;

struct image_plan {
    int original_height = 0;
    int original_width = 0;
    int content_height = 0;
    int content_width = 0;
    int padded_height = 0;
    int padded_width = 0;
    int patch_rows = 0;
    int patch_columns = 0;

    bool resized() const { return content_height != original_height || content_width != original_width; }
    size_t patch_count() const { return size_t(patch_rows) * patch_columns; }
};

// Mirror Python's ordered double-precision operations and int truncation.
// Algebraic integer-ratio simplifications can select a different pixel size.
// No upscaling. Extreme aspect ratios that produce a zero dimension are
// rejected, matching the original Pillow resize failure rather than clamped.
inline bool make_plan(int height, int width, image_plan & out) {
    out = {};
    if (height <= 0 || width <= 0) return false;
    const double aspect_ratio = double(width) / double(height);
    int resized_height = height;
    int resized_width = width;
    if (height > max_height) {
        resized_height = max_height;
        resized_width = int(double(resized_height) * aspect_ratio);
    }
    if (resized_width > max_width) {
        resized_width = max_width;
        resized_height = int(double(resized_width) / aspect_ratio);
    }
    if (resized_height <= 0 || resized_width <= 0 || resized_height > max_height || resized_width > max_width)
        return false;
    out.original_height = height;
    out.original_width = width;
    out.content_height = resized_height;
    out.content_width = resized_width;
    out.padded_height = ((resized_height + patch_size - 1) / patch_size) * patch_size;
    out.padded_width = ((resized_width + width_alignment - 1) / width_alignment) * width_alignment;
    out.patch_rows = out.padded_height / patch_size;
    out.patch_columns = out.padded_width / patch_size;
    return true;
}

inline bool valid_plan(const image_plan & plan) {
    image_plan expected;
    return make_plan(plan.original_height, plan.original_width, expected) &&
           plan.content_height == expected.content_height && plan.content_width == expected.content_width &&
           plan.padded_height == expected.padded_height && plan.padded_width == expected.padded_width &&
           plan.patch_rows == expected.patch_rows && plan.patch_columns == expected.patch_columns;
}

struct packed_image {
    image_plan geometry;
    // [patch row, patch column, channel, pixel row, pixel column]; flat F32.
    std::vector<float> patches;
};

// Accept tightly packed RGB8 or grayscale8 pixels after the ORIGINAL resize.
// Grayscale is replicated to R/G/B, equivalent to PIL L.convert("RGB").
// White padding is appended only at the bottom/right; it is normalized too.
// A packed row contains 768 floats. Channel order is NOT interleaved in a row.
inline bool pack_resized(const uint8_t * pixels, size_t byte_count, int height, int width, int channels,
                         const image_plan & plan, packed_image & out) {
    out = {};
    if (!core_env::on(gate) || !valid_plan(plan) || !pixels || (channels != 1 && channels != 3) ||
        height != plan.content_height || width != plan.content_width ||
        byte_count != size_t(height) * size_t(width) * size_t(channels) || byte_count > max_pixel_bytes ||
        sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
        return false;

    // Preserve the original float32 operation order: ToTensor divides by 255,
    // then Normalize subtracts a float32 mean and divides by a float32 std.
    const std::array<float, 3> mean = { 0.48145466f, 0.4578275f, 0.40821073f };
    const std::array<float, 3> stddev = { 0.26862954f, 0.26130258f, 0.27577711f };
    std::array<std::array<float, 256>, 3> normalized;
    for (int channel = 0; channel < 3; ++channel) {
        for (int value = 0; value < 256; ++value) {
            const float scaled = float(value) / 255.0f;
            const float centered = scaled - mean[channel];
            normalized[channel][value] = centered / stddev[channel];
        }
    }

    packed_image result;
    result.geometry = plan;
    result.patches.resize(plan.patch_count() * patch_elements);
    size_t cursor = 0;
    for (int row = 0; row < plan.patch_rows; ++row) {
        for (int column = 0; column < plan.patch_columns; ++column) {
            for (int channel = 0; channel < 3; ++channel) {
                for (int dy = 0; dy < patch_size; ++dy) {
                    const int y = row * patch_size + dy;
                    for (int dx = 0; dx < patch_size; ++dx) {
                        const int x = column * patch_size + dx;
                        uint8_t value = 255;
                        if (y < height && x < width) {
                            const size_t offset = (size_t(y) * width + x) * channels;
                            value = pixels[offset + (channels == 1 ? 0 : channel)];
                        }
                        result.patches[cursor++] = normalized[channel][value];
                    }
                }
            }
        }
    }
    out = std::move(result);
    return true;
}

} // namespace nemotron_parse_geometry
