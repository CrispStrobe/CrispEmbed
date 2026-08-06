#include "tesseract_pageseg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tesseract_pageseg {

struct blob_box {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0, area = 0;
};

static std::vector<ocr_detect::text_box> segment_components(const uint8_t * gray, int width, int height,
                                                            int threshold) {
    std::vector<uint8_t> seen((size_t)width * height, 0);
    std::vector<blob_box> blobs;
    std::vector<int> stack;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t seed = (size_t)y * width + x;
            if (seen[seed] || gray[seed] >= threshold) continue;
            seen[seed] = 1;
            stack.clear();
            stack.push_back((int)seed);
            blob_box box{ x, y, x, y, 0 };
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                const int py = p / width, px = p - py * width;
                ++box.area;
                box.x0 = std::min(box.x0, px);
                box.x1 = std::max(box.x1, px);
                box.y0 = std::min(box.y0, py);
                box.y1 = std::max(box.y1, py);
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx = px + dx, ny = py + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        const size_t np = (size_t)ny * width + nx;
                        if (!seen[np] && gray[np] < threshold) {
                            seen[np] = 1;
                            stack.push_back((int)np);
                        }
                    }
                }
            }
            if (box.area >= 2 && box.y1 - box.y0 + 1 >= 3) blobs.push_back(box);
        }
    }
    if (blobs.empty()) return {};
    std::vector<float> heights;
    for (const auto & b : blobs) heights.push_back((float)(b.y1 - b.y0 + 1));
    std::sort(heights.begin(), heights.end());
    const float tolerance = std::max(3.0f, heights[heights.size() / 2] * 0.8f);
    struct row {
        int x0, y0, x1, y1, count, area;
        float baseline;
    };
    std::vector<row> rows;
    std::sort(blobs.begin(), blobs.end(),
              [](const blob_box & a, const blob_box & b) { return a.y1 != b.y1 ? a.y1 < b.y1 : a.x0 < b.x0; });
    for (const auto & b : blobs) {
        int best = -1;
        float distance = tolerance;
        for (int i = 0; i < (int)rows.size(); ++i) {
            const float d = std::fabs(rows[i].baseline - (float)b.y1);
            if (d <= distance) {
                distance = d;
                best = i;
            }
        }
        if (best < 0)
            rows.push_back({ b.x0, b.y0, b.x1, b.y1, 1, b.area, (float)b.y1 });
        else {
            row & r = rows[best];
            r.x0 = std::min(r.x0, b.x0);
            r.y0 = std::min(r.y0, b.y0);
            r.x1 = std::max(r.x1, b.x1);
            r.y1 = std::max(r.y1, b.y1);
            r.area += b.area;
            r.baseline = (r.baseline * r.count + b.y1) / (r.count + 1.0f);
            ++r.count;
        }
    }
    std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) { return a.y0 < b.y0; });
    // Tesseract's make_initial_textrows() reassociates detached small blobs
    // with an established row using line-size spacing. Do the same here so
    // quotes and punctuation cannot become standalone text lines.
    const float reassociate_distance = std::max(tolerance, 10.0f);
    std::vector<row> reassociated;
    for (const auto & candidate : rows) {
        if (candidate.count >= 20) {
            reassociated.push_back(candidate);
            continue;
        }
        int best = -1;
        float best_distance = reassociate_distance;
        for (int i = 0; i < (int)reassociated.size(); ++i) {
            if (reassociated[i].count < 20) continue;
            const float distance = std::fabs(reassociated[i].baseline - candidate.baseline);
            if (distance <= best_distance) {
                best_distance = distance;
                best = i;
            }
        }
        if (best < 0) {
            reassociated.push_back(candidate);
        } else {
            row & target = reassociated[best];
            target.x0 = std::min(target.x0, candidate.x0);
            target.y0 = std::min(target.y0, candidate.y0);
            target.x1 = std::max(target.x1, candidate.x1);
            target.y1 = std::max(target.y1, candidate.y1);
            target.area += candidate.area;
            target.baseline = (target.baseline * target.count + candidate.baseline * candidate.count) /
                              (target.count + candidate.count);
            target.count += candidate.count;
        }
    }
    rows.swap(reassociated);
    std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) { return a.y0 < b.y0; });
    if (std::getenv("CRISPEMBED_TESSERACT_COMPONENT_DEBUG")) {
        for (const auto & r : rows) {
            std::fprintf(stderr, "component row x=%d..%d y=%d..%d count=%d base=%.1f\n", r.x0, r.x1, r.y0, r.y1,
                         r.count, r.baseline);
        }
    }
    std::vector<ocr_detect::text_box> out;
    // Tesseract's filter_blobs moves very small connected components to its
    // noise/small lists before row assignment.  Apply the equivalent row-level
    // guard here: a genuine text row has at least an x-height-sized vertical
    // span and substantially more ink than isolated scan noise.  Keep this
    // conservative because punctuation and detached dots are valid members of
    // an otherwise well-supported row.
    const int min_row_height = std::max(6, (int)std::lround(heights[heights.size() / 2] * 0.75f));
    for (const auto & r : rows) {
        if (r.count < 2 || r.x1 - r.x0 < std::max(8, width / 50) || r.y1 - r.y0 + 1 < min_row_height ||
            r.area < min_row_height * 2)
            continue;
        ocr_detect::text_box b{};
        b.x = (float)std::max(0, r.x0 - 2);
        b.y = (float)std::max(0, r.y0 - 2);
        b.w = (float)std::min(width - (int)b.x, r.x1 - r.x0 + 5);
        b.h = (float)std::min(height - (int)b.y, r.y1 - r.y0 + 5);
        b.score = 1.0f;
        out.push_back(b);
    }
    return out;
}

// Separator rejection (pageseg quality lane, 2026-08-06). The classical row
// bands include horizontal RULES and rule-bearing ornaments, and the LSTM
// turns each into a flood of garbage characters — on the Fraktur
// Reichsgesetzblatt page the two masthead rules alone emitted ~350 junk
// characters (+26% over official's char total) and were the bulk of the
// classical route's CER 0.412 vs the dbnet route's 0.235. Real Tesseract's
// layout analysis classifies and drops separators before recognition; the
// equivalent classical test, chosen from MEASURED crop statistics on that
// page (a first cut used max-scanline ink >= 80% and missed both rules —
// they are dashed/faded on the 1899 scan, max scanline only 33-35%): a
// rule's COLUMN COVERAGE (fraction of box columns with any ink) is
// 0.99-1.00 even when dashed, while every text row measured <= 0.76 (the
// dense blackletter masthead is the worst case). Guards: box height under
// 0.6x the median row height (rules 21-22 px vs median ~44; the smallest
// real text row was 29) and aspect >= 8, so a short wide dense-justified
// line cannot trip it. Applied by BOTH the projection and the
// component-legacy segmenters. CRISPEMBED_TESSERACT_PAGESEG_KEEP_RULES=1
// restores the unfiltered rows (regression lever).
static void reject_separator_rows(std::vector<ocr_detect::text_box> & out, const uint8_t * gray, int width, int height,
                                  int threshold) {
    if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_KEEP_RULES")) return;
    if (out.size() < 3) return;
    std::vector<float> heights;
    heights.reserve(out.size());
    for (const auto & b : out) heights.push_back(b.h);
    std::sort(heights.begin(), heights.end());
    const float median_h = heights[heights.size() / 2];
    std::vector<ocr_detect::text_box> kept;
    kept.reserve(out.size());
    for (const auto & b : out) {
        const int bx0 = std::max(0, (int)b.x), by0 = std::max(0, (int)b.y);
        const int bx1 = std::min(width - 1, bx0 + (int)b.w - 1);
        const int by1 = std::min(height - 1, by0 + (int)b.h - 1);
        const int bw = bx1 - bx0 + 1;
        const int bh = by1 - by0 + 1;
        bool separator = false;
        if (bw >= 32 && bh > 0 && (float)bh <= 0.6f * median_h && bw >= 8 * bh) {
            int covered = 0;
            for (int x = bx0; x <= bx1; ++x) {
                for (int y = by0; y <= by1; ++y) {
                    if (gray[y * width + x] < threshold) {
                        ++covered;
                        break;
                    }
                }
            }
            separator = covered >= (bw * 9) / 10;
        }
        if (separator) {
            if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG"))
                std::fprintf(stderr, "tesseract_pageseg: separator dropped x=%d..%d y=%d..%d (%dx%d)\n", bx0, bx1, by0,
                             by1, bw, bh);
            continue;
        }
        kept.push_back(b);
    }
    out = std::move(kept);
}

std::vector<ocr_detect::text_box> segment_gray(const uint8_t * gray, int width, int height) {
    std::vector<ocr_detect::text_box> out;
    if (!gray || width <= 0 || height <= 0) return out;

    // Historical paper is usually light, but estimate a robust foreground
    // threshold from the image rather than assuming a fixed white background.
    uint64_t sum = 0;
    for (int i = 0; i < width * height; ++i) sum += gray[i];
    const int mean = (int)(sum / (uint64_t)(width * height));
    // A dark threshold is intentional: historical paper has broad gray
    // background variation, so a near-background threshold would classify the
    // paper itself as ink and collapse all rows into one band. Keep enough
    // dark pixels per row to prevent isolated antialiasing specks from
    // bridging adjacent printed lines.
    const int threshold = std::clamp(mean - 90, 30, 120);
    const int min_row_ink = std::max(4, width / 130);
    if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG")) {
        std::fprintf(stderr, "tesseract_pageseg: size=%dx%d mean=%d threshold=%d min_row_ink=%d\n", width, height, mean,
                     threshold, min_row_ink);
    }
    if (std::getenv("CRISPEMBED_TESSERACT_COMPONENT_PAGESEG")) {
        const auto component_rows = segment_components(gray, width, height, threshold);
        if (component_rows.size() >= 2) return component_rows;
    }
    std::vector<int> rows(height, 0);
    for (int y = 0; y < height; ++y) {
        int count = 0;
        for (int x = 0; x < width; ++x) count += gray[y * width + x] < threshold;
        rows[y] = count;
    }

    // Join glyph rows into text bands. A small vertical gap closes holes in
    // ascenders/descenders while a larger gap separates printed lines.
    const int max_gap = std::max(2, height / 180);
    int start = -1, last = -1, gap = 0;
    auto emit = [&](int y0, int y1) {
        if (y0 < 0 || y1 < y0) return;
        int x0 = width, x1 = -1;
        int ink = 0;
        for (int y = y0; y <= y1; ++y) {
            for (int x = 0; x < width; ++x) {
                if (gray[y * width + x] < threshold) {
                    x0 = std::min(x0, x);
                    x1 = std::max(x1, x);
                    ++ink;
                }
            }
        }
        if (x1 < x0 || ink < min_row_ink) return;
        ocr_detect::text_box b{};
        b.x = (float)std::max(0, x0 - 2);
        b.y = (float)std::max(0, y0 - 2);
        b.w = (float)std::min(width - (int)b.x, x1 - x0 + 5);
        b.h = (float)std::min(height - (int)b.y, y1 - y0 + 5);
        b.score = 1.0f;
        out.push_back(b);
    };
    for (int y = 0; y < height; ++y) {
        if (rows[y] >= min_row_ink) {
            if (start < 0) start = y;
            last = y;
            gap = 0;
        } else if (start >= 0 && ++gap > max_gap) {
            emit(start, last);
            start = last = -1;
            gap = 0;
        }
    }
    if (start >= 0) emit(start, last);

    // Row ink can bridge neighboring lines on degraded paper. Estimate the
    // normal band height from the lower quartile, then split only clear
    // multi-line outliers. This keeps isolated tall glyphs intact while
    // recovering the regular line pitch in merged paragraph bands.
    if (out.size() >= 3) {
        std::vector<float> heights;
        heights.reserve(out.size());
        for (const auto & b : out) heights.push_back(b.h);
        std::sort(heights.begin(), heights.end());
        const float base_height = std::max(8.0f, heights[(heights.size() - 1) / 4]);
        std::vector<ocr_detect::text_box> split;
        split.reserve(out.size());
        for (const auto & b : out) {
            const int parts = b.h > base_height * 1.5f ? std::max(1, (int)std::lround(b.h / base_height)) : 1;
            if (parts == 1) {
                split.push_back(b);
                continue;
            }
            const float y0 = b.y;
            const float y1 = b.y + b.h;
            std::vector<float> edges((size_t)parts + 1);
            edges.front() = y0;
            edges.back() = y1;
            for (int edge = 1; edge < parts; ++edge) {
                const int expected = (int)std::lround(y0 + (y1 - y0) * (float)edge / (float)parts);
                const int radius = std::max(2, (int)std::lround(base_height * 0.45f));
                const int lo = std::max((int)std::ceil(y0) + 1, expected - radius);
                const int hi = std::min((int)std::floor(y1) - 1, expected + radius);
                int best = expected;
                int best_ink = rows[std::clamp(expected, 0, height - 1)];
                for (int yy = lo; yy <= hi; ++yy) {
                    if (rows[yy] < best_ink) {
                        best = yy;
                        best_ink = rows[yy];
                    }
                }
                edges[(size_t)edge] = (float)best;
            }
            for (int part = 0; part < parts; ++part) {
                const float py0 = edges[(size_t)part];
                const float py1 = edges[(size_t)part + 1];
                const int sy0 = std::max(0, (int)std::lround(py0) + 2);
                const int sy1 = std::min(height - 1, (int)std::lround(py1) - 3);
                int sx0 = width, sx1 = -1;
                int loose_x0 = width, loose_x1 = -1;
                if (sy1 >= sy0) {
                    std::vector<int> column_ink((size_t)width, 0);
                    for (int yy = sy0; yy <= sy1; ++yy) {
                        for (int xx = 0; xx < width; ++xx) column_ink[(size_t)xx] += gray[yy * width + xx] < threshold;
                    }
                    for (int xx = 0; xx < width; ++xx) {
                        if (column_ink[(size_t)xx] >= 2) {
                            loose_x0 = std::min(loose_x0, xx);
                            loose_x1 = std::max(loose_x1, xx);
                        }
                        if (column_ink[(size_t)xx] >= 4) {
                            sx0 = std::min(sx0, xx);
                            sx1 = std::max(sx1, xx);
                        }
                    }
                }
                if (sx1 >= sx0 && loose_x1 >= loose_x0 && sx0 > b.x + 5.0f && sx1 - sx0 > b.h * 10.0f) {
                    sx0 = loose_x0;
                    sx1 = loose_x1;
                }
                ocr_detect::text_box piece = b;
                if (sx1 >= sx0 && sy1 >= sy0) {
                    piece.x = (float)std::max(0, sx0 - 2);
                    piece.y = (float)std::max(0, sy0 - 2);
                    piece.w = (float)std::min(width - (int)piece.x, sx1 - sx0 + 5);
                    piece.h = (float)std::min(height - (int)piece.y, sy1 - sy0 + 5);
                } else {
                    piece.y = py0;
                    piece.h = std::max(1.0f, py1 - py0);
                }
                split.push_back(piece);
            }
        }
        out = std::move(split);
    }

    reject_separator_rows(out, gray, width, height, threshold);
    return out;
}

static std::vector<ocr_detect::text_box> segment_gray_components_legacy(const uint8_t * gray, int width, int height) {
    std::vector<ocr_detect::text_box> out;
    if (!gray || width <= 0 || height <= 0) return out;
    uint64_t sum = 0;
    for (int i = 0; i < width * height; ++i) sum += gray[i];
    const int threshold = std::clamp((int)(sum / (uint64_t)(width * height)) - 130, 60, 120);
    std::vector<uint8_t> seen((size_t)width * height, 0);
    struct blob {
        int x0, y0, x1, y1, area;
    };
    std::vector<blob> blobs;
    std::vector<int> stack;
    stack.reserve(256);
    int box_pad = 3;
    if (const char * pad_env = std::getenv("CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD")) {
        // Keep the historical 3-pixel expansion by default. Official
        // Tesseract line boxes are often tighter, so expose the geometry
        // independently from recognizer crop padding for controlled parity
        // experiments and alternate scan resolutions.
        box_pad = std::clamp(std::atoi(pad_env), 0, 16);
    }
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t seed = (size_t)y * width + x;
            if (seen[seed] || gray[seed] >= threshold) continue;
            seen[seed] = 1;
            stack.clear();
            stack.push_back((int)seed);
            blob b{ x, y, x, y, 0 };
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                const int py = p / width, px = p - py * width;
                b.x0 = std::min(b.x0, px);
                b.x1 = std::max(b.x1, px);
                b.y0 = std::min(b.y0, py);
                b.y1 = std::max(b.y1, py);
                ++b.area;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy) continue;
                        const int nx = px + dx, ny = py + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        const size_t q = (size_t)ny * width + nx;
                        if (!seen[q] && gray[q] < threshold) {
                            seen[q] = 1;
                            stack.push_back((int)q);
                        }
                    }
            }
            const int bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
            if (b.area >= 5 && bh >= 3 && bh <= height / 3 && bw <= width / 2) blobs.push_back(b);
        }
    if (blobs.empty()) return out;
    std::vector<int> heights;
    heights.reserve(blobs.size());
    for (const auto & b : blobs) heights.push_back(b.y1 - b.y0 + 1);
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    const int median_h = std::max(3, heights[heights.size() / 2]);
    const int max_row_gap = std::max(3, (int)std::lround(median_h * 0.65f));
    std::sort(blobs.begin(), blobs.end(),
              [](const blob & a, const blob & b) { return a.y0 == b.y0 ? a.x0 < b.x0 : a.y0 < b.y0; });
    struct row {
        std::vector<blob> blobs;
        int y0 = 0, y1 = 0;
    };
    std::vector<row> rows;
    for (const auto & b : blobs) {
        if (!rows.empty() && b.y0 <= rows.back().y1 + max_row_gap) {
            rows.back().blobs.push_back(b);
            rows.back().y1 = std::max(rows.back().y1, b.y1);
        } else {
            rows.push_back({ { b }, b.y0, b.y1 });
        }
    }
    for (const auto & r : rows) {
        if (r.blobs.size() < 2) continue;
        int x0 = width, x1 = -1;
        for (const auto & b : r.blobs) {
            x0 = std::min(x0, b.x0);
            x1 = std::max(x1, b.x1);
        }
        if (x1 < x0) continue;
        ocr_detect::text_box box{};
        box.x = (float)std::max(0, x0 - box_pad);
        box.y = (float)std::max(0, r.y0 - box_pad);
        box.w = (float)std::min(width - (int)box.x, x1 - x0 + 2 * box_pad + 1);
        box.h = (float)std::min(height - (int)box.y, r.y1 - r.y0 + 2 * box_pad + 1);
        box.score = 1.0f;
        out.push_back(box);
    }
    if (std::getenv("CRISPEMBED_TESSERACT_LEGACY_ROW_DEBUG")) {
        std::fprintf(stderr, "legacy pageseg: threshold=%d blobs=%zu median_h=%d max_row_gap=%d rows=%zu\n", threshold,
                     blobs.size(), median_h, max_row_gap, rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto & r = rows[i];
            std::fprintf(stderr, "  row=%zu blobs=%zu y=%d..%d x=%d..%d out=%s\n", i, r.blobs.size(), r.y0, r.y1,
                         r.blobs.front().x0, r.blobs.back().x1, i < out.size() ? "yes" : "filtered");
        }
    }
    std::sort(out.begin(), out.end(),
              [](const auto & a, const auto & b) { return a.y == b.y ? a.x < b.x : a.y < b.y; });
    return out;
}

std::vector<ocr_detect::text_box> segment_gray_components(const uint8_t * gray, int width, int height) {
    const bool baseline_override = std::getenv("CRISPEMBED_TESSERACT_COMPONENT_BASELINE") != nullptr;
    const bool use_fallback = !baseline_override;
    if (use_fallback) {
        auto legacy = segment_gray_components_legacy(gray, width, height);
        if (legacy.size() >= 2) {
            // Same dark-ink threshold convention as segment_gray; the
            // legacy segmenter's own threshold is function-local.
            uint64_t s = 0;
            for (int i = 0; i < width * height; ++i) s += gray[i];
            const int sep_threshold = std::clamp((int)(s / (uint64_t)(width * height)) - 90, 30, 120);
            reject_separator_rows(legacy, gray, width, height, sep_threshold);
            if (legacy.size() >= 2) return legacy;
        }
        // Keep the measured legacy grouping as the default, but do not turn a
        // difficult page into an empty or single-row classical result. The
        // baseline matcher below is still experimental; use it only when the
        // legacy path is under-segmented, and leave an explicit baseline
        // override available for A/B comparisons.
    }
    std::vector<ocr_detect::text_box> out;
    if (!gray || width <= 0 || height <= 0) return out;
    uint64_t sum = 0;
    for (int i = 0; i < width * height; ++i) sum += gray[i];
    const int threshold = std::clamp((int)(sum / (uint64_t)(width * height)) - 130, 60, 120);
    std::vector<uint8_t> seen((size_t)width * height, 0);
    struct blob {
        int x0, y0, x1, y1, area;
    };
    std::vector<blob> blobs;
    std::vector<int> stack;
    stack.reserve(256);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t seed = (size_t)y * width + x;
            if (seen[seed] || gray[seed] >= threshold) continue;
            seen[seed] = 1;
            stack.clear();
            stack.push_back((int)seed);
            blob b{ x, y, x, y, 0 };
            while (!stack.empty()) {
                const int p = stack.back();
                stack.pop_back();
                const int py = p / width, px = p - py * width;
                b.x0 = std::min(b.x0, px);
                b.x1 = std::max(b.x1, px);
                b.y0 = std::min(b.y0, py);
                b.y1 = std::max(b.y1, py);
                ++b.area;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (!dx && !dy) continue;
                        const int nx = px + dx, ny = py + dy;
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        const size_t q = (size_t)ny * width + nx;
                        if (!seen[q] && gray[q] < threshold) {
                            seen[q] = 1;
                            stack.push_back((int)q);
                        }
                    }
            }
            const int bw = b.x1 - b.x0 + 1, bh = b.y1 - b.y0 + 1;
            if (b.area >= 5 && bh >= 3 && bh <= height / 3 && bw <= width / 2) blobs.push_back(b);
        }
    if (blobs.empty()) return out;

    // Tesseract's old textord path first makes blobs and then assigns them to
    // rows.  Do the row assignment by vertical bands, rather than by nearest
    // centre: Fraktur has tall ascenders/descenders and a nearest-centre
    // heuristic splits one printed line into several rows.
    std::vector<int> heights;
    heights.reserve(blobs.size());
    for (const auto & b : blobs) heights.push_back(b.y1 - b.y0 + 1);
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    const int median_h = std::max(3, heights[heights.size() / 2]);
    const int default_row_gap = std::max(3, (int)std::lround(median_h * 0.65f));
    int max_row_gap = default_row_gap;
    if (const char * gap_env = std::getenv("CRISPEMBED_TESSERACT_PAGESEG_ROW_GAP")) {
        // Experimental tuning knob: Tesseract's row fitter is baseline-driven;
        // this pixel gap is the portable approximation for scanned pages.
        max_row_gap = std::clamp(std::atoi(gap_env), 0, std::max(3, height / 20));
    }
    if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG"))
        std::fprintf(stderr, "tesseract_pageseg: components=%zu median_h=%d row_gap=%d\n", blobs.size(), median_h,
                     max_row_gap);
    // Assign by fitted baseline, not by transitive y-range overlap. A tall
    // ascender can overlap the next line; Tesseract's makerow path keeps it
    // with the row whose baseline/vertical overlap is the best match.
    struct row {
        int x0, y0, x1, y1, count, area;
        float baseline;
    };
    std::vector<row> rows;
    std::sort(blobs.begin(), blobs.end(),
              [](const blob & a, const blob & b) { return a.y1 == b.y1 ? a.x0 < b.x0 : a.y1 < b.y1; });
    float baseline_tolerance = std::max((float)max_row_gap, median_h * 1.5f);
    if (const char * tolerance_env = std::getenv("CRISPEMBED_TESSERACT_PAGESEG_BASELINE_TOLERANCE")) {
        // Experimental override for scripts/fonts with unusually detached
        // ascenders. The default deliberately keeps adjacent printed rows
        // separate while allowing normal Fraktur glyph-height variation.
        baseline_tolerance = std::clamp((float)std::atof(tolerance_env), 1.0f, (float)height / 8.0f);
    }
    for (const auto & b : blobs) {
        int best = -1;
        float best_distance = baseline_tolerance;
        for (int i = 0; i < (int)rows.size(); ++i) {
            const float distance = std::fabs(rows[i].baseline - (float)b.y1);
            if (distance <= best_distance) {
                best_distance = distance;
                best = i;
            }
        }
        if (best < 0) {
            rows.push_back({ b.x0, b.y0, b.x1, b.y1, 1, b.area, (float)b.y1 });
        } else {
            row & r = rows[best];
            r.x0 = std::min(r.x0, b.x0);
            r.y0 = std::min(r.y0, b.y0);
            r.x1 = std::max(r.x1, b.x1);
            r.y1 = std::max(r.y1, b.y1);
            r.area += b.area;
            r.baseline = (r.baseline * r.count + b.y1) / (r.count + 1.0f);
            ++r.count;
        }
    }
    std::sort(rows.begin(), rows.end(), [](const row & a, const row & b) { return a.y0 < b.y0; });
    if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG")) {
        for (const auto & r : rows) {
            std::fprintf(stderr, "component-row x=%d..%d y=%d..%d count=%d base=%.1f\n", r.x0, r.x1, r.y0, r.y1,
                         r.count, r.baseline);
        }
    }
    for (const auto & r : rows) {
        if (r.count < 2 || r.x1 - r.x0 < std::max(8, width / 50) ||
            r.y1 - r.y0 + 1 < std::max(6, (int)std::lround(median_h * 0.75f)))
            continue;
        const int crop_y0 = std::max(r.y0, (int)std::floor(r.baseline - median_h * 1.6f));
        const int crop_y1 = std::min(r.y1, (int)std::ceil(r.baseline + median_h * 1.2f));
        int tight_x0 = width, tight_x1 = -1;
        int loose_x0 = width, loose_x1 = -1;
        for (int x = 0; x < width; ++x) {
            int ink = 0;
            for (int y = crop_y0; y <= crop_y1; ++y) ink += gray[y * width + x] < threshold;
            if (ink >= 2) {
                loose_x0 = std::min(loose_x0, x);
                loose_x1 = std::max(loose_x1, x);
            }
            if (ink >= 4) {
                tight_x0 = std::min(tight_x0, x);
                tight_x1 = std::max(tight_x1, x);
            }
        }
        if (r.count < 20) {
            // Tesseract's noise list prevents a detached speck from extending
            // a short row. Keep the widest nearby active cluster for such rows;
            // long rows intentionally retain separated word clusters.
            int best_run_x0 = width, best_run_x1 = -1;
            int run_x0 = width, last_active = -1;
            for (int x = 0; x < width; ++x) {
                int ink = 0;
                for (int y = crop_y0; y <= crop_y1; ++y) ink += gray[y * width + x] < threshold;
                if (ink >= 2) {
                    if (run_x0 == width || x - last_active > 8) {
                        if (last_active >= run_x0 && last_active - run_x0 > best_run_x1 - best_run_x0) {
                            best_run_x0 = run_x0;
                            best_run_x1 = last_active;
                        }
                        run_x0 = x;
                    }
                    last_active = x;
                }
            }
            if (last_active >= run_x0 && last_active - run_x0 > best_run_x1 - best_run_x0) {
                best_run_x0 = run_x0;
                best_run_x1 = last_active;
            }
            if (best_run_x1 >= best_run_x0) {
                tight_x0 = loose_x0 = best_run_x0;
                tight_x1 = loose_x1 = best_run_x1;
            }
        }
        if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_DEBUG") && r.count < 20) {
            std::fprintf(stderr, "component-row-tight x=%d..%d loose=%d..%d y=%d..%d\n", tight_x0, tight_x1, loose_x0,
                         loose_x1, crop_y0, crop_y1);
        }
        if (tight_x1 >= tight_x0 && loose_x1 >= loose_x0 && tight_x0 > r.x0 + 5 &&
            tight_x1 - tight_x0 > (r.y1 - r.y0 + 1) * 10) {
            tight_x0 = loose_x0;
            tight_x1 = loose_x1;
        }
        if (std::getenv("CRISPEMBED_TESSERACT_PAGESEG_ROW_BLOB_BOUNDS")) {
            // The vertical scan can see ink from an adjacent overlapping row
            // and widen the crop beyond the blobs actually assigned here.
            // Keep this A/B opt-in until it is validated across page fixtures.
            tight_x0 = r.x0;
            tight_x1 = r.x1;
            loose_x0 = r.x0;
            loose_x1 = r.x1;
        }
        const int box_x0 = tight_x1 >= tight_x0 ? tight_x0 : r.x0;
        const int box_x1 = tight_x1 >= tight_x0 ? tight_x1 : r.x1;
        ocr_detect::text_box box{};
        box.x = (float)std::max(0, box_x0 - 3);
        const int vertical_offset = r.count < 20 ? 3 : 0;
        box.y = (float)std::min(height - 1, std::max(0, crop_y0 + vertical_offset));
        box.w = (float)std::min(width - (int)box.x, box_x1 - box_x0 + 7);
        box.h = (float)std::min(height - (int)box.y, crop_y1 - crop_y0 + 2);
        box.score = 1.0f;
        out.push_back(box);
    }
    std::sort(out.begin(), out.end(),
              [](const auto & a, const auto & b) { return a.y == b.y ? a.x < b.x : a.y < b.y; });
    if (use_fallback && out.size() < 2) {
        // A single baseline row is not a useful page segmentation result. The
        // projection splitter is the safer fallback for this case and keeps
        // the baseline implementation explicitly opt-in for A/B diagnostics.
        return segment_gray(gray, width, height);
    }
    return out;
}

} // namespace tesseract_pageseg
