#include "tesseract_pageseg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tesseract_pageseg {

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
    return out;
}

} // namespace tesseract_pageseg
