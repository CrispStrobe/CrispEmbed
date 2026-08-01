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

std::vector<ocr_detect::text_box> segment_gray_components(const uint8_t * gray, int width, int height) {
    std::vector<ocr_detect::text_box> out;
    if (!gray || width <= 0 || height <= 0) return out;
    uint64_t sum = 0;
    for (int i = 0; i < width * height; ++i) sum += gray[i];
    const int threshold = std::clamp((int)(sum / (uint64_t)(width * height)) - 130, 60, 120);
    std::vector<uint8_t> seen((size_t)width * height, 0);
    struct blob { int x0, y0, x1, y1, area; };
    std::vector<blob> blobs;
    std::vector<int> stack;
    stack.reserve(256);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const size_t seed = (size_t)y * width + x;
        if (seen[seed] || gray[seed] >= threshold) continue;
        seen[seed] = 1; stack.clear(); stack.push_back((int)seed);
        blob b{x, y, x, y, 0};
        while (!stack.empty()) {
            const int p = stack.back(); stack.pop_back();
            const int py = p / width, px = p - py * width;
            b.x0 = std::min(b.x0, px); b.x1 = std::max(b.x1, px);
            b.y0 = std::min(b.y0, py); b.y1 = std::max(b.y1, py); ++b.area;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                if (!dx && !dy) continue;
                const int nx = px + dx, ny = py + dy;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                const size_t q = (size_t)ny * width + nx;
                if (!seen[q] && gray[q] < threshold) { seen[q] = 1; stack.push_back((int)q); }
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
        std::fprintf(stderr, "tesseract_pageseg: components=%zu median_h=%d row_gap=%d\n",
                     blobs.size(), median_h, max_row_gap);
    std::sort(blobs.begin(), blobs.end(), [](const blob & a, const blob & b) {
        return a.y0 == b.y0 ? a.x0 < b.x0 : a.y0 < b.y0;
    });
    struct row { std::vector<blob> blobs; int y0 = 0, y1 = 0; };
    std::vector<row> rows;
    for (const auto & b : blobs) {
        if (!rows.empty() && b.y0 <= rows.back().y1 + max_row_gap) {
            rows.back().blobs.push_back(b);
            rows.back().y1 = std::max(rows.back().y1, b.y1);
        } else {
            rows.push_back({{b}, b.y0, b.y1});
        }
    }
    for (auto & r : rows) {
        if (r.blobs.size() < 2) continue;
        int x0 = width, x1 = -1;
        for (const auto & b : r.blobs) { x0 = std::min(x0, b.x0); x1 = std::max(x1, b.x1); }
        if (x1 < x0) continue;
        ocr_detect::text_box box{};
        box.x = (float)std::max(0, x0 - 3);
        box.y = (float)std::max(0, r.y0 - 3);
        box.w = (float)std::min(width - (int)box.x, x1 - x0 + 7);
        box.h = (float)std::min(height - (int)box.y, r.y1 - r.y0 + 7);
        box.score = 1.0f;
        out.push_back(box);
    }
    std::sort(out.begin(), out.end(), [](const auto & a, const auto & b) { return a.y == b.y ? a.x < b.x : a.y < b.y; });
    return out;
}

} // namespace tesseract_pageseg
