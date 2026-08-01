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
    return out;
}

} // namespace tesseract_pageseg
