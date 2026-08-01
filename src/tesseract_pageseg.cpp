#include "tesseract_pageseg.h"

#include <algorithm>

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
    return out;
}

} // namespace tesseract_pageseg
