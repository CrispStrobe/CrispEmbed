// cc_detect.cpp — Classical text line detection via connected components.
//
// Algorithms cherry-picked from Leptonica (BSD-2-Clause, Dan Bloomberg).
// Uses morph_fast.h for 1-bit morphological operations.
//
// The approach (from Leptonica's pixGenTextlineMask):
//   1. Binarize → 1-bit packed image
//   2. Horizontal close (wide SE) → merge characters into text lines
//   3. (Optional) subtract vertical whitespace to split columns
//   4. Small open → remove noise
//   5. Connected component labeling → bounding boxes
//   6. Filter by minimum size → return sorted regions

#include "cc_detect.h"
#include "morph_fast.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Simple Otsu threshold (from scan_cleanup pattern)
// ---------------------------------------------------------------------------

static uint8_t otsu_threshold(const uint8_t * gray, int w, int h) {
    int hist[256] = {};
    int n = w * h;
    for (int i = 0; i < n; i++) hist[gray[i]]++;

    int total = n;
    double sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];

    double sumB = 0;
    int wB = 0;
    double max_var = 0;
    int best_t = 128;

    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = total - wB;
        if (wF == 0) break;
        sumB += t * hist[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double var = (double)wB * wF * (mB - mF) * (mB - mF);
        if (var > max_var) {
            max_var = var;
            best_t = t;
        }
    }
    return (uint8_t)best_t;
}

// ---------------------------------------------------------------------------
// Connected component labeling (4-connectivity, two-pass)
// ---------------------------------------------------------------------------
// Uses classic union-find algorithm. Operates on 1-bit packed image.
// Returns label image + bounding boxes.

struct UnionFind {
    std::vector<int> parent;
    std::vector<int> rank;

    int make_set() {
        int id = (int)parent.size();
        parent.push_back(id);
        rank.push_back(0);
        return id;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path compression
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
    }
};

static inline int get_bit(const uint32_t * line, int x) {
    return (line[x >> 5] >> (31 - (x & 31))) & 1;
}

struct BBox {
    int x0, y0, x1, y1;
    BBox() : x0(999999), y0(999999), x1(-1), y1(-1) {}
    void add(int x, int y) {
        if (x < x0) x0 = x;
        if (y < y0) y0 = y;
        if (x > x1) x1 = x;
        if (y > y1) y1 = y;
    }
};

static std::vector<BBox> cc_label_boxes(
    const uint32_t * bits, int w, int h, int wpl)
{
    // Two-pass labeling with union-find
    std::vector<int> labels(w * h, -1);
    UnionFind uf;

    // Pass 1: assign provisional labels
    for (int y = 0; y < h; y++) {
        const uint32_t * line = bits + y * wpl;
        for (int x = 0; x < w; x++) {
            if (!get_bit(line, x)) continue; // background

            int idx = y * w + x;
            int left = (x > 0 && get_bit(line, x - 1)) ? labels[idx - 1] : -1;
            int up = -1;
            if (y > 0) {
                const uint32_t * prev_line = bits + (y - 1) * wpl;
                if (get_bit(prev_line, x))
                    up = labels[(y - 1) * w + x];
            }

            if (left == -1 && up == -1) {
                labels[idx] = uf.make_set();
            } else if (left != -1 && up == -1) {
                labels[idx] = left;
            } else if (left == -1 && up != -1) {
                labels[idx] = up;
            } else {
                // Both neighbors have labels — union them
                uf.unite(left, up);
                labels[idx] = uf.find(left);
            }
        }
    }

    // Pass 2: resolve labels + compute bounding boxes
    std::vector<BBox> boxes;
    std::vector<int> label_map(uf.parent.size(), -1);
    int n_labels = 0;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            int lbl = labels[idx];
            if (lbl < 0) continue;
            int root = uf.find(lbl);
            if (label_map[root] < 0) {
                label_map[root] = n_labels++;
                boxes.emplace_back();
            }
            boxes[label_map[root]].add(x, y);
        }
    }

    return boxes;
}

// ---------------------------------------------------------------------------
// Subtract vertical whitespace
// ---------------------------------------------------------------------------
// Identify tall, narrow vertical runs of background. Invert image,
// close with tall vertical SE, AND with the mask → vertical whitespace.
// Subtract from the textline mask.

static void subtract_vws(uint32_t * mask, int w, int h, int wpl) {
    // Invert the mask to find background
    int n = wpl * h;
    uint32_t * inv = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!inv) return;
    for (int i = 0; i < n; i++) inv[i] = ~mask[i];

    // Open with tall vertical SE: keeps only tall vertical whitespace
    // o1.200 at 150dpi ≈ o1.100 at full res
    uint32_t * vws = morph_open_brick(inv, w, h, wpl, 1, 100);
    free(inv);
    if (!vws) return;

    // Subtract: mask &= ~vws
    for (int i = 0; i < n; i++) mask[i] &= ~vws[i];
    morph_free(vws);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cc_detect_params cc_detect_defaults(void) {
    cc_detect_params p;
    p.close_hsize = 30;
    p.close_vsize = 1;
    p.open_size = 3;
    p.min_width = 10;
    p.min_height = 5;
    p.binarize_threshold = 0; // auto Otsu
    return p;
}

cc_text_region * cc_detect_lines_params(
    const uint8_t * gray, int width, int height,
    cc_detect_params params, int * out_n)
{
    if (out_n) *out_n = 0;
    if (!gray || width <= 0 || height <= 0) return nullptr;

    // 1. Binarize
    uint8_t thresh = params.binarize_threshold;
    if (thresh == 0) thresh = otsu_threshold(gray, width, height);

    int wpl = 0;
    uint32_t * bits = morph_u8_to_1bit(gray, width, height, thresh, &wpl);
    if (!bits) return nullptr;

    // 2. Horizontal close → merge characters into lines
    uint32_t * closed = morph_close_brick(bits, width, height, wpl,
                                           params.close_hsize, params.close_vsize);
    morph_free(bits);
    if (!closed) return nullptr;

    // 3. Subtract vertical whitespace (column splitting)
    subtract_vws(closed, width, height, wpl);

    // 4. Small open → remove noise
    if (params.open_size > 1) {
        uint32_t * opened = morph_open_brick(closed, width, height, wpl,
                                              params.open_size, params.open_size);
        morph_free(closed);
        if (!opened) return nullptr;
        closed = opened;
    }

    // 5. Connected component labeling → bounding boxes
    auto boxes = cc_label_boxes(closed, width, height, wpl);
    morph_free(closed);

    // 6. Filter by minimum size and sort
    std::vector<cc_text_region> regions;
    for (auto & b : boxes) {
        int rw = b.x1 - b.x0 + 1;
        int rh = b.y1 - b.y0 + 1;
        if (rw >= params.min_width && rh >= params.min_height) {
            regions.push_back({b.x0, b.y0, rw, rh});
        }
    }

    // Sort: top-to-bottom, then left-to-right
    std::sort(regions.begin(), regions.end(), [](const cc_text_region & a, const cc_text_region & b) {
        int ay = a.y + a.h / 2, by = b.y + b.h / 2;
        if (ay != by) return ay < by;
        return a.x < b.x;
    });

    // Copy to C array
    int n = (int)regions.size();
    if (n == 0) return nullptr;
    auto * result = (cc_text_region *)malloc(n * sizeof(cc_text_region));
    if (!result) return nullptr;
    memcpy(result, regions.data(), n * sizeof(cc_text_region));
    if (out_n) *out_n = n;
    return result;
}

cc_text_region * cc_detect_lines(
    const uint8_t * gray, int width, int height, int * out_n)
{
    return cc_detect_lines_params(gray, width, height, cc_detect_defaults(), out_n);
}

void cc_detect_free(cc_text_region * regions) {
    free(regions);
}
