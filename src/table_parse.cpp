// table_parse.cpp — Rule-based table structure recognition.
//
// Line detection via 1-bit morphology (morph_fast.h):
//   - Horizontal lines: erode with wide horizontal SE → keeps only long h-runs
//   - Vertical lines: erode with tall vertical SE → keeps only long v-runs
//   - OR the h/v masks → grid lines
//   - Project onto X/Y axes → find grid boundaries
//
// For borderless tables (no lines found), falls back to:
//   - Vertical projection profile → column boundaries
//   - Horizontal projection profile → row boundaries

#include "table_parse.h"
#include "morph_fast.h"
#include "tesseract_lstm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────

static uint8_t otsu_threshold(const uint8_t * gray, int w, int h) {
    int hist[256] = {};
    int n = w * h;
    for (int i = 0; i < n; i++) hist[gray[i]]++;

    double sum = 0;
    for (int i = 0; i < 256; i++) sum += i * hist[i];

    double sumB = 0;
    int wB = 0;
    double max_var = 0;
    int best_t = 128;

    for (int t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB == 0) continue;
        int wF = n - wB;
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

// Horizontal projection: count foreground pixels per row
static std::vector<int> h_projection(const uint32_t * bits, int w, int h, int wpl) {
    std::vector<int> proj(h, 0);
    for (int y = 0; y < h; y++) {
        const uint32_t * line = bits + y * wpl;
        for (int x = 0; x < w; x++) {
            if ((line[x >> 5] >> (31 - (x & 31))) & 1)
                proj[y]++;
        }
    }
    return proj;
}

// Vertical projection: count foreground pixels per column
static std::vector<int> v_projection(const uint32_t * bits, int w, int h, int wpl) {
    std::vector<int> proj(w, 0);
    for (int y = 0; y < h; y++) {
        const uint32_t * line = bits + y * wpl;
        for (int x = 0; x < w; x++) {
            if ((line[x >> 5] >> (31 - (x & 31))) & 1)
                proj[x]++;
        }
    }
    return proj;
}

// Find positions where projection exceeds threshold (line candidates)
static std::vector<int> find_line_positions(const std::vector<int> & proj,
                                             int length, float min_fraction) {
    int thresh = (int)(length * min_fraction);
    std::vector<int> positions;
    int i = 0;
    while (i < (int)proj.size()) {
        if (proj[i] >= thresh) {
            // Find center of this line cluster
            int start = i;
            while (i < (int)proj.size() && proj[i] >= thresh) i++;
            positions.push_back((start + i - 1) / 2);
        } else {
            i++;
        }
    }
    return positions;
}

// Find boundaries from projection valleys (for borderless tables)
static std::vector<int> find_boundaries_from_valleys(const std::vector<int> & proj,
                                                      int length, int min_gap) {
    std::vector<int> bounds;
    bounds.push_back(0); // image edge

    int i = 0;
    while (i < (int)proj.size()) {
        // Find a gap (low projection region)
        if (proj[i] == 0) {
            int start = i;
            while (i < (int)proj.size() && proj[i] == 0) i++;
            int gap_len = i - start;
            if (gap_len >= min_gap) {
                bounds.push_back((start + i) / 2);
            }
        } else {
            i++;
        }
    }

    bounds.push_back((int)proj.size()); // image edge
    return bounds;
}

// ── Grid detection ────────────────────────────────────────────────────

struct Grid {
    std::vector<int> row_bounds; // y-coordinates of row boundaries
    std::vector<int> col_bounds; // x-coordinates of column boundaries
    int n_rows() const { return std::max(0, (int)row_bounds.size() - 1); }
    int n_cols() const { return std::max(0, (int)col_bounds.size() - 1); }
};

static Grid detect_grid(const uint8_t * gray, int w, int h) {
    Grid grid;

    // Binarize (foreground = dark pixels = 1).
    // For table images the foreground (lines/text) is dark and the background
    // is light. Compute Otsu then use the midpoint between the Otsu threshold
    // and 255 to be robust against thin lines that shift the Otsu value.
    uint8_t otsu_t = otsu_threshold(gray, w, h);
    // Use half-way between Otsu and white — ensures dark pixels are foreground
    uint8_t bin_thresh = (uint8_t)std::min(254, std::max((int)otsu_t + 1, ((int)otsu_t + 255) / 2));
    int wpl = 0;
    uint32_t * bits = morph_u8_to_1bit(gray, w, h, bin_thresh, &wpl);
    if (!bits) return grid;

    // Try to detect ruled lines first
    // Horizontal lines: erode with wide horizontal SE (keeps only long h-runs)
    int h_se_w = std::max(31, w / 4) | 1; // odd, at least 1/4 image width
    uint32_t * h_lines = morph_erode_brick(bits, w, h, wpl, h_se_w, 1);

    // Vertical lines: erode with tall vertical SE
    int v_se_h = std::max(31, h / 4) | 1;
    uint32_t * v_lines = morph_erode_brick(bits, w, h, wpl, 1, v_se_h);

    // Project h_lines onto Y axis → row boundaries
    auto h_proj = h_projection(h_lines, w, h, wpl);
    auto row_lines = find_line_positions(h_proj, w, 0.15f);

    // Project v_lines onto X axis → column boundaries
    auto v_proj = v_projection(v_lines, w, h, wpl);
    auto col_lines = find_line_positions(v_proj, h, 0.15f);

    if (getenv("CRISPEMBED_TABLE_DEBUG")) {
        fprintf(stderr, "table_parse: otsu=%d, bin=%d, h_se=%d, v_se=%d\n",
                otsu_t, bin_thresh, h_se_w, v_se_h);
        fprintf(stderr, "table_parse: h_proj peaks (>%d):", (int)(w * 0.15f));
        for (int y = 0; y < h; y++)
            if (h_proj[y] > (int)(w * 0.15f))
                fprintf(stderr, " y=%d(%d)", y, h_proj[y]);
        fprintf(stderr, "\n");
        fprintf(stderr, "table_parse: v_proj peaks (>%d):", (int)(h * 0.15f));
        for (int x = 0; x < w; x++)
            if (v_proj[x] > (int)(h * 0.15f))
                fprintf(stderr, " x=%d(%d)", x, v_proj[x]);
        fprintf(stderr, "\n");
        fprintf(stderr, "table_parse: row_lines=%zu, col_lines=%zu\n",
                row_lines.size(), col_lines.size());
    }

    morph_free(h_lines);
    morph_free(v_lines);

    bool has_ruled_lines = row_lines.size() >= 2 && col_lines.size() >= 2;

    if (has_ruled_lines) {
        // Use detected lines as grid boundaries
        grid.row_bounds = row_lines;
        grid.col_bounds = col_lines;

        // Ensure top and bottom edges are included
        if (grid.row_bounds.front() > h / 20)
            grid.row_bounds.insert(grid.row_bounds.begin(), 0);
        if (grid.row_bounds.back() < h - h / 20)
            grid.row_bounds.push_back(h);

        if (grid.col_bounds.front() > w / 20)
            grid.col_bounds.insert(grid.col_bounds.begin(), 0);
        if (grid.col_bounds.back() < w - w / 20)
            grid.col_bounds.push_back(w);
    } else {
        // Borderless table: use projection-based splitting
        // Dilate text to merge characters, then project
        uint32_t * dilated = morph_dilate_brick(bits, w, h, wpl, 5, 3);
        if (dilated) {
            auto h_text = h_projection(dilated, w, h, wpl);
            auto v_text = v_projection(dilated, w, h, wpl);

            // Row boundaries from horizontal gaps
            int min_row_gap = std::max(3, h / 40);
            grid.row_bounds = find_boundaries_from_valleys(h_text, w, min_row_gap);

            // Column boundaries from vertical gaps
            int min_col_gap = std::max(3, w / 20);
            grid.col_bounds = find_boundaries_from_valleys(v_text, h, min_col_gap);

            morph_free(dilated);
        }
    }

    morph_free(bits);

    // Validate: need at least 2 rows and 2 columns
    if (grid.n_rows() < 1 || grid.n_cols() < 1) {
        grid.row_bounds.clear();
        grid.col_bounds.clear();
    }

    return grid;
}

// ── Context ───────────────────────────────────────────────────────────

struct table_parse_context {
    tesseract_lstm_context * tess = nullptr;
    table_cell_ocr_fn ocr_fn = nullptr;
    void * ocr_user_data = nullptr;
    int n_threads = 2;
    bool bench = false;
};

table_parse_context * table_parse_init(const char * ocr_model_path, int n_threads) {
    auto * ctx = new table_parse_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 2;
    ctx->bench = (std::getenv("CRISPEMBED_TABLE_PARSE_BENCH") != nullptr);

    if (ocr_model_path && *ocr_model_path) {
        ctx->tess = tesseract_lstm_init(ocr_model_path, ctx->n_threads);
        if (!ctx->tess) {
            fprintf(stderr, "table_parse: failed to load OCR model %s\n", ocr_model_path);
            // Non-fatal: can still detect grid, just no OCR
        }
    }

    return ctx;
}

void table_parse_free(table_parse_context * ctx) {
    if (!ctx) return;
    if (ctx->tess) tesseract_lstm_free(ctx->tess);
    delete ctx;
}

void table_parse_set_ocr(table_parse_context * ctx,
                         table_cell_ocr_fn fn, void * user_data) {
    if (!ctx) return;
    ctx->ocr_fn = fn;
    ctx->ocr_user_data = user_data;
}

// OCR a cell crop
static std::string ocr_cell(table_parse_context * ctx,
                            const uint8_t * gray, int w, int h,
                            int x0, int y0, int x1, int y1) {
    int cw = x1 - x0;
    int ch = y1 - y0;
    if (cw <= 2 || ch <= 2) return "";

    // Crop with 1px padding inward to avoid border artifacts
    int pad = 1;
    int cx0 = std::min(x0 + pad, x1);
    int cy0 = std::min(y0 + pad, y1);
    int cx1 = std::max(x1 - pad, x0);
    int cy1 = std::max(y1 - pad, y0);
    cw = cx1 - cx0;
    ch = cy1 - cy0;
    if (cw <= 2 || ch <= 2) return "";

    std::vector<uint8_t> crop(cw * ch);
    for (int y = 0; y < ch; y++) {
        memcpy(crop.data() + y * cw, gray + (cy0 + y) * w + cx0, cw);
    }

    // Use callback if set, else built-in Tesseract
    if (ctx->ocr_fn) {
        const char * text = ctx->ocr_fn(ctx->ocr_user_data, crop.data(), cw, ch);
        return text ? text : "";
    }

    if (ctx->tess) {
        int len = 0;
        const char * text = tesseract_lstm_recognize(ctx->tess, crop.data(), cw, ch, &len);
        if (text && len > 0) return std::string(text, len);
    }

    return "";
}

// Escape HTML special characters
static std::string html_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '&': out += "&amp;"; break;
            case '"': out += "&quot;"; break;
            default:  out += c; break;
        }
    }
    return out;
}

// Trim whitespace
static std::string trim(const std::string & s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// ── Public API ────────────────────────────────────────────────────────

char * table_parse_to_html(table_parse_context * ctx,
                           const uint8_t * gray, int width, int height) {
    if (!ctx || !gray || width <= 0 || height <= 0) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t_grid = std::chrono::steady_clock::now();
    Grid grid = detect_grid(gray, width, height);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_grid).count();
        fprintf(stderr, "[table_parse-bench] grid detect: %.1f ms\n", ms);
    }

    if (grid.n_rows() < 1 || grid.n_cols() < 1) {
        fprintf(stderr, "table_parse: no grid detected (%dx%d)\n", width, height);
        return nullptr;
    }

    fprintf(stderr, "table_parse: %d rows × %d cols grid\n",
            grid.n_rows(), grid.n_cols());

    auto t_ocr = std::chrono::steady_clock::now();
    std::ostringstream html;
    html << "<table>\n";

    for (int r = 0; r < grid.n_rows(); r++) {
        html << "  <tr>\n";
        for (int c = 0; c < grid.n_cols(); c++) {
            int x0 = grid.col_bounds[c];
            int y0 = grid.row_bounds[r];
            int x1 = grid.col_bounds[c + 1];
            int y1 = grid.row_bounds[r + 1];

            std::string text = trim(ocr_cell(ctx, gray, width, height,
                                             x0, y0, x1, y1));

            // Use <th> for first row (likely header)
            const char * tag = (r == 0) ? "th" : "td";
            html << "    <" << tag << ">"
                 << html_escape(text)
                 << "</" << tag << ">\n";
        }
        html << "  </tr>\n";
    }

    html << "</table>\n";
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_ocr).count();
        fprintf(stderr, "[table_parse-bench] cell OCR: %.1f ms\n", ms);
    }

    std::string result = html.str();
    char * out = (char *)malloc(result.size() + 1);
    if (out) {
        memcpy(out, result.c_str(), result.size() + 1);
    }

    if (bench) {
        double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[table_parse-bench] total: %.1f ms\n", total_ms);
    }

    return out;
}

void table_parse_free_string(char * str) {
    free(str);
}

int table_parse_detect_grid(const uint8_t * gray, int width, int height,
                            int * out_n_rows, int * out_n_cols) {
    Grid grid = detect_grid(gray, width, height);
    if (out_n_rows) *out_n_rows = grid.n_rows();
    if (out_n_cols) *out_n_cols = grid.n_cols();
    return (grid.n_rows() > 0 && grid.n_cols() > 0) ? 0 : -1;
}
