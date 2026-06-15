// test_table_parse.cpp — test rule-based table structure recognition.
//
// Usage:
//   test-table-parse                          (synthetic tests, no model)
//   test-table-parse <tesseract.gguf>         (synthetic + OCR)
//   test-table-parse <tesseract.gguf> img.png (real image + OCR)

#include "table_parse.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// Generate a synthetic table image with ruled lines.
// nrows × ncols grid, white background, dark grid lines.
static std::vector<uint8_t> make_ruled_table(int w, int h,
                                              int nrows, int ncols,
                                              int line_width = 2) {
    std::vector<uint8_t> img(w * h, 240); // light gray bg

    // Horizontal lines
    for (int r = 0; r <= nrows; r++) {
        int y = r * h / nrows;
        for (int dy = 0; dy < line_width && y + dy < h; dy++)
            for (int x = 0; x < w; x++)
                img[(y + dy) * w + x] = 20;
    }

    // Vertical lines
    for (int c = 0; c <= ncols; c++) {
        int x = c * w / ncols;
        for (int dx = 0; dx < line_width && x + dx < w; dx++)
            for (int y = 0; y < h; y++)
                img[y * w + (x + dx)] = 20;
    }

    return img;
}

// Generate a borderless table: rows of text with wide column spacing.
static std::vector<uint8_t> make_borderless_table(int w, int h,
                                                    int nrows, int ncols) {
    std::vector<uint8_t> img(w * h, 255); // white bg

    int row_h = h / nrows;
    int col_w = w / ncols;
    int text_h = 4; // text-like lines
    int gap = w / 10; // wide gap between columns (needs to survive dilation)

    for (int r = 0; r < nrows; r++) {
        int y0 = r * row_h + row_h / 3;
        for (int c = 0; c < ncols; c++) {
            int x0 = c * col_w + gap;
            int x1 = (c + 1) * col_w - gap;
            // Draw dark text-like marks
            for (int dy = 0; dy < text_h && y0 + dy < h; dy++)
                for (int x = x0; x < x1 && x < w; x++)
                    img[(y0 + dy) * w + x] = 30;
        }
    }

    return img;
}

static void test_grid_detection_ruled() {
    printf("=== Ruled table grid detection ===\n");

    // 3×4 ruled table
    int w = 400, h = 300;
    auto img = make_ruled_table(w, h, 3, 4);

    int nr = 0, nc = 0;
    int rc = table_parse_detect_grid(img.data(), w, h, &nr, &nc);

    CHECK(rc == 0, "detect_grid returns 0");
    printf("  detected: %d rows × %d cols\n", nr, nc);
    CHECK(nr == 3, "3 rows detected");
    CHECK(nc == 4, "4 cols detected");
}

static void test_grid_detection_borderless() {
    printf("\n=== Borderless table grid detection ===\n");

    int w = 400, h = 200;
    auto img = make_borderless_table(w, h, 4, 3);

    int nr = 0, nc = 0;
    int rc = table_parse_detect_grid(img.data(), w, h, &nr, &nc);

    CHECK(rc == 0, "detect_grid returns 0");
    printf("  detected: %d rows × %d cols\n", nr, nc);
    CHECK(nr >= 3, "at least 3 rows detected");
    CHECK(nc >= 2, "at least 2 cols detected");
}

static void test_html_output() {
    printf("\n=== HTML output (no OCR) ===\n");

    int w = 400, h = 300;
    auto img = make_ruled_table(w, h, 3, 4);

    table_parse_context * ctx = table_parse_init(nullptr, 2);
    CHECK(ctx != nullptr, "init succeeds without OCR model");

    char * html = table_parse_to_html(ctx, img.data(), w, h);
    CHECK(html != nullptr, "to_html returns non-null");

    if (html) {
        printf("  HTML length: %d bytes\n", (int)strlen(html));
        CHECK(strstr(html, "<table>") != nullptr, "contains <table>");
        CHECK(strstr(html, "</table>") != nullptr, "contains </table>");
        CHECK(strstr(html, "<tr>") != nullptr, "contains <tr>");

        // Count rows
        int tr_count = 0;
        const char * p = html;
        while ((p = strstr(p, "<tr>")) != nullptr) { tr_count++; p++; }
        printf("  rows in HTML: %d\n", tr_count);
        CHECK(tr_count == 3, "HTML has 3 rows");

        // Count cells (th + td)
        int cell_count = 0;
        p = html;
        while ((p = strstr(p, "<t")) != nullptr) {
            if (p[2] == 'h' || p[2] == 'd') cell_count++;
            p++;
        }
        printf("  cells in HTML: %d\n", cell_count);
        CHECK(cell_count == 12, "HTML has 12 cells (3×4)");

        // First row should use <th>
        CHECK(strstr(html, "<th>") != nullptr, "first row uses <th>");

        printf("\n--- HTML output ---\n%s--- end ---\n", html);
        table_parse_free_string(html);
    }

    table_parse_free(ctx);
}

static void test_with_ocr(const char * model_path) {
    printf("\n=== HTML output with OCR ===\n");

    int w = 400, h = 300;
    auto img = make_ruled_table(w, h, 2, 3);

    table_parse_context * ctx = table_parse_init(model_path, 2);
    CHECK(ctx != nullptr, "init succeeds with OCR model");

    char * html = table_parse_to_html(ctx, img.data(), w, h);
    CHECK(html != nullptr, "to_html returns non-null");

    if (html) {
        printf("\n--- HTML with OCR ---\n%s--- end ---\n", html);
        table_parse_free_string(html);
    }

    table_parse_free(ctx);
}

int main(int argc, char ** argv) {
    // Synthetic tests (no model needed)
    test_grid_detection_ruled();
    test_grid_detection_borderless();
    test_html_output();

    // OCR test (optional)
    if (argc >= 2) {
        test_with_ocr(argv[1]);
    }

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
