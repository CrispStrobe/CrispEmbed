// tests/test_ocr_render.cpp — unit tests for OCR result renderers.
#include "ocr_render.h"
#include <cstdio>
#include <cstring>
#include <string>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

static bool contains(const char * haystack, const char * needle) {
    return strstr(haystack, needle) != nullptr;
}

// Build a sample 2-page document for testing
static void make_test_pages(ocr_render_page pages[2],
                             ocr_render_line lines[4],
                             ocr_render_word words[8]) {
    // Page 1: 2 lines, 2 words each
    words[0] = {"Hello", 10, 20, 50, 15, 0.99f};
    words[1] = {"World", 70, 20, 55, 15, 0.95f};
    words[2] = {"Line", 10, 50, 40, 15, 0.92f};
    words[3] = {"two", 60, 50, 35, 15, 0.88f};

    lines[0] = {&words[0], 2, 10, 20, 115, 15};
    lines[1] = {&words[2], 2, 10, 50, 85, 15};

    pages[0] = {&lines[0], 2, 400, 300, "page1.png"};

    // Page 2: 2 lines
    words[4] = {"Second", 15, 25, 65, 14, 0.97f};
    words[5] = {"page", 90, 25, 42, 14, 0.93f};
    words[6] = {"Last", 15, 55, 40, 14, 0.91f};
    words[7] = {"line.", 65, 55, 45, 14, 0.89f};

    lines[2] = {&words[4], 2, 15, 25, 117, 14};
    lines[3] = {&words[6], 2, 15, 55, 95, 14};

    pages[1] = {&lines[2], 2, 400, 300, "page2.png"};
}

static void test_text_renderer() {
    printf("\n=== Plain text renderer ===\n");

    ocr_render_page pages[2];
    ocr_render_line lines[4];
    ocr_render_word words[8];
    make_test_pages(pages, lines, words);

    ocr_renderer * r = ocr_render_create(OCR_RENDER_TEXT);
    ocr_render_begin(r);
    ocr_render_add_page(r, &pages[0]);
    ocr_render_add_page(r, &pages[1]);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);
    printf("  Output:\n---\n%s---\n", out);

    check("contains Hello World", contains(out, "Hello World"));
    check("contains Line two", contains(out, "Line two"));
    check("contains Second page", contains(out, "Second page"));
    check("contains page separator (\\f)", contains(out, "\f"));
    check("output size > 0", ocr_render_output_size(r) > 0);

    // Custom separator
    ocr_renderer * r2 = ocr_render_create(OCR_RENDER_TEXT);
    ocr_render_set_separator(r2, "\n---PAGE---\n");
    ocr_render_begin(r2);
    ocr_render_add_page(r2, &pages[0]);
    ocr_render_add_page(r2, &pages[1]);
    ocr_render_end(r2);
    check("custom separator", contains(ocr_render_output(r2), "---PAGE---"));

    ocr_render_free(r);
    ocr_render_free(r2);
}

static void test_hocr_renderer() {
    printf("\n=== hOCR renderer ===\n");

    ocr_render_page pages[2];
    ocr_render_line lines[4];
    ocr_render_word words[8];
    make_test_pages(pages, lines, words);

    ocr_renderer * r = ocr_render_create(OCR_RENDER_HOCR);
    ocr_render_begin(r);
    ocr_render_add_page(r, &pages[0]);
    ocr_render_add_page(r, &pages[1]);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);

    check("valid XML declaration", contains(out, "<?xml version=\"1.0\""));
    check("html root element", contains(out, "<html"));
    check("ocr_page class", contains(out, "class=\"ocr_page\""));
    check("ocr_line class", contains(out, "class=\"ocr_line\""));
    check("ocrx_word class", contains(out, "class=\"ocrx_word\""));
    check("bbox in title", contains(out, "bbox 0 0 400 300"));
    check("word Hello present", contains(out, ">Hello</span>"));
    check("word confidence", contains(out, "x_wconf 99"));
    check("page 2 present", contains(out, "page_2"));
    check("closes html", contains(out, "</html>"));

    ocr_render_free(r);
}

static void test_alto_renderer() {
    printf("\n=== ALTO renderer ===\n");

    ocr_render_page pages[2];
    ocr_render_line lines[4];
    ocr_render_word words[8];
    make_test_pages(pages, lines, words);

    ocr_renderer * r = ocr_render_create(OCR_RENDER_ALTO);
    ocr_render_begin(r);
    ocr_render_add_page(r, &pages[0]);
    ocr_render_add_page(r, &pages[1]);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);

    check("ALTO namespace", contains(out, "xmlns=\"http://www.loc.gov/standards/alto"));
    check("MeasurementUnit pixel", contains(out, "<MeasurementUnit>pixel</MeasurementUnit>"));
    check("CrispEmbed software", contains(out, "<softwareName>CrispEmbed</softwareName>"));
    check("Page element", contains(out, "<Page ID=\"page_0\""));
    check("Page dimensions", contains(out, "WIDTH=\"400\" HEIGHT=\"300\""));
    check("TextBlock element", contains(out, "<TextBlock"));
    check("TextLine element", contains(out, "<TextLine"));
    check("String element", contains(out, "<String CONTENT=\"Hello\""));
    check("SP element", contains(out, "<SP WIDTH="));
    check("word confidence WC", contains(out, "WC=\"0.99\""));
    check("closes alto", contains(out, "</alto>"));

    ocr_render_free(r);
}

static void test_pdf_renderer() {
    printf("\n=== PDF renderer ===\n");

    ocr_render_page pages[1];
    ocr_render_line lines[2];
    ocr_render_word words[4];
    // Reuse page 1 only
    words[0] = {"Hello", 10, 20, 50, 15, 0.99f};
    words[1] = {"World", 70, 20, 55, 15, 0.95f};
    lines[0] = {&words[0], 2, 10, 20, 115, 15};
    pages[0] = {&lines[0], 1, 400, 300, nullptr};

    ocr_renderer * r = ocr_render_create(OCR_RENDER_PDF);
    ocr_render_begin(r);
    ocr_render_add_page(r, &pages[0]);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);
    int size = ocr_render_output_size(r);

    check("PDF header", size > 0 && strncmp(out, "%PDF-1.4", 8) == 0);
    check("has %%EOF", contains(out, "%%EOF"));
    check("has xref", contains(out, "xref"));
    check("has Catalog", contains(out, "/Catalog"));
    check("has Helvetica font", contains(out, "/Helvetica"));

    ocr_render_free(r);
}

static void test_xml_escaping() {
    printf("\n=== XML escaping ===\n");

    ocr_render_word w = {"<b>Bold</b> & \"quoted\"", 0, 0, 100, 20, 1.0f};
    ocr_render_line l = {&w, 1, 0, 0, 100, 20};
    ocr_render_page p = {&l, 1, 200, 100, nullptr};

    ocr_renderer * r = ocr_render_create(OCR_RENDER_HOCR);
    ocr_render_begin(r);
    ocr_render_add_page(r, &p);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);
    check("& escaped to &amp;", contains(out, "&amp;"));
    check("< escaped to &lt;", contains(out, "&lt;b&gt;"));
    check("\" escaped to &quot;", contains(out, "&quot;"));
    check("no raw <b> tag", !contains(out, "<b>Bold"));

    ocr_render_free(r);
}

int main() {
    printf("OCR renderer — unit tests\n");

    test_text_renderer();
    test_hocr_renderer();
    test_alto_renderer();
    test_pdf_renderer();
    test_xml_escaping();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
