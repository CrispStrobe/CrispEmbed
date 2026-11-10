// tests/test_ocr_render.cpp — unit tests for OCR result renderers.
#include "ocr_render.h"
#include "core/clean_exit.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) {
        printf("  %s[PASS]%s %s\n", GREEN, RESET, name);
        n_pass++;
    } else {
        printf("  %s[FAIL]%s %s\n", RED, RESET, name);
        n_fail++;
    }
}

static bool contains(const char * haystack, const char * needle) {
    return strstr(haystack, needle) != nullptr;
}

// Build a sample 2-page document for testing
static void make_test_pages(ocr_render_page pages[2], ocr_render_line lines[4], ocr_render_word words[8]) {
    // Page 1: 2 lines, 2 words each
    words[0] = { "Hello", 10, 20, 50, 15, 0.99f };
    words[1] = { "World", 70, 20, 55, 15, 0.95f };
    words[2] = { "Line", 10, 50, 40, 15, 0.92f };
    words[3] = { "two", 60, 50, 35, 15, 0.88f };

    lines[0] = { &words[0], 2, 10, 20, 115, 15 };
    lines[1] = { &words[2], 2, 10, 50, 85, 15 };

    pages[0] = { &lines[0], 2, 400, 300, "page1.png" };

    // Page 2: 2 lines
    words[4] = { "Second", 15, 25, 65, 14, 0.97f };
    words[5] = { "page", 90, 25, 42, 14, 0.93f };
    words[6] = { "Last", 15, 55, 40, 14, 0.91f };
    words[7] = { "line.", 65, 55, 45, 14, 0.89f };

    lines[2] = { &words[4], 2, 15, 25, 117, 14 };
    lines[3] = { &words[6], 2, 15, 55, 95, 14 };

    pages[1] = { &lines[2], 2, 400, 300, "page2.png" };
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
    words[0] = { "Hello", 10, 20, 50, 15, 0.99f };
    words[1] = { "World", 70, 20, 55, 15, 0.95f };
    lines[0] = { &words[0], 2, 10, 20, 115, 15 };
    pages[0] = { &lines[0], 1, 400, 300, nullptr };

    ocr_renderer * r = ocr_render_create(OCR_RENDER_PDF);
    ocr_render_begin(r);
    ocr_render_add_page(r, &pages[0]);
    ocr_render_end(r);

    const char * out = ocr_render_output(r);
    int size = ocr_render_output_size(r);
    const std::string pdf(out, size_t(size));

    check("PDF header", size > 0 && strncmp(out, "%PDF-1.4", 8) == 0);
    check("has %%EOF", pdf.find("%%EOF") != std::string::npos);
    check("has xref", pdf.find("xref") != std::string::npos);
    check("has Catalog", pdf.find("/Catalog") != std::string::npos);
    check("has Unicode map", pdf.find("/ToUnicode") != std::string::npos);
    check("has embedded font", pdf.find("/FontFile2") != std::string::npos);
    check("no renderer error", ocr_render_error(r) == nullptr);
    check("PDF capabilities exclude unqualified bidi", crispembed_ocr_render_capabilities() == 47);

    ocr_render_free(r);
}

static void test_render_failures_and_long_text() {
    printf("\n=== Errors and long corrections ===\n");
    std::string text(10000, 'x');
    text += "\n\t\r日本語 😀 & <tag> \"end\"";
    ocr_render_word word = { text.c_str(), 0, 0, 200, 30, 0.8f };
    ocr_render_line line = { &word, 1, 0, 0, 200, 30 };
    ocr_render_page page = { &line, 1, 200, 100, nullptr };
    auto * renderer = ocr_render_create(OCR_RENDER_ALTO);
    ocr_render_begin(renderer);
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    const std::string xml(ocr_render_output(renderer));
    check("long ALTO content is complete", xml.find(std::string(10000, 'x')) != std::string::npos);
    check("ALTO retains Unicode suffix", xml.find("日本語 😀 &amp; &lt;tag&gt; &quot;end&quot;") != std::string::npos);
    check("ALTO attribute whitespace survives XML parsing", xml.find("&#xA;&#x9;&#xD;") != std::string::npos);
    ocr_render_free(renderer);

    for (ocr_render_format format : { OCR_RENDER_HOCR, OCR_RENDER_ALTO }) {
        renderer = ocr_render_create(format);
        word.confidence = -1;
        ocr_render_begin(renderer);
        ocr_render_add_page(renderer, &page);
        ocr_render_end(renderer);
        const std::string unknown(ocr_render_output(renderer));
        check("unknown confidence is omitted",
              unknown.find("x_wconf") == std::string::npos && unknown.find(" WC=") == std::string::npos);
        word.confidence = 0;
        ocr_render_begin(renderer);
        ocr_render_add_page(renderer, &page);
        ocr_render_end(renderer);
        const std::string zero(ocr_render_output(renderer));
        check("genuine zero confidence is preserved",
              zero.find("x_wconf 0") != std::string::npos || zero.find("WC=\"0.00\"") != std::string::npos);
        ocr_render_free(renderer);
    }

    renderer = ocr_render_create(OCR_RENDER_PDF);
    ocr_render_set_pdfa(renderer, 1);
    ocr_render_begin(renderer);
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("archival mode produces a complete document",
          ocr_render_error(renderer) == nullptr && ocr_render_output_size(renderer) > 0);
    const std::string archival(ocr_render_output(renderer), size_t(ocr_render_output_size(renderer)));
    check("archival profile is explicit", archival.find("pdfaid:conformance=\"B\"") != std::string::npos &&
                                              archival.find("pdfaid:part=\"2\"") != std::string::npos);
    ocr_render_set_pdfa(renderer, 0);
    ocr_render_begin(renderer);
    page.image_path = "missing-ocr-render-test-image.png";
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("missing PDF image fails explicitly", ocr_render_error(renderer) != nullptr);
    check("missing image does not silently disappear", ocr_render_output_size(renderer) == 0);
    ocr_render_begin(renderer);
    page.image_path = nullptr;
    word.w = 201;
    ocr_render_add_page(renderer, &page);
    check("out-of-page text is rejected", ocr_render_error(renderer) != nullptr);
    ocr_render_free(renderer);
}

static void test_xml_escaping() {
    printf("\n=== XML escaping ===\n");

    ocr_render_word w = { "<b>Bold</b> & \"quoted\"", 0, 0, 100, 20, 1.0f };
    ocr_render_line l = { &w, 1, 0, 0, 100, 20 };
    ocr_render_page p = { &l, 1, 200, 100, nullptr };

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

static void test_pdf_document_limits() {
    printf("\n=== Whole-document PDF limits ===\n");
    ocr_render_word word = { "Reviewed page", 0, 0, 200, 20, 0.9f };
    ocr_render_line line = { &word, 1, 0, 0, 200, 20 };
    ocr_render_page page = { &line, 1, 200, 100, nullptr };
    auto * renderer = ocr_render_create(OCR_RENDER_PDF);
    ocr_render_begin(renderer);
    ocr_render_set_pdf_limits(renderer, 2, 128 * 1024 * 1024);
    ocr_render_add_page(renderer, &page);
    ocr_render_add_page(renderer, &page);
    check("configured page limit accepts its boundary", ocr_render_error(renderer) == nullptr);
    ocr_render_add_page(renderer, &page);
    check("page beyond configured limit fails before rendering",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "page limit"));
    check("over-limit page count never returns partial PDF", ocr_render_output_size(renderer) == 0);

    ocr_render_begin(renderer);
    ocr_render_set_pdf_limits(renderer, 32, 4096);
    for (int i = 0; i < 32; i++) ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("whole-document byte budget rejects accumulated pages",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "byte limit") &&
              ocr_render_output_size(renderer) == 0);

    ocr_render_begin(renderer);
    std::string text(1024 * 1024, 'x');
    word.text = text.c_str();
    for (int i = 0; i < 17; i++) ocr_render_add_page(renderer, &page);
    check("whole-document input budget rejects accumulated text",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "text/geometry limit") &&
              ocr_render_output_size(renderer) == 0);

    ocr_render_begin(renderer);
    text += 'x';
    word.text = text.c_str();
    ocr_render_add_page(renderer, &page);
    check("oversized single text region is rejected before layout",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "region exceeds 1 MiB"));

    ocr_render_begin(renderer);
    word.text = "Recovered after limit";
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("begin resets budgets and recovers after failed job",
          ocr_render_error(renderer) == nullptr && ocr_render_output_size(renderer) > 0);

    ocr_render_begin(renderer);
    ocr_render_set_pdf_limits(renderer, 5001, 4096);
    check("callers cannot raise hard memory/page ceilings", ocr_render_error(renderer) != nullptr);
    ocr_render_free(renderer);
}

struct pdf_sink_probe {
    std::string bytes;
    size_t calls = 0, largest_slice = 0, accept_bytes = size_t(-1);
};

static int write_pdf_probe(void * context, const unsigned char * data, size_t size) {
    auto & sink = *static_cast<pdf_sink_probe *>(context);
    sink.calls++;
    sink.largest_slice = std::max(sink.largest_slice, size);
    if (size > sink.accept_bytes - sink.bytes.size()) return 1;
    sink.bytes.append(reinterpret_cast<const char *>(data), size);
    return 0;
}

static void test_pdf_stream_writer() {
    printf("\n=== Synchronous PDF writer ===\n");
    ocr_render_word word = { "Reviewed streamed page", 0, 0, 200, 20, 0.9f };
    ocr_render_line line = { &word, 1, 0, 0, 200, 20 };
    ocr_render_page page = { &line, 1, 200, 100, nullptr };
    auto * renderer = ocr_render_create(OCR_RENDER_PDF);
    pdf_sink_probe sink;
    check("stream capability is explicit", (crispembed_ocr_render_capabilities() & OCR_RENDER_CAP_STREAM) != 0);
    ocr_render_begin(renderer);
    ocr_render_set_pdf_writer(renderer, write_pdf_probe, &sink, 256ull * 1024 * 1024);
    ocr_render_add_page(renderer, &page);
    ocr_render_add_page(renderer, &page);
    check("writer callbacks wait until synchronous end", sink.calls == 0);
    ocr_render_end(renderer);
    check("stream finalizes a complete two-page document", !ocr_render_error(renderer) &&
                                                               contains(sink.bytes.c_str(), "/Count 2") &&
                                                               contains(sink.bytes.c_str(), "%%EOF"));
    check("stream emits incrementally in bounded slices", sink.calls > 10 && sink.largest_slice <= 64 * 1024);
    check("stream retains no byte-returning PDF buffer", ocr_render_output_size(renderer) == 0);

    ocr_render_begin(renderer);
    sink = {};
    sink.accept_bytes = 36;
    ocr_render_set_pdf_writer(renderer, write_pdf_probe, &sink, 256ull * 1024 * 1024);
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("writer rejection fails without advertising output",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "writer failed") &&
              ocr_render_output_size(renderer) == 0);
    const size_t failed_calls = sink.calls;
    ocr_render_end(renderer);
    check("no callback after writer failure", sink.calls == failed_calls);

    ocr_render_begin(renderer); // must clear the old callback/context
    ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("begin resets sink and recovers the original byte API",
          !ocr_render_error(renderer) && ocr_render_output_size(renderer) > 0 && sink.calls == failed_calls);

    ocr_render_begin(renderer);
    sink = {};
    ocr_render_set_pdf_writer(renderer, write_pdf_probe, &sink, 4096);
    for (int i = 0; i < 32; i++) ocr_render_add_page(renderer, &page);
    ocr_render_end(renderer);
    check("stream cap rejects before writing beyond its limit",
          ocr_render_error(renderer) && contains(ocr_render_error(renderer), "byte limit") && !sink.bytes.empty() &&
              sink.bytes.size() <= 4096 && !contains(sink.bytes.c_str(), "%%EOF"));

    ocr_render_begin(renderer);
    ocr_render_set_pdf_writer(renderer, write_pdf_probe, &sink, 2147483649ull);
    check("stream cap cannot exceed 2 GiB", ocr_render_error(renderer) != nullptr);
    ocr_render_begin(renderer);
    ocr_render_set_pdf_writer(renderer, nullptr, &sink, 4096);
    check("stream requires an actual callback", ocr_render_error(renderer) != nullptr);
    ocr_render_free(renderer);
}

static int crispembed_test_main() {
    printf("OCR renderer — unit tests\n");

    test_text_renderer();
    test_hocr_renderer();
    test_alto_renderer();
    test_pdf_renderer();
    test_xml_escaping();
    test_render_failures_and_long_text();
    test_pdf_document_limits();
    test_pdf_stream_writer();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
