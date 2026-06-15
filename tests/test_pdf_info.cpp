// tests/test_pdf_info.cpp — unit tests for PDF DPI profiling.
//
// Creates synthetic PDFs with known image dimensions and page sizes,
// then verifies that pdf_page_dpi() reports the correct DPI.

#include "pdf_info.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static int n_pass = 0, n_fail = 0;

static void check(const char * name, bool cond) {
    if (cond) { printf("  %s[PASS]%s %s\n", GREEN, RESET, name); n_pass++; }
    else      { printf("  %s[FAIL]%s %s\n", RED, RESET, name); n_fail++; }
}

// ---------------------------------------------------------------------------
// Generate a minimal valid PDF with one page and one image XObject.
//
// The image is a 1x1 white JPEG (smallest valid JPEG). What matters
// for DPI calculation is:
//   - /Width, /Height in the image XObject dict (pixel dimensions)
//   - Page /MediaBox (page size in points)
//   - CTM in content stream (display size in points)
// ---------------------------------------------------------------------------

static std::string make_test_pdf(int img_w, int img_h,
                                  float page_w_pt, float page_h_pt,
                                  float display_w_pt, float display_h_pt) {
    std::string pdf;
    char buf[512];

    // Minimal JPEG: 1x1 white pixel (but we set /Width /Height in the dict)
    // This is a valid 1x1 JPEG (SOI, APP0, SOF0, SOS, EOI)
    static const uint8_t mini_jpeg[] = {
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xC0, 0x00, 0x0B,
        0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11,
        0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01,
        0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
        0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0x7B, 0x40, 0xFF, 0xD9
    };
    int jpeg_len = sizeof(mini_jpeg);

    // Content stream: CTM + image paint
    std::string content;
    snprintf(buf, sizeof(buf), "q\n%.2f 0 0 %.2f 0 0 cm\n/Im1 Do\nQ\n",
             display_w_pt, display_h_pt);
    content = buf;

    // Object layout:
    //   1: Catalog
    //   2: Pages
    //   3: Page
    //   4: Image XObject
    //   5: Content stream

    struct { int id; long offset; } objs[5];
    int n_obj = 0;

    pdf = "%PDF-1.4\n";

    // Obj 1: Catalog
    objs[n_obj] = {1, (long)pdf.size()}; n_obj++;
    pdf += "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";

    // Obj 2: Pages
    objs[n_obj] = {2, (long)pdf.size()}; n_obj++;
    pdf += "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";

    // Obj 3: Page
    objs[n_obj] = {3, (long)pdf.size()}; n_obj++;
    snprintf(buf, sizeof(buf),
        "3 0 obj\n<< /Type /Page /Parent 2 0 R\n"
        "   /MediaBox [0 0 %.2f %.2f]\n"
        "   /Contents 5 0 R\n"
        "   /Resources << /XObject << /Im1 4 0 R >> >>\n"
        ">>\nendobj\n",
        page_w_pt, page_h_pt);
    pdf += buf;

    // Obj 4: Image XObject
    objs[n_obj] = {4, (long)pdf.size()}; n_obj++;
    snprintf(buf, sizeof(buf),
        "4 0 obj\n<< /Type /XObject /Subtype /Image\n"
        "   /Width %d /Height %d\n"
        "   /ColorSpace /DeviceRGB /BitsPerComponent 8\n"
        "   /Filter /DCTDecode /Length %d >>\nstream\n",
        img_w, img_h, jpeg_len);
    pdf += buf;
    pdf.append((const char *)mini_jpeg, jpeg_len);
    pdf += "\nendstream\nendobj\n";

    // Obj 5: Content stream
    objs[n_obj] = {5, (long)pdf.size()}; n_obj++;
    snprintf(buf, sizeof(buf),
        "5 0 obj\n<< /Length %d >>\nstream\n", (int)content.size());
    pdf += buf;
    pdf += content;
    pdf += "endstream\nendobj\n";

    // Xref table
    long xref_offset = (long)pdf.size();
    pdf += "xref\n";
    snprintf(buf, sizeof(buf), "0 %d\n", n_obj + 1);
    pdf += buf;
    pdf += "0000000000 65535 f \n";
    for (int i = 0; i < n_obj; i++) {
        snprintf(buf, sizeof(buf), "%010ld 00000 n \n", objs[i].offset);
        pdf += buf;
    }

    // Trailer
    snprintf(buf, sizeof(buf),
        "trailer\n<< /Size %d /Root 1 0 R >>\n"
        "startxref\n%ld\n%%%%EOF\n",
        n_obj + 1, xref_offset);
    pdf += buf;

    return pdf;
}

static std::string write_test_pdf(const std::string & content, const char * name) {
    std::string path = std::string("/tmp/test_") + name + ".pdf";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return "";
    fwrite(content.data(), 1, content.size(), f);
    fclose(f);
    return path;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_known_dpi(const char * label, int img_w, int img_h,
                            float page_w_pt, float page_h_pt,
                            float display_w_pt, float display_h_pt,
                            float expected_dpi) {
    printf("\n=== %s (expected %.0f DPI) ===\n", label, expected_dpi);

    std::string content = make_test_pdf(img_w, img_h, page_w_pt, page_h_pt,
                                         display_w_pt, display_h_pt);
    std::string path = write_test_pdf(content, label);
    check("PDF file created", !path.empty());

    pdf_page_dpi_result result = {};
    int ret = pdf_page_dpi(path.c_str(), 0, &result);
    check("pdf_page_dpi returns 0", ret == 0);

    if (ret == 0) {
        printf("  Computed: %.1f DPI (min=%.1f, max=%.1f, n_images=%d)\n",
               result.dpi, result.dpi_min, result.dpi_max, result.n_images);
        printf("  Page size: %.0f x %.0f pt\n", result.page_width_pt, result.page_height_pt);

        float tolerance = expected_dpi * 0.05f; // 5%
        float diff = std::abs(result.dpi - expected_dpi);
        char msg[128];
        snprintf(msg, sizeof(msg), "DPI within 5%% (%.1f vs expected %.1f, diff=%.1f)",
                 result.dpi, expected_dpi, diff);
        check(msg, diff <= tolerance);
        check("n_images == 1", result.n_images == 1);
    }

    remove(path.c_str());
}

static void test_no_images() {
    printf("\n=== PDF with no images ===\n");
    // Minimal PDF: just a page with no image
    std::string pdf = "%PDF-1.4\n";
    pdf += "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
    pdf += "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
    pdf += "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n";
    long xref_off = (long)pdf.size();
    pdf += "xref\n0 4\n";
    pdf += "0000000000 65535 f \n";
    // We need real offsets... let me just write the file and test
    // Actually the xref offsets must match. Simplify:
    char buf[128];
    std::string real_pdf = "%PDF-1.4\n";
    long off1 = (long)real_pdf.size();
    real_pdf += "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
    long off2 = (long)real_pdf.size();
    real_pdf += "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n";
    long off3 = (long)real_pdf.size();
    real_pdf += "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n";
    long xo = (long)real_pdf.size();
    real_pdf += "xref\n0 4\n";
    real_pdf += "0000000000 65535 f \n";
    snprintf(buf, sizeof(buf), "%010ld 00000 n \n", off1); real_pdf += buf;
    snprintf(buf, sizeof(buf), "%010ld 00000 n \n", off2); real_pdf += buf;
    snprintf(buf, sizeof(buf), "%010ld 00000 n \n", off3); real_pdf += buf;
    snprintf(buf, sizeof(buf), "trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n%ld\n%%%%EOF\n", xo);
    real_pdf += buf;

    std::string path = write_test_pdf(real_pdf, "no_images");
    check("PDF file created", !path.empty());

    pdf_page_dpi_result result = {};
    int ret = pdf_page_dpi(path.c_str(), 0, &result);
    check("returns 1 (no images)", ret == 1);
    check("n_images == 0", result.n_images == 0);

    remove(path.c_str());
}

static void test_bad_input() {
    printf("\n=== Bad input handling ===\n");

    pdf_page_dpi_result result = {};
    check("null path returns 1", pdf_page_dpi(nullptr, 0, &result) == 1);
    check("nonexistent file returns 1", pdf_page_dpi("/no/such/file.pdf", 0, &result) == 1);
    check("null result returns 1", pdf_page_dpi("/tmp/x.pdf", 0, nullptr) == 1);
    check("negative page returns 1", pdf_page_dpi("/tmp/x.pdf", -1, &result) == 1);

    // Non-PDF file
    std::string path = "/tmp/test_not_pdf.txt";
    FILE * f = fopen(path.c_str(), "w");
    if (f) { fprintf(f, "Hello world\n"); fclose(f); }
    check("non-PDF file returns 1", pdf_page_dpi(path.c_str(), 0, &result) == 1);
    remove(path.c_str());

    pdf_dpi_free(nullptr); // should not crash
    check("pdf_dpi_free(NULL) no crash", true);
}

static void test_all_pages() {
    printf("\n=== pdf_all_pages_dpi ===\n");

    // Create a PDF and test the all-pages API
    // 300 DPI: 2550x3300 pixels on 8.5x11" (612x792 pt)
    std::string content = make_test_pdf(2550, 3300, 612, 792, 612, 792);
    std::string path = write_test_pdf(content, "all_pages");

    int n = 0;
    pdf_page_dpi_result * results = pdf_all_pages_dpi(path.c_str(), &n);
    check("pdf_all_pages_dpi succeeds", results != nullptr);
    check("n_pages == 1", n == 1);

    if (results && n > 0) {
        printf("  Page 0: %.1f DPI\n", results[0].dpi);
        check("DPI ~300", std::abs(results[0].dpi - 300.0f) < 15.0f);
    }

    pdf_dpi_free(results);
    remove(path.c_str());
}

int main() {
    printf("PDF DPI Profiling — unit tests\n");

    // Test various known DPIs:
    // DPI = pixel_dim / (display_pt / 72)

    // 72 DPI: 612x792 pixels displayed at 612x792 pt (= 8.5x11")
    test_known_dpi("72dpi", 612, 792, 612, 792, 612, 792, 72.0f);

    // 150 DPI: 1275x1650 pixels displayed at 612x792 pt
    test_known_dpi("150dpi", 1275, 1650, 612, 792, 612, 792, 150.0f);

    // 300 DPI: 2550x3300 pixels displayed at 612x792 pt
    test_known_dpi("300dpi", 2550, 3300, 612, 792, 612, 792, 300.0f);

    // 600 DPI: 5100x6600 pixels displayed at 612x792 pt
    test_known_dpi("600dpi", 5100, 6600, 612, 792, 612, 792, 600.0f);

    // Non-standard: A4 page (595x842 pt) with 200 DPI image
    // 200 DPI: 1653x2339 pixels displayed at 595x842 pt
    test_known_dpi("200dpi_a4", 1653, 2339, 595, 842, 595, 842, 200.0f);

    // Small image on big page (image doesn't fill page):
    // 300 DPI image displayed at 306x396 pt (half page) on 612x792 page
    test_known_dpi("300dpi_half", 1275, 1650, 612, 792, 306, 396, 300.0f);

    test_no_images();
    test_bad_input();
    test_all_pages();

    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
