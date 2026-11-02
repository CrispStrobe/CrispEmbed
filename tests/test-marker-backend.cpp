// SPDX-License-Identifier: Apache-2.0
// Native input/geometry tests. Optional argv[1] selects a real PDFium library;
// no model, Python interpreter, server or network access is needed.
#include "marker_backend.h"
#include "core/clean_exit.h"
#include "core/image_out.h"
#include "../ggml/examples/stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace crisp_marker;

namespace {
int passed = 0, failed = 0;
void check(bool value, const char * label) {
    std::printf("[%s] %s\n", value ? "PASS" : "FAIL", label);
    value ? ++passed : ++failed;
}
void rejects(const std::function<void()> & call, const char * label) {
    bool threw = false;
    try {
        call();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, label);
}

struct scratch {
    std::filesystem::path path;
    scratch() {
        const auto time = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 100; ++i) {
            path = std::filesystem::current_path() /
                   ("marker-backend-test-" + std::to_string(time) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("cannot create test scratch directory");
    }
    ~scratch() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void write_bytes(const std::filesystem::path & path, const std::string & bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), std::streamsize(bytes.size()));
    if (!file) throw std::runtime_error("fixture write failed");
}

std::string stream_object(const std::string & stream) {
    return "<< /Length " + std::to_string(stream.size()) + " >>\nstream\n" + stream + "\nendstream";
}

std::string pdf_fixture() {
    const std::string content = "BT /F2 16 Tf 20 125 Td (Marker PDF) Tj ET\n"
                                "BT /F1 12 Tf 20 105 Td (Hello PDF \\200 \\201) Tj ET\n"
                                "BT /F3 10 Tf 20 85 Td (monospace) Tj ET\n"
                                "BT /F1 11 Tf 20 65 Td (left) Tj 100 0 Td (right) Tj ET\n"
                                "BT /F3 10 Tf 20 45 Td (    return 42) Tj ET\n"
                                "BT /F3 10 Tf 20 25 Td (a    b) Tj ET\n"
                                "BT /F1 10 Tf 0 155 Td (CLIPPED) Tj ET\n";
    const std::string resources = " /Resources << /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R >> >> ";
    const std::string bounds = " /MediaBox [0 0 240 160] /CropBox [10 10 230 150] ";
    const std::string cmap = "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
                             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
                             "/CMapName /MarkerUnicode def /CMapType 2 def\n"
                             "1 begincodespacerange <00> <ff> endcodespacerange\n"
                             "1 beginbfrange <00> <7f> <0000> endbfrange\n"
                             "2 beginbfchar <80> <03a9> <81> <d83dde03> endbfchar\n"
                             "endcmap CMapName currentdict /CMap defineresource pop end end";
    const std::vector<std::string> objects = {
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R 8 0 R] /Count 2 >>",
        "<< /Type /Page /Parent 2 0 R" + bounds + resources + "/Contents 7 0 R /Annots [11 0 R] >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding /ToUnicode 10 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding /ToUnicode 10 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Oblique /Encoding /WinAnsiEncoding /ToUnicode 10 0 R >>",
        stream_object(content),
        "<< /Type /Page /Parent 2 0 R" + bounds + resources + "/Rotate 90 /Contents 9 0 R >>",
        stream_object(content),
        stream_object(cmap),
        "<< /Type /Annot /Subtype /Link /Rect [20 103 80 118] /Border [0 0 0] /A << /S /URI /URI "
        "(https://example.org/marker) >> >>"
    };
    std::string bytes = "%PDF-1.7\n";
    std::vector<size_t> offsets{ 0 };
    for (size_t i = 0; i < objects.size(); ++i) {
        offsets.push_back(bytes.size());
        bytes += std::to_string(i + 1) + " 0 obj\n" + objects[i] + "\nendobj\n";
    }
    const size_t xref = bytes.size();
    bytes += "xref\n0 " + std::to_string(objects.size() + 1) + "\n0000000000 65535 f \n";
    for (size_t i = 1; i < offsets.size(); ++i) {
        char line[40];
        std::snprintf(line, sizeof(line), "%010llu 00000 n \n", static_cast<unsigned long long>(offsets[i]));
        bytes += line;
    }
    return bytes + "trailer\n<< /Size " + std::to_string(objects.size() + 1) + " /Root 1 0 R >>\nstartxref\n" +
           std::to_string(xref) + "\n%%EOF\n";
}

void geometry_tests(const scratch & files) {
    image pixels{ 28, 56, std::vector<uint8_t>(28 * 56 * 3, 255) };
    pixels.rgb[0] = 255;
    pixels.rgb[1] = 0;
    pixels.rgb[2] = 0;
    block region;
    region.bbox = { 0, 0, 28, 56 };
    region.angle = 90;
    const auto rotated = crop_image(pixels, region, 28, 56);
    check(rotated.width == 56 && rotated.height == 28, "crop quarter rotation swaps dimensions");
    const size_t target = size_t(27 * 56) * 3;
    check(rotated.rgb[target] == 255 && rotated.rgb[target + 1] == 0 && rotated.rgb[target + 2] == 0,
          "crop 90 degrees follows TeleOCR counterclockwise convention");

    image square{ 56, 56, std::vector<uint8_t>(56 * 56 * 3, 255) };
    region.bbox = { 0, 0, 56, 56 };
    region.angle = 0;
    region.polygon = { { 0, 0 }, { 56, 0 }, { 0, 56 } };
    const auto masked = crop_image(square, region, 56, 56);
    check(masked.rgb[(4 * 56 + 4) * 3] == 255 && masked.rgb[(50 * 56 + 50) * 3] == 0,
          "polygon masks at pixel centers with black exterior");

    image narrow{ 1000, 1, std::vector<uint8_t>(3000, 0) };
    region.bbox = { 0, 0, 1000, 1 };
    region.polygon.clear();
    const auto padded = crop_image(narrow, region, 1000, 1);
    check(padded.width == 1400 && padded.height == 28, "extreme crop is padded then upscaled with trained limits");
    check(padded.rgb[0] == 255, "crop aspect padding is white");

    region.bbox = { 0, 0, 0, 1 };
    rejects([&] { crop_image(narrow, region, 1000, 1); }, "empty crop rejects");
    region.bbox = { 0, 0, 1000, 1 };
    region.angle = 30;
    rejects([&] { crop_image(narrow, region, 1000, 1); }, "unsupported crop angle rejects");
    rejects([&] { crop_image(square, region, 0, 0); }, "invalid crop page size rejects");
    region.angle = 0;
    rejects([&] { crop_image(square, region, std::numeric_limits<double>::denorm_min(), 1); },
            "nonfinite crop scaling rejects before coordinate conversion");
    image broken{ 10, 10, {} };
    rejects([&] { encode_image_asset(broken, "broken.png"); }, "inconsistent RGB length rejects");
    rejects([&] { encode_image_asset(square, "../escape.png"); }, "asset path traversal rejects");

    const auto encoded = encode_image_asset(pixels, "owned.png");
    const std::string png(encoded.bytes.begin(), encoded.bytes.end());
    check(encoded.mime_type == "image/png" && png.find("document-image-extraction") != std::string::npos,
          "PNG asset carries truthful extraction provenance");
    write_bytes(files.path / "owned.png", png);
    const auto decoded = load_image((files.path / "owned.png").string(), 10000);
    check(decoded.width == pixels.width && decoded.height == pixels.height && decoded.rgb == pixels.rgb,
          "PNG encode and native decode preserve exact RGB bytes");
    rejects([&] { load_image((files.path / "owned.png").string(), 10); },
            "image dimension preflight enforces pixel limit");
    write_bytes(files.path / "bad.png", "not an image");
    rejects([&] { load_image((files.path / "bad.png").string(), 10000); }, "invalid encoded image rejects");

    // A real PNG eXIf chunk with IFD0 orientation 6 (clockwise 90 degrees).
    const uint8_t tiff[] = { 'I', 'I', 42, 0, 8, 0, 0, 0, 1, 0, 0x12, 1, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0 };
    std::string payload = "eXIf";
    payload.append(reinterpret_cast<const char *>(tiff), sizeof(tiff));
    std::string chunk;
    core_imgout::png_put_u32(chunk, sizeof(tiff));
    chunk += payload;
    core_imgout::png_put_u32(chunk,
                             core_imgout::png_crc(reinterpret_cast<const uint8_t *>(payload.data()), payload.size()));
    std::string oriented = png;
    oriented.insert(33, chunk);
    write_bytes(files.path / "oriented.png", oriented);
    const auto exif = load_image((files.path / "oriented.png").string(), 10000);
    const size_t corner = size_t(55) * 3;
    check(exif.width == 56 && exif.height == 28 && exif.rgb[corner + 1] == 0 && exif.rgb[corner + 2] == 0,
          "native image loading honors PNG EXIF orientation");

    const uint8_t rgba[] = { 255, 0, 0, 0, 0, 0, 255, 128 };
    std::string transparent;
    const auto collect = [](void * context, void * bytes, int length) {
        static_cast<std::string *>(context)->append(static_cast<const char *>(bytes), size_t(length));
    };
    if (!stbi_write_png_to_func(collect, &transparent, 2, 1, 4, rgba, 8))
        throw std::runtime_error("RGBA fixture encoding failed");
    write_bytes(files.path / "transparent.png", transparent);
    const auto composited = load_image((files.path / "transparent.png").string(), 100);
    check(composited.rgb == std::vector<uint8_t>({ 255, 255, 255, 127, 127, 255 }),
          "file RGBA is composited onto white identically to raw-image input");
}

void pdf_tests(const scratch & files, const std::string & library) {
    const auto input = files.path / "owned.pdf";
    write_bytes(input, pdf_fixture());
    pdf_provider provider(library, input.string(), "");
    check(provider.page_count() == 2, "native PDFium discovers two pages");
    const auto first = provider.read_page(0), rotated = provider.read_page(1);
    check(std::abs(first.width - 220) < 0.01 && std::abs(first.height - 140) < 0.01,
          "PDF coordinates use visible CropBox");
    check(std::abs(rotated.width - 140) < 0.01 && std::abs(rotated.height - 220) < 0.01,
          "PDF intrinsic rotation changes page dimensions once");
    std::string full;
    bool bold = false, mono = false, italic = false, linked = false, valid_boxes = true, words = false;
    for (const auto & row : first.text_lines) {
        full += text_of(row) + "\n";
        words = words || row.spans.size() > 1;
        for (const auto & part : row.spans) {
            bold = bold || part.bold;
            mono = mono || part.monospace;
            italic = italic || part.italic;
            linked = linked || part.url == "https://example.org/marker";
            valid_boxes = valid_boxes && part.bbox.x0 >= 0 && part.bbox.y0 >= 0 && part.bbox.x1 <= first.width &&
                          part.bbox.y1 <= first.height;
        }
    }
    check(full.find("Hello PDF") != std::string::npos && full.find("\xce\xa9") != std::string::npos,
          "PDF text extraction preserves ASCII and mapped Unicode");
    check(full.find("\xf0\x9f\x98\x83") != std::string::npos, "PDF text extraction preserves supplementary Unicode");
    check(full.find("CLIPPED") == std::string::npos, "PDF text outside CropBox is excluded");
    check(full.find("    return 42") != std::string::npos, "PDF leading code indentation survives extraction");
    check(full.find("a    b") != std::string::npos, "PDF repeated internal code spaces survive extraction");
    check(bold && mono && italic, "PDF spans retain bold monospace and italic styles");
    check(words && valid_boxes, "PDF spans retain word geometry inside page bounds");
    check(linked, "PDF URI link annotation maps to text spans");
    bool rotated_boxes = !rotated.text_lines.empty();
    for (const auto & row : rotated.text_lines)
        rotated_boxes = rotated_boxes && row.bbox.x0 >= 0 && row.bbox.y0 >= 0 && row.bbox.x1 <= rotated.width &&
                        row.bbox.y1 <= rotated.height;
    check(rotated_boxes, "PDF rotated character boxes align with rotated page");
    const auto dimensions_only = provider.read_page(1, false);
    check(dimensions_only.text_lines.empty() && dimensions_only.width == rotated.width &&
              dimensions_only.height == rotated.height,
          "force-OCR page metadata does not read or depend on embedded text");
    const auto raster = provider.render_page(1, 144, 200000);
    check(raster.width == 280 && raster.height == 440 && raster.rgb.size() == 280 * 440 * 3,
          "PDF rendering uses requested DPI and intrinsic rotation");
    check(std::find_if(raster.rgb.begin(), raster.rgb.end(), [](uint8_t v) { return v < 100; }) != raster.rgb.end(),
          "PDF rendering contains document glyphs");
    rejects([&] { provider.read_page(2); }, "PDF out-of-range page rejects");
    rejects([&] { provider.render_page(0, 144, 10); }, "PDF rendering pixel limit rejects before bitmap allocation");
    rejects([&] { provider.render_page(0, 0, 200000); }, "PDF invalid DPI rejects");
    rejects([&] { pdf_provider bad(library, (files.path / "missing.pdf").string(), ""); },
            "PDF missing file reports error");
    rejects([&] { pdf_provider different(input.string(), input.string(), ""); },
            "different active PDFium library rejects without deadlock");
    const auto concurrent = [&]() {
        pdf_provider second(library, input.string(), "");
        return second.read_page(0).text_lines.size() == first.text_lines.size();
    };
    auto left = std::async(std::launch::async, concurrent), right = std::async(std::launch::async, concurrent);
    check(left.get() && right.get(), "concurrent providers share a serialized native PDFium lifetime");
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    try {
        scratch files;
        geometry_tests(files);
        rejects([&] { pdf_provider missing("", "owned.pdf", ""); }, "PDFium requires explicit library path");
        if (argc > 1)
            pdf_tests(files, argv[1]);
        else
            std::puts("[SKIP] PDFium integration cases: pass an explicit native PDFium library as argv[1]");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[FAIL] unexpected exception: %s\n", error.what());
        ++failed;
    }
    std::printf("marker backend: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
