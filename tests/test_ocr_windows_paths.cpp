// Execute the real Windows image loader/writer and PDF image path. No models.
#include "ocr_render.h"
#include "core/clean_exit.h"
#ifndef OCR_PATHS_STANDALONE
#include "image_preprocess.h"
#include "scan_cleanup.h"
#endif
#include <windows.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int stbi_info(const char *, int *, int *, int *);
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
int stbi_write_png(const char *, int, int, int, const void *, int);
}

#ifndef OCR_PATHS_STANDALONE
// The production preprocessing TU references optional deskew. This path test
// keeps it disabled; a mistaken call must fail rather than use a fake result.
extern "C" int scan_cleanup_deskew_rgb(const uint8_t *, int, int, int, float, uint8_t **, int *, int *) {
    std::abort();
}
extern "C" void scan_cleanup_free_image(uint8_t *) { std::abort(); }
#endif

namespace fs = std::filesystem;
static int checks = 0;
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
    checks++;
}
static std::vector<unsigned char> read_bytes(const fs::path & path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "could not read fixture/output through its wide filesystem path");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
static bool decoded_equals(const std::string & path, const std::vector<unsigned char> & rgb) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(
        stbi_load(path.c_str(), &width, &height, &channels, 3), stbi_image_free);
    return pixels && width == 4 && height == 3 && std::equal(rgb.begin(), rgb.end(), pixels.get());
}
static void export_pdf(const std::string & image, const fs::path & output) {
    std::unique_ptr<ocr_renderer, decltype(&ocr_render_free)> renderer(ocr_render_create(OCR_RENDER_PDF), ocr_render_free);
    require(bool(renderer), "renderer allocation failed");
    ocr_render_begin(renderer.get());
    ocr_render_set_page_size_points(renderer.get(), 288, 216);
    ocr_render_word word{"Reviewed Unicode path", 0, 0, 4, 1, 1.0f};
    ocr_render_line line{&word, 1, 0, 0, 4, 1};
    ocr_render_page page{&line, 1, 4, 3, image.c_str()};
    ocr_render_add_page(renderer.get(), &page);
    ocr_render_end(renderer.get());
    const char * error = ocr_render_error(renderer.get());
    if (error) throw std::runtime_error(error);
    const char * bytes = ocr_render_output(renderer.get());
    const auto size = ocr_render_output_size(renderer.get());
    require(bytes && size > 0, "renderer produced no PDF");
    std::ofstream file(output, std::ios::binary);
    require(bool(file), "could not create a Unicode-named PDF output");
    file.write(bytes, static_cast<std::streamsize>(size));
    file.close();
    require(bool(file), "PDF output did not finish writing");
}

static int crispembed_test_main(int argc, char ** argv) {
    try {
        require(argc == 2, "supply one ASCII artifact directory");
        const fs::path root = fs::u8path(argv[1]);
        const fs::path nested = root / fs::u8path(u8"Pr\u00fcfung-\u6587\u4ef6-\U0001f4f7");
        fs::create_directories(nested);
        const fs::path ascii = root / "source.png";
        const fs::path unicode = nested / fs::u8path(u8"Seite-\u00e9-\u6771\u4eac-\U0001f600.png");
        const fs::path written = nested / fs::u8path(u8"Kopie-\u00fc-\u6587-\U0001f600.png");
        std::vector<unsigned char> rgb;
        for (int y = 0; y < 3; y++) for (int x = 0; x < 4; x++) {
            rgb.push_back(static_cast<unsigned char>(x * 60 + y * 10));
            rgb.push_back(static_cast<unsigned char>(y * 80 + x * 5));
            rgb.push_back(static_cast<unsigned char>(240 - x * 40 - y * 15));
        }
        require(stbi_write_png(ascii.u8string().c_str(), 4, 3, 3, rgb.data(), 12) != 0, "ASCII PNG write failed");
        require(decoded_equals(ascii.u8string(), rgb), "ASCII decoded pixels differ");
        fs::copy_file(ascii, unicode, fs::copy_options::overwrite_existing);
        const auto original = read_bytes(unicode);
        int width = 0, height = 0, channels = 0;
        const bool info = stbi_info(unicode.u8string().c_str(), &width, &height, &channels) != 0;
        const bool load = decoded_equals(unicode.u8string(), rgb);
        const bool write = stbi_write_png(written.u8string().c_str(), 4, 3, 3, rgb.data(), 12) != 0;
#ifdef OCR_PATHS_LEGACY
        // An explicit environment requirement keeps the historical failure
        // observable instead of silently passing on a UTF-8 ANSI-code-page host.
        require(GetACP() != CP_UTF8, "negative control requires a non-UTF-8 Windows ANSI code page");
        require(!info && !load && !write, "historical ANSI path unexpectedly accepted the UTF-8 path");
#else
        require(info && width == 4 && height == 3, "UTF-8 image dimensions are unavailable");
        require(load, "UTF-8 decoded pixels differ from the ASCII input");
        require(write && decoded_equals(written.u8string(), rgb), "UTF-8 PNG output was lost or changed");
        export_pdf(ascii.u8string(), root / "ascii.pdf");
        export_pdf(unicode.u8string(), nested / fs::u8path(u8"Gepr\u00fcft-\u6587-\U0001f600.pdf"));
#ifndef OCR_PATHS_STANDALONE
        image_preproc::config config;
        config.patch_size = 1; config.merge_size = 1; config.temporal_patch_size = 1;
        config.min_pixels = 12; config.max_pixels = 12; config.deskew = 0;
        image_preproc::result a, b;
        require(image_preproc::preprocess_file(ascii.u8string().c_str(), config, a), "ASCII preprocessing failed");
        require(image_preproc::preprocess_file(unicode.u8string().c_str(), config, b), "UTF-8 preprocessing failed");
        require(a.n_patches == 12 && b.n_patches == 12 && a.row_dim == b.row_dim && a.patches == b.patches,
                "UTF-8 preprocessing changed the decoded numeric input");
#endif
#endif
        require(read_bytes(unicode) == original, "source bytes changed during decoding/export");
        std::ofstream receipt(root / "native.json", std::ios::binary);
        receipt << "{\"passed\":true,\"checks\":" << checks << ",\"windows_ansi_codepage\":" << GetACP()
#ifdef OCR_PATHS_LEGACY
                << ",\"historical_negative_control\":true"
#else
                << ",\"historical_negative_control\":false"
#endif
#ifdef OCR_PATHS_STANDALONE
                << ",\"actual_production_translation_units\":false"
#else
                << ",\"actual_production_translation_units\":true"
#endif
                << "}\n";
        receipt.close();
        require(bool(receipt), "could not retain native path evidence");
        std::printf("Windows UTF-8 path checks passed: %d\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Windows UTF-8 path check failed: %s\n", error.what());
        return 1;
    }
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
