// SPDX-License-Identifier: Apache-2.0
// Native document input and TeleOCR backend. See NOTICE.marker and NOTICE.teleocr.
// PDFium public ABI: public/fpdfview.h and public/fpdf_text.h (BSD-3-Clause).
// Image/task conventions match CrispEmbed tools/teleocr_parse.py; no Python
// interpreter, Surya model, external executable or network request is used.
#include "marker_backend.h"

#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/image_out.h"
#include "image_preprocess.h"
#include "qwen2vl_ocr.h"
#include "../ggml/examples/stb_image.h"
#include "../ggml/examples/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define MARKER_PDF_CALL __stdcall
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
#include <dlfcn.h>
#define MARKER_PDF_CALL
#else
#define MARKER_PDF_CALL
#endif

namespace crisp_marker {
namespace {

constexpr size_t hard_pixel_limit = 64000000;
constexpr size_t image_file_limit = 256 * 1024 * 1024;
constexpr int dimension_limit = 32768;
constexpr int pdf_char_limit = 1000000;

[[noreturn]] void fail(const std::string & reason) {
    throw std::runtime_error("marker: " + reason);
}

size_t pixels_checked(int width, int height, size_t limit) {
    if (width <= 0 || height <= 0 || width > dimension_limit || height > dimension_limit || limit == 0)
        fail("invalid or excessive image dimensions");
    const size_t count = size_t(width) * size_t(height);
    if (count > std::min(limit, hard_pixel_limit)) fail("image exceeds the pixel limit");
    return count;
}

void check_image(const image & value, size_t limit = hard_pixel_limit) {
    const size_t count = pixels_checked(value.width, value.height, limit);
    if (value.rgb.size() != count * 3) fail("RGB buffer size does not match its dimensions");
}

image resize_image(const image & input, int width, int height) {
    check_image(input);
    const size_t count = pixels_checked(width, height, hard_pixel_limit);
    if (width == input.width && height == input.height) return input;
    image out;
    out.width = width;
    out.height = height;
    out.rgb.resize(count * 3);
    std::vector<float> resized(count * 3);
    image_preproc::resize_bicubic_pil_u8_hwc(input.rgb.data(), input.height, input.width, resized.data(), height, width,
                                             3);
    for (size_t i = 0; i < out.rgb.size(); ++i)
        out.rgb[i] = uint8_t(std::clamp(std::floor(resized[i] + 0.5f), 0.0f, 255.0f));
    return out;
}

// EXIF orientation numbers also express the exact lossless quarter rotations
// used for TeleOCR crops (8 = counterclockwise 90 degrees).
image orient_image(const image & input, int orientation) {
    check_image(input);
    if (orientation == 1) return input;
    if (orientation < 1 || orientation > 8) fail("invalid image orientation");
    image out;
    const bool transpose = orientation >= 5;
    out.width = transpose ? input.height : input.width;
    out.height = transpose ? input.width : input.height;
    out.rgb.resize(input.rgb.size());
    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            int dx = x, dy = y;
            switch (orientation) {
            case 2:
                dx = input.width - 1 - x;
                break;
            case 3:
                dx = input.width - 1 - x;
                dy = input.height - 1 - y;
                break;
            case 4:
                dy = input.height - 1 - y;
                break;
            case 5:
                dx = y;
                dy = x;
                break;
            case 6:
                dx = input.height - 1 - y;
                dy = x;
                break;
            case 7:
                dx = input.height - 1 - y;
                dy = input.width - 1 - x;
                break;
            case 8:
                dx = y;
                dy = input.width - 1 - x;
                break;
            default:
                break;
            }
            const size_t source = (size_t(y) * input.width + x) * 3;
            const size_t target = (size_t(dy) * out.width + dx) * 3;
            std::copy_n(input.rgb.data() + source, 3, out.rgb.data() + target);
        }
    }
    return out;
}

int tiff_orientation(const uint8_t * data, size_t size) {
    if (size < 8) return 1;
    const bool little = data[0] == 'I' && data[1] == 'I';
    if (!little && !(data[0] == 'M' && data[1] == 'M')) return 1;
    const auto u16 = [&](size_t p) -> uint32_t {
        if (p > size || size - p < 2) return 0;
        return little ? uint32_t(data[p]) | (uint32_t(data[p + 1]) << 8)
                      : (uint32_t(data[p]) << 8) | uint32_t(data[p + 1]);
    };
    const auto u32 = [&](size_t p) -> uint32_t {
        if (p > size || size - p < 4) return 0;
        return little ? u16(p) | (u16(p + 2) << 16) : (u16(p) << 16) | u16(p + 2);
    };
    if (u16(2) != 42) return 1;
    const size_t start = u32(4);
    if (start > size || size - start < 2) return 1;
    const size_t count = u16(start);
    if (count > (size - start - 2) / 12) return 1;
    for (size_t i = 0; i < count; ++i) {
        const size_t entry = start + 2 + 12 * i;
        if (u16(entry) == 0x0112 && u16(entry + 2) == 3 && u32(entry + 4) == 1) {
            const int orientation = int(u16(entry + 8));
            return orientation >= 1 && orientation <= 8 ? orientation : 1;
        }
    }
    return 1;
}

int encoded_orientation(const std::vector<uint8_t> & data) {
    // JPEG APP1 Exif; segment bounds are checked before consulting TIFF offsets.
    if (data.size() >= 4 && data[0] == 0xff && data[1] == 0xd8) {
        size_t p = 2;
        while (p + 4 <= data.size()) {
            if (data[p++] != 0xff) break;
            while (p < data.size() && data[p] == 0xff) ++p;
            if (p >= data.size()) break;
            const uint8_t marker = data[p++];
            if (marker == 0xda || marker == 0xd9) break;
            if (marker == 1 || (marker >= 0xd0 && marker <= 0xd7)) continue;
            if (p + 2 > data.size()) break;
            const size_t length = (size_t(data[p]) << 8) | data[p + 1];
            if (length < 2 || length > data.size() - p) break;
            if (marker == 0xe1 && length >= 8 && std::memcmp(data.data() + p + 2, "Exif\0\0", 6) == 0)
                return tiff_orientation(data.data() + p + 8, length - 8);
            p += length;
        }
    }
    // PNG eXIf uses a TIFF payload without the JPEG Exif prefix.
    static const uint8_t signature[] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    if (data.size() >= 8 && std::memcmp(data.data(), signature, 8) == 0) {
        for (size_t p = 8; p + 12 <= data.size();) {
            const size_t length =
                (size_t(data[p]) << 24) | (size_t(data[p + 1]) << 16) | (size_t(data[p + 2]) << 8) | data[p + 3];
            if (length > data.size() - p - 12) break;
            if (std::memcmp(data.data() + p + 4, "eXIf", 4) == 0) return tiff_orientation(data.data() + p + 8, length);
            p += length + 12;
        }
    }
    return 1;
}

std::string trim_ascii(const std::string & value) {
    const size_t first = value.find_first_not_of(" \t\r\n\f");
    if (first == std::string::npos) return {};
    return value.substr(first, value.find_last_not_of(" \t\r\n\f") - first + 1);
}

std::string lower_ascii(std::string value) {
    for (char & c : value)
        if (c >= 'A' && c <= 'Z') c = char(c + ('a' - 'A'));
    return value;
}

void append_utf8(std::string & target, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp < 0x80)
        target.push_back(char(cp));
    else if (cp < 0x800) {
        target.push_back(char(0xc0 | (cp >> 6)));
        target.push_back(char(0x80 | (cp & 0x3f)));
    } else if (cp < 0x10000) {
        target.push_back(char(0xe0 | (cp >> 12)));
        target.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
        target.push_back(char(0x80 | (cp & 0x3f)));
    } else {
        target.push_back(char(0xf0 | (cp >> 18)));
        target.push_back(char(0x80 | ((cp >> 12) & 0x3f)));
        target.push_back(char(0x80 | ((cp >> 6) & 0x3f)));
        target.push_back(char(0x80 | (cp & 0x3f)));
    }
}

box joined(const box & a, const box & b) {
    return { std::min(a.x0, b.x0), std::min(a.y0, b.y0), std::max(a.x1, b.x1), std::max(a.y1, b.y1) };
}

// PDFium's public ABI is not thread-safe. Every PDFium call, including module
// initialization/destruction, uses this process-wide lock. It intentionally
// outlives provider destructors at program shutdown.
std::mutex & pdf_mutex() {
    static auto * value = new std::mutex;
    return *value;
}

struct pdf_rectangle {
    float left, top, right, bottom;
};

struct pdf_api {
#if defined(_WIN32)
    HMODULE module = nullptr;
#else
    void * module = nullptr;
#endif
    std::string path;
    bool initialized = false;
#define PDF_FUNCTION(ret, name, ...)                                                                                   \
    using name##_fn = ret(MARKER_PDF_CALL *)(__VA_ARGS__);                                                             \
    name##_fn name = nullptr
    PDF_FUNCTION(void, FPDF_InitLibrary);
    PDF_FUNCTION(void, FPDF_DestroyLibrary);
    PDF_FUNCTION(void *, FPDF_LoadDocument, const char *, const char *);
    PDF_FUNCTION(void, FPDF_CloseDocument, void *);
    PDF_FUNCTION(unsigned long, FPDF_GetLastError);
    PDF_FUNCTION(int, FPDF_GetPageCount, void *);
    PDF_FUNCTION(void *, FPDF_LoadPage, void *, int);
    PDF_FUNCTION(void, FPDF_ClosePage, void *);
    PDF_FUNCTION(double, FPDF_GetPageWidth, void *);
    PDF_FUNCTION(double, FPDF_GetPageHeight, void *);
    PDF_FUNCTION(int, FPDF_GetPageBoundingBox, void *, pdf_rectangle *);
    PDF_FUNCTION(int, FPDF_PageToDevice, void *, int, int, int, int, int, double, double, int *, int *);
    PDF_FUNCTION(void *, FPDFText_LoadPage, void *);
    PDF_FUNCTION(void, FPDFText_ClosePage, void *);
    PDF_FUNCTION(int, FPDFText_CountChars, void *);
    PDF_FUNCTION(unsigned int, FPDFText_GetUnicode, void *, int);
    PDF_FUNCTION(void *, FPDFText_GetTextObject, void *, int);
    PDF_FUNCTION(int, FPDFText_IsGenerated, void *, int);
    PDF_FUNCTION(int, FPDFText_GetCharOrigin, void *, int, double *, double *);
    PDF_FUNCTION(int, FPDFText_GetCharBox, void *, int, double *, double *, double *, double *);
    PDF_FUNCTION(double, FPDFText_GetFontSize, void *, int);
    PDF_FUNCTION(unsigned long, FPDFText_GetFontInfo, void *, int, void *, unsigned long, int *);
    PDF_FUNCTION(int, FPDFText_GetFontWeight, void *, int);
    PDF_FUNCTION(int, FPDFLink_Enumerate, void *, int *, void **);
    PDF_FUNCTION(void *, FPDFLink_GetAction, void *);
    PDF_FUNCTION(int, FPDFLink_GetAnnotRect, void *, pdf_rectangle *);
    PDF_FUNCTION(unsigned long, FPDFAction_GetURIPath, void *, void *, void *, unsigned long);
    PDF_FUNCTION(void *, FPDFBitmap_CreateEx, int, int, int, void *, int);
    PDF_FUNCTION(void, FPDFBitmap_Destroy, void *);
    PDF_FUNCTION(void, FPDF_RenderPageBitmap, void *, void *, int, int, int, int, int, int);
#undef PDF_FUNCTION

    template <class T> void bind(T & function, const char * name, bool required = true) {
#if defined(_WIN32)
        function = reinterpret_cast<T>(GetProcAddress(module, name));
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
        function = reinterpret_cast<T>(dlsym(module, name));
#else
        (void)name;
#endif
        if (!function && required) fail(std::string("PDFium is missing the required API ") + name);
    }

    void close_module() noexcept {
        if (!module) return;
#if defined(_WIN32)
        FreeLibrary(module);
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
        dlclose(module);
#endif
        module = nullptr;
    }

    explicit pdf_api(const std::string & canonical_path) : path(canonical_path) {
        // The caller holds pdf_mutex() during construction.
#if defined(_WIN32)
        module = LoadLibraryExW(std::filesystem::u8path(path).c_str(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
#elif (defined(__unix__) || defined(__APPLE__)) && !defined(__EMSCRIPTEN__)
        module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#else
        fail("dynamic PDFium loading is unsupported on this platform");
#endif
        if (!module) fail("could not load the explicitly selected PDFium library");
        try {
#define PDF_BIND(name) bind(name, #name)
            PDF_BIND(FPDF_InitLibrary);
            PDF_BIND(FPDF_DestroyLibrary);
            PDF_BIND(FPDF_LoadDocument);
            PDF_BIND(FPDF_CloseDocument);
            PDF_BIND(FPDF_GetLastError);
            PDF_BIND(FPDF_GetPageCount);
            PDF_BIND(FPDF_LoadPage);
            PDF_BIND(FPDF_ClosePage);
            PDF_BIND(FPDF_GetPageWidth);
            PDF_BIND(FPDF_GetPageHeight);
            PDF_BIND(FPDF_GetPageBoundingBox);
            PDF_BIND(FPDF_PageToDevice);
            PDF_BIND(FPDFText_LoadPage);
            PDF_BIND(FPDFText_ClosePage);
            PDF_BIND(FPDFText_CountChars);
            PDF_BIND(FPDFText_GetUnicode);
            PDF_BIND(FPDFText_GetCharBox);
            PDF_BIND(FPDFText_GetFontSize);
            bind(FPDFText_GetTextObject, "FPDFText_GetTextObject", false);
            bind(FPDFText_IsGenerated, "FPDFText_IsGenerated", false);
            bind(FPDFText_GetCharOrigin, "FPDFText_GetCharOrigin", false);
            PDF_BIND(FPDFText_GetFontInfo);
            bind(FPDFText_GetFontWeight, "FPDFText_GetFontWeight", false);
            bind(FPDFLink_Enumerate, "FPDFLink_Enumerate", false);
            bind(FPDFLink_GetAction, "FPDFLink_GetAction", false);
            bind(FPDFLink_GetAnnotRect, "FPDFLink_GetAnnotRect", false);
            bind(FPDFAction_GetURIPath, "FPDFAction_GetURIPath", false);
            PDF_BIND(FPDFBitmap_CreateEx);
            PDF_BIND(FPDFBitmap_Destroy);
            PDF_BIND(FPDF_RenderPageBitmap);
#undef PDF_BIND
            FPDF_InitLibrary();
            initialized = true;
        } catch (...) {
            close_module();
            throw;
        }
    }

    ~pdf_api() {
        std::lock_guard<std::mutex> lock(pdf_mutex());
        if (initialized) FPDF_DestroyLibrary();
        close_module();
    }
};

std::shared_ptr<pdf_api> acquire_pdfium(const std::string & path) {
    if (path.empty() || path.find('\0') != std::string::npos) fail("an explicit PDFium library path is required");
    std::error_code error;
    const auto canonical = std::filesystem::canonical(std::filesystem::u8path(path), error);
    if (error || !std::filesystem::is_regular_file(canonical, error) || error)
        fail("PDFium path must identify an existing native library file");
    const std::string key = canonical.u8string();
    std::unique_lock<std::mutex> lock(pdf_mutex());
    static std::weak_ptr<pdf_api> active;
    if (auto api = active.lock()) {
        if (api->path != key) {
            // Another provider can concurrently release its last reference;
            // do not destroy this local reference while holding pdf_mutex().
            lock.unlock();
            fail("one PDFium library may be active at a time");
        }
        return api;
    }
    auto api = std::make_shared<pdf_api>(key);
    active = api;
    return api;
}

struct pdf_page_handle {
    pdf_api & api;
    void * value;
    pdf_page_handle(pdf_api & a, void * document, int index) : api(a), value(a.FPDF_LoadPage(document, index)) {
        if (!value) fail("PDFium could not load page " + std::to_string(index));
    }
    ~pdf_page_handle() { api.FPDF_ClosePage(value); }
};

struct pdf_text_handle {
    pdf_api & api;
    void * value;
    pdf_text_handle(pdf_api & a, void * page) : api(a), value(a.FPDFText_LoadPage(page)) {
        if (!value) fail("PDFium could not read the text layer");
    }
    ~pdf_text_handle() { api.FPDFText_ClosePage(value); }
};

void page_dimensions(pdf_api & api, void * page, double & width, double & height) {
    width = api.FPDF_GetPageWidth(page);
    height = api.FPDF_GetPageHeight(page);
    if (!std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0 || width > 100000 ||
        height > 100000)
        fail("invalid or excessive PDF page dimensions");
}

box page_box(pdf_api & api, void * page, double left, double bottom, double right, double top, double width,
             double height) {
    if (!std::isfinite(left) || !std::isfinite(bottom) || !std::isfinite(right) || !std::isfinite(top) ||
        std::max({ std::abs(left), std::abs(bottom), std::abs(right), std::abs(top) }) > 100000000)
        fail("invalid PDF character coordinates");
    // PDFium performs CropBox/MediaBox intersection and the page's intrinsic
    // rotation. An additional rotate=0 is required: reapplying GetRotation
    // would rotate twice. Millipoint device coordinates avoid integer-pixel
    // quantization when extracting text independently of the rendering DPI.
    const int device_w = std::max(1, int(std::llround(width * 1000)));
    const int device_h = std::max(1, int(std::llround(height * 1000)));
    box out{ std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
             -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max() };
    for (const point p :
         std::array<point, 4>{ { { left, bottom }, { left, top }, { right, bottom }, { right, top } } }) {
        int x = 0, y = 0;
        if (!api.FPDF_PageToDevice(page, 0, 0, device_w, device_h, 0, p.x, p.y, &x, &y))
            fail("PDFium could not transform page coordinates");
        const double px = double(x) * width / device_w, py = double(y) * height / device_h;
        out.x0 = std::min(out.x0, px);
        out.x1 = std::max(out.x1, px);
        out.y0 = std::min(out.y0, py);
        out.y1 = std::max(out.y1, py);
    }
    return out;
}

std::string pdf_error(unsigned long value) {
    switch (value) {
    case 2:
        return "cannot open PDF file";
    case 3:
        return "invalid or damaged PDF";
    case 4:
        return "PDF needs the correct password";
    case 5:
        return "unsupported PDF encryption/security handler";
    case 6:
        return "PDF page content error";
    default:
        return "PDFium could not open the PDF (error " + std::to_string(value) + ")";
    }
}

// PDFium's text stream collapses repeated spaces, including spaces explicitly
// present inside a PDF text object. Its glyph origins still retain the real
// advances. Reconstruct only fixed-pitch, non-generated spaces within one text
// object and require an almost integral number of space widths. This avoids
// interpreting column gaps or arbitrary prose kerning as code indentation.
size_t monospace_spaces(pdf_api & api, void * text, int index, int count) {
    if (!api.FPDFText_GetTextObject || !api.FPDFText_IsGenerated || !api.FPDFText_GetCharOrigin || index + 1 >= count ||
        api.FPDFText_IsGenerated(text, index) != 0)
        return 1;
    const uint32_t next = api.FPDFText_GetUnicode(text, index + 1);
    if (next == 0 || next == '\r' || next == '\n' || next == '\f') return 1;
    void * object = api.FPDFText_GetTextObject(text, index);
    if (!object || api.FPDFText_GetTextObject(text, index + 1) != object) return 1;
    int flags = 0;
    std::array<char, 512> font{};
    const auto required = api.FPDFText_GetFontInfo(text, index, font.data(), font.size(), &flags);
    font.back() = 0;
    const std::string name = required > 0 && required <= font.size() ? lower_ascii(font.data()) : "";
    if (!(flags & 1) && name.find("mono") == std::string::npos && name.find("courier") == std::string::npos) return 1;
    double x = 0, y = 0, nx = 0, ny = 0, left = 0, right = 0, bottom = 0, top = 0;
    if (!api.FPDFText_GetCharOrigin(text, index, &x, &y) || !api.FPDFText_GetCharOrigin(text, index + 1, &nx, &ny) ||
        !api.FPDFText_GetCharBox(text, index, &left, &right, &bottom, &top))
        return 1;
    const double advance = std::hypot(right - left, top - bottom);
    const double distance = std::hypot(nx - x, ny - y);
    if (!std::isfinite(advance) || !std::isfinite(distance) || advance < 0.01) return 1;
    const double ratio = distance / advance;
    if (!std::isfinite(ratio) || ratio < 1.8 || ratio > 256.1) return 1;
    const double rounded = std::round(ratio);
    if (std::abs(rounded - ratio) > 0.12) return 1;
    // The space's box is almost a baseline segment. Reject a displacement
    // perpendicular to it, which signals a line transition rather than spaces.
    const bool horizontal = std::abs(right - left) >= std::abs(top - bottom);
    const double cross = horizontal ? std::abs(ny - y) : std::abs(nx - x);
    if (cross > std::max(0.25, advance * 0.05)) return 1;
    return static_cast<size_t>(rounded);
}

} // namespace

struct pdf_provider::impl {
    std::shared_ptr<pdf_api> api;
    void * document = nullptr;
    int count = 0;
    ~impl() {
        if (document) {
            std::lock_guard<std::mutex> lock(pdf_mutex());
            api->FPDF_CloseDocument(document);
        }
        // Release the final module reference outside the lock: its destructor
        // locks independently before destroying PDFium.
    }
};

pdf_provider::pdf_provider(const std::string & library, const std::string & file, const std::string & password)
    : state(std::make_unique<impl>()) {
    if (file.empty() || file.find('\0') != std::string::npos || password.find('\0') != std::string::npos)
        fail("invalid PDF filename or password");
    std::ifstream input(std::filesystem::u8path(file), std::ios::binary);
    if (!input) fail("cannot open PDF file");
    state->api = acquire_pdfium(library);
    std::lock_guard<std::mutex> lock(pdf_mutex());
    state->document = state->api->FPDF_LoadDocument(file.c_str(), password.empty() ? nullptr : password.c_str());
    if (!state->document) fail(pdf_error(state->api->FPDF_GetLastError()));
    state->count = state->api->FPDF_GetPageCount(state->document);
    if (state->count < 1) fail("PDF has no readable pages");
}

pdf_provider::~pdf_provider() = default;
int pdf_provider::page_count() const {
    return state->count;
}

page pdf_provider::read_page(int index, bool extract_text) {
    if (index < 0 || index >= state->count) fail("PDF page index is out of range");
    std::lock_guard<std::mutex> lock(pdf_mutex());
    auto & api = *state->api;
    pdf_page_handle handle(api, state->document, index);
    page out;
    out.index = index;
    out.extraction_method = extract_text ? "pdfium-text" : "none";
    page_dimensions(api, handle.value, out.width, out.height);
    if (!extract_text) return out;
    pdf_rectangle visible{};
    if (!api.FPDF_GetPageBoundingBox(handle.value, &visible) || !std::isfinite(visible.left) ||
        !std::isfinite(visible.right) || !std::isfinite(visible.top) || !std::isfinite(visible.bottom) ||
        visible.right <= visible.left || visible.top <= visible.bottom)
        fail("PDFium returned an invalid visible page rectangle");
    pdf_text_handle text(api, handle.value);
    const int count = api.FPDFText_CountChars(text.value);
    if (count < 0 || count > pdf_char_limit) fail("PDF text layer exceeds the character limit or is unreadable");
    struct link_region {
        box bounds;
        std::string url;
    };
    std::vector<link_region> links;
    if (api.FPDFLink_Enumerate && api.FPDFLink_GetAction && api.FPDFLink_GetAnnotRect && api.FPDFAction_GetURIPath) {
        int position = 0;
        void * link = nullptr;
        size_t inspected = 0;
        while (api.FPDFLink_Enumerate(handle.value, &position, &link)) {
            if (++inspected > 10000) fail("PDF page exceeds the link annotation limit");
            void * action = api.FPDFLink_GetAction(link);
            if (!action) continue;
            const auto length = api.FPDFAction_GetURIPath(state->document, action, nullptr, 0);
            if (length < 2 || length > 8192) continue;
            std::string url(length, '\0');
            if (api.FPDFAction_GetURIPath(state->document, action, &url[0], length) != length) continue;
            const size_t terminator = url.find('\0');
            if (terminator == std::string::npos) continue;
            url.resize(terminator);
            pdf_rectangle rect{};
            if (!api.FPDFLink_GetAnnotRect(link, &rect) || !std::isfinite(rect.left) || !std::isfinite(rect.right) ||
                !std::isfinite(rect.top) || !std::isfinite(rect.bottom) || rect.right <= rect.left ||
                rect.top <= rect.bottom || rect.right < visible.left || rect.left > visible.right ||
                rect.top < visible.bottom || rect.bottom > visible.top)
                continue;
            rect.left = std::clamp(rect.left, visible.left, visible.right);
            rect.right = std::clamp(rect.right, visible.left, visible.right);
            rect.bottom = std::clamp(rect.bottom, visible.bottom, visible.top);
            rect.top = std::clamp(rect.top, visible.bottom, visible.top);
            links.push_back(
                { page_box(api, handle.value, rect.left, rect.bottom, rect.right, rect.top, out.width, out.height),
                  std::move(url) });
        }
    }
    line current;
    std::string pending_spaces;
    int unmapped = 0, invalid_boxes = 0, restored_spaces = 0;
    const auto flush = [&]() {
        if (!current.spans.empty()) out.text_lines.push_back(std::move(current));
        current = line{};
        pending_spaces.clear();
    };
    for (int i = 0; i < count; ++i) {
        const int character_index = i;
        uint32_t cp = api.FPDFText_GetUnicode(text.value, i);
        if (cp == '\r' || cp == '\n' || cp == '\f') {
            flush();
            continue;
        }
        if (cp == '\t') cp = ' ';
        if (cp == ' ') {
            // Keep code indentation and separators without enlarging an ink
            // box to include the gap between columns or table cells.
            const size_t spaces = monospace_spaces(api, text.value, i, count);
            restored_spaces += int(spaces - 1);
            if (current.spans.empty())
                pending_spaces.append(spaces, ' ');
            else
                current.spans.back().text.append(spaces, ' ');
            continue;
        }
        if (cp == 0 || cp > 0x10ffff) {
            cp = 0xfffd;
            ++unmapped;
        }
        if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < count) {
            const uint32_t next = api.FPDFText_GetUnicode(text.value, i + 1);
            if (next >= 0xdc00 && next <= 0xdfff) {
                cp = 0x10000 + ((cp - 0xd800) << 10) + next - 0xdc00;
                ++i;
            }
        }
        if (cp >= 0xd800 && cp <= 0xdfff) {
            cp = 0xfffd;
            ++unmapped;
        }
        if (cp < 32 || (cp >= 0x7f && cp <= 0x9f)) continue;
        double left = 0, right = 0, bottom = 0, top = 0;
        if (!api.FPDFText_GetCharBox(text.value, character_index, &left, &right, &bottom, &top)) {
            ++invalid_boxes;
            continue;
        }
        if (!std::isfinite(left) || !std::isfinite(right) || !std::isfinite(bottom) || !std::isfinite(top)) {
            ++invalid_boxes;
            continue;
        }
        if (right < visible.left || left > visible.right || top < visible.bottom || bottom > visible.top) continue;
        left = std::clamp(left, double(visible.left), double(visible.right));
        right = std::clamp(right, double(visible.left), double(visible.right));
        bottom = std::clamp(bottom, double(visible.bottom), double(visible.top));
        top = std::clamp(top, double(visible.bottom), double(visible.top));
        box bounds = page_box(api, handle.value, left, bottom, right, top, out.width, out.height);
        // Text outside the visible CropBox must not leak into the extracted
        // document. Partially visible glyph boxes are clipped to the page.
        if (bounds.x1 < 0 || bounds.y1 < 0 || bounds.x0 > out.width || bounds.y0 > out.height) continue;
        bounds.x0 = std::clamp(bounds.x0, 0.0, out.width);
        bounds.x1 = std::clamp(bounds.x1, 0.0, out.width);
        bounds.y0 = std::clamp(bounds.y0, 0.0, out.height);
        bounds.y1 = std::clamp(bounds.y1, 0.0, out.height);
        double size = api.FPDFText_GetFontSize(text.value, character_index);
        if (!std::isfinite(size) || size < 0 || size > 100000) size = 0;
        int flags = 0;
        std::array<char, 512> font{};
        const unsigned long required = api.FPDFText_GetFontInfo(text.value, character_index, font.data(),
                                                                static_cast<unsigned long>(font.size()), &flags);
        font.back() = 0;
        const std::string font_name = required > 0 && required <= font.size() ? lower_ascii(font.data()) : "";
        const int weight = api.FPDFText_GetFontWeight ? api.FPDFText_GetFontWeight(text.value, character_index) : 0;
        bool bold = weight >= 700 || font_name.find("bold") != std::string::npos || (flags & (1 << 18));
        bool italic = (flags & 64) || font_name.find("italic") != std::string::npos ||
                      font_name.find("oblique") != std::string::npos;
        bool mono = (flags & 1) || font_name.find("mono") != std::string::npos ||
                    font_name.find("courier") != std::string::npos;
        std::string url;
        const double center_x = (bounds.x0 + bounds.x1) * 0.5, center_y = (bounds.y0 + bounds.y1) * 0.5;
        for (const auto & link : links)
            if (center_x >= link.bounds.x0 && center_x <= link.bounds.x1 && center_y >= link.bounds.y0 &&
                center_y <= link.bounds.y1) {
                url = link.url;
                break;
            }
        if (current.spans.empty())
            current.bbox = bounds;
        else
            current.bbox = joined(current.bbox, bounds);
        const bool same_word = !current.spans.empty() && !current.spans.back().text.empty() &&
                               current.spans.back().text.back() != ' ' &&
                               bounds.x0 - current.spans.back().bbox.x1 <= std::max(1.0, size * 0.5);
        const bool same = same_word && std::abs(current.spans.back().font_size - size) < 0.05 &&
                          current.spans.back().bold == bold && current.spans.back().italic == italic &&
                          current.spans.back().monospace == mono && current.spans.back().url == url;
        if (!same) {
            span part;
            part.bbox = bounds;
            part.font_size = size;
            part.bold = bold;
            part.italic = italic;
            part.monospace = mono;
            part.url = std::move(url);
            if (current.spans.empty()) {
                part.text = std::move(pending_spaces);
                pending_spaces.clear();
            }
            current.spans.push_back(std::move(part));
        } else
            current.spans.back().bbox = joined(current.spans.back().bbox, bounds);
        append_utf8(current.spans.back().text, cp);
    }
    flush();
    if (unmapped) out.warnings.push_back(std::to_string(unmapped) + " PDF characters had invalid Unicode mappings");
    if (invalid_boxes) out.warnings.push_back(std::to_string(invalid_boxes) + " PDF characters had no usable boxes");
    if (restored_spaces)
        out.warnings.push_back(std::to_string(restored_spaces) +
                               " collapsed monospace spaces restored from PDF glyph advances");
    return out;
}

image pdf_provider::render_page(int index, int dpi, size_t max_pixels) {
    if (index < 0 || index >= state->count) fail("PDF page index is out of range");
    if (dpi < 36 || dpi > 1200) fail("PDF rendering DPI must be between 36 and 1200");
    std::lock_guard<std::mutex> lock(pdf_mutex());
    auto & api = *state->api;
    pdf_page_handle page_handle(api, state->document, index);
    double width = 0, height = 0;
    page_dimensions(api, page_handle.value, width, height);
    const double pixel_w = std::ceil(width * dpi / 72.0), pixel_h = std::ceil(height * dpi / 72.0);
    if (pixel_w > dimension_limit || pixel_h > dimension_limit) fail("PDF rendering exceeds the dimension limit");
    image out;
    out.width = int(pixel_w);
    out.height = int(pixel_h);
    const size_t count = pixels_checked(out.width, out.height, max_pixels);
    std::vector<uint8_t> bgra(count * 4, 255);
    void * bitmap = api.FPDFBitmap_CreateEx(out.width, out.height, 4 /* FPDFBitmap_BGRA */, bgra.data(), out.width * 4);
    if (!bitmap) fail("PDFium bitmap allocation failed");
    // No JavaScript, forms environment or document action is initialized.
    // FPDF_ANNOT includes visible annotations in the page image.
    api.FPDF_RenderPageBitmap(bitmap, page_handle.value, 0, 0, out.width, out.height, 0, 1 /* FPDF_ANNOT */);
    api.FPDFBitmap_Destroy(bitmap);
    out.rgb.resize(count * 3);
    for (size_t i = 0; i < count; ++i) {
        const unsigned alpha = bgra[i * 4 + 3];
        for (size_t c = 0; c < 3; ++c)
            out.rgb[i * 3 + c] = uint8_t((unsigned(bgra[i * 4 + 2 - c]) * alpha + 255 * (255 - alpha) + 127) / 255);
    }
    return out;
}

image load_image(const std::string & file, size_t max_pixels) {
    if (file.empty() || file.find('\0') != std::string::npos) fail("invalid image filename");
    std::ifstream stream(std::filesystem::u8path(file), std::ios::binary | std::ios::ate);
    if (!stream) fail("could not open input image");
    const auto length = stream.tellg();
    if (length <= 0 || uint64_t(length) > image_file_limit)
        fail("input image exceeds the encoded file-size limit or is empty");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), std::streamsize(bytes.size())))
        fail("could not read input image");
    int width = 0, height = 0, channels = 0;
    if (!stbi_info_from_memory(bytes.data(), int(bytes.size()), &width, &height, &channels))
        fail("input is not a supported raster image");
    pixels_checked(width, height, max_pixels);
    using owned_pixels = std::unique_ptr<unsigned char, decltype(&stbi_image_free)>;
    owned_pixels pixels(stbi_load_from_memory(bytes.data(), int(bytes.size()), &width, &height, &channels, 4),
                        stbi_image_free);
    if (!pixels) fail("raster image decoding failed");
    const size_t count = pixels_checked(width, height, max_pixels);
    image out;
    out.width = width;
    out.height = height;
    out.rgb.resize(count * 3);
    for (size_t i = 0; i < count; ++i) {
        const unsigned alpha = pixels.get()[i * 4 + 3];
        for (size_t c = 0; c < 3; ++c)
            out.rgb[i * 3 + c] = uint8_t((unsigned(pixels.get()[i * 4 + c]) * alpha + 255 * (255 - alpha) + 127) / 255);
    }
    return orient_image(out, encoded_orientation(bytes));
}

image crop_image(const image & input, const block & region, double page_width, double page_height) {
    check_image(input);
    if (!std::isfinite(page_width) || !std::isfinite(page_height) || page_width <= 0 || page_height <= 0)
        fail("invalid page dimensions for crop");
    const double sx = input.width / page_width, sy = input.height / page_height;
    if (!std::isfinite(sx) || !std::isfinite(sy) || sx <= 0 || sy <= 0) fail("invalid crop coordinate scale");
    const auto coordinate = [](double value, double scale, int limit) {
        if (!std::isfinite(value)) fail("nonfinite crop coordinates");
        return int(std::floor(std::clamp(value * scale, 0.0, double(limit))));
    };
    const int x0 = coordinate(region.bbox.x0, sx, input.width), y0 = coordinate(region.bbox.y0, sy, input.height);
    const int x1 = coordinate(region.bbox.x1, sx, input.width), y1 = coordinate(region.bbox.y1, sy, input.height);
    if (x1 <= x0 || y1 <= y0) fail("layout crop is empty at the rendered pixel resolution");
    image out;
    out.width = x1 - x0;
    out.height = y1 - y0;
    out.rgb.resize(size_t(out.width) * out.height * 3);
    for (int y = 0; y < out.height; ++y)
        std::copy_n(input.rgb.data() + (size_t(y + y0) * input.width + x0) * 3, size_t(out.width) * 3,
                    out.rgb.data() + size_t(y) * out.width * 3);
    if (!region.polygon.empty()) {
        if (region.polygon.size() < 3 || region.polygon.size() > 128) fail("invalid crop polygon");
        std::vector<point> polygon;
        for (const auto & p : region.polygon) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x > page_width ||
                p.y > page_height)
                fail("crop polygon lies outside the page");
            polygon.push_back({ std::floor(p.x * sx) - x0, std::floor(p.y * sy) - y0 });
        }
        for (int y = 0; y < out.height; ++y) {
            for (int x = 0; x < out.width; ++x) {
                const double px = x + 0.5, py = y + 0.5;
                bool inside = false;
                point previous = polygon.back();
                for (const point p : polygon) {
                    if ((p.y > py) != (previous.y > py)) {
                        const double edge = (previous.x - p.x) * (py - p.y) / (previous.y - p.y) + p.x;
                        if (px < edge) inside = !inside;
                    }
                    previous = p;
                }
                if (!inside) std::fill_n(out.rgb.data() + (size_t(y) * out.width + x) * 3, 3, uint8_t(0));
            }
        }
    }
    switch (region.angle) {
    case 0:
        break;
    case 90:
        out = orient_image(out, 8);
        break;
    case 180:
        out = orient_image(out, 3);
        break;
    case 270:
        out = orient_image(out, 6);
        break;
    default:
        fail("TeleOCR crop orientation must be a multiple of 90 degrees");
    }
    if (double(std::max(out.width, out.height)) / std::min(out.width, out.height) > 50) {
        const int width = out.width > out.height ? out.width : int(std::ceil(out.height / 50.0));
        const int height = out.width > out.height ? int(std::ceil(out.width / 50.0)) : out.height;
        image padded;
        padded.width = width;
        padded.height = height;
        padded.rgb.assign(pixels_checked(width, height, hard_pixel_limit) * 3, 255);
        const int dx = (width - out.width) / 2, dy = (height - out.height) / 2;
        for (int y = 0; y < out.height; ++y)
            std::copy_n(out.rgb.data() + size_t(y) * out.width * 3, size_t(out.width) * 3,
                        padded.rgb.data() + (size_t(y + dy) * width + dx) * 3);
        out = std::move(padded);
    }
    if (std::min(out.width, out.height) < 28) {
        const double scale = 28.0 / std::min(out.width, out.height);
        out = resize_image(out, int(std::ceil(out.width * scale)), int(std::ceil(out.height * scale)));
    }
    return out;
}

asset encode_image_asset(const image & input, const std::string & name) {
    check_image(input);
    if (name.empty() || name.size() > 255 || name == "." || name == ".." ||
        name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") !=
            std::string::npos)
        fail("unsafe image asset filename");
    std::string png;
    const auto collect = [](void * target, void * data, int count) {
        if (count > 0) static_cast<std::string *>(target)->append(static_cast<const char *>(data), size_t(count));
    };
    if (!stbi_write_png_to_func(collect, &png, input.width, input.height, 3, input.rgb.data(), input.width * 3))
        fail("could not encode extracted PNG asset");
    // Extracted/cropped pixels are not generated imagery. Record the actual
    // operation rather than attaching the enhancement helper's synthetic claim.
    static std::mutex png_metadata_mutex;
    {
        std::lock_guard<std::mutex> lock(png_metadata_mutex);
        core_imgout::png_insert_itxt(
            png, core_imgout::png_itxt(
                     "CrispEmbed", "software=CrispEmbed\nengine=marker-native\noperation=document-image-extraction\n"));
    }
    asset out;
    out.name = name;
    out.mime_type = "image/png";
    out.bytes.assign(png.begin(), png.end());
    return out;
}

struct teleocr_backend::impl {
    options opts;
    std::unique_ptr<qwen2vl_ocr_context, decltype(&qwen2vl_ocr_free)> context{ nullptr, qwen2vl_ocr_free };
    std::mutex mutex;
    struct generation {
        std::string text;
        int tokens = 0;
        double confidence = 0;
        bool cap = false;
    };

    generation generate(const image & input, const char * prompt, const std::string & kind, int cap) {
        check_image(input, opts.max_pixels);
        std::lock_guard<std::mutex> lock(mutex);
        if (!core_env::on("CRISPEMBED_TELEOCR")) fail("native TeleOCR requires CRISPEMBED_TELEOCR=1");
        const float frequency = kind == "layout"                                                            ? 0.0f
                                : (kind == "table" || kind == "char" || kind == "seal" || kind == "figure") ? 0.005f
                                                                                                            : 0.05f;
        if (!qwen2vl_ocr_set_generation_params(context.get(), 1.0f, frequency ? 1.0f : 0.0f, frequency, 100) ||
            !qwen2vl_ocr_set_prompt(context.get(), prompt))
            fail("TeleOCR rejected the task settings");
        qwen2vl_ocr_set_max_tokens(context.get(), cap);
        if (qwen2vl_ocr_get_max_tokens(context.get()) != cap) fail("TeleOCR rejected the token limit");
        int length = 0;
        const char * value =
            qwen2vl_ocr_recognize_raw(context.get(), input.rgb.data(), input.width, input.height, 3, &length);
        if (!value || length < 0) fail("native TeleOCR inference failed");
        generation out;
        out.text.assign(value, size_t(length));
        const float * scores = qwen2vl_ocr_confidences(context.get(), &out.tokens);
        if (!scores || out.tokens < 1 || out.tokens > cap) fail("native TeleOCR returned invalid token scores");
        for (int i = 0; i < out.tokens; ++i) {
            if (!std::isfinite(scores[i]) || scores[i] < 0 || scores[i] > 1)
                fail("native TeleOCR returned invalid confidence");
            out.confidence += scores[i];
        }
        out.confidence /= out.tokens;
        // Native public scores include EOS but do not expose the EOS ID. A
        // sequence ending exactly at cap is conservatively marked incomplete.
        out.cap = out.tokens >= cap;
        return out;
    }
};

teleocr_backend::teleocr_backend(const options & opts) : state(std::make_unique<impl>()) {
    if (!core_env::on("CRISPEMBED_TELEOCR")) fail("native TeleOCR requires CRISPEMBED_TELEOCR=1");
    if (!core_env::on("CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE"))
        fail("TeleOCR's vision tower is under the Qwen RESEARCH LICENSE (non-commercial); set "
             "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 to acknowledge");
    if (opts.model_path.empty() || opts.model_path.find('\0') != std::string::npos)
        fail("an explicit TeleOCR GGUF path is required");
    if (opts.n_threads < 1 || opts.n_threads > 256 || opts.max_tokens < 1 || opts.max_layout_tokens < 1 ||
        opts.max_tokens > 32768 || opts.max_layout_tokens > 32768 || opts.max_blocks < 1 || opts.max_blocks > 4096 ||
        opts.max_pixels < 1036 * 1036 || opts.max_pixels > hard_pixel_limit)
        fail("invalid TeleOCR backend limits");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> metadata(
        core_gguf::open_metadata(opts.model_path.c_str()), core_gguf::free_metadata);
    if (!metadata || core_gguf::kv_str(metadata.get(), "general.architecture", "") != "teleocr")
        fail("document OCR requires a strict TeleOCR-profile GGUF");
    metadata.reset();
    state->opts = opts;
    state->context.reset(qwen2vl_ocr_init(opts.model_path.c_str(), opts.n_threads));
    if (!state->context || !qwen2vl_ocr_get_generation_params(state->context.get(), nullptr, nullptr, nullptr, nullptr))
        fail("could not load the strict native TeleOCR model profile");
}

teleocr_backend::~teleocr_backend() = default;

std::vector<block> teleocr_backend::layout(const image & input, double page_width, double page_height,
                                           std::vector<std::string> & warnings, std::string * raw,
                                           int * generated_tokens, bool * token_cap_reached) {
    if (raw) raw->clear();
    if (generated_tokens) *generated_tokens = 0;
    if (token_cap_reached) *token_cap_reached = false;
    check_image(input, state->opts.max_pixels);
    const auto output = state->generate(resize_image(input, 1036, 1036), "\nAnalyze the image layout.", "layout",
                                        state->opts.max_layout_tokens);
    if (raw) *raw = output.text;
    if (generated_tokens) *generated_tokens = output.tokens;
    if (token_cap_reached) *token_cap_reached = output.cap;
    if (output.cap) warnings.push_back("TeleOCR layout reached the token cap; page structure may be incomplete");
    auto blocks = parse_teleocr_layout(output.text, page_width, page_height, state->opts.max_blocks, warnings);
    if (blocks.empty())
        warnings.push_back("TeleOCR returned no valid layout blocks; explicit document fallback required");
    for (auto & item : blocks) {
        item.source = "teleocr-native";
        item.confidence = output.confidence;
        if (output.cap) item.warnings.push_back("layout prediction reached the token cap");
    }
    return blocks;
}

void teleocr_backend::recognize(const image & input, block & region, double page_width, double page_height) {
    check_image(input, state->opts.max_pixels);
    const image crop = crop_image(input, region, page_width, page_height);
    std::string kind = region.raw_label.empty() ? lower_ascii(region.type) : region.raw_label;
    const char * prompt = "\nPlease output the text content from the image.";
    if (kind == "char") {
        prompt = "\nThis is a scientific figure. Please extract the table implied by this figure.";
    } else if (region.type == "Table" || kind == "table") {
        kind = "table";
        prompt = "\nThis is the image of a table. Please output the table in OTSL format.";
    } else if (region.type == "Equation" || kind == "equation" || kind == "equation_block") {
        kind = "equation";
        prompt = "\nPlease write out the expression of the formula in the image using LaTeX format.";
    } else if (region.type == "Code" || kind == "code" || kind == "algorithm") {
        kind = "code";
        prompt = "\nThe image contains a code snippet, please output the parsing result.";
    } else if (kind == "seal")
        prompt = "\nSeal Recognition:";
    const auto output = state->generate(crop, prompt, kind, state->opts.max_tokens);
    // A successful OCR replacement owns this block's content, including a
    // legitimate empty/EOS-only result. Do not let later text processors
    // resurrect a rejected PDF text layer from its old spans.
    region.lines.clear();
    region.raw = output.text;
    region.text = trim_ascii(output.text);
    region.html.clear();
    region.source = "teleocr-native";
    region.generated_tokens = output.tokens;
    region.token_cap_reached = output.cap;
    region.confidence = output.confidence;
    if (output.cap) region.warnings.push_back("TeleOCR recognition reached the token cap; content may be incomplete");
    if (kind == "table" || kind == "char") {
        region.html = otsl_to_html(output.text, region.warnings);
        if (kind == "char" && !region.html.empty()) {
            region.type = "Table";
            region.warnings.push_back("Scientific figure recognized as the implied table using TeleOCR's char task");
        }
        if (region.html.empty()) region.html = "<pre>" + escape_html(output.text) + "</pre>";
    } else if (kind == "equation") {
        // Strip only a balanced outer wrapper. Preserve literal dollar signs
        // and the full raw string for review instead of deleting all '$'.
        auto & value = region.text;
        if (value.size() >= 4 && ((value.compare(0, 2, "$$") == 0 && value.compare(value.size() - 2, 2, "$$") == 0) ||
                                  (value.compare(0, 2, "\\[") == 0 && value.compare(value.size() - 2, 2, "\\]") == 0) ||
                                  (value.compare(0, 2, "\\(") == 0 && value.compare(value.size() - 2, 2, "\\)") == 0)))
            value = trim_ascii(value.substr(2, value.size() - 4));
        else if (value.size() >= 2 && value.front() == '$' && value.back() == '$')
            value = trim_ascii(value.substr(1, value.size() - 2));
        region.html = "<math display=\"block\">" + escape_html(value) + "</math>";
    } else if (kind == "code") {
        auto & value = region.text;
        value = output.text;
        // Remove framing newlines only; indentation inside the code is data.
        const size_t first = value.find_first_not_of("\r\n");
        value = first == std::string::npos ? "" : value.substr(first, value.find_last_not_of("\r\n") - first + 1);
        if (value.compare(0, 3, "```") == 0) {
            const size_t newline = value.find('\n');
            if (newline != std::string::npos) {
                value.erase(0, newline + 1);
                if (value.size() >= 3 && value.compare(value.size() - 3, 3, "```") == 0) value.erase(value.size() - 3);
            }
        }
        if (value.compare(0, 2, "<_") == 0) {
            const size_t end = value.find("_>", 2);
            if (end != std::string::npos && end <= 66 && value.find('\n', 2) > end) value.erase(0, end + 2);
        }
        region.html = "<pre><code>" + escape_html(value) + "</code></pre>";
    } else {
        // Ordinary text, including lists, is extracted rather than silently
        // skipped. The document processors decide list/heading structure.
        region.html = "<p>" + escape_html(region.text) + "</p>";
    }
}

} // namespace crisp_marker
