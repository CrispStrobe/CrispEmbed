// SPDX-License-Identifier: MIT
// Owned PDFium ABI fixture for lifecycle tests, not a PDF parser or renderer.
// It accepts only inputs bearing MARKER_LIFECYCLE_FIXTURE and never uses models.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define FIXTURE_EXPORT extern "C" __declspec(dllexport)
#define FIXTURE_CALL __stdcall
#else
#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))
#define FIXTURE_CALL
#endif

namespace {
struct fixture_document {
    bool fail_second = false;
    std::string font_name = "FixtureSans";
    int font_flags = 0;
};
struct fixture_page {
    int index;
    std::string font_name;
    int font_flags;
};
struct fixture_text {
    int index;
    std::string font_name;
    int font_flags;
};
struct fixture_rectangle {
    float left, top, right, bottom;
};
thread_local unsigned long last_error = 0;
const std::array<std::string, 3> texts{ { "First owned page is durable.", "Second owned page is durable.",
                                          "Third owned page is complete." } };
const std::array<std::vector<uint16_t>, 3> labels{
    { { 'i', 'v', 0 }, { 0x7ae0, '-', '2', 0 }, { 0xd83d, 0xdcd6, '-', '3', 0 } }
};
constexpr double page_width = 600, page_height = 800;

bool valid_character(void * handle, int index) {
    if (!handle || index < 0) return false;
    const auto page = static_cast<fixture_text *>(handle)->index;
    return page >= 0 && page < 3 && static_cast<size_t>(index) < texts[page].size();
}
} // namespace

FIXTURE_EXPORT void FIXTURE_CALL FPDF_InitLibrary() {}
FIXTURE_EXPORT void FIXTURE_CALL FPDF_DestroyLibrary() {}
FIXTURE_EXPORT unsigned long FIXTURE_CALL FPDF_GetLastError() {
    return last_error;
}

FIXTURE_EXPORT void * FIXTURE_CALL FPDF_LoadDocument(const char * path, const char *) {
    last_error = 0;
    if (!path) {
        last_error = 2;
        return nullptr;
    }
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if (!file) {
        last_error = 2;
        return nullptr;
    }
    std::array<char, 4096> data{};
    file.read(data.data(), static_cast<std::streamsize>(data.size()));
    const std::string contents(data.data(), static_cast<size_t>(file.gcount()));
    if (contents.find("%PDF-") == std::string::npos || contents.find("MARKER_LIFECYCLE_FIXTURE") == std::string::npos) {
        last_error = 3;
        return nullptr;
    }
    fixture_document document;
    document.fail_second = contents.find("FAIL_SECOND_PAGE") != std::string::npos;
    const auto font = contents.find("\nFONT_NAME=");
    if (font != std::string::npos) {
        const auto end = contents.find('\n', font + 11);
        document.font_name = contents.substr(font + 11, end - (font + 11));
        if (document.font_name.empty() || document.font_name.size() > 256) {
            last_error = 3;
            return nullptr;
        }
        document.font_flags = contents.find("\nFONT_FIXED=1\n") != std::string::npos ? 1 : 0;
    }
    return new fixture_document(std::move(document));
}

FIXTURE_EXPORT void FIXTURE_CALL FPDF_CloseDocument(void * document) {
    delete static_cast<fixture_document *>(document);
}
FIXTURE_EXPORT int FIXTURE_CALL FPDF_GetPageCount(void * document) {
    return document ? 3 : 0;
}
FIXTURE_EXPORT void * FIXTURE_CALL FPDF_LoadPage(void * document, int index) {
    if (!document || index < 0 || index >= 3 ||
        (index == 1 && static_cast<fixture_document *>(document)->fail_second)) {
        last_error = 6;
        return nullptr;
    }
    const auto & source = *static_cast<fixture_document *>(document);
    return new fixture_page{ index, source.font_name, source.font_flags };
}
FIXTURE_EXPORT void FIXTURE_CALL FPDF_ClosePage(void * page) {
    delete static_cast<fixture_page *>(page);
}
FIXTURE_EXPORT double FIXTURE_CALL FPDF_GetPageWidth(void *) {
    return page_width;
}
FIXTURE_EXPORT double FIXTURE_CALL FPDF_GetPageHeight(void *) {
    return page_height;
}

FIXTURE_EXPORT unsigned long FIXTURE_CALL FPDF_GetPageLabel(void * document, int index, void * buffer,
                                                            unsigned long capacity) {
    if (!document || index < 0 || index >= 3) return 0;
    const auto & label = labels[static_cast<size_t>(index)];
    const auto bytes = static_cast<unsigned long>(label.size() * 2);
    if (buffer && capacity >= bytes) {
        auto * out = static_cast<uint8_t *>(buffer);
        for (size_t i = 0; i < label.size(); ++i) {
            out[i * 2] = static_cast<uint8_t>(label[i]);
            out[i * 2 + 1] = static_cast<uint8_t>(label[i] >> 8);
        }
    }
    return bytes; // PDFium reports bytes including the UTF-16LE NUL terminator.
}

FIXTURE_EXPORT int FIXTURE_CALL FPDF_GetPageBoundingBox(void * page, fixture_rectangle * out) {
    if (!page || !out) return 0;
    *out = { 0, static_cast<float>(page_height), static_cast<float>(page_width), 0 };
    return 1;
}
FIXTURE_EXPORT int FIXTURE_CALL FPDF_PageToDevice(void * page, int x, int y, int width, int height, int rotation,
                                                  double px, double py, int * dx, int * dy) {
    if (!page || !dx || !dy || width <= 0 || height <= 0 || rotation != 0) return 0;
    *dx = x + static_cast<int>(std::llround(px * width / page_width));
    *dy = y + static_cast<int>(std::llround((page_height - py) * height / page_height));
    return 1;
}

FIXTURE_EXPORT void * FIXTURE_CALL FPDFText_LoadPage(void * page) {
    if (!page) return nullptr;
    const auto & source = *static_cast<fixture_page *>(page);
    return new fixture_text{ source.index, source.font_name, source.font_flags };
}
FIXTURE_EXPORT void FIXTURE_CALL FPDFText_ClosePage(void * text) {
    delete static_cast<fixture_text *>(text);
}
FIXTURE_EXPORT int FIXTURE_CALL FPDFText_CountChars(void * text) {
    return text ? static_cast<int>(texts[static_cast<fixture_text *>(text)->index].size()) : -1;
}
FIXTURE_EXPORT unsigned int FIXTURE_CALL FPDFText_GetUnicode(void * text, int index) {
    return valid_character(text, index)
               ? static_cast<unsigned char>(texts[static_cast<fixture_text *>(text)->index][index])
               : 0;
}
FIXTURE_EXPORT int FIXTURE_CALL FPDFText_GetCharBox(void * text, int index, double * left, double * right,
                                                    double * bottom, double * top) {
    if (!valid_character(text, index) || !left || !right || !bottom || !top) return 0;
    *left = 40 + index * 6;
    *right = *left + 6;
    *bottom = 500;
    *top = 512;
    return 1;
}
FIXTURE_EXPORT double FIXTURE_CALL FPDFText_GetFontSize(void *, int) {
    return 12;
}
FIXTURE_EXPORT unsigned long FIXTURE_CALL FPDFText_GetFontInfo(void * text, int, void * buffer, unsigned long capacity,
                                                               int * flags) {
    if (!text) return 0;
    const auto & source = *static_cast<fixture_text *>(text);
    if (flags) *flags = source.font_flags;
    const auto size = static_cast<unsigned long>(source.font_name.size() + 1);
    if (buffer && capacity >= size) std::memcpy(buffer, source.font_name.c_str(), size);
    return size;
}
FIXTURE_EXPORT int FIXTURE_CALL FPDFText_GetFontWeight(void *, int) {
    return 400;
}

// Required loader symbols only. Text-mode tests must never render a bitmap;
// returning null makes an accidental raster path fail immediately.
FIXTURE_EXPORT void * FIXTURE_CALL FPDFBitmap_CreateEx(int, int, int, void *, int) {
    return nullptr;
}
FIXTURE_EXPORT void FIXTURE_CALL FPDFBitmap_Destroy(void *) {}
FIXTURE_EXPORT void FIXTURE_CALL FPDF_RenderPageBitmap(void *, void *, int, int, int, int, int, int) {}
