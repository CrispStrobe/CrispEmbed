#include "core/clean_exit.h"
#include "ocr_crop.h"
#include "ocr_detect.h"
#include "ppocrv6_ocr.h"
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct line_box {
    float x0, y0, x1, y1;
};

static std::vector<line_box> group_lines(std::vector<ocr_detect::text_box> boxes) {
    std::sort(boxes.begin(), boxes.end(),
              [](const auto & a, const auto & b) { return a.y + a.h * 0.5f < b.y + b.h * 0.5f; });
    std::vector<line_box> lines;
    for (const auto & b : boxes) {
        const float cy = b.y + b.h * 0.5f;
        int match = -1;
        for (int i = 0; i < (int)lines.size(); ++i) {
            const float ly = (lines[i].y0 + lines[i].y1) * 0.5f;
            if (std::abs(cy - ly) <= 0.5f * std::max(lines[i].y1 - lines[i].y0, b.h)) {
                match = i;
                break;
            }
        }
        if (match < 0)
            lines.push_back({ b.x, b.y, b.x + b.w, b.y + b.h });
        else {
            auto & l = lines[match];
            l.x0 = std::min(l.x0, b.x);
            l.y0 = std::min(l.y0, b.y);
            l.x1 = std::max(l.x1, b.x + b.w);
            l.y1 = std::max(l.y1, b.y + b.h);
        }
    }
    std::sort(lines.begin(), lines.end(),
              [](const auto & a, const auto & b) { return a.y0 == b.y0 ? a.x0 < b.x0 : a.y0 < b.y0; });
    return lines;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <ppocrv6-rec.gguf> <image>\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, channels = 0;
    unsigned char * pixels = stbi_load(argv[3], &w, &h, &channels, 3);
    if (!pixels) return 3;
    ocr_detect::context * det = nullptr;
    ppocrv6_ocr_context * rec = nullptr;
    int rc = 0;
    if (!ocr_detect::load(&det, argv[1], 1) || !(rec = ppocrv6_ocr_init(argv[2], 1)))
        rc = 4;
    else {
        auto boxes = ocr_detect::detect_rgb_ex(det, pixels, w, h, 3, ocr_detect::rapid_defaults());
        const bool word_mode =
            std::getenv("PPOCRV6_DBNET_MODE") && std::string(std::getenv("PPOCRV6_DBNET_MODE")) == "words";
        std::vector<line_box> lines = word_mode ? std::vector<line_box>() : group_lines(boxes);
        if (word_mode)
            for (const auto & b : boxes) lines.push_back({ b.x, b.y, b.x + b.w, b.y + b.h });
        if (const char * limit = std::getenv("PPOCRV6_DBNET_MAX_UNITS")) {
            const size_t n = std::strtoul(limit, nullptr, 10);
            if (n < lines.size()) lines.resize(n);
        }
        std::printf("dbnet-ppocrv6 mode=%s detector_boxes=%zu units=%zu\n", word_mode ? "words" : "lines", boxes.size(),
                    lines.size());
        for (size_t i = 0; i < lines.size(); ++i) {
            const auto & l = lines[i];
            const int x = std::max(0, (int)l.x0 - 2), y = std::max(0, (int)l.y0 - 2);
            const int cw = std::min(w - x, (int)(l.x1 - l.x0) + 4);
            const int ch = std::min(h - y, (int)(l.y1 - l.y0) + 4);
            int ow = 0, oh = 0;
            auto crop = ocr_crop::extract(pixels, w, h, 3, x, y, cw, ch, 0, &ow, &oh);
            int out_len = 0;
            const char * text = ppocrv6_ocr_recognize_raw(rec, crop.data(), ow, oh, 3, &out_len);
            std::printf("dbnet-ppocrv6 unit=%zu crop=%dx%d text=%.*s\n", i, ow, oh, out_len, text ? text : "");
        }
        if (lines.empty()) rc = 5;
    }
    ppocrv6_ocr_free(rec);
    ocr_detect::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
