#include "easyocr_ocr.h"
#include "ocr_crop.h"
#include "ocr_detect.h"
#include "core/clean_exit.h"
#include "stb_image.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
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
            const float line_cy = (lines[i].y0 + lines[i].y1) * 0.5f;
            if (std::abs(cy - line_cy) <= 0.5f * std::max(lines[i].y1 - lines[i].y0, b.h)) {
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
    std::sort(lines.begin(), lines.end(), [](const auto & a, const auto & b) { return a.y0 < b.y0; });
    return lines;
}

static std::vector<line_box> order_words(const std::vector<ocr_detect::text_box> & boxes) {
    std::vector<std::vector<line_box>> groups;
    for (const auto & b : boxes) {
        const float cy = b.y + b.h * 0.5f;
        int match = -1;
        for (int i = 0; i < (int)groups.size(); ++i) {
            float y0 = groups[i].front().y0, y1 = groups[i].front().y1;
            for (const auto & word : groups[i]) {
                y0 = std::min(y0, word.y0);
                y1 = std::max(y1, word.y1);
            }
            if (std::abs(cy - (y0 + y1) * 0.5f) <= 0.5f * std::max(y1 - y0, b.h)) {
                match = i;
                break;
            }
        }
        line_box word{ b.x, b.y, b.x + b.w, b.y + b.h };
        if (match < 0)
            groups.push_back({ word });
        else
            groups[match].push_back(word);
    }
    std::sort(groups.begin(), groups.end(), [](const auto & a, const auto & b) { return a.front().y0 < b.front().y0; });
    std::vector<line_box> ordered;
    for (auto & group : groups) {
        std::sort(group.begin(), group.end(), [](const auto & a, const auto & b) { return a.x0 < b.x0; });
        ordered.insert(ordered.end(), group.begin(), group.end());
    }
    return ordered;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dbnet.gguf> <easyocr.gguf> <image>\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, channels = 0;
    unsigned char * pixels = stbi_load(argv[3], &w, &h, &channels, 3);
    if (!pixels) return 3;
    ocr_detect::context * det = nullptr;
    easyocr_ocr_context * rec = nullptr;
    int rc = 0;
    if (!ocr_detect::load(&det, argv[1], 1) || !(rec = easyocr_ocr_init(argv[2], 1))) {
        rc = 4;
    } else {
        auto boxes = ocr_detect::detect_rgb_ex(det, pixels, w, h, 3, ocr_detect::rapid_defaults());
        const bool word_mode =
            std::getenv("EASYOCR_DBNET_MODE") && std::string(std::getenv("EASYOCR_DBNET_MODE")) == "words";
        std::vector<line_box> lines = word_mode ? std::vector<line_box>() : group_lines(std::move(boxes));
        if (word_mode) {
            lines = order_words(boxes);
        }
        std::printf("dbnet-easyocr mode=%s units=%zu\n", word_mode ? "words" : "lines", lines.size());
        for (size_t i = 0; i < lines.size(); ++i) {
            const auto & line = lines[i];
            const int x = std::max(0, (int)line.x0 - 2), y = std::max(0, (int)line.y0 - 2);
            const int cw = std::min(w - x, (int)(line.x1 - line.x0) + 4);
            const int ch = std::min(h - y, (int)(line.y1 - line.y0) + 4);
            int ow = 0, oh = 0;
            auto crop = ocr_crop::extract(pixels, w, h, 3, x, y, cw, ch, 0, &ow, &oh);
            const int recognizer_width = word_mode ? 200 : std::max(200, (int)std::ceil(64.0 * ow / std::max(1, oh)));
            if (!easyocr_ocr_set_width(rec, recognizer_width)) {
                rc = 6;
                continue;
            }
            int out_len = 0;
            const char * text = easyocr_ocr_recognize(rec, crop.data(), ow, oh, 3, &out_len);
            std::printf("dbnet-easyocr unit=%zu box=%.1f,%.1f %.1fx%.1f rec_conf=%.4f text=%.*s\n", i, line.x0, line.y0,
                        line.x1 - line.x0, line.y1 - line.y0, easyocr_ocr_last_confidence(rec), out_len,
                        text ? text : "");
        }
        if (lines.empty()) rc = 5;
    }
    easyocr_ocr_free(rec);
    ocr_detect::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
