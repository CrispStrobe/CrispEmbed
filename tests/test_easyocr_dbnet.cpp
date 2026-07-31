#include "easyocr_ocr.h"
#include "easyocr_layout.h"
#include "easyocr_postprocess.h"
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
        std::vector<easyocr_layout::region> regions;
        regions.reserve(boxes.size());
        for (const auto & b : boxes) regions.push_back({ b.x, b.y, b.w, b.h, b.score });
        const auto lines = easyocr_layout::order_regions(regions, word_mode ? easyocr_layout::ordering_mode::words
                                                                            : easyocr_layout::ordering_mode::lines);
        std::vector<easyocr_layout::word> handoff;
        std::printf("dbnet-easyocr mode=%s units=%zu\n", word_mode ? "words" : "lines", lines.size());
        for (size_t i = 0; i < lines.size(); ++i) {
            const auto & line = lines[i];
            const int x = std::max(0, (int)line.x - 2), y = std::max(0, (int)line.y - 2);
            const int cw = std::min(w - x, (int)line.w + 4);
            const int ch = std::min(h - y, (int)line.h + 4);
            int ow = 0, oh = 0;
            auto crop = ocr_crop::extract(pixels, w, h, 3, x, y, cw, ch, 0, &ow, &oh);
            const int recognizer_width = easyocr_postprocess::recognizer_canvas_width(ow, oh);
            if (!easyocr_ocr_set_width(rec, recognizer_width)) {
                rc = 6;
                continue;
            }
            int out_len = 0;
            const char * text = easyocr_ocr_recognize(rec, crop.data(), ow, oh, 3, &out_len);
            const float rec_conf = easyocr_ocr_last_confidence(rec);
            handoff.push_back({ text ? std::string(text, out_len) : std::string(), line.x, line.y, line.w, line.h,
                                rec_conf, 0, line.line, (int)i });
            std::printf("dbnet-easyocr unit=%zu box=%.1f,%.1f %.1fx%.1f det_conf=%.4f rec_conf=%.4f text=%.*s\n", i,
                        line.x, line.y, line.w, line.h, line.score, rec_conf, out_len, text ? text : "");
        }
        const auto normalized = easyocr_layout::normalize_boxes(handoff, w, h);
        std::printf("dbnet-easyocr handoff words=%zu normalized=%zu text=%s\n", handoff.size(), normalized.size(),
                    easyocr_layout::join_lines(handoff).c_str());
        if (lines.empty()) rc = 5;
    }
    easyocr_ocr_free(rec);
    ocr_detect::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
