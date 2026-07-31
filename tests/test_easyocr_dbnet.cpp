#include "easyocr_ocr.h"
#include "ocr_crop.h"
#include "ocr_detect.h"
#include "core/clean_exit.h"
#include "stb_image.h"

#include <algorithm>
#include <cstdio>

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
        std::printf("dbnet-easyocr boxes=%zu\n", boxes.size());
        for (size_t i = 0; i < boxes.size(); ++i) {
            const auto & b = boxes[i];
            const int x = std::max(0, (int)b.x - 2), y = std::max(0, (int)b.y - 2);
            const int cw = std::min(w - x, (int)b.w + 4), ch = std::min(h - y, (int)b.h + 4);
            int ow = 0, oh = 0;
            auto crop = ocr_crop::extract(pixels, w, h, 3, x, y, cw, ch, 0, &ow, &oh);
            int out_len = 0;
            const char * text = easyocr_ocr_recognize(rec, crop.data(), ow, oh, 3, &out_len);
            std::printf("dbnet-easyocr region=%zu box=%.1f,%.1f %.1fx%.1f text=%.*s\n", i, b.x, b.y, b.w, b.h, out_len,
                        text ? text : "");
        }
        if (boxes.empty()) rc = 5;
    }
    easyocr_ocr_free(rec);
    ocr_detect::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
