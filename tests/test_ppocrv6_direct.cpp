#include "core/clean_exit.h"
#include "ocr_crop.h"
#include "ppocrv6_det.h"
#include "ppocrv6_ocr.h"
#include "stb_image.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <ppocrv6-det.gguf> <ppocrv6-rec.gguf> <image>\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, channels = 0;
    auto * pixels = stbi_load(argv[3], &w, &h, &channels, 3);
    if (!pixels) return 3;
    auto * det = ppocrv6_det::init(argv[1], 1);
    auto * rec = ppocrv6_ocr_init(argv[2], 1);
    int rc = det && rec ? 0 : 4;
    if (rc == 0) {
        const auto boxes = ppocrv6_det::detect_raw(det, pixels, w, h, 3, 0.2f);
        std::printf("ppocrv6-direct detector_regions=%zu image=%dx%d\n", boxes.size(), w, h);
        for (size_t i = 0; i < boxes.size(); ++i) {
            const auto & b = boxes[i];
            int cw = 0, ch = 0;
            auto crop = ocr_crop::extract_quad(pixels, w, h, 3, b.qx, b.qy, 2, &cw, &ch);
            if (crop.empty()) continue;
            int n = 0;
            const char * text = ppocrv6_ocr_recognize_raw(rec, crop.data(), cw, ch, 3, &n);
            std::printf("ppocrv6-direct region=%zu score=%.4f quad=(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) "
                        "crop=%dx%d text=%.*s\n",
                        i, b.score, b.qx[0], b.qy[0], b.qx[1], b.qy[1], b.qx[2], b.qy[2], b.qx[3], b.qy[3], cw, ch, n,
                        text ? text : "");
        }
        if (boxes.empty()) rc = 5;
    }
    ppocrv6_ocr_free(rec);
    ppocrv6_det::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
