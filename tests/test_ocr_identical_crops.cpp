#include "easyocr_ocr.h"
#include "easyocr_postprocess.h"
#include "ocr_crop.h"
#include "tesseract_lstm.h"
#include "core/clean_exit.h"

#include "stb_image.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s <easyocr.gguf> <tesseract.gguf> <image> <x> <y> <w> <h>\n", argv[0]);
        return 2;
    }
    const int x = std::atoi(argv[4]), y = std::atoi(argv[5]);
    const int crop_w = std::atoi(argv[6]), crop_h = std::atoi(argv[7]);
    int width = 0, height = 0, channels = 0;
    unsigned char * image = stbi_load(argv[3], &width, &height, &channels, 3);
    if (!image || crop_w <= 0 || crop_h <= 0 || x < 0 || y < 0 || x + crop_w > width || y + crop_h > height) {
        std::fprintf(stderr, "invalid image or crop\n");
        if (image) stbi_image_free(image);
        return 3;
    }
    int out_w = 0, out_h = 0;
    const auto crop =
        ocr_crop::extract(image, width, height, 3, (float)x, (float)y, (float)crop_w, (float)crop_h, 0, &out_w, &out_h);
    stbi_image_free(image);
    if (crop.empty()) return 4;

    easyocr_ocr_context * easy = easyocr_ocr_init(argv[1], 4);
    tesseract_lstm_context * tess = tesseract_lstm_init(argv[2], 1);
    if (!easy || !tess) {
        easyocr_ocr_free(easy);
        tesseract_lstm_free(tess);
        return 5;
    }
    if (!easyocr_ocr_set_width(easy, easyocr_postprocess::recognizer_canvas_width(out_w, out_h))) {
        easyocr_ocr_free(easy);
        tesseract_lstm_free(tess);
        return 6;
    }
    int easy_len = 0, tess_len = 0;
    const char * easy_text = easyocr_ocr_recognize(easy, crop.data(), out_w, out_h, 3, &easy_len);
    std::vector<uint8_t> gray((size_t)out_w * out_h);
    for (int yy = 0; yy < out_h; ++yy) {
        for (int xx = 0; xx < out_w; ++xx) {
            const size_t p = ((size_t)yy * out_w + xx) * 3;
            gray[(size_t)yy * out_w + xx] = (uint8_t)((77 * crop[p] + 150 * crop[p + 1] + 29 * crop[p + 2]) >> 8);
        }
    }
    const char * tess_text = tesseract_lstm_recognize(tess, gray.data(), out_w, out_h, &tess_len);
    std::printf("crop=%dx%d easyocr=%.*s easy_conf=%.6f tesseract=%.*s tess_conf=%.6f\n", out_w, out_h, easy_len,
                easy_text ? easy_text : "", easyocr_ocr_last_confidence(easy), tess_len, tess_text ? tess_text : "",
                tesseract_lstm_mean_confidence(tess));
    easyocr_ocr_free(easy);
    tesseract_lstm_free(tess);
    return easy_text && tess_text ? 0 : 7;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
