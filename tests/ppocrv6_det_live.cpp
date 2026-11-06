// Test-only live probe: real detector, quadrilateral crops and recognizer.
#include "core/clean_exit.h"
#include "ocr_crop.h"
#include "ppocrv6_det.h"
#include "ppocrv6_ocr.h"
#include "stb_image.h"

#include <cstdio>
#include <fstream>
#include <string>

static bool write_floats(const std::string & path, const float * data, size_t n) {
    if (!data || !n) return false;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), std::streamsize(n * sizeof(float)));
    return bool(out);
}

static int run(int argc, char ** argv) {
    if (argc < 5) return 2;
    auto * det = ppocrv6_det::init(argv[1], 2);
    auto * rec = ppocrv6_ocr_init(argv[2], 2);
    if (!det || !rec) return 3;
    int rc = 0;
    for (int i = 4; i < argc; ++i) {
        int width = 0, height = 0, channels = 0;
        auto * image = stbi_load(argv[i], &width, &height, &channels, 3);
        if (!image) {
            rc = 4;
            break;
        }
        const auto boxes = ppocrv6_det::detect_file(det, argv[i], 0.2f);
        int h = 0, w = 0;
        const float * probability = ppocrv6_det::last_probability(det, &h, &w);
        const std::string prefix = std::string(argv[3]) + "/" + std::to_string(i - 4);
        if (!write_floats(prefix + "-prob.bin", probability, size_t(h) * size_t(w))) {
            stbi_image_free(image);
            rc = 5;
            break;
        }
        std::printf("CASE %d %d %d %zu\n", i - 4, h, w, boxes.size());
        for (size_t b = 0; b < boxes.size(); ++b) {
            const auto & box = boxes[b];
            int cw = 0, ch = 0;
            auto crop = ocr_crop::extract_quad(image, width, height, 3, box.qx, box.qy, 2, &cw, &ch);
            if (crop.empty()) {
                rc = 6;
                break;
            }
            int n = 0;
            const char * text = ppocrv6_ocr_recognize_raw(rec, crop.data(), cw, ch, 3, &n);
            if (!text || n < 0) {
                rc = 7;
                break;
            }
            std::printf("BOX %d %zu %.9g %.9g %.9g %.9g %.9g", i - 4, b, box.x, box.y, box.w, box.h, box.score);
            for (int k = 0; k < 4; ++k) std::printf(" %.9g %.9g", box.qx[k], box.qy[k]);
            std::printf(" ");
            for (int j = 0; j < n; ++j) std::printf("%02x", static_cast<unsigned char>(text[j]));
            std::printf("\n");
        }
        for (const char * name :
             { "graph_stage0", "backbone_stage0", "graph_med_adjust0", "med_adjust0", "graph_head_down", "head_down",
               "graph_head_final_pre", "head_final_pre", "graph_head_final" }) {
            size_t n = 0;
            const float * data = ppocrv6_det::last_stage(det, name, &n);
            if (data && !write_floats(prefix + "-" + name + ".bin", data, n)) rc = 8;
        }
        stbi_image_free(image);
        if (rc) break;
    }
    ppocrv6_ocr_free(rec);
    ppocrv6_det::free(det);
    return rc;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
