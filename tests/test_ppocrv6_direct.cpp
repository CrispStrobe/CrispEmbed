#include "core/clean_exit.h"
#include "ocr_crop.h"
#include "ppocrv6_det.h"
#include "ppocrv6_ocr.h"
#include "pplcnet_orientation.h"
#include "stb_image.h"
#include "../ggml/examples/stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "usage: %s <ppocrv6-det.gguf> <ppocrv6-rec.gguf> <image> [orientation.gguf]\n", argv[0]);
        return 2;
    }
    int w = 0, h = 0, channels = 0;
    auto * pixels = stbi_load(argv[3], &w, &h, &channels, 3);
    if (!pixels) return 3;
    auto * det = ppocrv6_det::init(argv[1], 1);
    auto * rec = ppocrv6_ocr_init(argv[2], 1);
    auto * ori = argc == 5 ? pplcnet_orientation::init(argv[4], 1) : nullptr;
    int rc = det && rec && (argc == 4 || ori) ? 0 : 4;
    if (rc == 0) {
        const auto started = std::chrono::steady_clock::now();
        const auto boxes = ppocrv6_det::detect_file(det, argv[3], 0.2f);
        size_t region_limit = boxes.size();
        if (const char * env = std::getenv("PPOCRV6_DIRECT_MAX_REGIONS"))
            region_limit = std::min(region_limit, std::strtoul(env, nullptr, 10));
        size_t rotated = 0;
        const char * crop_prefix = std::getenv("PPOCRV6_DIRECT_SAVE_CROPS");
        std::printf("ppocrv6-direct detector_regions=%zu image=%dx%d orientation=%s\n", boxes.size(), w, h,
                    ori ? "pplcnet" : "heuristic-disabled");
        for (size_t i = 0; i < region_limit; ++i) {
            const auto & b = boxes[i];
            int cw = 0, ch = 0;
            auto crop = ocr_crop::extract_quad(pixels, w, h, 3, b.qx, b.qy, 2, &cw, &ch);
            if (crop.empty()) continue;
            if (crop_prefix && *crop_prefix) {
                const std::string path = std::string(crop_prefix) + "-" + std::to_string(i) + ".ppm";
                std::ofstream out(path, std::ios::binary);
                out << "P6\n" << cw << " " << ch << "\n255\n";
                out.write(reinterpret_cast<const char *>(crop.data()), (std::streamsize)crop.size());
                if (!out) std::fprintf(stderr, "failed to save crop %s\n", path.c_str());
            }
            int angle = 0;
            float orientation_confidence = 0.0f;
            if (ori) {
                const auto classified = pplcnet_orientation::classify_raw(ori, crop.data(), cw, ch, 3);
                angle = classified.angle;
                orientation_confidence = classified.confidence;
                if (angle == 180) {
                    ocr_crop::rotate_180_rgb(crop, cw, ch);
                    rotated++;
                }
            }
            int n = 0;
            const char * text = ppocrv6_ocr_recognize_raw(rec, crop.data(), cw, ch, 3, &n);
            std::printf("ppocrv6-direct region=%zu score=%.4f quad=(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f)(%.1f,%.1f) "
                        "crop=%dx%d angle=%d orientation_confidence=%.5f text=%.*s\n",
                        i, b.score, b.qx[0], b.qy[0], b.qx[1], b.qy[1], b.qx[2], b.qy[2], b.qx[3], b.qy[3], cw, ch,
                        angle, orientation_confidence, n, text ? text : "");
        }
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::printf("ppocrv6-direct summary processed=%zu rotated_180=%zu elapsed_ms=%.3f\n", region_limit, rotated,
                    elapsed_ms);
        if (boxes.empty()) rc = 5;
    }
    ppocrv6_ocr_free(rec);
    pplcnet_orientation::free(ori);
    ppocrv6_det::free(det);
    stbi_image_free(pixels);
    core_util::clean_exit(rc);
    return rc;
}
