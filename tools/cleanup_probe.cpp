// Probe: which scan-cleanup step distorts a VLM-OCR input image?
#include "scan_cleanup.h"
#include <cstdio>
#include <cstring>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    const char* in = argv[1];
    int w, h, c;
    unsigned char* px = stbi_load(in, &w, &h, &c, 3);
    if (!px) { fprintf(stderr, "load fail\n"); return 1; }
    fprintf(stderr, "input: %dx%d\n", w, h);

    // Detected skew angle (on grayscale [0,1]) — the core regression metric.
    std::vector<float> gray((size_t)w*h);
    for (int i=0;i<w*h;i++) gray[i] = (0.299f*px[i*3]+0.587f*px[i*3+1]+0.114f*px[i*3+2])/255.0f;
    fprintf(stderr, "detect_angle = %.2f deg\n", scan_cleanup_detect_angle(gray.data(), w, h, 15.0f));

    scan_cleanup_ctx* ctx = scan_cleanup_init(nullptr, 4);

    struct Case { const char* name; int deskew, crop, whiten; };
    Case cases[] = {
        {"full_defaults", 1,1,1},
        {"deskew_only",   1,0,0},
        {"crop_only",     0,1,0},
        {"whiten_only",   0,0,1},
        {"none",          0,0,0},
    };
    for (auto& cs : cases) {
        scan_cleanup_params p = scan_cleanup_defaults();
        p.deskew = cs.deskew; p.crop_borders = cs.crop; p.whiten_background = cs.whiten;
        p.binarize = 0;
        unsigned char* out = nullptr; int ow = 0, oh = 0;
        int r = scan_cleanup_process(ctx, px, w, h, 3, p, &out, &ow, &oh);
        fprintf(stderr, "%-15s -> %dx%d (rc=%d)\n", cs.name, ow, oh, r);
        if (r == 0 && out) {
            char fn[256]; snprintf(fn, sizeof(fn), "%s.clean_%s.png", in, cs.name);
            stbi_write_png(fn, ow, oh, 3, out, ow*3);
            scan_cleanup_free_image(out);
        }
    }
    scan_cleanup_free(ctx);
    stbi_image_free(px);
    return 0;
}
