// tests/test_surya_det.cpp — basic load + forward pass test for surya detector
// Supports PNG/JPG/BMP via stb_image, or uses synthetic if no image given.
#include "surya_det.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// stb_image (implementation in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *ch, int desired_ch);
    void stbi_image_free(void *p);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <surya-det.gguf> [image.png|jpg|bmp]\n", argv[0]);
        return 1;
    }

    surya_det_context* ctx = surya_det_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const surya_det_hparams* hp = surya_det_get_hparams(ctx);
    printf("Model loaded: input=%dx%d, classes=%d\n",
           hp->input_w, hp->input_h, hp->num_classes);

    int w, h;
    std::vector<uint8_t> img;

    if (argc >= 3) {
        // Load image via stb_image
        int ch;
        stbi_uc* px = stbi_load(argv[2], &w, &h, &ch, 3);
        if (!px) {
            fprintf(stderr, "Failed to load image: %s\n", argv[2]);
            surya_det_free(ctx);
            return 1;
        }
        img.assign(px, px + w * h * 3);
        stbi_image_free(px);
        printf("Loaded image: %s (%dx%d)\n", argv[2], w, h);
    } else {
        // Create a synthetic test image (white with dark horizontal lines)
        w = 800; h = 600;
        img.resize(w * h * 3, 240);
    for (int y = 100; y < h - 100; y += 80) {
        for (int x = 50; x < w - 50; x++) {
            for (int c = 0; c < 3; c++) {
                img[(y * w + x) * 3 + c] = 30;
            }
            // Also the next few rows
            for (int dy = 1; dy < 12 && y + dy < h; dy++) {
                for (int c = 0; c < 3; c++) {
                    img[((y + dy) * w + x) * 3 + c] = 30;
                }
            }
        }
    }
    }

    printf("Running detection on %dx%d image...\n", w, h);
    int out_h = 0, out_w = 0;
    const float* heatmap = surya_det_detect(ctx, img.data(), w, h, 3, &out_h, &out_w);
    if (!heatmap) {
        fprintf(stderr, "Detection failed\n");
        surya_det_free(ctx);
        return 1;
    }

    printf("Heatmap: %dx%d (2 channels)\n", out_w, out_h);

    // Print stats for each channel
    for (int c = 0; c < 2; c++) {
        float mn = heatmap[c * out_h * out_w];
        float mx = mn;
        double sum = 0;
        int above_thresh = 0;
        for (int i = 0; i < out_h * out_w; i++) {
            float v = heatmap[c * out_h * out_w + i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
            if (v > 0.5f) above_thresh++;
        }
        printf("  Channel %d: min=%.4f max=%.4f mean=%.4f above_0.5=%d/%d\n",
               c, mn, mx, (float)(sum / (out_h * out_w)),
               above_thresh, out_h * out_w);
    }

    // Extract bounding boxes
    int n_boxes = 0;
    const surya_det_bbox* boxes = surya_det_get_boxes(ctx, w, h, 0.6f, 0.35f, &n_boxes);
    printf("\nDetected %d text regions:\n", n_boxes);
    for (int i = 0; i < n_boxes && i < 20; i++) {
        printf("  Box %d: (%.0f,%.0f)-(%.0f,%.0f) conf=%.3f\n",
               i, boxes[i].x0, boxes[i].y0, boxes[i].x1, boxes[i].y1,
               boxes[i].confidence);
    }
    if (n_boxes > 20) printf("  ... and %d more\n", n_boxes - 20);

    surya_det_free(ctx);
    printf("PASS\n");
    return 0;
}
