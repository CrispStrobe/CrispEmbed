// test_bttr_beam.cpp — Test beam search for BTTR.
// Usage: ./test-bttr-beam model.gguf image.f32 WxH [beam_width]

#include "bttr_ocr.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <image.f32> <WxH> [beam_width]\n", argv[0]);
        return 1;
    }

    int beam_width = argc >= 5 ? atoi(argv[4]) : 5;

    bttr_ocr_context * ctx = bttr_ocr_init(argv[1], 4);
    if (!ctx) { fprintf(stderr, "Failed to load\n"); return 1; }

    int W = 0, H = 0;
    sscanf(argv[3], "%dx%d", &W, &H);
    std::vector<float> img(W * H);
    FILE * f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", argv[2]); return 1; }
    fread(img.data(), sizeof(float), W * H, f);
    fclose(f);

    // Greedy
    int len = 0;
    const char * greedy = bttr_ocr_recognize(ctx, img.data(), W, H, &len);
    fprintf(stderr, "Greedy: %s\n", greedy ? greedy : "(null)");

    // Beam
    const char * beam = bttr_ocr_recognize_beam(ctx, img.data(), W, H, beam_width, &len);
    fprintf(stderr, "Beam(%d): %s\n", beam_width, beam ? beam : "(null)");

    bttr_ocr_free(ctx);
    return 0;
}
