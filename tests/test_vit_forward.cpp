// test_vit_forward.cpp — minimal test for vit_embed forward path.
//
// Usage:
//   ./test_vit_forward siglip-base.gguf test_pixels.bin [ref_embedding.bin]
//
// test_pixels.bin: float32 [3, H, W] preprocessed pixels (CHW, normalized)
// ref_embedding.bin: optional float32 [D] HF reference for cosine check

#include "../src/vit_embed.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static std::vector<float> read_f32(const char* path, size_t expected = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    size_t n = bytes / sizeof(float);
    if (expected > 0 && n != expected) {
        fprintf(stderr, "read_f32: %s has %zu floats, expected %zu\n", path, n, expected);
    }
    std::vector<float> data(n);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

static float cosine(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return (na > 1e-18 && nb > 1e-18) ? (float)(dot / (sqrt(na) * sqrt(nb))) : 0.0f;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <gguf> <pixels.bin> [ref.bin]\n", argv[0]);
        return 1;
    }

    const char* gguf_path = argv[1];
    const char* pixel_path = argv[2];
    const char* ref_path = argc > 3 ? argv[3] : nullptr;

    // Load model
    vit_embed::context* ctx = nullptr;
    if (!vit_embed::load(&ctx, gguf_path, 4)) {
        fprintf(stderr, "Failed to load %s\n", gguf_path);
        return 1;
    }

    int sz = vit_embed::image_size(ctx);
    int d = vit_embed::dim(ctx);
    fprintf(stderr, "Loaded: image=%dx%d dim=%d\n", sz, sz, d);

    // Load pixels
    size_t expected_pixels = 3 * sz * sz;
    auto pixels = read_f32(pixel_path, expected_pixels);
    if (pixels.size() != expected_pixels) {
        fprintf(stderr, "Pixel file size mismatch: %zu vs expected %zu\n",
                pixels.size(), expected_pixels);
        vit_embed::free(ctx);
        return 1;
    }

    // Encode
    fprintf(stderr, "Encoding...\n");
    auto emb = vit_embed::encode(ctx, pixels.data(), sz, sz);
    if (emb.empty()) {
        fprintf(stderr, "Encoding failed\n");
        vit_embed::free(ctx);
        return 1;
    }

    fprintf(stderr, "Output dim: %zu\n", emb.size());
    fprintf(stderr, "First 8: ");
    for (int i = 0; i < 8 && i < (int)emb.size(); i++)
        fprintf(stderr, "%.6f ", emb[i]);
    fprintf(stderr, "\n");

    // Check norm
    float norm = 0;
    for (float v : emb) norm += v * v;
    norm = sqrtf(norm);
    fprintf(stderr, "Norm: %.6f\n", norm);

    // Compare with reference
    if (ref_path) {
        auto ref = read_f32(ref_path, d);
        if ((int)ref.size() == d) {
            float cos = cosine(emb.data(), ref.data(), d);
            fprintf(stderr, "Cosine vs reference: %.6f  [%s]\n",
                    cos, cos > 0.95f ? "PASS" : "FAIL");
        }
    }

    // Print embedding to stdout
    for (int i = 0; i < (int)emb.size(); i++) {
        if (i > 0) printf(" ");
        printf("%.6f", emb[i]);
    }
    printf("\n");

    vit_embed::free(ctx);
    return 0;
}
