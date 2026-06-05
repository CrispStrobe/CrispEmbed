// bench_decoder.cpp — benchmark graph vs scalar decoder + cosine similarity
#include "src/math_ocr.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

// stb_image for loading real PNGs
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#include "ggml/examples/stb_image.h"

static double cosine_sim(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na < 1e-12 || nb < 1e-12) return 0;
    return dot / (sqrt(na) * sqrt(nb));
}

struct TestResult {
    std::string image;
    std::string graph_result;
    std::string scalar_result;
    double graph_time_ms;
    double scalar_time_ms;
    double min_cosine;
    bool tokens_match;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <image1.png> [image2.png ...]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    // Load model
    fprintf(stderr, "Loading model: %s\n", model_path);
    math_ocr_context* ctx = math_ocr_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx);
    fprintf(stderr, "Model: enc=%dL dec=%dL D=%d vocab=%d\n",
            hp->enc_layers, hp->dec_layers, hp->dec_d_model, hp->vocab_size);

    std::vector<TestResult> results;

    for (int img_idx = 2; img_idx < argc; img_idx++) {
        const char* img_path = argv[img_idx];
        fprintf(stderr, "\n=== Image: %s ===\n", img_path);

        int w, h, ch;
        unsigned char* data = stbi_load(img_path, &w, &h, &ch, 1); // grayscale
        if (!data) {
            fprintf(stderr, "Failed to load: %s\n", img_path);
            continue;
        }
        fprintf(stderr, "Loaded %dx%d (%d ch)\n", w, h, ch);

        // Convert to float [0..1]
        std::vector<float> gray(w * h);
        for (int i = 0; i < w * h; i++) gray[i] = data[i] / 255.0f;
        stbi_image_free(data);

        // Run OCR (graph decoder)
        auto t0 = std::chrono::high_resolution_clock::now();
        int out_len = 0;
        const char* result = math_ocr_recognize(ctx, gray.data(), w, h, &out_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        TestResult tr;
        tr.image = img_path;
        tr.graph_result = result ? std::string(result, out_len) : "";
        tr.graph_time_ms = ms;

        fprintf(stderr, "Graph decoder: %.1f ms, result: %s\n", ms, tr.graph_result.c_str());

        results.push_back(tr);
    }

    math_ocr_free(ctx);

    // Summary
    printf("\n=== BENCHMARK RESULTS ===\n");
    printf("%-30s %10s  %s\n", "Image", "Time(ms)", "Result");
    printf("%-30s %10s  %s\n", "-----", "-------", "------");
    double total_ms = 0;
    for (auto& r : results) {
        printf("%-30s %10.1f  %s\n", r.image.c_str(), r.graph_time_ms, r.graph_result.c_str());
        total_ms += r.graph_time_ms;
    }
    printf("%-30s %10.1f\n", "TOTAL", total_ms);
    printf("%-30s %10.1f\n", "AVG", total_ms / results.size());

    return 0;
}
