#include "core/clean_exit.h"
#include "ppocrv6_ocr.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <rec.gguf> <image> [image ...]\n", argv[0]);
        return 1;
    }
    auto * ctx = ppocrv6_ocr_init(argv[1], 1);
    if (!ctx) return 1;
    bool ok = true;
    std::vector<const unsigned char *> image_pixels;
    std::vector<int> widths, heights, channels;
    std::vector<std::string> scalar_text;
    const auto scalar_started = std::chrono::steady_clock::now();
    for (int i = 2; i < argc; ++i) {
        int w = 0, h = 0, ch = 0;
        auto * pixel = stbi_load(argv[i], &w, &h, &ch, 3);
        if (!pixel) {
            fprintf(stderr, "failed to load image: %s\n", argv[i]);
            ok = false;
            continue;
        }
        int n = 0;
        const char * text = ppocrv6_ocr_recognize_raw(ctx, pixel, w, h, 3, &n);
        scalar_text.emplace_back(text ? std::string(text, (size_t)n) : std::string());
        printf("text=%s\n", text ? text : "<null>");
        ok = ok && text != nullptr;
        image_pixels.push_back(pixel);
        widths.push_back(w);
        heights.push_back(h);
        channels.push_back(3);
    }
    const double scalar_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - scalar_started).count();
    std::vector<const uint8_t *> batch_pixels;
    std::vector<std::string> batch_text(image_pixels.size(), std::string(4096, '\0'));
    std::vector<char *> outputs;
    std::vector<int> capacities(image_pixels.size(), 4096), lengths(image_pixels.size(), 0);
    batch_pixels.reserve(image_pixels.size());
    outputs.reserve(image_pixels.size());
    for (size_t i = 0; i < image_pixels.size(); ++i) {
        batch_pixels.push_back(image_pixels[i]);
        outputs.push_back(batch_text[i].data());
    }
    if (!image_pixels.empty()) {
        const auto batch_started = std::chrono::steady_clock::now();
        const int completed = ppocrv6_ocr_recognize_raw_batch(ctx, batch_pixels.data(), widths.data(), heights.data(),
                                                              channels.data(), (int)image_pixels.size(), outputs.data(),
                                                              capacities.data(), lengths.data());
        const double batch_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - batch_started).count();
        printf("scalar_ms=%.3f batch_ms=%.3f batch_completed=%d\n", scalar_ms, batch_ms, completed);
        ok = ok && completed == (int)image_pixels.size();
        for (size_t i = 0; i < image_pixels.size(); ++i) {
            const std::string batched(outputs[i], (size_t)lengths[i]);
            ok = ok && batched == scalar_text[i];
            printf("batch[%zu]=%s parity=%s\n", i, batched.c_str(), batched == scalar_text[i] ? "PASS" : "FAIL");
        }
    }
    for (const auto * pixel : image_pixels) {
        stbi_image_free((void *)pixel);
    }
    ppocrv6_ocr_free(ctx);
    core_util::clean_exit(ok && !scalar_text.empty() ? 0 : 1);
}
