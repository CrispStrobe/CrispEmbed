#include "lighton4b_image_compose.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <limits>

using namespace qwen35_ocr;
static int checks = 0;
static void environment(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
#define CHECK(x)                                                                                                       \
    do {                                                                                                               \
        ++checks;                                                                                                      \
        if (!(x)) {                                                                                                    \
            std::cerr << "failed line " << __LINE__ << '\n';                                                           \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)
int main(int argc, char ** argv) {
    if (argc == 6) {
        environment("CRISPEMBED_QWEN35_OCR", "1");
        environment("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE", "1");
        const int height = std::atoi(argv[2]), width = std::atoi(argv[3]);
        if (height < 1 || width < 1 || height > 32768 || width > 32768 || int64_t(height) * width > 16777216) return 2;
        std::vector<uint8_t> rgb(size_t(height) * width * 3);
        std::ifstream input(argv[1], std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(rgb.data()), std::streamsize(rgb.size())) || input.peek() != EOF)
            return 2;
        image_inputs image;
        if (!lighton4b_preprocess_rgb(rgb.data(), height, width, image)) return 2;
        const std::string prefix = argv[4];
        std::ofstream patches(prefix + ".f32", std::ios::binary), resized(prefix + ".rgb", std::ios::binary);
        patches.write(reinterpret_cast<const char *>(image.patches.data()),
                      std::streamsize(image.patches.size() * sizeof(float)));
        resized.write(reinterpret_cast<const char *>(image.resized_rgb.data()),
                      std::streamsize(image.resized_rgb.size()));
        if (!patches || !resized || std::string(argv[5]) != "dump") return 2;
        std::cout << image.height << " " << image.width << " " << image.grid[0] << " " << image.grid[1] << " "
                  << image.grid[2] << "\n";
        return 0;
    }
    if (argc != 1) return 2;
    environment("CRISPEMBED_QWEN35_OCR", "1");
    environment("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE", "1");
    std::vector<float> text(6 * 2560), visual(4 * 2560);
    for (size_t i = 0; i < text.size(); ++i) text[i] = float(i);
    for (size_t i = 0; i < visual.size(); ++i) visual[i] = -float(i + 1);
    const std::vector<int> modalities{ 0, 1, 1, 1, 1, 0 }, grid{ 1, 4, 4 };
    lighton4b_image_prefill out;
    CHECK(lighton4b_image_prefill_cpu(text, modalities, grid, visual, out));
    CHECK(std::equal(text.begin(), text.begin() + 2560, out.hidden.begin()));
    CHECK(std::equal(visual.begin(), visual.end(), out.hidden.begin() + 2560));
    CHECK(std::equal(text.begin() + 5 * 2560, text.end(), out.hidden.begin() + 5 * 2560));
    CHECK(out.positions.axes == std::vector<float>({ 0, 1, 1, 1, 1, 3, 0, 1, 1, 2, 2, 3, 0, 1, 2, 1, 2, 3 }));
    CHECK(out.positions.rope_delta == -2);
    for (const char * gate : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE" }) {
        for (const char * value : { "", "0" }) {
            environment(gate, value);
            out.hidden = { 42 };
            CHECK(!lighton4b_image_prefill_cpu(text, modalities, grid, visual, out));
            CHECK(out.hidden.empty() && out.positions.axes.empty());
        }
        environment(gate, nullptr);
        CHECK(!lighton4b_image_prefill_cpu(text, modalities, grid, visual, out));
        environment(gate, "1");
    }
    for (auto bad : { std::vector<int>{ 1, 4, 3 }, std::vector<int>{ 2, 4, 4 }, std::vector<int>{ 1, 4, 4, 1, 4, 4 },
                      std::vector<int>{ 1, 64, 64 }, std::vector<int>{ 1, -2, 4 } })
        CHECK(!lighton4b_image_prefill_cpu(text, modalities, bad, visual, out));
    for (auto bad : { std::vector<int>{ 0, 1, 1, 0, 1, 1 }, std::vector<int>{ 0, 1, 1, 1, 0, 0 },
                      std::vector<int>{ 0, 2, 1, 1, 1, 0 } })
        CHECK(!lighton4b_image_prefill_cpu(text, bad, grid, visual, out));
    auto nonfinite = visual;
    nonfinite[123] = std::numeric_limits<float>::infinity();
    CHECK(!lighton4b_image_prefill_cpu(text, modalities, grid, nonfinite, out));
    nonfinite = text;
    nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!lighton4b_image_prefill_cpu(nonfinite, modalities, grid, visual, out));
    visual.pop_back();
    CHECK(!lighton4b_image_prefill_cpu(text, modalities, grid, visual, out));
    text.pop_back();
    CHECK(!lighton4b_image_prefill_cpu(text, modalities, grid, visual, out));
    // Actual reusable resize/normalization/patch layout, no model or framework.
    std::vector<uint8_t> rgb(256 * 256 * 3, 255);
    image_inputs image;
    CHECK(lighton4b_preprocess_rgb(rgb.data(), 256, 256, image));
    CHECK(image.grid == std::vector<int>({ 1, 16, 16 }));
    CHECK(image.patches.size() == 256 * 1536);
    CHECK(std::all_of(image.patches.begin(), image.patches.end(), [](float x) { return x == 1; }));
    // Independently authored channel/time and 2x2 merge-block coordinates.
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 256; ++x)
            for (int c = 0; c < 3; ++c) rgb[(y * 256 + x) * 3 + c] = uint8_t(y + x + c * 20);
    CHECK(lighton4b_preprocess_rgb(rgb.data(), 256, 256, image));
    auto normalized = [](int value) { return (float(value) - 127.5f) / 127.5f; };
    CHECK(image.patches[0] == normalized(0));
    CHECK(image.patches[256] == image.patches[0]);
    CHECK(image.patches[512] == normalized(20));
    CHECK(image.patches[1536] == normalized(16));
    CHECK(image.patches[2 * 1536] == normalized(16));
    CHECK(image.patches[3 * 1536] == normalized(32));
    CHECK(image.patches[4 * 1536] == normalized(32));
    std::vector<uint8_t> form(317 * 452 * 3, 255);
    CHECK(lighton4b_preprocess_rgb(form.data(), 317, 452, image));
    CHECK(image.grid == std::vector<int>({ 1, 20, 28 }));
    CHECK(image.patches.size() == 560 * 1536);
    CHECK(!lighton4b_preprocess_rgb(nullptr, 256, 256, image) && image.patches.empty());
    vision_weights weights;
    vision_config config;
    vision_outputs result;
    CHECK(!lighton4b_image_vision_cpu(weights, config, image, 1, result));
    std::cout << "passed=1 checks=" << checks << " model_inference=0\n";
}
