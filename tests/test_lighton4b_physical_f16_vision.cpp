#include "lighton4b_physical_f16_vision.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <memory>
using namespace qwen35_ocr;
static int checks = 0;
static void check(bool ok, const char * why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
static void flag(const char * k, const char * v) {
    setenv(k, v, 1);
}
int main() {
    try {
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION", "0");
        check(!lighton4b_physical_f16_vision_mode(true), "default disabled");
        check(lighton4b_physical_f16_vision_valid(true), "disabled inert");
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION", "1");
        check(!lighton4b_physical_f16_vision_mode(false) && lighton4b_physical_f16_vision_valid(false),
              "nonphysical inert");
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "0");
        check(!lighton4b_physical_f16_vision_valid(true), "required preprocessing closed");
        for (auto * k :
             { "CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE",
               "CRISPEMBED_LIGHTON4B_VISION_OWNER", "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "CRISPEMBED_QWEN35_OCR" })
            flag(k, "1");
        vision_weights weights;
        vision_config config;
        image_inputs image;
        vision_outputs output;
        output.merged = { 1 };
        check(!lighton4b_physical_f16_vision_cpu(weights, config, image, 4, false, output) && output.merged.empty(),
              "no unowned profile");
        if (!lighton4b_physical_f16_vision_available()) {
            check(!lighton4b_physical_f16_vision_valid(true), "unsupported platform closed");
            check(!lighton4b_physical_f16_vision_cpu(weights, config, image, 4, true, output) && output.merged.empty(),
                  "unsupported execution closed");
            std::cout << "PHYSICAL_F16_VISION_UNSUPPORTED PASS " << checks << "\n";
            return 0;
        }
        check(lighton4b_physical_f16_vision_valid(true), "supported dependency tuple");
        check(!lighton4b_physical_f16_vision_cpu(weights, config, image, 1, true, output), "four threads required");
        check(!lighton4b_physical_f16_vision_cpu(weights, config, image, 4, true, output),
              "invalid weights/config/grid rejected");
        // Model-free minimum-grid graph execution: shared synthetic zero matrices,
        // unit norm scales. No checkpoint or captured output is used.
        std::vector<float> zero(size_t(4096) * 4096, 0.f), ones(4096, 1.f);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 2u << 20, nullptr, true }), ggml_free);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> zb(
            ggml_backend_cpu_buffer_from_ptr(zero.data(), zero.size() * 4), ggml_backend_buffer_free);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> ob(
            ggml_backend_cpu_buffer_from_ptr(ones.data(), ones.size() * 4), ggml_backend_buffer_free);
        check(bool(ctx) && bool(zb) && bool(ob), "synthetic mapped storage");
        auto tensor = [&](int width, int rows, bool scale = false) {
            auto * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
            t->data = scale ? static_cast<void *>(ones.data()) : static_cast<void *>(zero.data());
            t->buffer = scale ? ob.get() : zb.get();
            return t;
        };
        vision_layer_weights layer;
        layer.norm1 = layer.norm2 = tensor(1024, 1, true);
        layer.norm1_bias = layer.norm2_bias = tensor(1024, 1);
        layer.qkv = tensor(1024, 3072); layer.qkv_bias = tensor(3072, 1);
        layer.out = tensor(1024, 1024); layer.out_bias = tensor(1024, 1);
        layer.up = tensor(1024, 4096); layer.up_bias = tensor(4096, 1);
        layer.down = tensor(4096, 1024); layer.down_bias = tensor(1024, 1);
        weights.layers.assign(24, layer);
        weights.patch = tensor(1536, 1024); weights.patch_bias = tensor(1024, 1);
        weights.position = tensor(1024, 2304);
        weights.merger_norm = tensor(1024, 1, true); weights.merger_norm_bias = tensor(1024, 1);
        weights.merger_up = tensor(4096, 4096); weights.merger_up_bias = tensor(4096, 1);
        weights.merger_down = tensor(4096, 2560); weights.merger_down_bias = tensor(2560, 1);
        config.width = 1024; config.heads = 16; config.intermediate = 4096; config.output = 2560;
        image.grid = {1, 2, 2}; image.height = image.width = 32;
        image.resized_rgb.assign(32 * 32 * 3, 0); image.patches.assign(4 * 1536, -1.f);
        check(lighton4b_physical_f16_vision_cpu(weights, config, image, 4, true, output),
              "synthetic minimum-grid production graph executes");
        auto zeros = [](const std::vector<float> & x, size_t count) {
            return x.size() == count && std::all_of(x.begin(), x.end(), [](float f) { return std::isfinite(f) && f == 0.f; });
        };
        check(output.hidden.size() == 24 && zeros(output.patch, 4096) && zeros(output.position, 4096) &&
              zeros(output.embedded, 4096) && zeros(output.merger_norm, 4096) && zeros(output.merged, 2560),
              "synthetic shapes and expected finite zeros");
        for (const auto & hidden : output.hidden) check(zeros(hidden, 4096), "synthetic hidden zeros");
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION", "0");
        check(!lighton4b_physical_f16_vision_cpu(weights, config, image, 4, true, output) && output.merged.empty(),
              "disabled direct component clears synthetic output");
        std::cout << "PHYSICAL_F16_VISION_SYNTHETIC PASS grid=2x2 layers=24 checkpoint_bytes=0\n";
        flag("CRISPEMBED_QWEN35_OCR", "0");
        check(!lighton4b_physical_f16_vision_valid(true), "OCR gate closed");
        std::cout << "PHYSICAL_F16_VISION PASS " << checks << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
