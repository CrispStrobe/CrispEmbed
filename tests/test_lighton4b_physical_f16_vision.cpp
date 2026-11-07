#include "lighton4b_physical_f16_vision.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
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
        flag("CRISPEMBED_QWEN35_OCR", "0");
        check(!lighton4b_physical_f16_vision_valid(true), "OCR gate closed");
        std::cout << "PHYSICAL_F16_VISION PASS " << checks << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
