#include "easyocr_craft.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <craft.gguf> <reference.gguf>\n", argv[0]);
        return 2;
    }
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 3;
    auto input = ref.get_f32("input_image");
    auto shape = ref.shape("input_image");
    if (!input.first || shape.size() != 4) return 4;
    // Ref stores GGML dimensions fastest-first: [W,H,C,B].
    easyocr_craft_context * ctx = easyocr_craft_init(argv[1], (int)shape[0], (int)shape[1]);
    if (!ctx || !easyocr_craft_forward(ctx, input.first, input.second)) {
        easyocr_craft_free(ctx);
        return 5;
    }
    int rc = easyocr_craft_diff(ctx, argv[2]);
    const int native_boxes = easyocr_craft_box_count(ctx, 0.7f, 0.4f, 0.4f);
    const std::string expected_text = ref.meta("easyocr.decoded");
    const int expected_boxes = expected_text.empty() ? -1 : std::atoi(expected_text.c_str());
    printf("easyocr-craft-decoded boxes=%d expected=%d %s\n", native_boxes, expected_boxes,
           native_boxes == expected_boxes ? "PASS" : "FAIL");
    if (native_boxes != expected_boxes) rc = 1;
    easyocr_craft_free(ctx);
    core_util::clean_exit(rc ? 6 : 0);
    return rc ? 6 : 0;
}
