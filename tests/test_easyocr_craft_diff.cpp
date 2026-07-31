#include "easyocr_craft.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"

#include <cstdio>

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
    easyocr_craft_free(ctx);
    core_util::clean_exit(rc ? 6 : 0);
    return rc ? 6 : 0;
}
