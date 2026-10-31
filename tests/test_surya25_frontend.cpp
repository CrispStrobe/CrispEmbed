// Surya OCR 2.5 format-4 frontend through real GGUF metadata (no weights needed):
//   test-surya25-frontend <metadata.gguf> < prompts.txt
// Each stdin line is a prompt ("-" = empty, i.e. the default block prompt).
// Prints "IDS <ids>" per prompt for the rendered one-image chat prompt (a single
// image token), after checking that the gates are honoured.
//   test-surya25-frontend <metadata.gguf> --image <rgb.bin> <width> <height> <patches.out>
// Runs the format-4 preprocessing (vLLM min/max pixels) and writes
// "height width t h w" on stdout plus float32 patches to <patches.out>.
//   test-surya25-frontend <metadata.gguf> --scale <rgb.bin> <width> <height> <scaled.out>
// Runs the package scale_to_fit port and prints "height width changed".
// Driven by tests/test_surya25_tokenizer_parity.py --frontend.
#include "qwen35_frontend.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

static int fail(const char * what) {
    std::printf("FAIL %s\n", what);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 7) return fail("usage");
    std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(argv[1]),
                                                                           core_gguf::free_metadata);
    if (!meta) return fail("metadata");
    qwen35_ocr::frontend_tokenizer t;
    setenv("CRISPEMBED_QWEN35_OCR", "1", 1);
    setenv("CRISPEMBED_SURYA25_FRONTEND", "0", 1); // "0" must mean OFF
    if (t.load(meta.get())) return fail("loaded with frontend gate =0");
    setenv("CRISPEMBED_SURYA25_FRONTEND", "1", 1);
    if (!t.load(meta.get()) || !t.is_surya25() || t.vocab_size() != 88526 || t.eos() != 2 || !t.is_stop(2) ||
        t.is_stop(0) || t.image_limits().min_pixels != 3136 || t.image_limits().max_pixels != 6291456)
        return fail("load");
    if (argc == 7 && std::string(argv[2]) == "--scale") {
        const int w = std::atoi(argv[4]), h = std::atoi(argv[5]);
        std::ifstream in(argv[3], std::ios::binary);
        std::vector<uint8_t> rgb(size_t(w) * h * 3), scaled;
        if (w < 1 || h < 1 || !in.read(reinterpret_cast<char *>(rgb.data()), std::streamsize(rgb.size())))
            return fail("image input");
        int sh = 0, sw = 0;
        if (!qwen35_ocr::surya_scale_to_fit(rgb.data(), h, w, scaled, sh, sw)) return fail("scale");
        const auto & bytes = scaled.empty() ? rgb : scaled;
        std::ofstream(argv[6], std::ios::binary)
            .write(reinterpret_cast<const char *>(bytes.data()), std::streamsize(bytes.size()));
        std::printf("%d %d %d\n", sh, sw, int(!scaled.empty()));
        return 0;
    }
    if (argc == 7) {
        const int w = std::atoi(argv[4]), h = std::atoi(argv[5]);
        std::ifstream in(argv[3], std::ios::binary);
        std::vector<uint8_t> rgb(size_t(w) * h * 3);
        if (w < 1 || h < 1 || !in.read(reinterpret_cast<char *>(rgb.data()), std::streamsize(rgb.size())))
            return fail("image input");
        qwen35_ocr::image_inputs image;
        if (!qwen35_ocr::preprocess_rgb(rgb.data(), h, w, t.image_limits(), image)) {
            std::printf("REJECT\n");
            return 0;
        }
        std::ofstream out(argv[6], std::ios::binary);
        out.write(reinterpret_cast<const char *>(image.patches.data()),
                  std::streamsize(image.patches.size() * sizeof(float)));
        std::printf("%d %d %d %d %d\n", image.height, image.width, image.grid[0], image.grid[1], image.grid[2]);
        return 0;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<int> ids, modalities;
        std::string rendered;
        if (!t.image_prompt(line == "-" ? "" : line, 1, ids, modalities, rendered)) return fail("image_prompt");
        std::printf("IDS");
        for (int id : ids) std::printf(" %d", id);
        std::printf("\n");
    }
    // Decode skips specials (<|im_end|>=2, <unk>=65424) and fuses the rest.
    std::vector<int> ids;
    std::string text;
    if (!t.encode("Grüße <b>x</b>", ids) || !t.decode(ids, text) || text != "Grüße <b>x</b>") return fail("roundtrip");
    ids.push_back(2);
    ids.push_back(65424);
    if (!t.decode(ids, text) || text != "Grüße <b>x</b>") return fail("special skip");
    unsetenv("CRISPEMBED_SURYA25_FRONTEND");
    if (t.encode("x", ids)) return fail("encode with gate unset");
    std::printf("PASS\n");
    return 0;
}
