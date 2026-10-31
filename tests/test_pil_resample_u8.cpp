// Hermetic guard for core_pil::resize_u8 plus a Pillow parity mode:
//   test-pil-resample-u8 <in.bin> <w> <h> <channels> <dw> <dh> <lanczos|bicubic> <out.bin>
// tests/test_surya25_tokenizer_parity.py --pil compares <out.bin> with Pillow.
#include "core/pil_resample_u8.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

static int failures = 0;
#define CHECK(c)                                                          \
    do {                                                                  \
        if (!(c)) {                                                       \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

int main(int argc, char ** argv) {
    if (argc == 9) {
        const int w = std::atoi(argv[2]), h = std::atoi(argv[3]), ch = std::atoi(argv[4]);
        const int dw = std::atoi(argv[5]), dh = std::atoi(argv[6]);
        const auto f = std::strcmp(argv[7], "lanczos") == 0 ? core_pil::filter::lanczos : core_pil::filter::bicubic;
        std::vector<uint8_t> in(size_t(w) * h * ch), out;
        std::ifstream src(argv[1], std::ios::binary);
        if (!src.read(reinterpret_cast<char *>(in.data()), std::streamsize(in.size())) ||
            !core_pil::resize_u8(in.data(), w, h, ch, dw, dh, f, out))
            return 2;
        std::ofstream(argv[8], std::ios::binary).write(reinterpret_cast<const char *>(out.data()), std::streamsize(out.size()));
        return 0;
    }
    // Same-size resize is the identity (Pillow skips both passes).
    std::vector<uint8_t> img(7 * 5 * 3), out;
    for (size_t i = 0; i < img.size(); ++i) img[i] = uint8_t(i * 37 + 11);
    CHECK(core_pil::resize_u8(img.data(), 7, 5, 3, 7, 5, core_pil::filter::lanczos, out) && out == img);
    // A constant image stays constant under any resize (coefficients renormalise to 1).
    std::vector<uint8_t> flat(9 * 4 * 3, 200);
    CHECK(core_pil::resize_u8(flat.data(), 9, 4, 3, 31, 2, core_pil::filter::lanczos, out) && out.size() == 31 * 2 * 3);
    bool constant = true;
    for (uint8_t v : out) constant &= v == 200;
    CHECK(constant);
    // Lanczos kernel: 1 at 0, 0 at nonzero integers inside the support, 0 outside.
    CHECK(core_pil::lanczos_filter(0.0) == 1.0);
    CHECK(std::fabs(core_pil::lanczos_filter(1.0)) < 1e-15 && std::fabs(core_pil::lanczos_filter(-2.0)) < 1e-15);
    CHECK(core_pil::lanczos_filter(3.0) == 0.0 && core_pil::lanczos_filter(-3.5) == 0.0);
    // Invalid arguments reject.
    CHECK(!core_pil::resize_u8(nullptr, 1, 1, 3, 1, 1, core_pil::filter::lanczos, out) && out.empty());
    CHECK(!core_pil::resize_u8(img.data(), 7, 5, 3, 0, 5, core_pil::filter::lanczos, out));
    std::printf("test-pil-resample-u8: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
