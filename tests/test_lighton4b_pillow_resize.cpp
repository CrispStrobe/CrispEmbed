#include "lighton4b_image_compose.h"
#include "core/clean_exit.h"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
using namespace qwen35_ocr;
static int checks = 0;
static void check(bool ok, const char * why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
static void flag(const char * k, const char * v) {
    setenv(k, v, 1);
}
static bool same(const image_inputs & a, const image_inputs & b) {
    return a.resized_rgb == b.resized_rgb && a.patches == b.patches && a.grid == b.grid && a.height == b.height &&
           a.width == b.width;
}
static std::vector<uint8_t> input(const char * path, int h, int w) {
    check(h > 0 && w > 0 && int64_t(h) * w <= 16777216, "bounded RGB");
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> rgb(size_t(h) * w * 3);
    f.read(reinterpret_cast<char *>(rgb.data()), rgb.size());
    check(size_t(f.gcount()) == rgb.size() && f.peek() == EOF, "exact RGB file");
    return rgb;
}
static void write(const std::string & p, const std::vector<uint8_t> & v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size());
    check(bool(f), "RGB write");
}
static void floats(const std::string & p, const std::vector<float> & v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
    check(bool(f), "F32 write");
}
static int crispembed_test_main(int argc, char ** argv) {
    try {
        if (argc == 7) {
            int h = std::stoi(argv[2]), w = std::stoi(argv[3]), oh = std::stoi(argv[4]), ow = std::stoi(argv[5]);
            auto rgb = input(argv[1], h, w);
            std::vector<uint8_t> out;
            check(lighton4b_pillow_resize_rgb(rgb.data(), h, w, oh, ow, out), "raw Pillow resize");
            write(std::string(argv[6]) + ".rgb", out);
            std::cout << "PILLOW_RAW PASS\n";
            return 0;
        }
        flag("CRISPEMBED_QWEN35_OCR", "1");
        flag("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE", "1");
        if (argc == 6 && std::string(argv[1]) == "--preprocess") {
            int h = std::stoi(argv[3]), w = std::stoi(argv[4]);
            auto rgb = input(argv[2], h, w);
            image_inputs result, legacy;
            flag("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "0");
            check(lighton4b_preprocess_rgb(rgb.data(), h, w, legacy, true), "legacy actual");
            flag("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "1");
            check(lighton4b_preprocess_rgb(rgb.data(), h, w, result, true), "physical actual");
            std::string prefix = argv[5];
            write(prefix + ".rgb", result.resized_rgb);
            floats(prefix + ".patches.f32", result.patches);
            write(prefix + ".legacy.rgb", legacy.resized_rgb);
            floats(prefix + ".legacy.patches.f32", legacy.patches);
            std::ofstream m(prefix + ".json");
            m << "{\"height\":" << result.height << ",\"width\":" << result.width
              << ",\"physical_owner\":true,\"gate\":true}\n";
            check(bool(m), "metadata");
            std::cout << "PILLOW_PREPROCESS PASS\n";
            return 0;
        }
        check(argc == 1, "closed unit/resize/preprocess CLI");
        std::vector<uint8_t> rgb(256 * 256 * 3), out;
        for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 37) % 256);
        check(lighton4b_pillow_resize_rgb(rgb.data(), 256, 256, 256, 256, out) && out == rgb, "identity");
        std::vector<uint8_t> solid(17 * 19 * 3, 171);
        for (auto size : { std::pair<int, int>{ 32, 64 }, { 7, 5 } }) {
            check(lighton4b_pillow_resize_rgb(solid.data(), 17, 19, size.first, size.second, out), "constant resize");
            check(std::all_of(out.begin(), out.end(), [](uint8_t x) { return x == 171; }), "constant preserved");
        }
        check(!lighton4b_pillow_resize_rgb(nullptr, 10, 10, 20, 20, out) && out.empty(), "null clears");
        check(!lighton4b_pillow_resize_rgb(rgb.data(), 0, 10, 20, 20, out) && out.empty(), "bad shape clears");
        check(!lighton4b_pillow_resize_rgb(rgb.data(), 32768, 1, 1, 32768, out) && out.empty(),
              "cross-aspect horizontal allocation bounded before input read");
        image_inputs legacy, physical, nonphysical;
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "0");
        check(lighton4b_preprocess_rgb(rgb.data(), 256, 256, legacy), "legacy default");
        check(lighton4b_preprocess_rgb(rgb.data(), 256, 256, physical, true) && same(physical, legacy),
              "gate-off physical identical");
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE", "1");
        check(lighton4b_preprocess_rgb(rgb.data(), 256, 256, nonphysical) && same(nonphysical, legacy),
              "nonphysical inert");
        check(!lighton4b_pillow_preprocess_rgb(rgb.data(), 256, 256, false, physical) && physical.patches.empty(),
              "helper requires owner");
        check(lighton4b_preprocess_rgb(rgb.data(), 256, 256, physical, true) && physical.resized_rgb == rgb &&
                  physical.grid == legacy.grid,
              "physical identity RGB/grid");
        check(physical.patches != legacy.patches, "source F32 normalization distinct");
        check(!lighton4b_preprocess_rgb(nullptr, 256, 256, physical, true) && physical.patches.empty(),
              "bad physical input clears");
        flag("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE", "0");
        check(!lighton4b_preprocess_rgb(rgb.data(), 256, 256, physical, true), "image gate closed");
        std::cout << "PHYSICAL_PILLOW_RESIZE PASS " << checks << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
