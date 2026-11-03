// Weight-free Nemotron frontend probe. No model is loaded, no resize performed.
// Reference pixels come from the hash-pinned ORIGINAL Python processor.
#include "nemotron_parse_geometry.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace nemotron_parse_geometry;

void set_gate(const char * value) {
#ifdef _WIN32
    if (_putenv_s(gate, value ? value : "") != 0) throw std::runtime_error("set gate failed");
#else
    if ((value ? setenv(gate, value, 1) : unsetenv(gate)) != 0) throw std::runtime_error("set gate failed");
#endif
}

bool self_controls() {
    image_plan plan;
    bool ok = make_plan(17, 65, plan) && plan.content_height == 17 && plan.content_width == 65 &&
              plan.padded_height == 32 && plan.padded_width == 128 && plan.patch_rows == 2 && plan.patch_columns == 8 &&
              plan.patch_count() == 16 && !plan.resized();
    image_plan empty;
    ok = ok && !make_plan(0, 1, empty) && !make_plan(-1, 10, empty) && !make_plan(2050, 1, empty) &&
         !make_plan(1, 1665, empty) && empty.patch_count() == 0;
    image_plan bounded;
    ok = ok && make_plan(INT_MAX, INT_MAX, bounded) && bounded.content_height == 1664 &&
         bounded.content_width == 1664 && bounded.resized();
    ok = ok && make_plan(4096, 1664, bounded) && bounded.content_height == 2048 && bounded.content_width == 832;

    std::vector<uint8_t> rgb(size_t(17) * 65 * 3);
    for (int y = 0; y < 17; ++y)
        for (int x = 0; x < 65; ++x)
            for (int c = 0; c < 3; ++c) rgb[(size_t(y) * 65 + x) * 3 + c] = uint8_t((y * 31 + x * 7 + c * 53) % 256);
    packed_image packed;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        set_gate(value);
        ok = ok && !pack_resized(rgb.data(), rgb.size(), 17, 65, 3, plan, packed) && packed.patches.empty();
    }
    set_gate("1");
    ok = ok && pack_resized(rgb.data(), rgb.size(), 17, 65, 3, plan, packed) && packed.patches.size() == 16 * 768;
    const float mean[3] = { .48145466f, .4578275f, .40821073f };
    const float stddev[3] = { .26862954f, .26130258f, .27577711f };
    // Distinct channels, rows and columns make CHW/HWC/patch-order errors visible.
    const int positions[][2] = { { 0, 0 },  { 0, 15 },  { 15, 0 },  { 0, 16 },
                                 { 16, 0 }, { 16, 64 }, { 17, 64 }, { 31, 127 } };
    for (const auto & position : positions) {
        const int y = position[0], x = position[1];
        const size_t patch = size_t(y / 16) * 8 + x / 16;
        for (int c = 0; c < 3; ++c) {
            const int pixel = y < 17 && x < 65 ? rgb[(size_t(y) * 65 + x) * 3 + c] : 255;
            const float expected = (float(pixel) / 255.0f - mean[c]) / stddev[c];
            const size_t index = patch * 768 + c * 256 + (y % 16) * 16 + x % 16;
            ok = ok && index < packed.patches.size() && packed.patches[index] == expected;
        }
    }
    ok = ok && !pack_resized(nullptr, rgb.size(), 17, 65, 3, plan, packed) && packed.patches.empty();
    ok = ok && !pack_resized(rgb.data(), rgb.size() - 1, 17, 65, 3, plan, packed);
    ok = ok && !pack_resized(rgb.data(), rgb.size(), 65, 17, 3, plan, packed);
    ok = ok && !pack_resized(rgb.data(), rgb.size(), 17, 65, 4, plan, packed);
    auto wrong = plan;
    wrong.padded_width = 80; // 16-aligned is insufficient; horizontal compression needs 64.
    ok = ok && !pack_resized(rgb.data(), rgb.size(), 17, 65, 3, wrong, packed);
    std::vector<uint8_t> gray(17 * 65, 27);
    ok = ok && pack_resized(gray.data(), gray.size(), 17, 65, 1, plan, packed);
    for (int c = 0; c < 3; ++c) ok = ok && packed.patches[c * 256] == (27.0f / 255.0f - mean[c]) / stddev[c];
    return ok;
}

int integer(const char * text) {
    size_t used = 0;
    const long long value = std::stoll(text, &used);
    if (used != std::strlen(text) || value < INT_MIN || value > INT_MAX) throw std::invalid_argument("integer");
    return int(value);
}

void plan_json(const image_plan & plan, size_t elements) {
    std::cout << "{\"ok\":true,\"content_sizes\":[" << plan.content_height << ',' << plan.content_width
              << "],\"image_sizes\":[" << plan.padded_height << ',' << plan.padded_width << "],\"patch_grid\":["
              << plan.patch_rows << ',' << plan.patch_columns << "],\"patch_count\":" << plan.patch_count()
              << ",\"resized\":" << (plan.resized() ? "true" : "false") << ",\"elements\":" << elements
              << ",\"resize_implemented\":false,\"model_execution\":false}\n";
}

void write_f32_le(const std::string & path, const std::vector<float> & values) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("output open failed");
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) == 1) {
        file.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * sizeof(float)));
    } else {
        for (float value : values) {
            uint32_t bits;
            std::memcpy(&bits, &value, 4);
            const char bytes[4] = { char(bits), char(bits >> 8), char(bits >> 16), char(bits >> 24) };
            file.write(bytes, 4);
        }
    }
    if (!file) throw std::runtime_error("output write failed");
}

int execute(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "self") {
        const bool ok = self_controls();
        std::cout << "{\"self_controls_pass\":" << (ok ? "true" : "false") << ",\"model_execution\":false}\n";
        return ok ? 0 : 1;
    }
    if (argc != 4 && argc != 7) throw std::invalid_argument("arguments");
    const std::string mode = argv[1];
    if ((argc == 4 && mode != "plan") || (argc == 7 && mode != "pack")) throw std::invalid_argument("mode");
    if (!core_env::on(gate)) {
        std::cout << "{\"ok\":false,\"error\":\"frontend gate is off\"}\n";
        return 1;
    }
    image_plan plan;
    if (!make_plan(integer(argv[2]), integer(argv[3]), plan)) {
        std::cout << "{\"ok\":false,\"error\":\"unsupported image dimensions\"}\n";
        return 1;
    }
    if (mode == "plan") {
        plan_json(plan, 0);
        return 0;
    }
    const int channels = integer(argv[4]);
    if (channels != 1 && channels != 3) throw std::invalid_argument("channels");
    const size_t count = size_t(plan.content_height) * plan.content_width * channels;
    std::ifstream file(argv[5], std::ios::binary);
    if (!file) throw std::runtime_error("input open failed");
    std::vector<uint8_t> pixels(count);
    file.read(reinterpret_cast<char *>(pixels.data()), std::streamsize(count));
    if (file.gcount() != std::streamsize(count) || file.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("input byte count");
    packed_image packed;
    if (!pack_resized(pixels.data(), pixels.size(), plan.content_height, plan.content_width, channels, plan, packed))
        throw std::runtime_error("pack failed");
    write_f32_le(argv[6], packed.patches);
    plan_json(plan, packed.patches.size());
    return 0;
}
} // namespace

int checked_execute(int argc, char ** argv) {
    try {
        return execute(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "Nemotron geometry probe: " << error.what() << '\n';
        std::cout << "{\"ok\":false,\"error\":\"invalid or unsupported frontend input\"}\n";
        return 2;
    }
}

int main(int argc, char ** argv) {
    core_util::clean_exit(checked_execute(argc, argv));
}
