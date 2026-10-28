// Test-only raw input boundary. No model, resize, normalization or backend.
#include "core/clean_exit.h"
#include "core/json.h"
#include "stb_image.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

static int body(int argc, char ** argv) {
    if (argc != 4) return 2;
    std::ofstream report(argv[3]);
    if (!report) return 2;
    try {
        int width = 0, height = 0, channels = 0;
        std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(stbi_load(argv[1], &width, &height, &channels, 3),
                                                                 stbi_image_free);
        if (!rgb) throw std::runtime_error(stbi_failure_reason() ? stbi_failure_reason() : "STB decode failed");
        if (width < 1 || height < 1 || uint64_t(width) * uint64_t(height) > 25000000)
            throw std::runtime_error("decoded image exceeds audit bound");
        const size_t count = size_t(width) * size_t(height) * 3;
        std::ofstream output(argv[2], std::ios::binary);
        output.write(reinterpret_cast<const char *>(rgb.get()), std::streamsize(count));
        output.close();
        if (!output) throw std::runtime_error("raw RGB write failed");
        report << "{\"completed\":true,\"width\":" << width << ",\"height\":" << height
               << ",\"file_channels\":" << channels << ",\"channels\":3,\"bytes\":" << count << "}\n";
        return report ? 0 : 2;
    } catch (const std::exception & error) {
        report << "{\"completed\":false,\"error\":\"" << core_json::json_escape_strict(error.what()) << "\"}\n";
        return 1;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
