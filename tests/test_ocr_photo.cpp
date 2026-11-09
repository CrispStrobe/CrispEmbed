// Model-free geometry controls and PPM driver for the phone-photo OCR panel.
// Compile the production crop/cleanup sources; no model or inference stub.
#include "ocr_crop.h"
#include "scan_cleanup.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

static void controls() {
    constexpr int w = 720, h = 600;
    std::vector<uint8_t> image(size_t(w) * h * 3, 255);
    // Independent known camera map: x=100+500u/(1+.65v),
    // y=80+700v/(1+.65v). Red/green encode original paper coordinates.
    for (int y = 0; y < h; ++y) {
        const double t = y - 80.0, v = t / (700.0 - .65 * t);
        for (int x = 0; x < w; ++x) {
            const double u = (x - 100.0) * (1.0 + .65 * v) / 500.0;
            image[(size_t(y) * w + x) * 3] = uint8_t(std::lround(255 * std::clamp(u, 0.0, 1.0)));
            image[(size_t(y) * w + x) * 3 + 1] = uint8_t(std::lround(255 * std::clamp(v, 0.0, 1.0)));
            image[(size_t(y) * w + x) * 3 + 2] = 73;
        }
    }
    float qx[4] = { 100, 600, float(100 + 500 / 1.65), 100 };
    float qy[4] = { 80, 80, float(80 + 700 / 1.65), float(80 + 700 / 1.65) };
    int ow = 0, oh = 0;
    gate("CRISPEMBED_OCR_CROP_PROJECTIVE", nullptr);
    const auto legacy = ocr_crop::extract_quad(image.data(), w, h, 3, qx, qy, 0, &ow, &oh);
    for (const auto * off : { "0", "false", "OFF", "no", "" }) {
        gate("CRISPEMBED_OCR_CROP_PROJECTIVE", off);
        require(ocr_crop::extract_quad(image.data(), w, h, 3, qx, qy, 0, &ow, &oh) == legacy,
                "all disabled projective gates must preserve legacy bytes");
    }
    gate("CRISPEMBED_OCR_CROP_PROJECTIVE", "1");
    const auto rectified = ocr_crop::extract_quad(image.data(), w, h, 3, qx, qy, 0, &ow, &oh);
    require(!rectified.empty() && rectified.size() == legacy.size(), "projective output geometry");
    double projective_error = 0, legacy_error = 0;
    int samples = 0;
    for (int y = 10; y < oh - 10; ++y)
        for (int x = 10; x < ow - 10; ++x) {
            const size_t i = (size_t(y) * ow + x) * 3;
            const double red = 255.0 * x / (ow - 1), green = 255.0 * y / (oh - 1);
            projective_error += std::abs(rectified[i] - red) + std::abs(rectified[i + 1] - green);
            legacy_error += std::abs(legacy[i] - red) + std::abs(legacy[i + 1] - green);
            require(rectified[i + 2] == 73, "RGB channels must not bleed");
            samples += 2;
        }
    projective_error /= samples;
    legacy_error /= samples;
    require(projective_error < .5, "projective crop must recover the independent paper coordinate ramp");
    require(legacy_error > 7, "fixture must distinguish a projective map from a bilinear corner warp");
    // Every permutation must retain the same distinct corners and pixels.
    int permutation[4] = { 0, 1, 2, 3 };
    int permutations = 0;
    do {
        float px[4], py[4];
        for (int i = 0; i < 4; ++i) {
            px[i] = qx[permutation[i]];
            py[i] = qy[permutation[i]];
        }
        require(ocr_crop::extract_quad(image.data(), w, h, 3, px, py, 0, &ow, &oh) == rectified,
                "quad order changes rectified pixels");
        ++permutations;
    } while (std::next_permutation(permutation, permutation + 4));
    float dx[4] = { 150, 250, 150, 50 }, dy[4] = { 50, 150, 250, 150 };
    require(!ocr_crop::extract_quad(image.data(), w, h, 3, dx, dy, 0, &ow, &oh).empty(),
            "diamond corners must remain distinct");
    dx[0] = dx[1];
    dy[0] = dy[1];
    require(ocr_crop::extract_quad(image.data(), w, h, 3, dx, dy, 0, &ow, &oh).empty() && ow == 0 && oh == 0,
            "duplicate quad must fail without an output");
    dx[0] = std::numeric_limits<float>::quiet_NaN();
    require(ocr_crop::extract_quad(image.data(), w, h, 3, dx, dy, 0, &ow, &oh).empty(), "NaN quad rejected");
    require(ocr_crop::extract_quad(image.data(), w, h, 3, qx, qy, 100000, &ow, &oh).empty(), "padding bounded");
    require(ocr_crop::extract_quad(image.data(), w, h, 2, qx, qy, 0, &ow, &oh).empty(), "channels validated");
    // Identity quad, RGBA and grayscale use the actual image edges, without
    // dropping the last row/column. Distances use the historical output size.
    const float ix[4] = { 0, 3, 3, 0 }, iy[4] = { 0, 0, 3, 3 };
    for (int channels : { 1, 4 }) {
        std::vector<uint8_t> rgba(4 * 4 * channels, 170);
        rgba.back() = 91;
        const auto out = ocr_crop::extract_quad(rgba.data(), 4, 4, channels, ix, iy, 0, &ow, &oh);
        require(ow == 3 && oh == 3 && out.back() == 91, "projective crop preserves final edge and channels");
    }
    gate("CRISPEMBED_OCR_CROP_PROJECTIVE", nullptr);

    std::vector<float> blank(1600 * 1200, 1.0f);
    gate("CRISPEMBED_SCAN_BOUNDED_DESKEW", "1");
    require(scan_cleanup_detect_angle_consensus(blank.data(), 1600, 1200, 15) == 0, "blank phone image unrotated");
    require(scan_cleanup_detect_angle(nullptr, 1600, 1200, 15) == 0, "null analysis rejected");
    require(scan_cleanup_detect_angle(blank.data(), 0, 1200, 15) == 0, "empty analysis rejected");
    require(scan_cleanup_detect_angle(blank.data(), 1600, 1200, std::numeric_limits<float>::infinity()) == 0,
            "nonfinite angle rejected");
    require(scan_cleanup_detect_angle_consensus(blank.data(), 1600, 1200, -1) == 0, "negative angle rejected");
    uint8_t * output = reinterpret_cast<uint8_t *>(1);
    require(scan_cleanup_deskew_rgb(image.data(), 100000, h, 3, 15, &output, &ow, &oh) == -1 && output == nullptr &&
                ow == 0 && oh == 0,
            "invalid RGB geometry clears every output");
    gate("CRISPEMBED_SCAN_BOUNDED_DESKEW", nullptr);
    std::cout << "{\"projective_mean_absolute_error\":" << projective_error
              << ",\"legacy_mean_absolute_error\":" << legacy_error << ",\"corner_permutations\":" << permutations
              << ",\"controls\":\"passed\"}\n";
}

struct ppm {
    int width = 0, height = 0;
    std::vector<uint8_t> pixels;
};

static ppm read_ppm(const char * path) {
    std::ifstream file(path, std::ios::binary);
    std::string magic;
    int maxval;
    ppm image;
    file >> magic >> image.width >> image.height >> maxval;
    require(file.good() && magic == "P6" && maxval == 255 && image.width > 0 && image.height > 0 &&
                size_t(image.width) * image.height <= 16u * 1024u * 1024u,
            "invalid P6 fixture");
    file.get();
    image.pixels.resize(size_t(image.width) * image.height * 3);
    file.read(reinterpret_cast<char *>(image.pixels.data()), std::streamsize(image.pixels.size()));
    require(file.good(), "truncated P6 fixture");
    return image;
}

static void write_ppm(const char * path, const uint8_t * pixels, int width, int height) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(pixels), std::streamsize(size_t(width) * height * 3));
    require(file.good(), "cannot write P6 result");
}

int main(int argc, char ** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "self") {
            controls();
            return 0;
        }
        require(argc >= 4, "usage: test-ocr-photo deskew|quad input.ppm output.ppm [corners]");
        auto image = read_ppm(argv[2]);
        if (std::string(argv[1]) == "whiten") {
            std::vector<float> gray(size_t(image.width) * image.height), whitened(gray.size());
            for (size_t i = 0; i < gray.size(); ++i)
                gray[i] =
                    (image.pixels[i * 3] * .299f + image.pixels[i * 3 + 1] * .587f + image.pixels[i * 3 + 2] * .114f) /
                    255;
            scan_cleanup_whiten(gray.data(), image.width, image.height, 51, whitened.data());
            for (size_t i = 0; i < gray.size(); ++i)
                for (int c = 0; c < 3; ++c)
                    image.pixels[i * 3 + c] = uint8_t(std::clamp(whitened[i] * 255.0f + .5f, 0.0f, 255.0f));
            write_ppm(argv[3], image.pixels.data(), image.width, image.height);
            std::cout << "{\"width\":" << image.width << ",\"height\":" << image.height << "}\n";
        } else if (std::string(argv[1]) == "deskew") {
            std::vector<float> gray(size_t(image.width) * image.height);
            for (size_t i = 0; i < gray.size(); ++i)
                gray[i] =
                    (image.pixels[i * 3] * .299f + image.pixels[i * 3 + 1] * .587f + image.pixels[i * 3 + 2] * .114f) /
                    255;
            const float angle = scan_cleanup_detect_angle_consensus(gray.data(), image.width, image.height, 15);
            uint8_t * out = nullptr;
            int width = 0, height = 0;
            require(scan_cleanup_deskew_rgb(image.pixels.data(), image.width, image.height, 3, 15, &out, &width,
                                            &height) == 0,
                    "native RGB deskew failed");
            write_ppm(argv[3], out ? out : image.pixels.data(), width, height);
            scan_cleanup_free_image(out);
            std::cout << "{\"angle\":" << angle << ",\"width\":" << width << ",\"height\":" << height << "}\n";
        } else {
            require(std::string(argv[1]) == "quad" && argc == 12, "quad requires four x,y corners");
            float x[4], y[4];
            for (int i = 0; i < 4; ++i) {
                x[i] = std::stof(argv[4 + 2 * i]);
                y[i] = std::stof(argv[5 + 2 * i]);
            }
            int width = 0, height = 0;
            auto out =
                ocr_crop::extract_quad(image.pixels.data(), image.width, image.height, 3, x, y, 0, &width, &height);
            require(!out.empty(), "native projective crop failed");
            write_ppm(argv[3], out.data(), width, height);
            std::cout << "{\"width\":" << width << ",\"height\":" << height << "}\n";
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
