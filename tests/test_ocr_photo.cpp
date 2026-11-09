// Model-free geometry controls and PPM driver for the phone-photo OCR panel.
// Compile the production crop/cleanup sources; no model or inference stub.
#include "ocr_crop.h"
#include "scan_cleanup.h"
#include "crispembed.h"

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

static void public_controls() {
    const float identity[8] = { 0, 0, 6, 0, 6, 4, 0, 4 };
    for (int channels : { 1, 3, 4 }) {
        std::vector<uint8_t> input(7 * 5 * channels);
        for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i % 251);
        uint8_t * output = nullptr;
        int ow = 0, oh = 0;
        require(crispembed_scan_photo_rotate(input.data(), input.size(), 7, 5, channels, 0, &output, &ow, &oh) == 0,
                "public zero rotation succeeds");
        require(ow == 7 && oh == 5 && std::equal(input.begin(), input.end(), output), "zero rotation is byte-exact");
        crispembed_scan_photo_free_image(output);
        for (const auto * state : { "0", "1" }) {
            gate("CRISPEMBED_OCR_CROP_PROJECTIVE", state);
            require(crispembed_scan_photo_perspective(input.data(), input.size(), 7, 5, channels, identity, &output,
                                                      &ow, &oh) == 0,
                    "public full-frame perspective succeeds regardless of legacy gate");
            require(ow == 7 && oh == 5 && std::equal(input.begin(), input.end(), output),
                    "full-frame perspective retains all pixels and channels");
            crispembed_scan_photo_free_image(output);
        }
        output = reinterpret_cast<uint8_t *>(1);
        require(crispembed_scan_photo_rotate(input.data(), input.size() - 1, 7, 5, channels, 2, &output, &ow, &oh) ==
                        -1 &&
                    !output && ow == 0 && oh == 0,
                "public rotation rejects short buffer and clears outputs");
        require(crispembed_scan_photo_perspective(input.data(), input.size() + 1, 7, 5, channels, identity, &output,
                                                  &ow, &oh) == -1 &&
                    !output,
                "public perspective rejects excess byte count");
        for (float bad_angle : { 21.0f, -21.0f, std::numeric_limits<float>::quiet_NaN() })
            require(crispembed_scan_photo_rotate(input.data(), input.size(), 7, 5, channels, bad_angle, &output, &ow,
                                                 &oh) == -1 &&
                        !output && ow == 0 && oh == 0,
                    "public rotation rejects invalid angle");
        for (int kind = 0; kind < 4; ++kind) {
            float bad_quad[8];
            std::copy(identity, identity + 8, bad_quad);
            if (kind == 0) std::swap(bad_quad[2], bad_quad[4]), std::swap(bad_quad[3], bad_quad[5]);
            if (kind == 1) bad_quad[0] = -1;
            if (kind == 2) bad_quad[4] = 7;
            if (kind == 3) bad_quad[0] = std::numeric_limits<float>::infinity();
            require(crispembed_scan_photo_perspective(input.data(), input.size(), 7, 5, channels, bad_quad, &output,
                                                      &ow, &oh) == -1 &&
                        !output && ow == 0 && oh == 0,
                    "public perspective rejects invalid labelled corners");
        }
    }
    gate("CRISPEMBED_OCR_CROP_PROJECTIVE", nullptr);
    std::vector<uint8_t> blank(90 * 120 * 3, 255);
    float correction = 12;
    // The legacy minimum-peak threshold truncates to zero below width 100.
    // Exercise both sides and the smallest accepted width, for white, gray
    // and dark uniform captures. These contain no evidence of an angle.
    for (int width : { 3, 90, 99, 100, 101, 160 }) {
        for (uint8_t value : { 0, 127, 255 }) {
            std::vector<uint8_t> uniform(size_t(width) * 120 * 3, value);
            correction = 12;
            require(crispembed_scan_photo_angle(uniform.data(), uniform.size(), width, 120, 3, 15, &correction) == 0 &&
                        correction == 0,
                    "uniform photo must not select a search-boundary angle");
        }
    }
    require(crispembed_scan_photo_angle(blank.data(), 7, 90, 120, 3, 15, &correction) == -1 && correction == 0,
            "angle API rejects short buffer");
    require(crispembed_scan_photo_angle(blank.data(), blank.size(), 90, 120, 3, 0, &correction) == -1,
            "angle API rejects invalid range");
    require(crispembed_scan_photo_angle(blank.data(), blank.size(), 90, 120, 3, 15, nullptr) == -1,
            "angle API rejects absent output");
    // A dark marker to the right of centre moves down for positive (clockwise)
    // rotation and up for negative. This checks the sign independently of skew.
    std::vector<uint8_t> marker(81 * 61, 255);
    for (int y = 28; y <= 32; ++y)
        for (int x = 58; x <= 62; ++x) marker[size_t(y) * 81 + x] = 0;
    for (float angle : { -10.0f, 10.0f }) {
        uint8_t * output = nullptr;
        int ow = 0, oh = 0;
        require(crispembed_scan_photo_rotate(marker.data(), marker.size(), 81, 61, 1, angle, &output, &ow, &oh) == 0,
                "public manual rotation succeeds");
        require(ow > 81 && oh > 61 && output[0] == 255, "rotation expands with white fill");
        double weight = 0, centre_y = 0;
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                const double ink = 255 - output[size_t(y) * ow + x];
                weight += ink;
                centre_y += y * ink;
            }
        crispembed_scan_photo_free_image(output);
        require(weight > 0 && (centre_y / weight - (oh - 1) * .5) * angle > 25,
                "manual rotation clockwise sign is correct");
    }
    crispembed_scan_photo_free_image(nullptr);
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
            public_controls();
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
        } else if (std::string(argv[1]) == "photo-deskew") {
            float correction = 0;
            require(crispembed_scan_photo_angle(image.pixels.data(), image.pixels.size(), image.width, image.height, 3,
                                                15, &correction) == 0,
                    "public angle failed");
            uint8_t * out = nullptr;
            int width = 0, height = 0;
            require(crispembed_scan_photo_rotate(image.pixels.data(), image.pixels.size(), image.width, image.height, 3,
                                                 correction, &out, &width, &height) == 0,
                    "public rotation failed");
            write_ppm(argv[3], out, width, height);
            crispembed_scan_photo_free_image(out);
            std::cout << "{\"angle\":" << -correction << ",\"correction_degrees\":" << correction
                      << ",\"width\":" << width << ",\"height\":" << height << "}\n";
        } else if (std::string(argv[1]) == "photo-quad") {
            require(argc == 12, "photo-quad requires four x,y corners");
            float corners[8];
            for (int i = 0; i < 8; ++i) corners[i] = std::stof(argv[4 + i]);
            uint8_t * out = nullptr;
            int width = 0, height = 0;
            require(crispembed_scan_photo_perspective(image.pixels.data(), image.pixels.size(), image.width,
                                                      image.height, 3, corners, &out, &width, &height) == 0,
                    "public perspective failed");
            write_ppm(argv[3], out, width, height);
            crispembed_scan_photo_free_image(out);
            std::cout << "{\"width\":" << width << ",\"height\":" << height << "}\n";
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
