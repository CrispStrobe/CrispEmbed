// mineru_table_image.h - native table-image masking/restoration for the MinerU
// document route. Translation of mineru-vl-utils post_process/
// table_image_processor.py (c9797cde, Apache-2.0); changed: C++, with these
// building blocks that reproduce the Python dependencies bit-exactly:
//   * python_random: CPython `random` (MT19937, seed(int), random(), choices)
//   * encode_jpeg_pillow: Pillow 12 `Image.save(format="JPEG")` defaults on
//     libjpeg-turbo (baseline, quality 75, 4:2:0, islow DCT, standard Huffman)
//   * Pillow crop/rotate(90/180/270, expand)/rectangle/getpixel semantics
// Token *text* rendering cannot be Pillow/FreeType pixel-identical without
// FreeType and the host font file; draw_token_text renders an embedded DejaVu
// Sans glyph atlas with the upstream size search. See docs/mineru25-pro.md.
#pragma once

#include "mineru_postprocess.h"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace mineru_img {

struct rgb_image {
    int width = 0, height = 0;
    std::vector<uint8_t> px; // RGB, row-major
    uint8_t * at(int x, int y) { return px.data() + (static_cast<size_t>(y) * width + x) * 3; }
    const uint8_t * at(int x, int y) const { return px.data() + (static_cast<size_t>(y) * width + x) * 3; }
};

// CPython random.Random.
class python_random {
public:
    python_random() { seed_os(); }
    explicit python_random(uint64_t seed) { seed_int(seed); }
    void seed_int(uint64_t value);
    void seed_os();
    uint32_t genrand();
    double random();

private:
    std::array<uint32_t, 624> mt_{};
    int index_ = 625;
    void init_genrand(uint32_t s);
};

// Pillow Image.crop with an integer box (zero fill outside the source).
rgb_image crop(const rgb_image & src, int x0, int y0, int x1, int y1);
// Pillow Image.rotate(angle, expand=True) for angle in {0, 90, 180, 270} (CCW).
rgb_image rotate(const rgb_image & src, int angle);
// Pillow ImageDraw.rectangle(box, fill) (inclusive corners, clipped).
void fill_rectangle(rgb_image & img, int x0, int y0, int x1, int y1, const std::array<uint8_t, 3> & color);

std::vector<uint8_t> encode_jpeg_pillow(const rgb_image & img);
std::string base64(const std::vector<uint8_t> & bytes);

// Approximate Pillow ImageDraw.text with the upstream optimal-font search.
// Returns false when the token does not fit (upstream then draws nothing).
bool draw_token_text(rgb_image & img, int box_x0, int box_y0, int box_w, int box_h, const std::string & token,
                     const std::array<uint8_t, 3> & color);

struct table_entry {
    size_t index;
    const mineru_pp::block * image;
};

// build_table_image_map: table index -> image indices (sorted by y, x).
std::vector<std::pair<size_t, std::vector<size_t>>> build_table_image_map(const mineru_pp::block_list & blocks,
                                                                          const std::vector<size_t> & table_indices,
                                                                          double threshold = 0.9);

// mask_and_encode_table_image: returns the masked/rotated table crop and the
// token -> data-URI map (insertion order).
rgb_image mask_and_encode_table_image(const rgb_image & page, const mineru_pp::block & table,
                                      const std::vector<const mineru_pp::block *> & images,
                                      const rgb_image & table_crop, python_random & rng,
                                      std::vector<std::pair<mineru_pp::ustring, mineru_pp::ustring>> & token_map);

} // namespace mineru_img
