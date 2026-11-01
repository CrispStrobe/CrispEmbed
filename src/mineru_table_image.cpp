// mineru_table_image.cpp - see mineru_table_image.h. Translation of
// mineru-vl-utils table_image_processor.py (c9797cde, Apache-2.0, MinerU Team);
// modified: C++, embedded JPEG encoder/RNG/glyph atlas replace Pillow/random.
#include "mineru_table_image.h"

#include "mineru_token_glyphs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace mineru_img {

// ------------------------------------------------------------- random ------
void python_random::init_genrand(uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < 624; ++i) mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
    index_ = 624;
}

void python_random::seed_int(uint64_t value) {
    // CPython random_seed: key = 32-bit little-endian words of abs(n), [0] for 0.
    std::vector<uint32_t> key;
    if (value == 0) key.push_back(0);
    while (value) {
        key.push_back(static_cast<uint32_t>(value & 0xffffffffu));
        value >>= 32;
    }
    init_genrand(19650218U);
    int i = 1, j = 0;
    const int n = 624, len = static_cast<int>(key.size());
    for (int k = std::max(n, len); k; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[static_cast<size_t>(j)] +
                 static_cast<uint32_t>(j);
        ++i, ++j;
        if (i >= n) {
            mt_[0] = mt_[n - 1];
            i = 1;
        }
        if (j >= len) j = 0;
    }
    for (int k = n - 1; k; --k) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - static_cast<uint32_t>(i);
        ++i;
        if (i >= n) {
            mt_[0] = mt_[n - 1];
            i = 1;
        }
    }
    mt_[0] = 0x80000000U;
    index_ = 624;
}

void python_random::seed_os() {
    std::random_device rd;
    uint64_t v = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    seed_int(v);
}

uint32_t python_random::genrand() {
    static const uint32_t mag01[2] = { 0x0U, 0x9908b0dfU };
    if (index_ >= 624) {
        int kk = 0;
        for (; kk < 624 - 397; ++kk) {
            const uint32_t y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
            mt_[kk] = mt_[kk + 397] ^ (y >> 1) ^ mag01[y & 1U];
        }
        for (; kk < 623; ++kk) {
            const uint32_t y = (mt_[kk] & 0x80000000U) | (mt_[kk + 1] & 0x7fffffffU);
            mt_[kk] = mt_[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1U];
        }
        const uint32_t y = (mt_[623] & 0x80000000U) | (mt_[0] & 0x7fffffffU);
        mt_[623] = mt_[396] ^ (y >> 1) ^ mag01[y & 1U];
        index_ = 0;
    }
    uint32_t y = mt_[static_cast<size_t>(index_++)];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

double python_random::random() {
    const uint32_t a = genrand() >> 5, b = genrand() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

// --------------------------------------------------------- image ops -------
rgb_image crop(const rgb_image & src, int x0, int y0, int x1, int y1) {
    rgb_image out;
    out.width = std::max(0, x1 - x0);
    out.height = std::max(0, y1 - y0);
    out.px.assign(static_cast<size_t>(out.width) * out.height * 3, 0);
    for (int y = 0; y < out.height; ++y)
        for (int x = 0; x < out.width; ++x) {
            const int sx = x0 + x, sy = y0 + y;
            if (sx < 0 || sy < 0 || sx >= src.width || sy >= src.height) continue;
            std::memcpy(out.at(x, y), src.at(sx, sy), 3);
        }
    return out;
}

rgb_image rotate(const rgb_image & src, int angle) {
    if (angle != 90 && angle != 180 && angle != 270) return src;
    rgb_image out;
    out.width = angle == 180 ? src.width : src.height;
    out.height = angle == 180 ? src.height : src.width;
    out.px.resize(src.px.size());
    for (int y = 0; y < src.height; ++y)
        for (int x = 0; x < src.width; ++x) {
            int rx, ry;
            if (angle == 90) { // ROTATE_90 (counter-clockwise)
                rx = y;
                ry = src.width - 1 - x;
            } else if (angle == 180) {
                rx = src.width - 1 - x;
                ry = src.height - 1 - y;
            } else {
                rx = src.height - 1 - y;
                ry = x;
            }
            std::memcpy(out.at(rx, ry), src.at(x, y), 3);
        }
    return out;
}

void fill_rectangle(rgb_image & img, int x0, int y0, int x1, int y1, const std::array<uint8_t, 3> & c) {
    if (x1 < x0 || y1 < y0) throw std::invalid_argument("rectangle x1/y1 must be >= x0/y0");
    for (int y = std::max(0, y0); y <= std::min(img.height - 1, y1); ++y)
        for (int x = std::max(0, x0); x <= std::min(img.width - 1, x1); ++x) std::memcpy(img.at(x, y), c.data(), 3);
}

// ---------------------------------------------------------------- JPEG -----
namespace {
const int zigzag[64] = { 0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,
                         41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
                         30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };
const int std_lum[64] = { 16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
                          14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
                          18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
                          49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99 };
const int std_chr[64] = { 17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99,
                          99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
                          99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99 };
const uint8_t dc_lum_bits[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
const uint8_t dc_chr_bits[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const uint8_t dc_vals[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
const uint8_t ac_lum_bits[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
const uint8_t ac_lum_vals[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71,
    0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72,
    0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37,
    0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
    0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
};
const uint8_t ac_chr_bits[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
const uint8_t ac_chr_vals[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22,
    0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1,
    0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36,
    0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
    0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
};

struct huff {
    uint16_t code[256] = {};
    uint8_t size[256] = {};
};
huff build_huff(const uint8_t * bits, const uint8_t * vals) {
    huff h;
    uint16_t code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len) {
        for (int i = 0; i < bits[len - 1]; ++i) {
            h.code[vals[k]] = code++;
            h.size[vals[k]] = static_cast<uint8_t>(len);
            ++k;
        }
        code <<= 1;
    }
    return h;
}

struct quant {
    int q[64]; // natural order quantval
    uint32_t recip[64], corr[64];
    int shift[64];
};
// libjpeg-turbo jcdctmgr.c compute_reciprocal, DCTELEM = 16-bit (SIMD builds).
void reciprocal(int divisor, uint32_t & recip, uint32_t & corr, int & shift) {
    if (divisor == 1) {
        recip = 1;
        corr = 0;
        shift = -16;
        return;
    }
    int b = 0;
    while ((1 << (b + 1)) <= divisor) ++b; // flss(divisor) - 1
    int r = 16 + b;
    uint64_t fq = (uint64_t(1) << r) / static_cast<uint64_t>(divisor);
    const uint64_t fr = (uint64_t(1) << r) % static_cast<uint64_t>(divisor);
    uint32_t c = static_cast<uint32_t>(divisor / 2);
    if (fr == 0) {
        fq >>= 1;
        --r;
    } else if (fr <= static_cast<uint64_t>(divisor) / 2U) {
        ++c;
    } else {
        ++fq;
    }
    recip = static_cast<uint32_t>(fq);
    corr = c;
    shift = r - 16;
}
quant make_quant(const int * basic) {
    quant t;
    for (int i = 0; i < 64; ++i) {
        long v = (static_cast<long>(basic[i]) * 50 + 50) / 100; // jpeg_quality_scaling(75) = 50
        if (v <= 0) v = 1;
        if (v > 255) v = 255; // force_baseline
        t.q[i] = static_cast<int>(v);
        reciprocal(t.q[i] << 3, t.recip[i], t.corr[i], t.shift[i]);
    }
    return t;
}

// jfdctint.c (islow), 16-bit DCTELEM semantics; data in natural order.
void fdct_islow(int * d) {
    const int CB = 13, P1 = 2;
    auto descale = [](long x, int n) { return static_cast<int>((x + (1L << (n - 1))) >> n); };
    for (int r = 0; r < 8; ++r) {
        int * p = d + r * 8;
        const long t0 = p[0] + p[7], t7 = p[0] - p[7], t1 = p[1] + p[6], t6 = p[1] - p[6];
        const long t2 = p[2] + p[5], t5 = p[2] - p[5], t3 = p[3] + p[4], t4 = p[3] - p[4];
        const long t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;
        p[0] = static_cast<int>((t10 + t11) << P1);
        p[4] = static_cast<int>((t10 - t11) << P1);
        long z1 = (t12 + t13) * 4433;
        p[2] = descale(z1 + t13 * 6270, CB - P1);
        p[6] = descale(z1 + t12 * (-15137), CB - P1);
        z1 = t4 + t7;
        long z2 = t5 + t6, z3 = t4 + t6, z4 = t5 + t7;
        const long z5 = (z3 + z4) * 9633;
        const long a4 = t4 * 2446, a5 = t5 * 16819, a6 = t6 * 25172, a7 = t7 * 12299;
        z1 *= -7373;
        z2 *= -20995;
        z3 *= -16069;
        z4 *= -3196;
        z3 += z5;
        z4 += z5;
        p[7] = descale(a4 + z1 + z3, CB - P1);
        p[5] = descale(a5 + z2 + z4, CB - P1);
        p[3] = descale(a6 + z2 + z3, CB - P1);
        p[1] = descale(a7 + z1 + z4, CB - P1);
    }
    for (int c = 0; c < 8; ++c) {
        int * p = d + c;
        const long t0 = p[0] + p[56], t7 = p[0] - p[56], t1 = p[8] + p[48], t6 = p[8] - p[48];
        const long t2 = p[16] + p[40], t5 = p[16] - p[40], t3 = p[24] + p[32], t4 = p[24] - p[32];
        const long t10 = t0 + t3, t13 = t0 - t3, t11 = t1 + t2, t12 = t1 - t2;
        p[0] = descale(t10 + t11, P1);
        p[32] = descale(t10 - t11, P1);
        long z1 = (t12 + t13) * 4433;
        p[16] = descale(z1 + t13 * 6270, CB + P1);
        p[48] = descale(z1 + t12 * (-15137), CB + P1);
        z1 = t4 + t7;
        long z2 = t5 + t6, z3 = t4 + t6, z4 = t5 + t7;
        const long z5 = (z3 + z4) * 9633;
        const long a4 = t4 * 2446, a5 = t5 * 16819, a6 = t6 * 25172, a7 = t7 * 12299;
        z1 *= -7373;
        z2 *= -20995;
        z3 *= -16069;
        z4 *= -3196;
        z3 += z5;
        z4 += z5;
        p[56] = descale(a4 + z1 + z3, CB + P1);
        p[40] = descale(a5 + z2 + z4, CB + P1);
        p[24] = descale(a6 + z2 + z3, CB + P1);
        p[8] = descale(a7 + z1 + z4, CB + P1);
    }
}

struct bit_writer {
    std::vector<uint8_t> & out;
    uint32_t buf = 0;
    int bits = 0;
    void put(uint32_t code, int size) {
        for (int i = size - 1; i >= 0; --i) {
            buf = (buf << 1) | ((code >> i) & 1U);
            if (++bits == 8) {
                out.push_back(static_cast<uint8_t>(buf));
                if ((buf & 0xff) == 0xff) out.push_back(0);
                buf = 0;
                bits = 0;
            }
        }
    }
    void flush() {
        while (bits) put(1, 1);
    }
};

void encode_block(bit_writer & w, const int * coef, int & last_dc, const huff & dc, const huff & ac) {
    auto nbits = [](int v) {
        int n = 0;
        unsigned a = static_cast<unsigned>(v < 0 ? -v : v);
        while (a) {
            ++n;
            a >>= 1;
        }
        return n;
    };
    int diff = coef[0] - last_dc;
    last_dc = coef[0];
    int n = nbits(diff);
    w.put(dc.code[n], dc.size[n]);
    if (n) w.put(static_cast<uint32_t>(diff < 0 ? diff - 1 : diff) & ((1U << n) - 1), n);
    int run = 0;
    for (int k = 1; k < 64; ++k) {
        const int v = coef[zigzag[k]];
        if (v == 0) {
            ++run;
            continue;
        }
        while (run > 15) {
            w.put(ac.code[0xf0], ac.size[0xf0]);
            run -= 16;
        }
        n = nbits(v);
        const int sym = (run << 4) + n;
        w.put(ac.code[sym], ac.size[sym]);
        w.put(static_cast<uint32_t>(v < 0 ? v - 1 : v) & ((1U << n) - 1), n);
        run = 0;
    }
    if (run > 0) w.put(ac.code[0], ac.size[0]);
}

void quantize(int * d, const quant & q, int * out) {
    for (int i = 0; i < 64; ++i) {
        int t = d[i];
        const bool neg = t < 0;
        if (neg) t = -t;
        uint32_t product = (static_cast<uint32_t>(static_cast<uint16_t>(t + static_cast<int>(q.corr[i])))) * q.recip[i];
        product >>= q.shift[i] + 16;
        int v = static_cast<int16_t>(static_cast<uint16_t>(product));
        out[i] = neg ? -v : v;
    }
}
} // namespace

std::vector<uint8_t> encode_jpeg_pillow(const rgb_image & img) {
    if (img.width <= 0 || img.height <= 0 || img.width > 65500 || img.height > 65500)
        throw std::invalid_argument("JPEG dimensions out of range");
    const int W = img.width, H = img.height;
    // Color conversion (jccolor.c rgb_ycc_convert, SCALEBITS 16).
    auto fix = [](double x) { return static_cast<long>(x * 65536.0 + 0.5); };
    static long tab[8][256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) {
            tab[0][i] = fix(0.29900) * i;
            tab[1][i] = fix(0.58700) * i;
            tab[2][i] = fix(0.11400) * i + (1L << 15);
            tab[3][i] = -fix(0.16874) * i;
            tab[4][i] = -fix(0.33126) * i;
            tab[5][i] = fix(0.5) * i + (128L << 16) + (1L << 15) - 1; // B->Cb, R->Cr
            tab[6][i] = -fix(0.41869) * i;
            tab[7][i] = -fix(0.08131) * i;
        }
        init = true;
    }
    const int y_wb = (W + 7) / 8, y_hb = (H + 7) / 8; // Y width_in_blocks / height_in_blocks
    const int c_wb = (W + 15) / 16, c_hb = (H + 15) / 16;
    const int mcux = c_wb, mcuy = c_hb;
    // Y plane padded by edge replication to y_wb*8 x (mcuy*16) rows.
    const int YW = y_wb * 8, YH = mcuy * 16, CW = c_wb * 8, CH = mcuy * 8;
    std::vector<uint8_t> Y(static_cast<size_t>(YW) * YH), Cb(static_cast<size_t>(c_wb * 16) * YH),
        Cr(static_cast<size_t>(c_wb * 16) * YH);
    const int FW = c_wb * 16; // full-res width for chroma downsampling input
    // Rows: real rows, then first pad to even (replicate), downsample, then
    // replicate last output row to fill the iMCU. For Y, equivalent to
    // replicating the last real row.
    const int H2 = H + (H & 1);
    for (int y = 0; y < H2; ++y) {
        const int sy = std::min(y, H - 1);
        for (int x = 0; x < FW; ++x) {
            const int sx = std::min(x, W - 1);
            const uint8_t * p = img.at(sx, sy);
            const int r = p[0], g = p[1], b = p[2];
            const int yy = static_cast<int>((tab[0][r] + tab[1][g] + tab[2][b]) >> 16);
            const int cb = static_cast<int>((tab[3][r] + tab[4][g] + tab[5][b]) >> 16);
            const int cr = static_cast<int>((tab[5][r] + tab[6][g] + tab[7][b]) >> 16);
            if (x < YW) Y[static_cast<size_t>(y) * YW + x] = static_cast<uint8_t>(yy);
            Cb[static_cast<size_t>(y) * FW + x] = static_cast<uint8_t>(cb);
            Cr[static_cast<size_t>(y) * FW + x] = static_cast<uint8_t>(cr);
        }
    }
    for (int y = H2; y < YH; ++y)
        std::memcpy(Y.data() + static_cast<size_t>(y) * YW, Y.data() + static_cast<size_t>(H2 - 1) * YW, YW);
    // h2v2 downsample.
    std::vector<uint8_t> cb2(static_cast<size_t>(CW) * CH), cr2(static_cast<size_t>(CW) * CH);
    const int crow = H2 / 2;
    for (int y = 0; y < crow; ++y)
        for (int plane = 0; plane < 2; ++plane) {
            const std::vector<uint8_t> & src = plane ? Cr : Cb;
            std::vector<uint8_t> & dst = plane ? cr2 : cb2;
            int bias = 1;
            for (int x = 0; x < CW; ++x) {
                const uint8_t * a = src.data() + static_cast<size_t>(2 * y) * FW + 2 * x;
                const uint8_t * b = a + FW;
                dst[static_cast<size_t>(y) * CW + x] = static_cast<uint8_t>((a[0] + a[1] + b[0] + b[1] + bias) >> 2);
                bias ^= 3;
            }
        }
    for (int y = crow; y < CH; ++y) {
        std::memcpy(cb2.data() + static_cast<size_t>(y) * CW, cb2.data() + static_cast<size_t>(crow - 1) * CW, CW);
        std::memcpy(cr2.data() + static_cast<size_t>(y) * CW, cr2.data() + static_cast<size_t>(crow - 1) * CW, CW);
    }
    static const quant ql = make_quant(std_lum), qc = make_quant(std_chr);
    static const huff hdl = build_huff(dc_lum_bits, dc_vals), hal = build_huff(ac_lum_bits, ac_lum_vals),
                      hdc = build_huff(dc_chr_bits, dc_vals), hac = build_huff(ac_chr_bits, ac_chr_vals);
    std::vector<uint8_t> out;
    auto put16 = [&](int v) {
        out.push_back(static_cast<uint8_t>(v >> 8));
        out.push_back(static_cast<uint8_t>(v & 0xff));
    };
    const uint8_t soi_app0[] = { 0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 'J',  'F',  'I',  'F',
                                 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00 };
    out.insert(out.end(), soi_app0, soi_app0 + sizeof(soi_app0));
    for (int t = 0; t < 2; ++t) {
        const quant & q = t ? qc : ql;
        out.push_back(0xFF);
        out.push_back(0xDB);
        put16(67);
        out.push_back(static_cast<uint8_t>(t));
        for (int k = 0; k < 64; ++k) out.push_back(static_cast<uint8_t>(q.q[zigzag[k]]));
    }
    out.push_back(0xFF);
    out.push_back(0xC0);
    put16(17);
    out.push_back(8);
    put16(H);
    put16(W);
    out.push_back(3);
    const uint8_t comps[] = { 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1 };
    out.insert(out.end(), comps, comps + 9);
    auto dht = [&](int cls_id, const uint8_t * bits, const uint8_t * vals) {
        int n = 0;
        for (int i = 0; i < 16; ++i) n += bits[i];
        out.push_back(0xFF);
        out.push_back(0xC4);
        put16(2 + 1 + 16 + n);
        out.push_back(static_cast<uint8_t>(cls_id));
        out.insert(out.end(), bits, bits + 16);
        out.insert(out.end(), vals, vals + n);
    };
    dht(0x00, dc_lum_bits, dc_vals);
    dht(0x10, ac_lum_bits, ac_lum_vals);
    dht(0x01, dc_chr_bits, dc_vals);
    dht(0x11, ac_chr_bits, ac_chr_vals);
    const uint8_t sos[] = { 0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00 };
    out.insert(out.end(), sos, sos + sizeof(sos));
    bit_writer w{ out };
    int dc_y = 0, dc_cb = 0, dc_cr = 0;
    int blk[64], coef[64];
    // Quantized Y blocks; dummy blocks copy the previous block's DC.
    auto y_block = [&](int bx, int by, int * res) {
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x) blk[y * 8 + x] = Y[static_cast<size_t>(by * 8 + y) * YW + bx * 8 + x] - 128;
        fdct_islow(blk);
        quantize(blk, ql, res);
    };
    for (int my = 0; my < mcuy; ++my)
        for (int mx = 0; mx < mcux; ++mx) {
            int ycoef[4][64];
            // jccoefct.c compress_data: within a real block row, right-edge
            // dummy blocks copy the DC of the block to their left; every
            // block of a dummy (bottom) row copies the DC of the last block
            // of the previous row in this MCU.
            for (int row = 0; row < 2; ++row) {
                const int by = my * 2 + row;
                for (int col = 0; col < 2; ++col) {
                    const int k = row * 2 + col, bx = mx * 2 + col;
                    if (by < y_hb && bx < y_wb) {
                        y_block(bx, by, ycoef[k]);
                        continue;
                    }
                    std::memset(ycoef[k], 0, sizeof(ycoef[k]));
                    ycoef[k][0] = by < y_hb ? ycoef[k - 1][0] : ycoef[row * 2 - 1][0];
                }
            }
            for (int k = 0; k < 4; ++k) encode_block(w, ycoef[k], dc_y, hdl, hal);
            for (int plane = 0; plane < 2; ++plane) {
                const std::vector<uint8_t> & src = plane ? cr2 : cb2;
                for (int y = 0; y < 8; ++y)
                    for (int x = 0; x < 8; ++x)
                        blk[y * 8 + x] = src[static_cast<size_t>(my * 8 + y) * CW + mx * 8 + x] - 128;
                fdct_islow(blk);
                quantize(blk, qc, coef);
                encode_block(w, coef, plane ? dc_cr : dc_cb, hdc, hac);
            }
        }
    w.flush();
    out.push_back(0xFF);
    out.push_back(0xD9);
    return out;
}

std::string base64(const std::vector<uint8_t> & bytes) {
    static const char * t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    for (; i + 2 < bytes.size(); i += 3) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += t[v & 63];
    }
    if (i + 1 == bytes.size()) {
        const uint32_t v = bytes[i] << 16;
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += "==";
    } else if (i + 2 == bytes.size()) {
        const uint32_t v = (bytes[i] << 16) | (bytes[i + 1] << 8);
        out += t[v >> 18];
        out += t[(v >> 12) & 63];
        out += t[(v >> 6) & 63];
        out += '=';
    }
    return out;
}

// --------------------------------------------------------- token text ------
namespace {
const mineru_glyph * glyph(char c) {
    for (const auto & g : mineru_glyphs)
        if (g.ch == c) return &g;
    return nullptr;
}
struct text_metrics {
    int l = 0, t = 0, r = 0, b = 0;
};
// Pillow getbbox approximation at size `s` from the 64 px reference.
text_metrics measure(const std::string & text, int s) {
    const double f = static_cast<double>(s) / mineru_glyph_ref_size;
    double pen = 0;
    double l = 1e9, t = 1e9, r = -1e9, b = -1e9;
    for (char c : text) {
        const mineru_glyph * g = glyph(c);
        if (!g) throw std::invalid_argument("token glyph missing");
        l = std::min(l, pen + g->l);
        r = std::max(r, pen + g->r);
        t = std::min(t, static_cast<double>(g->t));
        b = std::max(b, static_cast<double>(g->b));
        pen += g->advance;
    }
    text_metrics m;
    m.l = static_cast<int>(std::floor(l * f));
    m.t = static_cast<int>(std::floor(t * f));
    m.r = static_cast<int>(std::ceil(r * f));
    m.b = static_cast<int>(std::ceil(b * f));
    return m;
}
uint8_t coverage(const std::string & text, double rx, double ry) {
    // Reference-space coverage at point (rx, ry) relative to the pen origin.
    double pen = 0;
    for (char c : text) {
        const mineru_glyph * g = glyph(c);
        const double gx = rx - pen - g->l, gy = ry - g->t;
        if (gx >= 0 && gy >= 0 && gx < g->w && gy < g->h)
            return mineru_glyph_masks[g->offset + static_cast<int>(gy) * g->w + static_cast<int>(gx)];
        pen += g->advance;
    }
    return 0;
}
} // namespace

bool draw_token_text(rgb_image & img, int bx, int by, int bw, int bh, const std::string & token,
                     const std::array<uint8_t, 3> & color) {
    // _get_optimal_pil_font: binary search, fill_ratio 0.7, min 4, max(100, int(h*0.7)).
    int left = 4, right = std::max(100, static_cast<int>(bh * 0.7));
    int best_size = 4, best_w = 0, best_h = 0;
    for (int it = 0; it < 30 && left <= right; ++it) {
        const int mid = (left + right) / 2;
        const auto m = measure(token, mid);
        const int w = m.r - m.l, h = m.b - m.t;
        if (w <= bw * 0.7 && h <= bh * 0.7) {
            best_size = mid, best_w = w, best_h = h;
            left = mid + 1;
        } else
            right = mid - 1;
    }
    if (!(best_w <= bw && best_h <= bh)) return false;
    const double cx = bx + bw / 2.0, cy = by + bh / 2.0;
    const double ox = cx - best_w / 2.0, oy = cy - best_h / 2.0; // anchor "la" origin
    const auto m = measure(token, best_size);
    const double f = static_cast<double>(mineru_glyph_ref_size) / best_size;
    const int x0 = static_cast<int>(std::floor(ox)), y0 = static_cast<int>(std::floor(oy));
    for (int y = m.t; y < m.b; ++y)
        for (int x = m.l; x < m.r; ++x) {
            // 2x2 supersampled coverage from the reference atlas.
            int acc = 0;
            for (int sy = 0; sy < 2; ++sy)
                for (int sx = 0; sx < 2; ++sx)
                    acc += coverage(token, (x + 0.25 + 0.5 * sx) * f, (y + 0.25 + 0.5 * sy) * f);
            const int a = acc / 4;
            const int px = x0 + x, py = y0 + y;
            if (!a || px < 0 || py < 0 || px >= img.width || py >= img.height) continue;
            uint8_t * p = img.at(px, py);
            for (int c = 0; c < 3; ++c) {
                const int tmp = p[c] * (255 - a) + color[static_cast<size_t>(c)] * a + 128;
                p[c] = static_cast<uint8_t>((tmp + (tmp >> 8)) >> 8);
            }
        }
    return true;
}

// ----------------------------------------------------------- masking -------
namespace {
double inter_area(const std::array<double, 4> & a, const std::array<double, 4> & b) {
    const double x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]), x2 = std::min(a[2], b[2]),
                 y2 = std::min(a[3], b[3]);
    if (x2 <= x1 || y2 <= y1) return 0.0;
    return (x2 - x1) * (y2 - y1);
}
double bbox_area(const std::array<double, 4> & a) {
    return std::max(0.0, a[2] - a[0]) * std::max(0.0, a[3] - a[1]);
}
double overlap_ratio(const std::array<double, 4> & inner, const std::array<double, 4> & outer) {
    const double a = bbox_area(inner);
    if (a == 0) return 0.0;
    return inter_area(inner, outer) / a;
}
int py_int(double v) { // Python int() truncation
    return static_cast<int>(std::trunc(v));
}
} // namespace

std::vector<std::pair<size_t, std::vector<size_t>>> build_table_image_map(const mineru_pp::block_list & blocks,
                                                                          const std::vector<size_t> & tables,
                                                                          double threshold) {
    std::vector<std::pair<size_t, std::vector<size_t>>> out;
    for (size_t t : tables) out.push_back({ t, {} });
    if (tables.empty()) return out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]->type != "image") continue;
        long best = -1;
        double best_ratio = threshold, best_area = 0;
        bool has_area = false;
        for (size_t k = 0; k < tables.size(); ++k) {
            const auto & tb = *blocks[tables[k]];
            const double ratio = overlap_ratio(blocks[i]->bbox, tb.bbox);
            if (ratio < threshold) continue;
            const double area = bbox_area(tb.bbox);
            if (best < 0 || ratio > best_ratio || (ratio == best_ratio && has_area && area < best_area)) {
                best = static_cast<long>(k);
                best_ratio = ratio;
                best_area = area;
                has_area = true;
            }
        }
        if (best >= 0) out[static_cast<size_t>(best)].second.push_back(i);
    }
    // Duplicate table indices share one dict entry upstream (last assignment).
    for (auto & kv : out)
        std::stable_sort(kv.second.begin(), kv.second.end(), [&](size_t a, size_t b) {
            const auto & x = blocks[a]->bbox;
            const auto & y = blocks[b]->bbox;
            return x[1] < y[1] || (x[1] == y[1] && x[0] < y[0]);
        });
    return out;
}

rgb_image mask_and_encode_table_image(const rgb_image & page, const mineru_pp::block & table,
                                      const std::vector<const mineru_pp::block *> & images,
                                      const rgb_image & table_crop, python_random & rng,
                                      std::vector<std::pair<mineru_pp::ustring, mineru_pp::ustring>> & token_map) {
    const int width = page.width, height = page.height;
    const int abs_x1_t = py_int(table.bbox[0] * width), abs_y1_t = py_int(table.bbox[1] * height);
    const int tw = table_crop.width, th = table_crop.height;
    const int angle = (table.angle == 90 || table.angle == 180 || table.angle == 270) ? table.angle : 0;
    rgb_image masked = rotate(table_crop, angle);
    static const char * chars = "ACDGHKTWXYZ2345678";
    std::vector<std::string> used;
    for (const mineru_pp::block * ib : images) {
        const double ax1 = ib->bbox[0] * width, ay1 = ib->bbox[1] * height, ax2 = ib->bbox[2] * width,
                     ay2 = ib->bbox[3] * height;
        const int rx1 = py_int(std::max(0.0, ax1 - abs_x1_t)), ry1 = py_int(std::max(0.0, ay1 - abs_y1_t));
        const int rx2 = py_int(std::min(static_cast<double>(tw), ax2 - abs_x1_t)),
                  ry2 = py_int(std::min(static_cast<double>(th), ay2 - abs_y1_t));
        if (rx2 <= rx1 || ry2 <= ry1) continue;
        const rgb_image cimg = crop(page, py_int(ax1), py_int(ay1), py_int(ax2), py_int(ay2));
        if (cimg.width < 1 || cimg.height < 1) continue;
        if (used.size() >= 18u * 18u * 18u * 18u) throw std::runtime_error("Exhausted random table image token space");
        std::string code;
        while (true) {
            code.clear();
            for (int k = 0; k < 4; ++k) code += chars[static_cast<int>(std::floor(rng.random() * 18))];
            if (std::find(used.begin(), used.end(), code) == used.end()) {
                used.push_back(code);
                break;
            }
        }
        const std::string token = "[" + code + "]";
        token_map.push_back(
            { pyre::from_utf8(token),
              pyre::from_utf8(("data:image/jpeg;base64," + base64(encode_jpeg_pillow(rotate(cimg, angle)))).c_str()) });
        // _rotate_box_in_image
        int b0 = rx1, b1 = ry1, b2 = rx2, b3 = ry2;
        if (angle == 90) {
            b0 = ry1, b1 = tw - rx2, b2 = ry2, b3 = tw - rx1;
        } else if (angle == 180) {
            b0 = tw - rx2, b1 = th - ry2, b2 = tw - rx1, b3 = th - ry1;
        } else if (angle == 270) {
            b0 = th - ry2, b1 = rx1, b2 = th - ry1, b3 = rx2;
        }
        // _get_average_color on the current masked image.
        std::array<uint8_t, 3> avg{ { 255, 255, 255 } };
        {
            const int pad = 2, mx = (b0 + b2) / 2, my = (b1 + b3) / 2;
            const int pts[8][2] = {
                { b0 - pad, b1 - pad }, { mx, b1 - pad }, { b2 + pad, b1 - pad }, { b2 + pad, my },
                { b2 + pad, b3 + pad }, { mx, b3 + pad }, { b0 - pad, b3 + pad }, { b0 - pad, my }
            };
            long sr = 0, sg = 0, sb = 0;
            if (masked.width > 0 && masked.height > 0) {
                for (const auto & p : pts) {
                    const int x = std::max(0, std::min(p[0], masked.width - 1)),
                              y = std::max(0, std::min(p[1], masked.height - 1));
                    const uint8_t * px = masked.at(x, y);
                    sr += px[0], sg += px[1], sb += px[2];
                }
                avg = { { static_cast<uint8_t>(sr / 8), static_cast<uint8_t>(sg / 8), static_cast<uint8_t>(sb / 8) } };
            }
        }
        fill_rectangle(masked, b0, b1, b2, b3, avg);
        const int bw = b2 - b0, bh = b3 - b1;
        const double lum = 0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2];
        const std::array<uint8_t, 3> text_color =
            lum < 128 ? std::array<uint8_t, 3>{ { 255, 255, 255 } } : std::array<uint8_t, 3>{ { 0, 0, 0 } };
        draw_token_text(masked, b0, b1, bw, bh, token, text_color);
    }
    return masked;
}

} // namespace mineru_img
