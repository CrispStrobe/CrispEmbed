// image_out.h — how CrispEmbed emits an image, and what it says about it.
//
// PNG by default, because it is the only thing here that can carry provenance:
// raw Netpbm has no metadata container at all, and C2PA has no PPM binding.
// Two levels of marking, in increasing strength:
//
//   1. ALWAYS — a PNG iTXt chunk naming the engine that touched the pixels.
//      Standard PNG metadata, readable by exiftool/PIL/ImageMagick, and it
//      costs ~200 bytes. This is the Art. 50(2) machine-readable marking.
//
//   2. WHEN A SIGNING IDENTITY IS CONFIGURED — a real C2PA manifest.
//      Requires CRISPEMBED_C2PA_CERT + CRISPEMBED_C2PA_KEY (or the CLI flags).
//      We deliberately ship NO default key: a private key published in an MIT
//      repo lets anyone mint a manifest naming CrispEmbed as the software agent
//      for an image it never touched, and lets them re-sign after altering the
//      pixels — destroying both of the jobs a C2PA signature exists to do,
//      while looking like it does them. That is worse than no manifest.
//      scripts/make-c2pa-cert.sh generates a per-installation chain if you want
//      signing without sourcing a certificate; verifiers will still show
//      "unverified signer", but nobody can impersonate the project.
//
// Set CRISPEMBED_IMAGE_FORMAT=ppm to get the old raw Netpbm back (and with it
// only the header-comment marking of core/provenance.h).

#pragma once

#include "provenance.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace core_imgout {

// ── output format ──────────────────────────────────────────────────────
inline bool want_ppm() {
    const char * v = std::getenv("CRISPEMBED_IMAGE_FORMAT");
    return v && (std::strcmp(v, "ppm") == 0 || std::strcmp(v, "pgm") == 0 || std::strcmp(v, "netpbm") == 0);
}

// ── PNG chunk surgery ──────────────────────────────────────────────────
// stb_image_write emits a bare PNG; it has no API for ancillary chunks, so the
// iTXt is spliced in after IHDR (which the spec requires to be first).

inline uint32_t png_crc(const uint8_t * buf, size_t len) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[n] = c;
        }
        init = true;
    }
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < len; i++) c = table[(c ^ buf[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

inline void png_put_u32(std::string & s, uint32_t v) {
    s.push_back((char)((v >> 24) & 0xff));
    s.push_back((char)((v >> 16) & 0xff));
    s.push_back((char)((v >> 8) & 0xff));
    s.push_back((char)(v & 0xff));
}

// Uncompressed iTXt: keyword \0 compflag(0) compmethod(0) lang \0 transkey \0 text
inline std::string png_itxt(const std::string & keyword, const std::string & text) {
    std::string data;
    data += keyword;
    data.push_back('\0');
    data.push_back('\0'); // compression flag: uncompressed
    data.push_back('\0'); // compression method
    data.push_back('\0'); // language tag (empty)
    data.push_back('\0'); // translated keyword (empty)
    data += text;

    std::string chunk;
    png_put_u32(chunk, (uint32_t)data.size());
    const std::string type = "iTXt";
    std::string crc_input = type + data;
    chunk += type;
    chunk += data;
    png_put_u32(chunk, png_crc((const uint8_t *)crc_input.data(), crc_input.size()));
    return chunk;
}

// Insert after IHDR. Returns false if `png` is not a PNG we recognise, in which
// case the caller emits it unchanged rather than corrupting it.
inline bool png_insert_itxt(std::string & png, const std::string & chunk) {
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    if (png.size() < 8 + 25 || std::memcmp(png.data(), sig, 8) != 0) return false;
    // IHDR: 4 len + 4 type + 13 data + 4 crc = 25 bytes, immediately after the signature.
    if (std::memcmp(png.data() + 12, "IHDR", 4) != 0) return false;
    png.insert(8 + 25, chunk);
    return true;
}

// ── C2PA (optional, and only with a configured identity) ───────────────
// Defined in image_out.cpp so the c2pa header/library stay off every other
// translation unit. Returns false when unavailable or unconfigured; the caller
// then emits the iTXt-marked PNG, which is still marked, just not signed.
bool c2pa_sign_png(std::string & png, const char * engine);

// True when both a cert and a key are configured. Split out so the CLI can say
// why an image is unsigned instead of silently degrading.
inline bool c2pa_configured() {
    const char * c = std::getenv("CRISPEMBED_C2PA_CERT");
    const char * k = std::getenv("CRISPEMBED_C2PA_KEY");
    return c && *c && k && *k;
}

// ── the one emission point ─────────────────────────────────────────────
// `data` is interleaved 8-bit, `comp` 1 (gray) or 3 (RGB). Writes to `out`
// (stdout when null). `engine` names what touched the pixels.
bool emit(std::FILE * out, const uint8_t * data, int w, int h, int comp, const char * engine);

} // namespace core_imgout
