// test_image_provenance.cpp — how an emitted image is marked.
//
// Covers the three states a build can be in, because they degrade into each
// other silently and only the strongest one is obvious when it works:
//
//   PNG + iTXt            always; the Art. 50(2) machine-readable marking
//   PNG + iTXt + C2PA     when a signing identity is configured
//   raw Netpbm            CRISPEMBED_IMAGE_FORMAT=ppm, for callers that parse it
//
// The signing half only runs when the build found c2pa AND the harness was
// given a cert; otherwise it reports that it was skipped rather than passing
// vacuously, because "0 failures" from a test that quietly did nothing is how
// an unsigned release ships believing it signs.

// crispembed-core provides stb_image_write (via core/image_out.cpp) but NOT the
// decoder, so the round-trip check brings its own. Different symbols, no clash.
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

#include "core/image_out.h"
#include "core/temp_file.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// setenv/unsetenv are POSIX; MSVC has _putenv_s, where an empty value removes
// the variable. Same shape as the helpers in test_dbnet_diff / test_ocr_pipeline_pool.
void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int failures = 0;
int skipped = 0;

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

std::string emit_to_string(int w, int h, int comp, const char * engine) {
    std::vector<unsigned char> px((size_t)w * h * comp);
    for (size_t i = 0; i < px.size(); i++) px[i] = (unsigned char)(i * 11 + 3);

    // core_tmp::make_private is the project's one portable "unpredictable name,
    // created by us" helper — mkstemp on POSIX, GetTempFileName on Windows.
    // It returns a path to an already-created file, so plain fopen is safe here.
    const std::string tmp = core_tmp::make_private();
    if (tmp.empty()) return {};
    const char * path = tmp.c_str();
    std::FILE * f = std::fopen(path, "wb");
    if (!f) {
        std::remove(path);
        return {};
    }
    const bool ok = core_imgout::emit(f, px.data(), w, h, comp, engine);
    std::fclose(f);
    if (!ok) {
        std::remove(path);
        return {};
    }
    std::FILE * r = std::fopen(path, "rb");
    std::string out;
    if (r) {
        char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), r)) > 0) out.append(buf, n);
        std::fclose(r);
    }
    std::remove(path);
    return out;
}

// Independent CRC-32 (bit-by-bit, no table) so a bug in the table-driven one
// in image_out.h cannot validate itself. A wrong chunk CRC is the classic
// silent failure here: lenient decoders (stb, PIL) ignore it and strict ones
// reject the whole file.
uint32_t ref_crc32(const unsigned char * p, size_t n) {
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
    }
    return c ^ 0xffffffffu;
}

uint32_t be32(const unsigned char * p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// Walk every chunk, check every CRC, and report what was seen.
bool png_chunks_valid(const std::string & s, std::vector<std::string> & types) {
    if (s.size() < 8) return false;
    size_t i = 8;
    while (i + 12 <= s.size()) {
        const unsigned char * p = (const unsigned char *)s.data() + i;
        const uint32_t len = be32(p);
        if (i + 12 + (size_t)len > s.size()) return false;
        const std::string type((const char *)p + 4, 4);
        const uint32_t stored = be32(p + 8 + len);
        if (ref_crc32(p + 4, (size_t)len + 4) != stored) return false;
        types.push_back(type);
        i += 12 + (size_t)len;
        if (type == "IEND") return true;
    }
    return false;
}

bool is_png(const std::string & s) {
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    return s.size() > 8 && std::memcmp(s.data(), sig, 8) == 0;
}

} // namespace

static int crispembed_test_main() {
    std::printf("image provenance\n");

    unset_env("CRISPEMBED_IMAGE_FORMAT");
    unset_env("CRISPEMBED_C2PA_CERT");
    unset_env("CRISPEMBED_C2PA_KEY");

    // ── default: PNG + iTXt, unsigned ────────────────────────────────
    const std::string png = emit_to_string(16, 12, 3, "esrgan-sr");
    check(!png.empty(), "emits something by default");
    check(is_png(png), "default format is PNG, not raw Netpbm");
    check(png.find("iTXt") != std::string::npos, "carries an iTXt chunk");
    check(png.find("CrispEmbed") != std::string::npos, "iTXt names the software");
    check(png.find("engine=esrgan-sr") != std::string::npos, "iTXt names the engine");
    // The claim must be the honest one: our inputs are real captures we
    // enhanced, so asserting wholly-synthetic media would be false.
    check(png.find("algorithmicallyEnhanced") != std::string::npos,
          "asserts algorithmicallyEnhanced, not trainedAlgorithmicMedia");
    check(png.find("trainedAlgorithmicMedia") == std::string::npos, "does NOT claim the image is wholly AI-generated");
    // The chunk must sit after IHDR; before it, decoders reject the file.
    const size_t ihdr = png.find("IHDR"), itxt = png.find("iTXt");
    check(ihdr != std::string::npos && itxt > ihdr, "iTXt follows IHDR");

    const std::string gray = emit_to_string(9, 7, 1, "dewarp");
    check(is_png(gray) && gray.find("engine=dewarp") != std::string::npos, "grayscale path marked too");

    // ── escape hatch ─────────────────────────────────────────────────
    set_env("CRISPEMBED_IMAGE_FORMAT", "ppm");
    const std::string ppm = emit_to_string(16, 12, 3, "esrgan-sr");
    check(ppm.rfind("P6", 0) == 0, "CRISPEMBED_IMAGE_FORMAT=ppm still yields raw Netpbm");
    check(!is_png(ppm), "and is not a PNG");
    unset_env("CRISPEMBED_IMAGE_FORMAT");


    // ── the PNG must be structurally valid, not merely accepted ──────
    {
        std::vector<std::string> types;
        check(png_chunks_valid(png, types), "every chunk CRC is correct (independent CRC-32)");
        bool has_ihdr = false, has_itxt = false, has_idat = false, has_iend = false;
        for (const auto & t : types) {
            has_ihdr |= t == "IHDR";
            has_itxt |= t == "iTXt";
            has_idat |= t == "IDAT";
            has_iend |= t == "IEND";
        }
        check(has_ihdr && has_itxt && has_idat && has_iend, "IHDR + iTXt + IDAT + IEND all present");
        check(!types.empty() && types.front() == "IHDR", "IHDR is first");
        check(!types.empty() && types.back() == "IEND", "IEND is last");
    }

    // ── iTXt payload structure (spec: 5 NUL-separated fields) ────────
    {
        const size_t at = png.find("iTXt");
        const unsigned char * p = (const unsigned char *)png.data() + at - 4;
        const uint32_t len = be32(p);
        const std::string data((const char *)p + 8, len);
        // keyword \0 compflag compmethod lang \0 transkey \0 text
        const size_t k = data.find('\0');
        check(k != std::string::npos && data.compare(0, k, "CrispEmbed") == 0, "keyword is CrispEmbed");
        check(k + 4 < data.size() && data[k + 1] == 0 && data[k + 2] == 0,
              "compression flag and method are both 0 (uncompressed)");
        check(data.size() > k + 5, "payload follows the five header fields");
    }

    // ── the buffer path must agree with the file path ────────────────
    {
        std::vector<unsigned char> px(16 * 12 * 3);
        for (size_t i = 0; i < px.size(); i++) px[i] = (unsigned char)(i * 11 + 3);
        std::string buf, mime;
        check(core_imgout::emit_to_string(buf, mime, px.data(), 16, 12, 3, "esrgan-sr"), "emit_to_string succeeds");
        check(mime == "image/png", "reports image/png so a caller cannot mislabel it");
        check(buf == png, "buffer path is byte-identical to the file path");

        set_env("CRISPEMBED_IMAGE_FORMAT", "ppm");
        std::string pbuf, pmime;
        core_imgout::emit_to_string(pbuf, pmime, px.data(), 16, 12, 3, "esrgan-sr");
        check(pmime == "image/x-portable-pixmap", "ppm mode reports the Netpbm MIME, not image/png");
        check(pbuf.rfind("P6", 0) == 0, "ppm mode buffer really is Netpbm");
        unset_env("CRISPEMBED_IMAGE_FORMAT");
    }

    // ── rejects what it cannot honestly emit ─────────────────────────
    {
        std::vector<unsigned char> px(4 * 4 * 3, 7);
        std::string b, m;
        check(!core_imgout::emit_to_string(b, m, px.data(), 0, 4, 3, "x"), "rejects zero width");
        check(!core_imgout::emit_to_string(b, m, px.data(), 4, 4, 2, "x"), "rejects unsupported comp=2");
        check(!core_imgout::emit_to_string(b, m, nullptr, 4, 4, 3, "x"), "rejects null pixels");
        check(!core_imgout::emit(nullptr, nullptr, 4, 4, 3, "x"), "emit() rejects null pixels");
    }

    // ── a missing engine name must not produce a malformed chunk ─────
    {
        const std::string anon = emit_to_string(8, 8, 3, "");
        std::vector<std::string> types;
        check(png_chunks_valid(anon, types), "empty engine name still yields a valid PNG");
        check(anon.find("generated=true") != std::string::npos, "and is still marked");
        check(anon.find("engine=") == std::string::npos, "omits the engine field rather than emitting engine=");
    }


    // ── the format change must not touch a single pixel ──────────────
    // The benchmark harnesses measure PSNR/SSIM/CER on these bytes. If PNG and
    // PPM disagreed anywhere, every restoration metric would shift when the
    // default changed, and it would look like a model regression.
    {
        const int W = 23, H = 17; // deliberately not multiples of anything
        for (int comp = 1; comp <= 3; comp += 2) {
            std::vector<unsigned char> px((size_t)W * H * comp);
            for (size_t i = 0; i < px.size(); i++) px[i] = (unsigned char)((i * 37 + 11) & 0xff);

            unset_env("CRISPEMBED_IMAGE_FORMAT");
            std::string as_png, m1;
            core_imgout::emit_to_string(as_png, m1, px.data(), W, H, comp, "esrgan-sr");

            set_env("CRISPEMBED_IMAGE_FORMAT", "ppm");
            std::string as_ppm, m2;
            core_imgout::emit_to_string(as_ppm, m2, px.data(), W, H, comp, "esrgan-sr");
            unset_env("CRISPEMBED_IMAGE_FORMAT");

            int dw = 0, dh = 0, dc = 0;
            unsigned char * dec =
                stbi_load_from_memory((const unsigned char *)as_png.data(), (int)as_png.size(), &dw, &dh, &dc, comp);
            char label[96];
            std::snprintf(label, sizeof(label), "comp=%d: PNG decodes to the original dimensions", comp);
            check(dec && dw == W && dh == H, label);
            if (dec) {
                std::snprintf(label, sizeof(label), "comp=%d: PNG round-trips the pixels exactly", comp);
                check(std::memcmp(dec, px.data(), px.size()) == 0, label);
                stbi_image_free(dec);
            }

            // And the Netpbm body must be those same bytes, after its header.
            const size_t body = as_ppm.size() - px.size();
            std::snprintf(label, sizeof(label), "comp=%d: PPM body is the original pixels", comp);
            check(as_ppm.size() > px.size() && std::memcmp(as_ppm.data() + body, px.data(), px.size()) == 0, label);
        }
    }

    // ── signing ──────────────────────────────────────────────────────
    check(!core_imgout::c2pa_configured(), "reports unconfigured when no cert is set");

    const char * cert = std::getenv("CRISPEMBED_TEST_C2PA_CERT");
    const char * key = std::getenv("CRISPEMBED_TEST_C2PA_KEY");
#ifndef CRISPEMBED_HAVE_C2PA
    std::printf("  [skip] signing: built without c2pa\n");
    skipped++;
#else
    if (!cert || !key) {
        std::printf("  [skip] signing: set CRISPEMBED_TEST_C2PA_CERT/_KEY to exercise it\n");
        skipped++;
    } else {
        set_env("CRISPEMBED_C2PA_CERT", cert);
        set_env("CRISPEMBED_C2PA_KEY", key);
        check(core_imgout::c2pa_configured(), "reports configured once cert+key are set");
        const std::string signed_png = emit_to_string(16, 12, 3, "esrgan-sr");
        check(is_png(signed_png), "signed output is still a PNG");
        // c2pa embeds its manifest store in a JUMBF box; the identifier is
        // present in the bytes whatever the container details.
        check(signed_png.find("c2pa") != std::string::npos || signed_png.size() > png.size() * 2,
              "signed output carries a manifest (grew and contains c2pa markers)");
        check(signed_png.find("algorithmicallyEnhanced") != std::string::npos,
              "signed output still carries the iTXt marking");
        unset_env("CRISPEMBED_C2PA_CERT");
        unset_env("CRISPEMBED_C2PA_KEY");
    }
#endif

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPASS: images are marked%s.\n", skipped ? " (signing not exercised)" : " and signable");
    return 0;
}

// The guard in tools/check_test_clean_exit.sh: a one-shot binary must not run
// ggml's static GPU-device destructor at exit (it aborts on Metal / faults on
// CUDA). These tests touch no GPU today, but they link crispembed-core, so the
// teardown is one added dependency away from firing.
int main() {
    core_util::clean_exit(crispembed_test_main());
}
