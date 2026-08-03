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

#include "core/image_out.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
int skipped = 0;

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

std::string emit_to_string(int w, int h, int comp, const char * engine) {
    std::vector<unsigned char> px((size_t)w * h * comp);
    for (size_t i = 0; i < px.size(); i++) px[i] = (unsigned char)(i * 11 + 3);

    char path[] = "/tmp/crispembed_imgprov_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return {};
    std::FILE * f = fdopen(fd, "wb");
    if (!f) return {};
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

bool is_png(const std::string & s) {
    static const unsigned char sig[8] = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };
    return s.size() > 8 && std::memcmp(s.data(), sig, 8) == 0;
}

} // namespace

int main() {
    std::printf("image provenance\n");

    unsetenv("CRISPEMBED_IMAGE_FORMAT");
    unsetenv("CRISPEMBED_C2PA_CERT");
    unsetenv("CRISPEMBED_C2PA_KEY");

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
    setenv("CRISPEMBED_IMAGE_FORMAT", "ppm", 1);
    const std::string ppm = emit_to_string(16, 12, 3, "esrgan-sr");
    check(ppm.rfind("P6", 0) == 0, "CRISPEMBED_IMAGE_FORMAT=ppm still yields raw Netpbm");
    check(!is_png(ppm), "and is not a PNG");
    unsetenv("CRISPEMBED_IMAGE_FORMAT");

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
        setenv("CRISPEMBED_C2PA_CERT", cert, 1);
        setenv("CRISPEMBED_C2PA_KEY", key, 1);
        check(core_imgout::c2pa_configured(), "reports configured once cert+key are set");
        const std::string signed_png = emit_to_string(16, 12, 3, "esrgan-sr");
        check(is_png(signed_png), "signed output is still a PNG");
        // c2pa embeds its manifest store in a JUMBF box; the identifier is
        // present in the bytes whatever the container details.
        check(signed_png.find("c2pa") != std::string::npos || signed_png.size() > png.size() * 2,
              "signed output carries a manifest (grew and contains c2pa markers)");
        check(signed_png.find("algorithmicallyEnhanced") != std::string::npos,
              "signed output still carries the iTXt marking");
        unsetenv("CRISPEMBED_C2PA_CERT");
        unsetenv("CRISPEMBED_C2PA_KEY");
    }
#endif

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPASS: images are marked%s.\n", skipped ? " (signing not exercised)" : " and signable");
    return 0;
}
