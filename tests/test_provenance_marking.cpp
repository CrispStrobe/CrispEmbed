// test_provenance_marking.cpp — Art. 50(2) opt-in image marking.
//
// Three things worth pinning, because each has a way of failing silently:
//
//  1. OFF by default. POLICY.md §5's reasoned position is that document
//     restoration needs no marking; if this ever defaulted on, every existing
//     caller's output would change and the position would have been reversed
//     by accident rather than argued.
//  2. ON when asked, in a parseable shape.
//  3. A marked image is still decodable BY US. The marker is a Netpbm header
//     comment, which is only safe because stb_image's PNM loader skips '#'
//     runs — if that ever stops being true, marking would corrupt every
//     round-trip through the scan/dewarp paths, and the corruption would look
//     like a model bug rather than a metadata bug.

#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"

#include "core/provenance.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// setenv/unsetenv are POSIX; MSVC has _putenv_s, where an empty value removes
// the variable. Same shape as the helpers in test_dbnet_diff / test_ocr_pipeline_pool.
// Consequence for the "empty is off" case below: Windows cannot hold an
// empty-valued variable at all, so there it degenerates into the unset case.
// Both must be off, so the check is still meaningful — just weaker on Windows.
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

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

// Build a tiny PPM with whatever the marker currently yields, then decode it.
bool roundtrip(const std::string & comment, int & w, int & h) {
    const int W = 4, H = 3;
    std::string ppm = "P6\n" + comment + std::to_string(W) + " " + std::to_string(H) + "\n255\n";
    std::vector<unsigned char> px((size_t)W * H * 3);
    for (size_t i = 0; i < px.size(); i++) px[i] = (unsigned char)(i * 7);
    ppm.append((const char *)px.data(), px.size());

    int c = 0;
    unsigned char * out = stbi_load_from_memory((const unsigned char *)ppm.data(), (int)ppm.size(), &w, &h, &c, 3);
    if (!out) return false;
    const bool same = (w == W && h == H) && std::memcmp(out, px.data(), px.size()) == 0;
    stbi_image_free(out);
    return same;
}

} // namespace

int main() {
    std::printf("Art. 50(2) opt-in marking\n");

    unset_env("CRISPEMBED_MARK_GENERATED");
    check(!core_prov::marking_enabled(), "disabled when the variable is unset");
    check(core_prov::netpbm_comment("esrgan-sr").empty(), "emits nothing when disabled");

    set_env("CRISPEMBED_MARK_GENERATED", "0");
    check(!core_prov::marking_enabled(), "\"0\" is off, not merely 'set'");
    set_env("CRISPEMBED_MARK_GENERATED", "");
    check(!core_prov::marking_enabled(), "empty is off");

    set_env("CRISPEMBED_MARK_GENERATED", "1");
    check(core_prov::marking_enabled(), "enabled when set");

    const std::string c = core_prov::netpbm_comment("esrgan-sr");
    check(c.find("# CrispEmbed-Generated: true") != std::string::npos, "declares generated");
    check(c.find("# CrispEmbed-Engine: esrgan-sr") != std::string::npos, "names the engine");
    check(c.find("POLICY.md") != std::string::npos, "points at the policy");
    check(!c.empty() && c.back() == '\n', "ends on a newline so the dimensions start a line");
    // Every line must be a comment: a stray non-'#' line would be read as the
    // image dimensions and silently corrupt the header.
    bool all_comments = true;
    for (size_t i = 0; i < c.size();) {
        size_t e = c.find('\n', i);
        if (e == std::string::npos) e = c.size();
        if (e > i && c[i] != '#') all_comments = false;
        i = e + 1;
    }
    check(all_comments, "every emitted line is a '#' comment");

    int w = 0, h = 0;
    check(roundtrip(c, w, h), "marked PPM decodes byte-identically via stb_image");

    unset_env("CRISPEMBED_MARK_GENERATED");
    check(roundtrip(core_prov::netpbm_comment("esrgan-sr"), w, h), "unmarked PPM still decodes");

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPASS: marking is opt-in, well-formed, and decode-safe.\n");
    return 0;
}
