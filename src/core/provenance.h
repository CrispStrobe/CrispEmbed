// provenance.h — opt-in machine-readable marking for generated/modified images.
//
// EU AI Act Art. 50(2) requires providers of systems that generate or
// manipulate image content to mark outputs as artificially produced, in a
// machine-readable format. POLICY.md §5 sets out this project's reasoned
// position that document restoration is standard editing under Recital 134 and
// so falls outside that duty — and says just as plainly that the position is
// untested, that it weakens the further a use sits from the document case, and
// that "CrispEmbed adds no watermark or C2PA provenance marking to any output,
// so if you need marking you must add it yourself".
//
// This closes the last part. It does not overturn the position: marking is OFF
// by default, so the document case is unchanged. It exists so an integrator
// whose use sits away from that case can switch marking on instead of building
// it, which is what POLICY currently tells them to do.
//
//   CRISPEMBED_MARK_GENERATED=1     enable
//
// WHAT THIS IS NOT. It is a header comment, not a signature. Anyone can strip
// it with a text editor, it carries no cryptographic binding to the pixels, and
// it survives no format conversion that drops comments. Art. 50(2) contemplates
// solutions that are effective "as far as is technically feasible", and for raw
// Netpbm — which has no metadata container at all — this is the only in-band
// channel. If you need tamper-evident provenance, you need C2PA with a real
// signing identity, and that is still yours to add.
//
// Netpbm's grammar allows '#' comments anywhere whitespace is allowed in the
// header; stb_image's PNM loader skips them (stbi__pnm_skip_whitespace), so a
// marked image stays readable by this project's own decoder.

#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

namespace core_prov {

// Reads the environment each call rather than caching: a long-lived server
// should not need a restart to change whether its output is marked.
inline bool marking_enabled() {
    const char * v = std::getenv("CRISPEMBED_MARK_GENERATED");
    return v && *v && std::strcmp(v, "0") != 0;
}

// Comment block for a Netpbm header, or "" when marking is off. Emit directly
// after the magic number:
//
//     printf("P6\n%s%d %d\n255\n", core_prov::netpbm_comment("esrgan").c_str(), w, h);
//
// `engine` names what touched the pixels, because "AI-processed" alone does not
// tell a reader whether detail was synthesised (ESRGAN, NAFNet) or merely
// resampled (deskew, dewarp) — a distinction POLICY.md §5 argues matters and
// that a downstream reviewer cannot recover from the pixels.
inline std::string netpbm_comment(const char * engine) {
    if (!marking_enabled()) return std::string();
    std::string s = "# CrispEmbed-Generated: true\n";
    if (engine && *engine) {
        s += "# CrispEmbed-Engine: ";
        s += engine;
        s += "\n";
    }
    s += "# CrispEmbed-Note: AI-processed image. Not an authentic record of the "
         "original; restored or upscaled detail is a plausible completion, not "
         "recovered information.\n";
    s += "# CrispEmbed-Spec: https://github.com/CrispStrobe/CrispEmbed/blob/main/POLICY.md\n";
    return s;
}

} // namespace core_prov
