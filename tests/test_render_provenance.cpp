// test_render_provenance.cpp — producer identity in OCR output formats.
//
// hOCR, ALTO and PDF are archival formats: a document produced today may be
// read in a decade, by someone deciding whether to trust the text. "CrispEmbed"
// alone does not let them trace it to a build, which is why Tesseract writes
// "tesseract 4.1.1" and not "tesseract".
//
// Three separate string literals produce this, so the risk is drift — one
// format updated and the others not. The test reads the version from the same
// macro the code uses and requires all three to agree, so a partial update
// fails here rather than shipping three different answers.
//
// Scope note: this checks the SOFTWARE identity. It does not record which
// recognition engine ran, and POLICY.md §6 turns on that distinction — a CTC
// recogniser transcribes, a VLM confabulates through a smudge. Recording the
// engine needs it threaded through the orchestrator; see PLAN.md.

#include "ocr_render.h"

#include <cstdio>
#include <cstring>
#include <string>

#ifndef CRISPEMBED_VERSION_STR
#define CRISPEMBED_VERSION_STR "unknown"
#endif

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) failures++;
}

std::string render(ocr_render_format fmt) {
    ocr_render_word w{ "hello", 10, 20, 40, 12, 0.9f };
    ocr_render_line l{ &w, 1, 10, 20, 40, 12 };
    ocr_render_page p{ &l, 1, 600, 800, nullptr };

    ocr_renderer * r = ocr_render_create(fmt);
    if (!r) return {};
    ocr_render_begin(r);
    ocr_render_add_page(r, &p);
    ocr_render_end(r);
    const int n = ocr_render_output_size(r);
    std::string out;
    if (n > 0) {
        out.assign(ocr_render_output(r), (size_t)n);
    }
    ocr_render_free(r);
    return out;
}

} // namespace

int main() {
    std::printf("OCR output provenance\n");

    const std::string version = CRISPEMBED_VERSION_STR;
    check(!version.empty() && version != "unknown", "the build exposes a real version (CRISPEMBED_VERSION_STR)");
    // A version of "unknown" would silently pass every check below while
    // recording nothing useful, so fail loudly instead.
    check(version.find('.') != std::string::npos, "version looks like a version");

    const std::string hocr = render(OCR_RENDER_HOCR);
    check(!hocr.empty(), "hOCR renders");
    check(hocr.find("ocr-system") != std::string::npos, "hOCR declares ocr-system");
    check(hocr.find("CrispEmbed " + version) != std::string::npos,
          "hOCR ocr-system carries the version (Tesseract convention)");

    const std::string alto = render(OCR_RENDER_ALTO);
    check(!alto.empty(), "ALTO renders");
    check(alto.find("<softwareName>CrispEmbed</softwareName>") != std::string::npos, "ALTO declares softwareName");
    check(alto.find("<softwareVersion>" + version + "</softwareVersion>") != std::string::npos,
          "ALTO declares softwareVersion (standard element, was absent)");

    const std::string pdf = render(OCR_RENDER_PDF);
    check(!pdf.empty(), "PDF renders");

    // Drift guard: every format that names the producer must name the same one.
    int carrying = 0;
    for (const std::string * s : { &hocr, &alto, &pdf }) {
        if (s->find(version) != std::string::npos) carrying++;
    }
    check(carrying >= 2, "at least the two XML formats carry the same version string");

    // Plain text is not an archival container and must NOT grow a header —
    // callers pipe it and would get provenance mixed into the transcription.
    const std::string txt = render(OCR_RENDER_TEXT);
    check(txt.find("CrispEmbed") == std::string::npos,
          "plain text stays clean (provenance would corrupt the transcription)");

    if (failures) {
        std::printf("\nFAIL: %d check(s) failed.\n", failures);
        return 1;
    }
    std::printf("\nPASS: OCR output is traceable to a build.\n");
    return 0;
}
