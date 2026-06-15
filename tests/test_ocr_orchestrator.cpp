// test_ocr_orchestrator.cpp — deterministic tests for the OCR pipeline
// orchestrator: default_config() structure + the source-type classifier.
// No models needed (classifier is pure pixel heuristics).
//
// Usage: test-ocr-orchestrator   (synthetic; exits non-zero on failure)

#include "ocr_orchestrator.h"

// stbi_write_png is exported by the crispembed lib (ocr_orchestrator.cpp owns
// STB_IMAGE_WRITE_IMPLEMENTATION); just declare it here to avoid duplicate syms.
extern "C" int stbi_write_png(char const* filename, int w, int h, int comp,
                              const void* data, int stride_in_bytes);

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// Write an RGB(A) buffer to a temp PNG; returns the path.
static std::string write_temp(const std::vector<uint8_t>& px, int w, int h, int ch,
                              const char* name) {
    const char* dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    std::string path = std::string(dir) + "/orch_test_" + name + ".png";
    stbi_write_png(path.c_str(), w, h, ch, px.data(), w * ch);
    return path;
}

int main() {
    using namespace ocr_orchestrator;

    // ── default_config() structure ──────────────────────────────────────────
    {
        config cfg = default_config();
        CHECK(cfg.router, "default_config: router on");
        CHECK(!cfg.chains.empty(), "default_config: has chains");
        bool has_auto = false, has_scan = false, has_photo = false, has_shot = false;
        bool all_dbnet = true;
        for (auto& c : cfg.chains) {
            if (c.type == source_type::auto_detect)  has_auto = true;
            if (c.type == source_type::scanned_doc)  has_scan = true;
            if (c.type == source_type::photo)        has_photo = true;
            if (c.type == source_type::screenshot)   has_shot = true;
            for (auto& s : c.stages)
                if (s.eng != engine::dbnet_trocr) all_dbnet = false;
        }
        CHECK(has_auto && has_scan && has_photo && has_shot,
              "default_config: chains for auto/scan/photo/screenshot");
        CHECK(all_dbnet, "default_config: default stages are dbnet_trocr");
        // Per-source cleanup intent: scanned-doc binarizes, photo denoises.
        for (auto& c : cfg.chains) {
            if (c.type == source_type::scanned_doc && !c.stages.empty())
                CHECK(c.stages[0].cleanup.params.binarize == 1,
                      "scanned_doc chain binarizes");
            if (c.type == source_type::photo && !c.stages.empty())
                CHECK(c.stages[0].cleanup.denoise,
                      "photo chain denoises (NAFNet)");
        }
    }

    // ── classify_file() heuristics ───────────────────────────────────────────
    {
        // Colourful image → photo (mean saturation high).
        std::vector<uint8_t> photo(64 * 64 * 3);
        for (int i = 0; i < 64 * 64; i++) {
            photo[i*3+0] = 200; photo[i*3+1] = 30; photo[i*3+2] = 20; // saturated red
        }
        std::string p = write_temp(photo, 64, 64, 3, "photo");
        CHECK(classify_file(p.c_str()) == source_type::photo,
              "classify: saturated image → photo");

        // White page with sparse black lines → scanned_doc (low sat, high white).
        std::vector<uint8_t> doc(80 * 80 * 3, 255);
        for (int y = 20; y < 80; y += 20)
            for (int x = 0; x < 80; x++)
                for (int ch = 0; ch < 3; ch++) doc[(y*80+x)*3+ch] = 0;
        std::string d = write_temp(doc, 80, 80, 3, "doc");
        CHECK(classify_file(d.c_str()) == source_type::scanned_doc,
              "classify: white page w/ lines → scanned_doc");

        // Very wide grayscale strip → screenshot (aspect > 2.2).
        std::vector<uint8_t> wide(300 * 50 * 3, 240);
        std::string w = write_temp(wide, 300, 50, 3, "wide");
        CHECK(classify_file(w.c_str()) == source_type::screenshot,
              "classify: very wide image → screenshot");

        // Missing file → defaults to scanned_doc (no crash).
        CHECK(classify_file("/no/such/file.png") == source_type::scanned_doc,
              "classify: missing file → scanned_doc fallback");
    }

    printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
