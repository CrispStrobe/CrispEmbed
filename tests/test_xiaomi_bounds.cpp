#include "qwen35_frontend.h"
#include "qwen35_image_bounds.h"
#include "qwen35_vision.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

// Weight-free frontend/SDK bounds for the Xiaomi captures. Nothing here runs a
// model, so these assertions cannot substitute for the hosted native build.
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
static const char * kGates[] = { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_XIAOMI_OCR_FRONTEND",
                                 "CRISPEMBED_XIAOMI_OCR_INFERENCE" };

static bool preprocess(int height, int width, const qwen35_ocr::image_config & config, qwen35_ocr::image_inputs & out) {
    std::vector<uint8_t> rgb(size_t(height) * width * 3, 200);
    return qwen35_ocr::preprocess_rgb(rgb.data(), height, width, config, out);
}

static int run() {
    bool ok = true;
#define CHECK(label, expr)                                                                                             \
    do {                                                                                                               \
        const bool value = (expr);                                                                                     \
        if (!value) std::printf("FAIL %s\n", label);                                                                   \
        ok = ok && value;                                                                                              \
    } while (0)
    // Defaults are unchanged and independent of the Xiaomi gates.
    CHECK("constants", qwen35_ocr::kOrdinaryPatchLimit == 1024 && qwen35_ocr::kXiaomiPatchLimit == 9184);
    for (const char * name : kGates) gate(name, nullptr);
    CHECK("ungated_limit", qwen35_ocr::xiaomi_patch_limit() == 1024 && !qwen35_ocr::xiaomi_patch_limit_enabled());
    CHECK("default_image_config", qwen35_ocr::image_config{}.max_patches == 1024);

    // Partial Xiaomi gate sets must not widen the bound. The master OCR gate is
    // separate: it is enforced by the frontend, not by the limit helper.
    const char * xiaomi_gates[] = { "CRISPEMBED_XIAOMI_OCR_FRONTEND", "CRISPEMBED_XIAOMI_OCR_INFERENCE" };
    for (const char * name : xiaomi_gates) {
        for (const char * value : { "", "0" }) {
            for (const char * other : kGates) gate(other, nullptr);
            for (const char * other : kGates) gate(other, "1");
            gate(name, value);
            CHECK("partial_gate", qwen35_ocr::xiaomi_patch_limit() == 1024);
        }
    }

    for (const char * name : kGates) gate(name, "1");
    CHECK("gated_limit", qwen35_ocr::xiaomi_patch_limit() == 9184 && qwen35_ocr::xiaomi_patch_limit_enabled());
    // Ovis keeps its own four-gate 1024/2048 policy.
    CHECK("ovis_isolated", qwen35_ocr::ovisocr2_patch_limit() == 1024);

    const qwen35_ocr::image_config limited{ 65536, 16777216, qwen35_ocr::xiaomi_patch_limit() };
    qwen35_ocr::image_inputs output;
    // Captured pages: qianziwen 1984, receipt 5916, dense equations 9184 patches.
    struct case_row {
        const char * name;
        int height, width, rh, rw;
        int patches;
    } rows[] = { { "qianziwen", 510, 979, 512, 992, 1984 },
                 { "receipt", 932, 1644, 928, 1632, 5916 },
                 { "dense", 1800, 1303, 1792, 1312, 9184 } };
    for (const auto & row : rows) {
        const bool admitted = preprocess(row.height, row.width, limited, output);
        const bool shape = admitted && output.height == row.rh && output.width == row.rw &&
                           output.grid == std::vector<int>({ 1, row.rh / 16, row.rw / 16 }) &&
                           output.patches.size() == size_t(row.patches) * 1536;
        std::printf("%s XIAOMI_PATCH %s admitted=%d patches=%zu grid=%dx%d\n", shape ? "PASS" : "FAIL", row.name,
                    int(admitted), output.patches.size() / 1536, row.rh / 16, row.rw / 16);
        ok &= shape;
        // The ordinary configuration must reject the same page rather than resize it.
        ok &= !preprocess(row.height, row.width, qwen35_ocr::image_config{}, output) && output.patches.empty();
    }
    // One patch above the captured maximum is refused, not downscaled. 1920x1312
    // needs 1920*1312/256 = 9840 unmerged patches.
    CHECK("reject_9185", !preprocess(1920, 1312, limited, output) && output.patches.empty());
    // A configuration may not ask for a bound outside the audited range.
    CHECK("reject_zero", !preprocess(510, 979, qwen35_ocr::image_config{ 65536, 16777216, 0 }, output));
    CHECK("reject_over_max", !preprocess(510, 979, qwen35_ocr::image_config{ 65536, 16777216, 9185 }, output));

    // Redocumented vision contracts: a default config admits 1024 only.
    CHECK("default_vision_config", qwen35_ocr::vision_config{}.max_patches == 1024);
    qwen35_ocr::vision_config widened = qwen35_ocr::vision_config{};
    widened.max_patches = qwen35_ocr::kXiaomiPatchLimit;
    CHECK("widened_vision_config", widened.max_patches == 9184);

    for (const char * name : kGates) gate(name, nullptr);
    CHECK("restored_limit", qwen35_ocr::xiaomi_patch_limit() == 1024);
    // The ordinary configuration must reject each capture rather than resize it.
    for (const auto & row : rows)
        CHECK("ordinary_rejects",
              !preprocess(row.height, row.width, qwen35_ocr::image_config{}, output) && output.patches.empty());
    std::printf("%s XIAOMI_BOUNDS default1024=1 gated9184=1 qianziwen=1 receipt=1 dense=1 reject9185=1 "
                "ordinary_rejects=1 ovis_isolated=1\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main() {
    return run();
}
