#include "core/clean_exit.h"
#include "qwen35_frontend.h"
#include "qwen35_image_bounds.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

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
// The admitted patch bound is profile- and gate-dependent, so it must be
// re-derived whenever a gate changes. A plain two-field aggregate initializer
// would silently take the ordinary 1024-patch default.
static qwen35_ocr::image_config ovis_config() {
    qwen35_ocr::image_config config{ 448 * 448, 2880 * 2880 };
    config.max_patches = qwen35_ocr::ovisocr2_patch_limit();
    return config;
}
static int run() {
    const char * gates[] = { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR2_FRONTEND", "CRISPEMBED_OVISOCR2_INFERENCE",
                             "CRISPEMBED_OVISOCR2_LARGE_IMAGES" };
    for (const char * name : gates) gate(name, "1");
    bool ok = qwen35_ocr::ovisocr2_patch_limit() == 2048;
    for (const char * name : gates) {
        for (const char * value : { "", "0" }) {
            gate(name, value);
            ok &= qwen35_ocr::ovisocr2_patch_limit() == 1024;
        }
        gate(name, "1");
    }
    qwen35_ocr::image_inputs output;
    ok &= ovis_config().max_patches == 2048;
    std::vector<uint8_t> pixels(size_t(1024) * 544 * 3, 255);
    gate(gates[3], "0");
    ok &= ovis_config().max_patches == 1024;
    ok &= !qwen35_ocr::preprocess_rgb(pixels.data(), 480, 640, ovis_config(), output) && output.patches.empty();
    gate(gates[3], "1");
    ok &= ovis_config().max_patches == 2048;
    ok &= qwen35_ocr::preprocess_rgb(pixels.data(), 480, 640, ovis_config(), output) && output.height == 480 &&
          output.width == 640 && output.grid == std::vector<int>({ 1, 30, 40 });
    ok &= qwen35_ocr::preprocess_rgb(pixels.data(), 512, 1024, ovis_config(), output) &&
          output.grid == std::vector<int>({ 1, 32, 64 });
    ok &= !qwen35_ocr::preprocess_rgb(pixels.data(), 544, 1024, ovis_config(), output) && output.patches.empty();
    // The generic LightOn configuration keeps its old bound even when Ovis
    // experimental flags happen to be present in the process environment.
    ok &= !qwen35_ocr::preprocess_rgb(pixels.data(), 480, 640, {}, output) && output.patches.empty();
    gate(gates[3], "0");
    ok &= qwen35_ocr::preprocess_rgb(pixels.data(), 512, 512, ovis_config(), output) &&
          output.grid == std::vector<int>({ 1, 32, 32 });
    std::printf("%s OVISOCR2_IMAGE_BOUNDS default1024=1 optin2048=1 literal0=1 no_resize=1 reject2176=1 recovery=1\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main() {
    core_util::clean_exit(run());
}
