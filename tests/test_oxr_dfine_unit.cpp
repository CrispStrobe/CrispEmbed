// Model-free checks for the OXR D-FINE layout path: Python float repr used by
// the service-format JSON, the OpenCV Lanczos4 port's invariants, OXR's
// client-side selection (threshold, clip, class-wise NMS, order), the OXR ->
// canonical label table, and the license gates of the C API.
#include "core/clean_exit.h"
#include "core/opencv_lanczos4_u8.h"
#include "core/python_float_repr.h"
#include "crispembed.h"
#include "layout_detect.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                       \
            failures++;                                                                                                \
        }                                                                                                              \
    } while (0)

static int crispembed_test_main() {
    using core_pyrepr::python_float;
    // repr(float(...)) as CPython prints it.
    CHECK(python_float(0.5) == "0.5");
    CHECK(python_float(3.0) == "3.0");
    CHECK(python_float(-0.0) == "-0.0");
    CHECK(python_float(1e-5) == "1e-05");
    CHECK(python_float(0.0001) == "0.0001");
    CHECK(python_float(1.2345e-7) == "1.2345e-07");
    CHECK(python_float(1e16) == "1e+16");
    CHECK(python_float(123456789012345.0) == "123456789012345.0");
    CHECK(python_float((double)0.9643554688f) == "0.96435546875");
    CHECK(python_float(959.0) == "959.0");
    CHECK(python_float(-12.5) == "-12.5");

    // Lanczos4: identical size copies; a constant image stays constant.
    std::vector<uint8_t> src(7 * 5 * 3, 77), dst(13 * 9 * 3, 0);
    core_cv::opencv_resize_lanczos4_u8(src.data(), 7, 5, 3, dst.data(), 13, 9);
    bool constant = true;
    for (auto v : dst) constant &= v == 77;
    CHECK(constant);
    std::vector<uint8_t> same(src.size());
    core_cv::opencv_resize_lanczos4_u8(src.data(), 7, 5, 3, same.data(), 7, 5);
    CHECK(same == src);

    // OXR selection: threshold, degenerate rejection, clip, class-wise NMS, order kept.
    using layout_detect::candidate;
    std::vector<candidate> c = {
        { 3, 0.90f, { 10, 10, 110, 60 } },   // Text, kept
        { 3, 0.80f, { 12, 11, 112, 61 } },   // Text, suppressed by the first (IoU > 0.6)
        { 8, 0.70f, { 12, 11, 112, 61 } },   // Table, other class: kept
        { 7, 0.40f, { 0, 0, 50, 50 } },      // below 0.5
        { 6, 0.60f, { -20, -5, 30, 2000 } }, // clipped to the page
        { 10, 0.95f, { 5, 5, 6, 6 } },       // Stamp, non-canonical
    };
    auto r = layout_detect::oxr_select(c, 200, 300, 0.5f, 0.6f);
    CHECK(r.size() == 4);
    if (r.size() == 4) {
        CHECK(r[0].raw_class == 3 && r[1].raw_class == 8 && r[2].raw_class == 6 && r[3].raw_class == 10);
        CHECK(r[2].x1 == 0 && r[2].y1 == 0 && r[2].y2 == 300);
        CHECK(std::string(r[0].raw_label) == "Text" && r[0].label == layout_detect::label_id::text);
        CHECK(std::string(r[1].label_name) == "table" && r[1].canonical_exact);
        CHECK(std::string(r[3].raw_label) == "Stamp" && r[3].label == layout_detect::label_id::picture &&
              !r[3].canonical_exact);
    }
    CHECK(layout_detect::oxr_select(c, 200, 300, 0.5f, -1.0f).size() == 5); // NMS disabled

    bool exact = true;
    CHECK(layout_detect::oxr_canonical_label(9, &exact) == layout_detect::label_id::code && exact);
    CHECK(layout_detect::oxr_canonical_label(11, &exact) == layout_detect::label_id::picture && !exact);
    CHECK(std::string(layout_detect::oxr_label_name(11)) == "Chemical Structure");
    CHECK(layout_detect::oxr_label_name(12) == nullptr);

    // Gates: the C API refuses without both OXR variables, before touching files.
#ifdef _WIN32
    _putenv_s("CRISPEMBED_OXR", "");
#else
    unsetenv("CRISPEMBED_OXR");
#endif
    CHECK(crispembed_oxr_layout_init("missing.gguf", 1) == nullptr);
    CHECK(crispembed_oxr_layout_detect(nullptr, nullptr, 0, 0, 0, -1) == nullptr);
    layout_detect::context * ctx = nullptr;
    CHECK(!layout_detect::load(&ctx, "missing-oxr-dfine.gguf", 1));
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
