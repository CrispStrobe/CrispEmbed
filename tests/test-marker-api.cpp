// SPDX-License-Identifier: MIT
#include "crispembed.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
int checks = 0;
void check(bool ok, const char * label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        core_util::clean_exit(1);
    }
}
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_MARKER", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_MARKER", value, 1);
    else
        unsetenv("CRISPEMBED_MARKER");
#endif
}
} // namespace
static int crispembed_test_main(int argc, char ** argv) {
    auto c = crispembed_marker_default_config();
    check(c.struct_size == sizeof(c), "versioned config ABI");
    check(c.mode == 0 && c.last_page == -1 && c.extract_images == 1, "config defaults");
    c.mode = 1;
    // Shared core_env::on semantics: only unset, empty and literal 0 are off.
    for (const char * value : { static_cast<const char *>(nullptr), "0", "" }) {
        gate(value);
        check(crispembed_marker_init(&c) == nullptr, "default-off gate values");
        check(std::strstr(crispembed_marker_error(nullptr), "CRISPEMBED_MARKER") != nullptr, "gate diagnostic");
    }
    gate("1");
    check(crispembed_marker_init(nullptr) == nullptr, "null config");
    auto invalid = c;
    invalid.struct_size -= 1;
    check(crispembed_marker_init(&invalid) == nullptr, "config size guard");
    invalid = c;
    invalid.mode = 99;
    check(crispembed_marker_init(&invalid) == nullptr, "mode guard");
    invalid = c;
    invalid.n_threads = 0;
    check(crispembed_marker_init(&invalid) == nullptr, "thread guard");
    invalid = c;
    invalid.first_page = 5;
    invalid.last_page = 4;
    check(crispembed_marker_init(&invalid) == nullptr, "page range guard");
    invalid = c;
    invalid.max_pixels = UINT64_MAX;
    check(crispembed_marker_init(&invalid) == nullptr, "pixel overflow guard");
    invalid = c;
    invalid.max_tokens = 8193;
    check(crispembed_marker_init(&invalid) == nullptr, "token cap guard");
    invalid = c;
    invalid.include_page_headers = 2;
    check(crispembed_marker_init(&invalid) == nullptr, "boolean guard");
    invalid = c;
    invalid.mode = 0;
    check(crispembed_marker_init(&invalid) == nullptr, "missing model guard");
    auto * ctx = crispembed_marker_init(&c);
    check(ctx != nullptr, "model-free context");
    check(crispembed_marker_render(ctx, "json") == nullptr, "render requires conversion");
    check(crispembed_marker_metadata(ctx) == nullptr, "metadata requires conversion");
    check(crispembed_marker_convert(ctx, nullptr) == 0, "null file guard");
    check(crispembed_marker_convert(ctx, "missing-marker-test-file.pdf") == 0, "missing input guard");
    uint8_t pixels[3] = { 255, 255, 255 };
    check(crispembed_marker_convert_raw(ctx, pixels, 1, 1, 3) == 0, "digital mode rejects raster");
    size_t length = 7;
    check(crispembed_marker_asset_data(ctx, 0, &length) == nullptr && length == 0, "empty asset guard");
    check(crispembed_marker_asset_name(ctx, -1) == nullptr, "negative asset index");
    check(crispembed_marker_convert(nullptr, "x") == 0, "null conversion context");
    crispembed_marker_free(ctx);
    crispembed_marker_free(nullptr);
    if (argc == 3) {
        c.pdfium_path = argv[1];
        ctx = crispembed_marker_init(&c);
        check(ctx != nullptr, "PDF context");
        if (!crispembed_marker_convert(ctx, argv[2])) {
            std::fprintf(stderr, "%s\n", crispembed_marker_error(ctx));
            check(false, "PDF conversion");
        }
        check(true, "PDF conversion");
        for (const char * format : { "markdown", "html", "json", "chunks" }) {
            const char * s = crispembed_marker_render(ctx, format);
            check(s && *s, "actual PDF renderer");
        }
        const char * meta = crispembed_marker_metadata(ctx);
        check(meta && std::strstr(meta, "crispembed-marker-native"), "actual provenance");
        check(crispembed_marker_render(ctx, "bad-format") == nullptr, "bad renderer rejected");
        check(crispembed_marker_render(ctx, "markdown") != nullptr, "render recovery");
        gate("0");
        check(!crispembed_marker_convert(ctx, argv[2]), "gate checked on reuse");
        check(crispembed_marker_render(ctx, "json") == nullptr, "failed conversion clears old output");
        gate("1");
        check(crispembed_marker_convert(ctx, argv[2]) == 1, "conversion recovery");
        crispembed_marker_free(ctx);
    }
    std::printf("Marker C API: %d checks passed\n", checks);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
