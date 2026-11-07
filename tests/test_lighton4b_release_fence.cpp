#include "qwen35_ocr.h"
#include "gguf.h"
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "core/lighton4b_release_fence.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
static void gate(const char * n, const char * v) {
#ifdef _WIN32
    _putenv_s(n, v);
#else
    setenv(n, v, 1);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "release fence failed=%d\n", checks);
            std::exit(1);
        }
    };
    auto * metadata = gguf_init_empty();
    check(metadata != nullptr);
    auto profile = [&](const char * arch, uint32_t version, const char * name, const char * recipe) {
        gguf_set_val_str(metadata, "general.architecture", arch);
        gguf_set_val_u32(metadata, "qwen35_ocr.format_version", version);
        gguf_set_val_str(metadata, "qwen35_ocr.profile", name);
        gguf_set_val_str(metadata, "qwen35_ocr.mixed_quant_recipe", recipe);
    };
    check(!core_lighton4b_release_fence::blocked(nullptr));
    // Existing supported custom formats and unrelated legacy architectures stay unchanged.
    const char * names[] = { "", "ovisocr2", "ovisocr3", "xiaomi-ocr0" };
    for (uint32_t version = 1; version <= 4; ++version) {
        profile("qwen35_ocr", version, names[version - 1], "");
        check(!core_lighton4b_release_fence::blocked(metadata));
    }
    for (const char * arch : { "pix2tex_mfr", "qwen2vl", "qwen3vl", "lightonocr", "legacy-unknown" }) {
        profile(arch, 6, "lightonocr3-4b", "");
        check(!core_lighton4b_release_fence::blocked(metadata));
    }
    for (const char * value : { "0", "1" }) {
        for (const char * name :
             { "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_FRONTEND",
               "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_LIGHTON4B_Q6_K", "CRISPEMBED_LIGHTON4B_Q4_K_M",
               "CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD", "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL" })
            gate(name, value);
        gate("CRISPEMBED_QWEN35_OCR", "1");
        for (uint32_t version : { 0u, 1u, 5u, 6u }) {
            profile("qwen35", version, "", "");
            check(core_lighton4b_release_fence::blocked(metadata));
        }
        for (uint32_t version : { 5u, 6u }) {
            profile("qwen35_ocr", version, "lightonocr3-4b", version == 6 ? "lighton4b-upstream-q4km-physical-v1" : "");
            check(core_lighton4b_release_fence::blocked(metadata));
            gguf_set_val_bool(metadata, "qwen35_ocr.tie_word_embeddings", true);
            gguf_set_val_bool(metadata, "qwen35_ocr.zero_centered_text_norm", version == 5);
            check(gguf_write_to_file(metadata, argv[1], true));
            auto * valid = core_gguf::open_metadata(argv[1]);
            check(valid && core_gguf::kv_u32(valid, "qwen35_ocr.format_version", 0) == version &&
                  core_lighton4b_release_fence::blocked(valid));
            core_gguf::free_metadata(valid);
            auto * context = qwen35_ocr_init(argv[1], 1);
            check(context == nullptr);
            if (context) qwen35_ocr_free(context);
        }
        profile("qwen35_ocr", 1, "lightonocr3-4b", "");
        check(core_lighton4b_release_fence::blocked(metadata));
        profile("qwen35_ocr", 1, "", "lighton4b-upstream-q4km-physical-v1");
        check(core_lighton4b_release_fence::blocked(metadata));
    }
    gguf_free(metadata);
    std::printf("LIGHTON4B_RELEASE_FENCE passed=1 checks=%d metadata_only_public_rejections=4 "
                "raw_qwen35_early_rejection=1 supported_formats_unchanged=1 model_payload=0\n",
                checks);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
