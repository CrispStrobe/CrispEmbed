// Hosted admission/state check. No model forward or synthetic reference input.
#include "core/clean_exit.h"
#include "crispembed.h"
#include "qwen35_xiaomi0.h"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "1");
    gate("CRISPEMBED_XIAOMI_OCR0_INFERENCE", "0");
    using owner = std::unique_ptr<void, decltype(&crispembed_ocr_model_free)>;
    const auto rejects = [&]() {
        owner rejected(crispembed_ocr_model_init(argv[1], 2), crispembed_ocr_model_free);
        return !rejected;
    };
    bool ok = rejects();
    gate("CRISPEMBED_XIAOMI_OCR0_INFERENCE", "1");
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "0");
    ok &= rejects();
    gate("CRISPEMBED_XIAOMI_OCR0_FRONTEND", "1");
    owner model(crispembed_ocr_model_init(argv[1], 2), crispembed_ocr_model_free);
    if (!model) return 1;
    ok &= crispembed_ocr_model_get_max_tokens(model.get()) == 4096;
    crispembed_ocr_model_set_max_tokens(model.get(), 16);
    ok &= crispembed_ocr_model_get_max_tokens(model.get()) == 16;
    crispembed_ocr_model_set_max_tokens(model.get(), -1);
    ok &= crispembed_ocr_model_get_max_tokens(model.get()) == 16;
    crispembed_ocr_model_set_max_tokens(model.get(), 0);
    ok &= crispembed_ocr_model_get_max_tokens(model.get()) == 4096;
    using namespace qwen35_ocr::xiaomi0_profile;
    for (const char * prompt : { text, table, formula, document })
        ok &= crispembed_ocr_model_set_prompt(model.get(), prompt) == 1;
    ok &= crispembed_ocr_model_set_prompt(model.get(), "invented task") == 0;
    ok &= crispembed_ocr_model_set_prompt(model.get(), nullptr) == 1;
    ok &= crispembed_ocr_model_set_prompt(model.get(), "") == 1;
    int count = 123;
    ok &= !crispembed_ocr_model_confidences(model.get(), &count) && count == 0;
    std::printf("%s XIAOMI0_PUBLIC_ADMISSION no_forward=1 cap=4096 exact_tasks=4\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
