#include "core/clean_exit.h"
#include "crispembed.h"
#include "crispembed_diff.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
extern "C" unsigned char * stbi_load(const char *, int *, int *, int *, int);
extern "C" void stbi_image_free(void *);
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
static bool compare(const char * label, void * ctx, const char * text, int length, const crispembed_diff::Ref & ref) {
    const auto bytes = ref.get_f32("input.decoded_utf8"), ids = ref.get_f32("generated.ids");
    if (!bytes.first || !ids.first || !text || length < 0) return false;
    std::string expected;
    for (size_t i = 0; i < bytes.second; ++i) expected.push_back(char(bytes.first[i]));
    int n = 0;
    const auto * conf = crispembed_ocr_model_confidences(ctx, &n);
    bool ok = std::string(text, size_t(length)) == expected && conf && n == int(ids.second);
    double max_error = 0;
    if (ok)
        for (int i = 0; i < n; ++i) {
            auto scores = ref.get_f32("generated.scores." + std::to_string(i));
            if (!scores.first || scores.second != 248320) return false;
            const float best = *std::max_element(scores.first, scores.first + scores.second);
            double total = 0;
            for (size_t j = 0; j < scores.second; ++j) total += std::exp(double(scores.first[j]) - best);
            max_error = std::max(max_error, std::abs(double(conf[i]) - 1 / total));
        }
    ok &= max_error <= 1e-6;
    std::printf("%s PUBLIC %s bytes=%d scores=%d confidence_max_abs=%.9g mean=%.9g\n", ok ? "PASS" : "FAIL", label,
                length, n, max_error, crispembed_ocr_model_mean_confidence(ctx));
    return ok;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 6) return 2;
    std::printf("ABI stage_size=%zu cleanup_size=%zu params_size=%zu vlm_max_tokens_offset=%zu vlm_prompt_offset=%zu\n",
                sizeof(crispembed_ocr_stage), sizeof(crispembed_scan_cleanup_params),
                sizeof(crispembed_ocr_pipeline_params), offsetof(crispembed_ocr_stage, vlm_max_tokens),
                offsetof(crispembed_ocr_stage, vlm_prompt));
    gate("0");
    bool ok = !crispembed_ocr_model_init(argv[1], 4);
    gate("1");
    crispembed_diff::Ref ref, prompted;
    if (!ref.load(argv[3]) || !prompted.load(argv[5])) return 2;
    using owner = std::unique_ptr<void, decltype(&crispembed_ocr_model_free)>;
    owner model(crispembed_ocr_model_init(argv[1], 4), crispembed_ocr_model_free);
    if (!model) return 1;
    int w = 0, h = 0, c = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(stbi_load(argv[2], &w, &h, &c, 3),
                                                                      stbi_image_free);
    if (!pixels) return 2;
    crispembed_ocr_model_set_max_tokens(model.get(), 12);
    ok &= crispembed_ocr_model_get_max_tokens(model.get()) == 12;
    int len = 0;
    const char * text = crispembed_ocr_model_recognize(model.get(), pixels.get(), w, h, 3, &len);
    ok &= compare("raw-autodetect", model.get(), text, len, ref);
    std::vector<float> gray(size_t(w) * h);
    for (size_t i = 0; i < gray.size(); ++i) gray[i] = pixels.get()[3 * i] / 255.0f;
    text = crispembed_ocr_model_recognize_gray(model.get(), gray.data(), w, h, &len);
    ok &= compare("float-gray", model.get(), text, len, ref);
    ok &= crispembed_ocr_model_set_prompt(model.get(), "Transcribe the text.");
    text = crispembed_ocr_model_recognize(model.get(), pixels.get(), w, h, 3, &len);
    ok &= compare("custom-prompt", model.get(), text, len, prompted);
    ok &= crispembed_ocr_model_set_prompt(model.get(), nullptr);
    crispembed_ocr_model_set_max_tokens(model.get(), 2);
    text = crispembed_ocr_model_recognize(model.get(), pixels.get(), w, h, 3, &len);
    ok &= text && std::string(text, size_t(len)) == "Hello OCR";
    len = 99;
    gray[0] = NAN;
    text = crispembed_ocr_model_recognize_gray(model.get(), gray.data(), w, h, &len);
    int n = 99;
    ok &= !text && len == 0 && !crispembed_ocr_model_confidences(model.get(), &n) && n == 0;
    crispembed_ocr_model_set_max_tokens(model.get(), 12);
    text = crispembed_ocr_model_recognize(model.get(), pixels.get(), w, h, 3, &len);
    ok &= compare("recovered-prompt-reset", model.get(), text, len, ref);
    model.reset();
    crispembed_ocr_stage stage{};
    stage.source_type = 1;
    stage.engine = 20;
    stage.model_a = argv[1];
    stage.vlm_max_tokens = 12;
    stage.min_chars = 1;
    auto * pipeline =
        crispembed_ocr_pipeline_init_stages(0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &stage, 1, 4);
    if (!pipeline) return 1;
    const char * full = nullptr;
    float mean = 0;
    auto * regions = crispembed_ocr_pipeline_run(pipeline, argv[2], &n, &full, &mean);
    int metrics_count = 0;
    const auto * metrics = crispembed_ocr_pipeline_stage_metrics(pipeline, &metrics_count);
    ok &= regions && n == 1 && full && std::string(full) == "Hello OCR  \n12345" && mean > 0.99f && metrics &&
          metrics_count == 1 && std::string(metrics[0].engine) == "qwen35_ocr";
    std::printf("%s PUBLIC orchestrator-id20 regions=%d mean=%.9g\n", ok ? "PASS" : "FAIL", n, mean);
    crispembed_ocr_pipeline_free(pipeline);
    std::printf("%s unified public ABI, prompt/reset/cap/gray/confidence, orchestrator ID20 and master=0\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
