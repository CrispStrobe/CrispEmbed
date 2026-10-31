// Public C API controls with an intentionally fake low-level inference engine.
// Exercises ownership, validation, JSON and failure propagation, not model math.
#include "crispembed.h"
#include "oxr_ocr.h"
#include "core/clean_exit.h"
#include "core/json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

struct oxr_ocr_context {
    std::string prompt, error;
    int max_tokens = 2048;
    bool ok = false;
};
static int calls = 0, frees = 0, checks = 0, failures = 0;
static void check(bool condition, const char * label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    }
}
static void gate(const char * value) {
#if defined(_WIN32)
    _putenv_s("CRISPEMBED_OXR", value);
#else
    setenv("CRISPEMBED_OXR", value, 1);
#endif
}
static std::string string_field(const std::string & json, const char * key) {
    const size_t start = core_json::json_find_key_value(json, key);
    size_t end = start;
    std::string result;
    return core_json::json_decode_string(json, start, result, end) ? result : std::string();
}

extern "C" {
oxr_ocr_context * oxr_ocr_init(const char *, int) {
    return new oxr_ocr_context;
}
void oxr_ocr_free(oxr_ocr_context * ctx) {
    if (ctx) ++frees;
    delete ctx;
}
int oxr_ocr_set_prompt(oxr_ocr_context * ctx, const char * prompt) {
    ctx->prompt = prompt;
    return 1;
}
int oxr_ocr_set_repetition_penalty(oxr_ocr_context *, float) {
    return 1;
}
void oxr_ocr_set_max_tokens(oxr_ocr_context * ctx, int count) {
    ctx->max_tokens = count ? count : 2048;
}
const char * oxr_ocr_recognize_raw(oxr_ocr_context * ctx, const uint8_t *, int, int, int, int * length) {
    ++calls;
    ctx->ok = ctx->prompt != "fail";
    if (!ctx->ok) {
        ctx->error = "deliberate engine failure";
        return nullptr;
    }
    ctx->error.clear();
    static const std::string text = "ok\n\"quoted\"";
    *length = int(text.size());
    return text.c_str();
}
int oxr_ocr_last_ok(const oxr_ocr_context * ctx) {
    return ctx->ok;
}
const char * oxr_ocr_last_error(const oxr_ocr_context * ctx) {
    return ctx->error.c_str();
}
int oxr_ocr_stopped_eos(const oxr_ocr_context * ctx) {
    return ctx->max_tokens > 1;
}
int oxr_ocr_capped(const oxr_ocr_context * ctx) {
    return ctx->max_tokens == 1;
}
float oxr_ocr_mean_confidence(const oxr_ocr_context *) {
    return .9f;
}
const char * oxr_ocr_raw_text(const oxr_ocr_context *) {
    return "ok\n\"quoted\"<|im_end|>";
}
const int32_t * oxr_ocr_prompt_ids(const oxr_ocr_context *, int * count) {
    static const int32_t ids[] = { 151644, -200, 151645 };
    *count = 3;
    return ids;
}
const int32_t * oxr_ocr_completion_ids(const oxr_ocr_context * ctx, int * count) {
    static const int32_t ids[] = { 42, 151645 };
    *count = ctx->max_tokens == 1 ? 1 : 2;
    return ids;
}
const int32_t * oxr_ocr_grid_thw(const oxr_ocr_context *) {
    static const int32_t grid[] = { 1, 28, 28 };
    return grid;
}
int oxr_ocr_effective_max_tokens(const oxr_ocr_context * ctx) {
    return ctx->max_tokens;
}
}

static int execute() {
    gate("0");
    check(crispembed_oxr_init("fake.gguf", 1) == nullptr, "literal zero gate is off");
    gate("1");
    check(crispembed_oxr_init(nullptr, 1) == nullptr, "NULL model rejected");
    auto * ctx = crispembed_oxr_init("fake.gguf", 1);
    check(ctx != nullptr, "context owns initialized engine");
    const uint8_t pixel[] = { 1, 2, 3 };
    auto predict = [&](const char * prompt, int max_tokens = 3, float penalty = 1.f) {
        return crispembed_oxr_predict(ctx, pixel, 1, 1, 3, prompt, max_tokens, penalty);
    };
    const char * result = predict("task\n\"one\"");
    check(result != nullptr, "valid image request succeeds");
    std::string json = result ? result : "";
    check(string_field(json, "text") == "ok\n\"quoted\"", "JSON text escapes roundtrip");
    check(string_field(json, "prompt") == "task\n\"one\"", "JSON prompt escapes roundtrip");
    check(core_json::json_extract_bool(json, "eos", false), "EOS flag retained");
    check(!core_json::json_extract_bool(json, "capped", true), "complete result not capped");
    check(string_field(json, "finish_reason") == "stop", "finish reason stop");
    check(json.find("-200") != std::string::npos, "negative image placeholder retained");
    result = predict("task", 1);
    check(result && core_json::json_extract_bool(result, "capped", false), "partial output cap retained");
    check(result && string_field(result, "finish_reason") == "length", "finish reason length");
    result = predict("task", 0);
    check(result && core_json::json_extract_number(result, "effective_max_tokens", 0) == 2048,
          "zero default budget recorded");
    const int before = calls;
    check(predict(nullptr) == nullptr, "NULL prompt rejected");
    check(predict("task", -1) == nullptr, "negative cap rejected");
    check(predict("task", 32769) == nullptr, "excess cap rejected");
    check(predict("task", 3, 0.f) == nullptr, "zero penalty rejected");
    check(predict("task", 3, std::numeric_limits<float>::infinity()) == nullptr, "infinite penalty rejected");
    check(crispembed_oxr_predict(ctx, pixel, 1, 1, 2, "task", 3, 1.f) == nullptr, "two-channel input rejected");
    check(crispembed_oxr_predict(ctx, pixel, 100000001, 1, 3, "task", 3, 1.f) == nullptr, "oversize image rejected");
    check(crispembed_oxr_predict(ctx, nullptr, 1, 1, 3, "task", 3, 1.f) == nullptr,
          "NULL pixels with dimensions rejected");
    check(predict(std::string(65537, 'a').c_str()) == nullptr, "oversize prompt rejected");
    check(calls == before, "validation prevents engine calls");
    check(predict("fail") == nullptr, "engine failure propagates");
    check(std::string(crispembed_oxr_last_error(ctx)) == "deliberate engine failure", "engine error retained");
    check(crispembed_oxr_predict(ctx, nullptr, 0, 0, 0, "text-only", 3, 1.f) != nullptr, "text-only request supported");
    check(std::string(crispembed_oxr_last_error(ctx)).empty(), "next success clears previous error");
    gate("0");
    check(predict("task") == nullptr, "gate checked for each request");
    crispembed_oxr_free(ctx);
    crispembed_oxr_free(nullptr);
    check(frees == 1, "model freed exactly once");
    check(crispembed_oxr_predict(nullptr, pixel, 1, 1, 3, "task", 3, 1.f) == nullptr, "NULL context safe");
    std::printf("OXR public API controls: %d checks, %d failures (mock inference)\n", checks, failures);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(execute());
}
