// OXR-1.0 owning public API. See licenses/oxr-LICENSE and oxr-NOTICE.
#include "crispembed.h"
#include "core/env_gate.h"
#include "core/json.h"
#include "oxr_api_internal.h"
#include "oxr_ocr.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::string quote(const std::string & value) {
    return "\"" + core_json::json_escape_strict(value) + "\"";
}
template <typename T> void write_array(std::ostream & out, const T * values, int count) {
    out << '[';
    for (int i = 0; values && i < count; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}
} // namespace

extern "C" {
crispembed_oxr_context * crispembed_oxr_init(const char * path, int n_threads) {
    if (!path || !*path || !core_env::on("CRISPEMBED_OXR")) return nullptr;
    try {
        auto ctx = std::make_unique<crispembed_oxr_context>();
        ctx->model = oxr_ocr_init(path, n_threads);
        if (!ctx->model) return nullptr;
        return ctx.release();
    } catch (const std::exception &) {
        return nullptr;
    }
}

void crispembed_oxr_free(crispembed_oxr_context * ctx) {
    delete ctx;
}

} // extern C

// One native request; returns the result JSON or throws.
std::string oxr_predict_json(crispembed_oxr_context * ctx, const uint8_t * pixels, int width, int height, int channels,
                             const char * prompt, int max_tokens, float repetition_penalty, bool & eos_out,
                             bool & capped_out, std::string & text_out) {
    if (!core_env::on("CRISPEMBED_OXR")) throw std::runtime_error("CRISPEMBED_OXR=1 is required");
    if (!prompt || std::strlen(prompt) > 65536) throw std::invalid_argument("prompt is NULL or exceeds 65536 bytes");
    if (max_tokens < 0 || max_tokens > 32768) throw std::invalid_argument("max_new_tokens must be in [0,32768]");
    if (!std::isfinite(repetition_penalty) || repetition_penalty <= 0)
        throw std::invalid_argument("repetition_penalty must be finite and positive");
    if (pixels) {
        if (width <= 0 || height <= 0 || uint64_t(width) * uint64_t(height) > 100000000ULL ||
            (channels != 1 && channels != 3 && channels != 4))
            throw std::invalid_argument("expected gray/RGB/RGBA pixels with valid dimensions <=100 megapixels");
    } else if (width || height || channels) {
        throw std::invalid_argument("text-only input requires zero image dimensions");
    }
    if (!oxr_ocr_set_prompt(ctx->model, prompt) || !oxr_ocr_set_repetition_penalty(ctx->model, repetition_penalty))
        throw std::runtime_error(oxr_ocr_last_error(ctx->model));
    oxr_ocr_set_max_tokens(ctx->model, max_tokens);
    const auto started = std::chrono::steady_clock::now();
    int length = 0;
    const char * text = oxr_ocr_recognize_raw(ctx->model, pixels, width, height, channels, &length);
    if (!text || !oxr_ocr_last_ok(ctx->model)) throw std::runtime_error(oxr_ocr_last_error(ctx->model));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const bool eos = oxr_ocr_stopped_eos(ctx->model) != 0;
    const bool capped = oxr_ocr_capped(ctx->model) != 0;
    eos_out = eos;
    capped_out = capped;
    text_out.assign(text, size_t(length));
    const float confidence = oxr_ocr_mean_confidence(ctx->model);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12)
        << "{\"profile\":\"oxr-1.0-vllm-processor\",\"text\":" << quote(std::string(text, size_t(length)))
        << ",\"raw_text\":" << quote(oxr_ocr_raw_text(ctx->model)) << ",\"prompt\":" << quote(prompt)
        << ",\"input_ids\":";
    int count = 0;
    const int32_t * ids = oxr_ocr_prompt_ids(ctx->model, &count);
    write_array(out, ids, count);
    out << ",\"prompt_ids\":";
    write_array(out, ids, count);
    out << ",\"token_ids\":";
    ids = oxr_ocr_completion_ids(ctx->model, &count);
    write_array(out, ids, count);
    out << ",\"grid_thw\":";
    write_array(out, oxr_ocr_grid_thw(ctx->model), 3);
    out << ",\"eos\":" << (eos ? "true" : "false") << ",\"capped\":" << (capped ? "true" : "false")
        << ",\"finish_reason\":"
        << quote(eos      ? "stop"
                 : capped ? "length"
                          : "unknown")
        << ",\"max_tokens\":" << max_tokens << ",\"effective_max_tokens\":" << oxr_ocr_effective_max_tokens(ctx->model)
        << ",\"repetition_penalty\":" << repetition_penalty
        << ",\"repetition_history\":\"prompt-and-completion\",\"min_pixels\":200704,\"max_pixels\":921600"
        << ",\"image_width\":" << width << ",\"image_height\":" << height << ",\"inference_ms\":" << ms
        << ",\"confidence\":" << (std::isfinite(confidence) ? confidence : 0.f) << '}';
    return out.str();
}

extern "C" {
const char * crispembed_oxr_predict(crispembed_oxr_context * ctx, const uint8_t * pixels, int width, int height,
                                    int channels, const char * prompt, int max_tokens, float repetition_penalty) {
    if (!ctx) return nullptr;
    ctx->json.clear();
    ctx->error.clear();
    try {
        bool eos = false, capped = false;
        std::string text;
        ctx->json = oxr_predict_json(ctx, pixels, width, height, channels, prompt, max_tokens, repetition_penalty, eos,
                                     capped, text);
        return ctx->json.c_str();
    } catch (const std::exception & error) {
        ctx->error = error.what();
        return nullptr;
    }
}

const char * crispembed_oxr_last_error(const crispembed_oxr_context * ctx) {
    return ctx ? ctx->error.c_str() : "null OXR context";
}
} // extern C
