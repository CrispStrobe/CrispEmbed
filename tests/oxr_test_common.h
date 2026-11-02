// Test-only OXR runner utilities; never included by production inference.
#pragma once
#include "oxr_ocr.h"
#include "core/json.h"
#include "stb_image.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace oxr_test {
struct options {
    std::string model, image, reference;
    std::string prompt = "Recognize the content of the image.";
    int threads = 4, repeat = 3, max_tokens = 256, min_pixels = 200704, max_pixels = 921600;
    int hf_generation = 1;
    float repetition_penalty = 1.f;
    double max_abs = .002, min_cos = .99999;
};
inline int integer(const std::string & value) {
    size_t end = 0;
    const int result = std::stoi(value, &end);
    if (end != value.size()) throw std::invalid_argument("integer contains trailing characters");
    return result;
}
inline double number(const std::string & value) {
    size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result)) throw std::invalid_argument("invalid finite number");
    return result;
}
inline options parse(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("missing option value: " + key);
        const std::string value = argv[++i];
        if (key == "--model")
            result.model = value;
        else if (key == "--image")
            result.image = value;
        else if (key == "--reference")
            result.reference = value;
        else if (key == "--prompt")
            result.prompt = value;
        else if (key == "--threads")
            result.threads = integer(value);
        else if (key == "--repeat")
            result.repeat = integer(value);
        else if (key == "--max-tokens")
            result.max_tokens = integer(value);
        else if (key == "--min-pixels")
            result.min_pixels = integer(value);
        else if (key == "--max-pixels")
            result.max_pixels = integer(value);
        else if (key == "--hf-generation")
            result.hf_generation = integer(value);
        else if (key == "--repetition-penalty")
            result.repetition_penalty = float(number(value));
        else if (key == "--max-abs")
            result.max_abs = number(value);
        else if (key == "--min-cos")
            result.min_cos = number(value);
        else
            throw std::invalid_argument("unknown option: " + key);
    }
    if (result.model.empty() || result.image.empty() || result.threads < 1 || result.repeat < 1 ||
        result.repeat > 100 || result.max_tokens < 1 || result.max_tokens > 32768 || result.prompt.size() > 65536 ||
        result.min_pixels < 1 || result.max_pixels < result.min_pixels || result.max_pixels > 2408448 ||
        (result.hf_generation != 0 && result.hf_generation != 1) || !std::isfinite(result.repetition_penalty) ||
        result.repetition_penalty <= 0 || result.max_abs < 0 || result.min_cos < 0 || result.min_cos > 1)
        throw std::invalid_argument("invalid OXR runner configuration");
    return result;
}

struct image {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};
inline image load_image(const std::string & path) {
    image result;
    int channels = 0;
    auto * raw = stbi_load(path.c_str(), &result.width, &result.height, &channels, 4);
    if (!raw || result.width < 1 || result.height < 1 || uint64_t(result.width) * result.height > 100000000ULL) {
        if (raw) stbi_image_free(raw);
        throw std::runtime_error("image decode failed or invalid dimensions");
    }
    const size_t count = size_t(result.width) * result.height;
    result.rgb.resize(count * 3);
    for (size_t i = 0; i < count; ++i)
        for (size_t c = 0; c < 3; ++c) result.rgb[i * 3 + c] = raw[i * 4 + c];
    stbi_image_free(raw);
    return result;
}

using context = std::unique_ptr<oxr_ocr_context, decltype(&oxr_ocr_free)>;
inline context load_model(const options & opt) {
    context ctx(oxr_ocr_init(opt.model.c_str(), opt.threads), &oxr_ocr_free);
    if (!ctx) throw std::runtime_error("OXR model/profile/gate initialization failed");
    if (!oxr_ocr_set_prompt(ctx.get(), opt.prompt.c_str()) ||
        !oxr_ocr_set_pixel_limits(ctx.get(), opt.min_pixels, opt.max_pixels) ||
        !oxr_ocr_set_hf_generation(ctx.get(), opt.hf_generation) ||
        !oxr_ocr_set_repetition_penalty(ctx.get(), opt.repetition_penalty))
        throw std::runtime_error(oxr_ocr_last_error(ctx.get()));
    oxr_ocr_set_max_tokens(ctx.get(), opt.max_tokens);
    return ctx;
}

inline std::string quote(const std::string & value) {
    return "\"" + core_json::json_escape_strict(value) + "\"";
}
template <typename T> inline void array(std::ostream & out, const T * values, int count) {
    out << '[';
    for (int i = 0; values && i < count; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

inline std::string result_json(oxr_ocr_context * ctx, const options & opt, const image & img, const char * text,
                               int length, double elapsed_ms) {
    int count = 0;
    const bool eos = oxr_ocr_stopped_eos(ctx) != 0;
    const bool capped = oxr_ocr_capped(ctx) != 0;
    const float confidence = oxr_ocr_mean_confidence(ctx);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12) << "{\"text\":" << quote(std::string(text, size_t(length)))
        << ",\"raw_text\":" << quote(oxr_ocr_raw_text(ctx)) << ",\"prompt\":" << quote(opt.prompt) << ",\"input_ids\":";
    const int32_t * ids = oxr_ocr_prompt_ids(ctx, &count);
    array(out, ids, count);
    out << ",\"prompt_ids\":";
    array(out, ids, count);
    out << ",\"token_ids\":";
    ids = oxr_ocr_completion_ids(ctx, &count);
    array(out, ids, count);
    out << ",\"grid_thw\":";
    array(out, oxr_ocr_grid_thw(ctx), 3);
    out << ",\"eos\":" << (eos ? "true" : "false") << ",\"capped\":" << (capped ? "true" : "false")
        << ",\"finish_reason\":"
        << quote(eos      ? "stop"
                 : capped ? "length"
                          : "unknown")
        << ",\"max_tokens\":" << opt.max_tokens << ",\"effective_max_tokens\":" << oxr_ocr_effective_max_tokens(ctx)
        << ",\"repetition_penalty\":" << opt.repetition_penalty
        << ",\"hf_generation\":" << (opt.hf_generation ? "true" : "false")
        << ",\"repetition_history\":" << quote(opt.hf_generation ? "completion-only" : "prompt-and-completion")
        << ",\"min_pixels\":" << opt.min_pixels << ",\"max_pixels\":" << opt.max_pixels
        << ",\"image_width\":" << img.width << ",\"image_height\":" << img.height << ",\"inference_ms\":" << elapsed_ms
        << ",\"confidence\":" << (std::isfinite(confidence) ? confidence : 0.f) << '}';
    return out.str();
}
} // namespace oxr_test
