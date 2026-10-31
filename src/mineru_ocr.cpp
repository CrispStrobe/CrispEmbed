// MinerU2.5-Pro-2604 public API and bounded native page pipeline.
// Neural inference is entirely the shared Qwen2-VL GGML graph. The Python
// binding additionally exposes the complete pinned MinerU document helper.
#include "crispembed.h"
#include "core/env_gate.h"
#include "image_preprocess.h"
#include "mineru_layout.h"
#include "qwen2vl_ocr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct crispembed_mineru_context {
    qwen2vl_ocr_context * model = nullptr;
    std::string json, markdown, error;
    ~crispembed_mineru_context() {
        if (model) qwen2vl_ocr_free(model);
    }
};

namespace {
using clock_type = std::chrono::steady_clock;
double elapsed_ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

size_t pixel_count(int w, int h) {
    if (w <= 0 || h <= 0 || uint64_t(w) * uint64_t(h) > 100000000ULL)
        throw std::invalid_argument("image dimensions must be positive and at most 100 megapixels");
    return size_t(w) * size_t(h);
}

std::vector<uint8_t> rgb_pixels(const uint8_t * px, int w, int h, int ch) {
    const size_t count = pixel_count(w, h);
    if (!px || (ch != 1 && ch != 3 && ch != 4)) throw std::invalid_argument("expected gray, RGB or RGBA bytes");
    std::vector<uint8_t> rgb(count * 3);
    for (size_t i = 0; i < count; ++i)
        for (int c = 0; c < 3; ++c) {
            const unsigned value = px[i * size_t(ch) + (ch == 1 ? 0 : c)];
            // get_rgb_image calls PIL.convert("RGB"), which discards alpha.
            rgb[i * 3 + c] = uint8_t(value);
        }
    return rgb;
}

std::vector<uint8_t> pil_resize(const std::vector<uint8_t> & rgb, int w, int h, int nw, int nh) {
    if (w == nw && h == nh) return rgb;
    std::vector<float> resized(pixel_count(nw, nh) * 3);
    image_preproc::resize_bicubic_pil_u8_hwc(rgb.data(), h, w, resized.data(), nh, nw, 3);
    std::vector<uint8_t> out(resized.size());
    for (size_t i = 0; i < out.size(); ++i) out[i] = uint8_t(std::clamp(resized[i], 0.f, 255.f));
    return out;
}

std::vector<uint8_t> crop_pixels(const std::vector<uint8_t> & rgb, int w, int h, const mineru::block & block, int & cw,
                                 int & ch) {
    const auto rect = mineru::crop_geometry(block, w, h);
    const auto plan = mineru::normalize_crop_geometry(rect.width(), rect.height(), block.angle);
    std::vector<uint8_t> padded(pixel_count(plan.padded_width, plan.padded_height) * 3, 255);
    for (int y = 0; y < rect.height(); ++y)
        for (int x = 0; x < rect.width(); ++x) {
            int rx = x, ry = y;
            // Pillow's positive angles rotate counter-clockwise, including the
            // model's rotate_right token (the token denotes the needed correction).
            if (block.angle == 90) {
                rx = y;
                ry = rect.width() - 1 - x;
            } else if (block.angle == 180) {
                rx = rect.width() - 1 - x;
                ry = rect.height() - 1 - y;
            } else if (block.angle == 270) {
                rx = rect.height() - 1 - y;
                ry = x;
            }
            const size_t src = (size_t(rect.y0 + y) * w + rect.x0 + x) * 3;
            const size_t dst = (size_t(ry + plan.paste_y) * plan.padded_width + rx + plan.paste_x) * 3;
            std::memcpy(padded.data() + dst, rgb.data() + src, 3);
        }
    cw = plan.output_width;
    ch = plan.output_height;
    return pil_resize(padded, plan.padded_width, plan.padded_height, cw, ch);
}

template <typename T> void array_json(std::ostringstream & out, const T * values, int count) {
    out << '[';
    for (int i = 0; values && i < count; ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
}

struct prediction {
    std::string text, json;
    bool capped = false, eos = false;
    double ms = 0;
    float confidence = 0;
};

prediction predict(crispembed_mineru_context & ctx, const uint8_t * pixels, int w, int h, int ch, const char * prompt,
                   int max_tokens) {
    if (!core_env::on("CRISPEMBED_MINERU25_PRO")) throw std::runtime_error("CRISPEMBED_MINERU25_PRO=1 is required");
    if (max_tokens < 0 || max_tokens > 8192) throw std::invalid_argument("max_new_tokens must be in [0,8192]");
    if (!prompt) throw std::invalid_argument("prompt must not be NULL");
    qwen2vl_ocr_set_max_tokens(ctx.model, max_tokens);
    const auto started = clock_type::now();
    int length = 0;
    const char * text = nullptr;
    if (pixels) {
        if (!qwen2vl_ocr_set_prompt(ctx.model, prompt)) throw std::runtime_error("prompt tokenization failed");
        text = qwen2vl_ocr_recognize_raw(ctx.model, pixels, w, h, ch, &length);
    } else {
        if (w || h || ch) throw std::invalid_argument("text-only requests require zero image dimensions");
        text = qwen2vl_ocr_generate_text(ctx.model, prompt, &length);
    }
    if (!text || !qwen2vl_ocr_last_ok(ctx.model)) throw std::runtime_error("native MinerU inference failed");
    prediction result;
    result.ms = elapsed_ms(started);
    result.text.assign(text, size_t(length));
    result.capped = qwen2vl_ocr_capped(ctx.model) != 0;
    result.eos = qwen2vl_ocr_stopped_eos(ctx.model) != 0;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(12) << "{\"text\":" << mineru::json_quote(result.text)
        << ",\"raw_text\":" << mineru::json_quote(qwen2vl_ocr_raw_text(ctx.model))
        << ",\"prompt\":" << mineru::json_quote(prompt) << ",\"prompt_ids\":";
    int n = 0;
    const int32_t * ids = qwen2vl_ocr_prompt_ids(ctx.model, &n);
    array_json(out, ids, n);
    out << ",\"input_ids\":";
    array_json(out, ids, n);
    out << ",\"token_ids\":";
    ids = qwen2vl_ocr_completion_ids(ctx.model, &n);
    array_json(out, ids, n);
    out << ",\"grid_thw\":";
    array_json(out, qwen2vl_ocr_grid_thw(ctx.model), 3);
    out << ",\"eos\":" << (result.eos ? "true" : "false") << ",\"capped\":" << (result.capped ? "true" : "false")
        << ",\"max_tokens\":" << max_tokens
        << ",\"effective_max_tokens\":" << qwen2vl_ocr_effective_max_tokens(ctx.model) << ",\"image_width\":" << w
        << ",\"image_height\":" << h << ",\"inference_ms\":" << result.ms << ",\"confidence\":";
    const float confidence = qwen2vl_ocr_mean_confidence(ctx.model);
    result.confidence = std::isfinite(confidence) ? confidence : 0.0f;
    out << (std::isfinite(confidence) ? confidence : 0.0f) << '}';
    result.json = out.str();
    return result;
}
} // namespace

extern "C" {
crispembed_mineru_context * crispembed_mineru_init(const char * path, int threads) {
    if (!path || !*path || !core_env::on("CRISPEMBED_MINERU25_PRO")) return nullptr;
    try {
        auto ctx = std::make_unique<crispembed_mineru_context>();
        ctx->model = qwen2vl_ocr_init(path, threads);
        if (!ctx->model || !qwen2vl_ocr_is_mineru(ctx->model)) return nullptr;
        return ctx.release();
    } catch (const std::exception &) {
        return nullptr;
    }
}

void crispembed_mineru_free(crispembed_mineru_context * ctx) {
    delete ctx;
}

const char * crispembed_mineru_predict(crispembed_mineru_context * ctx, const uint8_t * px, int w, int h, int ch,
                                       const char * prompt, int max_tokens) {
    if (!ctx) return nullptr;
    ctx->json.clear();
    ctx->markdown.clear();
    ctx->error.clear();
    try {
        if (px) {
            auto rgb = rgb_pixels(px, w, h, ch);
            ctx->json = predict(*ctx, rgb.data(), w, h, 3, prompt, max_tokens).json;
        } else
            ctx->json = predict(*ctx, nullptr, w, h, ch, prompt, max_tokens).json;
        return ctx->json.c_str();
    } catch (const std::exception & e) {
        ctx->error = e.what();
        return nullptr;
    }
}

const char * crispembed_mineru_extract(crispembed_mineru_context * ctx, const uint8_t * px, int w, int h, int ch,
                                       int image_analysis, int max_tokens) {
    if (!ctx) return nullptr;
    ctx->json.clear();
    ctx->markdown.clear();
    ctx->error.clear();
    try {
        const auto started = clock_type::now();
        const auto rgb = rgb_pixels(px, w, h, ch);
        const auto layout_image = pil_resize(rgb, w, h, 1036, 1036);
        const auto layout =
            predict(*ctx, layout_image.data(), 1036, 1036, 3, mineru::prompt_for_type("[layout]"), max_tokens);
        auto blocks = mineru::parse_layout(layout.text);
        if (blocks.empty() && !mineru::trim(layout.text).empty())
            throw std::runtime_error("layout output contains no valid blocks; raw layout must be inspected");
        mineru::filter_internal_captions(blocks);
        // The native basic renderer cannot reproduce the full blueprint's
        // table-image mask/token restoration. Fail explicitly on that contract.
        for (size_t i = 0; i < blocks.size(); ++i)
            if (blocks[i].type == "image" && mineru::covered_by_type(blocks, i, { "table" }))
                throw std::runtime_error("tables containing images require CrispMinerU.two_step_extract (full helper)");
        bool any_capped = layout.capped;
        double confidence_sum = 0;
        std::vector<std::string> predictions;
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (!mineru::should_extract(blocks, i, image_analysis != 0)) continue;
            int cw = 0, ch_crop = 0;
            std::vector<uint8_t> crop;
            try {
                crop = crop_pixels(rgb, w, h, blocks[i], cw, ch_crop);
            } catch (const std::invalid_argument &) {
                continue;
            } // Python skips zero-size rounded crops.
            const auto content =
                predict(*ctx, crop.data(), cw, ch_crop, 3, mineru::prompt_for_type(blocks[i].type), max_tokens);
            blocks[i].content = content.text;
            blocks[i].has_content = true;
            any_capped = any_capped || content.capped;
            confidence_sum += content.confidence;
            predictions.push_back("{\"block_index\":" + std::to_string(i) + ",\"prediction\":" + content.json + "}");
        }
        mineru::post_process_basic(blocks);
        ctx->markdown = mineru::blocks_markdown(blocks);
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(12) << "{\"profile\":\"native-basic\",\"blueprint_revision\":"
            << mineru::json_quote(mineru::blueprint_revision()) << ",\"blocks\":" << mineru::blocks_json(blocks)
            << ",\"markdown\":" << mineru::json_quote(ctx->markdown)
            << ",\"capped\":" << (any_capped ? "true" : "false")
            << ",\"confidence\":" << (predictions.empty() ? 0.0 : confidence_sum / predictions.size())
            << ",\"inference_ms\":" << elapsed_ms(started) << ",\"layout\":" << layout.json << ",\"recognitions\":[";
        for (size_t i = 0; i < predictions.size(); ++i) {
            if (i) out << ',';
            out << predictions[i];
        }
        out << "]}";
        ctx->json = out.str();
        return ctx->json.c_str();
    } catch (const std::exception & e) {
        ctx->error = e.what();
        ctx->markdown.clear();
        return nullptr;
    }
}

const char * crispembed_mineru_last_markdown(const crispembed_mineru_context * ctx) {
    return ctx ? ctx->markdown.c_str() : "";
}
const char * crispembed_mineru_last_error(const crispembed_mineru_context * ctx) {
    return ctx ? ctx->error.c_str() : "NULL MinerU context";
}
} // extern C
