#include "qwen35_ocr.h"
#include "qwen35_frontend.h"
#include "qwen35_decoder.h"
#include "qwen35_positions.h"
#include "qwen35_vision.h"
#include "core/env_gate.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
// Provided by image_preprocess.cpp when integrated into the main library.
extern "C" {
int stbi_info(const char *, int *, int *, int *);
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
}
struct qwen35_ocr_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad loaded;
    qwen35_ocr::frontend_tokenizer tokenizer;
    qwen35_ocr::decoder_weights decoder;
    qwen35_ocr::vision_weights vision;
    int threads = 1, max_tokens = 2048;
    std::string text, error;
    std::vector<int> ids;
    std::vector<float> confidences;
    qwen35_ocr_stop stop = QWEN35_OCR_NONE;
    ~qwen35_ocr_context() {
        core_gguf::free_weights(loaded);
        if (backend) ggml_backend_free(backend);
    }
    void clear() {
        text.clear();
        error.clear();
        ids.clear();
        confidences.clear();
        stop = QWEN35_OCR_NONE;
    }
    const char * fail(const char * message) noexcept {
        clear();
        stop = QWEN35_OCR_ERROR;
        try {
            error = message;
        } catch (...) {
        }
        return nullptr;
    }
};
namespace {
bool shape(ggml_tensor * t, int x, int y = 1, int z = 1) {
    return t && t->ne[0] == x && t->ne[1] == y && t->ne[2] == z && t->ne[3] == 1 &&
           (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
}
bool valid_weights(const qwen35_ocr_context & c) {
    const auto & d = c.decoder;
    const auto & v = c.vision;
    if (c.loaded.tensors.size() != 473 || d.layers.size() != 24 || v.layers.size() != 12 ||
        !shape(d.embedding, 1024, 248320) || d.head != d.embedding || !shape(d.norm, 1024) ||
        !shape(v.patch, 1536, 768) || !shape(v.patch_bias, 768) || !shape(v.position, 768, 2304) ||
        !shape(v.merger_norm, 768) || !shape(v.merger_norm_bias, 768) || !shape(v.merger_up, 3072, 3072) ||
        !shape(v.merger_up_bias, 3072) || !shape(v.merger_down, 3072, 1024) || !shape(v.merger_down_bias, 1024))
        return false;
    for (const auto & l : d.layers) {
        if (!shape(l.input_norm, 1024) || !shape(l.post_norm, 1024) || !shape(l.gate, 1024, 3584) ||
            !shape(l.up, 1024, 3584) || !shape(l.down, 3584, 1024))
            return false;
        if (l.full_attention) {
            const auto & a = l.attention;
            if (!shape(a.q, 1024, 4096) || !shape(a.k, 1024, 512) || !shape(a.v, 1024, 512) ||
                !shape(a.out, 2048, 1024) || !shape(a.q_norm, 256) || !shape(a.k_norm, 256))
                return false;
        } else {
            const auto & a = l.linear;
            if (!shape(a.qkv, 1024, 6144) || !shape(a.z, 1024, 2048) || !shape(a.a, 1024, 16) ||
                !shape(a.b, 1024, 16) || !shape(a.out, 2048, 1024) || !shape(a.conv, 4, 1, 6144) ||
                !shape(a.a_log, 16) || !shape(a.dt_bias, 16) || !shape(a.norm, 128))
                return false;
        }
    }
    for (const auto & a : v.layers)
        if (!shape(a.norm1, 768) || !shape(a.norm1_bias, 768) || !shape(a.norm2, 768) || !shape(a.norm2_bias, 768) ||
            !shape(a.qkv, 768, 2304) || !shape(a.qkv_bias, 2304) || !shape(a.out, 768, 768) ||
            !shape(a.out_bias, 768) || !shape(a.up, 768, 3072) || !shape(a.up_bias, 3072) ||
            !shape(a.down, 3072, 768) || !shape(a.down_bias, 768))
            return false;
    return true;
}
bool image_bounds(int w, int h, int channels) {
    return w > 0 && h > 0 && w <= 32768 && h <= 32768 && int64_t(w) * h <= 16777216 &&
           double(std::max(w, h)) / std::min(w, h) <= 200 && (channels == 1 || channels == 3 || channels == 4);
}
const char * recognize(qwen35_ocr_context & c, const uint8_t * pixels, int w, int h, int channels,
                       const char * prompt) {
    if (!core_env::on("CRISPEMBED_QWEN35_OCR")) return c.fail("experimental OCR gate is off");
    if (!pixels || !image_bounds(w, h, channels)) return c.fail("invalid image dimensions or channels");
    size_t length = 0;
    if (prompt) {
        while (length <= 65536 && prompt[length]) ++length;
    }
    if (length > 65536) return c.fail("prompt exceeds byte limit");
    std::vector<uint8_t> rgb;
    if (channels != 3) {
        rgb.resize(size_t(w) * h * 3);
        for (size_t i = 0; i < size_t(w) * h; ++i)
            for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = pixels[i * channels + (channels == 1 ? 0 : k)];
        pixels = rgb.data();
    }
    qwen35_ocr::image_inputs image;
    if (!qwen35_ocr::preprocess_rgb(pixels, h, w, {}, image)) return c.fail("image exceeds native patch bound");
    std::vector<int> ids, modalities;
    std::string rendered;
    if (!c.tokenizer.image_prompt(prompt ? std::string(prompt, length) : "", int(image.patches.size() / 1536 / 4), ids,
                                  modalities, rendered))
        return c.fail("invalid or oversized image prompt");
    qwen35_ocr::position_outputs pos;
    if (!qwen35_ocr::image_positions(modalities, image.grid, 2, pos)) return c.fail("invalid multimodal positions");
    qwen35_ocr::vision_outputs visual;
    if (!qwen35_ocr::vision_cpu(c.vision, {}, image.patches.data(), int(image.patches.size() / 1536), image.grid,
                                c.threads, visual))
        return c.fail("vision inference failed");
    std::vector<float> hidden;
    if (!qwen35_ocr::embed_cpu(c.decoder.embedding, ids, hidden)) return c.fail("embedding failed");
    size_t image_token = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        if (modalities[i] == 1) {
            if ((image_token + 1) * 1024 > visual.merged.size()) return c.fail("image embedding count mismatch");
            std::copy_n(visual.merged.data() + image_token * 1024, 1024, hidden.data() + i * 1024);
            ++image_token;
        }
    if (image_token * 1024 != visual.merged.size()) return c.fail("image embedding count mismatch");
    // No cross-request cache: all carried state is local to this recognition.
    qwen35_ocr::decoder_outputs current;
    if (!qwen35_ocr::decoder_cpu(c.decoder, {}, hidden.data(), pos.axes.data(), 1, int(ids.size()), c.threads, nullptr,
                                 current))
        return c.fail("decoder prefill failed");
    for (int step = 0; step < c.max_tokens; ++step) {
        if (current.logits.size() != 248320 ||
            !std::all_of(current.logits.begin(), current.logits.end(), [](float x) { return std::isfinite(x); }))
            return c.fail("nonfinite or missing vocabulary scores");
        const auto best = std::max_element(current.logits.begin(), current.logits.end());
        const int id = int(best - current.logits.begin());
        double total = 0;
        for (float x : current.logits) total += std::exp(double(x) - *best);
        c.ids.push_back(id);
        c.confidences.push_back(float(1.0 / total));
        if (id == c.tokenizer.eos()) {
            c.stop = QWEN35_OCR_EOS;
            break;
        }
        if (step + 1 == c.max_tokens) {
            c.stop = QWEN35_OCR_MAX_TOKENS;
            break;
        }
        if (current.cache.tokens >= 4096) {
            c.stop = QWEN35_OCR_CONTEXT_LIMIT;
            break;
        }
        auto cache = std::move(current.cache);
        std::vector<float> coords;
        if (!qwen35_ocr::embed_cpu(c.decoder.embedding, { id }, hidden) ||
            !qwen35_ocr::cached_positions(cache.tokens, 1, pos.rope_delta, coords))
            return c.fail("cached input preparation failed");
        if (!qwen35_ocr::decoder_cpu(c.decoder, {}, hidden.data(), coords.data(), 1, 1, c.threads, &cache, current))
            return c.fail("cached decoder failed");
    }
    if (!c.tokenizer.decode(c.ids, c.text)) return c.fail("output decoding failed");
    return c.text.c_str();
}
} // namespace
extern "C" qwen35_ocr_context * qwen35_ocr_init(const char * path, int threads) {
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !path || !*path || threads < 1 || threads > 256) return nullptr;
    try {
        auto c = std::make_unique<qwen35_ocr_context>();
        c->threads = threads;
        std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> metadata(core_gguf::open_metadata(path),
                                                                                    core_gguf::free_metadata);
        auto * meta = metadata.get();
        if (!meta) return nullptr;
        const bool ready = core_gguf::kv_str(meta, "general.architecture", "") == "qwen35_ocr" &&
                           core_gguf::kv_u32(meta, "qwen35_ocr.format_version", 0) == 1 &&
                           core_gguf::kv_bool(meta, "qwen35_ocr.tie_word_embeddings", false) &&
                           core_gguf::kv_bool(meta, "qwen35_ocr.zero_centered_text_norm", false) &&
                           c->tokenizer.load(meta);
        metadata.reset();
        if (!ready) return nullptr;
        c->backend = ggml_backend_cpu_init();
        if (!c->backend || !core_gguf::load_weights(path, c->backend, "qwen35-ocr", c->loaded) ||
            !qwen35_ocr::map_small_decoder(c->loaded.tensors, c->decoder) ||
            !qwen35_ocr::map_small_vision(c->loaded.tensors, c->vision) || !valid_weights(*c))
            return nullptr;
        return c.release();
    } catch (...) {
        return nullptr;
    }
}
extern "C" void qwen35_ocr_free(qwen35_ocr_context * c) {
    delete c;
}
extern "C" int qwen35_ocr_set_max_tokens(qwen35_ocr_context * c, int n) {
    if (!c || n < 1 || n > 4096) return 0;
    c->max_tokens = n;
    return 1;
}
extern "C" int qwen35_ocr_get_max_tokens(const qwen35_ocr_context * c) {
    return c ? c->max_tokens : 0;
}
extern "C" const char * qwen35_ocr_recognize_raw(qwen35_ocr_context * c, const uint8_t * p, int w, int h, int ch,
                                                 const char * prompt, int * len) {
    if (len) *len = 0;
    if (!c) return nullptr;
    c->clear();
    try {
        const auto * text = recognize(*c, p, w, h, ch, prompt);
        if (text && len) *len = int(c->text.size());
        return text;
    } catch (...) {
        return c->fail("recognition exception");
    }
}
extern "C" const char * qwen35_ocr_recognize_file(qwen35_ocr_context * c, const char * path, const char * prompt,
                                                  int * len) {
    if (len) *len = 0;
    if (!c) return nullptr;
    c->clear();
    try {
        if (!core_env::on("CRISPEMBED_QWEN35_OCR")) return c->fail("experimental OCR gate is off");
        int w = 0, h = 0, ch = 0;
        if (!path || !*path || !stbi_info(path, &w, &h, &ch) || !image_bounds(w, h, 3))
            return c->fail("invalid image file");
        std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels(stbi_load(path, &w, &h, &ch, 3),
                                                                          stbi_image_free);
        if (!pixels) return c->fail("image file decoding failed");
        return qwen35_ocr_recognize_raw(c, pixels.get(), w, h, 3, prompt, len);
    } catch (...) {
        return c->fail("file recognition exception");
    }
}
extern "C" const int * qwen35_ocr_token_ids(const qwen35_ocr_context * c, int * n) {
    if (n) *n = c ? int(c->ids.size()) : 0;
    return c && !c->ids.empty() ? c->ids.data() : nullptr;
}
extern "C" const float * qwen35_ocr_confidences(const qwen35_ocr_context * c, int * n) {
    if (n) *n = c ? int(c->confidences.size()) : 0;
    return c && !c->confidences.empty() ? c->confidences.data() : nullptr;
}
extern "C" float qwen35_ocr_mean_confidence(const qwen35_ocr_context * c) {
    if (!c || c->confidences.empty()) return 0;
    double sum = 0;
    for (float p : c->confidences) sum += p;
    return float(sum / c->confidences.size());
}
extern "C" qwen35_ocr_stop qwen35_ocr_last_stop(const qwen35_ocr_context * c) {
    return c ? c->stop : QWEN35_OCR_ERROR;
}
extern "C" const char * qwen35_ocr_last_error(const qwen35_ocr_context * c) {
    return c ? c->error.c_str() : "null context";
}
