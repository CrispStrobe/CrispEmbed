// minicpm_ocr.cpp - experimental MiniCPM-OCR-1.0 orchestrator (CPU).
//
//   preprocess (minicpm_ocr_tiles.h + minicpm_ocr_vision.cpp: overview + slices)
//   -> SigLIP vision + 4x projector per image (minicpm_ocr_vision.cpp)
//   -> prompt ids (shared qwen35_ocr::frontend_tokenizer, chat template per spec)
//   -> embedding injection at <|image_pad|> (overview first, slices row-major)
//   -> qwen35 decoder (src/qwen35_decoder.cpp) with 1-D positions written as
//      equal t/h/w axes, greedy decode, stop on 248044 / 248046.
//
// Default-off (CRISPEMBED_MINICPM_OCR). Not parity-qualified yet (no reference
// capture run); every rule is pinned to transformers 5.21.0 source in the
// decision table of docs/minicpm-ocr-inference-spec.md.
#include "minicpm_ocr.h"

#include "core/env_gate.h"
#include "ggml-cpu.h"
#include "minicpm_ocr_tiles.h"
#include "minicpm_ocr_vision.h"
#include "qwen35_decoder.h"
#include "qwen35_frontend.h"
#include "qwen35_weight_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kImagePad = 248056, kImageStart = 248078, kImageEnd = 248079, kSliceStart = 248088, kSliceEnd = 248089,
              kImageIdStart = 248090, kImageIdEnd = 248091;
constexpr int kStopA = 248044, kStopB = 248046;
constexpr int kWidth = 1024, kMaxContext = 4096;
constexpr const char * kDefaultTask = "Text Recognition:";

// ---- debug dump (.npy, names follow tools/minicpm-ocr/reference.py) ----
void write_npy(const std::string & dir, const std::string & name, const void * data, const char * descr, size_t elem,
               const std::vector<size_t> & shape) {
    std::string dims;
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        dims += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
        count *= shape[i];
    }
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': (" + dims + "), }";
    while ((10 + header.size() + 1) % 64) header += ' ';
    header += '\n';
    FILE * f = std::fopen((dir + "/" + name + ".npy").c_str(), "wb");
    if (!f) return;
    const unsigned char magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    const uint16_t len = uint16_t(header.size());
    const unsigned char lenb[2] = { uint8_t(len & 0xff), uint8_t(len >> 8) };
    std::fwrite(magic, 1, 8, f);
    std::fwrite(lenb, 1, 2, f);
    std::fwrite(header.data(), 1, header.size(), f);
    std::fwrite(data, elem, count, f);
    std::fclose(f);
}
void dump_f32(const std::string & dir, const std::string & name, const std::vector<float> & v, size_t cols) {
    if (dir.empty() || v.empty() || !cols || v.size() % cols) return;
    write_npy(dir, name, v.data(), "<f4", 4, { v.size() / cols, cols });
}
void dump_i32(const std::string & dir, const std::string & name, const std::vector<int> & v) {
    if (dir.empty()) return;
    std::vector<int32_t> w(v.begin(), v.end());
    write_npy(dir, name, w.data(), "<i4", 4, { w.size() });
}

} // namespace

struct minicpm_ocr_context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad loaded;
    qwen35_ocr::frontend_tokenizer tok; // shared Qwen3.5 BPE (load_minicpm_ocr profile)
    minicpm_ocr::vision_weights vision;
    qwen35_ocr::decoder_weights decoder;
    qwen35_ocr::attention_config attention;
    qwen35_ocr::converted_weight_cache weights_cache;
    minicpm_ocr_tiles::config tiles;
    int threads = 1, max_tokens = 1024;
    std::string task = kDefaultTask, text, error;
    std::vector<int> ids;
    std::vector<float> confidences;
    ~minicpm_ocr_context() {
        weights_cache.clear();
        core_gguf::free_weights(loaded);
        if (backend) ggml_backend_free(backend);
    }
    const char * fail(const std::string & message) {
        text.clear();
        ids.clear();
        confidences.clear();
        error = message;
        return nullptr;
    }
};

namespace {

// Prompt (chat_template.jinja + MiniCPMOcrProcessor, transformers 5.21.0):
//   <|im_start|>user\n{image}\n{task}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
//   image = <image_id>0</image_id><image>{pad x N0}</image>{rows joined "\n";
//           row = (<slice>{pad x Ni}</slice>) x cols}
// No separator between </image> and the first <slice> (processing_minicpm_ocr.py:164-180).
// HF tokenizes the whole string after splitting on added tokens; every ordinary
// piece below is delimited by added tokens, so piecewise encoding is identical.
bool build_prompt(const qwen35_ocr::frontend_tokenizer & tok, const std::string & task,
                  const minicpm_ocr_tiles::plan & plan, std::vector<int> & ids) {
    ids.clear();
    auto text = [&](const std::string & s) {
        std::vector<int> piece;
        if (!tok.encode(s, piece)) return false;
        ids.insert(ids.end(), piece.begin(), piece.end());
        return true;
    };
    auto pads = [&](int n) { ids.insert(ids.end(), size_t(n), kImagePad); };
    if (!text("<|im_start|>user\n")) return false;
    ids.push_back(kImageIdStart);
    if (!text("0")) return false;
    ids.push_back(kImageIdEnd);
    ids.push_back(kImageStart);
    pads(plan.parts[0].tokens);
    ids.push_back(kImageEnd);
    int row = 0;
    for (size_t i = 1; i < plan.parts.size(); ++i) {
        if (plan.parts[i].row != row) {
            row = plan.parts[i].row;
            if (!text("\n")) return false;
        }
        ids.push_back(kSliceStart);
        pads(plan.parts[i].tokens);
        ids.push_back(kSliceEnd);
    }
    if (!text("\n" + task + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")) return false;
    return ids.size() < size_t(kMaxContext);
}

std::vector<float> equal_axes(int start, int count) {
    std::vector<float> axes(3 * size_t(count));
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < count; ++i) axes[a * size_t(count) + i] = float(start + i);
    return axes;
}

const char * recognize(minicpm_ocr_context & c, const uint8_t * pixels, int w, int h, int channels) {
    if (!core_env::on("CRISPEMBED_MINICPM_OCR") || !core_env::on("CRISPEMBED_QWEN35_OCR"))
        return c.fail("experimental gates off (CRISPEMBED_MINICPM_OCR and CRISPEMBED_QWEN35_OCR)");
    if (!pixels || w < 1 || h < 1 || w > 16384 || h > 16384 || int64_t(w) * h > 16777216 ||
        (channels != 1 && channels != 3 && channels != 4))
        return c.fail("invalid image dimensions or channels");
    const char * dump_env = std::getenv("CRISPEMBED_MINICPM_OCR_DUMP");
    const std::string dump = dump_env && *dump_env && std::strcmp(dump_env, "0") != 0 ? dump_env : "";
    std::vector<uint8_t> rgb;
    if (channels != 3) {
        rgb.resize(size_t(w) * h * 3);
        for (size_t i = 0; i < size_t(w) * h; ++i)
            for (int k = 0; k < 3; ++k) rgb[i * 3 + k] = pixels[i * channels + (channels == 1 ? 0 : k)];
        pixels = rgb.data();
    }
    minicpm_ocr_tiles::plan plan;
    std::vector<minicpm_ocr::image_part> parts;
    if (!minicpm_ocr::preprocess(pixels, w, h, c.tiles, plan, parts)) return c.fail("preprocessing failed");
    if (!dump.empty()) {
        std::string meta =
            "{\"grid\":[" + std::to_string(plan.grid.w) + "," + std::to_string(plan.grid.h) + "],\"parts\":[";
        for (size_t i = 0; i < parts.size(); ++i) {
            const auto & p = parts[i];
            meta += std::string(i ? "," : "") + "{\"w\":" + std::to_string(p.width) +
                    ",\"h\":" + std::to_string(p.height) + ",\"tokens\":" + std::to_string(p.tokens) +
                    ",\"overview\":" + (p.overview ? "true" : "false") + ",\"row\":" + std::to_string(p.row) +
                    ",\"col\":" + std::to_string(p.col) + "}";
            write_npy(dump, "native_pixels_" + std::to_string(i), p.chw.data(), "<f4", 4,
                      { 3, size_t(p.height), size_t(p.width) });
        }
        meta += "]}\n";
        if (FILE * f = std::fopen((dump + "/native_meta.json").c_str(), "wb")) {
            std::fwrite(meta.data(), 1, meta.size(), f);
            std::fclose(f);
        }
    }
    std::vector<int> ids;
    if (!build_prompt(c.tok, c.task, plan, ids)) return c.fail("prompt exceeds the native context bound");
    dump_i32(dump, "input_ids", ids);

    std::vector<float> merged;
    minicpm_ocr::vision_debug dbg;
    dbg.enabled = !dump.empty();
    for (const auto & p : parts)
        if (!minicpm_ocr::encode(c.vision, c.backend, p, c.threads, merged, &dbg))
            return c.fail("vision inference failed");
    if (!dump.empty()) {
        const size_t D = size_t(c.vision.width);
        dump_f32(dump, "model_vision_tower_embeddings", dbg.embeddings, D);
        dump_f32(dump, "model_vision_tower_encoder_layers_0", dbg.layer0, D);
        dump_f32(dump, "model_vision_tower_encoder_layers_6", dbg.layer6, D);
        dump_f32(dump, "model_vision_tower_encoder_layers_26", dbg.layer26, D);
        dump_f32(dump, "model_vision_tower_post_layernorm", dbg.post_ln, D);
        dump_f32(dump, "model_merger", dbg.merged, size_t(kWidth));
    }

    std::vector<float> hidden;
    if (!qwen35_ocr::embed_cpu(c.decoder.embedding, ids, hidden)) return c.fail("embedding failed");
    size_t image_token = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] == kImagePad) {
            if ((image_token + 1) * kWidth > merged.size()) return c.fail("image embedding count mismatch");
            std::copy_n(merged.data() + image_token * kWidth, kWidth, hidden.data() + i * kWidth);
            ++image_token;
        }
    if (image_token * kWidth != merged.size()) return c.fail("image embedding count mismatch");

    auto * cache_ptr = core_env::on("CRISPEMBED_QWEN35_WEIGHT_CACHE") ? &c.weights_cache : nullptr;
    // position_ids = None -> arange expanded to 4 equal rows (qwen3_5 modeling:1247-1256); with equal
    // t/h/w axes the default mrope_section interleave is the identity, i.e. plain 1-D RoPE
    // over 64 of 256 dims (partial_rotary_factor 0.25), theta 1e7, rotate_half (NeoX) pairs.
    auto positions = equal_axes(0, int(ids.size()));
    qwen35_ocr::decoder_outputs current;
    if (!qwen35_ocr::decoder_cpu(c.decoder, c.attention, hidden.data(), positions.data(), 1, int(ids.size()), c.threads,
                                 nullptr, current, cache_ptr))
        return c.fail("decoder prefill failed");
    if (!dump.empty()) {
        for (size_t i = 0; i < current.hidden.size(); ++i)
            dump_f32(dump, "model_language_model_layers_" + std::to_string(i), current.hidden[i], size_t(kWidth));
        dump_f32(dump, "model_language_model_norm", current.norm, size_t(kWidth));
        dump_f32(dump, "lm_head", current.logits, current.logits.size());
    }
    const size_t vocab = size_t(c.decoder.head->ne[1]);
    for (int step = 0; step < c.max_tokens; ++step) {
        if (current.logits.size() != vocab ||
            !std::all_of(current.logits.begin(), current.logits.end(), [](float x) { return std::isfinite(x); }))
            return c.fail("nonfinite or missing vocabulary scores");
        const auto best = std::max_element(current.logits.begin(), current.logits.end());
        const int id = int(best - current.logits.begin());
        double total = 0;
        for (float x : current.logits) total += std::exp(double(x) - *best);
        c.ids.push_back(id);
        c.confidences.push_back(float(1.0 / total));
        if (c.tok.is_stop(id)) break; // generation_config eos_token_id [248044, 248046]
        if (step + 1 == c.max_tokens || current.cache.tokens >= kMaxContext) break;
        auto cache = std::move(current.cache);
        if (!qwen35_ocr::embed_cpu(c.decoder.embedding, { id }, hidden)) return c.fail("embedding failed");
        positions = equal_axes(cache.tokens, 1);
        if (!qwen35_ocr::decoder_cpu(c.decoder, c.attention, hidden.data(), positions.data(), 1, 1, c.threads, &cache,
                                     current, cache_ptr))
            return c.fail("cached decoder failed");
    }
    if (!dump.empty()) {
        dump_i32(dump, "ids", c.ids);
        std::fprintf(stderr, "minicpm_ocr gen_ids (%zu):", c.ids.size());
        for (int id : c.ids) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");
    }
    if (!c.tok.decode(c.ids, c.text)) return c.fail("output decoding failed");
    return c.text.c_str();
}

} // namespace

extern "C" minicpm_ocr_context * minicpm_ocr_init(const char * path, int threads) {
    if (!core_env::on("CRISPEMBED_MINICPM_OCR") || !path || !*path || threads < 1 || threads > 256) return nullptr;
    if (!core_env::on("CRISPEMBED_QWEN35_OCR")) {
        std::fprintf(stderr, "minicpm_ocr: the reused qwen35 decoder also needs CRISPEMBED_QWEN35_OCR=1\n");
        return nullptr;
    }
    try {
        auto c = std::make_unique<minicpm_ocr_context>();
        c->threads = threads;
        std::string error;
        {
            std::unique_ptr<gguf_context, decltype(&core_gguf::free_metadata)> meta(core_gguf::open_metadata(path),
                                                                                    core_gguf::free_metadata);
            if (!meta) return nullptr;
            auto * m = meta.get();
            const bool ok = core_gguf::kv_str(m, "general.architecture", "") == "minicpm_ocr" &&
                            core_gguf::kv_u32(m, "minicpm_ocr.format_version", 0) == 1 &&
                            core_gguf::kv_str(m, "minicpm_ocr.downsample_mode", "") == "4x" &&
                            core_gguf::kv_bool(m, "minicpm_ocr.tie_word_embeddings", false) &&
                            core_gguf::kv_bool(m, "minicpm_ocr.zero_centered_text_norm", false);
            if (!ok) error = "not a minicpm_ocr format_version 1 / 4x GGUF";
            if (ok && !c->tok.load_minicpm_ocr(m)) error = "tokenizer metadata rejected";
            auto & t = c->tiles;
            t.scale_resolution = int(core_gguf::kv_u32(m, "minicpm_ocr.scale_resolution", 448));
            t.max_slice_nums = int(core_gguf::kv_u32(m, "minicpm_ocr.max_slice_nums", 9));
            t.patch_size = int(core_gguf::kv_u32(m, "minicpm_ocr.patch_size", 14));
            t.slice_mode = core_gguf::kv_bool(m, "minicpm_ocr.slice_mode", true);
            if (t.scale_resolution != 448 || t.max_slice_nums != 9 || t.patch_size != 14)
                error = "unaudited preprocessing parameters";
            const auto eos = core_gguf::kv_i32_array(m, "minicpm_ocr.source_eos_token_ids");
            if (eos != std::vector<int>{ kStopA, kStopB }) error = "EOS list differs from [248044, 248046]";
            auto & a = c->attention; // Qwen3.5-0.8B shape; theta 1e7, eps 1e-6
            a.theta = float(core_gguf::kv_f32(m, "minicpm_ocr.rope_theta", 10000000.0f));
            a.eps = float(core_gguf::kv_f32(m, "minicpm_ocr.text_rms_eps", 1e-6f));
            c->vision.eps = float(core_gguf::kv_f32(m, "minicpm_ocr.vision_ln_eps", 1e-6f));
            c->vision.merger_eps = float(core_gguf::kv_f32(m, "minicpm_ocr.merger_ln_eps", 1e-6f));
        }
        if (!error.empty()) {
            std::fprintf(stderr, "minicpm_ocr: %s\n", error.c_str());
            return nullptr;
        }
        c->backend = ggml_backend_cpu_init();
        if (!c->backend || !core_gguf::load_weights(path, c->backend, "minicpm-ocr", c->loaded)) return nullptr;
        if (!qwen35_ocr::map_small_decoder(c->loaded.tensors, c->decoder, 248100)) error = "decoder tensors";
        if (!error.empty() || !minicpm_ocr::map_vision(c->loaded.tensors, c->vision, error)) {
            std::fprintf(stderr, "minicpm_ocr: %s\n", error.c_str());
            return nullptr;
        }
        return c.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void minicpm_ocr_free(minicpm_ocr_context * c) {
    delete c;
}

extern "C" int minicpm_ocr_set_max_tokens(minicpm_ocr_context * c, int n) {
    if (!c || n < 1 || n > kMaxContext) return 0;
    c->max_tokens = n;
    return 1;
}

extern "C" int minicpm_ocr_get_max_tokens(const minicpm_ocr_context * c) {
    return c ? c->max_tokens : 0;
}

extern "C" int minicpm_ocr_set_prompt(minicpm_ocr_context * c, const char * prompt) {
    if (!c) return 0;
    if (!prompt || !*prompt) {
        c->task = kDefaultTask;
        return 1;
    }
    if (std::strlen(prompt) > 65536) return 0;
    c->task = prompt;
    return 1;
}

extern "C" const char * minicpm_ocr_recognize_raw(minicpm_ocr_context * c, const uint8_t * p, int w, int h, int ch,
                                                  int * len) {
    if (len) *len = 0;
    if (!c) return nullptr;
    c->text.clear();
    c->ids.clear();
    c->confidences.clear();
    c->error.clear();
    try {
        const char * text = recognize(*c, p, w, h, ch);
        if (text && len) *len = int(c->text.size());
        return text;
    } catch (...) {
        return c->fail("recognition exception");
    }
}

extern "C" const int * minicpm_ocr_token_ids(const minicpm_ocr_context * c, int * n) {
    if (n) *n = c ? int(c->ids.size()) : 0;
    return c && !c->ids.empty() ? c->ids.data() : nullptr;
}

extern "C" const float * minicpm_ocr_confidences(const minicpm_ocr_context * c, int * n) {
    if (n) *n = c ? int(c->confidences.size()) : 0;
    return c && !c->confidences.empty() ? c->confidences.data() : nullptr;
}

extern "C" const char * minicpm_ocr_last_error(const minicpm_ocr_context * c) {
    return c ? c->error.c_str() : "null context";
}
