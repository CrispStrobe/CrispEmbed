#include "qwen35_decoder.h"
#include "qwen35_math.h"
#include "core/env_gate.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace qwen35_ocr {
static bool shape(ggml_tensor * t, int columns, int rows) {
    return t && t->ne[0] == columns && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool rms(const std::vector<float> & x, ggml_tensor * norm, int width, float eps, std::vector<float> & y) {
    if (!shape(norm, width, 1)) return false;
    std::vector<float> weights(width);
    if (!tensor_floats(norm, weights.data())) return false;
    y.resize(x.size());
    for (size_t base = 0; base < x.size(); base += width) {
        const float scale = rms_scale_cpu(x.data() + base, width, eps);
        for (int i = 0; i < width; ++i) y[base + i] = (x[base + i] * scale) * (1.0f + weights[i]);
    }
    return true;
}
bool map_small_decoder(const core_gguf::tensor_map & tensors, decoder_weights & weights) {
    weights = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR")) return false;
    auto get = [&](const std::string & name) { return core_gguf::try_get(tensors, name.c_str()); };
    weights.norm = get("l.norm.weight");
    weights.embedding = get("l.embed_tokens.weight");
    weights.head = get("lm_head.weight");
    if (!weights.head) weights.head = weights.embedding;
    if (!shape(weights.embedding, 1024, 248320) || !shape(weights.head, 1024, 248320) || !shape(weights.norm, 1024, 1))
        return false;
    for (int i = 0; i < 24; ++i) {
        const auto prefix = "l.layers." + std::to_string(i) + ".";
        auto field = [&](const char * name) { return get(prefix + name); };
        decoder_layer_weights w;
        w.full_attention = i % 4 == 3;
        w.input_norm = field("input_layernorm.weight");
        w.post_norm = field("post_attention_layernorm.weight");
        w.gate = field("mlp.gate_proj.weight");
        w.up = field("mlp.up_proj.weight");
        w.down = field("mlp.down_proj.weight");
        if (!shape(w.input_norm, 1024, 1) || !shape(w.post_norm, 1024, 1) || !shape(w.gate, 1024, 3584) ||
            !shape(w.up, 1024, 3584) || !shape(w.down, 3584, 1024))
            return false;
        if (w.full_attention) {
            w.attention = { field("self_attn.q_proj.weight"), field("self_attn.k_proj.weight"),
                            field("self_attn.v_proj.weight"), field("self_attn.o_proj.weight"),
                            field("self_attn.q_norm.weight"), field("self_attn.k_norm.weight") };
        } else {
            w.linear = { field("linear_attn.in_proj_qkv.weight"),
                         field("linear_attn.in_proj_z.weight"),
                         field("linear_attn.in_proj_a.weight"),
                         field("linear_attn.in_proj_b.weight"),
                         field("linear_attn.out_proj.weight"),
                         field("linear_attn.conv1d.weight"),
                         field("linear_attn.A_log"),
                         field("linear_attn.dt_bias"),
                         field("linear_attn.norm.weight") };
        }
        weights.layers.push_back(w);
    }
    return true;
}
bool embed_cpu(ggml_tensor * embedding, const std::vector<int> & ids, std::vector<float> & hidden) {
    hidden.clear();
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !embedding || ids.empty() || ids.size() > 16384 ||
        embedding->ne[0] < 1 || embedding->ne[0] > 8192 || embedding->ne[2] != 1 || embedding->ne[3] != 1 ||
        (embedding->type != GGML_TYPE_F32 && embedding->type != GGML_TYPE_F16))
        return false;
    const size_t width = size_t(embedding->ne[0]), bytes = ggml_row_size(embedding->type, width);
    if (ids.size() * width > 16 * 1024 * 1024) return false;
    for (int id : ids)
        if (id < 0 || id >= embedding->ne[1]) return false;
    std::vector<uint8_t> row(bytes);
    hidden.resize(ids.size() * width);
    for (size_t t = 0; t < ids.size(); ++t) {
        ggml_backend_tensor_get(embedding, row.data(), size_t(ids[t]) * bytes, bytes);
        if (embedding->type == GGML_TYPE_F32)
            std::memcpy(hidden.data() + t * width, row.data(), bytes);
        else
            ggml_get_type_traits(embedding->type)->to_float(row.data(), hidden.data() + t * width, width);
    }
    return std::all_of(hidden.begin(), hidden.end(), [](float x) { return std::isfinite(x); });
}
bool decoder_cpu(const decoder_weights & w, const attention_config & c, const float * input, const float * positions,
                 int batch, int tokens, int threads, const decoder_cache * previous, decoder_outputs & output) {
    output = {};
    if (!core_env::on("CRISPEMBED_QWEN35_OCR") || !input || !positions || !w.embedding || w.layers.empty() ||
        w.layers.size() > 64 || batch < 1 || batch > 4 || tokens < 1 || tokens > 4096 || threads < 1 || threads > 256 ||
        !std::isfinite(c.eps) || c.eps <= 0)
        return false;
    const int width = int(w.embedding->ne[0]), past = previous ? previous->tokens : 0;
    if (width < 1 || width > 8192 || size_t(batch) * tokens * width > 16 * 1024 * 1024 || past < 0 ||
        past + tokens > 4096 || !w.head || w.head->ne[1] < 1 || w.head->ne[1] > 300000 ||
        !shape(w.head, width, int(w.head->ne[1])))
        return false;
    if (previous && (previous->batch != batch || previous->layers.size() != w.layers.size())) return false;
    const size_t count = size_t(batch) * tokens * width;
    std::vector<float> hidden(input, input + count), normalized;
    if (!std::all_of(hidden.begin(), hidden.end(), [](float x) { return std::isfinite(x); })) return false;
    output.cache.batch = batch;
    output.cache.tokens = past + tokens;
    for (size_t i = 0; i < w.layers.size(); ++i) {
        const auto & layer = w.layers[i];
        const auto * cache = previous ? &previous->layers[i] : nullptr;
        if (!rms(hidden, layer.input_norm, width, c.eps, normalized)) return false;
        decoder_layer_cache next;
        std::vector<float> mixed;
        if (layer.full_attention) {
            attention_outputs attention;
            output.attention_inputs[int(i)] = normalized;
            if (cache && cache->attention.tokens != past) return false;
            if (!attention_cpu(layer.attention, c, normalized.data(), positions, batch, tokens, threads,
                               cache ? &cache->attention : nullptr, attention))
                return false;
            output.attention_keys[int(i)] = attention.k_norm;
            mixed = std::move(attention.attention);
            next.attention = std::move(attention.cache);
        } else {
            projected_linear_outputs linear;
            if (cache &&
                (!layer.linear.a_log || !layer.linear.norm || !layer.linear.conv ||
                 cache->conv.size() != size_t(batch) * size_t(3 * layer.linear.a_log->ne[0] * layer.linear.norm->ne[0] *
                                                              layer.linear.conv->ne[0]) ||
                 cache->recurrent.size() !=
                     size_t(batch) * layer.linear.a_log->ne[0] * layer.linear.norm->ne[0] * layer.linear.norm->ne[0]))
                return false;
            if (!linear_cpu(layer.linear, normalized.data(), batch, tokens, threads,
                            cache ? cache->conv.data() : nullptr, cache ? cache->recurrent.data() : nullptr, linear))
                return false;
            mixed = std::move(linear.attention);
            next.conv = std::move(linear.core.conv_history);
            next.recurrent = std::move(linear.core.state);
        }
        if (mixed.size() != count) return false;
        for (size_t j = 0; j < count; ++j) hidden[j] += mixed[j];
        if (!rms(hidden, layer.post_norm, width, c.eps, normalized) || !layer.gate || layer.gate->ne[1] < 1 ||
            layer.gate->ne[1] > 32768 || !shape(layer.gate, width, int(layer.gate->ne[1])) ||
            !shape(layer.up, width, int(layer.gate->ne[1])) || !shape(layer.down, int(layer.gate->ne[1]), width))
            return false;
        std::vector<std::vector<float>> projections;
        if (!project_cpu({ layer.gate, layer.up }, normalized.data(), width, batch * tokens, threads, projections))
            return false;
        for (size_t j = 0; j < projections[0].size(); ++j) {
            const float gate = projections[0][j];
            projections[0][j] = (gate / (1.0f + std::exp(-gate))) * projections[1][j];
        }
        const auto activated = std::move(projections[0]);
        projections.clear();
        if (!project_cpu({ layer.down }, activated.data(), int(layer.gate->ne[1]), batch * tokens, threads,
                         projections))
            return false;
        for (size_t j = 0; j < count; ++j) hidden[j] += projections[0][j];
        output.hidden.push_back(hidden);
        output.cache.layers.push_back(std::move(next));
    }
    if (!rms(hidden, w.norm, width, c.eps, output.norm)) return false;
    std::vector<float> last(size_t(batch) * width);
    for (int b = 0; b < batch; ++b)
        std::copy_n(output.norm.data() + size_t((b * tokens) + tokens - 1) * width, width,
                    last.data() + size_t(b) * width);
    std::vector<std::vector<float>> logits;
    if (!project_cpu({ w.head }, last.data(), width, batch, threads, logits)) return false;
    output.logits = std::move(logits[0]);
    return true;
}
} // namespace qwen35_ocr
