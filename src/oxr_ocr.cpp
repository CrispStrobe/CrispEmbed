// OXR-1.0 native inference. See docs/oxr.md for source pins and qualification.
// No Python model or inference service is used by this implementation.
#include "oxr_ocr.h"
#include "oxr_graph_ops.h"
#include "oxr_model_profile.h"
#include "core/bpe.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "core/nfc.h"
#include "core/token_utf8.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct oxr_ocr_context {
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * cache_ctx = nullptr;
    ggml_backend_buffer_t cache_buffer = nullptr;
    ggml_tensor *cache_k = nullptr, *cache_v = nullptr;
    ggml_context * vte_ctx = nullptr;
    ggml_backend_buffer_t vte_buffer = nullptr;
    ggml_tensor * vte_transposed = nullptr;
    int cache_capacity = 0;
    int threads = 1, max_tokens = 2048, effective_max_tokens = 0;
    int min_pixels = oxr_profile::min_pixels, max_pixels = oxr_profile::vllm_max_pixels;
    float repetition_penalty = 1.0f;
    bool hf_generation = false;
    bool flash = true, reuse_decode = false;
    bool last_ok = false, eos = false, capped = false;
    std::string prompt = "Recognize the content of the image.", text, raw_text, error;
    std::vector<int32_t> prompt_ids, completion_ids;
    std::vector<float> confidences, position_table;
    int32_t grid[3] = { 0, 0, 0 };
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int32_t> token_to_id, merge_rank;
    oxr_ocr_stage_observer observer = nullptr;
    void * observer_user = nullptr;
    ~oxr_ocr_context() {
        if (sched) ggml_backend_sched_free(sched);
        if (cache_buffer) ggml_backend_buffer_free(cache_buffer);
        if (cache_ctx) ggml_free(cache_ctx);
        if (vte_buffer) ggml_backend_buffer_free(vte_buffer);
        if (vte_ctx) ggml_free(vte_ctx);
        core_gguf::free_weights(weights);
        if (cpu && cpu != backend) ggml_backend_free(cpu);
        if (backend) ggml_backend_free(backend);
    }
};

namespace {
constexpr int graph_nodes = 8192;
constexpr int vdim = 1152, vheads = 16, vhead_dim = 72, vlayers = 27;
constexpr int dim = 1024, layers = 28, heads = 16, kv_heads = 8, head_dim = 128;
// The converter stores the source prefix "visual_tokenizer.vit.vision_model." as
// "vit." (GGML_MAX_NAME) and records it in oxr.vision_tensor_prefix.
constexpr const char * vision = "vit.";

struct observation {
    std::string name;
    ggml_tensor * tensor;
    std::vector<int64_t> shape;
};
struct graph {
    oxr_ocr_context & owner;
    std::vector<uint8_t> metadata;
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    std::vector<observation> observations;
    graph(oxr_ocr_context & c) : owner(c) {
        ggml_backend_sched_reset(owner.sched);
        metadata.resize(ggml_tensor_overhead() * graph_nodes + ggml_graph_overhead_custom(graph_nodes, false) + 4096);
        ctx = ggml_init({ metadata.size(), metadata.data(), true });
        if (!ctx) throw std::runtime_error("OXR graph metadata allocation failed");
        gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    }
    ~graph() {
        ggml_backend_sched_reset(owner.sched);
        if (ctx) ggml_free(ctx);
    }
    void observe(const std::string & name, ggml_tensor * t, std::initializer_list<int64_t> shape) {
        if (!owner.observer) return;
        // A view (e.g. ggml_set_rows' result) shares its source's buffer, which
        // ggml_set_output does not protect: the allocator reused llm.input's rows
        // for a later layer, so the observer read llm.layer.1. Copy views.
        if (!ggml_is_contiguous(t) || t->view_src) t = ggml_cont(ctx, t);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        observations.push_back({ name, t, shape });
    }
    void output(ggml_tensor * t) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    void allocate() {
        if (!ggml_backend_sched_alloc_graph(owner.sched, gf)) throw std::runtime_error("OXR graph allocation failed");
    }
    void compute() {
        ggml_backend_cpu_set_n_threads(owner.cpu, owner.threads);
        if (ggml_backend_sched_graph_compute(owner.sched, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("OXR graph execution failed");
        for (const auto & o : observations) {
            std::vector<float> values(size_t(ggml_nelements(o.tensor)));
            ggml_backend_tensor_get(o.tensor, values.data(), 0, values.size() * sizeof(float));
            owner.observer(owner.observer_user, o.name.c_str(), values.data(), values.size(), o.shape.data(),
                           int(o.shape.size()));
        }
    }
};

ggml_tensor * weight(oxr_ocr_context & c, const std::string & key) {
    const auto it = c.weights.tensors.find(key);
    if (it == c.weights.tensors.end()) throw std::runtime_error("OXR missing tensor: " + key);
    return it->second;
}
ggml_tensor * mm(ggml_context * g, ggml_tensor * w, ggml_tensor * x) {
    auto * out = ggml_mul_mat(g, w, x);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}
ggml_tensor * linear(graph & g, ggml_tensor * x, const std::string & base, bool bias) {
    auto * out = mm(g.ctx, weight(g.owner, base + ".weight"), x);
    return bias ? ggml_add(g.ctx, out, weight(g.owner, base + ".bias")) : out;
}
ggml_tensor * norm(graph & g, ggml_tensor * x, const std::string & base, float epsilon, bool rms) {
    x = rms ? ggml_rms_norm(g.ctx, x, epsilon) : ggml_norm(g.ctx, x, epsilon);
    x = ggml_mul(g.ctx, x, weight(g.owner, base + ".weight"));
    return rms ? x : ggml_add(g.ctx, x, weight(g.owner, base + ".bias"));
}
void set(ggml_tensor * t, const void * data, size_t bytes) {
    ggml_backend_tensor_set(t, data, 0, bytes);
}
ggml_tensor * input(graph & g, const char * name, ggml_type type, int64_t a, int64_t b = 1, int64_t c = 1) {
    auto * t = ggml_new_tensor_3d(g.ctx, type, a, b, c);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}
std::vector<float> read(ggml_tensor * t) {
    std::vector<float> data(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
    return data;
}

void validate_tensors(oxr_ocr_context & c, const std::string & recipe) {
    size_t count = 0;
    auto check = [&](const std::string & name, std::initializer_list<int64_t> hf_shape, bool sensitive) {
        auto * t = weight(c, name);
        const std::vector<int64_t> shape(hf_shape);
        for (int i = 0; i < 4; ++i) {
            const int64_t n = size_t(i) < shape.size() ? shape[shape.size() - 1 - size_t(i)] : 1;
            if (t->ne[i] != n) throw std::runtime_error("OXR incompatible tensor shape: " + name);
        }
        const ggml_type expected_type = sensitive || recipe == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        // crispembed-quantize changes only these attention/MLP matrices (all used
        // through ggml_mul_mat); sensitive tensors must keep their recipe type.
        const bool quantized_matrix = !sensitive && (ggml_is_quantized(t->type) || t->type == GGML_TYPE_F16);
        if (t->type != expected_type && !quantized_matrix)
            throw std::runtime_error("OXR incompatible tensor precision: " + name);
        ++count;
    };
    check("llm.model.embed_tokens.weight", { 151936, 1024 }, true);
    check("llm.lm_head.weight", { 151936, 1024 }, true);
    check("llm.model.norm.weight", { 1024 }, true);
    check("vte.weight", { 65536, 1024 }, true);
    check("visual_tokenizer.head.0.weight", { 65532, 4608 }, true);
    check("visual_tokenizer.head.1.weight", { 65532 }, true);
    check("visual_tokenizer.head.1.bias", { 65532 }, true);
    check(std::string(vision) + "embeddings.patch_embedding.weight", { 1152, 3, 16, 16 }, true);
    check(std::string(vision) + "embeddings.patch_embedding.bias", { 1152 }, true);
    check(std::string(vision) + "embeddings.position_embedding.weight", { 1024, 1152 }, true);
    check(std::string(vision) + "post_layernorm.weight", { 1152 }, true);
    check(std::string(vision) + "post_layernorm.bias", { 1152 }, true);
    for (int i = 0; i < vlayers; ++i) {
        const std::string p = std::string(vision) + "encoder.layers." + std::to_string(i) + ".";
        for (const char * n : { "layer_norm1", "layer_norm2" }) {
            check(p + n + ".weight", { 1152 }, true);
            check(p + n + ".bias", { 1152 }, true);
        }
        for (const char * n : { "q_proj", "k_proj", "v_proj", "out_proj" }) {
            check(p + "self_attn." + n + ".weight", { 1152, 1152 }, false);
            check(p + "self_attn." + n + ".bias", { 1152 }, true);
        }
        check(p + "mlp.fc1.weight", { 4304, 1152 }, false);
        check(p + "mlp.fc1.bias", { 4304 }, true);
        check(p + "mlp.fc2.weight", { 1152, 4304 }, false);
        check(p + "mlp.fc2.bias", { 1152 }, true);
    }
    for (int i = 0; i < layers; ++i) {
        const std::string p = "llm.model.layers." + std::to_string(i) + ".";
        check(p + "input_layernorm.weight", { 1024 }, true);
        check(p + "post_attention_layernorm.weight", { 1024 }, true);
        check(p + "self_attn.q_norm.weight", { 128 }, true);
        check(p + "self_attn.k_norm.weight", { 128 }, true);
        check(p + "self_attn.q_proj.weight", { 2048, 1024 }, false);
        check(p + "self_attn.k_proj.weight", { 1024, 1024 }, false);
        check(p + "self_attn.v_proj.weight", { 1024, 1024 }, false);
        check(p + "self_attn.o_proj.weight", { 1024, 2048 }, false);
        check(p + "mlp.gate_proj.weight", { 3072, 1024 }, false);
        check(p + "mlp.up_proj.weight", { 3072, 1024 }, false);
        check(p + "mlp.down_proj.weight", { 1024, 3072 }, false);
    }
    if (count != 752 || c.weights.tensors.size() != count)
        throw std::runtime_error("OXR tensor inventory differs from pinned checkpoint");
}

void load(oxr_ocr_context & c, const char * path) {
    if (!core_env::on(oxr_profile::gate)) throw std::runtime_error("OXR requires CRISPEMBED_OXR=1");
    if (!path || !*path) throw std::invalid_argument("OXR requires an explicit GGUF path");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(core_gguf::open_metadata(path), gguf_free);
    if (!meta) throw std::runtime_error("OXR could not read GGUF metadata");
    auto str = [&](const char * key) { return core_gguf::kv_str(meta.get(), key, ""); };
    if (str("general.architecture") != "oxr" || str("oxr.model_family") != oxr_profile::family ||
        str("oxr.source.repository") != "SoMarkAI/OXR-1.0" ||
        str("oxr.blueprint.revision") != "c1a1dcb312ee9f691404f127785510453c0d9350" ||
        str("oxr.generation.repetition_history") != "prompt_and_completion" ||
        str("oxr.source.revision") != oxr_profile::revision || str("oxr.processor") != "siglip-pil-bilinear" ||
        str("oxr.tokenizer.normalizer") != "NFC" || str("tokenizer.ggml.model") != "gpt2" ||
        str("tokenizer.ggml.pre") != "qwen2")
        throw std::runtime_error("OXR requires the pinned oxr-1.0 conversion/preprocessor profile");
    const std::string recipe = str("oxr.conversion.recipe");
    if (recipe != "f32" && recipe != "f16-f32-sensitive") throw std::runtime_error("OXR unsupported conversion recipe");
    const std::pair<const char *, uint32_t> integers[] = {
        { "oxr.llm.hidden_size", 1024 },        { "oxr.llm.intermediate_size", 3072 },
        { "oxr.llm.num_hidden_layers", 28 },    { "oxr.llm.num_attention_heads", 16 },
        { "oxr.llm.num_key_value_heads", 8 },   { "oxr.llm.head_dim", 128 },
        { "oxr.llm.vocab_size", 151936 },       { "oxr.llm.max_position_embeddings", 32768 },
        { "oxr.vision.hidden_size", 1152 },     { "oxr.vision.intermediate_size", 4304 },
        { "oxr.vision.num_hidden_layers", 27 }, { "oxr.vision.num_attention_heads", 16 },
        { "oxr.vision.patch_size", 16 },        { "oxr.vision.hidden_stride", 2 },
        { "oxr.vision.image_size", 512 },       { "oxr.vision.window_size", 112 },
        { "oxr.context_length", 32768 },        { "oxr.visual_vocab_size", 65536 },
        { "oxr.tokenizer.vocab_size", 151669 }, { "oxr.processor.min_pixels", 200704 },
        { "oxr.processor.max_pixels", 921600 }
    };
    if (core_gguf::kv_str(meta.get(), "oxr.vision_tensor_prefix", "") != vision)
        throw std::runtime_error("OXR GGUF predates the short vision tensor names; reconvert it");
    for (const auto & item : integers)
        if (core_gguf::kv_u32(meta.get(), item.first, 0) != item.second)
            throw std::runtime_error(std::string("OXR incompatible metadata: ") + item.first);
    for (const auto & item : std::vector<std::pair<const char *, float>>{ { "oxr.llm.rms_norm_eps", 1e-6f },
                                                                          { "oxr.llm.rope_theta", 1e6f },
                                                                          { "oxr.vision.layer_norm_eps", 1e-6f },
                                                                          { "oxr.visual_head.layer_norm_eps", 1e-5f } })
        if (core_gguf::kv_f32(meta.get(), item.first, 0.0f) != item.second)
            throw std::runtime_error(std::string("OXR incompatible metadata: ") + item.first);
    c.vocab = core_gguf::kv_str_array(meta.get(), "tokenizer.ggml.tokens");
    auto merges = core_gguf::kv_str_array(meta.get(), "tokenizer.ggml.merges");
    if (c.vocab.size() != oxr_profile::real_vocab || merges.size() != 151387)
        throw std::runtime_error("OXR tokenizer inventory mismatch");
    const auto eos = core_gguf::kv_i32_array(meta.get(), "oxr.generation.eos_token_ids");
    const auto types = core_gguf::kv_i32_array(meta.get(), "tokenizer.ggml.token_type");
    if (eos != std::vector<int>({ 151643, 151645 }) || types.size() != c.vocab.size())
        throw std::runtime_error("OXR EOS or tokenizer-type inventory mismatch");
    for (size_t i = 0; i < types.size(); ++i)
        if (types[i] != (i < 151643 ? 1 : i <= 151656 ? 3 : 4))
            throw std::runtime_error("OXR tokenizer-type flags differ from pinned source");
    const char * added[] = {
        "<|endoftext|>",  "<|im_start|>",   "<|im_end|>",      "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>",  "<|box_end|>",    "<|quad_start|>",  "<|quad_end|>",         "<|vision_start|>",
        "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",   "<|video_pad|>",        "<tool_call>",
        "</tool_call>",   "<|fim_prefix|>", "<|fim_middle|>",  "<|fim_suffix|>",       "<|fim_pad|>",
        "<|repo_name|>",  "<|file_sep|>",   "<tool_response>", "</tool_response>",     "<think>",
        "</think>"
    };
    for (int i = 0; i < 26; ++i)
        if (c.vocab[size_t(151643 + i)] != added[i]) throw std::runtime_error("OXR added-token IDs differ from source");
    for (size_t i = 0; i < c.vocab.size(); ++i)
        if (!c.token_to_id.emplace(c.vocab[i], int32_t(i)).second)
            throw std::runtime_error("OXR duplicate vocabulary entry");
    for (size_t i = 0; i < merges.size(); ++i)
        if (!c.merge_rank.emplace(merges[i], int32_t(i)).second) throw std::runtime_error("OXR duplicate merge entry");
    c.flash = !core_env::explicitly_off("CRISPEMBED_OXR_FLASH_ATTN");
    c.reuse_decode = core_env::on("CRISPEMBED_OXR_REUSE_DECODE_GRAPH");
    c.backend = core_env::on("CRISPEMBED_OXR_GPU") && !core_env::on("CRISPEMBED_OXR_FORCE_CPU")
                    ? crispasr_init_gpu_backend()
                    : ggml_backend_cpu_init();
    if (!c.backend) throw std::runtime_error("OXR backend initialization failed");
    c.cpu = ggml_backend_is_cpu(c.backend) ? c.backend : ggml_backend_cpu_init();
    if (!c.cpu) throw std::runtime_error("OXR CPU fallback initialization failed");
    ggml_backend_cpu_set_n_threads(c.cpu, c.threads);
    if (!core_gguf::load_weights(path, c.backend, "oxr", c.weights))
        throw std::runtime_error("OXR GGUF weight loading failed");
    validate_tensors(c, recipe);
    c.position_table = read(weight(c, std::string(vision) + "embeddings.position_embedding.weight"));
    ggml_backend_t backends[] = { c.backend, c.cpu };
    c.sched = ggml_backend_sched_new(backends, nullptr, c.backend == c.cpu ? 1 : 2, graph_nodes, false, false);
    if (!c.sched) throw std::runtime_error("OXR scheduler initialization failed");
    // The visual-probability projection needs a contiguous [65536,1024]
    // matrix in GGML order. A merely transposed RHS violates CPU mul_mat's
    // contiguous-element stride contract. Transpose once per resident model,
    // retaining a 256MiB F32 buffer instead of redoing it for every image/chunk.
    c.vte_ctx = ggml_init({ ggml_tensor_overhead() + 1024, nullptr, true });
    if (!c.vte_ctx) throw std::runtime_error("OXR visual embedding metadata allocation failed");
    c.vte_transposed = ggml_new_tensor_2d(c.vte_ctx, GGML_TYPE_F32, oxr_profile::visual_vocab, dim);
    c.vte_buffer = ggml_backend_alloc_ctx_tensors(c.vte_ctx, c.backend);
    if (!c.vte_buffer) throw std::runtime_error("OXR visual embedding transpose allocation failed");
    graph transpose(c);
    auto * copy = ggml_cpy(transpose.ctx, ggml_transpose(transpose.ctx, weight(c, "vte.weight")), c.vte_transposed);
    transpose.output(copy);
    transpose.allocate();
    transpose.compute();
}

std::vector<int32_t> tokenize(oxr_ocr_context & c, const std::string & raw) {
    const std::string text = core_nfc::normalize(raw);
    std::vector<int32_t> result;
    size_t cursor = 0;
    while (cursor < text.size()) {
        size_t next = text.size();
        int special = -1;
        for (int id = 151643; id < oxr_profile::real_vocab; ++id) {
            const auto where = text.find(c.vocab[size_t(id)], cursor);
            if (where < next) {
                next = where;
                special = id;
            }
        }
        const auto ordinary = core_bpe::tokenize_qwen(c.token_to_id, c.merge_rank, text.substr(cursor, next - cursor));
        result.insert(result.end(), ordinary.begin(), ordinary.end());
        if (special < 0) break;
        result.push_back(special);
        cursor = next + c.vocab[size_t(special)].size();
    }
    return result;
}

ggml_tensor * attention(graph & g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, int hd,
                        int nh, int nkv, int tq, int tk) {
    const float scale = 1.0f / std::sqrt(float(hd));
    if (g.owner.flash) {
        auto * out = ggml_flash_attn_ext(g.ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        return ggml_reshape_2d(g.ctx, out, int64_t(hd) * nh, tq);
    }
    if (nkv != nh) {
        const int repeats = nh / nkv;
        auto expand = [&](ggml_tensor * t) {
            t = ggml_reshape_4d(g.ctx, t, hd, tk, 1, nkv);
            t = ggml_repeat(g.ctx, t, ggml_new_tensor_4d(g.ctx, t->type, hd, tk, repeats, nkv));
            return ggml_reshape_3d(g.ctx, t, hd, tk, nh);
        };
        // Cache views have gaps between heads, so make the read span contiguous
        // before flattening head groups for explicit repeat_kv.
        k = expand(ggml_cont(g.ctx, k));
        v = expand(ggml_cont(g.ctx, v));
    }
    auto * scores = mm(g.ctx, k, q);
    scores = ggml_soft_max_ext(g.ctx, scores, mask, scale, 0.0f);
    auto * vt = ggml_cont(g.ctx, ggml_permute(g.ctx, v, 1, 0, 2, 3));
    auto * out = mm(g.ctx, vt, scores);
    out = ggml_cont(g.ctx, ggml_permute(g.ctx, out, 0, 2, 1, 3));
    return ggml_reshape_2d(g.ctx, out, int64_t(hd) * nh, tq);
}

std::vector<float> encode_vision(oxr_ocr_context & c, const oxr_profile::image_input & image) {
    const int gh = image.grid[1], gw = image.grid[2], n = gh * gw, nmerged = n / 4;
    if (c.observer) {
        const int64_t shape[] = { n, 768 };
        c.observer(c.observer_user, "input.patches", image.patches.data(), image.patches.size(), shape, 2);
    }
    const auto pos = oxr_profile::position_embeddings(c.position_table.data(), gh, gw);
    const auto order = oxr_profile::window_groups(gh, gw);
    std::vector<int32_t> reverse(size_t(nmerged), 0);
    for (int i = 0; i < nmerged; ++i) reverse[size_t(order[size_t(i)])] = i;
    std::vector<float> cos(size_t(n) * vhead_dim), sin(cos.size());
    for (int i = 0; i < nmerged; ++i) {
        const int original = order[size_t(i)], my = original / (gw / 2), mx = original % (gw / 2);
        for (int sub = 0; sub < 4; ++sub) {
            const int y = my * 2 + sub / 2, x = mx * 2 + sub % 2;
            for (int d = 0; d < vhead_dim; ++d) {
                const int j = d % (vhead_dim / 2), which = j / (vhead_dim / 4), f = j % (vhead_dim / 4);
                const float inv = 1.0f / std::pow(10000.0f, float(2 * f) / float(vhead_dim / 2));
                const float angle = float(which == 0 ? y : x) * inv;
                cos[size_t(i * 4 + sub) * vhead_dim + d] = std::cos(angle);
                sin[size_t(i * 4 + sub) * vhead_dim + d] = std::sin(angle);
            }
        }
    }
    std::vector<float> features;
    {
        graph g(c);
        auto * patches = input(g, "oxr.patches", GGML_TYPE_F32, 768, n);
        auto * positions = input(g, "oxr.vision.positions", GGML_TYPE_F32, vdim, n);
        auto * perm = input(g, "oxr.vision.permutation", GGML_TYPE_I32, nmerged);
        auto * undo = input(g, "oxr.vision.inverse", GGML_TYPE_I32, nmerged);
        auto * tcos = input(g, "oxr.vision.cos", GGML_TYPE_F32, vhead_dim, 1, n);
        auto * tsin = input(g, "oxr.vision.sin", GGML_TYPE_F32, vhead_dim, 1, n);
        auto * patch_weight =
            ggml_reshape_2d(g.ctx, weight(c, std::string(vision) + "embeddings.patch_embedding.weight"), 768, vdim);
        auto * x = ggml_add(g.ctx, mm(g.ctx, patch_weight, patches),
                            weight(c, std::string(vision) + "embeddings.patch_embedding.bias"));
        g.observe("vision.patch_embed", x, { n, vdim });
        g.observe("vision.position", positions, { n, vdim });
        x = ggml_add(g.ctx, x, positions);
        g.observe("vision.embedding", x, { n, vdim });
        x = ggml_reshape_2d(g.ctx, ggml_get_rows(g.ctx, ggml_reshape_2d(g.ctx, x, vdim * 4, nmerged), perm), vdim, n);
        auto restore = [&](ggml_tensor * value) {
            return ggml_reshape_2d(g.ctx, ggml_get_rows(g.ctx, ggml_reshape_2d(g.ctx, value, vdim * 4, nmerged), undo),
                                   vdim, n);
        };
        auto rope = [&](ggml_tensor * value) {
            auto * first = ggml_view_3d(g.ctx, value, vhead_dim / 2, vheads, n, value->nb[1], value->nb[2], 0);
            auto * second = ggml_view_3d(g.ctx, value, vhead_dim / 2, vheads, n, value->nb[1], value->nb[2],
                                         size_t(vhead_dim / 2) * sizeof(float));
            auto * rotated = ggml_concat(g.ctx, ggml_neg(g.ctx, second), first, 0);
            return ggml_add(g.ctx, ggml_mul(g.ctx, value, tcos), ggml_mul(g.ctx, rotated, tsin));
        };
        for (int i = 0; i < vlayers; ++i) {
            const std::string p = std::string(vision) + "encoder.layers." + std::to_string(i) + ".";
            auto * residual = x;
            auto * normalized = norm(g, x, p + "layer_norm1", 1e-6f, false);
            auto * q =
                rope(ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.q_proj", true), vhead_dim, vheads, n));
            auto * k =
                rope(ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.k_proj", true), vhead_dim, vheads, n));
            auto * v =
                ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.v_proj", true), vhead_dim, vheads, n);
            q = ggml_cont(g.ctx, ggml_permute(g.ctx, q, 0, 2, 1, 3));
            k = ggml_cont(g.ctx, ggml_permute(g.ctx, k, 0, 2, 1, 3));
            v = ggml_cont(g.ctx, ggml_permute(g.ctx, v, 0, 2, 1, 3));
            // fullatt_block_indexes=null means FULL attention in every layer.
            auto * attn = attention(g, q, k, v, nullptr, vhead_dim, vheads, vheads, n, n);
            x = ggml_add(g.ctx, residual, linear(g, attn, p + "self_attn.out_proj", true));
            residual = x;
            normalized = norm(g, x, p + "layer_norm2", 1e-6f, false);
            auto * mlp = oxr_graph_ops::gelu_tanh_f32(g.ctx, linear(g, normalized, p + "mlp.fc1", true));
            x = ggml_add(g.ctx, residual, linear(g, mlp, p + "mlp.fc2", true));
            if (c.observer) g.observe("vision.layer." + std::to_string(i), restore(x), { n, vdim });
        }
        // OXR takes output.hidden_states[-1], BEFORE the stored post_layernorm.
        x = restore(x);
        auto * merged = ggml_reshape_2d(g.ctx, x, vdim * 4, nmerged);
        g.observe("vision.merge", merged, { nmerged, vdim * 4 });
        g.output(merged);
        g.allocate();
        set(patches, image.patches.data(), image.patches.size() * sizeof(float));
        set(positions, pos.data(), pos.size() * sizeof(float));
        set(perm, order.data(), order.size() * sizeof(int32_t));
        set(undo, reverse.data(), reverse.size() * sizeof(int32_t));
        set(tcos, cos.data(), cos.size() * sizeof(float));
        set(tsin, sin.data(), sin.size() * sizeof(float));
        g.compute();
        features = read(merged);
    }
    // The 65k visual vocabulary is row-independent. Bounded chunks avoid
    // materializing all probabilities at once for high-resolution pages.
    constexpr int chunk = 256;
    std::vector<float> embeddings(size_t(nmerged) * dim);
    std::vector<float> observed_logits, observed_probs;
    if (c.observer) {
        observed_logits.resize(size_t(nmerged) * oxr_profile::visual_logits);
        observed_probs.resize(observed_logits.size());
    }
    for (int start = 0; start < nmerged; start += chunk) {
        const int rows = std::min(chunk, nmerged - start);
        graph g(c);
        auto * in = input(g, "oxr.visual.features", GGML_TYPE_F32, vdim * 4, rows);
        auto * logits =
            norm(g, linear(g, in, "visual_tokenizer.head.0", false), "visual_tokenizer.head.1", 1e-5f, false);
        auto * probs = ggml_soft_max(g.ctx, logits);
        auto * padded = ggml_pad(g.ctx, probs, 4, 0, 0, 0);
        auto * projected = mm(g.ctx, c.vte_transposed, padded);
        if (c.observer) {
            ggml_set_output(logits);
            ggml_set_output(probs);
        }
        g.output(projected);
        g.allocate();
        set(in, features.data() + size_t(start) * vdim * 4, size_t(rows) * vdim * 4 * sizeof(float));
        g.compute();
        ggml_backend_tensor_get(projected, embeddings.data() + size_t(start) * dim, 0,
                                size_t(rows) * dim * sizeof(float));
        if (c.observer) {
            ggml_backend_tensor_get(logits, observed_logits.data() + size_t(start) * oxr_profile::visual_logits, 0,
                                    size_t(rows) * oxr_profile::visual_logits * sizeof(float));
            ggml_backend_tensor_get(probs, observed_probs.data() + size_t(start) * oxr_profile::visual_logits, 0,
                                    size_t(rows) * oxr_profile::visual_logits * sizeof(float));
        }
    }
    if (c.observer) {
        const int64_t head_shape[] = { nmerged, oxr_profile::visual_logits }, emb_shape[] = { nmerged, dim };
        c.observer(c.observer_user, "visual.logits", observed_logits.data(), observed_logits.size(), head_shape, 2);
        c.observer(c.observer_user, "visual.probs", observed_probs.data(), observed_probs.size(), head_shape, 2);
        c.observer(c.observer_user, "visual.embeddings", embeddings.data(), embeddings.size(), emb_shape, 2);
    }
    return embeddings;
}

void allocate_cache(oxr_ocr_context & c, int wanted) {
    wanted = std::min(oxr_profile::context_length, ((wanted + 255) / 256) * 256);
    if (c.cache_capacity >= wanted) return;
    if (c.cache_buffer) {
        ggml_backend_buffer_free(c.cache_buffer);
        c.cache_buffer = nullptr;
    }
    if (c.cache_ctx) {
        ggml_free(c.cache_ctx);
        c.cache_ctx = nullptr;
    }
    c.cache_capacity = 0;
    c.cache_ctx = ggml_init({ 2 * ggml_tensor_overhead() + 1024, nullptr, true });
    if (!c.cache_ctx) throw std::runtime_error("OXR KV metadata allocation failed");
    c.cache_k = ggml_new_tensor_4d(c.cache_ctx, GGML_TYPE_F32, head_dim, wanted, kv_heads, layers);
    c.cache_v = ggml_new_tensor_4d(c.cache_ctx, GGML_TYPE_F32, head_dim, wanted, kv_heads, layers);
    c.cache_buffer = ggml_backend_alloc_ctx_tensors(c.cache_ctx, c.backend);
    if (!c.cache_buffer) throw std::runtime_error("OXR F32 KV cache allocation failed");
    ggml_backend_buffer_clear(c.cache_buffer, 0);
    c.cache_capacity = wanted;
}

struct decoder_graph {
    std::unique_ptr<graph> owner;
    ggml_tensor *ids = nullptr, *positions = nullptr, *cache_index = nullptr;
    ggml_tensor *mask = nullptr, *image_values = nullptr, *image_indices = nullptr, *logits = nullptr;
    int queries = 0, keys = 0;
};
decoder_graph build_decoder(oxr_ocr_context & c, int tokens, int past, int extent, int visual_count, bool capture) {
    decoder_graph result;
    result.owner = std::make_unique<graph>(c);
    auto & g = *result.owner;
    result.queries = tokens;
    result.keys = extent;
    result.ids = input(g, "oxr.llm.ids", GGML_TYPE_I32, tokens);
    result.positions = input(g, "oxr.llm.positions", GGML_TYPE_I32, tokens);
    result.mask = input(g, "oxr.llm.mask", c.flash ? GGML_TYPE_F16 : GGML_TYPE_F32, extent, tokens);
    result.cache_index = input(g, "oxr.llm.cache_index", GGML_TYPE_I32, tokens);
    auto * x = ggml_get_rows(g.ctx, weight(c, "llm.model.embed_tokens.weight"), result.ids);
    if (visual_count) {
        result.image_values = input(g, "oxr.llm.visual", GGML_TYPE_F32, dim, visual_count);
        result.image_indices = input(g, "oxr.llm.visual_indices", GGML_TYPE_I32, visual_count);
        x = ggml_set_rows(g.ctx, x, result.image_values, result.image_indices);
    }
    if (capture) g.observe("llm.input", x, { tokens, dim });
    for (int i = 0; i < layers; ++i) {
        const std::string p = "llm.model.layers." + std::to_string(i) + ".";
        auto * residual = x;
        auto * normalized = norm(g, x, p + "input_layernorm", 1e-6f, true);
        auto * q =
            ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.q_proj", false), head_dim, heads, tokens);
        auto * k =
            ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.k_proj", false), head_dim, kv_heads, tokens);
        auto * v =
            ggml_reshape_3d(g.ctx, linear(g, normalized, p + "self_attn.v_proj", false), head_dim, kv_heads, tokens);
        q = norm(g, q, p + "self_attn.q_norm", 1e-6f, true);
        k = norm(g, k, p + "self_attn.k_norm", 1e-6f, true);
        q = ggml_rope_ext(g.ctx, q, result.positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        k = ggml_rope_ext(g.ctx, k, result.positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, 1e6f, 1.0f, 0.0f, 1.0f,
                          0.0f, 0.0f);
        q = ggml_cont(g.ctx, ggml_permute(g.ctx, q, 0, 2, 1, 3));
        k = ggml_cont(g.ctx, ggml_permute(g.ctx, k, 0, 2, 1, 3));
        v = ggml_cont(g.ctx, ggml_permute(g.ctx, v, 0, 2, 1, 3));
        auto cache_view = [&](ggml_tensor * cache) {
            return ggml_view_3d(g.ctx, cache, head_dim, c.cache_capacity, kv_heads, cache->nb[1], cache->nb[2],
                                size_t(i) * cache->nb[3]);
        };
        auto * kw = ggml_set_rows(g.ctx, cache_view(c.cache_k), k, result.cache_index);
        auto * vw = ggml_set_rows(g.ctx, cache_view(c.cache_v), v, result.cache_index);
        auto * keys = ggml_view_3d(g.ctx, kw, head_dim, extent, kv_heads, kw->nb[1], kw->nb[2], 0);
        auto * values = ggml_view_3d(g.ctx, vw, head_dim, extent, kv_heads, vw->nb[1], vw->nb[2], 0);
        auto * attn = attention(g, q, keys, values, result.mask, head_dim, heads, kv_heads, tokens, extent);
        x = ggml_add(g.ctx, residual, linear(g, attn, p + "self_attn.o_proj", false));
        residual = x;
        normalized = norm(g, x, p + "post_attention_layernorm", 1e-6f, true);
        auto * gate = ggml_silu(g.ctx, linear(g, normalized, p + "mlp.gate_proj", false));
        auto * up = linear(g, normalized, p + "mlp.up_proj", false);
        x = ggml_add(g.ctx, residual, linear(g, ggml_mul(g.ctx, gate, up), p + "mlp.down_proj", false));
        if (capture) g.observe("llm.layer." + std::to_string(i), x, { tokens, dim });
    }
    x = norm(g, x, "llm.model.norm", 1e-6f, true);
    if (capture) g.observe("llm.norm", x, { tokens, dim });
    x = ggml_view_2d(g.ctx, x, dim, 1, x->nb[1], size_t(tokens - 1) * x->nb[1]);
    result.logits = mm(g.ctx, weight(c, "llm.lm_head.weight"), x);
    if (capture) g.observe("llm.logits", result.logits, { 1, oxr_profile::vocab });
    g.output(result.logits);
    g.allocate();
    (void)past;
    return result;
}

void decoder_inputs(decoder_graph & g, const std::vector<int32_t> & ids, int past, bool initial_mask,
                    const std::vector<int32_t> * original_prompt = nullptr) {
    std::vector<int32_t> safe_ids(ids), positions(ids.size());
    for (size_t i = 0; i < safe_ids.size(); ++i) {
        if (safe_ids[i] < 0) safe_ids[i] = 0;
        positions[i] = past + int(i);
        if (original_prompt) {
            int ignored = 0;
            for (int j = 0; j <= past + int(i) && size_t(j) < original_prompt->size(); ++j)
                ignored += (*original_prompt)[size_t(j)] == 151643 ? 1 : 0;
            positions[i] -= ignored;
            if (size_t(past + int(i)) < original_prompt->size() && (*original_prompt)[size_t(past + int(i))] == 151643)
                positions[i] = 1;
        }
    }
    set(g.ids, safe_ids.data(), safe_ids.size() * sizeof(int32_t));
    set(g.positions, positions.data(), positions.size() * sizeof(int32_t));
    std::vector<int32_t> cache_indices(ids.size());
    std::iota(cache_indices.begin(), cache_indices.end(), past);
    set(g.cache_index, cache_indices.data(), cache_indices.size() * sizeof(int32_t));
    auto allowed = [&](int row, int key) {
        return key <= past + row &&
               !(original_prompt && size_t(key) < original_prompt->size() && (*original_prompt)[size_t(key)] == 151643);
    };
    if (initial_mask || ids.size() != 1) {
        if (g.mask->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> mask(size_t(g.keys) * g.queries);
            for (int row = 0; row < g.queries; ++row)
                for (int key = 0; key < g.keys; ++key)
                    mask[size_t(row) * g.keys + key] = ggml_fp32_to_fp16(allowed(row, key) ? 0.0f : -INFINITY);
            set(g.mask, mask.data(), mask.size() * sizeof(ggml_fp16_t));
        } else {
            std::vector<float> mask(size_t(g.keys) * g.queries);
            for (int row = 0; row < g.queries; ++row)
                for (int key = 0; key < g.keys; ++key)
                    mask[size_t(row) * g.keys + key] = allowed(row, key) ? 0.0f : -INFINITY;
            set(g.mask, mask.data(), mask.size() * sizeof(float));
        }
    } else {
        // A reused single-token graph only unmasks the newly written cache row.
        if (g.mask->type == GGML_TYPE_F16) {
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            ggml_backend_tensor_set(g.mask, &zero, size_t(past) * sizeof(zero), sizeof(zero));
        } else {
            const float zero = 0.0f;
            ggml_backend_tensor_set(g.mask, &zero, size_t(past) * sizeof(zero), sizeof(zero));
        }
    }
}

void prepare_request(oxr_ocr_context & c, int image_tokens) {
    if (image_tokens) {
        c.prompt_ids = tokenize(c, "<|im_start|>user\n");
        c.prompt_ids.push_back(oxr_profile::image_begin);
        c.prompt_ids.insert(c.prompt_ids.end(), size_t(image_tokens), oxr_profile::visual_atom);
        c.prompt_ids.push_back(oxr_profile::image_end);
        auto suffix = tokenize(c, c.prompt + "<|im_end|>\n<|im_start|>assistant\n");
        c.prompt_ids.insert(c.prompt_ids.end(), suffix.begin(), suffix.end());
    } else {
        c.prompt_ids = tokenize(c, "<|im_start|>user\n" + c.prompt + "<|im_end|>\n<|im_start|>assistant\n");
    }
    const int prompt_length = int(c.prompt_ids.size());
    if (prompt_length <= 0 || prompt_length >= oxr_profile::context_length)
        throw std::runtime_error("OXR prompt exceeds supported 32768-token total context");
    const int remaining = oxr_profile::context_length - prompt_length;
    c.effective_max_tokens = c.max_tokens;
    if (c.effective_max_tokens <= 0 || c.effective_max_tokens > remaining)
        throw std::runtime_error(
            "OXR requested completion exceeds remaining 32768-token context; request a smaller cap");
}

void generate(oxr_ocr_context & c, const std::vector<float> & image_embeddings) {
    const bool has_image = !image_embeddings.empty();
    const int visual_count = has_image ? int(image_embeddings.size() / dim) + 2 : 0;
    const int prompt_length = int(c.prompt_ids.size());
    std::vector<int32_t> indices;
    std::vector<float> visual;
    if (has_image) {
        const int first =
            int(std::find(c.prompt_ids.begin(), c.prompt_ids.end(), oxr_profile::image_begin) - c.prompt_ids.begin());
        indices.resize(size_t(visual_count));
        std::iota(indices.begin(), indices.end(), first);
        visual.resize(size_t(visual_count) * dim);
        ggml_backend_tensor_get(weight(c, "vte.weight"), visual.data(), size_t(65532) * dim * sizeof(float),
                                dim * sizeof(float));
        std::copy(image_embeddings.begin(), image_embeddings.end(), visual.begin() + dim);
        ggml_backend_tensor_get(weight(c, "vte.weight"), visual.data() + size_t(visual_count - 1) * dim,
                                size_t(65533) * dim * sizeof(float), dim * sizeof(float));
    }
    allocate_cache(c, prompt_length + c.effective_max_tokens);
    std::vector<float> logits;
    {
        auto prefill = build_decoder(c, prompt_length, 0, prompt_length, visual_count, c.observer != nullptr);
        decoder_inputs(prefill, c.prompt_ids, 0, true, c.hf_generation ? &c.prompt_ids : nullptr);
        if (visual_count) {
            set(prefill.image_values, visual.data(), visual.size() * sizeof(float));
            set(prefill.image_indices, indices.data(), indices.size() * sizeof(int32_t));
        }
        prefill.owner->compute();
        logits = read(prefill.logits);
    }
    std::vector<int32_t> history;
    if (!c.hf_generation) {
        history.reserve(size_t(prompt_length + c.effective_max_tokens));
        for (int32_t id : c.prompt_ids) history.push_back(oxr_profile::vllm_penalty_id(id));
    }
    std::unique_ptr<decoder_graph> cached;
    for (int step = 0; step < c.effective_max_tokens; ++step) {
        oxr_profile::repetition_penalty(logits, history, c.repetition_penalty);
        if (std::any_of(logits.begin(), logits.end(), [](float x) { return !std::isfinite(x); }))
            throw std::runtime_error("OXR decoder produced nonfinite logits");
        const int id = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        double denominator = 0.0;
        for (float x : logits) denominator += std::exp(double(x - logits[size_t(id)]));
        c.completion_ids.push_back(id);
        c.confidences.push_back(float(1.0 / denominator));
        history.push_back(id);
        if (oxr_profile::eos(id)) {
            c.eos = true;
            break;
        }
        if (step + 1 == c.effective_max_tokens) {
            c.capped = true;
            break;
        }
        const int past = prompt_length + step;
        const int extent = c.reuse_decode ? std::min(c.cache_capacity, ((past + 1 + 255) / 256) * 256) : past + 1;
        bool fresh = true;
        if (c.reuse_decode && cached && cached->keys == extent)
            fresh = false;
        else {
            // Destroy the old scheduler-owned graph before allocating the next.
            cached.reset();
            cached = std::make_unique<decoder_graph>(build_decoder(c, 1, past, extent, 0, false));
        }
        decoder_inputs(*cached, { id }, past, fresh, c.hf_generation ? &c.prompt_ids : nullptr);
        cached->owner->compute();
        logits = read(cached->logits);
    }
    std::string raw, filtered;
    for (int32_t id : c.completion_ids) {
        // Tokenizers' decoder silently drops the checkpoint's unused output
        // IDs151669..151935; retain those IDs in diagnostics and history.
        if (id >= oxr_profile::real_vocab) continue;
        const std::string piece = core_bpe::unicode_to_bytes(c.vocab[size_t(id)]);
        raw += piece;
        if (!oxr_profile::special(id)) filtered += piece;
    }
    c.raw_text = core_utf8::token_utf8_lossy(raw.data(), int(raw.size()));
    c.text = core_utf8::token_utf8_lossy(filtered.data(), int(filtered.size()));
}

void clear_result(oxr_ocr_context & c) {
    c.last_ok = false;
    c.eos = c.capped = false;
    c.effective_max_tokens = 0;
    c.text.clear();
    c.raw_text.clear();
    c.error.clear();
    c.prompt_ids.clear();
    c.completion_ids.clear();
    c.confidences.clear();
    c.grid[0] = c.grid[1] = c.grid[2] = 0;
}
} // namespace

extern "C" {
oxr_ocr_context * oxr_ocr_init(const char * path, int n_threads) {
    // License gate (pattern of the Qwen research gates): the OXR License is Apache-2.0
    // plus additional terms (licenses/oxr-LICENSE) that bind every surface loading the weights.
    if (!core_env::on("CRISPEMBED_ACCEPT_OXR_LICENSE")) {
        std::fprintf(stderr,
                     "oxr: OXR weights are under the OXR License (Apache-2.0 with additional terms): commercial use\n"
                     "  needs a SoMark license above USD 2M total funding or annual revenue; online services must\n"
                     "  credit SoMark OXR; no hosted API competing with SoMark's, at any size. Set\n"
                     "  CRISPEMBED_ACCEPT_OXR_LICENSE=1 to acknowledge. See licenses/oxr-LICENSE.\n");
        return nullptr;
    }
    try {
        auto c = std::make_unique<oxr_ocr_context>();
        c->threads = std::max(1, n_threads);
        load(*c, path);
        return c.release();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "oxr: %s\n", e.what());
        return nullptr;
    }
}
void oxr_ocr_free(oxr_ocr_context * c) {
    delete c;
}
int oxr_ocr_set_prompt(oxr_ocr_context * c, const char * p) {
    if (!c) return 0;
    if (!p || std::strlen(p) > 65536 || std::strstr(p, "<image>") || std::strstr(p, "<video>")) {
        c->error = "OXR prompt must be <=65536 bytes and contain no image/video placeholders";
        c->last_ok = false;
        return 0;
    }
    c->prompt = p;
    return 1;
}
const char * oxr_ocr_get_prompt(const oxr_ocr_context * c) {
    return c ? c->prompt.c_str() : nullptr;
}
void oxr_ocr_set_max_tokens(oxr_ocr_context * c, int value) {
    if (!c) return;
    c->max_tokens = value == 0 ? 2048 : value;
    if (value < 0 || value > oxr_profile::context_length) {
        c->error = "OXR max_tokens must be 0..32768 (0 resets the 2048 default)";
        c->last_ok = false;
    }
}
int oxr_ocr_get_max_tokens(const oxr_ocr_context * c) {
    return c ? c->max_tokens : 0;
}
int oxr_ocr_set_repetition_penalty(oxr_ocr_context * c, float penalty) {
    if (!c) return 0;
    if (!std::isfinite(penalty) || penalty <= 0.0f) {
        c->error = "OXR repetition penalty must be positive and finite";
        c->last_ok = false;
        return 0;
    }
    c->repetition_penalty = penalty;
    return 1;
}
float oxr_ocr_get_repetition_penalty(const oxr_ocr_context * c) {
    return c ? c->repetition_penalty : 1.0f;
}
int oxr_ocr_set_hf_generation(oxr_ocr_context * c, int value) {
    if (!c || (value != 0 && value != 1)) return 0;
    c->hf_generation = value != 0;
    return 1;
}
int oxr_ocr_get_hf_generation(const oxr_ocr_context * c) {
    return c && c->hf_generation ? 1 : 0;
}
int oxr_ocr_set_pixel_limits(oxr_ocr_context * c, int lo, int hi) {
    if (!c) return 0;
    if (lo <= 0 || hi < lo || hi > oxr_profile::hf_max_pixels) {
        c->error = "OXR pixel limits require 0 < min <= max <= 2408448";
        c->last_ok = false;
        return 0;
    }
    c->min_pixels = lo;
    c->max_pixels = hi;
    return 1;
}
int oxr_ocr_get_min_pixels(const oxr_ocr_context * c) {
    return c ? c->min_pixels : 0;
}
int oxr_ocr_get_max_pixels(const oxr_ocr_context * c) {
    return c ? c->max_pixels : 0;
}
const char * oxr_ocr_recognize_raw(oxr_ocr_context * c, const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (out_len) *out_len = 0;
    if (!c) return nullptr;
    clear_result(*c);
    try {
        if (c->max_tokens < 1 || c->max_tokens > oxr_profile::context_length)
            throw std::invalid_argument("OXR max_tokens must be 1..32768 after default reset");
        std::vector<float> visual;
        if (px) {
            if (w <= 0 || h <= 0 || w > 65536 || h > 65536 || int64_t(w) * h > 67108864 ||
                (ch != 1 && ch != 3 && ch != 4))
                throw std::invalid_argument("OXR invalid raw image dimensions/channels");
            auto prepared = oxr_profile::preprocess(px, w, h, ch, c->min_pixels, c->max_pixels);
            std::copy(std::begin(prepared.grid), std::end(prepared.grid), std::begin(c->grid));
            prepare_request(*c, prepared.grid[1] * prepared.grid[2] / 4);
            visual = encode_vision(*c, prepared);
        } else if (w != 0 || h != 0 || ch != 0)
            throw std::invalid_argument("OXR text-only request requires zero image dimensions");
        else
            prepare_request(*c, 0);
        generate(*c, visual);
        c->last_ok = true;
        if (out_len) *out_len = int(c->text.size());
        return c->text.c_str();
    } catch (const std::exception & e) {
        c->error = e.what();
        return nullptr;
    }
}
const char * oxr_ocr_recognize(oxr_ocr_context * c, const float * px, int w, int h, int * out_len) {
    if (out_len) *out_len = 0;
    if (!c) return nullptr;
    if (!px || w <= 0 || h <= 0 || w > 65536 || h > 65536 || int64_t(w) * h > 67108864) {
        clear_result(*c);
        c->error = "OXR invalid grayscale image";
        return nullptr;
    }
    try {
        std::vector<uint8_t> bytes(size_t(w) * h);
        for (size_t i = 0; i < bytes.size(); ++i) {
            if (!std::isfinite(px[i])) throw std::invalid_argument("OXR grayscale pixels must be finite");
            bytes[i] = uint8_t(std::clamp(std::lround(px[i] * 255.0f), 0L, 255L));
        }
        return oxr_ocr_recognize_raw(c, bytes.data(), w, h, 1, out_len);
    } catch (const std::exception & e) {
        clear_result(*c);
        c->error = e.what();
        return nullptr;
    }
}
const float * oxr_ocr_confidences(const oxr_ocr_context * c, int * count) {
    if (count) *count = c ? int(c->confidences.size()) : 0;
    return c && !c->confidences.empty() ? c->confidences.data() : nullptr;
}
float oxr_ocr_mean_confidence(const oxr_ocr_context * c) {
    return c && !c->confidences.empty()
               ? std::accumulate(c->confidences.begin(), c->confidences.end(), 0.0f) / float(c->confidences.size())
               : 0.0f;
}
int oxr_ocr_last_ok(const oxr_ocr_context * c) {
    return c && c->last_ok ? 1 : 0;
}
const char * oxr_ocr_last_error(const oxr_ocr_context * c) {
    return c ? c->error.c_str() : "OXR null context";
}
const char * oxr_ocr_raw_text(const oxr_ocr_context * c) {
    return c ? c->raw_text.c_str() : nullptr;
}
const int32_t * oxr_ocr_prompt_ids(const oxr_ocr_context * c, int * count) {
    if (count) *count = c ? int(c->prompt_ids.size()) : 0;
    return c && !c->prompt_ids.empty() ? c->prompt_ids.data() : nullptr;
}
const int32_t * oxr_ocr_completion_ids(const oxr_ocr_context * c, int * count) {
    if (count) *count = c ? int(c->completion_ids.size()) : 0;
    return c && !c->completion_ids.empty() ? c->completion_ids.data() : nullptr;
}
const int32_t * oxr_ocr_grid_thw(const oxr_ocr_context * c) {
    return c ? c->grid : nullptr;
}
int oxr_ocr_stopped_eos(const oxr_ocr_context * c) {
    return c && c->eos ? 1 : 0;
}
int oxr_ocr_capped(const oxr_ocr_context * c) {
    return c && c->capped ? 1 : 0;
}
int oxr_ocr_effective_max_tokens(const oxr_ocr_context * c) {
    return c ? c->effective_max_tokens : 0;
}
void oxr_ocr_set_stage_observer(oxr_ocr_context * c, oxr_ocr_stage_observer observer, void * user) {
    if (c) {
        c->observer = observer;
        c->observer_user = user;
    }
}
} // extern "C"
