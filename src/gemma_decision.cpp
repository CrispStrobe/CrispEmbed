// EmbeddingGemma 2 text encoder and schema-conditioned typed-decision head.
// Blueprint: pinned model's modeling_gemmadecision.py and Transformers 5.19.0
// modeling_embedding_gemma2.py. CPU reference path; opt in explicitly.
#include "gemma_decision.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
#include "tokenizer.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

struct gemma_decision_context {
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    BPETokenizer tokenizer;
    std::vector<std::string> vocab, special, layer_types;
    std::vector<int> special_ids, head_dims, kv_heads;
    std::map<std::string, std::vector<float>> head;
    std::string prompt, error;
    int H = 512, D = 768, L = 24, heads = 4, threads = 1, max_length = 8192, window = 512, width = 256;
    int bos = 2, eos = 1;
    float eps = 1e-6f, local_rope = 10000, global_rope = 1000000;
    ~gemma_decision_context() {
        if (sched) ggml_backend_sched_free(sched);
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
    ggml_tensor * tensor(const std::string & name) const {
        auto it = weights.tensors.find(name);
        if (it == weights.tensors.end()) throw std::runtime_error("missing tensor: " + name);
        return it->second;
    }
    const std::vector<float> & cpu(const std::string & name) {
        auto it = head.find(name);
        if (it == head.end()) it = head.emplace(name, core_cpu::to_f32(tensor(name))).first;
        return it->second;
    }
};

gemma_decision_context * gemma_decision_init(const char * path, int threads) {
    if (!path || !core_env::on("CRISPEMBED_GEMMA_DECISION")) return nullptr;
    auto c = std::make_unique<gemma_decision_context>();
    try {
        auto * g = core_gguf::open_metadata(path);
        if (!g) return nullptr;
        auto u = [&](const char * key, int def) {
            return (int)core_gguf::kv_u32(g, (std::string("gemma_decision.") + key).c_str(), def);
        };
        const std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        c->H = u("hidden_size", 0);
        c->D = u("embedding_dim", 0);
        c->L = u("n_layer", 0);
        c->heads = u("n_head", 0);
        c->max_length = u("max_length", 0);
        c->window = u("sliding_window", 0);
        c->width = u("head_width", 0);
        c->eps = core_gguf::kv_f32(g, "gemma_decision.rms_norm_eps", 1e-6f);
        c->local_rope = core_gguf::kv_f32(g, "gemma_decision.rope.sliding_attention", 10000);
        c->global_rope = core_gguf::kv_f32(g, "gemma_decision.rope.full_attention", 1000000);
        c->prompt = core_gguf::kv_str(g, "gemma_decision.prompt", "");
        c->vocab = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        auto merges = core_gguf::kv_str_array(g, "tokenizer.ggml.merges");
        c->special = core_gguf::kv_str_array(g, "gemma_decision.special_tokens");
        c->special_ids = core_gguf::kv_i32_array(g, "gemma_decision.special_ids");
        c->head_dims = core_gguf::kv_i32_array(g, "gemma_decision.layer_head_dim");
        c->kv_heads = core_gguf::kv_i32_array(g, "gemma_decision.layer_kv_heads");
        c->layer_types = core_gguf::kv_str_array(g, "gemma_decision.layer_types");
        c->bos = core_gguf::kv_u32(g, "tokenizer.ggml.bos_token_id", 2);
        c->eos = core_gguf::kv_u32(g, "tokenizer.ggml.eos_token_id", 1);
        core_gguf::free_metadata(g);
        if (arch != "gemma_decision" || c->H != 512 || c->D != 768 || c->L != 24 || c->heads != 4 || c->width != 256 ||
            c->max_length < 2 || c->max_length > 8192 || c->vocab.size() != 262144 || c->window != 512 ||
            c->head_dims.size() != (size_t)c->L || c->kv_heads.size() != (size_t)c->L ||
            c->layer_types.size() != (size_t)c->L || c->special.size() != c->special_ids.size())
            throw std::runtime_error("unsupported/incomplete GemmaDecision configuration");
        c->tokenizer.load(c->vocab, merges, -1, 0, -1, -1, true, c->max_length - 2);
        c->tokenizer.set_stable_spm_ties(true);
        c->threads = std::max(1, threads);
        c->backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(c->backend, c->threads);
        if (!core_gguf::load_weights(path, c->backend, "gemma_decision", c->weights, true)) return nullptr;
        auto shape = [&](const std::string & name, int cols, int rows = 1) {
            auto * t = c->tensor(name);
            if (t->ne[0] != cols || t->ne[1] != rows || ggml_nelements(t) != (int64_t)cols * rows)
                throw std::runtime_error("invalid tensor shape: " + name);
        };
        shape("encoder.language_model.embed_tokens.weight", c->H, (int)c->vocab.size());
        shape("encoder.language_model.ple.per_layer_model_projection.weight", c->H, c->H * c->L);
        shape("encoder.language_model.ple.per_layer_projection_norm.weight", c->H);
        shape("encoder.language_model.norm.weight", c->H);
        shape("encoder.language_model.embedding_projection.weight", c->H, c->D);
        for (int i = 0; i < c->L; ++i) {
            const std::string p = "encoder.language_model.layers." + std::to_string(i) + ".";
            const int hd = c->head_dims[i], kv = c->kv_heads[i];
            if ((hd != 256 && hd != 512) || kv <= 0 || c->heads % kv ||
                (c->layer_types[i] != "full_attention" && c->layer_types[i] != "sliding_attention"))
                throw std::runtime_error("invalid layer configuration");
            for (const auto & n :
                 { "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight",
                   "post_feedforward_layernorm.weight", "ple_block.post_per_layer_input_norm.weight" })
                shape(p + n, c->H);
            shape(p + "layer_scalar", 1);
            shape(p + "self_attn.q_norm.weight", hd);
            shape(p + "self_attn.k_norm.weight", hd);
            shape(p + "self_attn.q_proj.weight", c->H, hd * c->heads);
            shape(p + "self_attn.k_proj.weight", c->H, hd * kv);
            shape(p + "self_attn.v_proj.weight", c->H, hd * kv);
            shape(p + "self_attn.o_proj.weight", hd * c->heads, c->H);
            shape(p + "mlp.gate_proj.weight", c->H, 2048);
            shape(p + "mlp.up_proj.weight", c->H, 2048);
            shape(p + "mlp.down_proj.weight", 2048, c->H);
            shape(p + "ple_block.per_layer_input_gate.weight", c->H, c->H);
            shape(p + "ple_block.per_layer_projection.weight", c->H, c->H);
        }
        for (const auto & n : { "state_proj", "question_proj", "option_proj" }) {
            shape(std::string("head.") + n + ".weight", c->D, c->width);
            shape(std::string("head.") + n + ".bias", c->width);
        }
        shape("head.field.0.weight", 3 * c->width, c->width);
        shape("head.field.0.bias", c->width);
        shape("head.field.3.weight", c->width, c->width);
        shape("head.field.3.bias", c->width);
        shape("head.residual.0.weight", 4 * c->width, c->width);
        shape("head.residual.0.bias", c->width);
        shape("head.residual.3.weight", c->width, 1);
        shape("head.residual.3.bias", 1);
        for (const auto & n : { "field_norm", "option_norm" }) {
            shape(std::string("head.") + n + ".weight", c->width);
            shape(std::string("head.") + n + ".bias", c->width);
        }
        shape("head.type_emb.weight", c->width, 3);
        shape("temperatures", 3);
        for (const auto & n : { "head.gate", "head.log_joint_scale", "head.log_prior_scale" }) shape(n, 1);
        for (auto & item : c->weights.tensors) {
            if (item.first.rfind("head.", 0) == 0 || item.first == "temperatures") {
                const auto & v = c->cpu(item.first);
                for (float f : v)
                    if (!std::isfinite(f)) throw std::runtime_error("nonfinite head weight");
            }
        }
        for (float f : c->cpu("temperatures"))
            if (f <= 0) throw std::runtime_error("invalid calibration temperature");
        c->sched = ggml_backend_sched_new(&c->backend, nullptr, 1, 8192, false, false);
        if (!c->sched) throw std::runtime_error("scheduler initialization failed");
        return c.release();
    } catch (const std::exception & e) {
        fprintf(stderr, "gemma_decision: %s\n", e.what());
        return nullptr;
    }
}
void gemma_decision_free(gemma_decision_context * c) {
    delete c;
}
const char * gemma_decision_error(const gemma_decision_context * c) {
    return c ? c->error.c_str() : "invalid context";
}
const char * gemma_decision_token(const gemma_decision_context * c, int id) {
    return c && id >= 0 && id < (int)c->vocab.size() ? c->vocab[id].c_str() : "";
}
void gemma_decision_hparams(const gemma_decision_context * c, crispembed_hparams * out) {
    *out = {};
    out->n_embd = c->D;
    out->n_output = c->D;
    out->n_layer = c->L;
    out->n_head = c->heads;
    out->n_vocab = (int)c->vocab.size();
    out->n_max_tokens = c->max_length;
    out->n_intermediate = 2048;
}

static std::vector<int32_t> tokenize(gemma_decision_context * c, const std::string & text) {
    std::vector<int32_t> result{ c->bos };
    const std::string input = c->prompt + text;
    size_t at = 0;
    while (at < input.size() && result.size() < (size_t)c->max_length - 1) {
        size_t next = std::string::npos;
        int special = -1;
        for (size_t i = 0; i < c->special.size(); ++i) {
            size_t found = input.find(c->special[i], at);
            if (found < next || (found == next && special >= 0 && c->special[i].size() > c->special[special].size())) {
                next = found;
                special = (int)i;
            }
        }
        auto span = c->tokenizer.encode(input.substr(at, next - at));
        result.insert(result.end(), span.ids.begin(), span.ids.end());
        if (special < 0) break;
        result.push_back(c->special_ids[special]);
        at = next + c->special[special].size();
    }
    if (result.size() >= (size_t)c->max_length) result.resize(c->max_length - 1);
    result.push_back(c->eos);
    return result;
}

bool gemma_decision_encode(gemma_decision_context * c, const std::string & text, std::vector<float> & embedding,
                           std::vector<int32_t> * ids, std::vector<float> * raw_tokens) {
    embedding.clear();
    if (ids) ids->clear();
    if (raw_tokens) raw_tokens->clear();
    if (!c) return false;
    c->error.clear();
    ggml_context * g = nullptr;
    try {
        auto token_ids = tokenize(c, text);
        const int T = (int)token_ids.size(), H = c->H;
        const size_t bytes = 8192 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8192, false);
        g = ggml_init({ bytes, nullptr, true });
        auto * graph = ggml_new_graph_custom(g, 8192, false);
        auto * tok = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_input(tok);
        auto * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
        ggml_set_input(pos);
        auto * mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, T, T);
        ggml_set_input(mask);
        const bool dump = core_env::on("CRISPEMBED_GEMMA_DECISION_DUMP");
        std::vector<ggml_tensor *> probes;
        auto snapshot = [&](ggml_tensor * t, const std::string & name) {
            if (dump) {
                ggml_set_name(t, name.c_str());
                ggml_set_output(t);
                probes.push_back(t);
            }
            return t;
        };
        auto rms = [&](ggml_tensor * x, const std::string & w) {
            return ggml_mul(g, ggml_rms_norm(g, x, c->eps), c->tensor(w));
        };
        const std::string root = "encoder.language_model.";
        auto * cur =
            snapshot(ggml_scale(g, ggml_get_rows(g, c->tensor(root + "embed_tokens.weight"), tok), std::sqrt((float)H)),
                     "post_embed");
        auto * ple = ggml_scale(g, ggml_mul_mat(g, c->tensor(root + "ple.per_layer_model_projection.weight"), cur),
                                1 / std::sqrt((float)H));
        ple = rms(ggml_reshape_3d(g, ple, H, c->L, T), root + "ple.per_layer_projection_norm.weight");
        snapshot(ple, "ple");
        for (int i = 0; i < c->L; ++i) {
            const std::string p = root + "layers." + std::to_string(i) + ".";
            const int hd = c->head_dims[i], kv = c->kv_heads[i];
            auto * residual = cur;
            auto * x = rms(cur, p + "input_layernorm.weight");
            auto * Q =
                ggml_reshape_3d(g, ggml_mul_mat(g, c->tensor(p + "self_attn.q_proj.weight"), x), hd, c->heads, T);
            auto * K = ggml_reshape_3d(g, ggml_mul_mat(g, c->tensor(p + "self_attn.k_proj.weight"), x), hd, kv, T);
            auto * V = ggml_reshape_3d(g, ggml_mul_mat(g, c->tensor(p + "self_attn.v_proj.weight"), x), hd, kv, T);
            Q = rms(Q, p + "self_attn.q_norm.weight");
            K = rms(K, p + "self_attn.k_norm.weight");
            V = ggml_rms_norm(g, V, c->eps);
            const bool local = c->layer_types[i] == "sliding_attention";
            const float theta = local ? c->local_rope : c->global_rope;
            Q = ggml_rope_ext(g, Q, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 8192, theta, 1, 0, 1, 32, 1);
            K = ggml_rope_ext(g, K, pos, nullptr, hd, GGML_ROPE_TYPE_NEOX, 8192, theta, 1, 0, 1, 32, 1);
            auto repeat = [&](ggml_tensor * v) {
                v = ggml_reshape_4d(g, v, hd, 1, kv, T);
                auto * shape = ggml_new_tensor_4d(g, GGML_TYPE_F32, hd, c->heads / kv, kv, T);
                return ggml_reshape_3d(g, ggml_repeat(g, v, shape), hd, c->heads, T);
            };
            K = repeat(K);
            V = repeat(V);
            Q = ggml_permute(g, Q, 0, 2, 1, 3);
            K = ggml_permute(g, K, 0, 2, 1, 3);
            auto * scores = ggml_mul_mat(g, K, Q);
            // Reference scales attention by exactly 1, and normalizes V too.
            scores = ggml_soft_max_ext(g, scores, local ? mask : nullptr, 1, 0);
            V = ggml_cont(g, ggml_transpose(g, ggml_permute(g, V, 0, 2, 1, 3)));
            auto * attn = ggml_mul_mat(g, V, scores);
            attn = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3)), hd * c->heads, T);
            attn = rms(ggml_mul_mat(g, c->tensor(p + "self_attn.o_proj.weight"), attn),
                       p + "post_attention_layernorm.weight");
            cur = ggml_add(g, residual, attn);
            residual = cur;
            x = rms(cur, p + "pre_feedforward_layernorm.weight");
            auto * gate = ggml_gelu(g, ggml_mul_mat(g, c->tensor(p + "mlp.gate_proj.weight"), x));
            auto * up = ggml_mul_mat(g, c->tensor(p + "mlp.up_proj.weight"), x);
            x = rms(ggml_mul_mat(g, c->tensor(p + "mlp.down_proj.weight"), ggml_mul(g, gate, up)),
                    p + "post_feedforward_layernorm.weight");
            cur = ggml_add(g, residual, x);
            residual = cur;
            x = ggml_gelu(g, ggml_mul_mat(g, c->tensor(p + "ple_block.per_layer_input_gate.weight"), cur));
            auto * slice = ggml_view_2d(g, ple, H, T, ple->nb[2], (size_t)i * ple->nb[1]);
            x = ggml_mul(g, x, slice);
            x = rms(ggml_mul_mat(g, c->tensor(p + "ple_block.per_layer_projection.weight"), x),
                    p + "ple_block.post_per_layer_input_norm.weight");
            cur = snapshot(ggml_mul(g, ggml_add(g, residual, x), c->tensor(p + "layer_scalar")),
                           "layer_" + std::to_string(i));
        }
        cur = snapshot(rms(cur, root + "norm.weight"), "final_norm");
        cur = snapshot(ggml_mul_mat(g, c->tensor(root + "embedding_projection.weight"), cur), "tokens");
        ggml_set_output(cur);
        ggml_build_forward_expand(graph, cur);
        ggml_backend_sched_reset(c->sched);
        if (!ggml_backend_sched_alloc_graph(c->sched, graph)) throw std::runtime_error("graph allocation failed");
        std::vector<int32_t> positions(T);
        for (int i = 0; i < T; ++i) positions[i] = i;
        std::vector<float> m((size_t)T * T);
        for (int q = 0; q < T; ++q)
            for (int k = 0; k < T; ++k) m[(size_t)q * T + k] = std::abs(q - k) <= c->window ? 0 : -INFINITY;
        auto embedding_ids = token_ids;
        for (auto & id : embedding_ids)
            if (id == 258880 || id == 258881 || id == 258884) id = 0;
        ggml_backend_tensor_set(tok, embedding_ids.data(), 0, T * sizeof(int32_t));
        ggml_backend_tensor_set(pos, positions.data(), 0, T * sizeof(int32_t));
        if (mask->buffer) ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(c->sched, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph compute failed");
        std::vector<float> hidden((size_t)c->D * T);
        ggml_backend_tensor_get(cur, hidden.data(), 0, hidden.size() * sizeof(float));
        if (dump) {
            const char * path = std::getenv("CRISPEMBED_GEMMA_DECISION_DUMP_PATH");
            if (path && *path) {
                auto * meta = gguf_init_empty();
                std::vector<std::vector<float>> storage;
                storage.reserve(probes.size());
                auto * tensors = ggml_init({ probes.size() * ggml_tensor_overhead() + 1024, nullptr, true });
                for (auto * probe : probes) {
                    storage.emplace_back((size_t)ggml_nelements(probe));
                    auto & v = storage.back();
                    ggml_backend_tensor_get(probe, v.data(), 0, v.size() * sizeof(float));
                    auto * t = ggml_new_tensor(tensors, GGML_TYPE_F32, ggml_n_dims(probe), probe->ne);
                    ggml_set_name(t, probe->name);
                    t->data = v.data();
                    gguf_add_tensor(meta, t);
                }
                if (!gguf_write_to_file(meta, path, false)) throw std::runtime_error("dump write failed");
                gguf_free(meta);
                ggml_free(tensors);
            }
        }
        embedding.assign(c->D, 0);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < c->D; ++d) embedding[d] += hidden[(size_t)t * c->D + d] / T;
        double norm = 0;
        for (float f : embedding) norm += double(f) * f;
        norm = std::max(std::sqrt(norm), 1e-12);
        for (float & f : embedding) {
            f = (float)(f / norm);
            if (!std::isfinite(f)) throw std::runtime_error("nonfinite embedding");
        }
        if (ids) *ids = std::move(token_ids);
        if (raw_tokens) *raw_tokens = std::move(hidden);
        ggml_backend_sched_reset(c->sched);
        ggml_free(g);
        return true;
    } catch (const std::exception & e) {
        c->error = e.what();
        embedding.clear();
        ggml_backend_sched_reset(c->sched);
        if (g) ggml_free(g);
        return false;
    }
}

bool gemma_decision_scores(gemma_decision_context * c, const float * state, const float * question,
                           const float * options, int count, int kind, float * logits, float * probabilities) {
    if (!c || !state || !question || !options || !logits || !probabilities || count < 1 || count > 4096 || kind < 0 ||
        kind > 2)
        return false;
    try {
        const int W = c->width, D = c->D;
        auto linear = [&](const std::string & n, const std::vector<float> & x) {
            const auto & w = c->cpu(n + ".weight");
            const auto & b = c->cpu(n + ".bias");
            std::vector<float> y(b);
            for (size_t r = 0; r < y.size(); ++r)
                for (size_t j = 0; j < x.size(); ++j) y[r] += w[r * x.size() + j] * x[j];
            return y;
        };
        auto gelu = [](std::vector<float> x) {
            for (float & v : x) v = 0.5f * v * (1 + std::erf(v / std::sqrt(2.0f)));
            return x;
        };
        auto ln = [&](const std::string & n, std::vector<float> x) {
            double mean = 0, var = 0;
            for (float f : x) mean += f;
            mean /= x.size();
            for (float f : x) var += (f - mean) * (f - mean);
            var /= x.size();
            const auto & w = c->cpu(n + ".weight");
            const auto & b = c->cpu(n + ".bias");
            for (size_t j = 0; j < x.size(); ++j) x[j] = (float)((x[j] - mean) / std::sqrt(var + 1e-5)) * w[j] + b[j];
            return x;
        };
        auto normalize = [](std::vector<float> x, float eps) {
            double n = 0;
            for (float f : x) n += double(f) * f;
            n = std::max(std::sqrt(n), (double)eps);
            for (float & f : x) f = (float)(f / n);
            return x;
        };
        auto S = linear("head.state_proj", std::vector<float>(state, state + D));
        auto Q = linear("head.question_proj", std::vector<float>(question, question + D));
        std::vector<float> x = S;
        x.insert(x.end(), Q.begin(), Q.end());
        for (int j = 0; j < W; ++j) x.push_back(S[j] * Q[j]);
        auto field = linear("head.field.3", gelu(linear("head.field.0", x)));
        const auto & types = c->cpu("head.type_emb.weight");
        for (int j = 0; j < W; ++j) field[j] += types[kind * W + j];
        field = ln("head.field_norm", field);
        auto nf = normalize(field, 1e-8f);
        std::vector<float> anchor(D);
        for (int j = 0; j < D; ++j) anchor[j] = state[j] + question[j];
        anchor = normalize(anchor, 1e-12f);
        const float prior_scale = std::exp(std::min(c->cpu("head.log_prior_scale")[0], std::log(100.0f)));
        const float joint_scale = std::exp(std::min(c->cpu("head.log_joint_scale")[0], std::log(100.0f)));
        const float gate = 1 / (1 + std::exp(-c->cpu("head.gate")[0]));
        const float temperature = c->cpu("temperatures")[kind];
        for (int i = 0; i < count; ++i) {
            std::vector<float> option(options + (size_t)i * D, options + (size_t)(i + 1) * D);
            auto O = ln("head.option_norm", linear("head.option_proj", option));
            auto no = normalize(O, 1e-8f);
            std::vector<float> joint = field;
            joint.insert(joint.end(), O.begin(), O.end());
            for (int j = 0; j < W; ++j) joint.push_back(field[j] * O[j]);
            for (int j = 0; j < W; ++j) joint.push_back(std::fabs(field[j] - O[j]));
            float residual = linear("head.residual.3", gelu(linear("head.residual.0", joint)))[0];
            float cos = 0, prior = 0;
            for (int j = 0; j < W; ++j) cos += nf[j] * no[j];
            option = normalize(option, 1e-12f);
            for (int j = 0; j < D; ++j) prior += option[j] * anchor[j];
            logits[i] = prior_scale * prior + gate * (joint_scale * cos + residual);
            if (!std::isfinite(logits[i])) throw std::runtime_error("nonfinite decision logit");
        }
        float maximum = *std::max_element(logits, logits + count) / temperature;
        double denom = 0;
        for (int i = 0; i < count; ++i) {
            probabilities[i] = std::exp(logits[i] / temperature - maximum);
            denom += probabilities[i];
        }
        for (int i = 0; i < count; ++i) probabilities[i] = (float)(probabilities[i] / denom);
        return true;
    } catch (const std::exception & e) {
        c->error = e.what();
        return false;
    }
}
