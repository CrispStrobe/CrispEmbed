// Independent GGML decomposition of Apache-2.0 JointSchemaHead at
// mayflowergmbh/Josev-2B@6f0a31c8ed683aba79097eb3f48df2c9ab560afd.
// Publisher source/license identity: docs/josev2b-source-contract.json.
// No publisher Python code or backbone implementation is copied.
#include "josev2b_head.h"
#include "core/env_gate.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "ggml-alloc.h"
#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
#include "../tests/josev2b_head_observer.h"
#endif
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
#define JOSEV_MARK(label, tensor, ...) mark(label, tensor, __VA_ARGS__)
#else
#define JOSEV_MARK(...) ((void)0)
#endif
namespace josev2b_head {
struct context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    std::string name;
    ~context() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};
namespace {
constexpr const char * arch = "josev2b_head_reference";
constexpr size_t graph_capacity = 8192;
using dimensions = std::array<int64_t, 4>;
using schema = std::map<std::string, dimensions>;
bool fail(std::string & error, const std::string & message) {
    error = message;
    return false;
}
bool gate(std::string & error) {
    return core_env::on("CRISPEMBED_JOSEV_HEAD") || fail(error, "Josev head requires CRISPEMBED_JOSEV_HEAD=1");
}
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
std::string key(const char * suffix) {
    return std::string(arch) + "." + suffix;
}
schema expected_schema() {
    schema rows;
    auto add = [&](const std::string & name, int64_t x, int64_t y = 1) { rows["head." + name] = { x, y, 1, 1 }; };
    auto norm = [&](const std::string & name, int n) {
        add(name + ".weight", n);
        add(name + ".bias", n);
    };
    auto linear = [&](const std::string & name, int in, int out, bool bias = true) {
        add(name + ".weight", in, out);
        if (bias) add(name + ".bias", out);
    };
    auto mha = [&](const std::string & name) {
        add(name + ".in_proj_weight", 512, 1536);
        add(name + ".in_proj_bias", 1536);
        linear(name + ".out_proj", 512, 512);
    };
    norm("hidden_norm", 2048);
    for (const auto * name : { "memory_projection", "question_projection", "option_question_projection",
                               "global_projection", "option_context_projection", "option_lexical_projection" })
        linear(name, 2048, 512, false);
    add("type_embedding.weight", 512, 3);
    for (int i = 0; i < 2; ++i) {
        const auto p = "evidence_layers." + std::to_string(i) + ".";
        norm(p + "query_norm", 512);
        norm(p + "memory_norm", 512);
        mha(p + "attention");
        norm(p + "feedforward_norm", 512);
        linear(p + "feedforward.0", 512, 2048);
        linear(p + "feedforward.3", 2048, 512);
    }
    norm("option_summary_norm", 512);
    for (int i = 0; i < 4; ++i) {
        const auto p = "layers." + std::to_string(i) + ".";
        mha(p + "self_attn");
        mha(p + "multihead_attn");
        linear(p + "linear1", 512, 2048);
        linear(p + "linear2", 2048, 512);
        for (int j = 1; j <= 3; ++j) norm(p + "norm" + std::to_string(j), 512);
    }
    norm("field_norm", 512);
    norm("option_norm", 512);
    linear("residual_scorer.0", 2048, 512);
    linear("residual_scorer.3", 512, 1);
    for (const auto * name : { "prior_logit_scale", "joint_logit_scale", "residual_gate" }) add(name, 1);
    return rows;
}
bool metadata_valid(gguf_context * meta, const char * receipt, const char * worker, std::string & error) {
    if (!receipt || !worker || std::strcmp(receipt, original_receipt_sha256) ||
        std::strcmp(worker, original_worker_source))
        return fail(error, "Explicit reviewed original receipt/worker required");
    if (core_gguf::kv_str(meta, "general.architecture", "") != arch ||
        core_gguf::kv_str(meta, "general.license", "") != "apache-2.0" ||
        core_gguf::kv_str(meta, key("source_revision").c_str(), "") != "6f0a31c8ed683aba79097eb3f48df2c9ab560afd" ||
        core_gguf::kv_str(meta, key("checkpoint_sha256").c_str(), "") !=
            "8816d51356876ef9b055e51291a00dfc4a8c741189ca088b59604d039d7852c5" ||
        core_gguf::kv_str(meta, key("original_receipt_sha256").c_str(), "") != receipt ||
        core_gguf::kv_str(meta, key("original_worker_source").c_str(), "") != worker ||
        core_gguf::kv_str(meta, key("calibration_profile").c_str(), "") != "official-pytorch-no-extra-temperature")
        return fail(error, "Josev head original/source/calibration metadata differs");
    for (const auto & item : std::map<std::string, uint32_t>{ { "hidden_size", 2048 },
                                                              { "width", 512 },
                                                              { "routing_layers", 2 },
                                                              { "layers", 4 },
                                                              { "heads", 8 },
                                                              { "feedforward", 2048 } })
        if (core_gguf::kv_u32(meta, key(item.first.c_str()).c_str(), 0) != item.second)
            return fail(error, "Josev head configuration differs");
    const auto rows = expected_schema();
    int64_t elements = 0;
    if (rows.size() != 122 || gguf_get_n_tensors(meta) != 122)
        return fail(error, "Josev head requires exactly122 tensors");
    for (const auto & item : rows) {
        const int64_t id = gguf_find_tensor(meta, item.first.c_str());
        if (id < 0 || gguf_get_tensor_type(meta, id) != GGML_TYPE_F32)
            return fail(error, "Missing/non-F32 head tensor: " + item.first);
        const auto * ne = gguf_get_tensor_ne(meta, id);
        int64_t count = 1;
        for (int axis = 0; axis < 4; ++axis) {
            if (ne[axis] != item.second[axis]) return fail(error, "Head tensor shape differs: " + item.first);
            count *= ne[axis];
        }
        elements += count;
    }
    return elements == 30472708 || fail(error, "Head parameter inventory differs");
}
struct graph_builder {
    context & model;
    const input & values;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor *hidden = nullptr, *types = nullptr;
    std::vector<ggml_tensor *> lexical, logits, probabilities;
#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
    struct event {
        std::string label;
        size_t occurrence;
        std::vector<int64_t> shape;
        ggml_tensor * tensor;
    };
    bool observe = false;
    std::map<std::string, size_t> occurrences;
    std::vector<event> stages;
#endif
    graph_builder(context & model, const input & values) : model(model), values(values) {
        ctx = ggml_init({ graph_capacity * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false),
                          nullptr, true });
        require(ctx, "Head graph metadata allocation failed");
    }
    ~graph_builder() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }
    ggml_tensor * w(const std::string & name) { return model.weights.tensors.at("head." + name); }
#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
    void mark(const std::string & label, ggml_tensor * tensor, std::vector<int64_t> shape) {
        if (observe) {
            int64_t elements = 1;
            for (auto dimension : shape) {
                require(dimension > 0, "Invalid observed logical shape");
                elements *= dimension;
            }
            require(elements == ggml_nelements(tensor) && ggml_is_contiguous(tensor),
                    "Observed shape/layout differs from actual tensor");
            // Connected VIEW, never a disconnected ggml_view_tensor metadata alias.
            auto * view = ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                                       tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
            for (auto * owner = view; owner; owner = owner->view_src) ggml_set_output(owner);
            stages.push_back({ label, occurrences[label]++, std::move(shape), view });
        }
    }
#endif
    ggml_tensor * matmul(ggml_tensor * a, ggml_tensor * b) {
        auto * out = ggml_mul_mat(ctx, a, b);
        ggml_mul_mat_set_prec(out, GGML_PREC_F32);
        return out;
    }
    ggml_tensor * linear(const std::string & name, ggml_tensor * x, bool bias = true) {
        auto * out = matmul(w(name + ".weight"), x);
        return bias ? ggml_add(ctx, out, w(name + ".bias")) : out;
    }
    ggml_tensor * norm(const std::string & name, ggml_tensor * x) {
        return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w(name + ".weight")), w(name + ".bias"));
    }
    ggml_tensor * columns(ggml_tensor * x, int start, int count) {
        return ggml_view_2d(ctx, x, x->ne[0], count, x->nb[1], size_t(start) * x->nb[1]);
    }
    ggml_tensor * reduce_time(ggml_tensor * x, bool mean) {
        auto * transposed = ggml_cont(ctx, ggml_transpose(ctx, x));
        auto * reduced = mean ? ggml_mean(ctx, transposed) : ggml_sum_rows(ctx, transposed);
        return ggml_reshape_2d(ctx, reduced, x->ne[0], 1);
    }
    ggml_tensor * span_mean(ggml_tensor * x, span range) {
        return reduce_time(columns(x, range.start, range.end - range.start), true);
    }
    ggml_tensor * concatenate(const std::vector<ggml_tensor *> & parts, int dim) {
        require(!parts.empty(), "Empty head concatenation");
        auto * out = parts[0];
        for (size_t i = 1; i < parts.size(); ++i) out = ggml_concat(ctx, out, parts[i], dim);
        return out;
    }
    ggml_tensor * mha(const std::string & name, ggml_tensor * query, ggml_tensor * key_value) {
        auto project = [&](ggml_tensor * x, int index) {
            auto * weight = w(name + ".in_proj_weight");
            auto * bias = w(name + ".in_proj_bias");
            auto * slice = ggml_view_2d(ctx, weight, 512, 512, weight->nb[1], size_t(index) * 512 * weight->nb[1]);
            auto * b = ggml_view_1d(ctx, bias, 512, size_t(index) * 512 * sizeof(float));
            return ggml_add(ctx, matmul(slice, x), b);
        };
        auto heads = [&](ggml_tensor * x) {
            return ggml_permute(ctx, ggml_reshape_3d(ctx, x, 64, 8, x->ne[1]), 0, 2, 1, 3);
        };
        auto * q = heads(project(query, 0));
        auto * k = heads(project(key_value, 1));
        auto * v = heads(project(key_value, 2));
        // No mask, dropout, causal flag, RoPE or extra temperature in this head.
        auto * scores = ggml_soft_max(ctx, ggml_scale(ctx, matmul(k, q), 0.125f));
        auto * weighted = matmul(ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
        auto * merged =
            ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, weighted, 0, 2, 1, 3)), 512, query->ne[1]);
        return linear(name + ".out_proj", merged);
    }
    ggml_tensor * evidence(const std::string & name, ggml_tensor * query, ggml_tensor * memory) {
        auto * normalized_memory = norm(name + ".memory_norm", memory);
        auto * x = ggml_add(ctx, query, mha(name + ".attention", norm(name + ".query_norm", query), normalized_memory));
        auto * ff = linear(name + ".feedforward.3",
                           ggml_gelu_erf(ctx, linear(name + ".feedforward.0", norm(name + ".feedforward_norm", x))));
        return ggml_add(ctx, x, ff);
    }
    ggml_tensor * decoder(const std::string & name, ggml_tensor * fields, ggml_tensor * memory) {
        auto * n1 = norm(name + ".norm1", fields);
        auto * x = ggml_add(ctx, fields, mha(name + ".self_attn", n1, n1));
        auto * y = ggml_add(ctx, x, mha(name + ".multihead_attn", norm(name + ".norm2", x), memory));
        auto * ff = linear(name + ".linear2", ggml_gelu_erf(ctx, linear(name + ".linear1", norm(name + ".norm3", y))));
        return ggml_add(ctx, y, ff);
    }
    ggml_tensor * scale(const char * name) {
        // CLAMP aliases/mutates: duplicate the parameter; never modify loaded weights.
        return ggml_exp(ctx, ggml_clamp(ctx, ggml_dup(ctx, w(name)), -FLT_MAX, 4.605170185988091368f));
    }
    void build() {
        const int fields = int(values.fields.size());
        int total = 0;
        for (const auto & f : values.fields) total += int(f.options.size());
        hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, values.tokens);
        ggml_set_input(hidden);
        types = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, fields);
        ggml_set_input(types);
        JOSEV_MARK("head/hidden_norm/input/0", hidden, { 1, values.tokens, 2048 });
        auto * normalized = norm("hidden_norm", hidden);
        JOSEV_MARK("head/hidden_norm", normalized, { 1, values.tokens, 2048 });
        auto * memory = linear("memory_projection", normalized, false);
        JOSEV_MARK("head/memory_projection", memory, { values.tokens, 512 });
        auto * global = columns(normalized, values.tokens - 1, 1);
        std::vector<ggml_tensor *> questions, option_queries;
        for (size_t i = 0; i < values.fields.size(); ++i) {
            const auto & field = values.fields[i];
            const int count = int(field.options.size());
            auto * question = span_mean(normalized, field.question);
            questions.push_back(question);
            std::vector<ggml_tensor *> context_means;
            for (auto range : field.options) context_means.push_back(span_mean(normalized, range));
            auto * contextual = linear("option_context_projection", concatenate(context_means, 1), false);
            JOSEV_MARK("head/option_context_projection", contextual, { count, 512 });
            auto * lex = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, count);
            ggml_set_input(lex);
            lexical.push_back(lex);
            JOSEV_MARK("head/option_lexical_projection/input/0", lex, { count, 2048 });
            auto * projected_lexical = linear("option_lexical_projection", lex, false);
            JOSEV_MARK("head/option_lexical_projection", projected_lexical, { count, 512 });
            auto * projected_question = linear("option_question_projection", question, false);
            JOSEV_MARK("head/option_question_projection", projected_question, { 512 });
            option_queries.push_back(ggml_add(ctx, ggml_add(ctx, contextual, projected_lexical), projected_question));
        }
        auto * routed = concatenate(option_queries, 1);
        for (int i = 0; i < 2; ++i) {
            routed = evidence("evidence_layers." + std::to_string(i), routed, memory);
            JOSEV_MARK("head/evidence/" + std::to_string(i), routed, { 1, total, 512 });
        }
        auto * question_vectors = concatenate(questions, 1);
        JOSEV_MARK("head/question_projection/input/0", question_vectors, { fields, 2048 });
        auto * base = linear("question_projection", question_vectors, false);
        JOSEV_MARK("head/question_projection", base, { fields, 512 });
        std::vector<ggml_tensor *> splits, summaries;
        int offset = 0;
        for (int i = 0; i < fields; ++i) {
            const int count = int(values.fields[i].options.size());
            auto * split = columns(routed, offset, count);
            splits.push_back(split);
            offset += count;
            auto * routing =
                ggml_soft_max(ctx, ggml_scale(ctx, matmul(split, columns(base, i, 1)), 0.04419417382415922f));
            auto * weighted = ggml_mul(ctx, split, ggml_reshape_2d(ctx, routing, 1, count));
            summaries.push_back(reduce_time(weighted, false));
        }
        auto * summary = norm("option_summary_norm", concatenate(summaries, 1));
        JOSEV_MARK("head/option_summary_norm", summary, { fields, 512 });
        auto * global_projected = linear("global_projection", global, false);
        JOSEV_MARK("head/global_projection", global_projected, { 512 });
        auto * type_embedding = ggml_get_rows(ctx, w("type_embedding.weight"), types);
        JOSEV_MARK("head/type_embedding", type_embedding, { fields, 512 });
        auto * state = ggml_add(ctx, ggml_add(ctx, ggml_add(ctx, base, summary), global_projected), type_embedding);
        for (int i = 0; i < 4; ++i) {
            state = decoder("layers." + std::to_string(i), state, memory);
            JOSEV_MARK("head/field/" + std::to_string(i), state, { 1, fields, 512 });
        }
        state = norm("field_norm", state);
        JOSEV_MARK("head/field_norm", state, { fields, 512 });
        for (int i = 0; i < fields; ++i) {
            const int count = int(values.fields[i].options.size());
            auto * anchor = ggml_l2_norm(ctx, ggml_add(ctx, questions[i], global), 1e-12f);
            auto * lexical_anchor = ggml_l2_norm(ctx, lexical[i], 1e-12f);
            auto * prior = ggml_mul(ctx, matmul(lexical_anchor, anchor), scale("prior_logit_scale"));
            auto * options = norm("option_norm", splits[i]);
            JOSEV_MARK("head/option_norm", options, { count, 512 });
            auto * repeated = ggml_repeat(ctx, columns(state, i, 1), options);
            auto * cosine = ggml_sum_rows(
                ctx, ggml_mul(ctx, ggml_l2_norm(ctx, repeated, 1e-8f), ggml_l2_norm(ctx, options, 1e-8f)));
            auto * features = concatenate({ repeated, options, ggml_mul(ctx, repeated, options),
                                            ggml_abs(ctx, ggml_sub(ctx, repeated, options)) },
                                          0);
            JOSEV_MARK("head/residual_scorer/input/0", features, { count, 2048 });
            auto * residual = linear("residual_scorer.3", ggml_gelu_erf(ctx, linear("residual_scorer.0", features)));
            JOSEV_MARK("head/residual_scorer", residual, { count, 1 });
            auto * joint = ggml_add(ctx, ggml_mul(ctx, cosine, scale("joint_logit_scale")), residual);
            auto * final =
                ggml_add(ctx, ggml_reshape_1d(ctx, prior, count),
                         ggml_reshape_1d(ctx, ggml_mul(ctx, joint, ggml_sigmoid(ctx, w("residual_gate"))), count));
            logits.push_back(final);
            probabilities.push_back(ggml_soft_max(ctx, final));
            ggml_set_output(final);
            ggml_set_output(probabilities.back());
        }
        for (int i = 0; i < fields; ++i) {
            JOSEV_MARK("logits/" + values.fields[i].id, logits[i], { int64_t(values.fields[i].options.size()) });
            JOSEV_MARK("probabilities/" + values.fields[i].id, probabilities[i],
                       { int64_t(values.fields[i].options.size()) });
        }
        graph = ggml_new_graph_custom(ctx, graph_capacity, false);
#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
        for (const auto & stage : stages) ggml_build_forward_expand(graph, stage.tensor);
#endif
        for (size_t i = 0; i < logits.size(); ++i) {
            ggml_build_forward_expand(graph, logits[i]);
            ggml_build_forward_expand(graph, probabilities[i]);
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(model.backend, ggml_graph_node(graph, i)),
                    "Explicit backend lacks a head graph operation; no fallback");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "Head graph allocation failed");
    }
    bool compute(output & result, std::string & error) {
        ggml_backend_tensor_set(hidden, values.hidden.data(), 0, values.hidden.size() * sizeof(float));
        std::vector<int32_t> ids;
        for (const auto & f : values.fields) ids.push_back(f.type);
        ggml_backend_tensor_set(types, ids.data(), 0, ids.size() * sizeof(int32_t));
        for (size_t i = 0; i < lexical.size(); ++i)
            ggml_backend_tensor_set(lexical[i], values.fields[i].lexical_means.data(), 0,
                                    values.fields[i].lexical_means.size() * sizeof(float));
        if (ggml_backend_graph_compute(model.backend, graph) != GGML_STATUS_SUCCESS)
            return fail(error, "Head backend graph compute failed");
        output out;
        out.backend = model.name;
        for (size_t i = 0; i < logits.size(); ++i) {
            field_output f;
            f.id = values.fields[i].id;
            f.logits.resize(values.fields[i].options.size());
            f.probabilities.resize(f.logits.size());
            ggml_backend_tensor_get(logits[i], f.logits.data(), 0, f.logits.size() * sizeof(float));
            ggml_backend_tensor_get(probabilities[i], f.probabilities.data(), 0,
                                    f.probabilities.size() * sizeof(float));
            for (float x : f.logits)
                if (!std::isfinite(x)) return fail(error, "Nonfinite head logits");
            for (float x : f.probabilities)
                if (!std::isfinite(x) || x < 0 || x > 1) return fail(error, "Invalid head probabilities");
            out.fields.push_back(std::move(f));
        }
        result = std::move(out);
        return true;
    }
};
} // namespace

bool validate_input(const input & values, std::string & error) {
    error.clear();
    if (values.tokens < 1 || values.tokens > max_tokens ||
        values.hidden.size() != size_t(values.tokens) * hidden_size || values.fields.empty() ||
        values.fields.size() > max_fields)
        return fail(error, "Head input requires T1..512, F1..16 and exact contiguous F32 hidden shape");
    for (float x : values.hidden)
        if (!std::isfinite(x)) return fail(error, "Nonfinite original hidden input");
    std::set<std::string> ids;
    size_t options = 0;
    for (const auto & f : values.fields) {
        auto valid = [&](span s) { return s.start >= 0 && s.start < s.end && s.end <= values.tokens; };
        if (f.id.empty() || f.id.size() > 256 || f.id.find('\0') != std::string::npos || !ids.insert(f.id).second ||
            f.type < 0 || f.type > 2 || !valid(f.question) || f.options.empty())
            return fail(error, "Invalid ordered field ID/type/question/options");
        options += f.options.size();
        if (options > max_options || f.lexical_means.size() != f.options.size() * hidden_size)
            return fail(error, "Head requires total options<=128 and exact raw lexical means");
        for (auto s : f.options)
            if (!valid(s)) return fail(error, "Empty/out-of-range option span");
        for (float x : f.lexical_means)
            if (!std::isfinite(x)) return fail(error, "Nonfinite original lexical input");
    }
    return true;
}
context * init(const char * path, const char * receipt, const char * worker, const char * backend, int threads,
               std::string & error) {
    error.clear();
    if (!gate(error)) return nullptr;
    if (!path || !backend || threads < 1 || threads > 64) {
        fail(error, "Head model/backend/1..64 threads required");
        return nullptr;
    }
    auto * meta = core_gguf::open_metadata(path);
    if (!meta) {
        fail(error, "Cannot open head GGUF metadata");
        return nullptr;
    }
    const bool valid = metadata_valid(meta, receipt, worker, error);
    core_gguf::free_metadata(meta);
    if (!valid) return nullptr;
    try {
        std::unique_ptr<context> model(new context);
        if (std::strcmp(backend, "cpu") == 0) {
            model->backend = core_cpu_backend::init();
            require(core_cpu_backend::is_cpu(model->backend), "Head CPU backend unavailable");
            core_cpu_backend::set_n_threads(model->backend, threads);
        } else {
            ggml_backend_load_all();
            auto * dev = ggml_backend_dev_by_name(backend);
            require(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                        std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") == 0,
                    "Requested explicit CUDA device unavailable; no fallback");
            model->backend = ggml_backend_dev_init(dev, nullptr);
            require(model->backend, "CUDA backend initialization failed");
        }
        model->name = ggml_backend_name(model->backend);
        require(core_gguf::load_weights(path, model->backend, "josev2b_head", model->weights),
                "Head weight load failed");
        std::vector<float> chunk(65536);
        for (const auto & item : model->weights.tensors)
            for (size_t offset = 0; offset < size_t(ggml_nelements(item.second)); offset += chunk.size()) {
                const size_t count = std::min(chunk.size(), size_t(ggml_nelements(item.second)) - offset);
                ggml_backend_tensor_get(item.second, chunk.data(), offset * sizeof(float), count * sizeof(float));
                for (size_t i = 0; i < count; ++i)
                    require(std::isfinite(chunk[i]), "Nonfinite head weight: " + item.first);
            }
        return model.release();
    } catch (const std::exception & e) {
        fail(error, e.what());
        return nullptr;
    }
}
bool validate_packed_shapes(const core_gguf::tensor_map & tensors, std::string & error) {
    error.clear();
    try {
        const auto expected = expected_schema();
        require(tensors.size() == expected.size() && expected.size() == 122,
                "Borrowed head requires exactly122 tensors");
        for (const auto & row : expected) {
            auto found = tensors.find(row.first);
            require(found != tensors.end(), "Missing borrowed head tensor: " + row.first);
            auto * t = found->second;
            require(t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t),
                    "Packed head requires contiguous F32 metadata");
            for (int axis = 0; axis < 4; ++axis)
                require(t->ne[axis] == row.second[axis], "Borrowed head shape differs: " + row.first);
        }
        return true;
    } catch (const std::exception & e) {
        return fail(error, e.what());
    }
}
context * init_borrowed_cpu(const core_gguf::tensor_map & tensors, int threads, std::string & error) {
    error.clear();
    if (!gate(error)) return nullptr;
    if (!core_env::on("CRISPEMBED_JOSEV_PACKED")) {
        fail(error, "Josev packed gate is off");
        return nullptr;
    }
    try {
        require(threads >= 1 && threads <= 64, "Borrowed CPU head requires1..64 threads");
        require(validate_packed_shapes(tensors, error), "Packed head tensor schema rejected");
        for (const auto & item : tensors)
            require(item.second->buffer && ggml_backend_buffer_is_host(item.second->buffer),
                    "Borrowed head requires host storage");
        std::unique_ptr<context> model(new context);
        model->backend = core_cpu_backend::init();
        require(core_cpu_backend::is_cpu(model->backend), "Borrowed head CPU backend unavailable");
        core_cpu_backend::set_n_threads(model->backend, threads);
        model->name = ggml_backend_name(model->backend);
        // No buf/ctx ownership: free_weights clears these borrowed pointer entries
        // while the enclosing packed model still owns every GGML tensor/buffer.
        model->weights.tensors = tensors;
        std::vector<float> chunk(65536);
        for (const auto & item : tensors)
            for (size_t offset = 0; offset < size_t(ggml_nelements(item.second)); offset += chunk.size()) {
                const size_t count = std::min(chunk.size(), size_t(ggml_nelements(item.second)) - offset);
                ggml_backend_tensor_get(item.second, chunk.data(), offset * sizeof(float), count * sizeof(float));
                for (size_t i = 0; i < count; ++i)
                    require(std::isfinite(chunk[i]), "Nonfinite borrowed head weight: " + item.first);
            }
        return model.release();
    } catch (const std::exception & e) {
        fail(error, e.what());
        return nullptr;
    }
}
void free(context * ctx) {
    delete ctx;
}
const char * backend_name(const context * ctx) {
    return ctx ? ctx->name.c_str() : "";
}
bool run(context * ctx, const input & values, output & result, std::string & error) {
    result = {};
    error.clear();
    if (!gate(error)) return false;
    if (!ctx) return fail(error, "Null head context");
    if (!validate_input(values, error)) return false;
    try {
        graph_builder graph(*ctx, values);
        graph.build();
        return graph.compute(result, error);
    } catch (const std::exception & e) {
        return fail(error, e.what());
    }
}
#ifdef CRISPEMBED_JOSEV_HEAD_TESTING
namespace test {
bool run_observed(context * ctx, const input & values, output & result, const observer & callback,
                  std::string & error) {
    result = {};
    error.clear();
    if (!gate(error)) return false;
    if (!ctx || !callback) return fail(error, "Observer/context required");
    if (!validate_input(values, error)) return false;
    try {
        graph_builder graph(*ctx, values);
        graph.observe = true;
        graph.build();
        output pending;
        if (!graph.compute(pending, error)) return false;
        for (const auto & event : graph.stages) {
            stage out;
            out.label = event.label;
            out.occurrence = event.occurrence;
            out.shape = event.shape;
            out.values.resize(ggml_nelements(event.tensor));
            ggml_backend_tensor_get(event.tensor, out.values.data(), 0, out.values.size() * sizeof(float));
            for (float value : out.values) require(std::isfinite(value), "Nonfinite observed head stage");
            callback(out);
        }
        result = std::move(pending);
        return true;
    } catch (const std::exception & e) {
        return fail(error, e.what());
    }
}
bool metadata_controls(std::string & error) {
    try {
        for (int control = 0; control < 8; ++control) {
            auto * ctx = ggml_init({ 128 * ggml_tensor_overhead(), nullptr, true });
            require(ctx, "Metadata fixture allocation failed");
            auto * meta = gguf_init_empty();
            require(meta, "Metadata fixture initialization failed");
            gguf_set_val_str(meta, "general.architecture", arch);
            gguf_set_val_str(meta, "general.license", "apache-2.0");
            gguf_set_val_str(meta, key("source_revision").c_str(), "6f0a31c8ed683aba79097eb3f48df2c9ab560afd");
            gguf_set_val_str(meta, key("checkpoint_sha256").c_str(),
                             "8816d51356876ef9b055e51291a00dfc4a8c741189ca088b59604d039d7852c5");
            gguf_set_val_str(meta, key("original_receipt_sha256").c_str(), original_receipt_sha256);
            gguf_set_val_str(meta, key("original_worker_source").c_str(), original_worker_source);
            gguf_set_val_str(meta, key("calibration_profile").c_str(), "official-pytorch-no-extra-temperature");
            for (const auto & item : std::map<std::string, uint32_t>{ { "hidden_size", 2048 },
                                                                      { "width", 512 },
                                                                      { "routing_layers", 2 },
                                                                      { "layers", 4 },
                                                                      { "heads", 8 },
                                                                      { "feedforward", 2048 } })
                gguf_set_val_u32(meta, key(item.first.c_str()).c_str(), item.second);
            int index = 0;
            for (const auto & item : expected_schema()) {
                if (control == 1 && index++ == 0) continue;
                auto shape = item.second;
                if (control == 2 && index++ == 0) shape[0]++;
                auto * tensor = ggml_new_tensor_4d(ctx, control == 3 && index++ == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                                   shape[0], shape[1], shape[2], shape[3]);
                ggml_set_name(tensor, item.first.c_str());
                gguf_add_tensor(meta, tensor);
            }
            if (control == 4) gguf_set_val_u32(meta, key("heads").c_str(), 4);
            if (control == 5) gguf_set_val_str(meta, key("calibration_profile").c_str(), "extra-temperature");
            if (control == 6) gguf_set_val_str(meta, key("original_receipt_sha256").c_str(), "wrong");
            if (control == 7) {
                auto * extra = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
                ggml_set_name(extra, "head.extra");
                gguf_add_tensor(meta, extra);
            }
            std::string rejection;
            const bool valid = metadata_valid(meta, original_receipt_sha256, original_worker_source, rejection);
            gguf_free(meta);
            ggml_free(ctx);
            require(valid == (control == 0), "Weight-free head metadata control failed");
        }
        return true;
    } catch (const std::exception & e) {
        return fail(error, e.what());
    }
}
} // namespace test
#endif
} // namespace josev2b_head

#undef JOSEV_MARK
