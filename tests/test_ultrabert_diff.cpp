// Test-only prepared-ID UltraBERT graph. No production architecture dispatch.
// Original F32/eager tensors come from dump_ultrabert_reference.py.
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Resources {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    ggml_context * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~Resources() {
        if (allocator) ggml_gallocr_free(allocator);
        if (graph) ggml_free(graph);
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};
using Outputs = std::map<std::string, std::vector<float>>;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void append(Outputs & out, const std::string & name, const std::vector<float> & values) {
    auto & target = out[name];
    target.insert(target.end(), values.begin(), values.end());
}

void run_row(Resources & resources, const std::vector<int32_t> & ids, const std::vector<float> & keep, Outputs & out) {
    const int T = (int)ids.size(), H = 768, D = 128, heads = 6;
    require(T > 0 && T <= 512 && keep.size() == ids.size(), "Invalid bounded input");
    for (int i = 0; i < T; ++i) {
        require(ids[i] >= 0 && ids[i] < 49152 && (keep[i] == 0 || keep[i] == 1), "Invalid ID/mask");
    }
    require(ids[0] == 0 && keep[0] == 1, "BOS must be attended");
    auto weight = [&](const std::string & name, int64_t ne0, int64_t ne1 = 1) {
        auto found = resources.weights.tensors.find(name);
        require(found != resources.weights.tensors.end(), "Missing weight: " + name);
        auto * t = found->second;
        require(t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == 1 && t->ne[3] == 1 &&
                    (t->type == GGML_TYPE_F32 || (ne1 > 1 && t->type == GGML_TYPE_F16)),
                "Weight shape/type differs: " + name);
        return t;
    };
    const size_t graph_size = 8192;
    resources.graph = ggml_init({ 32 * 1024 * 1024, nullptr, true });
    require(resources.graph != nullptr, "Graph metadata allocation failed");
    auto * g = resources.graph;
    auto * gf = ggml_new_graph_custom(g, graph_size, false);
    auto * input = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    auto * positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    auto * local_mask = ggml_new_tensor_3d(g, GGML_TYPE_F32, T, T, 1);
    auto * global_mask = ggml_new_tensor_3d(g, GGML_TYPE_F32, T, T, 1);
    for (auto * t : { input, positions, local_mask, global_mask }) ggml_set_input(t);
    std::vector<float> local((size_t)T * T), global(local.size());
    std::vector<int32_t> pos(T);
    for (int q = 0; q < T; ++q) {
        pos[q] = q;
        for (int k = 0; k < T; ++k) {
            const size_t index = (size_t)q * T + k;
            global[index] = keep[k] ? 0 : std::numeric_limits<float>::lowest();
            local[index] = keep[k] && std::abs(q - k) <= 128 ? 0 : std::numeric_limits<float>::lowest();
        }
    }
    std::vector<std::pair<std::string, ggml_tensor *>> stages;
    auto observe = [&](const std::string & name, ggml_tensor * t) {
        ggml_set_output(t);
        stages.emplace_back(name, t);
        return t;
    };
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(g, ggml_rms_norm(g, t, 1e-6f), w); };
    auto * cur = observe("emb_out", ggml_get_rows(g, weight("token_embd.weight", H, 49152), input));
    for (int i = 0; i < 24; ++i) {
        const std::string prefix = "layer_" + std::to_string(i) + ".";
        const std::string wp = "blk." + std::to_string(i) + ".";
        auto * residual = observe(prefix + "input", cur);
        auto * normalized = observe(prefix + "attn_norm", norm(cur, weight(wp + "attn_norm.weight", H)));
        auto * q = observe(prefix + "q", ggml_mul_mat(g, weight(wp + "attn_q.weight", H, H), normalized));
        auto * k = observe(prefix + "k", ggml_mul_mat(g, weight(wp + "attn_k.weight", H, H), normalized));
        auto * v = observe(prefix + "v", ggml_mul_mat(g, weight(wp + "attn_v.weight", H, H), normalized));
        const bool is_global = i % 3 == 2;
        const float theta = is_global ? 2560000.f : 10000.f;
        q = ggml_rope_ext(g, ggml_reshape_3d(g, q, D, heads, T), positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, theta,
                          1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(g, ggml_reshape_3d(g, k, D, heads, T), positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, theta,
                          1.f, 0.f, 1.f, 0.f, 0.f);
        q = ggml_permute(g, q, 0, 2, 1, 3);
        k = ggml_permute(g, k, 0, 2, 1, 3);
        v = ggml_permute(g, ggml_reshape_3d(g, v, D, heads, T), 0, 2, 1, 3);
        auto * scores = ggml_scale(g, ggml_mul_mat(g, k, q), 1.f / std::sqrt((float)D));
        scores = ggml_soft_max(g, ggml_add(g, scores, is_global ? global_mask : local_mask));
        auto * vt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
        auto * attention = ggml_mul_mat(g, vt, scores);
        attention = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, attention, 0, 2, 1, 3)), H, T);
        attention = observe(prefix + "attn_out", ggml_mul_mat(g, weight(wp + "attn_output.weight", H, H), attention));
        cur = observe(prefix + "post_attn_residual", ggml_add(g, residual, attention));
        residual = cur;
        normalized = observe(prefix + "ffn_norm", norm(cur, weight(wp + "ffn_norm.weight", H)));
        auto * gate = observe(prefix + "gate", ggml_mul_mat(g, weight(wp + "ffn_gate.weight", H, 1024), normalized));
        auto * up = observe(prefix + "up", ggml_mul_mat(g, weight(wp + "ffn_up.weight", H, 1024), normalized));
        auto * ffn = ggml_mul_mat(g, weight(wp + "ffn_down.weight", 1024, H), ggml_mul(g, ggml_silu(g, gate), up));
        ffn = observe(prefix + "ffn_out", ffn);
        cur = observe(prefix + "output", ggml_add(g, residual, ffn));
        append(out, prefix + "attention_mask", is_global ? global : local);
    }
    cur = observe("final_hidden", norm(cur, weight("output_norm.weight", H)));
    ggml_build_forward_expand(gf, cur);
    resources.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(resources.backend));
    require(resources.allocator && ggml_gallocr_alloc_graph(resources.allocator, gf), "Graph allocation failed");
    ggml_backend_tensor_set(input, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(local_mask, local.data(), 0, local.size() * sizeof(float));
    ggml_backend_tensor_set(global_mask, global.data(), 0, global.size() * sizeof(float));
    require(ggml_backend_graph_compute(resources.backend, gf) == GGML_STATUS_SUCCESS, "Graph compute failed");
    for (auto & stage : stages) {
        std::vector<float> values((size_t)ggml_nelements(stage.second));
        ggml_backend_tensor_get(stage.second, values.data(), 0, values.size() * sizeof(float));
        append(out, stage.first, values);
    }
    std::vector<float> cls(H);
    ggml_backend_tensor_get(cur, cls.data(), 0, H * sizeof(float));
    append(out, "cls_raw", cls);
    double magnitude = 0;
    for (float value : cls) magnitude += (double)value * value;
    const double divisor = std::max(1e-12, std::sqrt(magnitude));
    for (float & value : cls) value = (float)(value / divisor);
    append(out, "cls_normalized", cls);
    ggml_gallocr_free(resources.allocator);
    resources.allocator = nullptr;
    ggml_free(resources.graph);
    resources.graph = nullptr;
}

// Separate graphs over frozen original inputs. These never feed ordinary inference.
void replay_operators(Resources & resources, const crispembed_diff::Ref & ref) {
    auto embedding = ref.get_f32("emb_out");
    auto residual = ref.get_f32("layer_14.post_attn_residual");
    auto normalized = ref.get_f32("layer_14.ffn_norm");
    auto gates = ref.get_f32("layer_14.gate");
    auto ups = ref.get_f32("layer_14.up");
    require(normalized.second > 0 && normalized.second % 768 == 0 && embedding.second == normalized.second &&
                residual.second == normalized.second && gates.second == normalized.second / 768 * 1024 &&
                ups.second == gates.second,
            "Invalid original operator inputs");
    const int64_t rows = (int64_t)normalized.second / 768;
    require(rows <= 1024, "Operator replay row bound exceeded");
    resources.graph = ggml_init({ 8 * 1024 * 1024, nullptr, true });
    require(resources.graph != nullptr, "Operator metadata allocation failed");
    auto * g = resources.graph;
    auto * gf = ggml_new_graph_custom(g, 512, false);
    auto * e = ggml_new_tensor_2d(g, GGML_TYPE_F32, 768, rows);
    auto * r = ggml_new_tensor_2d(g, GGML_TYPE_F32, 768, rows);
    auto * n = ggml_new_tensor_2d(g, GGML_TYPE_F32, 768, rows);
    auto * original_gate = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1024, rows);
    auto * original_up = ggml_new_tensor_2d(g, GGML_TYPE_F32, 1024, rows);
    for (auto * t : { e, r, n, original_gate, original_up }) ggml_set_input(t);
    const auto & weights = resources.weights.tensors;
    auto * first_norm = ggml_mul(g, ggml_rms_norm(g, e, 1e-6f), weights.at("blk.0.attn_norm.weight"));
    auto * norm = ggml_mul(g, ggml_rms_norm(g, r, 1e-6f), weights.at("blk.14.ffn_norm.weight"));
    auto * gate = ggml_mul_mat(g, weights.at("blk.14.ffn_gate.weight"), n);
    auto * up = ggml_mul_mat(g, weights.at("blk.14.ffn_up.weight"), n);
    auto * mlp = ggml_mul_mat(g, weights.at("blk.14.ffn_down.weight"), ggml_mul(g, ggml_silu(g, gate), up));
    auto * glu_down =
        ggml_mul_mat(g, weights.at("blk.14.ffn_down.weight"), ggml_mul(g, ggml_silu(g, original_gate), original_up));
    const std::vector<ggml_tensor *> nodes = { first_norm, norm, gate, up, mlp, glu_down };
    const std::vector<std::string> modes = { "original_embedding", "original_residual", "original_ffn_norm",
                                             "original_ffn_norm",  "original_ffn_norm", "original_gate_up" };
    const std::vector<std::string> names = { "layer_0.attn_norm", "layer_14.ffn_norm", "layer_14.gate",
                                             "layer_14.up",       "layer_14.ffn_out",  "layer_14.ffn_out" };
    for (auto * t : nodes) {
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
    }
    resources.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(resources.backend));
    require(resources.allocator && ggml_gallocr_alloc_graph(resources.allocator, gf), "Operator allocation failed");
    ggml_backend_tensor_set(e, embedding.first, 0, embedding.second * sizeof(float));
    ggml_backend_tensor_set(r, residual.first, 0, residual.second * sizeof(float));
    ggml_backend_tensor_set(n, normalized.first, 0, normalized.second * sizeof(float));
    ggml_backend_tensor_set(original_gate, gates.first, 0, gates.second * sizeof(float));
    ggml_backend_tensor_set(original_up, ups.first, 0, ups.second * sizeof(float));
    require(ggml_backend_graph_compute(resources.backend, gf) == GGML_STATUS_SUCCESS, "Operator compute failed");
    for (size_t i = 0; i < nodes.size(); ++i) {
        std::vector<float> values((size_t)ggml_nelements(nodes[i]));
        ggml_backend_tensor_get(nodes[i], values.data(), 0, values.size() * sizeof(float));
        const auto report = ref.compare(names[i], values.data(), values.size(), 0);
        bool finite = true;
        for (float value : values) finite = finite && std::isfinite(value);
        const bool pass = finite && report.found && std::isfinite(report.cos_min) && std::isfinite(report.max_abs) &&
                          std::isfinite(report.mean_abs) && std::isfinite(report.rms) &&
                          std::isfinite(report.mine_norm) && std::isfinite(report.ref_norm) &&
                          report.cos_min >= 0.99999 && report.max_abs <= 0.002;
        printf("OPERATOR\t%s\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n", modes[i].c_str(), names[i].c_str(),
               report.cos_min, report.max_abs, report.mean_abs, report.rms, report.mine_norm, report.ref_norm,
               pass ? 1 : 0);
    }
    ggml_gallocr_free(resources.allocator);
    resources.allocator = nullptr;
    ggml_free(resources.graph);
    resources.graph = nullptr;
}

int run(int argc, char ** argv) {
    // Explicit test-only opt-in, with zero behaving as off. Gate checked before IO.
    if (!core_env::on("CRISPEMBED_ULTRABERT")) return 2;
    require(argc == 3 || (argc == 4 && std::string(argv[3]) == "--operator-audit"),
            "Usage: test-ultrabert-diff model.gguf reference.gguf [--operator-audit]");
    crispembed_diff::Ref ref;
    require(ref.load(argv[2]), "Reference load failed");
    require(ref.meta("ref.revision") == "367d5343b5a0164d57b810cedf5de4b8983d5984" &&
                ref.meta("ref.model") == "dmis-lab/UltraBERT-Embed-150M",
            "Wrong original reference source");
    auto * metadata = core_gguf::open_metadata(argv[1]);
    require(metadata != nullptr, "Model metadata load failed");
    const auto windows = core_gguf::kv_i32_array(metadata, "ultrabert.attention.window_radius");
    const auto thetas = core_gguf::kv_f32_array(metadata, "ultrabert.rope.freq_base_per_layer");
    bool contract =
        core_gguf::kv_u32(metadata, "ultrabert.embedding_length", 0) == 768 &&
        core_gguf::kv_u32(metadata, "ultrabert.block_count", 0) == 24 &&
        core_gguf::kv_u32(metadata, "ultrabert.feed_forward_length", 0) == 1024 &&
        core_gguf::kv_u32(metadata, "ultrabert.attention.head_count", 0) == 6 &&
        core_gguf::kv_u32(metadata, "ultrabert.attention.head_count_kv", 0) == 6 &&
        core_gguf::kv_u32(metadata, "ultrabert.attention.key_length", 0) == 128 &&
        core_gguf::kv_f32(metadata, "ultrabert.attention.layer_norm_rms_epsilon", 0) == 1e-6f &&
        core_gguf::kv_str(metadata, "ultrabert.rope.layout", "") == "split-half" &&
        core_gguf::kv_str(metadata, "ultrabert.source.revision", "") == "367d5343b5a0164d57b810cedf5de4b8983d5984" &&
        !core_gguf::kv_bool(metadata, "ultrabert.pooling.normalize_default", true) &&
        core_gguf::kv_str(metadata, "general.architecture", "") == "ultrabert" &&
        core_gguf::kv_str(metadata, "ultrabert.pooling", "") == "cls" &&
        !core_gguf::kv_bool(metadata, "ultrabert.attention.causal", true) && windows.size() == 24 &&
        thetas.size() == 24;
    for (int i = 0; contract && i < 24; ++i) {
        contract = windows[i] == (i % 3 == 2 ? -1 : 128) && thetas[i] == (i % 3 == 2 ? 2560000.f : 10000.f);
    }
    core_gguf::free_metadata(metadata);
    require(contract, "Unsupported UltraBERT metadata contract");
    Resources resources;
    resources.backend = ggml_backend_cpu_init();
    require(resources.backend != nullptr, "CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(resources.backend, 2);
    require(core_gguf::load_weights(argv[1], resources.backend, "ultrabert-test", resources.weights),
            "Weight load failed");
    require(resources.weights.tensors.size() == 218, "Unexpected tensor coverage");
    auto id_data = ref.get_f32("input_ids");
    auto mask_data = ref.get_f32("attention_mask");
    auto shape = ref.shape("input_ids");
    require(shape.size() == 2 && shape[0] > 0 && shape[0] <= 512 && shape[1] > 0 && shape[1] <= 2 &&
                id_data.second == (size_t)(shape[0] * shape[1]) && mask_data.second == id_data.second,
            "Invalid reference input shape");
    const int T = (int)shape[0], B = (int)shape[1];
    Outputs outputs;
    for (int b = 0; b < B; ++b) {
        std::vector<int32_t> ids(T);
        std::vector<float> mask(T);
        for (int t = 0; t < T; ++t) {
            float value = id_data.first[b * T + t];
            require(std::isfinite(value) && value >= 0 && value < 49152 && value == std::floor(value),
                    "Invalid prepared ID");
            ids[t] = (int32_t)value;
            mask[t] = mask_data.first[b * T + t];
        }
        run_row(resources, ids, mask, outputs);
    }
    // Positions are broadcast by the original Python model, even for B=2.
    for (int i = 0; i < 24; ++i) {
        std::vector<float> positions(T);
        for (int t = 0; t < T; ++t) positions[t] = (float)t;
        outputs["layer_" + std::to_string(i) + ".position_ids"] = positions;
    }
    outputs["input_ids"] = std::vector<float>(id_data.first, id_data.first + id_data.second);
    outputs["attention_mask"] = std::vector<float>(mask_data.first, mask_data.first + mask_data.second);
    int failed = 0, numeric = 0, exact = 0;
    for (const auto & output : outputs) {
        const auto & name = output.first;
        const auto & values = output.second;
        auto expected = ref.get_f32(name);
        require(expected.first && expected.second == values.size(), "Missing/wrong-size reference: " + name);
        const bool structural = name == "input_ids" || name == "attention_mask" ||
                                name.find(".attention_mask") != std::string::npos ||
                                name.find(".position_ids") != std::string::npos;
        if (structural) {
            const bool pass = std::equal(values.begin(), values.end(), expected.first);
            printf("EXACT\t%s\t%d\n", name.c_str(), pass ? 1 : 0);
            failed += !pass;
            ++exact;
        } else {
            bool finite = true;
            for (float value : values) finite = finite && std::isfinite(value);
            const auto r = ref.compare(name, values.data(), values.size(), 0);
            const bool pass = finite && r.found && std::isfinite(r.cos_min) && std::isfinite(r.max_abs) &&
                              std::isfinite(r.mean_abs) && std::isfinite(r.rms) && std::isfinite(r.mine_norm) &&
                              std::isfinite(r.ref_norm) && r.cos_min >= 0.99999 && r.max_abs <= 0.002;
            printf("STAGE\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\n", name.c_str(), r.cos_min, r.max_abs,
                   r.mean_abs, r.rms, r.mine_norm, r.ref_norm, pass ? 1 : 0);
            failed += !pass;
            ++numeric;
        }
    }
    require(numeric == 292 && exact == 50, "Incomplete native stage comparisons");
    printf("TOTAL\t%d\t%d\t%d\n", numeric, exact, failed);
    if (argc == 4) replay_operators(resources, ref);
    return failed ? 1 : 0;
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        fprintf(stderr, "UltraBERT test error: %s\n", error.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
