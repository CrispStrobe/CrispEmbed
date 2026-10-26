// Validation-only actual ggml RoPE and frozen-table rotation controls. No model weights.
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "crispembed_diff.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Values = std::map<std::string, std::vector<float>>;
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}
std::vector<float> read(const crispembed_diff::Ref & ref, const std::string & name, size_t count) {
    auto value = ref.get_f32(name);
    require(value.first && value.second == count, "Missing/wrong-size target: " + name);
    std::vector<float> out(value.first, value.first + count);
    require(std::all_of(out.begin(), out.end(), [](float x) { return std::isfinite(x); }), "Nonfinite input");
    return out;
}
struct Work {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~Work() {
        if (allocator) ggml_gallocr_free(allocator);
        if (ctx) ggml_free(ctx);
    }
};
// A separate ordinary graph is compared bit-for-bit with the observed graph.
Values execute(ggml_backend_t backend, const std::vector<float> & q, const std::vector<float> & k,
               const std::vector<float> & cosine, const std::vector<float> & sine,
               const std::vector<int32_t> & positions, float theta, bool observed) {
    const int tokens = int(positions.size()), d = 128, heads = 6;
    Work work;
    work.ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    require(work.ctx, "Graph metadata allocation failed");
    auto * ctx = work.ctx;
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    auto * qi = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, tokens);
    auto * ki = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, heads, tokens);
    auto * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    for (auto * t : { qi, ki, pos }) ggml_set_input(t);
    auto rope = [&](ggml_tensor * x) {
        return ggml_rope_ext(ctx, x, pos, nullptr, d, GGML_ROPE_TYPE_NEOX, 0, theta, 1.f, 0.f, 1.f, 0.f, 0.f);
    };
    std::map<std::string, ggml_tensor *> outputs = { { "q_normal", rope(qi) }, { "k_normal", rope(ki) } };
    ggml_tensor *co = nullptr, *si = nullptr, *basis = nullptr;
    std::vector<float> unit;
    if (observed) {
        co = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, tokens);
        si = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, tokens);
        basis = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, tokens);
        for (auto * t : { co, si, basis }) ggml_set_input(t);
        unit.resize(size_t(tokens) * d);
        for (int t = 0; t < tokens; ++t) std::fill_n(unit.begin() + size_t(t) * d, d / 2, 1.f);
        outputs["basis"] = rope(basis); // Actual operator exposes cos/sin on the split-half basis.
        auto source_rotation = [&](ggml_tensor * x) {
            auto * left = ggml_view_3d(ctx, x, d / 2, heads, tokens, x->nb[1], x->nb[2], 0);
            auto * right = ggml_view_3d(ctx, x, d / 2, heads, tokens, x->nb[1], x->nb[2], (d / 2) * sizeof(float));
            auto * rotated = ggml_concat(ctx, ggml_neg(ctx, right), left, 0);
            return ggml_add(ctx, ggml_mul(ctx, x, co), ggml_mul(ctx, rotated, si));
        };
        outputs["q_source"] = source_rotation(qi);
        outputs["k_source"] = source_rotation(ki);
    }
    for (auto & row : outputs) {
        ggml_set_output(row.second);
        ggml_build_forward_expand(graph, row.second);
    }
    work.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(work.allocator && ggml_gallocr_alloc_graph(work.allocator, graph), "Graph allocation failed");
    ggml_backend_tensor_set(qi, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(ki, k.data(), 0, k.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    std::vector<int32_t> copied_positions(positions.size());
    ggml_backend_tensor_get(pos, copied_positions.data(), 0, copied_positions.size() * sizeof(int32_t));
    require(copied_positions == positions, "Position transfer changed bits");
    if (observed) {
        ggml_backend_tensor_set(co, cosine.data(), 0, cosine.size() * sizeof(float));
        ggml_backend_tensor_set(si, sine.data(), 0, sine.size() * sizeof(float));
        ggml_backend_tensor_set(basis, unit.data(), 0, unit.size() * sizeof(float));
        std::vector<float> copied_unit(unit.size());
        ggml_backend_tensor_get(basis, copied_unit.data(), 0, copied_unit.size() * sizeof(float));
        require(exact(copied_unit, unit), "Basis transfer changed bits");
    }
    // Verify the supplied F32 values, not merely an asserted source label.
    for (auto row : { std::make_pair(qi, &q), std::make_pair(ki, &k) }) {
        std::vector<float> copied(row.second->size());
        ggml_backend_tensor_get(row.first, copied.data(), 0, copied.size() * sizeof(float));
        require(exact(copied, *row.second), "Input transfer changed bits");
    }
    if (observed)
        for (auto row : { std::make_pair(co, &cosine), std::make_pair(si, &sine) }) {
            std::vector<float> copied(row.second->size());
            ggml_backend_tensor_get(row.first, copied.data(), 0, copied.size() * sizeof(float));
            require(exact(copied, *row.second), "Frozen table transfer changed bits");
        }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "RoPE graph compute failed");
    Values result;
    for (auto & row : outputs) {
        auto & values = result[row.first];
        values.resize(size_t(ggml_nelements(row.second)));
        ggml_backend_tensor_get(row.second, values.data(), 0, values.size() * sizeof(float));
    }
    return result;
}
int run(int argc, char ** argv) {
    // Existing experimental architecture gate is checked before all file IO.
    if (!core_env::on("CRISPEMBED_ULTRABERT")) return 2;
    require(argc == 3, "Usage: test-ultrabert-rope operator-reference.gguf raw-output-directory");
    crispembed_diff::Ref ref;
    require(ref.load(argv[1]) && ref.meta("ref.scope") == "same-input-local-rope-v1" &&
                ref.meta("ref.model") == "dmis-lab/UltraBERT-Embed-150M" &&
                ref.meta("ref.revision") == "367d5343b5a0164d57b810cedf5de4b8983d5984",
            "Wrong local operator reference");
    auto * backend = ggml_backend_cpu_init();
    require(backend, "CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(backend, 2);
    int metrics = 0, controls = 0, failures = 0;
    const std::string raw_root = argv[2];
    auto compare = [&](const std::string & mode, const std::string & target, const std::vector<float> & values) {
        const auto path = raw_root + "/" + mode;
        std::filesystem::create_directories(path);
#ifdef _WIN32
        _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", path.c_str());
#else
        setenv("CRISPEMBED_DIFF_DUMP_DIR", path.c_str(), 1);
#endif
        auto oracle = read(ref, target, values.size());
        auto r = ref.compare(target, values.data(), values.size());
        bool passed = r.found && r.n_elem == values.size() && r.cos_min >= .99999f && r.max_abs <= .002f &&
                      std::isfinite(r.cos_min) && std::isfinite(r.max_abs);
        for (float v : values) passed &= std::isfinite(v);
        ++metrics;
        failures += !passed;
        std::printf("ROPE\t%s\t%s\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%d\t%d\n", mode.c_str(), target.c_str(),
                    r.cos_min, r.max_abs, r.mean_abs, r.rms, r.mine_norm, r.ref_norm, int(passed),
                    int(exact(values, oracle)));
    };
    for (int layer = 0; layer < 24; ++layer) {
        const std::string prefix = "layer_" + std::to_string(layer) + ".";
        const auto shape = ref.shape(prefix + "q_input");
        require(shape.size() == 3 && shape[0] == 768 && shape[1] >= 1 && shape[1] <= 512 && shape[2] >= 1 &&
                    shape[2] <= 2,
                "Invalid bounded Q shape");
        const int tokens = int(shape[1]), batch = int(shape[2]);
        const size_t n = size_t(batch) * tokens;
        auto qs = read(ref, prefix + "q_input", n * 768), ks = read(ref, prefix + "k_input", n * 768),
             cos = read(ref, prefix + "cos_frozen", n * 128), sin = read(ref, prefix + "sin_frozen", n * 128),
             pos = read(ref, prefix + "positions", n);
        Values all;
        bool neutral = true;
        for (int b = 0; b < batch; ++b) {
            std::vector<float> q(qs.begin() + size_t(b) * tokens * 768, qs.begin() + size_t(b + 1) * tokens * 768),
                k(ks.begin() + size_t(b) * tokens * 768, ks.begin() + size_t(b + 1) * tokens * 768),
                co(cos.begin() + size_t(b) * tokens * 128, cos.begin() + size_t(b + 1) * tokens * 128),
                si(sin.begin() + size_t(b) * tokens * 128, sin.begin() + size_t(b + 1) * tokens * 128);
            std::vector<int32_t> positions(tokens);
            for (int t = 0; t < tokens; ++t) {
                const float p = pos[size_t(b) * tokens + t];
                require(p >= 0 && p < 512 && p == std::floor(p), "Invalid frozen position ID");
                positions[t] = int32_t(p);
            }
            const float theta = layer % 3 == 2 ? 2560000.f : 10000.f;
            const auto ordinary = execute(backend, q, k, co, si, positions, theta, false);
            const auto observed = execute(backend, q, k, co, si, positions, theta, true);
            neutral &= exact(ordinary.at("q_normal"), observed.at("q_normal")) &&
                       exact(ordinary.at("k_normal"), observed.at("k_normal"));
            for (auto & row : observed)
                all[row.first].insert(all[row.first].end(), row.second.begin(), row.second.end());
        }
        std::printf("ROPEEXACT\tobserver.layer_%d\t%d\n", layer, int(neutral));
        std::printf("ROPEEXACT\tinputs.layer_%d\t1\n", layer);
        controls += 2;
        failures += !neutral;
        for (const char * scope : { "frozen", "current" }) {
            const std::string suffix = "_" + std::string(scope), mode = "normal_" + std::string(scope);
            compare(mode, prefix + "basis" + suffix, all.at("basis"));
            compare(mode, prefix + "q" + suffix, all.at("q_normal"));
            compare(mode, prefix + "k" + suffix, all.at("k_normal"));
        }
        compare("source_frozen", prefix + "q_frozen", all.at("q_source"));
        compare("source_frozen", prefix + "k_frozen", all.at("k_source"));
    }
    ggml_backend_free(backend);
    std::printf("ROPEEND\t%d\t%d\t%d\n", metrics, controls, failures);
    require(metrics == 192 && controls == 48, "Incomplete RoPE comparisons");
    return failures ? 1 : 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "RoPE audit error: %s\n", e.what());
        core_util::clean_exit(2);
    }
}
