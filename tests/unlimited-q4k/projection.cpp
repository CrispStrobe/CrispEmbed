// Test-only same-input expert projection. Model arithmetic/source are unchanged.
// CPU graph uses actual stored-type mul_mat_id and a separate dequant-F32 arm.
#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
bool include(const char * name, void * target) {
    return *static_cast<std::string *>(target) == name;
}
void save(const std::string & path, const std::vector<float> & values) {
    for (float x : values) require(std::isfinite(x), "nonfinite projection");
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(values.data()), std::streamsize(values.size() * 4));
    require(bool(out), "projection file write");
}
struct context {
    ggml_context * c = nullptr;
    ggml_backend_buffer_t b = nullptr;
    ~context() {
        if (b) ggml_backend_buffer_free(b);
        if (c) ggml_free(c);
    }
};
std::vector<float> project(ggml_backend_t backend, ggml_tensor * weight, const float * h, const int32_t * ids, int rows,
                           bool dequantized) {
    context g;
    g.c = ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true });
    require(g.c, "projection graph context");
    ggml_tensor * x = ggml_new_tensor_3d(g.c, GGML_TYPE_F32, 1280, 6, rows);
    ggml_tensor * selected = ggml_new_tensor_2d(g.c, GGML_TYPE_I32, 6, rows);
    ggml_tensor * w = dequantized ? ggml_new_tensor_3d(g.c, GGML_TYPE_F32, 1280, 896, 64) : weight;
    ggml_tensor * result = ggml_mul_mat_id(g.c, w, x, selected);
    ggml_set_input(x);
    ggml_set_input(selected);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph_custom(g.c, 64, false);
    ggml_build_forward_expand(graph, result);
    g.b = ggml_backend_alloc_ctx_tensors(g.c, backend);
    require(g.b, "projection graph backend allocation");
    std::vector<float> repeated(size_t(rows) * 6 * 1280);
    for (int row = 0; row < rows; ++row)
        for (int k = 0; k < 6; ++k)
            std::copy_n(h + size_t(row) * 1280, 1280, repeated.data() + (size_t(row) * 6 + k) * 1280);
    ggml_backend_tensor_set(x, repeated.data(), 0, repeated.size() * 4);
    ggml_backend_tensor_set(selected, ids, 0, size_t(rows) * 6 * 4);
    if (dequantized) {
        const auto * traits = ggml_get_type_traits(weight->type);
        require(traits && traits->to_float, "dequant trait");
        const size_t row_bytes = ggml_row_size(weight->type, 1280);
        std::vector<uint8_t> raw(row_bytes);
        std::vector<float> decoded(1280);
        for (size_t row = 0; row < size_t(896) * 64; ++row) {
            ggml_backend_tensor_get(weight, raw.data(), row * row_bytes, row_bytes);
            traits->to_float(raw.data(), decoded.data(), 1280);
            for (float value : decoded) require(std::isfinite(value), "nonfinite dequantized weight");
            ggml_backend_tensor_set(w, decoded.data(), row * 1280 * 4, 1280 * 4);
        }
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "projection graph compute");
    std::vector<float> output(size_t(rows) * 6 * 896);
    ggml_backend_tensor_get(result, output.data(), 0, output.size() * 4);
    return output;
}
int body(int argc, char ** argv) {
    if (argc != 6) return 2;
    try {
        std::string name = argv[2];
        require(name.rfind("l.blk.", 0) == 0 && (name.find(".ffn_gate_exps.weight") != std::string::npos ||
                                                 name.find(".ffn_up_exps.weight") != std::string::npos),
                "selected projection name");
        const int threads = std::stoi(argv[5]);
        require(threads >= 1 && threads <= 16, "thread bound");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        require(bool(backend), "explicit CPU backend");
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        core_gguf::WeightLoad weights;
        struct cleanup {
            core_gguf::WeightLoad & w;
            ~cleanup() { core_gguf::free_weights(w); }
        } owner{ weights };
        require(core_gguf::load_weights_filtered(argv[1], backend.get(), include, &name, "unlimited-q4-probe", weights),
                "selected weight load");
        require(weights.tensors.size() == 1, "exact single weight coverage");
        auto * w = weights.tensors.at(name);
        require(w && w->ne[0] == 1280 && w->ne[1] == 896 && w->ne[2] == 64 && w->ne[3] == 1 &&
                    (w->type == GGML_TYPE_F16 || w->type == GGML_TYPE_Q4_K),
                "actual stored expert type/shape");
        crispembed_diff::Ref input;
        require(input.load(argv[3]), "authenticated input adapter load");
        auto h = input.get_f32("hn"), ids = input.get_f32("ids");
        require(h.first && ids.first && h.second % 1280 == 0 && ids.second == h.second / 1280 * 6,
                "exact selected input/ID extents");
        const int rows = int(h.second / 1280);
        require(rows > 0 && rows <= 16 && input.shape("hn") == std::vector<int64_t>({ 1280, 1, rows }) &&
                    input.shape("ids") == std::vector<int64_t>({ 6, 1, rows }),
                "exact C-order input axes");
        std::vector<int32_t> selected(ids.second);
        for (size_t i = 0; i < h.second; ++i) require(std::isfinite(h.first[i]), "finite input");
        for (size_t i = 0; i < ids.second; ++i) {
            const float value = ids.first[i];
            require(std::isfinite(value) && value >= 0 && value < 64 && std::floor(value) == value, "exact router ID");
            selected[i] = int32_t(value);
        }
        const auto native = project(backend.get(), w, h.first, selected.data(), rows, false);
        const auto control = project(backend.get(), w, h.first, selected.data(), rows, true);
        save(std::string(argv[4]) + ".native.f32", native);
        save(std::string(argv[4]) + ".dequant.f32", control);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "unlimited-q4 projection: %s\n", e.what());
        return 2;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
