// Test-only operator audit. No model loader or production routing changes.
// source_topk adapts PyTorch v2.8.0 TopKImpl.h at commit
// ba56102387ef21a3b04b357e5b183d48f0afefc7; BSD-3-Clause notice: NOTICE.pytorch.
#include "core/clean_exit.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void require(bool v, const char * s) {
    if (!v) throw std::runtime_error(s);
}
// Exact finite F32 E=8,K=2 CPU branch. No index secondary comparator.
static std::array<int32_t, 2> source_topk(const float * p) {
    using element = std::pair<float, int64_t>;
    std::vector<element> queue(8);
    for (int i = 0; i < 8; ++i) queue[i] = { p[i], i };
    const auto compare = [](const element & x, const element & y) {
        return (std::isnan(x.first) && !std::isnan(y.first)) || x.first > y.first;
    };
    // k*64<=dim_size is false (128<=8). sorted=True sorts [begin,begin+k-1).
    std::nth_element(queue.begin(), queue.begin() + 1, queue.end(), compare);
    std::sort(queue.begin(), queue.begin() + 1, compare);
    return { int32_t(queue[0].second), int32_t(queue[1].second) };
}
template <typename T> static void dump(const std::string & name, const std::vector<T> & v) {
    std::ofstream f(name, std::ios::binary);
    require(bool(f), "Output open failed");
    f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
    require(bool(f), "Output write failed");
}
struct owned {
    ggml_context * context = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~owned() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};
static int body(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--compiler-info") {
            std::printf("compiler=%s\n", __VERSION__);
#ifdef __GLIBCXX__
            std::printf("__GLIBCXX__=%d\n", __GLIBCXX__);
#endif
#ifdef _GLIBCXX_RELEASE
            std::printf("_GLIBCXX_RELEASE=%d\n", _GLIBCXX_RELEASE);
#endif
#ifdef _LIBCPP_VERSION
            std::printf("_LIBCPP_VERSION=%d\n", _LIBCPP_VERSION);
#endif
            return 0;
        }
        require(argc == 5, "Expected packet output-prefix probabilities|logits threads");
        const bool logits = std::string(argv[3]) == "logits";
        require(logits || std::string(argv[3]) == "probabilities", "Invalid operator mode");
        require(std::string(argv[4]) == "1" || std::string(argv[4]) == "2", "Threads must be 1 or 2");
        std::ifstream f(argv[1], std::ios::binary);
        char magic[8];
        uint32_t n = 0;
        f.read(magic, 8);
        f.read(reinterpret_cast<char *>(&n), 4);
        require(bool(f) && std::memcmp(magic, "MRTOPK1\0", 8) == 0 && n >= 1 && n <= 64,
                "Invalid bounded router packet");
        std::vector<float> input(size_t(n) * 8);
        f.read(reinterpret_cast<char *>(input.data()), std::streamsize(input.size() * 4));
        require(bool(f) && f.peek() == std::char_traits<char>::eof(), "Packet extent differs");
        for (float x : input) require(std::isfinite(x), "Nonfinite packet input");
        owned o;
        o.context = ggml_init({ 4 * 1024 * 1024, nullptr, true });
        require(o.context != nullptr, "Context initialization failed");
        o.backend = ggml_backend_cpu_init();
        require(o.backend != nullptr, "Explicit CPU backend required");
        ggml_backend_cpu_set_n_threads(o.backend, std::stoi(argv[4]));
        auto * x = ggml_new_tensor_2d(o.context, GGML_TYPE_F32, 8, n);
        auto * probs = logits ? ggml_soft_max(o.context, x) : x;
        auto * sorted = ggml_argsort(o.context, probs, GGML_SORT_ORDER_DESC);
        auto * chosen = ggml_view_2d(o.context, sorted, 2, n, 8 * sizeof(int32_t), 0);
        auto * picked = ggml_reshape_2d(
            o.context, ggml_get_rows(o.context, ggml_reshape_3d(o.context, probs, 1, 8, n), chosen), 2, n);
        auto * eps = ggml_new_tensor_1d(o.context, GGML_TYPE_F32, 1);
        auto * weight = ggml_div(o.context, picked, ggml_add(o.context, ggml_sum_rows(o.context, picked), eps));
        auto * graph = ggml_new_graph(o.context);
        ggml_build_forward_expand(graph, weight);
        o.buffer = ggml_backend_alloc_ctx_tensors(o.context, o.backend);
        require(o.buffer != nullptr, "Actual backend allocation required");
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
        const float denominator_epsilon = 1e-9f;
        ggml_backend_tensor_set(eps, &denominator_epsilon, 0, 4);
        require(ggml_backend_graph_compute(o.backend, graph) == GGML_STATUS_SUCCESS, "CPU operator graph failed");
        std::vector<float> p(input.size()), selected(size_t(n) * 2), weights(selected.size());
        std::vector<int32_t> indices(selected.size()), cpp_indices(selected.size());
        std::vector<float> cpp_values(selected.size()), cpp_weights(selected.size());
        ggml_backend_tensor_get(probs, p.data(), 0, p.size() * 4);
        ggml_backend_tensor_get(picked, selected.data(), 0, selected.size() * 4);
        ggml_backend_tensor_get(weight, weights.data(), 0, weights.size() * 4);
        // chosen is a strided view; read the contiguous actual argsort owner.
        std::vector<int32_t> all_indices(size_t(n) * 8);
        ggml_backend_tensor_get(sorted, all_indices.data(), 0, all_indices.size() * 4);
        for (size_t row = 0; row < n; ++row) {
            auto ids = source_topk(p.data() + row * 8);
            const float sum = p[row * 8 + ids[0]] + p[row * 8 + ids[1]] + denominator_epsilon;
            for (size_t j = 0; j < 2; ++j) {
                const size_t at = row * 2 + j;
                indices[at] = all_indices[row * 8 + j];
                cpp_indices[at] = ids[j];
                cpp_values[at] = p[row * 8 + ids[j]];
                cpp_weights[at] = cpp_values[at] / sum;
            }
        }
        const std::string prefix(argv[2]);
        dump(prefix + ".probabilities.f32", p);
        dump(prefix + ".ggml_indices.i32", indices);
        dump(prefix + ".ggml_values.f32", selected);
        dump(prefix + ".ggml_weights.f32", weights);
        dump(prefix + ".source_indices.i32", cpp_indices);
        dump(prefix + ".source_values.f32", cpp_values);
        dump(prefix + ".source_weights.f32", cpp_weights);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "Router operator rejected: %s\n", e.what());
        return 2;
    }
}
int main(int argc, char ** argv) {
    core_util::clean_exit(body(argc, argv));
}
