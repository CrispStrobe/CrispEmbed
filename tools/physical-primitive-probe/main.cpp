// Reference diagnostic only; builds separately against either pinned ggml tree.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
static bool read(const char * path, void * data, size_t size) {
    std::ifstream f(path, std::ios::binary);
    return bool(f.read(static_cast<char *>(data), size)) && f.peek() == EOF;
}
static int run(int argc, char ** argv) {
    // OP={norm,direct,dequant}, TYPE={f32,q4_k,q6_k}, ROWS TOKENS WEIGHTS INPUT OUTPUT.
    if (argc != 8) return 2;
    const std::string op = argv[1], name = argv[2];
    const int width = 2560, rows = std::stoi(argv[3]), tokens = std::stoi(argv[4]);
    if ((op != "norm" && op != "direct" && op != "dequant") || rows < 1 || rows > 8192 || tokens < 1 || tokens > 16)
        return 2;
    const ggml_type type = name == "f32" ? GGML_TYPE_F32 : name == "q4_k" ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    if (name != "f32" && name != "q4_k" && name != "q6_k") return 2;
    if (op == "norm" && (rows != 1 || type != GGML_TYPE_F32)) return 2;
    // Maximum temporary F32 projection matrix is 80MiB, never full model/head.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 128ull * 1024 * 1024, nullptr, false }),
                                                            ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!ctx || !backend) return 1;
    ggml_backend_cpu_set_n_threads(backend.get(), 4);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    auto * weight = ggml_new_tensor_2d(ctx.get(), type, width, rows);
    if (!read(argv[5], weight->data, ggml_nbytes(weight)) || !read(argv[6], input->data, ggml_nbytes(input))) return 1;
    const auto * x = static_cast<const float *>(input->data);
    for (size_t i = 0; i < size_t(width * tokens); ++i)
        if (!std::isfinite(x[i])) return 1;
    ggml_tensor * result;
    if (op == "norm")
        result = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), input, 1e-6f), weight);
    else {
        if (op == "dequant" && type != GGML_TYPE_F32) {
            auto * f32 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
            const auto * traits = ggml_get_type_traits(type);
            if (!traits || !traits->to_float) return 1;
            traits->to_float(weight->data, static_cast<float *>(f32->data), int64_t(width) * rows);
            weight = f32;
        }
        result = ggml_mul_mat(ctx.get(), weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    }
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) return 1;
    const auto * values = static_cast<const float *>(result->data);
    for (int64_t i = 0; i < ggml_nelements(result); ++i)
        if (!std::isfinite(values[i])) return 1;
    std::ofstream out(argv[7], std::ios::binary);
    if (!out.write(reinterpret_cast<const char *>(result->data), ggml_nbytes(result))) return 1;
    std::printf("PHYSICAL_PRIMITIVE pass=1 op=%s type=%s width=2560 rows=%d tokens=%d threads=4\n", op.c_str(),
                name.c_str(), rows, tokens);
    return 0;
}
int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "primitive: %s\n", e.what());
        return 1;
    }
}
