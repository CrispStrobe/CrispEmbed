// Reference-only attention graph. Compile separately against each pinned ggml.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace fs = std::filesystem;
static void need(bool ok, const char * s) { if (!ok) throw std::runtime_error(s); }
static std::vector<float> read(const fs::path & p, size_t n) {
    std::vector<float> v(n); std::ifstream f(p, std::ios::binary);
    need(bool(f.read(reinterpret_cast<char *>(v.data()), n * 4)) && f.peek() == EOF, "input size/read");
    for (float x : v) need(std::isfinite(x), "nonfinite input"); return v;
}
static void dump(const fs::path & p, ggml_tensor * t) {
    need(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "dump layout");
    const auto * x = static_cast<const float *>(t->data);
    bool nonzero = false;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) { need(std::isfinite(x[i]), "nonfinite output"); nonzero |= x[i] != 0.f; }
    need(nonzero, "unmaterialized/zero observed output on fixed nonzero capture");
    std::ofstream f(p, std::ios::binary); need(bool(f.write(static_cast<const char *>(t->data), ggml_nbytes(t))), "output write");
}
int main(int argc, char ** argv) try {
    // INPUT_DIR OUTPUT_DIR {4,16} {f16,f32} {composed,observed}
    need(argc == 7, "usage: graph INPUT OUTPUT TOKENS CACHE MODE PAST");
    const fs::path input = argv[1], output = argv[2]; fs::create_directories(output);
    const int n = std::stoi(argv[3]), past = std::stoi(argv[6]), total = past + n; need((n == 1 || n == 4 || n == 16) && past >= 0 && total <= 16, "tokens/past");
    const std::string type = argv[4], mode = argv[5]; need(type == "f16" || type == "f32", "cache type");
    need(mode == "composed" || mode == "observed", "mode"); const bool observe = mode == "observed";
    const auto qt = read(input / "q.f32", 16 * 16 * 256), kt = read(input / "k.f32", 16 * 4 * 256), vt = read(input / "v.f32", 16 * 4 * 256), gates = read(input / "gate.f32", 16 * 16 * 256);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({128ull << 20, nullptr, false}), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    need(bool(ctx) && bool(backend), "backend init"); ggml_backend_cpu_set_n_threads(backend.get(), 4);
    auto * c = ctx.get(); const auto cachetype = type == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
    auto * qcur = ggml_new_tensor_3d(c, GGML_TYPE_F32, 256, 16, n);
    auto * kcur = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1024, total);
    auto * vcur = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, total * 1024);
    auto * gate = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4096, n);
    std::memcpy(qcur->data, qt.data() + past * 4096, ggml_nbytes(qcur)); std::memcpy(kcur->data, kt.data(), ggml_nbytes(kcur));
    std::memcpy(vcur->data, vt.data(), ggml_nbytes(vcur)); std::memcpy(gate->data, gates.data() + past * 4096, ggml_nbytes(gate));
    auto * kcache = ggml_new_tensor_2d(c, cachetype, 1024, 2048);
    auto * vcache = ggml_new_tensor_2d(c, cachetype, 2048, 1024);
    std::memset(kcache->data, 0, ggml_nbytes(kcache)); std::memset(vcache->data, 0, ggml_nbytes(vcache));
    auto * kidx = ggml_new_tensor_1d(c, GGML_TYPE_I64, total);
    auto * vidx = ggml_new_tensor_1d(c, GGML_TYPE_I64, total * 1024);
    for (int t = 0; t < total; ++t) static_cast<int64_t *>(kidx->data)[t] = t;
    for (int t = 0; t < total; ++t) for (int lane = 0; lane < 1024; ++lane)
        static_cast<int64_t *>(vidx->data)[t * 1024 + lane] = lane * 2048 + t;
    auto * kwrite = ggml_set_rows(c, kcache, kcur, kidx);
    auto * vflat = ggml_view_2d(c, vcache, 1, 2048 * 1024, ggml_type_size(cachetype), 0);
    auto * vwrite = ggml_set_rows(c, vflat, vcur, vidx);
    const size_t item = ggml_type_size(cachetype);
    auto * k = ggml_view_4d(c, kcache, 256, 4, 256, 1, 256 * item, 1024 * item, 1024 * 2048 * item, 0);
    auto * v = ggml_view_4d(c, vcache, 256, 4, 256, 1, 2048 * 256 * item, 2048 * item, 2048 * 1024 * item, 0);
    k = ggml_permute(c, k, 0, 2, 1, 3); v = ggml_permute(c, v, 0, 2, 1, 3);
    auto * q = ggml_permute(c, qcur, 0, 2, 1, 3);
    auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 32);
    for (int t = 0; t < 32; ++t) for (int s = 0; s < 256; ++s)
        static_cast<float *>(mask->data)[t * 256 + s] = s <= past + t && t < n ? 0.f : -INFINITY;
    auto * logits = ggml_mul_mat(c, k, q); ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    auto * probs = ggml_soft_max_ext(c, logits, mask, .0625f, 0.f);
    auto * pv = ggml_mul_mat(c, v, probs);
    auto * context = ggml_cont_2d(c, ggml_permute(c, pv, 0, 2, 1, 3), 4096, n);
    auto * sigmoid = ggml_sigmoid(c, gate); auto * gated = ggml_mul(c, context, sigmoid);
    auto * graph = ggml_new_graph_custom(c, 128, false);
    ggml_build_forward_expand(graph, kwrite); ggml_build_forward_expand(graph, vwrite);
    const std::vector<std::pair<const char *, ggml_tensor *>> stages = {{"logits", logits}, {"probs", probs}, {"context", context}, {"sigmoid", sigmoid}, {"gated", gated}};
    if (observe) for (const auto & stage : stages) { ggml_set_output(stage.second); ggml_build_forward_expand(graph, stage.second); }
    ggml_build_forward_expand(graph, gated);
    need(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph compute");
    if (observe) for (const auto & stage : stages) dump(output / (std::string(stage.first) + ".f32"), stage.second);
    else dump(output / "gated.f32", gated);
    std::cout << "ATTENTION_GRAPH tokens=" << n << " cache=" << type << " observed=" << observe << " PASS\n";
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
