// Weight-free CPU graph contracts: contiguous VTE projection, F32 head72
// attention, dynamic KV writes, graph reuse, a256-slot boundary and a shorter
// subsequent request. This test must execute in the hosted build gate.
#include "core/clean_exit.h"
#include "oxr_graph_ops.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int crispembed_test_main() {
    auto * ctx = ggml_init({ 32 * 1024 * 1024, nullptr, false });
    auto backend = ggml_backend_cpu_init();
    if (!ctx || !backend) return 1;
    ggml_backend_cpu_set_n_threads(backend, 2);
    int checks = 0, failed = 0;
    const auto check = [&](bool condition, const char * name) {
        ++checks;
        if (!condition) {
            ++failed;
            std::fprintf(stderr, "FAIL %s\n", name);
        }
    };
    auto * gelu_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    const float gelu_values[] = { -8.0f, -3.12345f, -1.0003f, -0.10007f, 0.10007f, 1.0003f, 3.12345f, 8.0f };
    std::memcpy(gelu_input->data, gelu_values, sizeof(gelu_values));
    auto * gelu = oxr_graph_ops::gelu_tanh_f32(ctx, gelu_input);
    auto * gelu_graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(gelu_graph, gelu);
    check(ggml_backend_graph_compute(backend, gelu_graph) == GGML_STATUS_SUCCESS, "F32 tanh GELU graph executes");
    bool gelu_equal = true;
    for (int i = 0; i < 8; ++i) {
        const float x = gelu_values[i];
        const float expected = .5f * x * (1.0f + std::tanh(.7978845608028654f * (x + .044715f * x * x * x)));
        gelu_equal = gelu_equal && std::fabs(static_cast<float *>(gelu->data)[i] - expected) <= 2e-6f;
    }
    check(gelu_equal, "GELU avoids hidden F16 lookup rounding");
    auto * vte = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 8);
    auto * transposed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 3);
    auto * raw = static_cast<float *>(vte->data);
    for (int i = 0; i < 24; ++i) raw[i] = float(i - 7) / 16.0f;
    auto * copy_graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(copy_graph, ggml_cpy(ctx, ggml_transpose(ctx, vte), transposed));
    check(ggml_backend_graph_compute(backend, copy_graph) == GGML_STATUS_SUCCESS, "persistent VTE transpose executes");
    check(transposed->nb[0] == sizeof(float), "transposed VTE has legal contiguous element stride");
    auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    const float logit_values[] = { 1.0f, 0.2f, -0.7f, 0.0f, -0.5f, 0.8f, 0.1f, 0.3f };
    std::memcpy(logits->data, logit_values, sizeof(logit_values));
    auto * probs = ggml_pad(ctx, ggml_soft_max(ctx, logits), 4, 0, 0, 0);
    auto * output = ggml_mul_mat(ctx, transposed, probs);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    auto * project_graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(project_graph, output);
    check(ggml_backend_graph_compute(backend, project_graph) == GGML_STATUS_SUCCESS,
          "padded visual softmax and projection execute");
    bool projected = true;
    for (int row = 0; row < 2; ++row) {
        double sum = 0.0;
        for (int v = 0; v < 4; ++v) sum += std::exp(double(logit_values[row * 4 + v]));
        for (int d = 0; d < 3; ++d) {
            double want = 0.0;
            for (int v = 0; v < 4; ++v) want += std::exp(double(logit_values[row * 4 + v])) / sum * raw[v * 3 + d];
            projected = projected && std::fabs(static_cast<float *>(output->data)[row * 3 + d] - want) < 1e-6;
        }
    }
    check(projected, "visual projection matches independent scalar softmax and weighted embedding sum");

    constexpr int hd = 72, nh = 4, nkv = 2, capacity = 512;
    auto * cache_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, capacity, nkv);
    auto * cache_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, capacity, nkv);
    auto kval = [](int row, int head, int d) { return float(((row * 3 + head * 7 + d) % 31) - 15) / 53.0f; };
    auto vval = [](int row, int head, int d) { return float(((row * 5 + head * 3 + d * 2) % 43) - 21) / 37.0f; };
    for (int head = 0; head < nkv; ++head)
        for (int row = 0; row < capacity; ++row)
            for (int d = 0; d < hd; ++d) {
                const int index = (head * capacity + row) * hd + d;
                static_cast<float *>(cache_k->data)[index] = kval(row, head, d);
                static_cast<float *>(cache_v->data)[index] = vval(row, head, d);
            }
    ggml_cgraph * current = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr, *index = nullptr, *mask = nullptr, *answer = nullptr;
    int last_extent = 0;
    for (int past : { 254, 255, 256, 257, 5 }) {
        const int extent = ((past + 1 + 255) / 256) * 256;
        if (extent != last_extent) {
            q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, nh);
            k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, nkv);
            v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, 1, nkv);
            index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, extent, 1);
            auto * written_k = ggml_set_rows(ctx, cache_k, k, index);
            auto * written_v = ggml_set_rows(ctx, cache_v, v, index);
            auto * keys = ggml_view_3d(ctx, written_k, hd, extent, nkv, written_k->nb[1], written_k->nb[2], 0);
            auto * values = ggml_view_3d(ctx, written_v, hd, extent, nkv, written_v->nb[1], written_v->nb[2], 0);
            answer = ggml_flash_attn_ext(ctx, q, keys, values, mask, 1.0f / std::sqrt(float(hd)), 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(answer, GGML_PREC_F32);
            current = ggml_new_graph_custom(ctx, 256, false);
            ggml_build_forward_expand(current, answer);
            last_extent = extent;
        }
        for (int head = 0; head < nh; ++head)
            for (int d = 0; d < hd; ++d)
                static_cast<float *>(q->data)[head * hd + d] = float(((head * 3 + d) % 13) - 6) / 29.0f;
        for (int head = 0; head < nkv; ++head)
            for (int d = 0; d < hd; ++d) {
                static_cast<float *>(k->data)[head * hd + d] = kval(past, head, d);
                static_cast<float *>(v->data)[head * hd + d] = vval(past, head, d);
                // Poison the old destination to prove set_rows feeds attention.
                static_cast<float *>(cache_k->data)[(head * capacity + past) * hd + d] = 100.0f;
                static_cast<float *>(cache_v->data)[(head * capacity + past) * hd + d] = -100.0f;
            }
        *static_cast<int32_t *>(index->data) = past;
        for (int i = 0; i < extent; ++i)
            static_cast<ggml_fp16_t *>(mask->data)[i] = ggml_fp32_to_fp16(i <= past ? 0.0f : -INFINITY);
        check(ggml_backend_graph_compute(backend, current) == GGML_STATUS_SUCCESS,
              "reused F32 head72 GQA/cache graph executes");
        bool equal = true;
        for (int head = 0; head < nh; ++head) {
            const int kvhead = head / (nh / nkv);
            std::vector<double> scores(size_t(past + 1));
            for (int row = 0; row <= past; ++row) {
                double score = 0.0;
                for (int d = 0; d < hd; ++d)
                    score += static_cast<float *>(q->data)[head * hd + d] * double(kval(row, kvhead, d));
                scores[size_t(row)] = score / std::sqrt(double(hd));
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double total = 0.0;
            for (double & score : scores) total += score = std::exp(score - maximum);
            for (int d = 0; d < hd; ++d) {
                double want = 0.0;
                for (int row = 0; row <= past; ++row) want += scores[size_t(row)] / total * vval(row, kvhead, d);
                equal = equal && std::fabs(static_cast<float *>(answer->data)[head * hd + d] - want) < 2e-6;
            }
        }
        check(equal, "cache write, mask, GQA and output agree with independent scalar attention");
    }
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("OXR graph contracts: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
