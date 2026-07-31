// Verify that the selected ggml backend can allocate and execute a graph.
// This intentionally tests device execution, not merely compile-time flags.

#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <vector>

int main() {
    ggml_backend_t backend = crispasr_init_gpu_backend();
    if (!backend) {
        std::fprintf(stderr, "backend smoke: no backend available\n");
        return 1;
    }
    const auto device = ggml_backend_get_device(backend);
    const char * name = device ? ggml_backend_dev_name(device) : ggml_backend_name(backend);
    const auto type = device ? ggml_backend_dev_type(device) : GGML_BACKEND_DEVICE_TYPE_CPU;

    std::vector<uint8_t> arena(ggml_tensor_overhead() * 8 + ggml_graph_overhead());
    ggml_init_params params{ arena.size(), arena.data(), true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 1;
    }
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    const bool allocated = alloc && ggml_gallocr_alloc_graph(alloc, graph);
    std::vector<float> av(16, 1.25f), bv(16, 2.75f), ov(16, 0.0f);
    if (allocated) {
        ggml_backend_tensor_set(ggml_graph_get_tensor(graph, a->name), av.data(), 0, av.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(graph, b->name), bv.data(), 0, bv.size() * sizeof(float));
    }
    const bool computed = allocated && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (computed)
        ggml_backend_tensor_get(ggml_graph_get_tensor(graph, out->name), ov.data(), 0, ov.size() * sizeof(float));
    bool correct = computed;
    for (float v : ov) correct = correct && v > 3.99f && v < 4.01f;
    std::printf("backend-smoke name=%s type=%d nodes=%d computed=%d correct=%d\n", name ? name : "unknown", (int)type,
                ggml_graph_n_nodes(graph), computed ? 1 : 0, correct ? 1 : 0);
    if (alloc) ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    core_util::clean_exit(correct ? 0 : 2);
    return correct ? 0 : 2;
}
