// test_conv_ext.cpp — test ggml_conv_2d with external (pre-allocated) weight tensor
// This tests whether conv_2d works when the kernel is in a separate buffer.

#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <layout.gguf>\n", argv[0]);
        return 1;
    }

    // Load weights (same way as layout_detect)
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(argv[1], backend, "test", wl)) {
        fprintf(stderr, "Failed to load weights\n");
        return 1;
    }

    // Find stem weight
    auto it = wl.tensors.find("model.backbone.conv1.conv1_1.conv.weight");
    if (it == wl.tensors.end()) {
        fprintf(stderr, "Stem weight not found\n");
        return 1;
    }
    ggml_tensor* ext_w = it->second;
    printf("External weight: [%lld,%lld] type=%d\n",
           (long long)ext_w->ne[0], (long long)ext_w->ne[1], ext_w->type);
    float wv[3];
    ggml_backend_tensor_get(ext_w, wv, 0, 12);
    printf("  first3: %.6f %.6f %.6f\n", wv[0], wv[1], wv[2]);

    // Build a tiny graph: reshape ext_w to 4D, cast to F16, conv with 8x8 input
    size_t buf_size = ggml_tensor_overhead() * 20 + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    ggml_init_params p = { buf_size, buf.data(), true };
    ggml_context* g = ggml_init(p);

    // Input
    ggml_tensor* x = ggml_new_tensor_3d(g, GGML_TYPE_F32, 8, 8, 3);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    // Reshape external weight to 4D
    ggml_tensor* w4 = ggml_reshape_4d(g, ext_w, 3, 3, 3, 32);
    ggml_tensor* w16 = ggml_cast(g, w4, GGML_TYPE_F16);

    // Conv
    ggml_tensor* y = ggml_conv_2d(g, w16, x, 1, 1, 1, 1, 1, 1);
    ggml_set_name(y, "y");
    ggml_set_output(y);

    // Build graph
    ggml_cgraph* gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, y);
    printf("Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    // Allocate
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, gf)) {
        fprintf(stderr, "Alloc failed\n");
        return 1;
    }

    // Set input to ones
    std::vector<float> input_data(8*8*3, 1.0f);
    ggml_tensor* xt = ggml_graph_get_tensor(gf, "x");
    if (xt) ggml_backend_tensor_set(xt, input_data.data(), 0, input_data.size() * 4);
    else printf("x not found!\n");

    // Compute
    ggml_backend_graph_compute(backend, gf);

    // Read output
    ggml_tensor* yt = ggml_graph_get_tensor(gf, "y");
    if (yt) {
        float ov[5];
        ggml_backend_tensor_get(yt, ov, 0, 5 * sizeof(float));
        printf("Output [%lld,%lld,%lld]: %.4f %.4f %.4f %.4f %.4f\n",
               (long long)yt->ne[0], (long long)yt->ne[1], (long long)yt->ne[2],
               ov[0], ov[1], ov[2], ov[3], ov[4]);
    } else {
        // Try direct pointer
        float ov[5];
        ggml_backend_tensor_get(y, ov, 0, 5 * sizeof(float));
        printf("Output (direct) [%lld,%lld,%lld]: %.4f %.4f %.4f %.4f %.4f\n",
               (long long)y->ne[0], (long long)y->ne[1], (long long)y->ne[2],
               ov[0], ov[1], ov[2], ov[3], ov[4]);
    }

    ggml_gallocr_free(alloc);
    ggml_free(g);
    core_gguf::free_weights(wl);
    ggml_backend_free(backend);
    return 0;
}
