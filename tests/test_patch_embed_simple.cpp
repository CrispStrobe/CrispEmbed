// Minimal test: load patch_embed weight from GGUF, load patches from
// reference GGUF, do matmul, compare first 5 values.

#include "crispembed_diff.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <ref.gguf>\n", argv[0]);
        return 1;
    }

    // Load reference patches
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;

    auto [patches_data, patches_n] = ref.get_f32("input_patches");
    auto [pe_ref, pe_ref_n] = ref.get_f32("vis_patch_embed");
    auto patches_shape = ref.shape("input_patches");  // [1176, 1564]
    int patch_dim = (int)patches_shape[0];  // 1176
    int n_patches = (int)patches_shape[1];  // 1564

    printf("Patches: %d x %d\n", n_patches, patch_dim);
    printf("  first 5 values: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           patches_data[0], patches_data[1], patches_data[2],
           patches_data[3], patches_data[4]);

    // Load model weight
    ggml_context* mctx = nullptr;
    gguf_init_params gp = { .no_alloc = false, .ctx = &mctx };
    gguf_context* gctx = gguf_init_from_file(argv[1], gp);

    int H = 1280;

    // Find patch_embed weight
    ggml_tensor* W = nullptr;
    for (ggml_tensor* t = ggml_get_first_tensor(mctx); t; t = ggml_get_next_tensor(mctx, t)) {
        if (strcmp(ggml_get_name(t), "v.patch_embed.weight") == 0) {
            W = t;
            break;
        }
    }
    if (!W) {
        fprintf(stderr, "v.patch_embed.weight not found\n");
        return 1;
    }
    printf("Weight: ne=[%lld, %lld], type=%d\n",
           (long long)W->ne[0], (long long)W->ne[1], W->type);

    // Print first 5 weight values
    const float* w_data = (const float*)W->data;
    printf("  W first 5: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
           w_data[0], w_data[1], w_data[2], w_data[3], w_data[4]);

    // Manual matmul: result[i,j] = sum_k W[k,i] * X[k,j]
    // where W ne=(1176,1280), X ne=(1176,1564)
    // Result ne=(1280, 1564)
    // W[k,i] = w_data[i * 1176 + k] (column-major: ne[0] is contiguous)
    // X[k,j] = patches_data[j * 1176 + k]
    // result[i,j] = result_data[j * 1280 + i]

    printf("\nManual matmul (first 5 dims of patch 0):\n");
    for (int i = 0; i < 5; i++) {
        double sum = 0;
        for (int k = 0; k < patch_dim; k++) {
            sum += (double)w_data[i * patch_dim + k] * (double)patches_data[k];
        }
        printf("  dim %d: %.6f (ref: %.6f)\n", i, (float)sum, pe_ref[i]);
    }

    // Now try ggml graph
    printf("\nggml graph matmul:\n");
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 4);

    size_t ctx_size = ggml_tensor_overhead() * 10 + ggml_graph_overhead();
    ggml_init_params ip = { ctx_size, nullptr, true };
    ggml_context* cctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph(cctx);

    ggml_tensor* X = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, patch_dim, n_patches);
    ggml_set_name(X, "X");
    ggml_set_input(X);

    ggml_tensor* result = ggml_mul_mat(cctx, W, X);
    ggml_set_name(result, "result");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_tensor* X_graph = ggml_graph_get_tensor(gf, "X");
    ggml_backend_tensor_set(X_graph, patches_data, 0,
                            (size_t)n_patches * patch_dim * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    ggml_tensor* res = ggml_graph_get_tensor(gf, "result");
    std::vector<float> res_data(H * n_patches);
    ggml_backend_tensor_get(res, res_data.data(), 0, res_data.size() * sizeof(float));

    printf("  ggml first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           res_data[0], res_data[1], res_data[2], res_data[3], res_data[4]);
    printf("  ref  first 5: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           pe_ref[0], pe_ref[1], pe_ref[2], pe_ref[3], pe_ref[4]);

    ggml_gallocr_free(alloc);
    ggml_free(cctx);
    ggml_backend_free(backend);
    gguf_free(gctx);

    return 0;
}
