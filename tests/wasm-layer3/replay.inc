// TEST OVERLAY ONLY. No production export or new model/API path.
extern "C" CRISPEMBED_API int crispembed_layer3_replay(crispembed_context * ctx, const char * name,
        const float * values, int count, float * output, int capacity) {
    if (!ctx || !name || !values || !output || ctx->model.hparams.n_layer != 6) return -1;
    const auto & layer = ctx->model.layers[3];
    ggml_tensor * weight = nullptr;
    if (!strcmp(name, "blk.3.attn_q.weight")) weight = layer.q_w;
    else if (!strcmp(name, "blk.3.attn_k.weight")) weight = layer.k_w;
    else if (!strcmp(name, "blk.3.attn_v.weight")) weight = layer.v_w;
    else if (!strcmp(name, "blk.3.attn_output.weight")) weight = layer.o_w;
    else if (!strcmp(name, "blk.3.ffn_up.weight")) weight = layer.fc1_w;
    else if (!strcmp(name, "blk.3.ffn_down.weight")) weight = layer.fc2_w;
    if (!weight || strcmp(ggml_get_name(weight), name) || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            count != weight->ne[0]*12 || capacity < weight->ne[1]*12) return -2;
    for (int i = 0; i < count; ++i) if (!std::isfinite(values[i])) return -3;
    std::vector<uint8_t> metadata(4*1024*1024);
    ggml_init_params parameters = {metadata.size(), metadata.data(), true};
    ggml_context * graph_context = ggml_init(parameters);
    if (!graph_context) return -4;
    ggml_cgraph * graph = ggml_new_graph_custom(graph_context, 128, false);
    ggml_tensor * input = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, weight->ne[0], 12);
    ggml_set_input(input);
    ggml_tensor * result = ggml_mul_mat(graph_context, weight, input);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    ggml_backend_sched_reset(ctx->sched);
    int rc = -5;
    if (ggml_backend_sched_alloc_graph(ctx->sched, graph)) {
        ggml_backend_tensor_set(input, values, 0, count*sizeof(float));
        if (sched_graph_compute(ctx->sched, graph, ctx->n_threads)) {
            rc = (int) ggml_nelements(result);
            ggml_backend_tensor_get(result, output, 0, rc*sizeof(float));
        }
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_free(graph_context);
    return rc;
}
