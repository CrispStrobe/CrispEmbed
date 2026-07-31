#include "easyocr_craft.h"

#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct easyocr_craft_context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * feature = nullptr;
    ggml_tensor * scores = nullptr;
    ggml_tensor * basenet[5] = {};
    int width = 0;
    int height = 0;
    std::vector<float> input_host;
};

static ggml_tensor * req(easyocr_craft_context * c, const std::string & name) {
    return core_gguf::require(c->wl.tensors, name.c_str(), "easyocr-craft");
}

static ggml_tensor * conv(ggml_context * g, easyocr_craft_context * c, ggml_tensor * x, const std::string & name,
                          int kw, int kh, int pw, int ph, int dw = 1, int dh = 1, bool relu = true) {
    ggml_tensor * y = ggml_conv_2d(g, req(c, name + ".weight"), x, 1, 1, pw, ph, dw, dh);
    y = ggml_add(g, y, ggml_reshape_4d(g, req(c, name + ".bias"), 1, 1, req(c, name + ".bias")->ne[0], 1));
    return relu ? ggml_relu(g, y) : y;
}

static ggml_tensor * double_conv(ggml_context * g, easyocr_craft_context * c, ggml_tensor * x,
                                 const std::string & name) {
    x = conv(g, c, x, name + ".conv.0", 1, 1, 0, 0);
    return conv(g, c, x, name + ".conv.3", 3, 3, 1, 1);
}

static bool build_graph(easyocr_craft_context * c) {
    ggml_init_params ip = { 96u * 1024u * 1024u, nullptr, true };
    c->graph_ctx = ggml_init(ip);
    if (!c->graph_ctx) return false;
    ggml_context * g = c->graph_ctx;
    c->input = ggml_new_tensor_4d(g, GGML_TYPE_F32, c->width, c->height, 3, 1);
    ggml_set_name(c->input, "input_image");
    ggml_set_input(c->input);
    ggml_tensor * x = c->input;
    auto pool = [&](int k, int s, int p) { x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, k, k, s, s, p, p); };

    // VGG-16 BN feature taps used by EasyOCR's CRAFT implementation.
    x = ggml_conv_2d(g, req(c, "basenet.slice1.0.weight"), x, 1, 1, 1, 1, 1, 1);
    x = ggml_add(g, x, ggml_reshape_4d(g, req(c, "basenet.slice1.0.bias"), 1, 1, 64, 1));
    x = ggml_relu(g, x);
    x = conv(g, c, x, "basenet.slice1.3", 3, 3, 1, 1);
    pool(2, 2, 0);
    x = conv(g, c, x, "basenet.slice1.7", 3, 3, 1, 1);
    x = conv(g, c, x, "basenet.slice1.10", 3, 3, 1, 1, 1, 1, false);
    x = ggml_relu(g, x);
    ggml_tensor * source4 = x;
    c->basenet[4] = source4;
    ggml_set_name(source4, "basenet_4");
    ggml_set_output(source4);

    pool(2, 2, 0);
    x = conv(g, c, x, "basenet.slice2.14", 3, 3, 1, 1);
    x = conv(g, c, x, "basenet.slice2.17", 3, 3, 1, 1, 1, 1, false);
    x = ggml_relu(g, x);
    ggml_tensor * source3 = x;
    c->basenet[3] = source3;
    ggml_set_name(source3, "basenet_3");
    ggml_set_output(source3);

    x = conv(g, c, x, "basenet.slice3.20", 3, 3, 1, 1);
    x = ggml_relu(g, x);
    pool(2, 2, 0);
    x = conv(g, c, x, "basenet.slice3.24", 3, 3, 1, 1);
    x = conv(g, c, x, "basenet.slice3.27", 3, 3, 1, 1, 1, 1, false);
    x = ggml_relu(g, x);
    ggml_tensor * source2 = x;
    c->basenet[2] = source2;
    ggml_set_name(source2, "basenet_2");
    ggml_set_output(source2);

    x = ggml_relu(g, x);
    x = conv(g, c, x, "basenet.slice4.30", 3, 3, 1, 1);
    pool(2, 2, 0);
    x = conv(g, c, x, "basenet.slice4.34", 3, 3, 1, 1);
    x = conv(g, c, x, "basenet.slice4.37", 3, 3, 1, 1, 1, 1, false);
    ggml_tensor * source1 = x;
    c->basenet[1] = source1;
    ggml_set_name(source1, "basenet_1");
    ggml_set_output(source1);

    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 3, 3, 1, 1, 1, 1);
    x = conv(g, c, x, "basenet.slice5.1", 3, 3, 6, 6, 6, 6, false);
    x = conv(g, c, x, "basenet.slice5.2", 1, 1, 0, 0, 1, 1, false);
    ggml_tensor * source0 = x;
    c->basenet[0] = source0;
    ggml_set_name(source0, "basenet_0");
    ggml_set_output(source0);

    x = ggml_concat(g, source0, source1, 2);
    x = double_conv(g, c, x, "upconv1");
    x = ggml_interpolate(g, x, source2->ne[0], source2->ne[1], x->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    x = ggml_concat(g, x, source2, 2);
    x = double_conv(g, c, x, "upconv2");
    x = ggml_interpolate(g, x, source3->ne[0], source3->ne[1], x->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    x = ggml_concat(g, x, source3, 2);
    x = double_conv(g, c, x, "upconv3");
    x = ggml_interpolate(g, x, source4->ne[0], source4->ne[1], x->ne[2], 1, GGML_SCALE_MODE_BILINEAR);
    x = ggml_concat(g, x, source4, 2);
    c->feature = double_conv(g, c, x, "upconv4");
    ggml_set_name(c->feature, "feature");
    ggml_set_output(c->feature);

    x = conv(g, c, c->feature, "conv_cls.0", 3, 3, 1, 1);
    x = conv(g, c, x, "conv_cls.2", 3, 3, 1, 1);
    x = conv(g, c, x, "conv_cls.4", 3, 3, 1, 1);
    x = conv(g, c, x, "conv_cls.6", 1, 1, 0, 0);
    c->scores = conv(g, c, x, "conv_cls.8", 1, 1, 0, 0, 1, 1, false);
    ggml_set_name(c->scores, "scores");
    ggml_set_output(c->scores);
    c->graph = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(c->graph, c->scores);
    c->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    return c->alloc && ggml_gallocr_alloc_graph(c->alloc, c->graph);
}

easyocr_craft_context * easyocr_craft_init(const char * model_path, int width, int height) {
    auto * c = new easyocr_craft_context();
    c->width = width;
    c->height = height;
    c->backend = std::getenv("EASYOCR_CRAFT_FORCE_CPU") ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!c->backend || !core_gguf::load_weights(model_path, c->backend, "easyocr-craft", c->wl) || !build_graph(c)) {
        easyocr_craft_free(c);
        return nullptr;
    }
    return c;
}

void easyocr_craft_free(easyocr_craft_context * c) {
    if (!c) return;
    if (c->alloc) ggml_gallocr_free(c->alloc);
    if (c->graph_ctx) ggml_free(c->graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

bool easyocr_craft_forward(easyocr_craft_context * c, const float * input, size_t n_elem) {
    if (!c || n_elem != (size_t)c->width * c->height * 3) return false;
    c->input_host.assign(input, input + n_elem);
    ggml_backend_tensor_set(c->input, input, 0, n_elem * sizeof(float));
    return ggml_backend_graph_compute(c->backend, c->graph) == GGML_STATUS_SUCCESS;
}

int easyocr_craft_diff(easyocr_craft_context * c, const char * path) {
    crispembed_diff::Ref ref;
    if (!c || !ref.load(path)) return 1;
    int failures = 0;
    for (const char * name :
         { "input_image", "basenet_0", "basenet_1", "basenet_2", "basenet_3", "basenet_4", "feature", "scores" }) {
        auto rr = ref.get_f32(name);
        if (!rr.first) continue;
        ggml_tensor * t = !strcmp(name, "feature") ? c->feature : !strcmp(name, "scores") ? c->scores : c->input;
        if (!strncmp(name, "basenet_", 8)) t = c->basenet[name[8] - '0'];
        std::vector<float> data((size_t)ggml_nelements(t));
        if (!strcmp(name, "input_image")) {
            data = c->input_host;
        } else {
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
        }
        if (!strcmp(name, "scores")) {
            // EasyOCR returns [N,H,W,C] while GGML's score tensor is [W,H,C].
            // Reorder to the Python output's contiguous NHWC layout before
            // comparing the decoded detector boundary.
            const int64_t w = t->ne[0], h = t->ne[1], channels = t->ne[2];
            std::vector<float> nhwc(data.size());
            for (int64_t y = 0; y < h; ++y)
                for (int64_t x = 0; x < w; ++x)
                    for (int64_t k = 0; k < channels; ++k)
                        nhwc[(size_t)((y * w + x) * channels + k)] = data[(size_t)(x + w * y + w * h * k)];
            data.swap(nhwc);
        }
        auto report = ref.compare(name, data.data(), data.size(), 0);
        printf("easyocr-craft-diff %-12s n=%zu cos=%.7f global=%.7f mine=%.6g ref=%.6g %s\n", name, report.n_elem,
               report.cos_min, report.cos_global, report.mine_norm, report.ref_norm,
               report.cos_global >= 0.99f ? "PASS" : "FAIL");
        if (report.cos_global < 0.99f) failures++;
    }
    return failures;
}
