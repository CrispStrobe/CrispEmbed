#include "easyocr_ocr.h"

#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "image_preprocess.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

struct easyocr_ocr_context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * logits = nullptr;
    int width = 200;
    int height = 64;
    int classes = 0;
    int hidden = 256;
    std::vector<std::string> tokens;
    std::vector<float> input_host;
    std::string result;
};

static ggml_tensor * req(easyocr_ocr_context * c, const char * name) {
    return core_gguf::require(c->wl.tensors, name, "easyocr");
}

static ggml_tensor * linear(ggml_context * g, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(g, ggml_mul_mat(g, w, x), b);
}

static ggml_tensor * lstm_direction(ggml_context * g, ggml_tensor * seq, int T, int in_dim, int hidden,
                                    ggml_tensor * wih, ggml_tensor * whh, ggml_tensor * bih, ggml_tensor * bhh,
                                    bool reverse) {
    std::vector<ggml_tensor *> out((size_t)T);
    ggml_tensor * h = ggml_new_tensor_1d(g, GGML_TYPE_F32, hidden);
    ggml_tensor * c = ggml_new_tensor_1d(g, GGML_TYPE_F32, hidden);
    ggml_set_name(h, reverse ? "lstm_rev_h0" : "lstm_fwd_h0");
    ggml_set_name(c, reverse ? "lstm_rev_c0" : "lstm_fwd_c0");
    ggml_set_input(h);
    ggml_set_input(c);

    for (int step = 0; step < T; ++step) {
        const int t = reverse ? T - 1 - step : step;
        ggml_tensor * x = ggml_view_1d(g, seq, in_dim, (size_t)t * in_dim * sizeof(float));
        ggml_tensor * gates =
            ggml_add(g, ggml_add(g, ggml_mul_mat(g, wih, x), bih), ggml_add(g, ggml_mul_mat(g, whh, h), bhh));
        ggml_tensor * gi = ggml_sigmoid(g, ggml_cont(g, ggml_view_1d(g, gates, hidden, 0)));
        ggml_tensor * gf =
            ggml_sigmoid(g, ggml_cont(g, ggml_view_1d(g, gates, hidden, (size_t)hidden * sizeof(float))));
        ggml_tensor * gg =
            ggml_tanh(g, ggml_cont(g, ggml_view_1d(g, gates, hidden, (size_t)2 * hidden * sizeof(float))));
        ggml_tensor * go =
            ggml_sigmoid(g, ggml_cont(g, ggml_view_1d(g, gates, hidden, (size_t)3 * hidden * sizeof(float))));
        c = ggml_add(g, ggml_mul(g, gf, c), ggml_mul(g, gi, gg));
        h = ggml_mul(g, go, ggml_tanh(g, c));
        if (ggml_nelements(h) != hidden) {
            fprintf(stderr, "easyocr: LSTM hidden shape at t=%d is %lld, expected %d\n", t,
                    (long long)ggml_nelements(h), hidden);
            return nullptr;
        }
        out[(size_t)t] = ggml_reshape_2d(g, h, hidden, 1);
    }

    ggml_tensor * result = out[0];
    for (int t = 1; t < T; ++t) result = ggml_concat(g, result, out[(size_t)t], 1);
    return result;
}

static bool build_graph(easyocr_ocr_context * c) {
    ggml_init_params ip = { 64u * 1024u * 1024u, nullptr, true };
    c->graph_ctx = ggml_init(ip);
    if (!c->graph_ctx) return false;
    ggml_context * g = c->graph_ctx;

    c->input = ggml_new_tensor_4d(g, GGML_TYPE_F32, c->width, c->height, 1, 1);
    ggml_set_name(c->input, "input_image");
    ggml_set_input(c->input);
    ggml_tensor * x = c->input;

    auto conv = [&](int i, int sw, int sh, int pw, int ph) {
        char n[96];
        snprintf(n, sizeof(n), "FeatureExtraction.ConvNet.%d.weight", i);
        ggml_tensor * w = req(c, n);
        snprintf(n, sizeof(n), "FeatureExtraction.ConvNet.%d.bias", i);
        ggml_tensor * b = req(c, n);
        x = ggml_conv_2d(g, w, x, sw, sh, pw, ph, 1, 1);
        if (b) x = ggml_add(g, x, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1));
        x = ggml_relu(g, x);
    };
    conv(0, 1, 1, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    conv(3, 1, 1, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    conv(6, 1, 1, 1, 1);
    conv(8, 1, 1, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 1, 2, 1, 2, 0, 0);
    conv(11, 1, 1, 1, 1);
    conv(14, 1, 1, 1, 1);
    x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 1, 2, 1, 2, 0, 0);
    conv(18, 1, 1, 0, 0);
    ggml_set_name(x, "features");
    ggml_set_output(x);

    const int T = 49;
    const int D = 256;
    // EasyOCR applies AdaptiveAvgPool2d((None, 1)) after permuting
    // [B,C,H,W] -> [B,W,C,H]. In ggml layout this is [W,H,C,B] ->
    // [H,C,W,B], average over the H axis, then restore [C,W].
    x = ggml_cont(g, ggml_permute(g, x, 2, 0, 1, 3));
    x = ggml_pool_2d(g, x, GGML_OP_POOL_AVG, x->ne[0], 1, x->ne[0], 1, 0, 0);
    x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
    if (ggml_nelements(x) != D * T) {
        fprintf(stderr, "easyocr: sequence shape is %lld, expected %d\n", (long long)ggml_nelements(x), D * T);
        return false;
    }
    x = ggml_reshape_2d(g, x, D, T);
    ggml_set_name(x, "sequence_input");
    ggml_set_output(x);

    auto rn = [&](int layer, const char * suffix) {
        char n[128];
        snprintf(n, sizeof(n), "SequenceModeling.%d.rnn.%s", layer, suffix);
        return req(c, n);
    };
    auto ln = [&](int layer, const char * suffix) {
        char n[128];
        snprintf(n, sizeof(n), "SequenceModeling.%d.linear.%s", layer, suffix);
        return req(c, n);
    };
    ggml_tensor * f0 = lstm_direction(g, x, T, D, c->hidden, rn(0, "weight_ih_l0"), rn(0, "weight_hh_l0"),
                                      rn(0, "bias_ih_l0"), rn(0, "bias_hh_l0"), false);
    ggml_tensor * r0 =
        lstm_direction(g, x, T, D, c->hidden, rn(0, "weight_ih_l0_reverse"), rn(0, "weight_hh_l0_reverse"),
                       rn(0, "bias_ih_l0_reverse"), rn(0, "bias_hh_l0_reverse"), true);
    x = linear(g, ln(0, "weight"), ln(0, "bias"), ggml_concat(g, f0, r0, 0));
    ggml_set_name(x, "bilstm_0");
    ggml_set_output(x);
    ggml_tensor * f1 = lstm_direction(g, x, T, D, c->hidden, rn(1, "weight_ih_l0"), rn(1, "weight_hh_l0"),
                                      rn(1, "bias_ih_l0"), rn(1, "bias_hh_l0"), false);
    ggml_tensor * r1 =
        lstm_direction(g, x, T, D, c->hidden, rn(1, "weight_ih_l0_reverse"), rn(1, "weight_hh_l0_reverse"),
                       rn(1, "bias_ih_l0_reverse"), rn(1, "bias_hh_l0_reverse"), true);
    x = linear(g, ln(1, "weight"), ln(1, "bias"), ggml_concat(g, f1, r1, 0));
    ggml_set_name(x, "bilstm_1");
    ggml_set_output(x);
    c->logits = linear(g, req(c, "Prediction.weight"), req(c, "Prediction.bias"), x);
    ggml_set_name(c->logits, "logits");
    ggml_set_output(c->logits);

    c->graph = ggml_new_graph_custom(g, 16384, false);
    ggml_build_forward_expand(c->graph, c->logits);
    c->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    return c->alloc && ggml_gallocr_alloc_graph(c->alloc, c->graph);
}

easyocr_ocr_context * easyocr_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    auto * c = new easyocr_ocr_context();
    c->backend = crispasr_init_gpu_backend();
    if (!c->backend || !core_gguf::load_weights(model_path, c->backend, "easyocr", c->wl)) {
        easyocr_ocr_free(c);
        return nullptr;
    }
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (meta) {
        c->width = (int)core_gguf::kv_u32(meta, "easyocr.input_width", 200);
        c->height = (int)core_gguf::kv_u32(meta, "easyocr.input_height", 64);
        c->classes = (int)core_gguf::kv_u32(meta, "easyocr.num_classes", 0);
        c->tokens = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        core_gguf::free_metadata(meta);
    }
    if (!build_graph(c)) {
        easyocr_ocr_free(c);
        return nullptr;
    }
    return c;
}

void easyocr_ocr_free(easyocr_ocr_context * c) {
    if (!c) return;
    if (c->alloc) ggml_gallocr_free(c->alloc);
    if (c->graph_ctx) ggml_free(c->graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

const char * easyocr_ocr_recognize(easyocr_ocr_context * c, const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!c || !px || w <= 0 || h <= 0 || ch <= 0) return nullptr;
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t * p = px + ((size_t)y * w + x) * ch;
            uint8_t * q = rgb.data() + ((size_t)y * w + x) * 3;
            q[0] = q[1] = q[2] = ch == 1 ? p[0] : (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
        }
    std::vector<float> resized((size_t)c->height * c->width * 3);
    const int rw = std::min(c->width, std::max(1, (int)std::ceil((double)c->height * w / h)));
    image_preproc::resize_bicubic_u8_hwc(rgb.data(), h, w, resized.data(), c->height, rw, 3);
    c->input_host.assign((size_t)c->height * c->width, 0.0f);
    for (int y = 0; y < c->height; ++y)
        for (int x = 0; x < rw; ++x) {
            float v = resized[((size_t)y * rw + x) * 3] / 255.0f;
            c->input_host[(size_t)y * c->width + x] = v * 2.0f - 1.0f;
        }
    for (int y = 0; y < c->height; ++y)
        for (int x = rw; x < c->width; ++x)
            c->input_host[(size_t)y * c->width + x] = c->input_host[(size_t)y * c->width + rw - 1];
    ggml_backend_tensor_set(c->input, c->input_host.data(), 0, c->input_host.size() * sizeof(float));
    std::vector<float> zero((size_t)c->hidden, 0.0f);
    for (const char * n : { "lstm_fwd_h0", "lstm_fwd_c0", "lstm_rev_h0", "lstm_rev_c0" })
        ggml_backend_tensor_set(ggml_graph_get_tensor(c->graph, n), zero.data(), 0, zero.size() * sizeof(float));
    if (ggml_backend_graph_compute(c->backend, c->graph) != GGML_STATUS_SUCCESS) return nullptr;

    std::vector<float> logits((size_t)c->classes * 49);
    ggml_backend_tensor_get(c->logits, logits.data(), 0, logits.size() * sizeof(float));
    c->result.clear();
    int prev = 0;
    for (int t = 0; t < 49; ++t) {
        int best = 0;
        for (int k = 1; k < c->classes; ++k)
            if (logits[(size_t)t * c->classes + k] > logits[(size_t)t * c->classes + best]) best = k;
        if (best && best != prev && best < (int)c->tokens.size()) c->result += c->tokens[(size_t)best];
        prev = best;
    }
    if (out_len) *out_len = (int)c->result.size();
    return c->result.c_str();
}
