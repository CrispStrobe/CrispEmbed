#include "easyocr_ocr.h"

#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "easyocr_postprocess.h"
#include "image_preprocess.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "core/env_gate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_diff_report(const char * name, const crispembed_diff::Report & r, bool pass) {
    printf("easyocr-diff %-16s n=%zu max=%.6g mean=%.6g rms=%.6g cos=%.7f global=%.7f mine=%.6g ref=%.6g %s\n", name,
           r.n_elem, r.max_abs, r.mean_abs, r.rms, r.cos_min, r.cos_global, r.mine_norm, r.ref_norm,
           pass ? "PASS" : "FAIL");
}

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
    int output_channels = 256;
    int time_steps = 0;
    int network = 0;
    std::vector<std::string> tokens;
    std::vector<float> input_host;
    std::string result;
    float last_confidence = 0.0f;
    easyocr_ocr_timing last_timing = {};
};

static ggml_tensor * req(easyocr_ocr_context * c, const char * name) {
    return core_gguf::require(c->wl.tensors, name, "easyocr");
}

static ggml_tensor * linear(ggml_context * g, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    return ggml_add(g, ggml_mul_mat(g, w, x), b);
}

// EasyOCR's get_image_list() calls cv2.resize with Image.Resampling.LANCZOS.
// That PIL enum has value 1, which OpenCV interprets as INTER_LINEAR. Keep the
// native path byte-compatible with that actual upstream pipeline rather than
// substituting the standalone reference dumper's bicubic resize. OpenCV's
// generic CV_8U INTER_LINEAR path uses 11-bit separable coefficients and an
// integer horizontal buffer; keeping those details here avoids accumulating a
// float interpolation residual before the recognizer sees the pixels.
static void resize_easyocr_linear_u8(const uint8_t * src, int src_h, int src_w, float * dst, int dst_h, int dst_w) {
    constexpr int coef_bits = 11;
    constexpr int coef_scale = 1 << coef_bits;
    std::vector<int> xofs((size_t)dst_w), yofs((size_t)dst_h);
    std::vector<int16_t> xalpha((size_t)dst_w * 2), ybeta((size_t)dst_h * 2);
    const float scale_x = (float)src_w / (float)dst_w;
    const float scale_y = (float)src_h / (float)dst_h;
    for (int x = 0; x < dst_w; ++x) {
        float f = ((float)x + 0.5f) * scale_x - 0.5f;
        int sx = (int)std::floor(f);
        f -= (float)sx;
        if (sx < 0) {
            f = 0.0f;
            sx = 0;
        }
        if (sx >= src_w - 1) {
            f = 0.0f;
            sx = src_w - 1;
        }
        xofs[(size_t)x] = sx;
        xalpha[(size_t)x * 2 + 0] = (int16_t)std::lround((1.0f - f) * coef_scale);
        xalpha[(size_t)x * 2 + 1] = (int16_t)std::lround(f * coef_scale);
    }
    for (int y = 0; y < dst_h; ++y) {
        float f = ((float)y + 0.5f) * scale_y - 0.5f;
        int sy = (int)std::floor(f);
        f -= (float)sy;
        if (sy < 0) {
            f = 0.0f;
            sy = 0;
        }
        if (sy >= src_h - 1) {
            f = 0.0f;
            sy = src_h - 1;
        }
        yofs[(size_t)y] = sy;
        ybeta[(size_t)y * 2 + 0] = (int16_t)std::lround((1.0f - f) * coef_scale);
        ybeta[(size_t)y * 2 + 1] = (int16_t)std::lround(f * coef_scale);
    }

    std::vector<int32_t> horizontal((size_t)src_h * dst_w);
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int sx = xofs[(size_t)x];
            const int a0 = xalpha[(size_t)x * 2 + 0];
            const int a1 = xalpha[(size_t)x * 2 + 1];
            const uint8_t * row = src + (size_t)y * src_w;
            horizontal[(size_t)y * dst_w + x] = (int32_t)row[sx] * a0 + (int32_t)row[std::min(sx + 1, src_w - 1)] * a1;
        }
    }
    constexpr int final_shift = coef_bits * 2;
    constexpr int final_round = 1 << (final_shift - 1);
    for (int y = 0; y < dst_h; ++y) {
        const int sy = yofs[(size_t)y];
        const int b0 = ybeta[(size_t)y * 2 + 0];
        const int b1 = ybeta[(size_t)y * 2 + 1];
        for (int x = 0; x < dst_w; ++x) {
            const int64_t value = (int64_t)horizontal[(size_t)sy * dst_w + x] * b0 +
                                  (int64_t)horizontal[(size_t)std::min(sy + 1, src_h - 1) * dst_w + x] * b1;
            dst[(size_t)y * dst_w + x] =
                (float)std::max<int64_t>(0, std::min<int64_t>(255, (value + final_round) >> final_shift));
        }
    }
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
    // These are reset from the host before every recognition. Keep their
    // storage live for the complete graph so the allocator cannot reuse an
    // initial-state buffer as an intermediate from an earlier timestep.
    ggml_set_output(h);
    ggml_set_output(c);

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

    auto conv_named = [&](ggml_tensor * in, const std::string & base, int kw, int kh, int sw, int sh, int pw, int ph,
                          bool activate) {
        ggml_tensor * w = req(c, (base + ".weight").c_str());
        ggml_tensor * b = req(c, (base + ".bias").c_str());
        ggml_tensor * y = ggml_conv_2d(g, w, in, sw, sh, pw, ph, 1, 1);
        if (b) y = ggml_add(g, y, ggml_reshape_4d(g, b, 1, 1, b->ne[0], 1));
        return activate ? ggml_relu(g, y) : y;
    };

    if (c->network == 1) {
        const std::string root = "FeatureExtraction.ConvNet.";
        auto block = [&](ggml_tensor * in, const std::string & prefix, bool downsample) {
            ggml_tensor * y = conv_named(in, root + prefix + ".conv1", 3, 3, 1, 1, 1, 1, true);
            y = conv_named(y, root + prefix + ".conv2", 3, 3, 1, 1, 1, 1, false);
            ggml_tensor * skip =
                downsample ? conv_named(in, root + prefix + ".downsample.0", 1, 1, 1, 1, 0, 0, false) : in;
            return ggml_relu(g, ggml_add(g, y, skip));
        };
        x = conv_named(x, root + "conv0_1", 3, 3, 1, 1, 1, 1, true);
        x = conv_named(x, root + "conv0_2", 3, 3, 1, 1, 1, 1, true);
        x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        x = block(x, "layer1.0", true);
        x = conv_named(x, root + "conv1", 3, 3, 1, 1, 1, 1, true);
        x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
        x = block(x, "layer2.0", true);
        x = block(x, "layer2.1", false);
        x = conv_named(x, root + "conv2", 3, 3, 1, 1, 1, 1, true);
        // PyTorch maxpool3 is kernel=(2,2), stride=(H=2,W=1), padding=(H=0,W=1).
        // ggml orders the spatial arguments as width, height.
        x = ggml_pool_2d(g, x, GGML_OP_POOL_MAX, 2, 2, 1, 2, 1, 0);
        for (int i = 0; i < 5; ++i) x = block(x, "layer3." + std::to_string(i), i == 0);
        x = conv_named(x, root + "conv3", 3, 3, 1, 1, 1, 1, true);
        for (int i = 0; i < 3; ++i) x = block(x, "layer4." + std::to_string(i), false);
        x = conv_named(x, root + "conv4_1", 2, 2, 1, 2, 1, 0, true);
        x = conv_named(x, root + "conv4_2", 2, 2, 1, 1, 0, 0, true);
    } else {
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
    }
    x = ggml_cont(g, x);
    ggml_set_name(x, "features");
    ggml_set_output(x);

    const int T = (int)x->ne[0];
    const int D = (int)x->ne[2];
    c->time_steps = T;
    c->output_channels = D;
    // EasyOCR applies AdaptiveAvgPool2d((None, 1)) to [B,C,H,W], averaging
    // the height axis while preserving width. In ggml's [W,H,C,B] layout,
    // move H to the pooler's x axis, average it, then restore [C,W].
    x = ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3)); // [H,W,C,B]
    x = ggml_pool_2d(g, x, GGML_OP_POOL_AVG, x->ne[0], 1, x->ne[0], 1, 0, 0);
    x = ggml_cont(g, ggml_permute(g, x, 2, 1, 0, 3)); // [C,W,1,B]
    if (ggml_nelements(x) != D * T) {
        fprintf(stderr, "easyocr: sequence shape is %lld, expected %d\n", (long long)ggml_nelements(x), D * T);
        return false;
    }
    x = ggml_cont(g, ggml_reshape_2d(g, x, D, T));
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

    // The BiLSTM is statically unrolled over the input width. EasyOCR line
    // crops can be substantially wider than the default 200px batch width.
    // Scale graph capacity with time steps so dynamic-width line recognition
    // remains a persistent ggml graph rather than falling back to CPU code.
    const size_t graph_nodes = std::max<size_t>(16384, (size_t)c->time_steps * 700);
    c->graph = ggml_new_graph_custom(g, graph_nodes, false);
    ggml_build_forward_expand(c->graph, c->logits);
    c->alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c->backend));
    return c->alloc && ggml_gallocr_alloc_graph(c->alloc, c->graph);
}

easyocr_ocr_context * easyocr_ocr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    auto * c = new easyocr_ocr_context();
    const bool force_cpu = std::getenv("EASYOCR_FORCE_CPU") != nullptr;
    c->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend_shared();
    if (!c->backend || !core_gguf::load_weights(model_path, c->backend, "easyocr", c->wl)) {
        easyocr_ocr_free(c);
        return nullptr;
    }
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (meta) {
        c->width = (int)core_gguf::kv_u32(meta, "easyocr.input_width", 200);
        c->height = (int)core_gguf::kv_u32(meta, "easyocr.input_height", 64);
        c->classes = (int)core_gguf::kv_u32(meta, "easyocr.num_classes", 0);
        c->hidden = (int)core_gguf::kv_u32(meta, "easyocr.hidden_size", c->hidden);
        c->output_channels = (int)core_gguf::kv_u32(meta, "easyocr.output_channels", c->output_channels);
        c->network = (int)core_gguf::kv_u32(meta, "easyocr.network", 0);
        c->tokens = core_gguf::kv_str_array(meta, "tokenizer.tokens");
        core_gguf::free_metadata(meta);
    }
    if (!build_graph(c)) {
        easyocr_ocr_free(c);
        return nullptr;
    }
    return c;
}

bool easyocr_ocr_set_width(easyocr_ocr_context * c, int width) {
    if (!c || width <= 0 || width == c->width) return c != nullptr;
    if (c->alloc) {
        ggml_gallocr_free(c->alloc);
        c->alloc = nullptr;
    }
    if (c->graph_ctx) {
        ggml_free(c->graph_ctx);
        c->graph_ctx = nullptr;
        c->graph = nullptr;
    }
    c->width = width;
    return build_graph(c);
}

void easyocr_ocr_free(easyocr_ocr_context * c) {
    if (!c) return;
    if (c->alloc) ggml_gallocr_free(c->alloc);
    if (c->graph_ctx) ggml_free(c->graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) crispasr_free_gpu_backend(c->backend);
    delete c;
}

const char * easyocr_ocr_recognize(easyocr_ocr_context * c, const uint8_t * px, int w, int h, int ch, int * out_len) {
    if (!c || !px || w <= 0 || h <= 0 || ch <= 0) return nullptr;
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<uint8_t> gray((size_t)w * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const uint8_t * p = px + ((size_t)y * w + x) * ch;
            gray[(size_t)y * w + x] = ch == 1 ? p[0] : (uint8_t)((299 * p[0] + 587 * p[1] + 114 * p[2] + 500) / 1000);
        }
    std::vector<float> resized((size_t)c->height * c->width);
    const int rw = std::min(c->width, std::max(1, (int)std::ceil((double)c->height * w / h)));
    resize_easyocr_linear_u8(gray.data(), h, w, resized.data(), c->height, rw);
    c->input_host.assign((size_t)c->height * c->width, 0.0f);
    for (int y = 0; y < c->height; ++y)
        for (int x = 0; x < rw; ++x) {
            float v = resized[(size_t)y * rw + x] / 255.0f;
            c->input_host[(size_t)y * c->width + x] = v * 2.0f - 1.0f;
        }
    for (int y = 0; y < c->height; ++y)
        for (int x = rw; x < c->width; ++x)
            c->input_host[(size_t)y * c->width + x] = c->input_host[(size_t)y * c->width + rw - 1];
    const auto preprocess_end = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(c->input, c->input_host.data(), 0, c->input_host.size() * sizeof(float));
    std::vector<float> zero((size_t)c->hidden, 0.0f);
    for (const char * n : { "lstm_fwd_h0", "lstm_fwd_c0", "lstm_rev_h0", "lstm_rev_c0" })
        ggml_backend_tensor_set(ggml_graph_get_tensor(c->graph, n), zero.data(), 0, zero.size() * sizeof(float));
    const auto graph_start = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(c->backend, c->graph) != GGML_STATUS_SUCCESS) return nullptr;
    const auto graph_end = std::chrono::steady_clock::now();

    std::vector<float> logits((size_t)c->classes * c->time_steps);
    ggml_backend_tensor_get(c->logits, logits.data(), 0, logits.size() * sizeof(float));
    std::vector<int> best_tokens;
    std::vector<float> nonblank_probabilities;
    best_tokens.reserve(c->time_steps);
    nonblank_probabilities.reserve(c->time_steps);
    for (int t = 0; t < c->time_steps; ++t) {
        float max_logit = logits[(size_t)t * c->classes];
        for (int k = 1; k < c->classes; ++k) max_logit = std::max(max_logit, logits[(size_t)t * c->classes + k]);
        float sum_exp = 0.0f;
        for (int k = 0; k < c->classes; ++k) sum_exp += std::exp(logits[(size_t)t * c->classes + k] - max_logit);
        float best_probability = 0.0f;
        int best = 0;
        for (int k = 0; k < c->classes; ++k) {
            const float probability = std::exp(logits[(size_t)t * c->classes + k] - max_logit) / sum_exp;
            if (probability > best_probability) {
                best_probability = probability;
                best = k;
            }
        }
        best_tokens.push_back(best);
        if (best != 0) nonblank_probabilities.push_back(best_probability);
    }
    std::vector<std::string> vocabulary;
    if (c->tokens.size() > 1) vocabulary.assign(c->tokens.begin() + 1, c->tokens.end());
    int invalid_token = -1;
    if (!easyocr_postprocess::ctc_greedy_decode(best_tokens, vocabulary, &c->result, &invalid_token)) {
        fprintf(stderr, "easyocr: invalid CTC token %d (vocabulary size=%zu)\n", invalid_token, vocabulary.size());
        c->result.clear();
    }
    c->last_confidence = easyocr_postprocess::confidence_custom_mean(nonblank_probabilities);
    if (out_len) *out_len = (int)c->result.size();
    const auto total_end = std::chrono::steady_clock::now();
    c->last_timing.preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - total_start).count();
    c->last_timing.graph_ms = std::chrono::duration<double, std::milli>(graph_end - graph_start).count();
    c->last_timing.decode_ms = std::chrono::duration<double, std::milli>(total_end - graph_end).count();
    c->last_timing.total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    if (core_env::on("EASYOCR_BENCH")) {
        fprintf(stderr, "[easyocr-bench] preprocess=%.3f graph=%.3f decode=%.3f total=%.3f ms width=%d\n",
                c->last_timing.preprocess_ms, c->last_timing.graph_ms, c->last_timing.decode_ms,
                c->last_timing.total_ms, c->width);
    }
    return c->result.c_str();
}

float easyocr_ocr_last_confidence(const easyocr_ocr_context * c) {
    return c ? c->last_confidence : 0.0f;
}

bool easyocr_ocr_last_timing(const easyocr_ocr_context * c, easyocr_ocr_timing * timing) {
    if (!c || !timing) return false;
    *timing = c->last_timing;
    return true;
}

static bool copy_graph_tensor(ggml_cgraph * graph, const char * name, std::vector<float> & raw,
                              ggml_tensor ** out_tensor) {
    ggml_tensor * t = ggml_graph_get_tensor(graph, name);
    if (!t || t->type != GGML_TYPE_F32) {
        fprintf(stderr, "easyocr-diff: missing/non-F32 graph tensor '%s'\n", name);
        return false;
    }
    raw.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size() * sizeof(float));
    if (out_tensor) *out_tensor = t;
    return true;
}

// Convert GGML's fastest-dimension-first graph views to the contiguous
// PyTorch tensors emitted by dump_easyocr_reference.py.  The mappings are
// explicit even where contiguous storage makes the resulting byte order equal:
// [W,H,C] -> [C,H,W], [D,T] -> [T,D], and [V,T] -> [T,V].
static bool to_reference_layout(const char * name, const ggml_tensor * t, const std::vector<float> & raw,
                                std::vector<float> & ordered) {
    const int64_t n0 = t->ne[0], n1 = t->ne[1], n2 = t->ne[2], n3 = t->ne[3];
    ordered.resize(raw.size());
    if (!strcmp(name, "features") || !strcmp(name, "input_image")) {
        for (int64_t b = 0; b < n3; ++b)
            for (int64_t c = 0; c < n2; ++c)
                for (int64_t y = 0; y < n1; ++y)
                    for (int64_t x = 0; x < n0; ++x) {
                        const size_t i = (size_t)x + (size_t)n0 * (y + n1 * (c + n2 * b));
                        ordered[i] = raw[i];
                    }
        return true;
    }
    if (!strcmp(name, "sequence_input") || !strcmp(name, "bilstm_0") || !strcmp(name, "bilstm_1") ||
        !strcmp(name, "logits")) {
        const int64_t d = n0, tlen = n1;
        for (int64_t tpos = 0; tpos < tlen; ++tpos)
            for (int64_t dim = 0; dim < d; ++dim) ordered[(size_t)tpos * d + dim] = raw[(size_t)dim + (size_t)d * tpos];
        return true;
    }
    return false;
}

int easyocr_ocr_diff(easyocr_ocr_context * c, const char * ref_path) {
    if (!c || !c->graph || !ref_path) return 1;
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) return 1;

    static const char * names[] = {
        "input_image", "features", "sequence_input", "bilstm_0", "bilstm_1", "logits",
    };
    int failures = 0;
    for (const char * name : names) {
        if (!ref.has(name)) {
            fprintf(stderr, "easyocr-diff: reference is missing '%s'\n", name);
            failures++;
            continue;
        }
        std::vector<float> raw, ordered;
        if (!strcmp(name, "input_image")) {
            ordered = c->input_host;
            auto report = ref.compare(name, ordered.data(), ordered.size(), 0);
            const bool pass = report.is_pass(0.99f);
            print_diff_report(name, report, pass);
            if (!pass) failures++;
            continue;
        }
        ggml_tensor * t = nullptr;
        if (!copy_graph_tensor(c->graph, name, raw, &t) || !to_reference_layout(name, t, raw, ordered)) {
            failures++;
            continue;
        }
        const int row_dim = !strcmp(name, "features") ? 0 : !strcmp(name, "sequence_input") ? 1 : 0;
        auto report = ref.compare(name, ordered.data(), ordered.size(), row_dim);
        const bool sparse_feature_pass = !strcmp(name, "features") && report.cos_global >= 0.99f;
        const bool pass = report.is_pass(0.99f) || sparse_feature_pass;
        print_diff_report(name, report, pass);
        if (std::getenv("EASYOCR_DIFF_DEBUG") && row_dim >= 0 &&
            (!pass || !strcmp(name, "sequence_input") || !strcmp(name, "bilstm_1") || !strcmp(name, "logits"))) {
            auto ref_values = ref.get_f32(name);
            const size_t row_size = row_dim < (int)ref.shape(name).size() ? (size_t)ref.shape(name)[row_dim] : 0;
            if (row_size > 0) {
                const size_t rows = std::min(ordered.size(), ref_values.second) / row_size;
                float worst = 2.0f;
                size_t worst_row = 0;
                float mine_row_norm = 0.0f;
                float ref_row_norm = 0.0f;
                for (size_t row = 0; row < rows; ++row) {
                    double dot = 0.0, mine_sq = 0.0, ref_sq = 0.0;
                    for (size_t j = 0; j < row_size; ++j) {
                        const float mine = ordered[row * row_size + j];
                        const float truth = ref_values.first[row * row_size + j];
                        dot += (double)mine * truth;
                        mine_sq += (double)mine * mine;
                        ref_sq += (double)truth * truth;
                    }
                    const float cos = mine_sq > 1e-18 && ref_sq > 1e-18
                                          ? (float)(dot / (std::sqrt(mine_sq) * std::sqrt(ref_sq)))
                                          : (mine_sq <= 1e-18 && ref_sq <= 1e-18 ? 1.0f : 0.0f);
                    if (cos < worst) {
                        worst = cos;
                        worst_row = row;
                        mine_row_norm = (float)std::sqrt(mine_sq);
                        ref_row_norm = (float)std::sqrt(ref_sq);
                    }
                }
                printf("easyocr-diff-debug %-16s worst_row=%zu cos=%.7f mine=%.6g ref=%.6g\n", name, worst_row, worst,
                       mine_row_norm, ref_row_norm);
                if (!strcmp(name, "logits")) {
                    std::vector<size_t> order(row_size);
                    for (size_t j = 0; j < row_size; ++j) order[j] = j;
                    const size_t top = std::min<size_t>(5, row_size);
                    std::partial_sort(order.begin(), order.begin() + top, order.end(), [&](size_t a, size_t b) {
                        return std::fabs(ordered[worst_row * row_size + a] -
                                         ref_values.first[worst_row * row_size + a]) >
                               std::fabs(ordered[worst_row * row_size + b] -
                                         ref_values.first[worst_row * row_size + b]);
                    });
                    for (size_t k = 0; k < top; ++k) {
                        const size_t cls = order[k];
                        printf("easyocr-diff-debug logits-row=%zu class=%zu mine=%.7g ref=%.7g diff=%.7g\n", worst_row,
                               cls, ordered[worst_row * row_size + cls], ref_values.first[worst_row * row_size + cls],
                               ordered[worst_row * row_size + cls] - ref_values.first[worst_row * row_size + cls]);
                    }
                }
            }
        }
        if (!strcmp(name, "logits") && std::getenv("EASYOCR_DIFF_DEBUG")) {
            auto ref_logits = ref.get_f32(name);
            const int vocab = (int)t->ne[0];
            const int steps = (int)t->ne[1];
            int mismatches = 0;
            for (int step = 0; step < steps; ++step) {
                int mine_best = 0;
                int ref_best = 0;
                for (int cls = 1; cls < vocab; ++cls) {
                    if (ordered[(size_t)step * vocab + cls] > ordered[(size_t)step * vocab + mine_best])
                        mine_best = cls;
                    if (ref_logits.first[(size_t)step * vocab + cls] >
                        ref_logits.first[(size_t)step * vocab + ref_best])
                        ref_best = cls;
                }
                if (mine_best != ref_best) {
                    printf("easyocr-diff logits-debug step=%d mine=%d ref=%d mine_value=%.7g ref_value=%.7g\n", step,
                           mine_best, ref_best, ordered[(size_t)step * vocab + mine_best],
                           ref_logits.first[(size_t)step * vocab + ref_best]);
                    mismatches++;
                }
            }
            printf("easyocr-diff logits-debug argmax_mismatches=%d/%d\n", mismatches, steps);
        }
        if (!pass) failures++;
    }
    return failures;
}
