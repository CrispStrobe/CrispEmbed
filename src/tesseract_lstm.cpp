// tesseract_lstm.cpp — Tesseract LSTM line-recognition engine via ggml.
//
// Implements the VGSL forward pass:
//   Input → Convolve 3×3 stacking → FC+tanh → MaxPool 3×3 →
//   XYTranspose → SummLSTM (y-summarize) → XYTranspose →
//   LSTM (forward) → LSTM (reverse) → LSTM (forward) → Softmax →
//   CTC greedy decode.
//
// All computation is CPU-side (no ggml graph) — models are tiny (~1-5 MB).
// Weights are dequantized to F32 on load and cached.

#include "tesseract_lstm.h"
#include "tesseract_dawg.h"
#include "tesseract_dawg_score.h"
#include "tesseract_recoder.h"

#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "core/env_gate.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LSTM layer weights (dequantized F32, cached)
// ---------------------------------------------------------------------------

struct lstm_weights {
    std::vector<float> W_ih; // (4*ns, ni)
    std::vector<float> W_hh; // (4*ns, ns)
    std::vector<float> bias; // (4*ns,)
    // Packed cached rows: [W_ih | W_hh] for each gate.  The int-mode hot
    // path can then accumulate one contiguous row against [input | hidden].
    std::vector<int8_t> W_q;
    std::vector<int32_t> bias_q;
    std::vector<float> q_scale;
    int ni;
    int ns; // hidden size
};

struct lstm_scratch {
    std::vector<float> h, c, gates;
    std::vector<int8_t> input_q, hidden_q, activation_q;
};

static int tesseract_round_int(float value) {
    return value >= 0.0f ? (int)floorf(value + 0.5f) : -(int)floorf(-value + 0.5f);
}

// The converter stores an int-mode row as raw_int8 * stored_scale, where the
// original Tesseract runtime uses raw_int8 inputs and stored_scale / 127.
// Reconstruct the row scale and accumulate in int32 to preserve that path.
static float int8_row_dot(const float * weights, int n, float bias, const float * input) {
    float max_abs = fabsf(bias);
    for (int i = 0; i < n; ++i) max_abs = std::max(max_abs, fabsf(weights[i]));
    if (max_abs == 0.0f) return 0.0f;
    const float stored_scale = max_abs / 127.0f;
    int32_t acc = tesseract_round_int(bias / stored_scale) * 127;
    for (int i = 0; i < n; ++i) {
        const int wi = tesseract_round_int(weights[i] / stored_scale);
        const int xi = tesseract_round_int(input[i] * 127.0f);
        acc += wi * xi;
    }
    return (float)acc * stored_scale / 127.0f;
}

static float int8_lstm_row_dot(const float * w_ih, const float * w_hh, int ni, int ns, float bias, const float * input,
                               const float * hidden) {
    float max_abs = fabsf(bias);
    for (int i = 0; i < ni; ++i) max_abs = std::max(max_abs, fabsf(w_ih[i]));
    for (int i = 0; i < ns; ++i) max_abs = std::max(max_abs, fabsf(w_hh[i]));
    if (max_abs == 0.0f) return 0.0f;
    const float stored_scale = max_abs / 127.0f;
    int32_t acc = tesseract_round_int(bias / stored_scale) * 127;
    for (int i = 0; i < ni; ++i) {
        acc += tesseract_round_int(w_ih[i] / stored_scale) * tesseract_round_int(input[i] * 127.0f);
    }
    for (int i = 0; i < ns; ++i) {
        acc += tesseract_round_int(w_hh[i] / stored_scale) * tesseract_round_int(hidden[i] * 127.0f);
    }
    return (float)acc * stored_scale / 127.0f;
}

static void prepare_lstm_int_weights(lstm_weights & lw) {
    const int gates = 4 * lw.ns;
    lw.bias_q.resize(gates);
    lw.q_scale.resize(gates);
    lw.W_q.resize((size_t)gates * (lw.ni + lw.ns));
    for (int g = 0; g < gates; ++g) {
        float max_abs = fabsf(lw.bias[g]);
        for (int i = 0; i < lw.ni; ++i) max_abs = std::max(max_abs, fabsf(lw.W_ih[g * lw.ni + i]));
        for (int i = 0; i < lw.ns; ++i) max_abs = std::max(max_abs, fabsf(lw.W_hh[g * lw.ns + i]));
        lw.q_scale[g] = max_abs / 127.0f;
        if (lw.q_scale[g] == 0.0f) continue;
        lw.bias_q[g] = tesseract_round_int(lw.bias[g] / lw.q_scale[g]);
        int8_t * packed = lw.W_q.data() + (size_t)g * (lw.ni + lw.ns);
        for (int i = 0; i < lw.ni; ++i) packed[i] = (int8_t)tesseract_round_int(lw.W_ih[g * lw.ni + i] / lw.q_scale[g]);
        for (int i = 0; i < lw.ns; ++i)
            packed[lw.ni + i] = (int8_t)tesseract_round_int(lw.W_hh[g * lw.ns + i] / lw.q_scale[g]);
    }
}

static float int8_lstm_row_dot_cached(const lstm_weights & lw, int gate, const int8_t * activation_q) {
    const float scale = lw.q_scale[gate];
    if (scale == 0.0f) return 0.0f;
    int32_t acc = lw.bias_q[gate] * 127;
    const int n = lw.ni + lw.ns;
    const int8_t * w = lw.W_q.data() + (size_t)gate * n;
    for (int i = 0; i < n; ++i) acc += (int32_t)w[i] * activation_q[i];
    return (float)acc * scale / 127.0f;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct tesseract_lstm_context {
    // Hyperparameters
    int input_height;
    int num_classes;
    int null_char;
    int num_lstm_layers;
    uint32_t training_flags;
    int32_t sample_iteration;
    bool int_mode;
    std::string vgsl_spec;

    // Conv FC weights
    std::vector<float> conv_w; // (16, 9)  — [out][in] row-major
    std::vector<float> conv_b; // (16,)
    int conv_out;              // 16

    // LSTM layers
    std::vector<lstm_weights> lstm;

    // Output FC weights
    std::vector<float> out_w; // (n_classes, last_lstm_ns)
    std::vector<float> out_b; // (n_classes,)

    // Per-LSTM metadata
    std::vector<std::string> lstm_types; // "y_sum", "fwd", "rev"

    // Reverse recoder: output_class → unichar_id (-1 if unmapped)
    std::vector<int> output_to_unichar;
    std::vector<std::vector<int>> recoder_codes;

    // Optional parsed language models. Loading is diagnostic-only until
    // dictionary scoring has passed official-output parity.
    std::map<std::string, tesseract_dawg::Dawg> dawgs;

    // Unicharset tokens
    std::vector<std::string> tokens;

    // Losslessly preserved DAWG payloads for future decoder work. The native
    // production decoder deliberately does not score these yet.
    std::vector<std::string> dawg_components;
    std::map<std::string, std::string> dawg_payloads;
    std::map<std::string, tesseract_dawg_context *> dawg_contexts;

    ~tesseract_lstm_context() {
        for (auto & entry : dawg_contexts) tesseract_dawg_free(entry.second);
    }

    // Inference results
    std::string result_buf;
    std::vector<float> char_confs;
    float sequence_confidence = 0.0f;
    float word_confidence = 0.0f;

    // Diff mode: capture per-stage intermediates
    bool dump_mode = false;
    std::map<std::string, std::vector<float>> captures;

    bool bench = false;
    bool cache_int = true;
    bool reuse_scratch = false;

    // GGUF loader state
    core_gguf::WeightLoad wl;
    // Dequantized weight cache
    std::map<const void *, std::vector<float>> dequant_cache;
    // Per-context activation scratch. Each orchestrator worker owns one
    // recognizer context, so these buffers can be reused across line crops
    // without sharing mutable state between workers.
    std::vector<float> scratch_input;
    std::vector<float> scratch_convolve;
    std::vector<float> scratch_fc;
    std::vector<float> scratch_pool;
    std::vector<float> scratch_transposed;
    std::vector<float> scratch_seq_a;
    std::vector<float> scratch_seq_b;
    std::vector<float> scratch_logits;
    lstm_scratch scratch_lstm;
};

// ---------------------------------------------------------------------------
// Tensor dequantization helper
// ---------------------------------------------------------------------------

static const float * tensor_f32(tesseract_lstm_context * ctx, struct ggml_tensor * t) {
    // Weights resident on CUDA/Vulkan/SYCL/HIP have a DEVICE pointer in t->data
    // that must not be returned/read on the host. Keep the zero-copy fast path
    // only for host-visible buffers (CPU / Metal); otherwise dequant through the
    // backend buffer via ggml_backend_tensor_get.
    const bool host = !t->buffer || ggml_backend_buffer_is_host(t->buffer);
    if (t->type == GGML_TYPE_F32 && host) {
        return (const float *)t->data;
    }
    auto it = ctx->dequant_cache.find(t->data);
    if (it != ctx->dequant_cache.end()) {
        return it->second.data();
    }
    const int64_t n = ggml_nelements(t);
    auto & buf = ctx->dequant_cache[t->data];
    buf.resize(n);
    std::vector<uint8_t> raw;
    const void * src_bytes;
    if (t->buffer) {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        src_bytes = raw.data();
    } else {
        src_bytes = t->data;
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf.data(), src_bytes, n * sizeof(float));
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits->to_float) {
            traits->to_float(src_bytes, buf.data(), n);
        } else {
            fprintf(stderr, "tesseract_lstm: cannot dequantize type %d\n", t->type);
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
    }
    return buf.data();
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

static bool load_model(tesseract_lstm_context * ctx, const char * path) {
    // Pass 1: metadata
    gguf_context * meta = core_gguf::open_metadata(path);
    if (!meta) return false;

    ctx->input_height = (int)core_gguf::kv_u32(meta, "tesseract_lstm.input_height", 36);
    ctx->num_classes = (int)core_gguf::kv_u32(meta, "tesseract_lstm.num_classes", 111);
    ctx->null_char = (int)core_gguf::kv_u32(meta, "tesseract_lstm.null_char", 110);
    ctx->num_lstm_layers = (int)core_gguf::kv_u32(meta, "tesseract_lstm.num_lstm_layers", 4);
    ctx->training_flags = core_gguf::kv_u32(meta, "tesseract_lstm.training_flags", 0);
    ctx->sample_iteration = core_gguf::kv_i32(meta, "tesseract_lstm.sample_iteration", 0);
    ctx->int_mode = core_gguf::kv_bool(meta, "tesseract_lstm.int_mode", (ctx->training_flags & 1) != 0);
    ctx->vgsl_spec = core_gguf::kv_str(meta, "tesseract_lstm.vgsl_spec", "");

    // New converters preserve DAWG payloads as base64 metadata. Older GGUFs
    // have no manifest and remain valid with zero components. Reject a
    // malformed new manifest rather than silently claiming dictionary data is
    // available for a future scoring implementation.
    ctx->dawg_components = core_gguf::kv_str_array(meta, "tesseract_lstm.dawg_components");
    for (const std::string & name : ctx->dawg_components) {
        const std::string key = "tesseract_lstm.dawg." + name + ".base64";
        const std::string payload = core_gguf::kv_str(meta, key.c_str(), "");
        if (payload.empty()) {
            fprintf(stderr, "tesseract_lstm: DAWG manifest entry '%s' has no payload\n", name.c_str());
            core_gguf::free_metadata(meta);
            return false;
        }
        char dawg_error[128];
        if (!tesseract_dawg_validate_base64(payload.c_str(), dawg_error, sizeof(dawg_error))) {
            fprintf(stderr, "tesseract_lstm: invalid DAWG '%s': %s\n", name.c_str(), dawg_error);
            core_gguf::free_metadata(meta);
            return false;
        }
        ctx->dawg_payloads.emplace(name, payload);
        tesseract_dawg_context * parsed = tesseract_dawg_init_base64(payload.c_str(), dawg_error, sizeof(dawg_error));
        if (!parsed) {
            fprintf(stderr, "tesseract_lstm: failed to cache DAWG '%s': %s\n", name.c_str(), dawg_error);
            core_gguf::free_metadata(meta);
            return false;
        }
        ctx->dawg_contexts.emplace(name, parsed);
    }

    // Tokens
    ctx->tokens = core_gguf::kv_str_array(meta, "tokenizer.tokens");
    // Tesseract reserves unichar id 0 for the space character (UNICHAR_SPACE).
    // The GGUF converter dropped it (stored ""), which is why word spaces never
    // appeared in the output — restore it so the LSTM's space class emits ' '.
    if (!ctx->tokens.empty() && ctx->tokens[0].empty()) {
        ctx->tokens[0] = " ";
    }

    // Reverse recoder
    std::vector<int> rev = core_gguf::kv_i32_array(meta, "tesseract_lstm.output_to_unichar");
    ctx->output_to_unichar = std::move(rev);
    std::vector<int> recoder_flat = core_gguf::kv_i32_array(meta, "tesseract_lstm.recoder_map");
    std::vector<int> recoder_offsets = core_gguf::kv_i32_array(meta, "tesseract_lstm.recoder_offsets");
    if (recoder_offsets.size() >= 2) {
        ctx->recoder_codes.reserve(recoder_offsets.size() - 1);
        for (size_t i = 0; i + 1 < recoder_offsets.size(); ++i) {
            const int begin = recoder_offsets[i];
            const int end = recoder_offsets[i + 1];
            if (begin >= 0 && end >= begin && end <= (int)recoder_flat.size()) {
                ctx->recoder_codes.emplace_back(recoder_flat.begin() + begin, recoder_flat.begin() + end);
            }
        }
    }

    // DAWG payloads are opt-in. Existing GGUFs do not contain them, and
    // loading them must not alter the default recognition path.
    if (std::getenv("CRISPEMBED_TESSERACT_DAWG_LOAD") != nullptr) {
        const auto dawg_names = core_gguf::kv_str_array(meta, "tesseract_lstm.dawg_names");
        for (const auto & name : dawg_names) {
            const std::string key = "tesseract_lstm.dawg." + name;
            const auto bytes = core_gguf::kv_u8_array(meta, key.c_str());
            tesseract_dawg::Dawg dawg;
            std::string error;
            if (tesseract_dawg::parse(bytes, dawg, &error)) {
                ctx->dawgs.emplace(name, std::move(dawg));
            } else {
                fprintf(stderr, "tesseract_lstm: ignoring invalid DAWG %s: %s\n", name.c_str(), error.c_str());
            }
        }
        fprintf(stderr, "tesseract_lstm: loaded %zu optional DAWG graph(s)\n", ctx->dawgs.size());
    }

    // LSTM types
    ctx->lstm_types = core_gguf::kv_str_array(meta, "tesseract_lstm.lstm_types");

    core_gguf::free_metadata(meta);

    // Pass 2: weights
    // This recognizer is entirely CPU-side (see tesseract_lstm_init: n_threads
    // is unused and every weight below is copied into host vectors, after
    // which the dequant cache is dropped). The backend exists only to pull the
    // GGUF through core_gguf::load_weights, so asking for a GPU one spins up
    // Metal -- shader library and all -- for a sub-2 MB model and then frees
    // it again. Measured at 1.07 s warm and 4.97 s cold against 4.8 ms for the
    // CPU backend, i.e. ~85% of a one-shot CLI invocation spent initialising a
    // device this engine never uses. CRISPEMBED_TESSERACT_GPU_LOAD restores the
    // old behaviour for bisection.
    const bool gpu_load = std::getenv("CRISPEMBED_TESSERACT_GPU_LOAD") != nullptr &&
                          std::getenv("CRISPEMBED_TESSERACT_FORCE_CPU") == nullptr;
    ggml_backend_t backend = gpu_load ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(path, backend, "tesseract_lstm", ctx->wl)) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_free(backend);

    const auto & T = ctx->wl.tensors;
    auto req = [&](const char * name) -> ggml_tensor * { return core_gguf::require(T, name, "tesseract_lstm"); };

    // Conv FC
    ggml_tensor * cw = req("conv.weight");
    ggml_tensor * cb = req("conv.bias");
    if (!cw || !cb) return false;
    const float * cw_f = tensor_f32(ctx, cw);
    const float * cb_f = tensor_f32(ctx, cb);
    int conv_ni = (int)cw->ne[0];   // 9
    ctx->conv_out = (int)cw->ne[1]; // 16
    ctx->conv_w.assign(cw_f, cw_f + conv_ni * ctx->conv_out);
    ctx->conv_b.assign(cb_f, cb_f + ctx->conv_out);

    // LSTM layers
    ctx->lstm.resize(ctx->num_lstm_layers);
    char buf[128];
    for (int i = 0; i < ctx->num_lstm_layers; i++) {
        auto & lw = ctx->lstm[i];
        snprintf(buf, sizeof(buf), "lstm.%d.weight_ih", i);
        ggml_tensor * wih = req(buf);
        snprintf(buf, sizeof(buf), "lstm.%d.weight_hh", i);
        ggml_tensor * whh = req(buf);
        snprintf(buf, sizeof(buf), "lstm.%d.bias", i);
        ggml_tensor * b = req(buf);
        if (!wih || !whh || !b) return false;

        lw.ni = (int)wih->ne[0];
        lw.ns = (int)wih->ne[1] / 4;

        const float * wih_f = tensor_f32(ctx, wih);
        const float * whh_f = tensor_f32(ctx, whh);
        const float * b_f = tensor_f32(ctx, b);

        int gate_size = 4 * lw.ns;
        lw.W_ih.assign(wih_f, wih_f + gate_size * lw.ni);
        lw.W_hh.assign(whh_f, whh_f + gate_size * lw.ns);
        lw.bias.assign(b_f, b_f + gate_size);
        prepare_lstm_int_weights(lw);
    }

    // Output FC
    ggml_tensor * ow = req("output.weight");
    ggml_tensor * ob = req("output.bias");
    if (!ow || !ob) return false;
    const float * ow_f = tensor_f32(ctx, ow);
    const float * ob_f = tensor_f32(ctx, ob);
    int out_ni = (int)ow->ne[0];
    int out_no = (int)ow->ne[1];
    ctx->out_w.assign(ow_f, ow_f + out_ni * out_no);
    ctx->out_b.assign(ob_f, ob_f + out_no);

    // Clear dequant cache — we've copied everything we need
    ctx->dequant_cache.clear();

    fprintf(stderr, "tesseract_lstm: loaded %s (%d LSTM layers, %d classes, height=%d, int_mode=%s, dawg=%zu)\n",
            ctx->vgsl_spec.c_str(), ctx->num_lstm_layers, ctx->num_classes, ctx->input_height,
            ctx->int_mode ? "true" : "false", ctx->dawg_components.size());

    return true;
}

// ---------------------------------------------------------------------------
// LSTM forward (single direction)
// ---------------------------------------------------------------------------

// Tesseract's TF_INT_MODE stores NetworkIO activations as signed int8 values
// representing [-1, 1]. Its rounding is away from zero at half values.
static float quantize_int_activation(float value) {
    float scaled = value * 127.0f;
    int rounded = scaled >= 0.0f ? (int)floorf(scaled + 0.5f) : -(int)floorf(-scaled + 0.5f);
    rounded = std::max(-127, std::min(127, rounded));
    return (float)rounded / 127.0f;
}

static float quantize_int_input(float value) {
    float scaled = value * 128.0f;
    int rounded = scaled >= 0.0f ? (int)floorf(scaled + 0.5f) : -(int)floorf(-scaled + 0.5f);
    rounded = std::max(-127, std::min(127, rounded));
    return (float)rounded / 127.0f;
}

// Tesseract uses generated 4096-entry LUTs with 1/256 input spacing for its
// LSTM tanh/logistic functions. The upstream generator evaluates math.tanh
// and math.exp in double precision, then stores the literals as TFloat. Build
// the same tables once; synthesizing float-math entries per call changes int8
// quantization boundaries.
static const std::array<float, 4096> & tesseract_tanh_table() {
    static const std::array<float, 4096> table = [] {
        std::array<float, 4096> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>(std::tanh(static_cast<double>(i) / 256.0));
        }
        return values;
    }();
    return table;
}

static const std::array<float, 4096> & tesseract_logistic_table() {
    static const std::array<float, 4096> table = [] {
        std::array<float, 4096> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            const double x = static_cast<double>(i) / 256.0;
            values[i] = static_cast<float>(1.0 / (1.0 + std::exp(-x)));
        }
        return values;
    }();
    return table;
}

static float tesseract_tanh(float value) {
    const bool negative = value < 0.0f;
    const float x = fabsf(value) * 256.0f;
    const unsigned index = (unsigned)x;
    if (index >= 4095) return negative ? -1.0f : 1.0f;
    const float frac = x - (float)index;
    const auto & table = tesseract_tanh_table();
    const float result = table[index] + (table[index + 1] - table[index]) * frac;
    return negative ? -result : result;
}

static float tesseract_logistic(float value) {
    const bool negative = value < 0.0f;
    const float x = fabsf(value) * 256.0f;
    const unsigned index = (unsigned)x;
    if (index >= 4095) return negative ? 0.0f : 1.0f;
    const float frac = x - (float)index;
    const auto & table = tesseract_logistic_table();
    const float result = table[index] + (table[index + 1] - table[index]) * frac;
    return negative ? 1.0f - result : result;
}

static void lstm_forward(const float * input, // (T, ni)
                         float * output,      // (T, ns)
                         int T, int ni, int ns,
                         const float * W_ih, // (4*ns, ni)
                         const float * W_hh, // (4*ns, ns)
                         const float * bias, // (4*ns,)
                         bool reverse, bool int_mode, const lstm_weights * cached = nullptr,
                         lstm_scratch * scratch = nullptr) {
    // Gate order (PyTorch): i, f, g, o
    const int gs = 4 * ns;
    std::vector<float> local_h, local_c, local_gates;
    std::vector<int8_t> local_input_q, local_hidden_q, local_activation_q;
    auto & h = scratch ? scratch->h : local_h;
    auto & c = scratch ? scratch->c : local_c;
    auto & gates = scratch ? scratch->gates : local_gates;
    auto & input_q = scratch ? scratch->input_q : local_input_q;
    auto & hidden_q = scratch ? scratch->hidden_q : local_hidden_q;
    auto & activation_q = scratch ? scratch->activation_q : local_activation_q;
    h.assign(ns, 0.0f);
    c.assign(ns, 0.0f);
    gates.resize(gs);
    input_q.resize(int_mode ? ni : 0);
    hidden_q.resize(int_mode ? ns : 0);
    activation_q.resize(int_mode && cached ? ni + ns : 0);

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;
        const float * xt = input + t * ni;
        if (int_mode) {
            for (int i = 0; i < ni; ++i) input_q[i] = (int8_t)tesseract_round_int(xt[i] * 127.0f);
            for (int i = 0; i < ns; ++i) hidden_q[i] = (int8_t)tesseract_round_int(h[i] * 127.0f);
            if (cached) {
                memcpy(activation_q.data(), input_q.data(), ni);
                memcpy(activation_q.data() + ni, hidden_q.data(), ns);
            }
        }

        // gates = W_ih @ x + W_hh @ h + bias (SIMD-accelerated dot products)
        for (int g = 0; g < gs; g++) {
            gates[g] = int_mode
                           ? (cached ? int8_lstm_row_dot_cached(*cached, g, activation_q.data())
                                     : int8_lstm_row_dot(W_ih + g * ni, W_hh + g * ns, ni, ns, bias[g], xt, h.data()))
                           : bias[g] + core_cpu::dot_product(W_ih + g * ni, xt, ni) +
                                 core_cpu::dot_product(W_hh + g * ns, h.data(), ns);
        }

        for (int j = 0; j < ns; j++) {
            float i_gate = tesseract_logistic(gates[0 * ns + j]);
            float f_gate = tesseract_logistic(gates[1 * ns + j]);
            float g_gate = tesseract_tanh(gates[2 * ns + j]);
            float o_gate = tesseract_logistic(gates[3 * ns + j]);

            c[j] = f_gate * c[j] + i_gate * g_gate;
            if (c[j] > 100.0f) c[j] = 100.0f;
            if (c[j] < -100.0f) c[j] = -100.0f;
            h[j] = o_gate * tesseract_tanh(c[j]);
            if (int_mode) h[j] = quantize_int_activation(h[j]);
        }

        memcpy(output + t * ns, h.data(), ns * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// SummLSTM: run LSTM over height dimension, keep last hidden state per column
// ---------------------------------------------------------------------------

static void summ_lstm_forward(const float * input, // (height, width, channels) — row-major after XYTranspose
                              float * output,      // (width, ns) — one hidden state per column
                              int height, int width, int channels, int ns,
                              const float * W_ih, // (4*ns, channels)
                              const float * W_hh, // (4*ns, ns)
                              const float * bias, // (4*ns,)
                              bool int_mode, const lstm_weights * cached = nullptr, lstm_scratch * scratch = nullptr) {
    // After XYTranspose: height = original_width, width = original_height
    // For each row (height position), run LSTM across the width (original height).
    // State resets at each row boundary.
    const int gs = 4 * ns;
    std::vector<float> local_h, local_c, local_gates;
    std::vector<int8_t> local_input_q, local_hidden_q, local_activation_q;
    auto & h = scratch ? scratch->h : local_h;
    auto & c = scratch ? scratch->c : local_c;
    auto & gates = scratch ? scratch->gates : local_gates;
    auto & input_q = scratch ? scratch->input_q : local_input_q;
    auto & hidden_q = scratch ? scratch->hidden_q : local_hidden_q;
    auto & activation_q = scratch ? scratch->activation_q : local_activation_q;
    h.resize(ns);
    c.resize(ns);
    gates.resize(gs);
    input_q.resize(int_mode ? channels : 0);
    hidden_q.resize(int_mode ? ns : 0);
    activation_q.resize(int_mode && cached ? channels + ns : 0);

    for (int row = 0; row < height; row++) {
        // Reset state per row
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);

        for (int col = 0; col < width; col++) {
            const float * xt = input + (row * width + col) * channels;
            if (int_mode) {
                for (int i = 0; i < channels; ++i) input_q[i] = (int8_t)tesseract_round_int(xt[i] * 127.0f);
                for (int i = 0; i < ns; ++i) hidden_q[i] = (int8_t)tesseract_round_int(h[i] * 127.0f);
                if (cached) {
                    memcpy(activation_q.data(), input_q.data(), channels);
                    memcpy(activation_q.data() + channels, hidden_q.data(), ns);
                }
            }

            // SIMD-accelerated gate computation
            for (int g = 0; g < gs; g++) {
                gates[g] = int_mode ? (cached ? int8_lstm_row_dot_cached(*cached, g, activation_q.data())
                                              : int8_lstm_row_dot(W_ih + g * channels, W_hh + g * ns, channels, ns,
                                                                  bias[g], xt, h.data()))
                                    : bias[g] + core_cpu::dot_product(W_ih + g * channels, xt, channels) +
                                          core_cpu::dot_product(W_hh + g * ns, h.data(), ns);
            }

            for (int j = 0; j < ns; j++) {
                float i_gate = tesseract_logistic(gates[0 * ns + j]);
                float f_gate = tesseract_logistic(gates[1 * ns + j]);
                float g_gate = tesseract_tanh(gates[2 * ns + j]);
                float o_gate = tesseract_logistic(gates[3 * ns + j]);

                c[j] = f_gate * c[j] + i_gate * g_gate;
                if (c[j] > 100.0f) c[j] = 100.0f;
                if (c[j] < -100.0f) c[j] = -100.0f;
                h[j] = o_gate * tesseract_tanh(c[j]);
                if (int_mode) h[j] = quantize_int_activation(h[j]);
            }
        }

        // Keep last hidden state
        memcpy(output + row * ns, h.data(), ns * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Image normalization (matches Tesseract's ComputeBlackWhite + SetPixel)
// ---------------------------------------------------------------------------

static void normalize_image(const uint8_t * pixels, int width, int height,
                            float * out) // (height, width)
{
    // ComputeBlackWhite: scan middle row for local min/max
    int mid_y = height / 2;
    std::vector<float> mins, maxes;

    if (width >= 3) {
        float prev = (float)pixels[mid_y * width + 0];
        float curr = (float)pixels[mid_y * width + 1];
        for (int x = 1; x + 1 < width; x++) {
            float next = (float)pixels[mid_y * width + x + 1];
            if ((curr < prev && curr <= next) || (curr <= prev && curr < next)) mins.push_back(curr);
            if ((curr > prev && curr >= next) || (curr >= prev && curr > next)) maxes.push_back(curr);
            prev = curr;
            curr = next;
        }
    }
    if (mins.empty()) mins.push_back(0.0f);
    if (maxes.empty()) maxes.push_back(255.0f);

    // Tesseract's STATS::ile uses a 0..255 histogram and interpolates within
    // the bucket containing frac * total_count. It is not a sorted-sample
    // percentile; that distinction changes the int8 input rounding.
    auto percentile_histogram = [](const std::vector<float> & values, float q) {
        int buckets[256] = {};
        for (float value : values) {
            const int bucket = std::max(0, std::min(255, (int)value));
            buckets[bucket]++;
        }
        const int total = (int)values.size();
        const double target = std::max(1.0, std::min((double)total, (double)q * total));
        int sum = 0;
        for (int index = 0; index <= 255 && sum < target; ++index) {
            sum += buckets[index];
            if (sum >= target && buckets[index] > 0) {
                // STATS::ile increments its loop index before returning.
                return (float)(index + 1) - (float)((sum - target) / buckets[index]);
            }
        }
        return 0.0f;
    };
    float black = percentile_histogram(mins, 0.25f);
    float white = percentile_histogram(maxes, 0.75f);
    float contrast = (white - black) / 2.0f;
    if (contrast <= 0.0f) contrast = 1.0f;

    for (int i = 0; i < width * height; i++) {
        out[i] = ((float)pixels[i] - black) / contrast - 1.0f;
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

// Helper: capture a buffer for diff comparison
static void capture(tesseract_lstm_context * ctx, const char * name, const float * data, size_t n) {
    if (ctx->dump_mode) ctx->captures[name].assign(data, data + n);
}

struct ctc_beam_state {
    std::vector<int> prefix;
    float p_blank = -INFINITY;
    float p_nonblank = -INFINITY;
};

static float log_add(float a, float b) {
    if (!std::isfinite(a)) return b;
    if (!std::isfinite(b)) return a;
    const float hi = std::max(a, b);
    return hi + log1pf(expf(std::min(a, b) - hi));
}

static float beam_add(float a, float b, bool viterbi) {
    return viterbi ? std::max(a, b) : log_add(a, b);
}

static std::vector<int> ctc_prefix_beam_decode(const std::vector<float> & logits, int timesteps, int classes, int blank,
                                               int beam_width, bool viterbi,
                                               const std::vector<std::vector<int>> * recoder = nullptr,
                                               float * score_out = nullptr,
                                               const std::map<std::string, tesseract_dawg::Dawg> * dawgs = nullptr,
                                               const std::vector<std::string> * tokens = nullptr,
                                               bool dawg_prefix_score = false) {
    std::vector<ctc_beam_state> beam(1);
    beam[0].p_blank = 0.0f;
    for (int t = 0; t < timesteps; ++t) {
        std::vector<ctc_beam_state> next;
        const float * probs = logits.data() + t * classes;
        auto find_or_add = [&](const std::vector<int> & prefix) -> ctc_beam_state & {
            for (auto & state : next) {
                if (state.prefix == prefix) return state;
            }
            next.push_back({ prefix });
            return next.back();
        };

        for (const auto & state : beam) {
            const float total = log_add(state.p_blank, state.p_nonblank);
            for (int c = 0; c < classes; ++c) {
                const float lp = logf(std::max(probs[c], 1.0e-30f));
                if (c == blank) {
                    auto & dst = find_or_add(state.prefix);
                    dst.p_blank = beam_add(dst.p_blank, total + lp, viterbi);
                } else if (!state.prefix.empty() && state.prefix.back() == c) {
                    auto & same = find_or_add(state.prefix);
                    same.p_nonblank = beam_add(same.p_nonblank, state.p_nonblank + lp, viterbi);
                    std::vector<int> extended = state.prefix;
                    extended.push_back(c);
                    if (recoder == nullptr || tesseract_recoder::prefix_legal(extended, *recoder, true)) {
                        auto & dst = find_or_add(extended);
                        dst.p_nonblank = beam_add(dst.p_nonblank, state.p_blank + lp, viterbi);
                    }
                } else {
                    std::vector<int> extended = state.prefix;
                    extended.push_back(c);
                    if (recoder == nullptr || tesseract_recoder::prefix_legal(extended, *recoder, true)) {
                        auto & dst = find_or_add(extended);
                        dst.p_nonblank = beam_add(dst.p_nonblank, total + lp, viterbi);
                    }
                }
            }
        }

        std::sort(
            next.begin(), next.end(),
            [viterbi, recoder, dawgs, tokens, dawg_prefix_score](const ctc_beam_state & a, const ctc_beam_state & b) {
                auto rank = [&](const ctc_beam_state & state) {
                    float score = beam_add(state.p_blank, state.p_nonblank, viterbi);
                    if (recoder && dawgs && tokens)
                        score += tesseract_dawg_score::word_bonus(state.prefix, *recoder, *tokens, *dawgs, false,
                                                                  dawg_prefix_score);
                    return score;
                };
                return rank(a) > rank(b);
            });
        if ((int)next.size() > beam_width) next.resize(beam_width);
        beam.swap(next);
    }

    if (beam.empty()) return {};
    if (recoder != nullptr) {
        const ctc_beam_state * best = nullptr;
        float best_rank = -INFINITY;
        for (const auto & state : beam) {
            if (tesseract_recoder::prefix_legal(state.prefix, *recoder, false)) {
                float rank = beam_add(state.p_blank, state.p_nonblank, viterbi);
                if (dawgs && tokens)
                    rank += tesseract_dawg_score::word_bonus(state.prefix, *recoder, *tokens, *dawgs, true,
                                                             dawg_prefix_score);
                if (!best || rank > best_rank) {
                    best = &state;
                    best_rank = rank;
                }
            }
        }
        if (best) {
            if (score_out) *score_out = beam_add(best->p_blank, best->p_nonblank, viterbi);
            return best->prefix;
        }
        return {};
    }
    if (score_out) *score_out = beam_add(beam.front().p_blank, beam.front().p_nonblank, viterbi);
    return beam.front().prefix;
}

// Keep the LSTM-boundary helper name used by the kernel contract while the
// reusable implementation remains in the recoder module.
static bool recode_classes_to_unichars(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                                       std::vector<int> & unichars, std::vector<int> & starts) {
    return tesseract_recoder::compose_classes(labels, codes, unichars, starts);
}

static void forward(tesseract_lstm_context * ctx,
                    const float * image, // (H, W) normalized
                    int H, int W) {
    ctx->captures.clear();
    const int conv_out = ctx->conv_out; // 16
    const bool int_mode = ctx->int_mode;

    std::vector<float> local_input;
    auto & input_values = ctx->reuse_scratch ? ctx->scratch_input : local_input;
    input_values.assign(image, image + H * W);
    if (int_mode) {
        for (float & value : input_values) value = quantize_int_input(value);
    }
    capture(ctx, "input_image", input_values.data(), H * W);

    // 1. Convolve 3×3 stacking (no learned weights) + FC+tanh
    // For each pixel (y,x): stack 3×3 neighborhood → 9 features
    // Then FC: out = tanh(W @ stacked + bias)
    std::vector<float> local_convolve, local_fc;
    auto & convolve_out = ctx->reuse_scratch ? ctx->scratch_convolve : local_convolve;
    auto & fc_out = ctx->reuse_scratch ? ctx->scratch_fc : local_fc;
    convolve_out.resize((size_t)H * W * 9);
    fc_out.resize((size_t)H * W * conv_out);
    uint64_t rng_seed = (uint64_t)((int64_t)ctx->sample_iteration * 0x10000001LL);
    auto random_int = [&]() -> int32_t {
        rng_seed = rng_seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return (int32_t)(rng_seed >> 33);
    };
    auto random_signed = [&](float range) -> float {
        return range * (2.0f * (float)random_int() / 2147483647.0f - 1.0f);
    };
    random_int(); // LSTMRecognizer::SetRandomSeed discards one value.
    {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = 0;
                float * stacked = convolve_out.data() + (y * W + x) * 9;
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int sx = x + dx, sy = y + dy;
                        if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
                            stacked[idx++] = input_values[sy * W + sx];
                        } else if (int_mode) {
                            stacked[idx++] = (float)tesseract_round_int(random_signed(127.0f)) / 127.0f;
                        } else {
                            stacked[idx++] = random_signed(1.0f);
                        }
                    }
                }
                // FC: tanh(W @ stacked + bias)
                float * dst = fc_out.data() + (y * W + x) * conv_out;
                for (int o = 0; o < conv_out; o++) {
                    float val = ctx->conv_b[o];
                    const float * w_row = ctx->conv_w.data() + o * 9;
                    if (int_mode) {
                        val = int8_row_dot(w_row, 9, ctx->conv_b[o], stacked);
                    } else {
                        for (int j = 0; j < 9; j++) val += w_row[j] * stacked[j];
                    }
                    dst[o] = tesseract_tanh(val);
                    if (int_mode) dst[o] = quantize_int_activation(dst[o]);
                }
            }
        }
    }

    capture(ctx, "after_convolve", convolve_out.data(), convolve_out.size());
    capture(ctx, "after_conv_fc", fc_out.data(), fc_out.size());

    // 2. MaxPool 3×3
    int H2 = H / 3, W2 = W / 3;
    std::vector<float> local_pool;
    auto & mp_out = ctx->reuse_scratch ? ctx->scratch_pool : local_pool;
    mp_out.assign((size_t)H2 * W2 * conv_out, -1e30f);
    for (int y = 0; y < H2; y++) {
        for (int x = 0; x < W2; x++) {
            float * dst = mp_out.data() + (y * W2 + x) * conv_out;
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    int sy = y * 3 + dy, sx = x * 3 + dx;
                    if (sy < H && sx < W) {
                        const float * src = fc_out.data() + (sy * W + sx) * conv_out;
                        for (int c = 0; c < conv_out; c++) dst[c] = std::max(dst[c], src[c]);
                    }
                }
            }
        }
    }

    capture(ctx, "after_maxpool", mp_out.data(), mp_out.size());

    // 3. XYTranspose + SummLSTM
    // Transpose: (H2, W2, C) → (W2, H2, C)
    std::vector<float> local_transposed;
    auto & transposed = ctx->reuse_scratch ? ctx->scratch_transposed : local_transposed;
    transposed.resize((size_t)H2 * W2 * conv_out);
    for (int y = 0; y < H2; y++)
        for (int x = 0; x < W2; x++)
            memcpy(transposed.data() + (x * H2 + y) * conv_out, mp_out.data() + (y * W2 + x) * conv_out,
                   conv_out * sizeof(float));

    // Find SummLSTM layer (first one, type "y_sum")
    int lstm_idx = 0;
    assert(lstm_idx < ctx->num_lstm_layers);
    const auto & lw0 = ctx->lstm[lstm_idx];
    int ns0 = lw0.ns;

    std::vector<float> local_seq_a, local_seq_b;
    auto & seq_a = ctx->reuse_scratch ? ctx->scratch_seq_a : local_seq_a;
    auto & seq_b = ctx->reuse_scratch ? ctx->scratch_seq_b : local_seq_b;
    seq_a.resize((size_t)W2 * ns0);
    summ_lstm_forward(transposed.data(), seq_a.data(), W2, H2, conv_out, ns0, lw0.W_ih.data(), lw0.W_hh.data(),
                      lw0.bias.data(), int_mode, ctx->cache_int ? &lw0 : nullptr,
                      ctx->reuse_scratch ? &ctx->scratch_lstm : nullptr);
    lstm_idx++;
    capture(ctx, "after_lstm_0", seq_a.data(), seq_a.size());

    // 4. Remaining LSTM layers (1-D over the time axis = W2)
    int T = W2;
    std::vector<float> * cur_seq = &seq_a;
    std::vector<float> * next_seq = &seq_b;
    int cur_dim = ns0;

    while (lstm_idx < ctx->num_lstm_layers) {
        const auto & lw = ctx->lstm[lstm_idx];
        bool rev = (lstm_idx < (int)ctx->lstm_types.size() && ctx->lstm_types[lstm_idx] == "rev");

        next_seq->resize((size_t)T * lw.ns);
        lstm_forward(cur_seq->data(), next_seq->data(), T, cur_dim, lw.ns, lw.W_ih.data(), lw.W_hh.data(),
                     lw.bias.data(), rev, int_mode, ctx->cache_int ? &lw : nullptr,
                     ctx->reuse_scratch ? &ctx->scratch_lstm : nullptr);

        std::swap(cur_seq, next_seq);
        cur_dim = lw.ns;
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "after_lstm_%d", lstm_idx);
            capture(ctx, buf, cur_seq->data(), cur_seq->size());
        }
        lstm_idx++;
    }

    // 5. Softmax output
    int n_classes = ctx->num_classes;
    std::vector<float> local_logits;
    auto & logits = ctx->reuse_scratch ? ctx->scratch_logits : local_logits;
    logits.resize((size_t)T * n_classes);
    for (int t = 0; t < T; t++) {
        const float * x = cur_seq->data() + t * cur_dim;
        float * dst = logits.data() + t * n_classes;
        float max_val = -1e30f;
        for (int c = 0; c < n_classes; c++) {
            float val = ctx->out_b[c];
            const float * w_row = ctx->out_w.data() + c * cur_dim;
            if (int_mode) {
                val = int8_row_dot(w_row, cur_dim, ctx->out_b[c], x);
            } else {
                for (int j = 0; j < cur_dim; j++) val += w_row[j] * x[j];
            }
            dst[c] = val;
            if (val > max_val) max_val = val;
        }
        // Softmax
        float sum = 0.0f;
        for (int c = 0; c < n_classes; c++) {
            dst[c] = expf(dst[c] - max_val);
            sum += dst[c];
        }
        for (int c = 0; c < n_classes; c++) dst[c] /= sum;
    }

    capture(ctx, "logits", logits.data(), logits.size());

    // 6. CTC decode. Beam search is opt-in for parity experiments; production
    // remains greedy until the decoder is validated against Tesseract's
    // recode beam and dictionary behavior.
    ctx->result_buf.clear();
    ctx->char_confs.clear();
    ctx->sequence_confidence = 0.0f;
    ctx->word_confidence = 0.0f;
    std::vector<int> labels;
    const char * beam_env = std::getenv("CRISPEMBED_TESSERACT_BEAM_WIDTH");
    const int beam_width = beam_env ? std::max(1, atoi(beam_env)) : 1;
    const char * recode_env = std::getenv("CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH");
    const int recode_width = recode_env ? std::max(1, atoi(recode_env)) : 1;
    const bool dawg_score =
        recode_width > 1 && std::getenv("CRISPEMBED_TESSERACT_DAWG_SCORE") != nullptr && !ctx->dawgs.empty();
    const bool dawg_prefix_score = dawg_score && std::getenv("CRISPEMBED_TESSERACT_DAWG_PREFIX_SCORE") != nullptr;
    bool beam_decoded = false;
    float beam_log_score = -INFINITY;
    std::vector<float> greedy_label_confs;
    if (recode_width > 1) {
        labels = ctc_prefix_beam_decode(logits, T, n_classes, ctx->null_char, recode_width, false, &ctx->recoder_codes,
                                        &beam_log_score, dawg_score ? &ctx->dawgs : nullptr,
                                        dawg_score ? &ctx->tokens : nullptr, dawg_prefix_score);
        beam_decoded = true;
    } else if (beam_width > 1) {
        labels =
            ctc_prefix_beam_decode(logits, T, n_classes, ctx->null_char, beam_width, false, nullptr, &beam_log_score);
        beam_decoded = true;
    } else {
        int prev = -1;
        for (int t = 0; t < T; t++) {
            const float * probs = logits.data() + t * n_classes;
            int best = 0;
            float best_p = probs[0];
            for (int c = 1; c < n_classes; c++) {
                if (probs[c] > best_p) {
                    best = c;
                    best_p = probs[c];
                }
            }
            labels.push_back(best);
            greedy_label_confs.push_back(best_p);
        }
    }
    std::vector<int> collapsed_labels;
    std::vector<float> collapsed_confs;
    if (beam_decoded) {
        collapsed_labels = labels;
    } else {
        int prev = -1;
        for (size_t i = 0; i < labels.size(); ++i) {
            const int best = labels[i];
            if (best != ctx->null_char && best != prev) {
                collapsed_labels.push_back(best);
                collapsed_confs.push_back(i < greedy_label_confs.size() ? greedy_label_confs[i] : 0.0f);
            }
            prev = best;
        }
    }

    const bool compose_recoder =
        !ctx->recoder_codes.empty() && std::getenv("CRISPEMBED_TESSERACT_RECODE_COMPOSE") != nullptr;
    std::vector<int> composed_uids, composed_starts;
    bool composed = compose_recoder &&
                    recode_classes_to_unichars(collapsed_labels, ctx->recoder_codes, composed_uids, composed_starts);
    if (compose_recoder && !composed) {
        // Keep valid multi-class segments visible around an unmapped class in
        // diagnostic mode. The production/default path remains unchanged.
        composed = tesseract_recoder::compose_classes_partial(collapsed_labels, ctx->recoder_codes, composed_uids,
                                                              composed_starts);
    }
    if (composed) {
        for (size_t i = 0; i < composed_uids.size(); ++i) {
            const int uid = composed_uids[i];
            if (uid < 0 || uid >= (int)ctx->tokens.size()) {
                // Preserve an unmapped composed class in diagnostics. Dropping
                // it silently makes native output look shorter than Python's
                // reference and hides recoder coverage gaps.
                ctx->result_buf += "<class>";
                continue;
            }
            ctx->result_buf += ctx->tokens[uid];
            if (!beam_decoded) {
                const int begin = composed_starts[i];
                const int end = i + 1 < composed_starts.size() ? composed_starts[i + 1] : (int)collapsed_confs.size();
                float confidence = 1.0f;
                for (int j = begin; j < end && j < (int)collapsed_confs.size(); ++j)
                    confidence = std::min(confidence, collapsed_confs[j]);
                ctx->char_confs.push_back(confidence);
            }
        }
    } else {
        for (size_t i = 0; i < collapsed_labels.size(); ++i) {
            const int best = collapsed_labels[i];
            int uid = -1;
            if (best >= 0 && best < (int)ctx->output_to_unichar.size()) uid = ctx->output_to_unichar[best];
            if (uid >= 0 && uid < (int)ctx->tokens.size()) {
                ctx->result_buf += ctx->tokens[uid];
                if (!beam_decoded) ctx->char_confs.push_back(collapsed_confs[i]);
            } else {
                // Keep unmapped output classes visible instead of silently
                // dropping them or exposing implementation-specific IDs.
                ctx->result_buf += "<class>";
            }
        }
    }
    if (beam_decoded && std::isfinite(beam_log_score)) {
        // A CTC beam score is a sequence probability, not a character
        // probability. Normalize by timesteps to avoid length-dependent
        // underflow and expose it separately from char_confs.
        ctx->sequence_confidence = std::clamp(expf(beam_log_score / std::max(1, T)), 0.0f, 1.0f);
    } else if (!greedy_label_confs.empty()) {
        double sum = 0.0;
        for (float p : greedy_label_confs) sum += p;
        ctx->sequence_confidence = (float)(sum / greedy_label_confs.size());
        // Tesseract word certainty is the minimum over emitted characters,
        // not over blank/repeated CTC timesteps. char_confs contains exactly
        // the mapped, collapsed greedy output symbols.
        if (!ctx->char_confs.empty()) {
            float min_certainty = 0.0f;
            for (float p : ctx->char_confs) min_certainty = std::min(min_certainty, std::log(std::max(p, 1.0e-20f)));
            ctx->word_confidence = std::clamp(1.0f + 0.05f * min_certainty, 0.0f, 1.0f);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

tesseract_lstm_context * tesseract_lstm_init(const char * model_path, int n_threads) {
    (void)n_threads; // all CPU-side, single-threaded for now

    auto * ctx = new tesseract_lstm_context();
    if (!load_model(ctx, model_path)) {
        delete ctx;
        return nullptr;
    }
    ctx->bench = core_env::on("CRISPEMBED_TESSERACT_BENCH");
    ctx->cache_int = (std::getenv("CRISPEMBED_TESSERACT_DISABLE_INT_CACHE") == nullptr);
    ctx->reuse_scratch = (std::getenv("CRISPEMBED_TESSERACT_REUSE_SCRATCH") != nullptr);
    return ctx;
}

void tesseract_lstm_free(tesseract_lstm_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

const char * tesseract_lstm_recognize(tesseract_lstm_context * ctx, const uint8_t * pixels, int width, int height,
                                      int * out_len) {
    if (!ctx || !pixels || width <= 0 || height <= 0) {
        if (out_len) *out_len = 0;
        return "";
    }

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Tesseract's ImageData::PreScale calls Leptonica pixScale. For the usual
    // upscaling path this is pixScaleGrayLI: top-left-corner linear
    // interpolation on a 1/16 fixed-point grid, with edge replication.
    auto t0 = std::chrono::steady_clock::now();
    const uint8_t * src = pixels;
    int W = width, H = height;
    std::vector<uint8_t> resized;
    if (ctx->input_height > 0 && height != ctx->input_height) {
        const int dh = ctx->input_height;
        int dw = (int)std::lround((double)width * dh / (double)height);
        if (dw < 1) dw = 1;
        resized.resize((size_t)dw * dh);
        const float scx = 16.0f * (float)width / (float)dw;
        const float scy = 16.0f * (float)height / (float)dh;
        for (int y = 0; y < dh; y++) {
            const int ypm = (int)(scy * (float)y);
            const int yp = ypm >> 4;
            const int yf = ypm & 0x0f;
            const int y1 = std::min(yp + 1, height - 1);
            for (int x = 0; x < dw; x++) {
                const int xpm = (int)(scx * (float)x);
                const int xp = xpm >> 4;
                const int xf = xpm & 0x0f;
                const int x1 = std::min(xp + 1, width - 1);
                const int v00 = pixels[yp * width + xp];
                const int v10 = pixels[yp * width + x1];
                const int v01 = pixels[y1 * width + xp];
                const int v11 = pixels[y1 * width + x1];
                resized[(size_t)y * dw + x] = (uint8_t)(((16 - xf) * (16 - yf) * v00 + xf * (16 - yf) * v10 +
                                                         (16 - xf) * yf * v01 + xf * yf * v11 + 128) /
                                                        256);
            }
        }
        src = resized.data();
        W = dw;
        H = dh;
    }

    if (std::getenv("TESSERACT_DIFF_DEBUG")) {
        uint8_t lo = src[0], hi = src[0];
        int lo_i = 0;
        for (int i = 1; i < W * H; ++i) {
            if (src[i] < lo) {
                lo = src[i];
                lo_i = i;
            }
            hi = std::max(hi, src[i]);
        }
        fprintf(stderr, "C++ resized debug: %dx%d min=%u max=%u first=%u %u %u %u\n", W, H, (unsigned)lo, (unsigned)hi,
                (unsigned)src[0], (unsigned)src[1], (unsigned)src[2], (unsigned)src[3]);
        fprintf(stderr, "C++ resized min location: x=%d y=%d\n", lo_i % W, lo_i / W);
    }

    // Normalize image (Tesseract-style ComputeBlackWhite + SetPixel)
    std::vector<float> normalized((size_t)W * H);
    normalize_image(src, W, H, normalized.data());
    if (bench)
        fprintf(stderr, "[tesseract-bench] preprocess: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    // Run forward pass (LSTM layers + softmax)
    t0 = std::chrono::steady_clock::now();
    forward(ctx, normalized.data(), H, W);
    if (bench)
        fprintf(stderr, "[tesseract-bench] LSTM layers: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    if (bench)
        fprintf(stderr, "[tesseract-bench] total: %.1f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float * tesseract_lstm_confidences(const tesseract_lstm_context * ctx, int * n_chars) {
    if (!ctx || ctx->char_confs.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confs.size();
    return ctx->char_confs.data();
}

float tesseract_lstm_mean_confidence(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->sequence_confidence : 0.0f;
}

float tesseract_lstm_word_confidence(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->word_confidence : 0.0f;
}

int tesseract_lstm_input_height(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->input_height : 0;
}

int tesseract_lstm_num_classes(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->num_classes : 0;
}

int tesseract_lstm_dawg_component_count(const tesseract_lstm_context * ctx) {
    return ctx ? (int)ctx->dawg_components.size() : 0;
}

int tesseract_lstm_dawg_contains(const tesseract_lstm_context * ctx, const char * component, const int * ids,
                                 size_t count) {
    if (!ctx || !component) return 0;
    const auto it = ctx->dawg_contexts.find(component);
    return it == ctx->dawg_contexts.end() ? 0 : tesseract_dawg_context_contains(it->second, ids, count);
}

int tesseract_lstm_dawg_has_prefix(const tesseract_lstm_context * ctx, const char * component, const int * ids,
                                   size_t count) {
    if (!ctx || !component) return 0;
    const auto it = ctx->dawg_contexts.find(component);
    return it == ctx->dawg_contexts.end() ? 0 : tesseract_dawg_context_has_prefix(it->second, ids, count);
}

int tesseract_lstm_dawg_state(const tesseract_lstm_context * ctx, const char * component, const int * ids,
                              size_t count) {
    if (!ctx || !component) return TESSERACT_DAWG_INVALID_PREFIX;
    const auto it = ctx->dawg_contexts.find(component);
    return it == ctx->dawg_contexts.end() ? TESSERACT_DAWG_INVALID_PREFIX
                                          : tesseract_dawg_context_state(it->second, ids, count);
}

const char * tesseract_lstm_vgsl_spec(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->vgsl_spec.c_str() : "";
}

int tesseract_lstm_dawg_count(const tesseract_lstm_context * ctx) {
    return ctx ? (int)ctx->dawgs.size() : 0;
}

int tesseract_lstm_dawg_matches(const tesseract_lstm_context * ctx, const char * name, const int * unichars,
                                int n_unichars, int complete) {
    if (!ctx || !name || n_unichars < 0 || (n_unichars > 0 && !unichars)) return -1;
    const auto it = ctx->dawgs.find(name);
    if (it == ctx->dawgs.end()) return -1;
    const std::vector<int> ids =
        n_unichars > 0 ? std::vector<int>(unichars, unichars + n_unichars) : std::vector<int>();
    return tesseract_dawg::prefix_matches(it->second, ids, complete != 0) ? 1 : 0;
}

int tesseract_lstm_dawg_matches_utf8(const tesseract_lstm_context * ctx, const char * name, const char * text,
                                     int complete) {
    if (!ctx || !name || !text) return -1;
    if (ctx->dawgs.find(name) == ctx->dawgs.end()) return -1;
    const std::string input(text);
    std::vector<int> previous(input.size() + 1, -1);
    std::vector<int> previous_uid(input.size() + 1, -1);
    previous[0] = 0;
    for (size_t pos = 0; pos < input.size(); ++pos) {
        if (previous[pos] < 0) continue;
        for (int uid = 0; uid < (int)ctx->tokens.size(); ++uid) {
            const std::string & token = ctx->tokens[uid];
            if (token.empty() || pos + token.size() > input.size()) continue;
            if (!std::equal(token.begin(), token.end(), input.begin() + pos)) continue;
            const size_t end = pos + token.size();
            if (previous[end] < 0) {
                previous[end] = (int)pos;
                previous_uid[end] = uid;
            }
        }
    }
    if (previous[input.size()] < 0) return 0;
    std::vector<int> ids;
    for (size_t end = input.size(); end > 0;) {
        const int uid = previous_uid[end];
        const int begin = previous[end];
        if (uid < 0 || begin < 0) return 0;
        ids.push_back(uid);
        end = (size_t)begin;
    }
    std::reverse(ids.begin(), ids.end());
    return tesseract_lstm_dawg_matches(ctx, name, ids.data(), (int)ids.size(), complete);
}

void tesseract_lstm_set_dump(tesseract_lstm_context * ctx, int enabled) {
    if (ctx) ctx->dump_mode = (enabled != 0);
}

const float * tesseract_lstm_get_capture(const tesseract_lstm_context * ctx, const char * name, int * n_elem) {
    if (!ctx || !name) {
        if (n_elem) *n_elem = 0;
        return nullptr;
    }
    auto it = ctx->captures.find(name);
    if (it == ctx->captures.end()) {
        if (n_elem) *n_elem = 0;
        return nullptr;
    }
    if (n_elem) *n_elem = (int)it->second.size();
    return it->second.data();
}
