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

#include "ggml.h"
#include "ggml-backend.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"

#include <algorithm>
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
    std::vector<float> W_ih;   // (4*ns, ni)
    std::vector<float> W_hh;   // (4*ns, ns)
    std::vector<float> bias;   // (4*ns,)
    int ni;
    int ns;  // hidden size
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct tesseract_lstm_context {
    // Hyperparameters
    int input_height;
    int num_classes;
    int null_char;
    int num_lstm_layers;
    std::string vgsl_spec;

    // Conv FC weights
    std::vector<float> conv_w;   // (16, 9)  — [out][in] row-major
    std::vector<float> conv_b;   // (16,)
    int conv_out;                // 16

    // LSTM layers
    std::vector<lstm_weights> lstm;

    // Output FC weights
    std::vector<float> out_w;    // (n_classes, last_lstm_ns)
    std::vector<float> out_b;    // (n_classes,)

    // Per-LSTM metadata
    std::vector<std::string> lstm_types;  // "y_sum", "fwd", "rev"

    // Reverse recoder: output_class → unichar_id (-1 if unmapped)
    std::vector<int> output_to_unichar;

    // Unicharset tokens
    std::vector<std::string> tokens;

    // Inference results
    std::string result_buf;
    std::vector<float> char_confs;

    // Diff mode: capture per-stage intermediates
    bool dump_mode = false;
    std::map<std::string, std::vector<float>> captures;

    bool bench = false;

    // GGUF loader state
    core_gguf::WeightLoad wl;
    // Dequantized weight cache
    std::map<const void *, std::vector<float>> dequant_cache;
};

// ---------------------------------------------------------------------------
// Tensor dequantization helper
// ---------------------------------------------------------------------------

static const float * tensor_f32(tesseract_lstm_context * ctx, struct ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return (const float *)t->data;
    }
    auto it = ctx->dequant_cache.find(t->data);
    if (it != ctx->dequant_cache.end()) {
        return it->second.data();
    }
    const int64_t n = ggml_nelements(t);
    auto & buf = ctx->dequant_cache[t->data];
    buf.resize(n);
    const auto * traits = ggml_get_type_traits(t->type);
    if (traits->to_float) {
        traits->to_float(t->data, buf.data(), n);
    } else {
        fprintf(stderr, "tesseract_lstm: cannot dequantize type %d\n", t->type);
        std::fill(buf.begin(), buf.end(), 0.0f);
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
    ctx->num_classes   = (int)core_gguf::kv_u32(meta, "tesseract_lstm.num_classes", 111);
    ctx->null_char     = (int)core_gguf::kv_u32(meta, "tesseract_lstm.null_char", 110);
    ctx->num_lstm_layers = (int)core_gguf::kv_u32(meta, "tesseract_lstm.num_lstm_layers", 4);
    ctx->vgsl_spec     = core_gguf::kv_str(meta, "tesseract_lstm.vgsl_spec", "");

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

    // LSTM types
    ctx->lstm_types = core_gguf::kv_str_array(meta, "tesseract_lstm.lstm_types");

    core_gguf::free_metadata(meta);

    // Pass 2: weights
    ggml_backend_t backend = ggml_backend_init_best();
    if (!core_gguf::load_weights(path, backend, "tesseract_lstm", ctx->wl)) {
        ggml_backend_free(backend);
        return false;
    }
    ggml_backend_free(backend);

    const auto & T = ctx->wl.tensors;
    auto req = [&](const char * name) -> ggml_tensor * {
        return core_gguf::require(T, name, "tesseract_lstm");
    };

    // Conv FC
    ggml_tensor * cw = req("conv.weight");
    ggml_tensor * cb = req("conv.bias");
    if (!cw || !cb) return false;
    const float * cw_f = tensor_f32(ctx, cw);
    const float * cb_f = tensor_f32(ctx, cb);
    int conv_ni = (int)cw->ne[0];  // 9
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
        const float * b_f   = tensor_f32(ctx, b);

        int gate_size = 4 * lw.ns;
        lw.W_ih.assign(wih_f, wih_f + gate_size * lw.ni);
        lw.W_hh.assign(whh_f, whh_f + gate_size * lw.ns);
        lw.bias.assign(b_f, b_f + gate_size);
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

    fprintf(stderr, "tesseract_lstm: loaded %s (%d LSTM layers, %d classes, height=%d)\n",
            ctx->vgsl_spec.c_str(), ctx->num_lstm_layers, ctx->num_classes, ctx->input_height);

    return true;
}

// ---------------------------------------------------------------------------
// LSTM forward (single direction)
// ---------------------------------------------------------------------------

static void lstm_forward(
    const float * input,   // (T, ni)
    float * output,        // (T, ns)
    int T, int ni, int ns,
    const float * W_ih,    // (4*ns, ni)
    const float * W_hh,    // (4*ns, ns)
    const float * bias,    // (4*ns,)
    bool reverse)
{
    // Gate order (PyTorch): i, f, g, o
    const int gs = 4 * ns;
    std::vector<float> h(ns, 0.0f);
    std::vector<float> c(ns, 0.0f);
    std::vector<float> gates(gs);
    std::vector<float> hh_out(gs);  // W_hh @ h scratch (hoisted out of step loop)

    for (int step = 0; step < T; step++) {
        int t = reverse ? (T - 1 - step) : step;
        const float * xt = input + t * ni;

        // gates = W_ih @ x + bias  (SIMD GEMV via linear_cpu)
        core_cpu::linear_cpu(xt, gates.data(), ni, gs, W_ih, bias);
        // gates += W_hh @ h
        core_cpu::linear_cpu(h.data(), hh_out.data(), ns, gs, W_hh, nullptr);
        for (int g = 0; g < gs; g++) gates[g] += hh_out[g];

        for (int j = 0; j < ns; j++) {
            float i_gate = 1.0f / (1.0f + expf(-gates[0*ns + j]));
            float f_gate = 1.0f / (1.0f + expf(-gates[1*ns + j]));
            float g_gate = tanhf(gates[2*ns + j]);
            float o_gate = 1.0f / (1.0f + expf(-gates[3*ns + j]));

            c[j] = f_gate * c[j] + i_gate * g_gate;
            if (c[j] > 100.0f) c[j] = 100.0f;
            if (c[j] < -100.0f) c[j] = -100.0f;
            h[j] = o_gate * tanhf(c[j]);
        }

        memcpy(output + t * ns, h.data(), ns * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// SummLSTM: run LSTM over height dimension, keep last hidden state per column
// ---------------------------------------------------------------------------

static void summ_lstm_forward(
    const float * input,   // (height, width, channels) — row-major after XYTranspose
    float * output,        // (width, ns) — one hidden state per column
    int height, int width, int channels, int ns,
    const float * W_ih,    // (4*ns, channels)
    const float * W_hh,    // (4*ns, ns)
    const float * bias)    // (4*ns,)
{
    // After XYTranspose: height = original_width, width = original_height
    // For each row (height position), run LSTM across the width (original height).
    // State resets at each row boundary.
    const int gs = 4 * ns;
    std::vector<float> h(ns);
    std::vector<float> c(ns);
    std::vector<float> gates(gs);

    for (int row = 0; row < height; row++) {
        // Reset state per row
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);

        for (int col = 0; col < width; col++) {
            const float * xt = input + (row * width + col) * channels;

            for (int g = 0; g < gs; g++) {
                float val = bias[g];
                const float * wih_row = W_ih + g * channels;
                for (int j = 0; j < channels; j++)
                    val += wih_row[j] * xt[j];
                const float * whh_row = W_hh + g * ns;
                for (int j = 0; j < ns; j++)
                    val += whh_row[j] * h[j];
                gates[g] = val;
            }

            for (int j = 0; j < ns; j++) {
                float i_gate = 1.0f / (1.0f + expf(-gates[0*ns + j]));
                float f_gate = 1.0f / (1.0f + expf(-gates[1*ns + j]));
                float g_gate = tanhf(gates[2*ns + j]);
                float o_gate = 1.0f / (1.0f + expf(-gates[3*ns + j]));

                c[j] = f_gate * c[j] + i_gate * g_gate;
                if (c[j] > 100.0f) c[j] = 100.0f;
                if (c[j] < -100.0f) c[j] = -100.0f;
                h[j] = o_gate * tanhf(c[j]);
            }
        }

        // Keep last hidden state
        memcpy(output + row * ns, h.data(), ns * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Image normalization (matches Tesseract's ComputeBlackWhite + SetPixel)
// ---------------------------------------------------------------------------

static void normalize_image(
    const uint8_t * pixels, int width, int height,
    float * out)  // (height, width)
{
    // ComputeBlackWhite: scan middle row for local min/max
    int mid_y = height / 2;
    std::vector<float> mins, maxes;

    if (width >= 3) {
        float prev = (float)pixels[mid_y * width + 0];
        float curr = (float)pixels[mid_y * width + 1];
        for (int x = 1; x + 1 < width; x++) {
            float next = (float)pixels[mid_y * width + x + 1];
            if ((curr < prev && curr <= next) || (curr <= prev && curr < next))
                mins.push_back(curr);
            if ((curr > prev && curr >= next) || (curr >= prev && curr > next))
                maxes.push_back(curr);
            prev = curr;
            curr = next;
        }
    }
    if (mins.empty()) mins.push_back(0.0f);
    if (maxes.empty()) maxes.push_back(255.0f);

    // 25th percentile of mins, 75th percentile of maxes
    std::sort(mins.begin(), mins.end());
    std::sort(maxes.begin(), maxes.end());
    float black = mins[(int)(mins.size() * 0.25f)];
    float white = maxes[(int)(maxes.size() * 0.75f)];
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
static void capture(tesseract_lstm_context * ctx, const char * name,
                    const float * data, size_t n) {
    if (ctx->dump_mode)
        ctx->captures[name].assign(data, data + n);
}

static void forward(tesseract_lstm_context * ctx,
                    const float * image,  // (H, W) normalized
                    int H, int W)
{
    ctx->captures.clear();
    const int conv_out = ctx->conv_out;  // 16

    capture(ctx, "input_image", image, H * W);

    // 1. Convolve 3×3 stacking (no learned weights) + FC+tanh
    // For each pixel (y,x): stack 3×3 neighborhood → 9 features
    // Then FC: out = tanh(W @ stacked + bias)
    std::vector<float> fc_out(H * W * conv_out);
    {
        std::vector<float> stacked(9);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int idx = 0;
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int sx = x + dx, sy = y + dy;
                        stacked[idx++] = (sx >= 0 && sx < W && sy >= 0 && sy < H)
                                         ? image[sy * W + sx] : 0.0f;
                    }
                }
                // FC: tanh(W @ stacked + bias)
                float * dst = fc_out.data() + (y * W + x) * conv_out;
                for (int o = 0; o < conv_out; o++) {
                    float val = ctx->conv_b[o];
                    const float * w_row = ctx->conv_w.data() + o * 9;
                    for (int j = 0; j < 9; j++)
                        val += w_row[j] * stacked[j];
                    dst[o] = tanhf(val);
                }
            }
        }
    }

    capture(ctx, "after_conv_fc", fc_out.data(), fc_out.size());

    // 2. MaxPool 3×3
    int H2 = H / 3, W2 = W / 3;
    std::vector<float> mp_out(H2 * W2 * conv_out, -1e30f);
    for (int y = 0; y < H2; y++) {
        for (int x = 0; x < W2; x++) {
            float * dst = mp_out.data() + (y * W2 + x) * conv_out;
            for (int dy = 0; dy < 3; dy++) {
                for (int dx = 0; dx < 3; dx++) {
                    int sy = y*3+dy, sx = x*3+dx;
                    if (sy < H && sx < W) {
                        const float * src = fc_out.data() + (sy * W + sx) * conv_out;
                        for (int c = 0; c < conv_out; c++)
                            dst[c] = std::max(dst[c], src[c]);
                    }
                }
            }
        }
    }

    capture(ctx, "after_maxpool", mp_out.data(), mp_out.size());

    // 3. XYTranspose + SummLSTM
    // Transpose: (H2, W2, C) → (W2, H2, C)
    std::vector<float> transposed(H2 * W2 * conv_out);
    for (int y = 0; y < H2; y++)
        for (int x = 0; x < W2; x++)
            memcpy(transposed.data() + (x * H2 + y) * conv_out,
                   mp_out.data() + (y * W2 + x) * conv_out,
                   conv_out * sizeof(float));

    // Find SummLSTM layer (first one, type "y_sum")
    int lstm_idx = 0;
    assert(lstm_idx < ctx->num_lstm_layers);
    const auto & lw0 = ctx->lstm[lstm_idx];
    int ns0 = lw0.ns;

    std::vector<float> summ_out(W2 * ns0);
    summ_lstm_forward(transposed.data(), summ_out.data(),
                      W2, H2, conv_out, ns0,
                      lw0.W_ih.data(), lw0.W_hh.data(), lw0.bias.data());
    lstm_idx++;
    capture(ctx, "after_lstm_0", summ_out.data(), summ_out.size());

    // 4. Remaining LSTM layers (1-D over the time axis = W2)
    int T = W2;
    std::vector<float> cur_seq = std::move(summ_out);  // (T, ns0)
    int cur_dim = ns0;

    while (lstm_idx < ctx->num_lstm_layers) {
        const auto & lw = ctx->lstm[lstm_idx];
        bool rev = (lstm_idx < (int)ctx->lstm_types.size() &&
                    ctx->lstm_types[lstm_idx] == "rev");

        std::vector<float> next_seq(T * lw.ns);
        lstm_forward(cur_seq.data(), next_seq.data(),
                     T, cur_dim, lw.ns,
                     lw.W_ih.data(), lw.W_hh.data(), lw.bias.data(), rev);

        cur_seq = std::move(next_seq);
        cur_dim = lw.ns;
        {
            char buf[32];
            snprintf(buf, sizeof(buf), "after_lstm_%d", lstm_idx);
            capture(ctx, buf, cur_seq.data(), cur_seq.size());
        }
        lstm_idx++;
    }

    // 5. Softmax output
    int n_classes = ctx->num_classes;
    std::vector<float> logits(T * n_classes);
    for (int t = 0; t < T; t++) {
        const float * x = cur_seq.data() + t * cur_dim;
        float * dst = logits.data() + t * n_classes;
        float max_val = -1e30f;
        for (int c = 0; c < n_classes; c++) {
            float val = ctx->out_b[c];
            const float * w_row = ctx->out_w.data() + c * cur_dim;
            for (int j = 0; j < cur_dim; j++)
                val += w_row[j] * x[j];
            dst[c] = val;
            if (val > max_val) max_val = val;
        }
        // Softmax
        float sum = 0.0f;
        for (int c = 0; c < n_classes; c++) {
            dst[c] = expf(dst[c] - max_val);
            sum += dst[c];
        }
        for (int c = 0; c < n_classes; c++)
            dst[c] /= sum;
    }

    capture(ctx, "logits", logits.data(), logits.size());

    // 6. CTC greedy decode
    ctx->result_buf.clear();
    ctx->char_confs.clear();
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
        if (best != ctx->null_char && best != prev) {
            // Map output class → unichar via reverse recoder
            int uid = -1;
            if (best < (int)ctx->output_to_unichar.size())
                uid = ctx->output_to_unichar[best];
            if (uid >= 0 && uid < (int)ctx->tokens.size()) {
                ctx->result_buf += ctx->tokens[uid];
                ctx->char_confs.push_back(best_p);
            }
        }
        prev = best;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

tesseract_lstm_context * tesseract_lstm_init(const char * model_path, int n_threads) {
    (void)n_threads;  // all CPU-side, single-threaded for now

    auto * ctx = new tesseract_lstm_context();
    if (!load_model(ctx, model_path)) {
        delete ctx;
        return nullptr;
    }
    ctx->bench = (std::getenv("CRISPEMBED_TESSERACT_BENCH") != nullptr);
    return ctx;
}

void tesseract_lstm_free(tesseract_lstm_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

const char * tesseract_lstm_recognize(
    tesseract_lstm_context * ctx,
    const uint8_t * pixels,
    int width, int height,
    int * out_len)
{
    if (!ctx || !pixels || width <= 0 || height <= 0) {
        if (out_len) *out_len = 0;
        return "";
    }

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Tesseract's LSTM is trained on lines normalized to a fixed height
    // (input_height, typically 36); the conv/maxpool + SummLSTM stack assumes
    // it (H2 = H/3). Resize the input line to input_height (bilinear,
    // aspect-preserving) before normalization — otherwise a differently-sized
    // line produces garbage. (recognize() is documented to do this.)
    auto t0 = std::chrono::steady_clock::now();
    const uint8_t * src = pixels;
    int W = width, H = height;
    std::vector<uint8_t> resized;
    if (ctx->input_height > 0 && height != ctx->input_height) {
        const int dh = ctx->input_height;
        int dw = (int)std::lround((double)width * dh / (double)height);
        if (dw < 1) dw = 1;
        resized.resize((size_t)dw * dh);
        for (int y = 0; y < dh; y++) {
            float sy = ((y + 0.5f) * height / dh) - 0.5f;
            int y0 = (int)std::floor(sy);
            float fy = sy - y0;
            int y0c = std::min(std::max(y0, 0), height - 1);
            int y1c = std::min(std::max(y0 + 1, 0), height - 1);
            for (int x = 0; x < dw; x++) {
                float sx = ((x + 0.5f) * width / dw) - 0.5f;
                int x0 = (int)std::floor(sx);
                float fx = sx - x0;
                int x0c = std::min(std::max(x0, 0), width - 1);
                int x1c = std::min(std::max(x0 + 1, 0), width - 1);
                float p00 = pixels[y0c * width + x0c];
                float p01 = pixels[y0c * width + x1c];
                float p10 = pixels[y1c * width + x0c];
                float p11 = pixels[y1c * width + x1c];
                float top = p00 + (p01 - p00) * fx;
                float bot = p10 + (p11 - p10) * fx;
                resized[(size_t)y * dw + x] = (uint8_t)std::lround(top + (bot - top) * fy);
            }
        }
        src = resized.data();
        W = dw;
        H = dh;
    }

    // Normalize image (Tesseract-style ComputeBlackWhite + SetPixel)
    std::vector<float> normalized((size_t)W * H);
    normalize_image(src, W, H, normalized.data());
    if (bench) fprintf(stderr, "[tesseract-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // Run forward pass (LSTM layers + softmax)
    t0 = std::chrono::steady_clock::now();
    forward(ctx, normalized.data(), H, W);
    if (bench) fprintf(stderr, "[tesseract-bench] LSTM layers: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[tesseract-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float * tesseract_lstm_confidences(
    const tesseract_lstm_context * ctx,
    int * n_chars)
{
    if (!ctx || ctx->char_confs.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confs.size();
    return ctx->char_confs.data();
}

int tesseract_lstm_input_height(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->input_height : 0;
}

int tesseract_lstm_num_classes(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->num_classes : 0;
}

const char * tesseract_lstm_vgsl_spec(const tesseract_lstm_context * ctx) {
    return ctx ? ctx->vgsl_spec.c_str() : "";
}

void tesseract_lstm_set_dump(tesseract_lstm_context * ctx, int enabled) {
    if (ctx) ctx->dump_mode = (enabled != 0);
}

const float * tesseract_lstm_get_capture(
    const tesseract_lstm_context * ctx,
    const char * name,
    int * n_elem)
{
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
