// bert_ner.cpp — Fixed-label NER: BERT encoder + Linear classifier + BIO decode.

#include "bert_ner.h"
#include "crispembed.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace bert_ner {

struct context {
    crispembed_context* enc = nullptr;  // existing encoder

    // Classifier head weights (loaded from GGUF separately)
    int n_labels = 0;
    std::vector<std::string> labels;        // [n_labels] label names
    std::vector<float> cls_weight;          // [n_labels * hidden_dim]
    std::vector<float> cls_bias;            // [n_labels]
    int hidden_dim = 0;

    // Last result storage
    std::vector<entity> last_entities;
};

bool load(context** out, const char* model_path, int n_threads) {
    if (!out || !model_path) return false;

    auto* ctx = new context;

    // Load encoder via existing CrispEmbed API.
    ctx->enc = crispembed_init(model_path, n_threads);
    if (!ctx->enc) {
        fprintf(stderr, "bert_ner: failed to load encoder from %s\n", model_path);
        delete ctx;
        return false;
    }

    ctx->hidden_dim = crispembed_get_hparams(ctx->enc)->n_embd;

    // Read NER classifier weights from the GGUF separately.
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        crispembed_free(ctx->enc);
        delete ctx;
        return false;
    }

    ctx->n_labels = core_gguf::kv_u32(gctx, "ner.num_labels", 0);
    if (ctx->n_labels == 0) {
        fprintf(stderr, "bert_ner: no ner.num_labels in %s — not a NER model\n", model_path);
        core_gguf::free_metadata(gctx);
        crispembed_free(ctx->enc);
        delete ctx;
        return false;
    }

    // Read label names
    ctx->labels = core_gguf::kv_str_array(gctx, "ner.labels");
    if ((int)ctx->labels.size() != ctx->n_labels) {
        // Fall back to generated names
        ctx->labels.resize(ctx->n_labels);
        for (int i = 0; i < ctx->n_labels; i++) {
            char key[64];
            snprintf(key, sizeof(key), "ner.label.%d", i);
            ctx->labels[i] = core_gguf::kv_str(gctx, key, ("LABEL_" + std::to_string(i)).c_str());
        }
    }

    core_gguf::free_metadata(gctx);

    // Load classifier weight tensors (need a separate backend for the classifier)
    bool force_cpu = (getenv("BERT_NER_FORCE_CPU") && atoi(getenv("BERT_NER_FORCE_CPU")));
    ggml_backend_t cls_backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!cls_backend) cls_backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(cls_backend))
        ggml_backend_cpu_set_n_threads(cls_backend, n_threads);
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, cls_backend, "bert_ner", wl)) {
        fprintf(stderr, "bert_ner: warning: could not load weight tensors\n");
        ggml_backend_free(cls_backend);
    }

    auto read_tensor_f32 = [&](const char* name, std::vector<float>& dst, size_t expected) -> bool {
        ggml_tensor* t = core_gguf::try_get(wl.tensors, name);
        if (!t) return false;
        size_t n = ggml_nelements(t);
        if (n != expected) {
            fprintf(stderr, "bert_ner: %s: expected %zu elements, got %zu\n", name, expected, n);
            return false;
        }
        dst.resize(n);
        // Read via backend API (tensors live on the backend buffer)
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<uint16_t> f16(n);
            ggml_backend_tensor_get(t, f16.data(), 0, n * sizeof(uint16_t));
            for (size_t i = 0; i < n; i++)
                dst[i] = ggml_fp16_to_fp32(f16[i]);
        } else {
            fprintf(stderr, "bert_ner: %s: unsupported type %d\n", name, t->type);
            return false;
        }
        return true;
    };

    size_t w_size = (size_t)ctx->n_labels * ctx->hidden_dim;
    if (!read_tensor_f32("ner.classifier.weight", ctx->cls_weight, w_size)) {
        fprintf(stderr, "bert_ner: missing ner.classifier.weight\n");
        core_gguf::free_weights(wl);
        if (cls_backend) ggml_backend_free(cls_backend);
        crispembed_free(ctx->enc);
        delete ctx;
        return false;
    }

    // Bias is optional
    ctx->cls_bias.resize(ctx->n_labels, 0.0f);
    read_tensor_f32("ner.classifier.bias", ctx->cls_bias, ctx->n_labels);

    core_gguf::free_weights(wl);
    if (cls_backend) ggml_backend_free(cls_backend);

    fprintf(stderr, "bert_ner: loaded %d labels, hidden=%d\n", ctx->n_labels, ctx->hidden_dim);
    for (int i = 0; i < ctx->n_labels; i++)
        fprintf(stderr, "  label[%d] = %s\n", i, ctx->labels[i].c_str());

    *out = ctx;
    return true;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

std::vector<entity> extract(context* ctx, const char* text) {
    ctx->last_entities.clear();
    if (!ctx || !text || !text[0]) return ctx->last_entities;

    // Step 1: Get per-token hidden states from the encoder.
    // encode_tokens returns L2-normalized vectors, but for NER we need
    // the raw (unnormalized) hidden states. We'll use the raw encoder output.
    // However, crispembed_encode_tokens normalizes. Let's still use it —
    // the classifier was trained on the raw output, but for a well-trained
    // NER model the normalization doesn't change the argmax (just scales).
    //
    // Actually, we need raw hidden states. Let me use encode_tokens and
    // then undo the L2 normalization — but we don't have the original norms.
    // Better approach: the raw encoder output is what we need.
    // The encode_tokens API normalizes, which is wrong for classification.
    // Let me use a workaround: call the encoder and read the raw output.

    int n_tokens = 0, dim = 0;
    const float* hidden = crispembed_encode_tokens_raw(ctx->enc, text, &n_tokens, &dim);
    if (!hidden || n_tokens == 0) return ctx->last_entities;

    const int H = dim;
    const int L = ctx->n_labels;

    // Step 2: Apply classifier head: logits[t, l] = sum_h(W[l,h] * hidden[t,h]) + b[l]
    // cls_weight is [n_labels, hidden_dim] row-major (from PyTorch)
    std::vector<float> logits(n_tokens * L);
    for (int t = 0; t < n_tokens; t++) {
        const float* h = hidden + t * H;
        for (int l = 0; l < L; l++) {
            float sum = ctx->cls_bias[l];
            const float* w = ctx->cls_weight.data() + l * H;
            for (int d = 0; d < H; d++)
                sum += w[d] * h[d];
            logits[t * L + l] = sum;
        }
    }

    // Step 3: Argmax + softmax score per token
    struct token_pred {
        int label_id;
        float score;
    };
    std::vector<token_pred> preds(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        const float* row = logits.data() + t * L;
        int best = 0;
        for (int l = 1; l < L; l++)
            if (row[l] > row[best]) best = l;

        // Softmax score for the predicted label
        float max_val = *std::max_element(row, row + L);
        float sum = 0.0f;
        for (int l = 0; l < L; l++) sum += expf(row[l] - max_val);
        float score = expf(row[best] - max_val) / sum;

        preds[t] = {best, score};
    }

    // Step 4: BIO decode — merge consecutive B-X / I-X tokens into spans.
    // Get token-to-character mapping from the tokenizer.
    const int32_t* token_ids = crispembed_last_token_ids(ctx->enc);
    int tok_kind = crispembed_tokenizer_kind(ctx->enc);  // 1=WordPiece, 2=SentencePiece

    // Reconstruct character offsets from tokens.
    // For WordPiece: tokens starting with "##" continue the previous word.
    // For SentencePiece: tokens starting with "▁" start a new word.
    std::string input_text(text);
    std::vector<int> tok_char_start(n_tokens, -1);
    std::vector<int> tok_char_end(n_tokens, -1);

    int char_pos = 0;
    for (int t = 0; t < n_tokens; t++) {
        const char* tok_str = crispembed_token_str(ctx->enc, token_ids[t]);
        if (!tok_str || !tok_str[0]) continue;

        std::string ts(tok_str);
        // Skip special tokens ([CLS], [SEP], <s>, </s>)
        if (ts == "[CLS]" || ts == "[SEP]" || ts == "<s>" || ts == "</s>" ||
            ts == "[PAD]" || ts == "<pad>") continue;

        // WordPiece: strip "##" prefix
        bool is_continuation = false;
        if (tok_kind == 1 && ts.size() > 2 && ts[0] == '#' && ts[1] == '#') {
            ts = ts.substr(2);
            is_continuation = true;
        }
        // SentencePiece: strip "▁" prefix (U+2581, 3 bytes in UTF-8)
        if (tok_kind == 2 && ts.size() >= 3 &&
            (uint8_t)ts[0] == 0xE2 && (uint8_t)ts[1] == 0x96 && (uint8_t)ts[2] == 0x81) {
            ts = ts.substr(3);
            is_continuation = false;
        }

        // Find this token's text in the input, starting from char_pos
        if (!is_continuation) {
            // Skip whitespace to find next word start
            while (char_pos < (int)input_text.size() && input_text[char_pos] == ' ')
                char_pos++;
        }

        // Match the token text (case-insensitive for BERT uncased models)
        size_t match_len = ts.size();
        if (char_pos + (int)match_len <= (int)input_text.size()) {
            tok_char_start[t] = char_pos;
            tok_char_end[t] = char_pos + (int)match_len;
            char_pos += (int)match_len;
        }
    }

    // Step 5: Group B-X / I-X sequences into entity spans
    int span_start_tok = -1;
    int span_start_char = -1;
    int span_end_char = -1;
    std::string span_label;
    float span_score_sum = 0.0f;
    int span_count = 0;

    auto flush_span = [&]() {
        if (span_label.empty() || span_start_char < 0) return;
        entity e;
        e.start_char = span_start_char;
        e.end_char = span_end_char;
        e.text = input_text.substr(span_start_char, span_end_char - span_start_char);
        e.label = span_label;
        e.score = span_score_sum / std::max(span_count, 1);
        ctx->last_entities.push_back(std::move(e));
    };

    for (int t = 0; t < n_tokens; t++) {
        if (tok_char_start[t] < 0) continue;  // skip special tokens

        const std::string& label = ctx->labels[preds[t].label_id];

        bool is_begin = label.size() > 2 && label[0] == 'B' && label[1] == '-';
        bool is_inside = label.size() > 2 && label[0] == 'I' && label[1] == '-';
        std::string base_label = (is_begin || is_inside) ? label.substr(2) : "";

        if (is_begin) {
            flush_span();
            span_label = base_label;
            span_start_char = tok_char_start[t];
            span_end_char = tok_char_end[t];
            span_score_sum = preds[t].score;
            span_count = 1;
        } else if (is_inside && !span_label.empty() && base_label == span_label) {
            // Extend current span
            span_end_char = tok_char_end[t];
            span_score_sum += preds[t].score;
            span_count++;
        } else {
            flush_span();
            span_label.clear();
            span_start_char = -1;
        }
    }
    flush_span();

    return ctx->last_entities;
}

int num_labels(context* ctx) {
    return ctx ? ctx->n_labels : 0;
}

const char* label_name(context* ctx, int label_id) {
    if (!ctx || label_id < 0 || label_id >= ctx->n_labels) return "";
    return ctx->labels[label_id].c_str();
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->enc) crispembed_free(ctx->enc);
    delete ctx;
}

} // namespace bert_ner
