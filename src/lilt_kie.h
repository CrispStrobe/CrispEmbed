// lilt_kie.h — LiLT (Language-independent Layout Transformer) for KIE.
//
// Dual-stream encoder: RoBERTa text + layout transformer with BiACM
// (bidirectional attention complementation). Supports token classification
// for form understanding (FUNSD: question/answer/header labeling).
//
// Usage:
//   lilt_kie::context* ctx;
//   lilt_kie::load(&ctx, "lilt-funsd-f32.gguf", 4);
//
//   int32_t ids[] = {0, 10566, 35, ...};
//   int32_t bbox[][4] = {{0,0,0,0}, {10,50,90,80}, ...};
//   auto result = lilt_kie::classify(ctx, ids, bbox, n_tokens);
//   for (auto& tok : result) printf("%s → %s\n", tok.text, tok.label);
//
//   lilt_kie::free(ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lilt_kie {

struct token_result {
    int         token_id;
    int         label_id;   // argmax of logits
    std::string label;      // label string from id2label
    float       score;      // softmax probability of predicted label
};

struct context;

// Load LiLT GGUF model. Returns true on success.
bool load(context** ctx, const char* model_path, int n_threads = 4);

// Run token classification. Returns per-token predictions.
// input_ids: [n_tokens] token ids (including BOS/EOS)
// bbox: [n_tokens][4] bounding boxes (x0, y0, x1, y1), each in [0, 1000]
// n_tokens: sequence length
std::vector<token_result> classify(context* ctx,
                                    const int32_t* input_ids,
                                    const int32_t* bbox,  // flat [n_tokens * 4]
                                    int n_tokens);

// Run with dump mode: returns named per-layer intermediates for parity testing.
struct dump_tensor {
    std::string name;
    std::vector<float> data;
    int n_elem;
};
std::vector<dump_tensor> classify_dump(context* ctx,
                                        const int32_t* input_ids,
                                        const int32_t* bbox,
                                        int n_tokens);

// Get label name by id.
const char* label_name(context* ctx, int label_id);

// Get number of labels.
int num_labels(context* ctx);

// Free all resources.
void free(context* ctx);

} // namespace lilt_kie
