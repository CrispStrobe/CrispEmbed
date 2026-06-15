// bert_ner.h — Fixed-label NER via BERT/XLM-R encoder + token classification head.
//
// Uses the existing CrispEmbed encoder (encode_tokens) with a simple
// Linear(hidden_dim, num_labels) head on top. Labels are baked into the
// model (e.g. CoNLL-03: PER/LOC/ORG/MISC). BIO decoding merges consecutive
// B-/I- tokens into entity spans with character offsets.
//
// Usage:
//   bert_ner::context* ctx;
//   bert_ner::load(&ctx, "bert-base-ner-f32.gguf", 4);
//   auto entities = bert_ner::extract(ctx, "Barack Obama was born in Hawaii");
//   // → [{text="Barack Obama", label="PER", start=0, end=12, score=0.99}, ...]
//   bert_ner::free(ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bert_ner {

struct entity {
    int         start_char;  // character offset in input text
    int         end_char;    // character offset (exclusive)
    std::string text;        // extracted span
    std::string label;       // entity label (e.g. "PER", "LOC")
    float       score;       // mean softmax probability across span tokens
};

struct context;

// Load BERT/XLM-R NER model from GGUF.
// The GGUF must contain ner.classifier.weight and ner.labels metadata.
bool load(context** ctx, const char* model_path, int n_threads = 4);

// Extract named entities. Returns entity spans with character offsets.
std::vector<entity> extract(context* ctx, const char* text);

// Get number of labels.
int num_labels(context* ctx);

// Get label name by id.
const char* label_name(context* ctx, int label_id);

// Free all resources.
void free(context* ctx);

} // namespace bert_ner
