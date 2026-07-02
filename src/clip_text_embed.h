// clip_text_embed.h — CLIP text encoder for cross-modal text↔image retrieval.
//
// Loads a GGUF produced by convert-clip-text-to-gguf.py and runs the
// text transformer: tokenize → embed → N × (LN→CausalAttn→LN→MLP) →
// final_ln → EOS pool → text_projection → L2 normalize.
//
// The output vector lives in the same space as CLIP vision embeddings,
// enabling cosine-similarity text-image search.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace clip_text {

struct context;

bool load(context ** ctx, const char * path, int n_threads = 1);
int dim(const context * ctx);

// Encode text to CLIP embedding. Returns L2-normalized vector.
std::vector<float> encode(context * ctx, const char * text);

void free(context * ctx);

} // namespace clip_text
