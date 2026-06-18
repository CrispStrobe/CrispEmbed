// lfm2_embed.h — LFM2.5 bidirectional encoder for CLS-pooled text embeddings.
//
// Backbone: 16-layer ShortConv+GQA hybrid (same as GLiNER-LFM), 1024-dim.
// Pooling: CLS token (position 0) of final hidden state, L2-normalized.
//
// GGUF arch = "lfm2"; tensor prefix "lfm.*".
//
// Usage (internal — called from crispembed.cpp dispatch):
//   lfm2_embed_ctx * ctx = lfm2_embed_load(path, backend);
//   std::vector<float> emb = lfm2_embed_encode(ctx, "query: hello world");
//   lfm2_embed_free(ctx);

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <string>

struct lfm2_embed_ctx;

// Load LFM2 embedding model from GGUF.  backend must already be initialised
// (reuses the crispembed Metal/CPU backend).  Returns nullptr on failure.
lfm2_embed_ctx * lfm2_embed_load(const char * path, ggml_backend_t backend);

// Free all resources.
void lfm2_embed_free(lfm2_embed_ctx * ctx);

// Encode text (with any prefix already prepended) → L2-normalised 1024-dim
// CLS vector.  Returns empty on error.
std::vector<float> lfm2_embed_encode(lfm2_embed_ctx * ctx, const char * text);

// Output dimension (hidden_size = 1024).
int lfm2_embed_n_embd(const lfm2_embed_ctx * ctx);

// Dump mode: encode + capture per-layer intermediates for parity testing.
// Fills *names and *data with one entry per captured stage.
// Shape is (T * H) row-major for 2D tensors, (H,) for CLS vectors.
struct lfm2_dump_entry {
    std::string name;
    std::vector<float> data;
    int T = 0, H = 0;  // 2D: T tokens, H hidden; 1D: T=1, H=dim
};
std::vector<lfm2_dump_entry> lfm2_embed_encode_dump(lfm2_embed_ctx * ctx, const char * text);
