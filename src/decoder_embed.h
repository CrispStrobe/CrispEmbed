// decoder_embed.h — Decoder-style embedding model (Qwen3/LLaMA architecture).
//
// For models like Qwen3-Embedding, Octen, F2LLM, Jina v5, Harrier
// that use causal attention + RoPE + SwiGLU/GELU FFN + last-token pooling.
//
// Architecture:
//   Token embeddings + RoPE
//   N × (RMSNorm → Causal MHA with RoPE → residual → RMSNorm → FFN → residual)
//   RMSNorm
//   Last-token extraction → L2 normalize

#pragma once

#include "crispembed.h"
#include <string>

// Check if a model is decoder-style based on GGUF metadata
bool is_decoder_model(const char * model_path);

// TODO: Implement decoder graph builder
// The decoder graph needs:
// - RMSNorm (not LayerNorm)
// - Causal attention mask
// - RoPE positional encoding
// - SwiGLU FFN (gate * up * silu)
// - Last-token pooling
//
// These ops are all available in ggml (used by CrispASR's voxtral/qwen3).
// The converter needs to handle Qwen3's weight naming convention.
