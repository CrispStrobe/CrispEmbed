#pragma once
#include "qwen35_attention.h"
#include "qwen35_linear.h"
#include "core/gguf_loader.h"
namespace qwen35_ocr {
struct decoder_layer_weights {
    bool full_attention = false;
    linear_weights linear{};
    attention_weights attention{};
    ggml_tensor * input_norm = nullptr;
    ggml_tensor * post_norm = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
};
struct decoder_weights {
    std::vector<decoder_layer_weights> layers;
    ggml_tensor * norm = nullptr;
    ggml_tensor * embedding = nullptr;
    ggml_tensor * head = nullptr;
};
struct decoder_layer_cache {
    std::vector<float> conv, recurrent;
    attention_cache attention;
};
struct decoder_cache {
    int batch = 0, tokens = 0;
    std::vector<decoder_layer_cache> layers;
};
struct decoder_outputs {
    std::vector<std::vector<float>> hidden;
    std::vector<float> norm, logits;
    decoder_cache cache;
    std::map<int, std::vector<float>> attention_inputs, attention_keys;
};
// Explicit small-model schedule; no automatic support for unknown checkpoints.
bool map_small_decoder(const core_gguf::tensor_map & tensors, decoder_weights & weights);
bool embed_cpu(ggml_tensor * embedding, const std::vector<int> & ids, std::vector<float> & hidden);
// Consumes merged vision/text embeddings. Frontend/tokenizer are separate.
bool decoder_cpu(const decoder_weights & weights, const attention_config & config, const float * hidden,
                 const float * positions, int batch, int tokens, int threads, const decoder_cache * previous,
                 decoder_outputs & output, converted_weight_cache * weights_cache = nullptr);
// Internal full-prefill primitive for a dedicated, strictly validated Josev
// F32 profile. No generation, previous cache or vocabulary projection.
bool decoder_hidden_cpu(const decoder_weights & weights, const attention_config & config, const float * hidden,
                        const float * positions, int tokens, int threads, decoder_outputs & output);
} // namespace qwen35_ocr
