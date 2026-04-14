#pragma once

#include "core/gguf_loader.h"
#include "tokenizer.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <vector>

struct dec_layer {
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * q_w = nullptr, * q_b = nullptr;
    ggml_tensor * k_w = nullptr, * k_b = nullptr;
    ggml_tensor * v_w = nullptr, * v_b = nullptr;
    ggml_tensor * o_w = nullptr, * o_b = nullptr;
    ggml_tensor * q_norm_w = nullptr;
    ggml_tensor * k_norm_w = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * gate_w = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * down_w = nullptr;
};

struct dec_model {
    int n_vocab = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int n_layer = 0;
    int n_intermediate = 0;
    int n_max_pos = 8192;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;

    ggml_tensor * token_embd = nullptr;
    ggml_tensor * output_norm = nullptr;
    std::vector<dec_layer> layers;
};

bool load_decoder_model(dec_model & m, core_gguf::WeightLoad & wl,
                         const char * path, ggml_backend_t backend);

std::vector<float> decoder_encode_tokens(
    const dec_model & m, ggml_backend_t backend,
    const embed_tokens & tokens, int n_threads);
