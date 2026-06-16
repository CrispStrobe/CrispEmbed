// internvl2_ocr.cpp — InternVL2/2.5 OCR inference engine.
//
// Architecture (InternVL2.5-2B):
//
// Vision encoder (InternViT-300M):
//   image → Conv2D patches → + CLS + position embedding
//   24 × pre-LayerNorm ViT block:
//     LayerNorm → fused QKV + bias → attention → * LayerScale → residual
//     LayerNorm → GELU MLP (fc1, fc2) + bias → * LayerScale → residual
//   Remove CLS → pixel unshuffle (4:1, dim 1024→4096)
//   → LayerNorm → Linear → GELU → Linear → (n_merged, 2048)
//
// LLM decoder (InternLM2.5-1.8B):
//   embed_tokens → splice image_embeds at <IMG_CONTEXT> positions
//   24 × pre-RMSNorm InternLM2 block:
//     RMSNorm → Q/K/V (split from wqkv, GQA 16/8) + RoPE → attention → residual
//     RMSNorm → SwiGLU FFN (w1=gate, w3=up, w2=down) → residual
//   RMSNorm → lm_head → logits → greedy decode
//
// Per-layer diff comparison via crispembed_diff.h when ctx.diff_ref_path is set.

#include "internvl2_ocr.h"
#include "image_preprocess.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"
#include "core/bpe.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace internvl2_ocr {

// ── Tokenizer decode ─────────────────────────────────────────────────

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// GPT-2 byte_decoder: inverse of bytes_to_unicode().
// Maps each Unicode codepoint used in GPT-2 BPE tokens back to the
// original byte value. Built lazily on first call.
static const std::vector<int> &gpt2_byte_decoder() {
    static std::vector<int> table;
    if (!table.empty()) return table;
    table.assign(65536, -1);
    // Build forward map (byte → unicode) then invert
    auto fwd = core_bpe::byte_encoder();
    for (int b = 0; b < 256; b++) {
        int cp = fwd[b];
        if (cp >= 0 && cp < (int)table.size()) {
            table[cp] = b;
        }
    }
    return table;
}

// Decode a GPT-2 BPE token string to raw bytes.
// Each character in the token is a Unicode codepoint that maps to a byte.
static std::string gpt2_token_to_bytes(const std::string &token) {
    const auto &dec = gpt2_byte_decoder();
    std::string result;
    // Parse UTF-8 codepoints from the token string
    const uint8_t *p = (const uint8_t *)token.data();
    const uint8_t *end = p + token.size();
    while (p < end) {
        uint32_t cp;
        if (*p < 0x80) {
            cp = *p++;
        } else if (*p < 0xE0) {
            cp = (*p++ & 0x1F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        } else if (*p < 0xF0) {
            cp = (*p++ & 0x0F) << 12;
            if (p < end) cp |= (*p++ & 0x3F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        } else {
            cp = (*p++ & 0x07) << 18;
            if (p < end) cp |= (*p++ & 0x3F) << 12;
            if (p < end) cp |= (*p++ & 0x3F) << 6;
            if (p < end) cp |= (*p++ & 0x3F);
        }
        if (cp < dec.size() && dec[cp] >= 0) {
            result += (char)dec[cp];
        }
    }
    return result;
}

std::string tokenizer::decode(const int32_t *ids, int n) const {
    // Detect tokenizer type: SentencePiece (InternLM2, vocab ~92K)
    // vs GPT-2 BPE (Qwen2, vocab ~151K)
    bool is_gpt2_bpe = vocab_size > 100000;

    std::string result;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= vocab_size) continue;
        if (id == bos_id || id == eos_id) continue;
        if (id == im_end_id) break;

        const std::string &piece = id_to_piece[id];
        if (piece.empty()) continue;

        // Skip special/control tokens
        if (piece[0] == '[' && piece.back() == ']') continue;  // [UNUSED_*]
        if (piece == "<s>" || piece == "</s>") continue;
        if (piece.size() > 3 && piece[0] == '<' && piece[1] == '|' &&
            piece.back() == '>' && piece[piece.size()-2] == '|') continue;  // <|...|>

        if (is_gpt2_bpe) {
            // GPT-2 BPE: each char is a byte-mapped codepoint
            result += gpt2_token_to_bytes(piece);
        } else {
            // SentencePiece: <0xNN> = byte fallback, ▁ = space
            if (piece.size() == 6 && piece[0] == '<' && piece[1] == '0' &&
                piece[2] == 'x' && piece[5] == '>') {
                int h = hex_val(piece[3]);
                int l = hex_val(piece[4]);
                if (h >= 0 && l >= 0) {
                    result += (char)(h * 16 + l);
                    continue;
                }
            }
            std::string text = piece;
            size_t pos = 0;
            while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) {
                text.replace(pos, 3, " ");
                pos += 1;
            }
            result += text;
        }
    }

    // Trim leading space
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    return result;
}

std::string tokenizer::decode(const std::vector<int32_t> &ids) const {
    return decode(ids.data(), (int)ids.size());
}

void tokenizer::build_reverse_map() {
    piece_to_id.clear();
    for (int i = 0; i < (int)id_to_piece.size(); i++)
        piece_to_id[id_to_piece[i]] = i;
    // Find special tokens
    auto find = [&](const char *s) -> int {
        auto it = piece_to_id.find(s);
        return it != piece_to_id.end() ? it->second : -1;
    };
    if (im_start_id < 0) im_start_id = find("<|im_start|>");
    if (im_end_id < 0) im_end_id = find("<|im_end|>");
    newline_id = find("\n");
    if (newline_id < 0) newline_id = find("Ċ");  // GPT-2 BPE '\n'
}

std::vector<int32_t> tokenizer::encode(const std::string &text) const {
    // Greedy longest-match encoding using vocab.
    // Not full BPE but sufficient for short prompts.
    std::vector<int32_t> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        int best_len = 0;
        int32_t best_id = -1;
        // Try longest match first (up to 16 bytes)
        for (int len = std::min((int)(text.size() - pos), 16); len >= 1; len--) {
            std::string piece = text.substr(pos, len);
            auto it = piece_to_id.find(piece);
            if (it != piece_to_id.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            pos += best_len;
        } else {
            // Byte fallback: <0xNN>
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)text[pos]);
            auto it = piece_to_id.find(hex);
            ids.push_back(it != piece_to_id.end() ? it->second : 0);
            pos++;
        }
    }
    return ids;
}

std::vector<int32_t> tokenizer::build_prompt(const std::string &user_text,
                                              int n_image_tokens) const {
    // InternVL2 chat template:
    // <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
    // <|im_start|>user\n<IMG_CONTEXT>*N\n{text}<|im_end|>\n
    // <|im_start|>assistant\n
    std::vector<int32_t> ids;

    auto add_special = [&](int id) { if (id >= 0) ids.push_back(id); };
    auto add_text = [&](const std::string &s) {
        auto toks = encode(s);
        ids.insert(ids.end(), toks.begin(), toks.end());
    };

    // System
    add_special(im_start_id);
    add_text("system");
    add_special(newline_id);
    add_text("You are a helpful assistant.");
    add_special(im_end_id);
    add_special(newline_id);

    // User with image
    add_special(im_start_id);
    add_text("user");
    add_special(newline_id);
    for (int i = 0; i < n_image_tokens; i++)
        ids.push_back(image_token_id);
    add_special(newline_id);
    add_text(user_text);
    add_special(im_end_id);
    add_special(newline_id);

    // Assistant
    add_special(im_start_id);
    add_text("assistant");
    add_special(newline_id);

    return ids;
}

namespace {

// ── Hparams loading ──────────────────────────────────────────────────

bool load_hparams(context &ctx, const char *path) {
    gguf_context *g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char *k, uint32_t d) {
        return core_gguf::kv_u32(g, k, d);
    };
    auto f32v = [&](const char *k, float d) {
        return core_gguf::kv_f32(g, k, d);
    };
    auto boolv = [&](const char *k, bool d) -> bool {
        int idx = gguf_find_key(g, k);
        return idx >= 0 ? gguf_get_val_bool(g, idx) : d;
    };

    auto &vhp = ctx.m.vhp;
    auto &php = ctx.m.php;
    auto &lhp = ctx.m.lhp;

    // Vision
    vhp.num_hidden_layers   = u32("internvl2.vision.num_hidden_layers", vhp.num_hidden_layers);
    vhp.hidden_size         = u32("internvl2.vision.hidden_size", vhp.hidden_size);
    vhp.intermediate_size   = u32("internvl2.vision.intermediate_size", vhp.intermediate_size);
    vhp.num_attention_heads = u32("internvl2.vision.num_attention_heads", vhp.num_attention_heads);
    vhp.patch_size          = u32("internvl2.vision.patch_size", vhp.patch_size);
    vhp.image_size          = u32("internvl2.vision.image_size", vhp.image_size);
    vhp.layer_norm_eps      = f32v("internvl2.vision.layer_norm_eps", vhp.layer_norm_eps);
    vhp.qkv_bias            = boolv("internvl2.vision.qkv_bias", vhp.qkv_bias);

    // Derived
    vhp.head_dim = vhp.hidden_size / vhp.num_attention_heads;
    vhp.n_patches_per_side = vhp.image_size / vhp.patch_size;
    vhp.n_patches = vhp.n_patches_per_side * vhp.n_patches_per_side;
    vhp.n_positions = vhp.n_patches + 1;  // +CLS

    // Image preprocessor
    int idx = gguf_find_key(g, "internvl2.vision.image_mean");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *data = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_mean[i] = data[i];
    }
    idx = gguf_find_key(g, "internvl2.vision.image_std");
    if (idx >= 0 && gguf_get_arr_n(g, idx) >= 3) {
        auto *data = (const float *)gguf_get_arr_data(g, idx);
        for (int i = 0; i < 3; i++) vhp.image_std[i] = data[i];
    }

    // Projector
    php.downsample_ratio = f32v("internvl2.downsample_ratio", php.downsample_ratio);
    php.n_merged_tokens  = u32("internvl2.vision.num_merged_tokens", php.n_merged_tokens);
    php.merge_dim        = u32("internvl2.vision.merge_dim", php.merge_dim);
    php.output_dim       = u32("internvl2.hidden_size",  // same as LLM hidden
                               php.output_dim);

    // LLM
    lhp.vocab_size              = u32("internvl2.vocab_size", lhp.vocab_size);
    lhp.hidden_size             = u32("internvl2.hidden_size", lhp.hidden_size);
    lhp.intermediate_size       = u32("internvl2.intermediate_size", lhp.intermediate_size);
    lhp.num_hidden_layers       = u32("internvl2.num_hidden_layers", lhp.num_hidden_layers);
    lhp.num_attention_heads     = u32("internvl2.num_attention_heads", lhp.num_attention_heads);
    lhp.num_key_value_heads     = u32("internvl2.num_key_value_heads", lhp.num_key_value_heads);
    lhp.head_dim                = lhp.hidden_size / lhp.num_attention_heads;
    lhp.max_position_embeddings = u32("internvl2.max_position_embeddings", lhp.max_position_embeddings);
    lhp.rms_norm_eps            = f32v("internvl2.rms_norm_eps", lhp.rms_norm_eps);
    lhp.rope_theta              = f32v("internvl2.rope_theta", lhp.rope_theta);
    lhp.rope_scaling_factor     = f32v("internvl2.rope_scaling_factor", lhp.rope_scaling_factor);
    lhp.tie_word_embeddings     = boolv("internvl2.tie_word_embeddings", lhp.tie_word_embeddings);
    lhp.image_token_id          = u32("internvl2.image_token_id", lhp.image_token_id);

    // Dynamic resolution
    lhp.max_dynamic_patch = u32("internvl2.max_dynamic_patch", lhp.max_dynamic_patch);
    lhp.min_dynamic_patch = u32("internvl2.min_dynamic_patch", lhp.min_dynamic_patch);
    lhp.use_thumbnail     = boolv("internvl2.use_thumbnail", lhp.use_thumbnail);

    // Tokenizer special tokens
    lhp.bos_token_id = u32("internvl2.tokenizer.bos_id", lhp.bos_token_id);
    lhp.eos_token_id = u32("internvl2.tokenizer.eos_id", lhp.eos_token_id);

    php.output_dim = lhp.hidden_size;  // projector output = LLM hidden

    // Tokenizer: load vocab for decode
    auto &tok = ctx.tok;
    tok.bos_id = (int)lhp.bos_token_id;
    tok.eos_id = (int)lhp.eos_token_id;
    tok.image_token_id = (int)lhp.image_token_id;

    int im_end_idx = gguf_find_key(g, "internvl2.tokenizer.im_end_id");
    if (im_end_idx >= 0) {
        tok.im_end_id = (int)gguf_get_val_u32(g, im_end_idx);
    }

    // Load vocab from standard GGUF tokenizer keys
    int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        int n = gguf_get_arr_n(g, vocab_idx);
        tok.id_to_piece.resize(n);
        for (int i = 0; i < n; i++) {
            tok.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
        }
        tok.vocab_size = n;
        tok.build_reverse_map();
    }

    core_gguf::free_metadata(g);
    return true;
}

// ── Tensor loading ───────────────────────────────────────────────────

bool load_tensors(context &ctx, const char *path) {
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx.backend, "internvl2_ocr", wl)) {
        return false;
    }
    ctx.model_ctx = wl.ctx;
    ctx.model_buf = wl.buf;

    auto &m = ctx.m;
    auto get = [&](const std::string &name) -> ggml_tensor * {
        auto it = wl.tensors.find(name);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    // Vision encoder
    m.patch_embed_w = get("v.patch_embed.weight");
    m.patch_embed_b = get("v.patch_embed.bias");
    m.class_embedding = get("v.class_embedding");
    m.position_embedding = get("v.position_embedding");

    m.vis_blocks.resize(m.vhp.num_hidden_layers);
    for (uint32_t i = 0; i < m.vhp.num_hidden_layers; i++) {
        auto &blk = m.vis_blocks[i];
        std::string p = "v.blk." + std::to_string(i) + ".";
        blk.norm1_w  = get(p + "norm1.weight");
        blk.norm1_b  = get(p + "norm1.bias");
        blk.norm2_w  = get(p + "norm2.weight");
        blk.norm2_b  = get(p + "norm2.bias");
        blk.ls1      = get(p + "ls1");
        blk.ls2      = get(p + "ls2");
        blk.qkv_w    = get(p + "attn_qkv.weight");
        blk.qkv_b    = get(p + "attn_qkv.bias");
        blk.proj_w   = get(p + "attn_proj.weight");
        blk.proj_b   = get(p + "attn_proj.bias");
        blk.fc1_w    = get(p + "ffn_fc1.weight");
        blk.fc1_b    = get(p + "ffn_fc1.bias");
        blk.fc2_w    = get(p + "ffn_fc2.weight");
        blk.fc2_b    = get(p + "ffn_fc2.bias");
    }

    // MLP projector
    m.proj.norm_w = get("v.proj.norm.weight");
    m.proj.norm_b = get("v.proj.norm.bias");
    m.proj.fc1_w  = get("v.proj.fc1.weight");
    m.proj.fc1_b  = get("v.proj.fc1.bias");
    m.proj.fc2_w  = get("v.proj.fc2.weight");
    m.proj.fc2_b  = get("v.proj.fc2.bias");

    // LLM decoder
    m.embed_tokens = get("l.embed_tokens.weight");

    m.llm_layers.resize(m.lhp.num_hidden_layers);
    for (uint32_t i = 0; i < m.lhp.num_hidden_layers; i++) {
        auto &ly = m.llm_layers[i];
        std::string p = "l.blk." + std::to_string(i) + ".";
        ly.attn_norm_w = get(p + "attn_norm.weight");
        ly.ffn_norm_w  = get(p + "ffn_norm.weight");
        ly.q_w         = get(p + "attn_q.weight");
        ly.q_b         = get(p + "attn_q.bias");
        ly.k_w         = get(p + "attn_k.weight");
        ly.k_b         = get(p + "attn_k.bias");
        ly.v_w         = get(p + "attn_v.weight");
        ly.v_b         = get(p + "attn_v.bias");
        ly.o_w         = get(p + "attn_o.weight");
        ly.ffn_gate_w  = get(p + "ffn_gate.weight");
        ly.ffn_up_w    = get(p + "ffn_up.weight");
        ly.ffn_down_w  = get(p + "ffn_down.weight");
    }

    m.output_norm_w = get("l.output_norm.weight");
    m.lm_head_w     = get("l.lm_head.weight");

    return true;
}

// ── Vision encoder graph ─────────────────────────────────────────────

// Build vision encoder graph for a single 448×448 tile.
// Input: (3*P*P, n_patches) patch pixels
// Output: (hidden_size, n_positions) = (1024, 1025) hidden states

struct vision_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *pixel_in = nullptr;   // input: (patch_flat_dim, n_patches)
    ggml_tensor *output = nullptr;     // output: (D, n_pos) including CLS
    std::vector<ggml_tensor *> layer_outputs;  // for diff comparison
};

vision_graph build_vision_graph(context &ctx) {
    vision_graph vg;
    const auto &vhp = ctx.m.vhp;
    const int D = (int)vhp.hidden_size;        // 1024
    const int nh = (int)vhp.num_attention_heads; // 16
    const int hd = D / nh;                       // 64
    const int inter = (int)vhp.intermediate_size; // 4096
    const int P = (int)vhp.patch_size;           // 14
    const int n_patches = (int)vhp.n_patches;   // 1024
    const int n_pos = (int)vhp.n_positions;     // 1025
    const int patch_flat = 3 * P * P;            // 588
    const float ln_eps = vhp.layer_norm_eps;
    const int n_layers = (int)vhp.num_hidden_layers;

    size_t meta_size = (size_t)(n_layers * 40 + 100) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(16384, false);
    ctx.compute_meta.resize(meta_size);
    ggml_init_params ip{meta_size, ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);
    vg.gctx = g;

    ggml_cgraph *gf = ggml_new_graph_custom(g, 16384, false);

    // Input patches: (patch_flat, n_patches) = (588, 1024)
    ggml_tensor *pixel_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, patch_flat, n_patches);
    ggml_set_name(pixel_in, "pixel_in");
    ggml_set_input(pixel_in);
    vg.pixel_in = pixel_in;

    // Patch embedding: mul_mat(patch_embed_w, pixel_in) + bias
    // patch_embed_w: (patch_flat, D) → result: (D, n_patches)
    ggml_tensor *x = ggml_mul_mat(g, ctx.m.patch_embed_w, pixel_in);
    if (ctx.m.patch_embed_b) {
        x = ggml_add(g, x, ctx.m.patch_embed_b);
    }

    // Prepend CLS token: class_embedding is (1, D) or (1, 1, D)
    // We need to concat along the sequence dimension (dim 1 in ggml = columns)
    // x is (D, n_patches), CLS is (D, 1), result is (D, n_patches+1)
    ggml_tensor *cls = ggml_reshape_2d(g, ctx.m.class_embedding, D, 1);
    x = ggml_concat(g, cls, x, 1);  // (D, 1+n_patches) = (D, n_pos)

    // Add position embedding: (D, n_pos)
    ggml_tensor *pos = ggml_reshape_2d(g, ctx.m.position_embedding, D, n_pos);
    x = ggml_add(g, x, pos);

    ggml_set_name(x, "vis_patch_embed");
    ggml_set_output(x);

    // ── LayerNorm helper ──
    auto layer_norm = [&](ggml_tensor *t, ggml_tensor *w, ggml_tensor *b) -> ggml_tensor * {
        ggml_tensor *y = ggml_norm(g, t, ln_eps);
        y = ggml_mul(g, y, w);
        if (b) y = ggml_add(g, y, b);
        return y;
    };

    // ── Transformer layers ──
    for (int i = 0; i < n_layers; i++) {
        auto &blk = ctx.m.vis_blocks[i];
        int T = n_pos;  // sequence length (1025)

        // ── Self-attention ──
        // Pre-norm
        ggml_tensor *h = layer_norm(x, blk.norm1_w, blk.norm1_b);

        // Fused QKV: (D, T) → (3*D, T) via mul_mat
        ggml_tensor *qkv = ggml_mul_mat(g, blk.qkv_w, h);  // (3*D, T)
        if (blk.qkv_b) qkv = ggml_add(g, qkv, blk.qkv_b);

        // Split Q, K, V: each (D, T) → reshape to (hd, nh, T)
        // qkv is (3*D, T) in ggml layout = ne[0]=3*D, ne[1]=T
        ggml_tensor *Q = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 0);
        ggml_tensor *K = ggml_view_2d(g, qkv, D, T, qkv->nb[1], D * sizeof(float));
        ggml_tensor *V = ggml_view_2d(g, qkv, D, T, qkv->nb[1], 2 * D * sizeof(float));

        // Reshape for multi-head: (hd, nh, T)
        Q = ggml_reshape_3d(g, ggml_cont(g, Q), hd, nh, T);
        K = ggml_reshape_3d(g, ggml_cont(g, K), hd, nh, T);
        V = ggml_reshape_3d(g, ggml_cont(g, V), hd, nh, T);

        // Attention: use flash_attn_ext for efficiency
        // flash_attn_ext expects: Q(hd, T, nh), K(hd, T_k, nh), V(hd, T_k, nh)
        Q = ggml_permute(g, Q, 0, 2, 1, 3);  // (hd, T, nh)
        K = ggml_permute(g, K, 0, 2, 1, 3);  // (hd, T, nh)
        V = ggml_permute(g, V, 0, 2, 1, 3);  // (hd, T, nh)
        // V needs to be (hd, T, nh) for flash_attn, which expects V contiguous
        V = ggml_cont(g, V);
        K = ggml_cont(g, K);

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor *attn_out = ggml_flash_attn_ext(g, Q, K, V,
                                                      nullptr,  // no mask (bidirectional)
                                                      scale, 0.0f, 0.0f);
        // attn_out: (hd, T, nh) → reshape to (D, T)
        attn_out = ggml_reshape_2d(g, attn_out, D, T);

        // Output projection
        ggml_tensor *attn_proj = ggml_mul_mat(g, blk.proj_w, attn_out);
        if (blk.proj_b) attn_proj = ggml_add(g, attn_proj, blk.proj_b);

        // LayerScale + residual
        if (blk.ls1) attn_proj = ggml_mul(g, attn_proj, blk.ls1);
        x = ggml_add(g, x, attn_proj);

        // ── MLP ──
        // Pre-norm
        h = layer_norm(x, blk.norm2_w, blk.norm2_b);

        // fc1 → GELU → fc2
        ggml_tensor *mlp = ggml_mul_mat(g, blk.fc1_w, h);
        if (blk.fc1_b) mlp = ggml_add(g, mlp, blk.fc1_b);
        mlp = ggml_gelu(g, mlp);
        mlp = ggml_mul_mat(g, blk.fc2_w, mlp);
        if (blk.fc2_b) mlp = ggml_add(g, mlp, blk.fc2_b);

        // LayerScale + residual
        if (blk.ls2) mlp = ggml_mul(g, mlp, blk.ls2);
        x = ggml_add(g, x, mlp);

        char name[64];
        snprintf(name, sizeof(name), "vis_layer_%d", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
        vg.layer_outputs.push_back(x);
    }

    ggml_build_forward_expand(gf, x);
    vg.gf = gf;
    vg.output = x;
    return vg;
}

// ── Pixel unshuffle (host-side, no ggml graph needed) ───────────────

// Pixel unshuffle v2: (n_patches, D) → (n_merged, merge_dim)
// n_patches = H*W, D = hidden_size
// n_merged = (H*ratio)*(W*ratio), merge_dim = D / ratio^2
void pixel_unshuffle_v2(const float *in, float *out,
                        int n_patches_per_side, int hidden_size,
                        float downsample_ratio) {
    int H = n_patches_per_side;
    int W = n_patches_per_side;
    int D = hidden_size;
    int scale = (int)(1.0f / downsample_ratio);  // 2
    int H2 = H / scale;
    int W2 = W / scale;
    int D2 = D * scale * scale;  // 4096

    // Allocate temp buffer for reshaping
    // Step 1: reshape (H, W, D) → (H, W/s, D*s)
    std::vector<float> tmp1(H * W2 * D * scale);
    for (int h = 0; h < H; h++) {
        for (int w2 = 0; w2 < W2; w2++) {
            for (int s = 0; s < scale; s++) {
                int w_orig = w2 * scale + s;
                const float *src = in + (h * W + w_orig) * D;
                float *dst = tmp1.data() + (h * W2 + w2) * D * scale + s * D;
                std::memcpy(dst, src, D * sizeof(float));
            }
        }
    }

    // Step 2: permute(1, 0, 2) → (W/s, H, D*s)
    int D_s = D * scale;
    std::vector<float> tmp2(W2 * H * D_s);
    for (int h = 0; h < H; h++) {
        for (int w2 = 0; w2 < W2; w2++) {
            const float *src = tmp1.data() + (h * W2 + w2) * D_s;
            float *dst = tmp2.data() + (w2 * H + h) * D_s;
            std::memcpy(dst, src, D_s * sizeof(float));
        }
    }

    // Step 3: reshape (W/s, H, D*s) → (W/s, H/s, D*s*s)
    std::vector<float> tmp3(W2 * H2 * D2);
    for (int w2 = 0; w2 < W2; w2++) {
        for (int h2 = 0; h2 < H2; h2++) {
            for (int s = 0; s < scale; s++) {
                int h_orig = h2 * scale + s;
                const float *src = tmp2.data() + (w2 * H + h_orig) * D_s;
                float *dst = tmp3.data() + (w2 * H2 + h2) * D2 + s * D_s;
                std::memcpy(dst, src, D_s * sizeof(float));
            }
        }
    }

    // Step 4: permute(1, 0, 2) → (H/s, W/s, D*s*s)
    for (int w2 = 0; w2 < W2; w2++) {
        for (int h2 = 0; h2 < H2; h2++) {
            const float *src = tmp3.data() + (w2 * H2 + h2) * D2;
            float *dst = out + (h2 * W2 + w2) * D2;
            std::memcpy(dst, src, D2 * sizeof(float));
        }
    }
}

// ── Projector graph ──────────────────────────────────────────────────

struct proj_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *input = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<uint8_t> meta;  // must outlive gctx
};

proj_graph build_proj_graph(context &ctx, int n_merged) {
    proj_graph pg;
    const auto &php = ctx.m.php;
    const int merge_dim = (int)php.merge_dim;      // 4096
    const int out_dim = (int)php.output_dim;        // 2048

    size_t meta_size = 32 * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(256, false);
    pg.meta.resize(meta_size);
    ggml_init_params ip{meta_size, pg.meta.data(), true};
    ggml_context *g = ggml_init(ip);

    ggml_cgraph *gf = ggml_new_graph_custom(g, 256, false);

    // Input: (merge_dim, n_merged) = (4096, 256)
    ggml_tensor *x = ggml_new_tensor_2d(g, GGML_TYPE_F32, merge_dim, n_merged);
    ggml_set_name(x, "proj_in");
    ggml_set_input(x);
    pg.input = x;

    // LayerNorm
    ggml_tensor *h = ggml_norm(g, x, 1e-6f);
    h = ggml_mul(g, h, ctx.m.proj.norm_w);
    if (ctx.m.proj.norm_b) h = ggml_add(g, h, ctx.m.proj.norm_b);

    // Linear 4096 → 2048
    h = ggml_mul_mat(g, ctx.m.proj.fc1_w, h);
    if (ctx.m.proj.fc1_b) h = ggml_add(g, h, ctx.m.proj.fc1_b);

    // GELU
    h = ggml_gelu(g, h);

    // Linear 2048 → 2048
    h = ggml_mul_mat(g, ctx.m.proj.fc2_w, h);
    if (ctx.m.proj.fc2_b) h = ggml_add(g, h, ctx.m.proj.fc2_b);

    ggml_set_name(h, "proj_out");
    ggml_set_output(h);

    ggml_build_forward_expand(gf, h);
    pg.gf = gf;
    pg.gctx = g;
    pg.output = h;
    return pg;
}

// ── KV cache allocation ──────────────────────────────────────────────

bool alloc_kv_cache(context &ctx, int max_seq) {
    const auto &lhp = ctx.m.lhp;
    const int hd = (int)lhp.head_dim;
    const int nkv = (int)lhp.num_key_value_heads;
    const int n_layers = (int)lhp.num_hidden_layers;

    // KV cache: (hd, max_seq, n_kv_heads, n_layers) F16
    size_t n_tensors = 2;  // k + v
    size_t ctx_size = n_tensors * ggml_tensor_overhead() + 256;
    ggml_init_params ip{ctx_size, nullptr, /*no_alloc=*/true};
    ctx.kvc.ctx = ggml_init(ip);

    ctx.kvc.k = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16,
                                     hd, max_seq, nkv, n_layers);
    ctx.kvc.v = ggml_new_tensor_4d(ctx.kvc.ctx, GGML_TYPE_F16,
                                     hd, max_seq, nkv, n_layers);
    ggml_set_name(ctx.kvc.k, "kv_k");
    ggml_set_name(ctx.kvc.v, "kv_v");

    ctx.kvc.buf = ggml_backend_alloc_ctx_tensors(ctx.kvc.ctx, ctx.backend);
    if (!ctx.kvc.buf) {
        fprintf(stderr, "internvl2_ocr: KV cache alloc failed (max_seq=%d)\n", max_seq);
        ggml_free(ctx.kvc.ctx);
        ctx.kvc.ctx = nullptr;
        return false;
    }

    // Zero-fill the cache
    ggml_backend_buffer_clear(ctx.kvc.buf, 0);

    ctx.kvc.max_seq = max_seq;
    ctx.kvc.n_past = 0;
    ctx.kvc.allocated = true;

    size_t kv_bytes = ggml_backend_buffer_get_size(ctx.kvc.buf);
    if (ctx.verbosity >= 1) {
        printf("  KV cache: %d layers, max_seq=%d, %.1f MB\n",
               n_layers, max_seq, (float)kv_bytes / 1024 / 1024);
    }
    return true;
}

void free_kv_cache(context &ctx) {
    if (ctx.kvc.buf) { ggml_backend_buffer_free(ctx.kvc.buf); ctx.kvc.buf = nullptr; }
    if (ctx.kvc.ctx) { ggml_free(ctx.kvc.ctx); ctx.kvc.ctx = nullptr; }
    ctx.kvc.allocated = false;
    ctx.kvc.n_past = 0;
}

// ── LLM decoder graph ───────────────────────────────────────────────

struct llm_graph {
    ggml_cgraph *gf = nullptr;
    ggml_context *gctx = nullptr;
    ggml_tensor *token_in = nullptr;    // (1, T) int32 token IDs
    ggml_tensor *output = nullptr;      // (D, T) hidden states
    ggml_tensor *logits_out = nullptr;  // (V, T) logits
    std::vector<ggml_tensor *> layer_outputs;
};

// Build LLM graph with optional KV cache.
// If kvc is null or unallocated, builds the old uncached graph (for parity testing).
// If kvc is allocated, writes new K/V at n_past, reads [0..n_past+T].
llm_graph build_llm_graph(context &ctx, int n_tokens, int n_past,
                          bool use_kv_cache = false) {
    llm_graph lg;
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int nh = (int)lhp.num_attention_heads;
    const int nkv = (int)lhp.num_key_value_heads;
    const int hd = (int)lhp.head_dim;
    const int inter = (int)lhp.intermediate_size;
    const int V = (int)lhp.vocab_size;
    const int T = n_tokens;
    const int n_layers = (int)lhp.num_hidden_layers;
    const float rms_eps = lhp.rms_norm_eps;
    const int kv_repeat = nh / nkv;
    const int Lk = use_kv_cache ? (n_past + T) : T;  // total KV length

    // KV cache path needs ~80 tensors/layer (projections, views, cpy, cont,
    // reshape, repeat for GQA, permute, flash_attn, ffn). Uncached ~50.
    int tensors_per_layer = use_kv_cache ? 80 : 50;
    size_t meta_size = (size_t)(n_layers * tensors_per_layer + 300) * ggml_tensor_overhead()
                       + ggml_graph_overhead_custom(32768, false);
    ctx.compute_meta.resize(meta_size);
    ggml_init_params ip{meta_size, ctx.compute_meta.data(), true};
    ggml_context *g = ggml_init(ip);

    ggml_cgraph *gf = ggml_new_graph_custom(g, 32768, false);

    // Input token IDs
    ggml_tensor *token_in = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(token_in, "token_ids");
    ggml_set_input(token_in);
    lg.token_in = token_in;

    // Token embedding lookup
    ggml_tensor *x = ggml_get_rows(g, ctx.m.embed_tokens, token_in);

    ggml_set_name(x, "llm_embed");
    ggml_set_output(x);

    // Vision-text splicing: replace embeddings at image_token positions
    // with vision encoder output. Uses mask-based splicing:
    //   x = embed * keep_mask + image_patches
    // where keep_mask is 1.0 for text tokens, 0.0 for image positions,
    // and image_patches has vision embeds at image positions, 0 elsewhere.
    ggml_tensor *image_embeds_in = nullptr;
    ggml_tensor *splice_mask = nullptr;
    if (n_past == 0) {
        // Only splice during prefill (n_past == 0)
        image_embeds_in = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(image_embeds_in, "image_embeds");
        ggml_set_input(image_embeds_in);

        splice_mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
        ggml_set_name(splice_mask, "splice_mask");
        ggml_set_input(splice_mask);

        // x = x * mask + image_embeds * (1 - mask)
        // Simplify: x = x * mask + image_embeds_in
        // where image_embeds_in is already zeroed at text positions
        x = ggml_add(g, ggml_mul(g, x, splice_mask), image_embeds_in);
    }

    // RoPE position IDs
    ggml_tensor *rope_pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(rope_pos, "rope_pos");
    ggml_set_input(rope_pos);

    // Causal mask: (Lk, T) F16 — flash_attn_ext mask shape is (kv_len, q_len)
    // For uncached: (T, T). For cached: (n_past+T, T).
    ggml_tensor *mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "causal_mask");
    ggml_set_input(mask);

    auto rmsnorm = [&](ggml_tensor *t, ggml_tensor *w) -> ggml_tensor * {
        return ggml_mul(g, ggml_rms_norm(g, t, rms_eps), w);
    };

    for (int i = 0; i < n_layers; i++) {
        auto &ly = ctx.m.llm_layers[i];

        // ── Self-attention ──
        ggml_tensor *h = rmsnorm(x, ly.attn_norm_w);

        // Q/K/V projections (optional bias for Qwen2)
        ggml_tensor *Q = ggml_mul_mat(g, ly.q_w, h);
        if (ly.q_b) Q = ggml_add(g, Q, ly.q_b);
        ggml_tensor *K_new = ggml_mul_mat(g, ly.k_w, h);
        if (ly.k_b) K_new = ggml_add(g, K_new, ly.k_b);
        ggml_tensor *V_new = ggml_mul_mat(g, ly.v_w, h);
        if (ly.v_b) V_new = ggml_add(g, V_new, ly.v_b);

        Q = ggml_reshape_3d(g, Q, hd, nh, T);
        K_new = ggml_reshape_3d(g, K_new, hd, nkv, T);
        V_new = ggml_reshape_3d(g, V_new, hd, nkv, T);

        // Apply RoPE
        Q = ggml_rope_ext(g, Q, rope_pos, nullptr,
                          hd, GGML_ROPE_TYPE_NEOX, 0,
                          lhp.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_new = ggml_rope_ext(g, K_new, rope_pos, nullptr,
                              hd, GGML_ROPE_TYPE_NEOX, 0,
                              lhp.rope_theta,
                              1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor *Kfull, *Vfull;

        if (use_kv_cache && ctx.kvc.allocated) {
            // ── Write new K/V into cache at [n_past..n_past+T) ──
            ggml_tensor *K_new_perm = ggml_permute(g, K_new, 0, 2, 1, 3); // (hd, T, nkv)
            ggml_tensor *V_new_perm = ggml_permute(g, V_new, 0, 2, 1, 3);

            // View into cache for this layer at n_past offset
            ggml_tensor *k_view = ggml_view_4d(g, ctx.kvc.k,
                hd, T, nkv, 1,
                ctx.kvc.k->nb[1], ctx.kvc.k->nb[2], ctx.kvc.k->nb[3],
                (size_t)i * ctx.kvc.k->nb[3] + (size_t)n_past * ctx.kvc.k->nb[1]);
            ggml_tensor *v_view = ggml_view_4d(g, ctx.kvc.v,
                hd, T, nkv, 1,
                ctx.kvc.v->nb[1], ctx.kvc.v->nb[2], ctx.kvc.v->nb[3],
                (size_t)i * ctx.kvc.v->nb[3] + (size_t)n_past * ctx.kvc.v->nb[1]);

            ggml_build_forward_expand(gf, ggml_cpy(g, K_new_perm, k_view));
            ggml_build_forward_expand(gf, ggml_cpy(g, V_new_perm, v_view));

            // ── Read full K/V history [0..n_past+T) ──
            ggml_tensor *k_layer = ggml_view_3d(g, ctx.kvc.k,
                hd, Lk, nkv,
                ctx.kvc.k->nb[1], ctx.kvc.k->nb[2],
                (size_t)i * ctx.kvc.k->nb[3]);
            ggml_tensor *v_layer = ggml_view_3d(g, ctx.kvc.v,
                hd, Lk, nkv,
                ctx.kvc.v->nb[1], ctx.kvc.v->nb[2],
                (size_t)i * ctx.kvc.v->nb[3]);

            // Cont to make contiguous (F16→F16 copy)
            Kfull = ggml_cont(g, k_layer);  // (hd, Lk, nkv)
            Vfull = ggml_cont(g, v_layer);

            // GQA expansion: (hd, Lk, nkv) → (hd, Lk, nh)
            if (kv_repeat > 1) {
                Kfull = ggml_reshape_4d(g, Kfull, hd, Lk, 1, nkv);
                ggml_tensor *K_tgt = ggml_new_tensor_4d(g, Kfull->type, hd, Lk, kv_repeat, nkv);
                Kfull = ggml_repeat(g, Kfull, K_tgt);
                Kfull = ggml_reshape_3d(g, Kfull, hd, Lk, nh);

                Vfull = ggml_reshape_4d(g, Vfull, hd, Lk, 1, nkv);
                ggml_tensor *V_tgt = ggml_new_tensor_4d(g, Vfull->type, hd, Lk, kv_repeat, nkv);
                Vfull = ggml_repeat(g, Vfull, V_tgt);
                Vfull = ggml_reshape_3d(g, Vfull, hd, Lk, nh);
            }
        } else {
            // ── No KV cache: use current K/V directly ──
            if (kv_repeat > 1) {
                K_new = ggml_reshape_4d(g, K_new, hd, 1, nkv, T);
                ggml_tensor *K_tgt = ggml_new_tensor_4d(g, K_new->type, hd, kv_repeat, nkv, T);
                K_new = ggml_repeat(g, K_new, K_tgt);
                K_new = ggml_reshape_3d(g, K_new, hd, nh, T);

                V_new = ggml_reshape_4d(g, V_new, hd, 1, nkv, T);
                ggml_tensor *V_tgt = ggml_new_tensor_4d(g, V_new->type, hd, kv_repeat, nkv, T);
                V_new = ggml_repeat(g, V_new, V_tgt);
                V_new = ggml_reshape_3d(g, V_new, hd, nh, T);
            }
            // Permute to (hd, T, nh) for flash_attn
            Kfull = ggml_cont(g, ggml_permute(g, K_new, 0, 2, 1, 3));
            Vfull = ggml_cont(g, ggml_permute(g, V_new, 0, 2, 1, 3));
        }

        // ── Flash attention ──
        // Q: (hd, nh, T) → permute to (hd, T, nh)
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));

        float scale = 1.0f / std::sqrt((float)hd);
        ggml_tensor *attn_out = ggml_flash_attn_ext(g, Q, Kfull, Vfull,
                                                      mask, scale, 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(g, attn_out, D, T);

        // Output projection
        attn_out = ggml_mul_mat(g, ly.o_w, attn_out);

        x = ggml_add(g, x, attn_out);

        // ── SwiGLU FFN ──
        h = rmsnorm(x, ly.ffn_norm_w);
        ggml_tensor *gate = ggml_mul_mat(g, ly.ffn_gate_w, h);
        gate = ggml_silu(g, gate);
        ggml_tensor *up = ggml_mul_mat(g, ly.ffn_up_w, h);
        ggml_tensor *ffn = ggml_mul(g, gate, up);
        ffn = ggml_mul_mat(g, ly.ffn_down_w, ffn);
        x = ggml_add(g, x, ffn);

        char name[64];
        snprintf(name, sizeof(name), "llm_layer_%d", i);
        ggml_set_name(x, name);
        ggml_set_output(x);
        lg.layer_outputs.push_back(x);
    }

    // Final norm
    x = rmsnorm(x, ctx.m.output_norm_w);

    // LM head
    if (ctx.m.lm_head_w) {
        ggml_tensor *logits = ggml_mul_mat(g, ctx.m.lm_head_w, x);
        ggml_set_name(logits, "logits");
        ggml_set_output(logits);
        lg.logits_out = logits;
        ggml_build_forward_expand(gf, logits);
    } else {
        ggml_build_forward_expand(gf, x);
    }

    lg.gf = gf;
    lg.gctx = g;
    lg.output = x;
    return lg;
}

}  // anonymous namespace

// ── Public API ───────────────────────────────────────────────────────

bool load(context &ctx, const char *gguf_path, int n_threads, int verbosity) {
    ctx.n_threads = n_threads;
    ctx.verbosity = verbosity;

    if (verbosity >= 1) {
        printf("internvl2_ocr: loading %s\n", gguf_path);
    }

    // Load hparams
    if (!load_hparams(ctx, gguf_path)) {
        fprintf(stderr, "internvl2_ocr: failed to load hparams\n");
        return false;
    }

    if (verbosity >= 1) {
        printf("  Vision: %uL, %ud, %uH, patch=%u, image=%u\n",
               ctx.m.vhp.num_hidden_layers, ctx.m.vhp.hidden_size,
               ctx.m.vhp.num_attention_heads, ctx.m.vhp.patch_size,
               ctx.m.vhp.image_size);
        printf("  Projector: ratio=%.1f, %u→%u tokens, dim %u→%u\n",
               ctx.m.php.downsample_ratio, ctx.m.vhp.n_patches,
               ctx.m.php.n_merged_tokens, ctx.m.php.merge_dim,
               ctx.m.php.output_dim);
        printf("  LLM: %uL, %ud, %uH/%uKV, inter=%u, vocab=%u\n",
               ctx.m.lhp.num_hidden_layers, ctx.m.lhp.hidden_size,
               ctx.m.lhp.num_attention_heads, ctx.m.lhp.num_key_value_heads,
               ctx.m.lhp.intermediate_size, ctx.m.lhp.vocab_size);
    }

    // Init backend — prefer GPU when available
    bool force_cpu = (getenv("INTERNVL2_OCR_FORCE_CPU") && atoi(getenv("INTERNVL2_OCR_FORCE_CPU")));
    ctx.backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx.backend) ctx.backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx.backend))
        ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
    // CPU fallback for scheduler (sched asserts last backend is CPU)
    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);

    // Load tensors
    if (!load_tensors(ctx, gguf_path)) {
        fprintf(stderr, "internvl2_ocr: failed to load tensors\n");
        return false;
    }

    // Init scheduler — GPU + CPU fallback when available
    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr,
                                       (int)backends.size(), 16384, false, false);

    if (verbosity >= 1) {
        const char *bname = ggml_backend_is_cpu(ctx.backend) ? "CPU" : "GPU";
        printf("  Ready (%s, %d threads)\n", bname, n_threads);
    }

    return true;
}

void free_(context &ctx) {
    free_kv_cache(ctx);
    if (ctx.sched) { ggml_backend_sched_free(ctx.sched); ctx.sched = nullptr; }
    if (ctx.model_buf) { ggml_backend_buffer_free(ctx.model_buf); ctx.model_buf = nullptr; }
    if (ctx.model_ctx) { ggml_free(ctx.model_ctx); ctx.model_ctx = nullptr; }
    if (ctx.backend_cpu) { ggml_backend_free(ctx.backend_cpu); ctx.backend_cpu = nullptr; }
    if (ctx.backend) { ggml_backend_free(ctx.backend); ctx.backend = nullptr; }
}

bool encode_vision_tile(context &ctx, const float *pixels, vision_result &out) {
    const auto &vhp = ctx.m.vhp;
    const int D = (int)vhp.hidden_size;
    const int P = (int)vhp.patch_size;
    const int n_patches = (int)vhp.n_patches;
    const int n_pos = (int)vhp.n_positions;
    const int patch_flat = 3 * P * P;

    // Extract patches from (3, 448, 448) → (n_patches, 3*P*P)
    int n_ph = (int)vhp.n_patches_per_side;
    int n_pw = n_ph;
    std::vector<float> patches(n_patches * patch_flat);
    int idx = 0;
    for (int ph = 0; ph < n_ph; ph++) {
        for (int pw = 0; pw < n_pw; pw++) {
            for (int c = 0; c < 3; c++) {
                for (int py = 0; py < P; py++) {
                    for (int px = 0; px < P; px++) {
                        int y = ph * P + py;
                        int x = pw * P + px;
                        patches[idx * patch_flat + c * P * P + py * P + px] =
                            pixels[c * vhp.image_size * vhp.image_size + y * vhp.image_size + x];
                    }
                }
            }
            idx++;
        }
    }

    // Build and compute vision graph
    vision_graph vg = build_vision_graph(ctx);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, vg.gf)) {
        fprintf(stderr, "internvl2_ocr: vision graph alloc failed\n");
        ggml_free(vg.gctx);
        return false;
    }

    // Set input
    ggml_backend_tensor_set(vg.pixel_in, patches.data(), 0,
                            n_patches * patch_flat * sizeof(float));

    // Compute
    ggml_backend_sched_graph_compute(ctx.sched, vg.gf);

    // Read output: (D, n_pos)
    out.n_tokens = n_pos;
    out.hidden_dim = D;
    out.hidden = (float *)malloc(n_pos * D * sizeof(float));
    ggml_backend_tensor_get(vg.output, out.hidden, 0,
                            n_pos * D * sizeof(float));

    // Diff comparison if enabled
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            // Compare patch embed
            {
                float *vis_pe = (float *)malloc(n_pos * D * sizeof(float));
                ggml_tensor *pe_t = ggml_graph_get_tensor(vg.gf, "vis_patch_embed");
                if (pe_t) {
                    ggml_backend_tensor_get(pe_t, vis_pe, 0, n_pos * D * sizeof(float));
                    auto r = ref.compare("vis_patch_embed", vis_pe, n_pos * D);
                    printf("  vis_patch_embed: cos=%.6f max_abs=%.6f %s\n",
                           r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
                free(vis_pe);
            }
            // Compare layer outputs
            for (size_t i = 0; i < vg.layer_outputs.size(); i++) {
                char name[64];
                snprintf(name, sizeof(name), "vis_layer_%zu", i);
                float *buf = (float *)malloc(n_pos * D * sizeof(float));
                ggml_backend_tensor_get(vg.layer_outputs[i], buf, 0,
                                        n_pos * D * sizeof(float));
                auto r = ref.compare(name, buf, n_pos * D);
                printf("  %s: cos=%.6f max_abs=%.6f %s\n",
                       name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                free(buf);
            }
        }
    }

    ggml_free(vg.gctx);
    return true;
}

bool project_vision(context &ctx, const float *vis_hidden, int n_patches,
                    project_result &out) {
    const auto &vhp = ctx.m.vhp;
    const auto &php = ctx.m.php;
    const int n_merged = (int)php.n_merged_tokens;
    const int merge_dim = (int)php.merge_dim;
    const int out_dim = (int)php.output_dim;

    // Pixel unshuffle: (n_patches, D) → (n_merged, merge_dim)
    std::vector<float> unshuffled(n_merged * merge_dim);
    pixel_unshuffle_v2(vis_hidden, unshuffled.data(),
                       (int)vhp.n_patches_per_side, (int)vhp.hidden_size,
                       php.downsample_ratio);

    // Build and compute projector graph
    proj_graph pg = build_proj_graph(ctx, n_merged);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, pg.gf)) {
        fprintf(stderr, "internvl2_ocr: projector graph alloc failed\n");
        ggml_free(pg.gctx);
        return false;
    }

    ggml_backend_tensor_set(pg.input, unshuffled.data(), 0,
                            n_merged * merge_dim * sizeof(float));

    ggml_backend_sched_graph_compute(ctx.sched, pg.gf);

    out.n_tokens = n_merged;
    out.embed_dim = out_dim;
    out.embeds = (float *)malloc(n_merged * out_dim * sizeof(float));
    ggml_backend_tensor_get(pg.output, out.embeds, 0,
                            n_merged * out_dim * sizeof(float));

    ggml_free(pg.gctx);
    return true;
}

bool encode_vision(context &ctx, const float *tiles, int n_tiles,
                   vision_pipeline_result &out) {
    const auto &vhp = ctx.m.vhp;
    const auto &php = ctx.m.php;
    const int img_size = (int)vhp.image_size;
    const int tile_pixels = 3 * img_size * img_size;
    const int n_merged_per_tile = (int)php.n_merged_tokens;
    const int out_dim = (int)php.output_dim;

    int total_tokens = n_tiles * n_merged_per_tile;
    float *all_embeds = (float *)malloc(total_tokens * out_dim * sizeof(float));

    for (int t = 0; t < n_tiles; t++) {
        // Encode tile
        vision_result vr;
        if (!encode_vision_tile(ctx, tiles + t * tile_pixels, vr)) {
            free(all_embeds);
            return false;
        }

        // Remove CLS token: skip first row
        const float *no_cls = vr.hidden + vr.hidden_dim;  // skip (D,) for CLS
        int n_patches = vr.n_tokens - 1;

        // Project
        project_result pr;
        if (!project_vision(ctx, no_cls, n_patches, pr)) {
            free(vr.hidden);
            free(all_embeds);
            return false;
        }

        // Copy to output buffer
        std::memcpy(all_embeds + t * n_merged_per_tile * out_dim,
                    pr.embeds, n_merged_per_tile * out_dim * sizeof(float));

        free(vr.hidden);
        free(pr.embeds);
    }

    out.image_embeds = all_embeds;
    out.n_image_tokens = total_tokens;
    out.embed_dim = out_dim;
    return true;
}

bool run_llm_forward(context &ctx, const int32_t *token_ids, int n_tokens,
                     llm_result &out, const image_input *img) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int V = (int)lhp.vocab_size;
    const int T = n_tokens;

    llm_graph lg = build_llm_graph(ctx, T, /*n_past=*/0, /*use_kv_cache=*/false);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        fprintf(stderr, "internvl2_ocr: LLM graph alloc failed\n");
        ggml_free(lg.gctx);
        return false;
    }

    // Set token IDs
    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // Set position IDs: sequential 0..T-1
    std::vector<int32_t> pos_ids(T);
    for (int i = 0; i < T; i++) pos_ids[i] = i;
    ggml_tensor *rope_pos = ggml_graph_get_tensor(lg.gf, "rope_pos");
    if (rope_pos) {
        ggml_backend_tensor_set(rope_pos, pos_ids.data(), 0, T * sizeof(int32_t));
    }

    // Set causal mask: upper triangular = -inf (F16)
    std::vector<ggml_fp16_t> mask_data(T * T);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            mask_data[i * T + j] = ggml_fp32_to_fp16((j > i) ? -INFINITY : 0.0f);
        }
    }
    ggml_tensor *mask = ggml_graph_get_tensor(lg.gf, "causal_mask");
    if (mask) {
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                T * T * sizeof(ggml_fp16_t));
    }

    // Set splice inputs to identity (no image tokens in parity test)
    ggml_tensor *splice_m = ggml_graph_get_tensor(lg.gf, "splice_mask");
    ggml_tensor *img_e = ggml_graph_get_tensor(lg.gf, "image_embeds");
    if (splice_m) {
        std::vector<float> ones((size_t)D * T, 1.0f);
        ggml_backend_tensor_set(splice_m, ones.data(), 0, D * T * sizeof(float));
    }
    if (img_e) {
        std::vector<float> zeros((size_t)D * T, 0.0f);
        ggml_backend_tensor_set(img_e, zeros.data(), 0, D * T * sizeof(float));
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    // Read output
    out.n_tokens = T;
    out.hidden_dim = D;
    out.hidden = (float *)malloc(T * D * sizeof(float));
    ggml_backend_tensor_get(lg.output, out.hidden, 0, T * D * sizeof(float));

    if (lg.logits_out) {
        out.vocab_size = V;
        out.logits = (float *)malloc(T * V * sizeof(float));
        ggml_backend_tensor_get(lg.logits_out, out.logits, 0,
                                T * V * sizeof(float));
    }

    // Diff comparison
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str())) {
            // Compare embedding
            {
                float *buf = (float *)malloc(T * D * sizeof(float));
                ggml_tensor *e = ggml_graph_get_tensor(lg.gf, "llm_embed");
                if (e) {
                    ggml_backend_tensor_get(e, buf, 0, T * D * sizeof(float));
                    auto r = ref.compare("llm_embed", buf, T * D);
                    printf("  llm_embed: cos=%.6f max_abs=%.6f %s\n",
                           r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
                free(buf);
            }
            // Compare debug intermediates (if present in ref)
            // Dump raw rms_norm output for debugging
            // Compare layer outputs
            for (size_t i = 0; i < lg.layer_outputs.size(); i++) {
                char name[64];
                snprintf(name, sizeof(name), "llm_layer_%zu", i);
                float *buf = (float *)malloc(T * D * sizeof(float));
                ggml_backend_tensor_get(lg.layer_outputs[i], buf, 0,
                                        T * D * sizeof(float));
                auto r = ref.compare(name, buf, T * D);
                printf("  %s: cos=%.6f max_abs=%.6f %s\n",
                       name, r.cos_min, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                free(buf);
            }
        }
    }

    ggml_free(lg.gctx);
    return true;
}

// Helper: run a single LLM forward step with KV cache.
// Returns logits for the last token only (V floats).
// Splice data for vision-text interleaving during prefill.
struct splice_data {
    const float *image_embeds = nullptr;  // (D, n_image_tokens)
    int n_image_tokens = 0;
    // Map: for each token position, if it's an image token, the index
    // into image_embeds; else -1.
    const int *token_to_image = nullptr;  // (n_tokens,)
};

static bool run_cached_step(context &ctx, const int32_t *token_ids, int n_tokens,
                            int n_past, std::vector<float> &last_logits_out,
                            const splice_data *splice = nullptr) {
    const auto &lhp = ctx.m.lhp;
    const int D = (int)lhp.hidden_size;
    const int V = (int)lhp.vocab_size;
    const int T = n_tokens;
    const int Lk = n_past + T;

    llm_graph lg = build_llm_graph(ctx, T, n_past, /*use_kv_cache=*/true);

    ggml_backend_sched_reset(ctx.sched);
    if (!ggml_backend_sched_alloc_graph(ctx.sched, lg.gf)) {
        fprintf(stderr, "internvl2_ocr: cached step graph alloc failed\n");
        ggml_free(lg.gctx);
        return false;
    }

    // Set token IDs
    ggml_backend_tensor_set(lg.token_in, token_ids, 0, T * sizeof(int32_t));

    // Set position IDs: [n_past, n_past+1, ..., n_past+T-1]
    std::vector<int32_t> pos_ids(T);
    for (int i = 0; i < T; i++) pos_ids[i] = n_past + i;
    ggml_tensor *rope_pos = ggml_graph_get_tensor(lg.gf, "rope_pos");
    if (rope_pos) {
        ggml_backend_tensor_set(rope_pos, pos_ids.data(), 0, T * sizeof(int32_t));
    }

    // Set causal mask: (Lk, T) — each new token attends to all past + self
    // Row i (query at position n_past+i) can attend to positions [0..n_past+i]
    std::vector<ggml_fp16_t> mask_data((size_t)Lk * T);
    for (int qi = 0; qi < T; qi++) {
        int q_pos = n_past + qi;
        for (int ki = 0; ki < Lk; ki++) {
            mask_data[(size_t)qi * Lk + ki] =
                ggml_fp32_to_fp16((ki > q_pos) ? -INFINITY : 0.0f);
        }
    }
    ggml_tensor *mask = ggml_graph_get_tensor(lg.gf, "causal_mask");
    if (mask) {
        ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                (size_t)Lk * T * sizeof(ggml_fp16_t));
    }

    // Set vision splice inputs (only during prefill, n_past == 0)
    if (splice && n_past == 0) {
        const int D = (int)ctx.m.lhp.hidden_size;

        // Build image_embeds tensor: (D, T) with vision embeds at image positions
        std::vector<float> img_data((size_t)D * T, 0.0f);
        // Build keep_mask: (D, T) with 1.0 for text, 0.0 for image
        std::vector<float> mask_f((size_t)D * T, 1.0f);

        if (splice->token_to_image) {
            for (int t = 0; t < T; t++) {
                int img_idx = splice->token_to_image[t];
                if (img_idx >= 0 && img_idx < splice->n_image_tokens) {
                    // This position is an image token
                    for (int d = 0; d < D; d++) {
                        img_data[(size_t)t * D + d] =
                            splice->image_embeds[(size_t)img_idx * D + d];
                        mask_f[(size_t)t * D + d] = 0.0f;
                    }
                }
            }
        }

        ggml_tensor *img_t = ggml_graph_get_tensor(lg.gf, "image_embeds");
        ggml_tensor *mask_t = ggml_graph_get_tensor(lg.gf, "splice_mask");
        if (img_t) {
            ggml_backend_tensor_set(img_t, img_data.data(), 0,
                                    (size_t)D * T * sizeof(float));
        }
        if (mask_t) {
            ggml_backend_tensor_set(mask_t, mask_f.data(), 0,
                                    (size_t)D * T * sizeof(float));
        }
    } else if (n_past == 0) {
        // No splice data, but graph has splice inputs — fill with identity
        const int D = (int)ctx.m.lhp.hidden_size;
        ggml_tensor *img_t = ggml_graph_get_tensor(lg.gf, "image_embeds");
        ggml_tensor *mask_t = ggml_graph_get_tensor(lg.gf, "splice_mask");
        if (img_t) {
            std::vector<float> zeros((size_t)D * T, 0.0f);
            ggml_backend_tensor_set(img_t, zeros.data(), 0,
                                    (size_t)D * T * sizeof(float));
        }
        if (mask_t) {
            std::vector<float> ones((size_t)D * T, 1.0f);
            ggml_backend_tensor_set(mask_t, ones.data(), 0,
                                    (size_t)D * T * sizeof(float));
        }
    }

    // Compute
    ggml_backend_sched_graph_compute(ctx.sched, lg.gf);

    // Read logits for the last token only
    if (lg.logits_out) {
        last_logits_out.resize(V);
        // logits: (V, T) — last token at offset (T-1)*V
        ggml_backend_tensor_get(lg.logits_out, last_logits_out.data(),
                                (size_t)(T - 1) * V * sizeof(float),
                                V * sizeof(float));
    }

    ggml_free(lg.gctx);
    return true;
}

bool generate(context &ctx,
              const float *image_embeds, int n_image_tokens, int embed_dim,
              const int32_t *prompt_token_ids, int n_prompt_tokens,
              int max_new_tokens,
              generate_result &out) {
    const auto &lhp = ctx.m.lhp;
    const int V = (int)lhp.vocab_size;
    const int eos_id = (int)lhp.eos_token_id;
    const int im_end_id = ctx.tok.im_end_id;
    const int max_seq = n_prompt_tokens + max_new_tokens + 16;

    // Allocate KV cache if needed
    if (!ctx.kvc.allocated || ctx.kvc.max_seq < max_seq) {
        free_kv_cache(ctx);
        if (!alloc_kv_cache(ctx, max_seq)) {
            fprintf(stderr, "internvl2_ocr: KV cache alloc failed\n");
            return false;
        }
    }
    ctx.kvc.n_past = 0;  // reset for new generation

    // Build splice mapping if we have image embeddings
    splice_data sd = {};
    std::vector<int> token_to_image;
    if (image_embeds && n_image_tokens > 0) {
        sd.image_embeds = image_embeds;
        sd.n_image_tokens = n_image_tokens;
        // Build token→image mapping: find image_token_id positions in prompt
        int img_idx = 0;
        token_to_image.resize(n_prompt_tokens, -1);
        int img_token_id = (int)lhp.image_token_id;
        for (int t = 0; t < n_prompt_tokens && img_idx < n_image_tokens; t++) {
            if (prompt_token_ids[t] == img_token_id) {
                token_to_image[t] = img_idx++;
            }
        }
        sd.token_to_image = token_to_image.data();
        if (ctx.verbosity >= 1) {
            fprintf(stderr, "  Spliced %d image tokens into %d prompt tokens\n",
                    img_idx, n_prompt_tokens);
        }
    }

    // ── Prefill: process all prompt tokens at once ──
    std::vector<float> logits;
    const splice_data *sd_ptr = (image_embeds && n_image_tokens > 0) ? &sd : nullptr;
    if (!run_cached_step(ctx, prompt_token_ids, n_prompt_tokens, 0, logits, sd_ptr)) {
        fprintf(stderr, "internvl2_ocr: prefill failed\n");
        return false;
    }
    ctx.kvc.n_past = n_prompt_tokens;

    // Greedy argmax on prefill logits (last token)
    int best_id = 0;
    float best_score = -INFINITY;
    for (int v = 0; v < V; v++) {
        if (logits[v] > best_score) {
            best_score = logits[v];
            best_id = v;
        }
    }
    out.token_ids.push_back(best_id);
    if (ctx.verbosity >= 1) {
        fprintf(stderr, "  gen[0]: token=%d score=%.2f (prefill %d tokens)\n",
                best_id, best_score, n_prompt_tokens);
    }
    if (best_id == eos_id || best_id == im_end_id) {
        out.text = ctx.tok.decode(out.token_ids);
        return true;
    }

    // ── Decode: one token at a time with KV cache ──
    for (int gen = 1; gen < max_new_tokens; gen++) {
        int32_t next_token = best_id;

        if (!run_cached_step(ctx, &next_token, 1, ctx.kvc.n_past, logits)) {
            fprintf(stderr, "internvl2_ocr: decode step %d failed\n", gen);
            return false;
        }
        ctx.kvc.n_past += 1;

        best_id = 0;
        best_score = -INFINITY;
        for (int v = 0; v < V; v++) {
            if (logits[v] > best_score) {
                best_score = logits[v];
                best_id = v;
            }
        }

        out.token_ids.push_back(best_id);
        if (ctx.verbosity >= 1) {
            fprintf(stderr, "  gen[%d]: token=%d score=%.2f\n", gen, best_id, best_score);
        }
        if (best_id == eos_id || best_id == im_end_id) break;
    }

    // Decode generated tokens to text
    out.text = ctx.tok.decode(out.token_ids);
    return true;
}

}  // namespace internvl2_ocr

// ── C ABI wrappers ───────────────────────────────────────────────────

struct internvl2_ocr_context {
    internvl2_ocr::context ctx;
    std::string last_text;
    std::string prompt = "OCR this image.";
    int max_tokens = 512;
};

internvl2_ocr_context * internvl2_ocr_init(const char *model_path, int n_threads) {
    auto *c = new internvl2_ocr_context();
    if (!internvl2_ocr::load(c->ctx, model_path, n_threads, 1)) {
        delete c;
        return nullptr;
    }
    return c;
}

void internvl2_ocr_free(internvl2_ocr_context *ctx) {
    if (ctx) {
        internvl2_ocr::free_(ctx->ctx);
        delete ctx;
    }
}

void internvl2_ocr_set_prompt(internvl2_ocr_context *ctx, const char *prompt) {
    if (ctx && prompt) ctx->prompt = prompt;
}

void internvl2_ocr_set_max_tokens(internvl2_ocr_context *ctx, int max_tokens) {
    if (ctx) ctx->max_tokens = max_tokens;
}

const char * internvl2_ocr_recognize_raw(
    internvl2_ocr_context *ctx,
    const uint8_t *pixel_bytes,
    int width, int height, int channels,
    int *out_len) {
    if (!ctx || !pixel_bytes) {
        if (out_len) *out_len = 0;
        return "";
    }

    // Use proper dynamic tiling preprocessor from image_preprocess.cpp
    const auto &vhp = ctx->ctx.m.vhp;
    image_preproc::internvl_config cfg;
    cfg.image_size        = (int)vhp.image_size;   // 448
    cfg.min_dynamic_patch = 1;
    cfg.max_dynamic_patch = 12;
    cfg.use_thumbnail     = true;
    for (int i = 0; i < 3; i++) {
        cfg.mean[i] = vhp.image_mean[i];
        cfg.std[i]  = vhp.image_std[i];
    }

    image_preproc::internvl_result pp;
    if (!image_preproc::preprocess_internvl_rgb(pixel_bytes, height, width, channels, cfg, pp)) {
        fprintf(stderr, "internvl2_ocr: image preprocessing failed\n");
        if (out_len) *out_len = 0;
        return "";
    }

    fprintf(stderr, "internvl2_ocr: %dx%d → %d tiles (%dx%d grid, %dpx)\n",
            width, height, pp.n_tiles, pp.grid_rows, pp.grid_cols, pp.tile_size);

    // Encode vision with all tiles (dynamic tiling + optional thumbnail)
    internvl2_ocr::vision_pipeline_result vpr;
    if (!internvl2_ocr::encode_vision(ctx->ctx, pp.tiles.data(), pp.n_tiles, vpr)) {
        if (out_len) *out_len = 0;
        return "";
    }

    // Build prompt tokens using proper tokenizer
    int n_img_tokens = vpr.n_image_tokens;
    std::vector<int32_t> prompt_tokens = ctx->ctx.tok.build_prompt(
        ctx->prompt, n_img_tokens);

    // Generate
    internvl2_ocr::generate_result gen;
    bool ok = internvl2_ocr::generate(ctx->ctx,
        vpr.image_embeds, vpr.n_image_tokens, vpr.embed_dim,
        prompt_tokens.data(), (int)prompt_tokens.size(),
        ctx->max_tokens, gen);

    free(vpr.image_embeds);

    if (!ok) {
        if (out_len) *out_len = 0;
        return "";
    }

    ctx->last_text = ctx->ctx.tok.decode(gen.token_ids);

    if (out_len) *out_len = (int)ctx->last_text.size();
    return ctx->last_text.c_str();
}

const char * internvl2_ocr_recognize(
    internvl2_ocr_context *ctx,
    const float *pixels,
    int width, int height,
    int *out_len) {
    // Convert grayscale float to RGB uint8 and delegate
    if (!ctx || !pixels) {
        if (out_len) *out_len = 0;
        return "";
    }
    std::vector<uint8_t> rgb(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, pixels[i] * 255.0f));
        rgb[i * 3 + 0] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    return internvl2_ocr_recognize_raw(ctx, rgb.data(), width, height, 3, out_len);
}
