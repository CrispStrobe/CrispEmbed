// decoder_embed.cpp — Qwen3/LLaMA/Gemma3 decoder embedding graph via ggml.
// Uses ggml_backend_sched for GPU dispatch (same pattern as encoder).
//
// Multimodal extension (BidirLM-Omni): when `dec_image_input` is supplied,
// the graph (a) splices `image_embeds` rows into the post-token-embed
// hidden state at every `image_token_id` position, (b) adds
// `deepstack[k]` rows at the same positions after the first n_deepstack
// transformer layers, and (c) uses 3D interleaved-MRoPE position ids
// derived from `grid_thw`. See HF BidirLMOmniModel.forward + get_rope_index.

#include "decoder_embed_internal.h"
#include "crispembed.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

bool load_decoder_model(dec_model & m, core_gguf::WeightLoad & wl,
                         const char * path, ggml_backend_t backend) {
    gguf_init_params gp = { true, nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;

    // Type-safe GGUF value readers (handle uint32, int32, bool, float)
    auto u32_safe = [&](int k) -> int {
        auto type = gguf_get_kv_type(g, k);
        if (type == GGUF_TYPE_UINT32) return (int)gguf_get_val_u32(g, k);
        if (type == GGUF_TYPE_INT32) return (int)gguf_get_val_i32(g, k);
        if (type == GGUF_TYPE_BOOL) return gguf_get_val_bool(g, k) ? 1 : 0;
        if (type == GGUF_TYPE_UINT16) return (int)gguf_get_val_u16(g, k);
        if (type == GGUF_TYPE_FLOAT32) return (int)gguf_get_val_f32(g, k);
        return 0;
    };
    auto f32_safe = [&](int k) -> float {
        auto type = gguf_get_kv_type(g, k);
        if (type == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(g, k);
        if (type == GGUF_TYPE_UINT32) return (float)gguf_get_val_u32(g, k);
        return 0.0f;
    };
    auto u32 = [&](const char * key, int def) -> int {
        int k = gguf_find_key(g, key);
        return k >= 0 ? u32_safe(k) : def;
    };
    auto f32v = [&](const char * key, float def) -> float {
        int k = gguf_find_key(g, key);
        return k >= 0 ? f32_safe(k) : def;
    };

    // Detect architecture for key prefix fallback.
    // Converters write {arch}.* keys (gemma3.*, qwen3.*) for Ollama compat.
    // Older GGUFs use decoder.* keys.
    std::string arch_pfx = "decoder";
    {
        int64_t ki = gguf_find_key(g, "general.architecture");
        if (ki >= 0) {
            std::string arch = gguf_get_val_str(g, ki);
            if (arch == "gemma3" || arch == "qwen3" || arch == "llama")
                arch_pfx = arch;
        }
    }
    auto a_u32 = [&](const char * suffix, int def) -> int {
        // Try decoder.suffix, then {arch}.suffix
        std::string k1 = "decoder." + std::string(suffix);
        int v = u32(k1.c_str(), -1);
        if (v >= 0) return v;
        std::string k2 = arch_pfx + "." + std::string(suffix);
        return u32(k2.c_str(), def);
    };
    auto a_f32 = [&](const char * suffix, float def) -> float {
        std::string k1 = "decoder." + std::string(suffix);
        int k = gguf_find_key(g, k1.c_str());
        if (k >= 0) return gguf_get_val_f32(g, k);
        std::string k2 = arch_pfx + "." + std::string(suffix);
        k = gguf_find_key(g, k2.c_str());
        return k >= 0 ? gguf_get_val_f32(g, k) : def;
    };

    m.n_vocab = a_u32("vocab_size", 0);  // inferred from tensor if 0
    m.n_embd = a_u32("hidden_size",
               a_u32("embedding_length", 1024));
    m.n_head = a_u32("num_attention_heads",
               a_u32("attention.head_count", 16));
    m.n_kv_head = a_u32("num_key_value_heads",
                  a_u32("attention.head_count_kv", m.n_head));
    m.n_layer = a_u32("num_hidden_layers",
                a_u32("block_count", 28));
    m.n_intermediate = a_u32("intermediate_size",
                      a_u32("feed_forward_length", 3072));
    m.n_max_pos = a_u32("max_position_embeddings",
                  a_u32("context_length", 8192));
    m.rms_norm_eps = a_f32("rms_norm_eps",
                    a_f32("attention.layer_norm_rms_epsilon", 1e-6f));
    m.rope_theta = a_f32("rope_theta",
                   a_f32("rope.freq_base", 10000.0f));
    // Sliding-window alternating rope_theta (Gemma3-style): local layers use a shorter theta.
    // Stored as decoder.rope_theta_local (crisp) or {arch}.rope.freq_base_local (Ollama).
    m.rope_theta_local = a_f32("rope_theta_local",
                         a_f32("rope.freq_base_local", 0.0f));
    m.global_attn_every_n = a_u32("global_attn_every_n_layers",
                            a_u32("attention.sliding_window_layer_period", 0));
    m.is_bidirectional = a_u32("is_bidirectional", 0) != 0;
    // Pooling: decoder.pooling_method or {arch}.pooling_type (Ollama: 3=last)
    m.pooling_method = a_u32("pooling_method", -1);
    if (m.pooling_method < 0) {
        int pt = a_u32("pooling_type", 3);  // Ollama default: 3=last
        // Ollama {1=mean,2=cls,3=last} → internal {1=mean,1=cls(?),2=last}
        m.pooling_method = (pt == 3) ? 2 : (pt == 1) ? 1 : 2;
    }
    m.activation = a_u32("activation", 0);
    m.head_dim = a_u32("head_dim",
                 a_u32("attention.key_length", 0));
    m.attn_scale = a_f32("attn_scale",
                   a_f32("attention.key_length_scale", 0.0f));
    m.embed_scale = a_f32("embed_scale", 1.0f);
    // Detect Gemma3 from architecture.
    // gemma_norm=true  → runtime applies (1+w)*rms_norm(x); use for CrispEmbed-native
    //                    GGUFs where weights are stored raw (HF convention).
    // Ollama-format GGUFs (arch_pfx=="gemma3") pre-bake +1 into every norm weight,
    // so the correct computation is w*rms_norm(x) (gemma_norm stays false).
    // embed_scale and GELU activation still apply for both formats.
    m.gemma_norm = a_u32("gemma_norm", 0) != 0;
    if (arch_pfx == "gemma3") {
        if (m.embed_scale == 1.0f) m.embed_scale = std::sqrt((float)m.n_embd);
        // Gemma3 uses gelu_pytorch_tanh FFN activation (not SiLU)
        if (m.activation == 0) m.activation = 2;
        // gemma_norm intentionally NOT set here: Ollama GGUFs pre-bake the +1 offset.
    }
    bool normalize_embeddings = a_u32("normalize_embeddings", 1) != 0;

    // MRoPE / multimodal metadata (BidirLM-Omni). All optional — absent ⇒
    // text-only path with standard RoPE.
    int sec_idx = gguf_find_key(g, "decoder.mrope_section");
    if (sec_idx >= 0) {
        const int n = (int)gguf_get_arr_n(g, sec_idx);
        const uint32_t * arr = (const uint32_t *)gguf_get_arr_data(g, sec_idx);
        for (int i = 0; i < std::min(n, 3); i++) {
            m.mrope_section[i] = (int)arr[i];
        }
    }
    m.vision_start_token_id = u32("decoder.vision_start_token_id", -1);
    m.vision_end_token_id   = u32("decoder.vision_end_token_id",   -1);
    m.image_token_id        = u32("decoder.image_token_id",        -1);
    m.spatial_merge_size    = u32("decoder.spatial_merge_size",     1);

    // ---- Fallbacks for older GGUFs (pre-b277c3d) that lack the decoder.*
    // multimodal keys but ship a vision tower in `bidirlm.vision.*`. We
    // (a) read spatial_merge from bidirlm.vision.spatial_merge_size,
    // (b) recover image/vision_start/vision_end token ids by string match in
    //     tokenizer.ggml.tokens (the tokens array is always present), and
    // (c) default mrope_section to the BidirLM-Omni reference [24, 20, 20]
    //     when vision metadata exists but mrope_section does not.
    const bool has_vision = gguf_find_key(g, "bidirlm.vision.depth") >= 0;
    if (m.spatial_merge_size <= 1 && has_vision) {
        m.spatial_merge_size = u32("bidirlm.vision.spatial_merge_size", 1);
        fprintf(stderr,
                "decoder_embed: stale GGUF — recovered spatial_merge_size=%d "
                "from bidirlm.vision.spatial_merge_size.\n", m.spatial_merge_size);
    }
    if (has_vision && m.mrope_section[0] == 0
                   && m.mrope_section[1] == 0
                   && m.mrope_section[2] == 0) {
        m.mrope_section[0] = 24;
        m.mrope_section[1] = 20;
        m.mrope_section[2] = 20;
        fprintf(stderr,
                "decoder_embed: stale GGUF — defaulting mrope_section to [24,20,20] "
                "(BidirLM-Omni reference). Re-export with the latest converter for "
                "explicit metadata.\n");
    }
    if (has_vision &&
        (m.image_token_id < 0 || m.vision_start_token_id < 0
         || m.vision_end_token_id < 0)) {
        const int tok_key = gguf_find_key(g, "tokenizer.ggml.tokens");
        if (tok_key >= 0) {
            const int nv = (int)gguf_get_arr_n(g, tok_key);
            auto find_token = [&](const char * s) -> int {
                for (int i = 0; i < nv; i++) {
                    const char * tok = gguf_get_arr_str(g, tok_key, i);
                    if (tok && std::strcmp(tok, s) == 0) return i;
                }
                return -1;
            };
            if (m.image_token_id < 0) {
                int id = find_token("<|image_pad|>");
                if (id >= 0) m.image_token_id = id;
            }
            if (m.vision_start_token_id < 0) {
                int id = find_token("<|vision_start|>");
                if (id >= 0) m.vision_start_token_id = id;
            }
            if (m.vision_end_token_id < 0) {
                int id = find_token("<|vision_end|>");
                if (id >= 0) m.vision_end_token_id = id;
            }
            if (m.image_token_id >= 0) {
                fprintf(stderr,
                    "decoder_embed: stale GGUF — recovered image_token_id=%d, "
                    "vision_start=%d, vision_end=%d by token-string lookup.\n",
                    m.image_token_id, m.vision_start_token_id,
                    m.vision_end_token_id);
            }
        }
    }

    // NOTE: do NOT free g here — it is used later for LoRA metadata reads.
    // g is freed after the LoRA loading block below.

    if (!core_gguf::load_weights(path, backend, "decoder_embed", wl))
        return false;

    auto get = [&](const std::string & n) -> ggml_tensor * {
        auto it = wl.tensors.find(n);
        return it != wl.tensors.end() ? it->second : nullptr;
    };

    m.token_embd = get("token_embd.weight");
    m.output_norm = get("output_norm.weight");
    if (!m.output_norm) m.output_norm = get("final_norm.weight");

    // Infer n_vocab and n_embd from token_embd shape if not in metadata
    if (m.token_embd) {
        int64_t te_embd = m.token_embd->ne[0];
        int64_t te_vocab = m.token_embd->ne[1];
        if (m.n_vocab == 0 || m.n_vocab != (int)te_vocab) {
            fprintf(stderr, "decoder_embed: n_vocab %d → %d (from token_embd)\n",
                    m.n_vocab, (int)te_vocab);
            m.n_vocab = (int)te_vocab;
        }
        if (te_embd != m.n_embd) {
            fprintf(stderr, "decoder_embed: n_embd %d → %d (from token_embd)\n",
                    m.n_embd, (int)te_embd);
            m.n_embd = (int)te_embd;
        }
    }

    // Count actual layers from loaded tensors
    {
        int counted = 0;
        for (const auto& kv : wl.tensors) {
            int layer_id = -1;
            if (sscanf(kv.first.c_str(), "dec.%d.", &layer_id) == 1) {
                if (layer_id + 1 > counted) counted = layer_id + 1;
            }
        }
        if (counted > 0 && counted != m.n_layer) {
            fprintf(stderr, "decoder_embed: n_layer %d → %d (from tensors)\n",
                    m.n_layer, counted);
            m.n_layer = counted;
        }
    }

    m.layers.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; i++) {
        // Try both dec.N.* (CrispEmbed-native) and blk.N.* (Ollama) naming
        auto p = "dec." + std::to_string(i) + ".";
        auto b = "blk." + std::to_string(i) + ".";
        auto & L = m.layers[i];
        auto get2 = [&](const char* s1, const char* s2) -> ggml_tensor* {
            auto t = get(p + s1);
            return t ? t : get(b + s2);
        };
        L.attn_norm_w = get2("attn_norm.weight", "attn_norm.weight");
        L.q_w = get2("attn.q.weight", "attn_q.weight");
        L.q_b = get2("attn.q.bias", "attn_q.bias");
        L.k_w = get2("attn.k.weight", "attn_k.weight");
        L.k_b = get2("attn.k.bias", "attn_k.bias");
        L.v_w = get2("attn.v.weight", "attn_v.weight");
        L.v_b = get2("attn.v.bias", "attn_v.bias");
        L.o_w = get2("attn.o.weight", "attn_output.weight");
        L.o_b = get2("attn.o.bias", "attn_output.bias");
        L.q_norm_w = get2("attn.q_norm.weight", "attn_q_norm.weight");
        L.k_norm_w = get2("attn.k_norm.weight", "attn_k_norm.weight");
        L.ffn_norm_w = get2("ffn_norm.weight", "ffn_norm.weight");
        L.gate_w = get2("ffn.gate.weight", "ffn_gate.weight");
        L.up_w = get2("ffn.up.weight", "ffn_up.weight");
        L.down_w = get2("ffn.down.weight", "ffn_down.weight");
        L.post_attn_norm_w = get2("post_attn_norm.weight", "post_attention_norm.weight");
        L.pre_ffn_norm_w = get2("pre_ffn_norm.weight", "pre_ffn_norm.weight");
        L.post_ffn_norm_w = get2("post_ffn_norm.weight", "post_ffw_norm.weight");
    }

    // Load optional post-pooling Dense projection layers (dense.0.weight, dense.1.weight, ...)
    for (int di = 0; ; di++) {
        std::string key = "dense." + std::to_string(di) + ".weight";
        auto it = wl.tensors.find(key);
        if (it == wl.tensors.end()) break;
        ggml_tensor * t = it->second;
        // Shape: [out_features, in_features] in ggml (ne[0]=in, ne[1]=out)
        int in_dim  = (int)t->ne[0];
        int out_dim = (int)t->ne[1];
        size_t n_elem = (size_t)in_dim * out_dim;
        dec_model::DenseLayer dl;
        dl.in_dim  = in_dim;
        dl.out_dim = out_dim;
        dl.weight.resize(n_elem);
        // Copy from backend tensor to CPU float vector
        ggml_backend_tensor_get(t, dl.weight.data(), 0, n_elem * sizeof(float));
        m.dense_proj.push_back(std::move(dl));
        fprintf(stderr, "decoder_embed: dense.%d.weight [%d→%d]\n", di, in_dim, out_dim);
    }

    // Load LoRA adapter tensors if present
    {
        // Read adapter list from GGUF metadata
        int key_id = gguf_find_key(g, "decoder.lora_adapters");
        if (key_id >= 0 && gguf_get_kv_type(g, key_id) == GGUF_TYPE_ARRAY) {
            int n_adapters = (int)gguf_get_arr_n(g, key_id);
            int lora_rank = a_u32("lora_rank", 0);
            float lora_alpha = a_f32("lora_alpha", 0.0f);
            std::string lora_default;
            {
                int dk = gguf_find_key(g, "decoder.lora_default");
                if (dk >= 0) lora_default = gguf_get_val_str(g, dk);
            }

            // Collect adapter names
            std::vector<std::string> adapter_names;
            for (int ai = 0; ai < n_adapters; ai++) {
                adapter_names.push_back(gguf_get_arr_str(g, key_id, ai));
            }

            // Build projection suffix list that matches the GGUF tensor naming
            // Detect naming from first adapter's tensors
            std::string LP_detected = "blk";  // default
            for (auto & kv : wl.tensors) {
                if (kv.first.substr(0, 5) == "lora.") {
                    // e.g. "lora.retrieval.blk.0.attn_q.A.weight" → LP = "blk"
                    // or   "lora.retrieval.dec.0.attn.q.A.weight" → LP = "dec"
                    size_t dot2 = kv.first.find('.', 5);  // after "lora.ADAPTER."
                    if (dot2 != std::string::npos) {
                        size_t dot3 = kv.first.find('.', dot2 + 1);
                        if (dot3 != std::string::npos)
                            LP_detected = kv.first.substr(dot2 + 1, dot3 - dot2 - 1);
                    }
                    break;
                }
            }

            // Projection suffixes to scan (both Ollama and CrispEmbed naming)
            struct ProjMap { const char * suffix; lora_pair lora_layer::* field; };
            std::vector<ProjMap> proj_maps;
            if (LP_detected == "blk") {
                proj_maps = {
                    {"attn_q",      &lora_layer::q},
                    {"attn_k",      &lora_layer::k},
                    {"attn_v",      &lora_layer::v},
                    {"attn_output", &lora_layer::o},
                    {"ffn_gate",    &lora_layer::gate},
                    {"ffn_up",      &lora_layer::up},
                    {"ffn_down",    &lora_layer::down},
                };
            } else {
                proj_maps = {
                    {"attn.q",   &lora_layer::q},
                    {"attn.k",   &lora_layer::k},
                    {"attn.v",   &lora_layer::v},
                    {"attn.o",   &lora_layer::o},
                    {"ffn.gate", &lora_layer::gate},
                    {"ffn.up",   &lora_layer::up},
                    {"ffn.down", &lora_layer::down},
                };
            }

            for (const auto & aname : adapter_names) {
                lora_adapter adapter;
                adapter.name  = aname;
                adapter.rank  = lora_rank;
                adapter.alpha = lora_alpha;
                adapter.layers.resize(m.n_layer);

                int loaded = 0;
                for (int li = 0; li < m.n_layer; li++) {
                    for (auto & pm : proj_maps) {
                        std::string a_key = "lora." + aname + "." + LP_detected + "." +
                                            std::to_string(li) + "." + pm.suffix + ".A.weight";
                        std::string b_key = "lora." + aname + "." + LP_detected + "." +
                                            std::to_string(li) + "." + pm.suffix + ".B.weight";
                        auto it_a = wl.tensors.find(a_key);
                        auto it_b = wl.tensors.find(b_key);
                        if (it_a == wl.tensors.end() || it_b == wl.tensors.end()) continue;

                        ggml_tensor * ta = it_a->second;
                        ggml_tensor * tb = it_b->second;
                        lora_pair & lp = adapter.layers[li].*pm.field;
                        lp.rank    = (int)ta->ne[1];  // A is [rank, in_dim] → ne[0]=in_dim, ne[1]=rank
                        lp.in_dim  = (int)ta->ne[0];
                        lp.out_dim = (int)tb->ne[1];  // B is [out_dim, rank] → ne[0]=rank, ne[1]=out_dim

                        // Read tensor data → F32
                        size_t na = (size_t)lp.rank * lp.in_dim;
                        size_t nb = (size_t)lp.out_dim * lp.rank;
                        lp.A.resize(na);
                        lp.B.resize(nb);

                        // Tensors may be F16 — read raw then convert
                        if (ta->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> buf(na);
                            ggml_backend_tensor_get(ta, buf.data(), 0, na * sizeof(ggml_fp16_t));
                            for (size_t k = 0; k < na; k++) lp.A[k] = ggml_fp16_to_fp32(buf[k]);
                        } else {
                            ggml_backend_tensor_get(ta, lp.A.data(), 0, na * sizeof(float));
                        }
                        if (tb->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> buf(nb);
                            ggml_backend_tensor_get(tb, buf.data(), 0, nb * sizeof(ggml_fp16_t));
                            for (size_t k = 0; k < nb; k++) lp.B[k] = ggml_fp16_to_fp32(buf[k]);
                        } else {
                            ggml_backend_tensor_get(tb, lp.B.data(), 0, nb * sizeof(float));
                        }
                        loaded++;
                    }
                }
                m.lora_adapters.push_back(std::move(adapter));
                fprintf(stderr, "decoder_embed: LoRA adapter '%s': %d projections, rank=%d, alpha=%.1f\n",
                        aname.c_str(), loaded, lora_rank, lora_alpha);
            }

            // Auto-activate default adapter
            if (!lora_default.empty() && !m.lora_adapters.empty()) {
                fprintf(stderr, "decoder_embed: activating default LoRA '%s'\n", lora_default.c_str());
                decoder_set_lora(m, backend, lora_default);
            }
        }
    }

    gguf_free(g);  // Free metadata context (was kept open for LoRA metadata reads)

    const char * pool_str = (m.pooling_method == 1) ? "mean" : "last-token";
    fprintf(stderr, "decoder_embed: loaded %d layers, %d dims, %d vocab, %d heads (%d kv), rope_theta=%.0f%s, pool=%s%s%s\n",
            m.n_layer, m.n_embd, m.n_vocab, m.n_head, m.n_kv_head, m.rope_theta,
            (m.rope_theta_local > 0.0f && m.global_attn_every_n > 0)
                ? (" (local=" + std::to_string((int)m.rope_theta_local)
                   + " every " + std::to_string(m.global_attn_every_n) + ")").c_str()
                : "",
            pool_str, m.is_bidirectional ? ", bidirectional" : "",
            m.lora_adapters.empty() ? "" : (", " + std::to_string(m.lora_adapters.size()) + " LoRA adapters").c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Full-graph decoder with scheduler support (GPU + CPU)
// ---------------------------------------------------------------------------

// Build 3D MRoPE position ids in HF BidirLMOmniModel.get_rope_index style.
// Output `pos_thw` is laid out as 4 contiguous int32 arrays of length T:
//   [t..., h..., w..., e...]   with e == t (the "extra" channel ggml IMROPE
// uses at sectors that fall outside the t/h/w slices when `mrope_section`
// does not cover the full head_dim/2 — for BidirLM sections=[24,20,20] those
// are sectors 61, 62 of the 64-pair head; pinning e=t makes ggml IMROPE
// numerically agree with HF's apply_interleaved_mrope at those sectors).
static void build_mrope_positions_3d(
        const dec_model & m,
        const embed_tokens & tokens,
        const dec_image_input * img,
        std::vector<int32_t> & pos_thw) {
    const int T = (int)tokens.ids.size();
    pos_thw.assign((size_t)4 * T, 0);
    int32_t * pos_t = pos_thw.data();
    int32_t * pos_h = pos_thw.data() + (size_t)T;
    int32_t * pos_w = pos_thw.data() + (size_t)2 * T;
    int32_t * pos_e = pos_thw.data() + (size_t)3 * T;

    const int spatial_merge = std::max(1, m.spatial_merge_size);
    const int img_tok = m.image_token_id;
    const bool has_image = (img && img->image_embeds && img->n_image_tokens > 0
                             && img->grid_thw && img->n_images > 0
                             && img_tok >= 0);

    int last_max = -1;
    int img_idx = 0;
    int st = 0;
    while (st < T) {
        // Find next image_token_id at or after st (only if we still have
        // images to consume). Image tokens are assumed contiguous.
        int ed = T;
        if (has_image && img_idx < img->n_images) {
            int p = st;
            while (p < T && tokens.ids[p] != img_tok) p++;
            if (p < T) ed = p;
        }

        // Text segment [st, ed).
        const int text_len = ed - st;
        const int st_idx = last_max + 1;
        for (int i = 0; i < text_len; i++) {
            pos_t[st + i] = st_idx + i;
            pos_h[st + i] = st_idx + i;
            pos_w[st + i] = st_idx + i;
        }
        if (text_len > 0) last_max = st_idx + text_len - 1;

        if (!has_image || img_idx >= img->n_images || ed >= T) {
            st = ed;
            // Tail text after final image (or no images): we already filled
            // [st, ed). If ed < T but we have no more images, treat the
            // remainder as another text segment.
            if (ed < T) {
                const int tail = T - ed;
                const int tail_st = last_max + 1;
                for (int i = 0; i < tail; i++) {
                    pos_t[ed + i] = tail_st + i;
                    pos_h[ed + i] = tail_st + i;
                    pos_w[ed + i] = tail_st + i;
                }
                if (tail > 0) last_max = tail_st + tail - 1;
            }
            break;
        }

        // Image block at [ed, ed + n_img_tok).
        const int t_grid = img->grid_thw[img_idx * 3 + 0];
        const int h_grid = img->grid_thw[img_idx * 3 + 1] / spatial_merge;
        const int w_grid = img->grid_thw[img_idx * 3 + 2] / spatial_merge;
        const int n_img_tok = t_grid * h_grid * w_grid;
        const int img_st = last_max + 1;
        int idx_in = 0;
        for (int ti = 0; ti < t_grid; ti++) {
            for (int hi = 0; hi < h_grid; hi++) {
                for (int wi = 0; wi < w_grid; wi++) {
                    const int p = ed + idx_in;
                    if (p < T) {
                        pos_t[p] = img_st + ti;
                        pos_h[p] = img_st + hi;
                        pos_w[p] = img_st + wi;
                    }
                    idx_in++;
                }
            }
        }
        last_max = img_st + std::max({t_grid, h_grid, w_grid}) - 1;
        img_idx++;
        st = ed + n_img_tok;
    }

    // e channel mirrors t.
    for (int t = 0; t < T; t++) pos_e[t] = pos_t[t];
}

std::vector<float> decoder_encode_tokens(
    const dec_model & m, ggml_backend_t backend,
    const embed_tokens & tokens, int n_threads,
    ggml_backend_sched_t sched,
    std::vector<uint8_t> * compute_meta,
    const dec_image_input * img) {

    const int T = (int)tokens.ids.size();
    const int H = m.n_embd;
    const int n_heads = m.n_head;
    const int n_kv_heads = m.n_kv_head;
    int q_dim = m.layers[0].q_w ? (int)m.layers[0].q_w->ne[1] : H;
    const int head_dim = (m.head_dim > 0) ? m.head_dim : (q_dim / n_heads);
    q_dim = n_heads * head_dim;
    const float eps = m.rms_norm_eps;

    const bool has_image = (img && img->image_embeds && img->n_image_tokens > 0
                             && m.image_token_id >= 0);
    // Switch to ggml_rope_multi (3D interleaved MRoPE) only when image input
    // is present. For text-only inputs the t/h/w channels are all equal, so
    // standard NEOX rope_ext is mathematically equivalent and keeps the
    // existing text-only parity tests bit-identical.
    const bool use_mrope = has_image
                           && (m.mrope_section[0] + m.mrope_section[1]
                               + m.mrope_section[2]) > 0;
    // Diagnostic: skip DeepStack injection while keeping the image-embed
    // splice + 3D MRoPE. If parity improves with DEEPSTACK_OFF=1, the bug is
    // in the deepstack hook; otherwise look at splice / MRoPE.
    const bool skip_deepstack = std::getenv("CRISPEMBED_SKIP_DEEPSTACK") != nullptr;
    const int n_ds = (has_image && !skip_deepstack)
                     ? std::min(img->n_deepstack, m.n_layer) : 0;

    // Graph context: no_alloc=true when using scheduler, false otherwise
    bool use_sched = (sched != nullptr && compute_meta != nullptr);
    int graph_size = std::max(4096, m.n_layer * 50 + 256);

    size_t mem;
    std::vector<uint8_t> local_buf;
    uint8_t * buf_ptr;

    if (use_sched) {
        mem = compute_meta->size();
        buf_ptr = compute_meta->data();
    } else {
        size_t per_layer = (size_t)H * T * 4 * 30
                         + (size_t)T * T * n_heads * 4 * 2
                         + (size_t)m.n_intermediate * T * 4 * 3;
        mem = per_layer * m.n_layer
            + (size_t)H * T * 4 * 10
            + ggml_tensor_overhead() * (size_t)(m.n_layer * 50 + 200)
            + ggml_graph_overhead_custom(graph_size, false)
            + 64 * 1024 * 1024;
        local_buf.resize(mem);
        buf_ptr = local_buf.data();
    }

    ggml_init_params ip = { mem, buf_ptr, use_sched };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // --- Token embedding ---
    ggml_tensor * ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    ggml_set_name(ids_t, "tok_ids");
    ggml_set_input(ids_t);

    ggml_tensor * cur = ggml_get_rows(gctx, m.token_embd, ids_t);

    if (m.embed_scale != 1.0f) {
        cur = ggml_scale(gctx, cur, m.embed_scale);
    }

    // Image-embed splice: replace token-embed rows at every image_token_id
    // position with the corresponding `image_embeds` row. We do this with a
    // host-prepared (1, T) keep-mask (0 at image positions, 1 elsewhere) and
    // an (H, T) patch tensor (image_embeds at image positions, 0 elsewhere).
    // Equivalent to HF's `inputs_embeds.masked_scatter(image_mask, image_embeds)`.
    ggml_tensor * input_keep_mask = nullptr;
    ggml_tensor * input_patch     = nullptr;
    if (has_image) {
        input_keep_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, T);
        ggml_set_name(input_keep_mask, "in_keep_mask");
        ggml_set_input(input_keep_mask);

        input_patch = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
        ggml_set_name(input_patch, "in_patch");
        ggml_set_input(input_patch);

        cur = ggml_mul(gctx, cur, input_keep_mask);
        cur = ggml_add(gctx, cur, input_patch);
    }

    // DeepStack patches: per-layer (H, T) tensors holding `deepstack[k]`
    // rows at image positions and 0 elsewhere. Added after the k-th layer's
    // residual+ffn output. Mirrors HF BidirLMOmniTextModel deepstack hook.
    std::vector<ggml_tensor *> ds_patches(n_ds, nullptr);
    for (int k = 0; k < n_ds; k++) {
        char nm[24];
        std::snprintf(nm, sizeof(nm), "ds_patch_%d", k);
        ds_patches[k] = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
        ggml_set_name(ds_patches[k], nm);
        ggml_set_input(ds_patches[k]);
    }

    // Gemma3 ones tensor for (1 + weight) RMSNorm
    ggml_tensor * ones_h = nullptr;
    ggml_tensor * ones_hd = nullptr;
    if (m.gemma_norm) {
        ones_h = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, H);
        ggml_set_name(ones_h, "ones_h");
        ggml_set_input(ones_h);  // will set to 1.0f
        if (m.layers[0].q_norm_w) {
            ones_hd = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, head_dim);
            ggml_set_name(ones_hd, "ones_hd");
            ggml_set_input(ones_hd);
        }
    }

    auto rms_norm = [&](ggml_tensor * x, ggml_tensor * w) -> ggml_tensor * {
        x = ggml_rms_norm(gctx, x, eps);
        if (m.gemma_norm) {
            ggml_tensor * ones = (w->ne[0] == H) ? ones_h : ones_hd;
            return ggml_mul(gctx, x, ggml_add(gctx, w, ones));
        }
        return ggml_mul(gctx, x, w);
    };

    // --- Position IDs for RoPE ---
    // Standard 1D RoPE: (T,). MRoPE / IMROPE: (4*T,) laid out [t,h,w,e].
    const int pos_size = use_mrope ? 4 * T : T;
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, pos_size);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    // --- Transformer layers ---
    for (int il = 0; il < m.n_layer; il++) {
        const auto & L = m.layers[il];
        ggml_tensor * residual = cur;

        if (L.attn_norm_w) cur = rms_norm(cur, L.attn_norm_w);

        ggml_tensor * Q = ggml_mul_mat(gctx, L.q_w, cur);
        if (L.q_b) Q = ggml_add(gctx, Q, L.q_b);
        ggml_tensor * K = ggml_mul_mat(gctx, L.k_w, cur);
        if (L.k_b) K = ggml_add(gctx, K, L.k_b);
        ggml_tensor * V = ggml_mul_mat(gctx, L.v_w, cur);
        if (L.v_b) V = ggml_add(gctx, V, L.v_b);

        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads, T);
        K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, T);
        V = ggml_reshape_3d(gctx, V, head_dim, n_kv_heads, T);

        if (L.q_norm_w) Q = rms_norm(Q, L.q_norm_w);
        if (L.k_norm_w) K = rms_norm(K, L.k_norm_w);

        // Per-layer rope_theta: Gemma3 sliding-window layers use a shorter theta
        float layer_rope_theta = m.rope_theta;
        if (m.rope_theta_local > 0.0f && m.global_attn_every_n > 0) {
            bool is_global = ((il + 1) % m.global_attn_every_n == 0);
            layer_rope_theta = is_global ? m.rope_theta : m.rope_theta_local;
        }

        if (use_mrope) {
            int sections[GGML_MROPE_SECTIONS] = {
                m.mrope_section[0], m.mrope_section[1], m.mrope_section[2], 0,
            };
            const int rope_mode = GGML_ROPE_TYPE_IMROPE;
            Q = ggml_rope_multi(gctx, Q, pos, nullptr,
                                head_dim, sections, rope_mode, 0,
                                layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_multi(gctx, K, pos, nullptr,
                                head_dim, sections, rope_mode, 0,
                                layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        } else {
            int rope_mode = 2;
            Q = ggml_rope_ext(gctx, Q, pos, nullptr,
                               head_dim, rope_mode, 0,
                               layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(gctx, K, pos, nullptr,
                               head_dim, rope_mode, 0,
                               layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // Attention: permute → scores → mask → softmax → value
        Q = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

        float scale = (m.attn_scale > 0.0f)
                        ? (1.0f / sqrtf(m.attn_scale))
                        : (1.0f / sqrtf((float)head_dim));
        ggml_tensor * scores = ggml_mul_mat(gctx, K, Q);
        scores = ggml_scale(gctx, scores, scale);
        if (!m.is_bidirectional) {
            scores = ggml_diag_mask_inf(gctx, scores, 0);
        }
        scores = ggml_soft_max(gctx, scores);

        ggml_tensor * V_perm = ggml_cont(gctx, ggml_permute(gctx, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(gctx, V_perm, scores);
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(gctx, attn, q_dim, T);

        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);

        // Gemma3: post-attention norm applied to attention output before residual add
        if (L.post_attn_norm_w) attn = rms_norm(attn, L.post_attn_norm_w);

        if (L.pre_ffn_norm_w) {
            if (L.ffn_norm_w) attn = rms_norm(attn, L.ffn_norm_w);
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            cur = rms_norm(cur, L.pre_ffn_norm_w);
        } else {
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            if (L.ffn_norm_w) cur = rms_norm(cur, L.ffn_norm_w);
        }

        if (L.gate_w && L.up_w && L.down_w) {
            ggml_tensor * gate = ggml_mul_mat(gctx, L.gate_w, cur);
            if (m.activation == 2 || m.activation == 1) {
                gate = ggml_gelu(gctx, gate);
            } else {
                gate = ggml_silu(gctx, gate);
            }
            ggml_tensor * up = ggml_mul_mat(gctx, L.up_w, cur);
            ggml_tensor * ffn = ggml_mul(gctx, gate, up);
            ffn = ggml_mul_mat(gctx, L.down_w, ffn);
            if (L.post_ffn_norm_w) ffn = rms_norm(ffn, L.post_ffn_norm_w);
            cur = ggml_add(gctx, residual, ffn);
        } else {
            cur = residual;
        }

        if (il < n_ds && ds_patches[il]) {
            cur = ggml_add(gctx, cur, ds_patches[il]);
        }
    }

    if (m.output_norm) cur = rms_norm(cur, m.output_norm);

    ggml_set_name(cur, "decoder_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // --- Build host-side input buffers ---
    std::vector<int32_t> pos_data;
    if (use_mrope) {
        build_mrope_positions_3d(m, tokens, has_image ? img : nullptr, pos_data);
    } else {
        pos_data.resize(T);
        for (int t = 0; t < T; t++) pos_data[t] = t;
    }

    std::vector<float> in_keep_data;
    std::vector<float> in_patch_data;
    std::vector<std::vector<float>> ds_patch_data;
    if (has_image) {
        in_keep_data.assign((size_t)T, 1.0f);
        in_patch_data.assign((size_t)H * T, 0.0f);
        ds_patch_data.resize(n_ds);
        for (int k = 0; k < n_ds; k++) {
            ds_patch_data[k].assign((size_t)H * T, 0.0f);
        }

        const int img_tok = m.image_token_id;
        const int n_image_tokens = img->n_image_tokens;
        const float * src_img = img->image_embeds;             // (n_image, H)
        const float * src_ds = img->deepstack;                 // (n_ds, n_image, H)
        int img_idx = 0;
        for (int t = 0; t < T && img_idx < n_image_tokens; t++) {
            if (tokens.ids[t] != img_tok) continue;
            in_keep_data[t] = 0.0f;
            std::memcpy(in_patch_data.data() + (size_t)t * H,
                        src_img + (size_t)img_idx * H,
                        (size_t)H * sizeof(float));
            for (int k = 0; k < n_ds; k++) {
                if (!src_ds) break;
                const float * srow = src_ds
                    + (size_t)k * n_image_tokens * H
                    + (size_t)img_idx * H;
                std::memcpy(ds_patch_data[k].data() + (size_t)t * H,
                            srow, (size_t)H * sizeof(float));
            }
            img_idx++;
        }
        if (img_idx != n_image_tokens) {
            fprintf(stderr,
                "decoder: image token count mismatch — input has %d "
                "image_token_id placeholders, vision tower produced %d "
                "merged tokens (image_token_id=%d). Encoding will likely "
                "be wrong.\n",
                img_idx, n_image_tokens, img_tok);
        }
    }

    // --- Set inputs and compute ---
    if (use_sched) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "decoder: failed to allocate graph\n");
            ggml_free(gctx);
            return {};
        }

        // Set input data
        std::vector<int32_t> tok_data(tokens.ids.begin(), tokens.ids.end());
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_ids"),
                                tok_data.data(), 0, T * sizeof(int32_t));

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                                pos_data.data(),
                                0,
                                pos_data.size() * sizeof(int32_t));

        if (m.gemma_norm) {
            std::vector<float> ones(H, 1.0f);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_h"),
                                    ones.data(), 0, H * sizeof(float));
            if (ones_hd) {
                std::vector<float> ones_hd_data(head_dim, 1.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_hd"),
                                        ones_hd_data.data(), 0, head_dim * sizeof(float));
            }
        }

        if (has_image) {
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "in_keep_mask"),
                                    in_keep_data.data(), 0,
                                    (size_t)T * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "in_patch"),
                                    in_patch_data.data(), 0,
                                    (size_t)H * T * sizeof(float));
            for (int k = 0; k < n_ds; k++) {
                char nm[24];
                std::snprintf(nm, sizeof(nm), "ds_patch_%d", k);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm),
                                        ds_patch_data[k].data(), 0,
                                        (size_t)H * T * sizeof(float));
            }
        }

        // Compute via scheduler
        ggml_backend_sched_graph_compute(sched, gf);
    } else {
        // CPU fallback (set data directly)
        int32_t * id = (int32_t *)ids_t->data;
        for (int t = 0; t < T; t++) id[t] = tokens.ids[t];

        int32_t * pd = (int32_t *)pos->data;
        std::memcpy(pd, pos_data.data(), pos_data.size() * sizeof(int32_t));

        if (ones_h) {
            float * d = (float *)ones_h->data;
            for (int i = 0; i < H; i++) d[i] = 1.0f;
        }
        if (ones_hd) {
            float * d = (float *)ones_hd->data;
            for (int i = 0; i < head_dim; i++) d[i] = 1.0f;
        }

        if (has_image) {
            std::memcpy(input_keep_mask->data, in_keep_data.data(),
                        (size_t)T * sizeof(float));
            std::memcpy(input_patch->data, in_patch_data.data(),
                        (size_t)H * T * sizeof(float));
            for (int k = 0; k < n_ds; k++) {
                std::memcpy(ds_patches[k]->data, ds_patch_data[k].data(),
                            (size_t)H * T * sizeof(float));
            }
        }

        struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, NULL);
        std::vector<uint8_t> work;
        if (cplan.work_size > 0) {
            work.resize(cplan.work_size);
            cplan.work_data = work.data();
        }
        ggml_graph_compute(gf, &cplan);
    }

    // --- Read output ---
    ggml_tensor * out = ggml_graph_get_tensor(gf, "decoder_out");
    std::vector<float> hidden(H * T);
    if (use_sched) {
        ggml_backend_tensor_get(out, hidden.data(), 0, H * T * sizeof(float));
    } else {
        memcpy(hidden.data(), out->data, H * T * sizeof(float));
    }

    ggml_free(gctx);

    std::vector<float> pooled(H, 0.0f);

    if (m.pooling_method == 1) {
        // Mean pooling over non-padding tokens (BidirLM-style)
        int n_valid = 0;
        for (int t = 0; t < T; t++) {
            if (!tokens.attn_mask[t]) continue;
            const float * row = hidden.data() + (size_t)t * H;
            for (int i = 0; i < H; i++) pooled[i] += row[i];
            n_valid++;
        }
        if (n_valid > 0) {
            const float inv = 1.0f / (float)n_valid;
            for (int i = 0; i < H; i++) pooled[i] *= inv;
        }
    } else {
        // Last-token pooling (Qwen3/Gemma3)
        int last_t = 0;
        for (int t = T - 1; t >= 0; t--) {
            if (tokens.attn_mask[t]) { last_t = t; break; }
        }
        memcpy(pooled.data(), hidden.data() + (size_t)last_t * H, H * sizeof(float));
    }

    // Post-pooling Dense projection layers (SentenceTransformer-style, e.g. EmbeddingGemma)
    for (const auto & dl : m.dense_proj) {
        std::vector<float> out_vec(dl.out_dim, 0.0f);
        // weight is [out_dim, in_dim] row-major; compute: out[o] = sum_j(weight[o*in+j]*pooled[j])
        const float * w = dl.weight.data();
        for (int o = 0; o < dl.out_dim; o++) {
            float acc = 0.0f;
            for (int j = 0; j < dl.in_dim; j++) acc += w[o * dl.in_dim + j] * pooled[j];
            out_vec[o] = acc;
        }
        pooled = std::move(out_vec);
    }

    // L2 normalize
    float norm = 0;
    for (int i = 0; i < (int)pooled.size(); i++) norm += pooled[i] * pooled[i];
    norm = sqrtf(std::max(norm, 1e-12f));
    for (int i = 0; i < (int)pooled.size(); i++) pooled[i] /= norm;

    return pooled;
}

// ---------------------------------------------------------------------------
// LoRA adapter hot-swap
// ---------------------------------------------------------------------------

// Helper: dequantize a ggml tensor to F32 on CPU
static std::vector<float> dequant_tensor(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        // Quantized: read raw bytes then dequantize
        size_t raw_size = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_size);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_size);
        auto traits = ggml_get_type_traits(t->type);
        if (traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            // Fallback: zero-fill (should not happen for supported types)
            std::fill(out.begin(), out.end(), 0.0f);
        }
    }
    return out;
}

// Helper: write F32 data to a ggml tensor, requantizing if needed
static void write_tensor_f32(ggml_tensor * t, const float * data) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp32_to_fp16(data[i]);
        ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        // Quantized: requantize from F32
        size_t raw_size = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_size);
        auto traits = ggml_get_type_traits(t->type);
        if (traits->from_float_ref) {
            traits->from_float_ref(data, raw.data(), n);
        }
        ggml_backend_tensor_set(t, raw.data(), 0, raw_size);
    }
}

// Helper: CPU matmul C = B @ A where B is [M, K] and A is [K, N], result C is [M, N]
// (row-major convention: C[i,j] = sum_k B[i,k] * A[k,j])
static void matmul_f32(const float * B, const float * A,
                        float * C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += B[i * K + k] * A[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

bool decoder_set_lora(dec_model & m, ggml_backend_t backend,
                      const std::string & adapter_name) {
    (void)backend;

    // Empty name = unmerge
    if (adapter_name.empty()) {
        if (m.active_lora.empty()) return true;  // already base
        // Restore all base weights
        struct TensorRef { ggml_tensor ** ptrs[7]; const char * names[7]; };
        for (int li = 0; li < m.n_layer; li++) {
            dec_layer & L = m.layers[li];
            ggml_tensor * tensors[] = { L.q_w, L.k_w, L.v_w, L.o_w, L.gate_w, L.up_w, L.down_w };
            const char * suffixes[] = { "q", "k", "v", "o", "gate", "up", "down" };
            for (int pi = 0; pi < 7; pi++) {
                if (!tensors[pi]) continue;
                std::string key = "base." + std::to_string(li) + "." + suffixes[pi];
                auto it = m.base_weights_f32.find(key);
                if (it != m.base_weights_f32.end()) {
                    write_tensor_f32(tensors[pi], it->second.data());
                }
            }
        }
        fprintf(stderr, "decoder_embed: LoRA unmerged (restored base weights)\n");
        m.active_lora.clear();
        return true;
    }

    // Same adapter already active
    if (adapter_name == m.active_lora) return true;

    // Find adapter
    const lora_adapter * adapter = nullptr;
    for (const auto & a : m.lora_adapters) {
        if (a.name == adapter_name) { adapter = &a; break; }
    }
    if (!adapter) {
        fprintf(stderr, "decoder_embed: LoRA adapter '%s' not found\n", adapter_name.c_str());
        return false;
    }

    // If another adapter is active, unmerge first
    if (!m.active_lora.empty()) {
        decoder_set_lora(m, backend, "");
    }

    // Lazy base weight snapshot (first time only)
    if (m.base_weights_f32.empty()) {
        for (int li = 0; li < m.n_layer; li++) {
            dec_layer & L = m.layers[li];
            ggml_tensor * tensors[] = { L.q_w, L.k_w, L.v_w, L.o_w, L.gate_w, L.up_w, L.down_w };
            const char * suffixes[] = { "q", "k", "v", "o", "gate", "up", "down" };
            const lora_layer & ll = adapter->layers[li];
            const lora_pair * pairs[] = { &ll.q, &ll.k, &ll.v, &ll.o, &ll.gate, &ll.up, &ll.down };
            for (int pi = 0; pi < 7; pi++) {
                if (!tensors[pi] || pairs[pi]->empty()) continue;
                std::string key = "base." + std::to_string(li) + "." + suffixes[pi];
                m.base_weights_f32[key] = dequant_tensor(tensors[pi]);
            }
        }
        fprintf(stderr, "decoder_embed: base weight snapshot taken (%zu tensors)\n",
                m.base_weights_f32.size());
    }

    // Merge: W' = W_base + (alpha/rank) * B @ A
    float scale = adapter->alpha / (float)adapter->rank;
    int merged = 0;
    for (int li = 0; li < m.n_layer; li++) {
        dec_layer & L = m.layers[li];
        ggml_tensor * tensors[] = { L.q_w, L.k_w, L.v_w, L.o_w, L.gate_w, L.up_w, L.down_w };
        const char * suffixes[] = { "q", "k", "v", "o", "gate", "up", "down" };
        const lora_layer & ll = adapter->layers[li];
        const lora_pair * pairs[] = { &ll.q, &ll.k, &ll.v, &ll.o, &ll.gate, &ll.up, &ll.down };

        for (int pi = 0; pi < 7; pi++) {
            if (!tensors[pi] || pairs[pi]->empty()) continue;
            const lora_pair & lp = *pairs[pi];
            std::string key = "base." + std::to_string(li) + "." + suffixes[pi];

            auto it = m.base_weights_f32.find(key);
            if (it == m.base_weights_f32.end()) continue;

            // Start from base weights
            std::vector<float> merged_w = it->second;

            // Compute delta = B @ A  (B is [out, rank], A is [rank, in])
            // Result is [out, in] — same shape as the weight
            std::vector<float> delta(lp.out_dim * lp.in_dim);
            matmul_f32(lp.B.data(), lp.A.data(), delta.data(),
                       lp.out_dim, lp.rank, lp.in_dim);

            // W' = W_base + scale * delta
            for (size_t k = 0; k < merged_w.size(); k++) {
                merged_w[k] += scale * delta[k];
            }

            // Write back (requantize if needed)
            write_tensor_f32(tensors[pi], merged_w.data());
            merged++;
        }
    }

    m.active_lora = adapter_name;
    fprintf(stderr, "decoder_embed: LoRA '%s' merged (%d projections, scale=%.3f)\n",
            adapter_name.c_str(), merged, scale);
    return true;
}

// ---------------------------------------------------------------------------
// Batched decoder encoding
// ---------------------------------------------------------------------------

std::vector<std::vector<float>> decoder_encode_tokens_batch(
    const dec_model & m, ggml_backend_t backend,
    const std::vector<embed_tokens> & batch, int n_threads,
    ggml_backend_sched_t sched,
    std::vector<uint8_t> * compute_meta) {

    const int B = (int)batch.size();

    // For B<=1, use the single-text path (avoid mask overhead)
    if (B <= 1) {
        std::vector<std::vector<float>> results;
        for (const auto & tokens : batch) {
            results.push_back(decoder_encode_tokens(m, backend, tokens, n_threads,
                                                     sched, compute_meta));
        }
        return results;
    }

    // Compute T_max and per-text lengths
    std::vector<int> lens(B);
    int T_max = 0;
    for (int b = 0; b < B; b++) {
        lens[b] = (int)batch[b].ids.size();
        T_max = std::max(T_max, lens[b]);
    }
    const int T_total = T_max * B;

    const int H = m.n_embd;
    const int n_heads = m.n_head;
    const int n_kv_heads = m.n_kv_head;
    int q_dim = m.layers[0].q_w ? (int)m.layers[0].q_w->ne[1] : H;
    const int head_dim = (m.head_dim > 0) ? m.head_dim : (q_dim / n_heads);
    q_dim = n_heads * head_dim;
    const float eps = m.rms_norm_eps;

    // Build padded token ids and position ids
    std::vector<int32_t> tok_data(T_total, 0);
    std::vector<int32_t> pos_data(T_total, 0);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < lens[b]; t++) {
            tok_data[b * T_max + t] = batch[b].ids[t];
            pos_data[b * T_max + t] = t;
        }
    }

    // Build block-diagonal causal mask as F32 [T_total, T_total]
    // mask[q_pos, k_pos] = 0.0 if same text AND q >= k AND k < len
    //                    = -INFINITY otherwise
    // Padding positions get self-attention (mask[pad,pad]=0) to prevent
    // softmax(all -inf) = NaN. The padding output isn't pooled.
    std::vector<float> mask_data((size_t)T_total * T_total, -INFINITY);
    for (int b = 0; b < B; b++) {
        for (int q = 0; q < lens[b]; q++) {
            if (m.is_bidirectional) {
                // Bidirectional: all valid positions within same text can attend
                for (int k = 0; k < lens[b]; k++) {
                    mask_data[(size_t)(b * T_max + q) * T_total + (b * T_max + k)] = 0.0f;
                }
            } else {
                // Causal: q can attend to k only if k <= q
                for (int k = 0; k <= q; k++) {
                    mask_data[(size_t)(b * T_max + q) * T_total + (b * T_max + k)] = 0.0f;
                }
            }
        }
        // Padding positions: self-attend to prevent NaN in softmax
        for (int p = lens[b]; p < T_max; p++) {
            mask_data[(size_t)(b * T_max + p) * T_total + (b * T_max + p)] = 0.0f;
        }
    }

    // --- Build graph ---
    bool use_sched = (sched != nullptr && compute_meta != nullptr);
    int graph_size = std::max(4096, m.n_layer * 50 + 256);

    size_t mem;
    std::vector<uint8_t> local_buf;
    uint8_t * buf_ptr;

    if (use_sched) {
        mem = compute_meta->size();
        buf_ptr = compute_meta->data();
    } else {
        size_t per_layer = (size_t)H * T_total * 4 * 30
                         + (size_t)T_total * T_total * n_heads * 4 * 2
                         + (size_t)m.n_intermediate * T_total * 4 * 3;
        mem = per_layer * m.n_layer
            + (size_t)H * T_total * 4 * 10
            + ggml_tensor_overhead() * (size_t)(m.n_layer * 50 + 200)
            + ggml_graph_overhead_custom(graph_size, false)
            + 64 * 1024 * 1024;
        local_buf.resize(mem);
        buf_ptr = local_buf.data();
    }

    ggml_init_params ip = { mem, buf_ptr, use_sched };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, graph_size, false);

    // Token embedding
    ggml_tensor * ids_t = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_total);
    ggml_set_name(ids_t, "tok_ids");
    ggml_set_input(ids_t);

    ggml_tensor * cur = ggml_get_rows(gctx, m.token_embd, ids_t);

    if (m.embed_scale != 1.0f) {
        cur = ggml_scale(gctx, cur, m.embed_scale);
    }

    // Gemma3 ones tensors
    ggml_tensor * ones_h = nullptr;
    ggml_tensor * ones_hd = nullptr;
    if (m.gemma_norm) {
        ones_h = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, H);
        ggml_set_name(ones_h, "ones_h");
        ggml_set_input(ones_h);
        if (m.layers[0].q_norm_w) {
            ones_hd = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, head_dim);
            ggml_set_name(ones_hd, "ones_hd");
            ggml_set_input(ones_hd);
        }
    }

    auto rms_norm_b = [&](ggml_tensor * x, ggml_tensor * w) -> ggml_tensor * {
        x = ggml_rms_norm(gctx, x, eps);
        if (m.gemma_norm) {
            ggml_tensor * ones = (w->ne[0] == H) ? ones_h : ones_hd;
            return ggml_mul(gctx, x, ggml_add(gctx, w, ones));
        }
        return ggml_mul(gctx, x, w);
    };

    // Position IDs
    ggml_tensor * pos = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T_total);
    ggml_set_name(pos, "pos_ids");
    ggml_set_input(pos);

    // Attention mask: [T_total, T_total, 1, 1] passed as kq_mask
    ggml_tensor * attn_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T_total, T_total);
    ggml_set_name(attn_mask, "attn_mask");
    ggml_set_input(attn_mask);

    // --- Transformer layers ---
    for (int il = 0; il < m.n_layer; il++) {
        const auto & L = m.layers[il];
        ggml_tensor * residual = cur;

        if (L.attn_norm_w) cur = rms_norm_b(cur, L.attn_norm_w);

        ggml_tensor * Q = ggml_mul_mat(gctx, L.q_w, cur);
        if (L.q_b) Q = ggml_add(gctx, Q, L.q_b);
        ggml_tensor * K = ggml_mul_mat(gctx, L.k_w, cur);
        if (L.k_b) K = ggml_add(gctx, K, L.k_b);
        ggml_tensor * V = ggml_mul_mat(gctx, L.v_w, cur);
        if (L.v_b) V = ggml_add(gctx, V, L.v_b);

        Q = ggml_reshape_3d(gctx, Q, head_dim, n_heads, T_total);
        K = ggml_reshape_3d(gctx, K, head_dim, n_kv_heads, T_total);
        V = ggml_reshape_3d(gctx, V, head_dim, n_kv_heads, T_total);

        if (L.q_norm_w) Q = rms_norm_b(Q, L.q_norm_w);
        if (L.k_norm_w) K = rms_norm_b(K, L.k_norm_w);

        float layer_rope_theta = m.rope_theta;
        if (m.rope_theta_local > 0.0f && m.global_attn_every_n > 0) {
            bool is_global = ((il + 1) % m.global_attn_every_n == 0);
            layer_rope_theta = is_global ? m.rope_theta : m.rope_theta_local;
        }

        int rope_mode = 2;  // NEOX
        Q = ggml_rope_ext(gctx, Q, pos, nullptr,
                           head_dim, rope_mode, 0,
                           layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(gctx, K, pos, nullptr,
                           head_dim, rope_mode, 0,
                           layer_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Manual attention with mask (no ggml_diag_mask_inf — we use explicit mask)
        Q = ggml_cont(gctx, ggml_permute(gctx, Q, 0, 2, 1, 3));
        K = ggml_cont(gctx, ggml_permute(gctx, K, 0, 2, 1, 3));
        V = ggml_cont(gctx, ggml_permute(gctx, V, 0, 2, 1, 3));

        float scale = (m.attn_scale > 0.0f)
                        ? (1.0f / sqrtf(m.attn_scale))
                        : (1.0f / sqrtf((float)head_dim));
        ggml_tensor * scores = ggml_mul_mat(gctx, K, Q);
        scores = ggml_scale(gctx, scores, scale);
        // Add block-diagonal mask
        scores = ggml_add(gctx, scores, attn_mask);
        scores = ggml_soft_max(gctx, scores);

        ggml_tensor * V_perm = ggml_cont(gctx, ggml_permute(gctx, V, 1, 0, 2, 3));
        ggml_tensor * attn = ggml_mul_mat(gctx, V_perm, scores);
        attn = ggml_cont(gctx, ggml_permute(gctx, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(gctx, attn, q_dim, T_total);

        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);

        if (L.post_attn_norm_w) attn = rms_norm_b(attn, L.post_attn_norm_w);

        if (L.pre_ffn_norm_w) {
            if (L.ffn_norm_w) attn = rms_norm_b(attn, L.ffn_norm_w);
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            cur = rms_norm_b(cur, L.pre_ffn_norm_w);
        } else {
            cur = ggml_add(gctx, residual, attn);
            residual = cur;
            if (L.ffn_norm_w) cur = rms_norm_b(cur, L.ffn_norm_w);
        }

        if (L.gate_w && L.up_w && L.down_w) {
            ggml_tensor * gate = ggml_mul_mat(gctx, L.gate_w, cur);
            if (m.activation == 2 || m.activation == 1) {
                gate = ggml_gelu(gctx, gate);
            } else {
                gate = ggml_silu(gctx, gate);
            }
            ggml_tensor * up = ggml_mul_mat(gctx, L.up_w, cur);
            ggml_tensor * ffn = ggml_mul(gctx, gate, up);
            ffn = ggml_mul_mat(gctx, L.down_w, ffn);
            if (L.post_ffn_norm_w) ffn = rms_norm_b(ffn, L.post_ffn_norm_w);
            cur = ggml_add(gctx, residual, ffn);
        } else {
            cur = residual;
        }
    }

    if (m.output_norm) cur = rms_norm_b(cur, m.output_norm);

    ggml_set_name(cur, "decoder_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // --- Set inputs and compute ---
    if (use_sched) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "decoder_batch: failed to allocate graph\n");
            ggml_free(gctx);
            // Fall back to sequential
            std::vector<std::vector<float>> results;
            for (const auto & tokens : batch) {
                results.push_back(decoder_encode_tokens(m, backend, tokens, n_threads,
                                                         sched, compute_meta));
            }
            return results;
        }

        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_ids"),
                                tok_data.data(), 0, T_total * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"),
                                pos_data.data(), 0, T_total * sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "attn_mask"),
                                mask_data.data(), 0,
                                (size_t)T_total * T_total * sizeof(float));
        if (m.gemma_norm) {
            std::vector<float> ones(H, 1.0f);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_h"),
                                    ones.data(), 0, H * sizeof(float));
            if (ones_hd) {
                std::vector<float> ones_hd_data(head_dim, 1.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "ones_hd"),
                                        ones_hd_data.data(), 0, head_dim * sizeof(float));
            }
        }

        ggml_backend_sched_graph_compute(sched, gf);
    } else {
        // CPU fallback
        memcpy(ids_t->data, tok_data.data(), T_total * sizeof(int32_t));
        memcpy(pos->data, pos_data.data(), T_total * sizeof(int32_t));
        memcpy(attn_mask->data, mask_data.data(),
               (size_t)T_total * T_total * sizeof(float));

        if (ones_h) {
            float * d = (float *)ones_h->data;
            for (int i = 0; i < H; i++) d[i] = 1.0f;
        }
        if (ones_hd) {
            float * d = (float *)ones_hd->data;
            for (int i = 0; i < head_dim; i++) d[i] = 1.0f;
        }

        struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads, NULL);
        std::vector<uint8_t> work;
        if (cplan.work_size > 0) {
            work.resize(cplan.work_size);
            cplan.work_data = work.data();
        }
        ggml_graph_compute(gf, &cplan);
    }

    // --- Read output and pool per text ---
    ggml_tensor * out = ggml_graph_get_tensor(gf, "decoder_out");
    std::vector<float> hidden(H * T_total);
    if (use_sched) {
        ggml_backend_tensor_get(out, hidden.data(), 0, (size_t)H * T_total * sizeof(float));
    } else {
        memcpy(hidden.data(), out->data, (size_t)H * T_total * sizeof(float));
    }

    ggml_free(gctx);

    // Per-text pooling
    std::vector<std::vector<float>> results(B);
    for (int b = 0; b < B; b++) {
        std::vector<float> pooled(H, 0.0f);

        if (m.pooling_method == 1) {
            // Mean pooling
            int n_valid = 0;
            for (int t = 0; t < lens[b]; t++) {
                const float * row = hidden.data() + (size_t)(b * T_max + t) * H;
                for (int i = 0; i < H; i++) pooled[i] += row[i];
                n_valid++;
            }
            if (n_valid > 0) {
                const float inv = 1.0f / (float)n_valid;
                for (int i = 0; i < H; i++) pooled[i] *= inv;
            }
        } else {
            // Last-token pooling
            int last_t = lens[b] - 1;
            if (last_t < 0) last_t = 0;
            memcpy(pooled.data(), hidden.data() + (size_t)(b * T_max + last_t) * H,
                   H * sizeof(float));
        }

        // Dense projection
        for (const auto & dl : m.dense_proj) {
            std::vector<float> out_vec(dl.out_dim, 0.0f);
            const float * w = dl.weight.data();
            for (int o = 0; o < dl.out_dim; o++) {
                float acc = 0.0f;
                for (int j = 0; j < dl.in_dim; j++) acc += w[o * dl.in_dim + j] * pooled[j];
                out_vec[o] = acc;
            }
            pooled = std::move(out_vec);
        }

        // L2 normalize
        float norm = 0;
        for (int i = 0; i < (int)pooled.size(); i++) norm += pooled[i] * pooled[i];
        norm = sqrtf(std::max(norm, 1e-12f));
        for (int i = 0; i < (int)pooled.size(); i++) pooled[i] /= norm;

        results[b] = std::move(pooled);
    }

    return results;
}
