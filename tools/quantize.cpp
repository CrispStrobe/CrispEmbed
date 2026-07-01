// quantize.cpp — GGUF tensor re-quantization tool for CrispEmbed.
//
// Adapted from CrispASR/examples/crispasr-quantize/main.cpp.
// Takes any GGUF model and re-quantizes all eligible tensors to the
// target type, preserving metadata and non-quantizable tensors
// (norms, positional embeddings, biases, small tables).
//
// Usage:
//   crispembed-quantize input.gguf output.gguf q4_k
//   crispembed-quantize input.gguf output.gguf q8_0

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static const std::map<std::string, enum ggml_ftype> FTYPE_MAP = {
    {"f16",  GGML_FTYPE_MOSTLY_F16},
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
    {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
    {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
    {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
    {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

// When set, LLM decoder weight matrices (prefix "l.") are kept at F16 instead of
// being quantized. Small decoders (e.g. GOT-OCR2's 0.5B Qwen2) are catastrophically
// sensitive to q8_0/k-quant weights — llm_layer_0 cos drops to ~0.936 (vs 0.9999 at
// F16) and the OCR output degenerates. Enable with --decoder-f16. See issue #25.
static bool g_decoder_f16 = false;

static bool quantize_model(const std::string & fname_inp, const std::string & fname_out, ggml_ftype ftype) {
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_F16:  qtype = GGML_TYPE_F16;  break;
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        default:
            fprintf(stderr, "unsupported quantization type %d\n", ftype);
            return false;
    }

    printf("Loading model from '%s'\n", fname_inp.c_str());

    // Load with no_alloc=true so we can read tensor data from file directly
    struct ggml_context * ctx_in_ggml = nullptr;
    struct gguf_init_params params = { /*no_alloc*/ true, /*ctx*/ &ctx_in_ggml };
    struct gguf_context * ctx_in = gguf_init_from_file(fname_inp.c_str(), params);
    if (!ctx_in || !ctx_in_ggml) {
        fprintf(stderr, "Failed to load model from '%s'\n", fname_inp.c_str());
        return false;
    }

    // Build output GGUF with same metadata
    struct gguf_context * ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    const int n_tensors = gguf_get_n_tensors(ctx_in);

    // CNN/face model detection: scan tensor names for known prefixes
    {
        bool is_cnn_model = false;
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_in, i);
            std::string sn(name);
            if (sn.rfind("cnn.", 0) == 0 ||
                sn.rfind("scrfd.", 0) == 0 ||
                sn.rfind("arcface.", 0) == 0 ||
                sn.rfind("sface.", 0) == 0) {
                is_cnn_model = true;
                break;
            }
        }
        if (is_cnn_model) {
            fprintf(stderr, "Warning: CNN/face model detected — conv2d tensors will be kept at original precision\n");
        }
    }

    // Pre-scan: flatten 4D conv weights to 2D [OC, IC*KH*KW] in the tensor
    // metadata so the output header has correct shapes. Data is transposed
    // during the per-tensor write loop below.
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_in_ggml, name);
        if (ggml_n_dims(t) == 4 && t->type == GGML_TYPE_F32) {
            // Flatten [KW, KH, IC, OC] → [IC*KH*KW, OC]
            int64_t flat_cols = t->ne[0] * t->ne[1] * t->ne[2];
            int64_t OC = t->ne[3];
            t->ne[0] = flat_cols;
            t->ne[1] = OC;
            t->ne[2] = 1;
            t->ne[3] = 1;
            t->nb[0] = sizeof(float);
            t->nb[1] = flat_cols * sizeof(float);
            t->nb[2] = t->nb[1] * OC;
            t->nb[3] = t->nb[2];
        }
        gguf_add_tensor(ctx_out, t);
    }

    // Write output file
    printf("Writing quantized model to '%s'\n", fname_out.c_str());
    FILE * fout = fopen(fname_out.c_str(), "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open '%s' for writing\n", fname_out.c_str());
        gguf_free(ctx_in);
        gguf_free(ctx_out);
        ggml_free(ctx_in_ggml);
        return false;
    }

    // Write metadata placeholder (will be overwritten at the end)
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size, 0);
    fwrite(meta_data.data(), 1, meta_size, fout);

    // Open input file for data reading
    FILE * fin = fopen(fname_inp.c_str(), "rb");
    const size_t data_offset_in = gguf_get_data_offset(ctx_in);

    std::vector<float>   f32_data;
    std::vector<uint8_t> q_data;

    int n_quantized = 0, n_kept = 0;
    size_t total_orig = 0, total_new = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_in_ggml, name);

        enum ggml_type type = t->type;
        size_t size = ggml_nbytes(t);
        size_t offset = data_offset_in + gguf_get_tensor_offset(ctx_in, i);

        printf("[%3d/%3d] %-45s - %6s, ", i + 1, n_tensors, name, ggml_type_name(type));

        // Decide whether to quantize this tensor:
        // - Must be F32 or F16 source
        // - Must be 2D (weight matrices)
        // - Must contain "weight" or ".w" in name
        // - Must NOT contain "norm" in name
        // - Token/position/type embeddings: only Q8_0/F16, skip aggressive quants
        std::string sname(name);

        // Guard 1: patch_embed tensors — always copy as-is (they are conv2d kernels)
        // patch_embed, downsample, merger — used in host-side computation,
        // must stay F32 (ggml_backend_tensor_get reads as float).
        if (sname.find("patch_embed") != std::string::npos ||
            sname.find("downsample") != std::string::npos ||
            sname.find("merger") != std::string::npos) {
            printf("note: %s — copying as-is (host-side computation)\n", name);
            size_t sz = ggml_nbytes(t);
            size_t off = data_offset_in + gguf_get_tensor_offset(ctx_in, i);
#ifdef _WIN32
            _fseeki64(fin, (int64_t)off, SEEK_SET);
#else
            fseeko(fin, (off_t)off, SEEK_SET);
#endif
            std::vector<uint8_t> raw(sz);
            if (fread(raw.data(), 1, sz, fin) != sz) {
                fprintf(stderr, "failed to read raw data for patch_embed tensor\n");
                fclose(fin); fclose(fout);
                return false;
            }
            fwrite(raw.data(), 1, sz, fout);
            size_t pad = GGML_PAD(sz, GGUF_DEFAULT_ALIGNMENT) - sz;
            for (size_t j = 0; j < pad; j++) fputc(0, fout);
            total_orig += sz;
            total_new += sz;
            n_kept++;
            continue;
        }

        // Guard 2: LoRA adapter tensors — keep at source precision (F16/F32)
        // LoRA A/B matrices are low-rank (rank=32 typical) and quantizing
        // them destroys the decomposition quality. They're small anyway.
        if (sname.find("lora.") != std::string::npos) {
            printf("note: LoRA tensor — keeping as %s\n", ggml_type_name(type));
            size_t sz = ggml_nbytes(t);
            size_t off = data_offset_in + gguf_get_tensor_offset(ctx_in, i);
#ifdef _WIN32
            _fseeki64(fin, (int64_t)off, SEEK_SET);
#else
            fseeko(fin, (off_t)off, SEEK_SET);
#endif
            std::vector<uint8_t> raw(sz);
            if (fread(raw.data(), 1, sz, fin) != sz) {
                fprintf(stderr, "failed to read LoRA tensor data\n");
                fclose(fin); fclose(fout);
                return false;
            }
            fwrite(raw.data(), 1, sz, fout);
            size_t pad = GGML_PAD(sz, GGUF_DEFAULT_ALIGNMENT) - sz;
            for (size_t j = 0; j < pad; j++) fputc(0, fout);
            total_orig += sz;
            total_new += sz;
            n_kept++;
            continue;
        }

        // Guard 3: 5D+ tensors — copy as-is (no known use case)
        // 4D tensors were pre-flattened to 2D above, so ggml_n_dims <= 3 here.
        // 3D tensors (MoE expert weights) are quantized via the standard path.
        if (ggml_n_dims(t) >= 5) {
            int ndims = ggml_n_dims(t);
            printf("note: skipping %d-D tensor — copying as-is\n", ndims);
            size_t sz = ggml_nbytes(t);
            size_t off = data_offset_in + gguf_get_tensor_offset(ctx_in, i);
#ifdef _WIN32
            _fseeki64(fin, (int64_t)off, SEEK_SET);
#else
            fseeko(fin, (off_t)off, SEEK_SET);
#endif
            std::vector<uint8_t> raw(sz);
            if (fread(raw.data(), 1, sz, fin) != sz) {
                fprintf(stderr, "failed to read raw data for %d-D tensor\n", ndims);
                fclose(fin); fclose(fout);
                return false;
            }
            fwrite(raw.data(), 1, sz, fout);
            size_t pad = GGML_PAD(sz, GGUF_DEFAULT_ALIGNMENT) - sz;
            for (size_t j = 0; j < pad; j++) fputc(0, fout);
            total_orig += sz;
            total_new += sz;
            n_kept++;
            continue;
        }

        bool is_embd = sname.find("embd") != std::string::npos ||
                        sname.find("embed") != std::string::npos ||
                        sname.find("token_types") != std::string::npos;
        // Skip tiny embedding tables (token_types has only 2 rows)
        // — quantizing these breaks Ollama's binary ops (f32 + q8_0)
        bool is_tiny_embd = (t->ne[1] <= 4) &&
                            (sname.find("token_types") != std::string::npos ||
                             sname.find("type_embd") != std::string::npos);
        // Position/class embeddings, LayerScale, and NAFNet beta/gamma
        // used in ggml_add/ggml_mul — must stay F32 (binary ops don't
        // support F32 + Q8_0/F16 operands, and these are tiny scale factors)
        bool is_add_operand = sname.find("position_embedding") != std::string::npos ||
                              sname.find("class_embedding") != std::string::npos ||
                              sname.find(".ls1") != std::string::npos ||
                              sname.find(".ls2") != std::string::npos ||
                              sname.find(".beta") != std::string::npos ||
                              sname.find(".gamma") != std::string::npos;
        if (is_add_operand) {
            is_tiny_embd = true;  // force copy-as-is
        }
        bool quantize = (ggml_is_quantized(qtype) || qtype == GGML_TYPE_F16) &&
                        (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) &&
                        (ggml_n_dims(t) >= 2) &&
                        !is_tiny_embd;
        const int64_t ncols = t->ne[0];
        ggml_type qtype_used = qtype;

        // Embedding tables: use Q8_0 for aggressive quants to preserve quality
        // while still compressing (embedding tables are huge, ~50% of model)
        if (quantize && is_embd &&
            qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16) {
            qtype_used = GGML_TYPE_Q8_0;
        }

        // Vision encoder weights: keep at Q8_0 minimum for OCR quality. The
        // vision encoder directly determines text recognition accuracy, so
        // aggressive quantization (Q4_K, Q3_K, Q2_K) degrades it. Covers
        // "v.*" (SAM ViT / merger), "qe.*" (the DeepSeek-OCR Qwen2 vision
        // encoder), and "vis.*" (granite_vision SigLIP, smoldocling) — the
        // latter does NOT start with "v." so it was being aggressively
        // quantized. Worse, SigLIP's D=1152 is not 256-divisible, so a Q4_K
        // target fell back to Q4_0 (legacy 4-bit) on the vision weights; Q8_0
        // has block size 32 (1152 % 32 == 0) so it applies cleanly here.
        // Also keep the multimodal projector ("proj.*") at Q8_0: it is a tiny
        // 2-layer bridge from vision features into the LLM embedding space, and
        // quantizing it to Q4_K measurably hurt parity (HF-blueprint projector
        // cos 0.929 at Q4_K vs ~0.95 at Q8_0) for negligible size.
        bool is_vision_weight = sname.rfind("v.", 0) == 0 ||
                                sname.rfind("c.", 0) == 0 ||
                                sname.rfind("qe.", 0) == 0 ||
                                sname.rfind("vis.", 0) == 0 ||
                                sname.rfind("proj.", 0) == 0;
        if (quantize && is_vision_weight &&
            qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(vision→Q8_0) ");
        }

        // MoE router / gating weights (DeepSeek-V2: "*.mlp_gate.weight", also
        // the generic "ffn_gate_inp"): these pick which experts run, so even
        // small quant error flips the top-k selection and corrupts the output.
        // Keep them at Q8_0 minimum (they are tiny: n_experts × hidden).
        bool is_moe_router = sname.find("mlp_gate.weight") != std::string::npos ||
                             sname.find("ffn_gate_inp") != std::string::npos;
        if (quantize && is_moe_router &&
            qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(moe-router→Q8_0) ");
        }

        // LM head / output projection: produces the token logits over a large
        // vocabulary, so Q4_K error here directly perturbs the softmax and flips
        // borderline greedy picks (measured: Unlimited-OCR decoder logits cos vs
        // HF was only 0.926 with a Q4_K lm_head, dropping from ~0.979 at the last
        // hidden state). Keep it at Q8_0 minimum — cheap relative to the experts
        // (~+90 MB on a 2 GB model). Matches "lm_head.weight" (this model) and
        // the generic llama.cpp "output.weight" (but NOT "output_norm.weight").
        bool is_lm_head = sname.find("lm_head.weight") != std::string::npos ||
                          sname == "output.weight" ||
                          sname.find(".output.weight") != std::string::npos;
        if (quantize && is_lm_head &&
            qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(lm-head→Q8_0) ");
        }

        // LLM decoder weights (prefix "l.": attn_*, ffn_*, embed_tokens): keep at
        // F16 when --decoder-f16 is set. A tiny 0.5B Qwen2 decoder loses too much
        // to q8_0/k-quants (llm_layer_0 cos 0.936 → garbage OCR); F16 restores
        // cos 0.9999 and correct output. Norms/biases are 1-D and already copied.
        if (quantize && g_decoder_f16 && sname.rfind("l.", 0) == 0 &&
            qtype_used != GGML_TYPE_F16 && qtype_used != GGML_TYPE_F32) {
            qtype_used = GGML_TYPE_F16;
            printf("(decoder→F16) ");
        }

        int64_t qk = ggml_blck_size(qtype_used);

        // Fallback chain for K-quants: if row width isn't 256-aligned,
        // fall back to a legacy quant with block size 32.
        if (quantize && ncols % qk != 0) {
            ggml_type fallback = GGML_TYPE_COUNT;
            switch (qtype) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K: fallback = GGML_TYPE_Q4_0; break;
                case GGML_TYPE_Q5_K: fallback = GGML_TYPE_Q5_0; break;
                case GGML_TYPE_Q6_K: fallback = GGML_TYPE_Q8_0; break;
                default: break;
            }
            if (fallback != GGML_TYPE_COUNT && ncols % ggml_blck_size(fallback) == 0) {
                qtype_used = fallback;
                qk = ggml_blck_size(qtype_used);
                printf("(fallback %s) ", ggml_type_name(qtype_used));
            } else {
                printf("skip (ncols %lld not div by %lld)\n", (long long)ncols, (long long)qk);
                quantize = false;
            }
        }

#ifdef _WIN32
        _fseeki64(fin, (int64_t)offset, SEEK_SET);
#else
        fseeko(fin, (off_t)offset, SEEK_SET);
#endif

        if (quantize) {
            printf("quantizing to %s... ", ggml_type_name(qtype_used));

            const int64_t nelements = ggml_nelements(t);
            f32_data.resize(nelements);

            if (type == GGML_TYPE_F32) {
                if (fread(f32_data.data(), sizeof(float), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f32 data\n");
                    fclose(fin); fclose(fout);
                    return false;
                }
            } else {
                std::vector<ggml_fp16_t> f16_data(nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    fclose(fin); fclose(fout);
                    return false;
                }
                for (int64_t j = 0; j < nelements; j++) {
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
                }
            }

            const size_t max_q_size = ggml_row_size(qtype_used, t->ne[0]) * (nelements / t->ne[0]);
            q_data.resize(max_q_size);

            size_t q_size = ggml_quantize_chunk(qtype_used, f32_data.data(), q_data.data(),
                                                 0, nelements / t->ne[0], t->ne[0], nullptr);

            fwrite(q_data.data(), 1, q_size, fout);
            gguf_set_tensor_type(ctx_out, name, qtype_used);

            // Alignment padding
            size_t pad = GGML_PAD(q_size, GGUF_DEFAULT_ALIGNMENT) - q_size;
            for (size_t j = 0; j < pad; j++) fputc(0, fout);

            total_new += q_size;
            n_quantized++;
            printf("%.1f MB -> %.1f MB\n", size / 1048576.0, q_size / 1048576.0);
        } else {
            // Copy tensor as-is
            std::vector<uint8_t> raw_data(size);
            if (fread(raw_data.data(), 1, size, fin) != size) {
                fprintf(stderr, "failed to read raw data\n");
                fclose(fin); fclose(fout);
                return false;
            }
            fwrite(raw_data.data(), 1, size, fout);

            size_t pad = GGML_PAD(size, GGUF_DEFAULT_ALIGNMENT) - size;
            for (size_t j = 0; j < pad; j++) fputc(0, fout);

            total_new += size;
            n_kept++;
            printf("copy %.1f MB\n", size / 1048576.0);
        }
        total_orig += size;
    }

    // Rewrite metadata header now that tensor types/offsets are final
    rewind(fout);
    gguf_get_meta_data(ctx_out, meta_data.data());
    fwrite(meta_data.data(), 1, meta_size, fout);

    fclose(fin);
    fclose(fout);
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ctx_in_ggml);

    printf("\n%d quantized, %d kept\n", n_quantized, n_kept);
    printf("%.0f MB -> %.0f MB (%.1fx compression)\n",
           total_orig / 1048576.0, total_new / 1048576.0,
           (double)total_orig / (total_new > 0 ? (double)total_new : 1.0));

    return true;
}

int main(int argc, char ** argv) {
    // Collect positional args, allowing an optional --decoder-f16 flag anywhere.
    std::vector<std::string> pos;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--decoder-f16") g_decoder_f16 = true;
        else pos.push_back(a);
    }
    if (pos.size() != 3) {
        fprintf(stderr, "usage: %s <input.gguf> <output.gguf> <type> [--decoder-f16]\n\n", argv[0]);
        fprintf(stderr, "  --decoder-f16  keep LLM decoder weights (prefix 'l.') at F16\n");
        fprintf(stderr, "                 (required for small decoders like GOT-OCR2's 0.5B; see #25)\n\n");
        fprintf(stderr, "Supported types:\n");
        for (auto & [name, _] : FTYPE_MAP) {
            fprintf(stderr, "  %s\n", name.c_str());
        }
        return 1;
    }

    const std::string fname_inp = pos[0];
    const std::string fname_out = pos[1];
    const char * type_str = pos[2].c_str();

    auto it = FTYPE_MAP.find(type_str);
    if (it == FTYPE_MAP.end()) {
        fprintf(stderr, "Unknown quantization type: %s\n", type_str);
        return 1;
    }

    if (!quantize_model(fname_inp, fname_out, it->second)) {
        fprintf(stderr, "Failed to quantize model\n");
        return 1;
    }

    return 0;
}
