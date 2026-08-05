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

#include "../src/core/imatrix_alias.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>

static const std::map<std::string, enum ggml_ftype> FTYPE_MAP = {
    { "f16", GGML_FTYPE_MOSTLY_F16 },       { "q4_0", GGML_FTYPE_MOSTLY_Q4_0 }, { "q4_1", GGML_FTYPE_MOSTLY_Q4_1 },
    { "q5_0", GGML_FTYPE_MOSTLY_Q5_0 },     { "q5_1", GGML_FTYPE_MOSTLY_Q5_1 }, { "q8_0", GGML_FTYPE_MOSTLY_Q8_0 },
    { "q2_k", GGML_FTYPE_MOSTLY_Q2_K },     { "q3_k", GGML_FTYPE_MOSTLY_Q3_K }, { "q4_k", GGML_FTYPE_MOSTLY_Q4_K },
    { "q5_k", GGML_FTYPE_MOSTLY_Q5_K },     { "q6_k", GGML_FTYPE_MOSTLY_Q6_K }, { "iq4_nl", GGML_FTYPE_MOSTLY_IQ4_NL },
    { "iq4_xs", GGML_FTYPE_MOSTLY_IQ4_XS },
};

// When set, LLM decoder weight matrices (prefix "l.") are kept at F16 instead of
// being quantized. Small decoders (e.g. GOT-OCR2's 0.5B Qwen2) are catastrophically
// sensitive to q8_0/k-quant weights — llm_layer_0 cos drops to ~0.936 (vs 0.9999 at
// F16) and the OCR output degenerates. Enable with --decoder-f16. See issue #25.
static bool g_decoder_f16 = false;
static bool g_ppocrv6_q8_head = false;

// Per-tensor importance vectors loaded from a CrispEmbed imatrix file
// (see src/imatrix.cpp). Keyed by weight name; value length == n_per_row.
// importance[c] = sum_of_squares[c] / count. Passed to ggml_quantize_chunk,
// which uses it to minimise activation-weighted error for k-quants / IQ-quants.
static std::map<std::string, std::vector<float>> g_imatrix;

// BERT-family runtimes pre-merge the per-layer attn q/k/v weights into one F32
// tensor at load time, so the imatrix collector records that matmul's input
// statistics under the merged name — there is no per-weight entry for
// attn.{q,k,v}.weight. The name contract and the q/k/v -> merged alias live
// in src/core/imatrix_alias.h, shared with the runtime's naming site in
// src/crispembed.cpp and guarded by tests/test_imatrix_alias.cpp.
using core_imatrix::qkv_merged_alias;

// How to resolve importance for an attn q/k/v weight that has BOTH a direct
// entry and a merged-QKV alias entry (`CRISPEMBED_QUANT_IMATRIX_QKV`):
//
//   direct (default) — the shipped behaviour: a direct entry wins.
//   merged           — the alias wins; the direct entry is ignored.
//   sum              — element-wise sum of both vectors.
//
// It matters for DeBERTa-v2 (mxbai-rerank-*): disentangled attention applies
// L.q_w / L.k_w a SECOND time, to the relative-position embeddings, under the
// weights' own GGUF names. The collector therefore files a direct
// blk.N.attn_{q,k}.weight entry whose input statistics come from the shared
// rel-position tensor, not the token stream, and that entry shadows the
// correct merged-alias vector. See tests/results/mxbai-qk-imatrix/SUMMARY.md.
enum class qkv_imatrix_mode { direct, merged, sum };

static qkv_imatrix_mode g_qkv_mode = qkv_imatrix_mode::direct;

static void init_qkv_imatrix_mode() {
    const char * e = std::getenv("CRISPEMBED_QUANT_IMATRIX_QKV");
    if (!e || !*e) return;
    if (strcmp(e, "merged") == 0) {
        g_qkv_mode = qkv_imatrix_mode::merged;
    } else if (strcmp(e, "sum") == 0) {
        g_qkv_mode = qkv_imatrix_mode::sum;
    } else if (strcmp(e, "direct") != 0) {
        fprintf(stderr, "imatrix: unknown CRISPEMBED_QUANT_IMATRIX_QKV='%s' (direct|merged|sum), using direct\n", e);
        return;
    }
    fprintf(stderr, "imatrix: attn q/k/v importance mode = %s\n", e);
}

static bool load_imatrix(const std::string & path) {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &ctx };
    struct gguf_context * g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        fprintf(stderr, "imatrix: failed to open '%s'\n", path.c_str());
        return false;
    }
    const int64_t nt = gguf_get_n_tensors(g);
    int loaded = 0;
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_F32) continue;
        const int64_t ne0 = t->ne[0];
        const float * d = (const float *)t->data;
        std::string ck = std::string("count.") + name;
        int64_t kid = gguf_find_key(g, ck.c_str());
        uint64_t count = (kid >= 0) ? gguf_get_val_u64(g, kid) : 0;
        if (count == 0) continue;
        std::vector<float> imp((size_t)ne0);
        const double inv = 1.0 / (double)count;
        for (int64_t c = 0; c < ne0; c++) imp[c] = (float)((double)d[c] * inv);
        g_imatrix[name] = std::move(imp);
        loaded++;
    }
    gguf_free(g);
    ggml_free(ctx);
    fprintf(stderr, "imatrix: loaded importance vectors for %d tensors from '%s'\n", loaded, path.c_str());
    return loaded > 0;
}

static bool quantize_model(const std::string & fname_inp, const std::string & fname_out, ggml_ftype ftype) {
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
    case GGML_FTYPE_MOSTLY_F16:
        qtype = GGML_TYPE_F16;
        break;
    case GGML_FTYPE_MOSTLY_Q4_0:
        qtype = GGML_TYPE_Q4_0;
        break;
    case GGML_FTYPE_MOSTLY_Q4_1:
        qtype = GGML_TYPE_Q4_1;
        break;
    case GGML_FTYPE_MOSTLY_Q5_0:
        qtype = GGML_TYPE_Q5_0;
        break;
    case GGML_FTYPE_MOSTLY_Q5_1:
        qtype = GGML_TYPE_Q5_1;
        break;
    case GGML_FTYPE_MOSTLY_Q8_0:
        qtype = GGML_TYPE_Q8_0;
        break;
    case GGML_FTYPE_MOSTLY_Q2_K:
        qtype = GGML_TYPE_Q2_K;
        break;
    case GGML_FTYPE_MOSTLY_Q3_K:
        qtype = GGML_TYPE_Q3_K;
        break;
    case GGML_FTYPE_MOSTLY_Q4_K:
        qtype = GGML_TYPE_Q4_K;
        break;
    case GGML_FTYPE_MOSTLY_Q5_K:
        qtype = GGML_TYPE_Q5_K;
        break;
    case GGML_FTYPE_MOSTLY_Q6_K:
        qtype = GGML_TYPE_Q6_K;
        break;
    case GGML_FTYPE_MOSTLY_IQ4_NL:
        qtype = GGML_TYPE_IQ4_NL;
        break;
    case GGML_FTYPE_MOSTLY_IQ4_XS:
        qtype = GGML_TYPE_IQ4_XS;
        break;
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
    const int arch_key = gguf_find_key(ctx_in, "general.architecture");
    const bool is_ppocrv6 = arch_key >= 0 && std::string(gguf_get_val_str(ctx_in, arch_key)) == "ppocrv6";
    const bool is_tesseract_lstm = arch_key >= 0 && std::string(gguf_get_val_str(ctx_in, arch_key)) == "tesseract_lstm";
    const bool is_internvl2 = arch_key >= 0 && std::string(gguf_get_val_str(ctx_in, arch_key)) == "internvl2";
    const bool ppocr_q8_late = is_ppocrv6 && ftype == GGML_FTYPE_MOSTLY_Q8_0 && g_ppocrv6_q8_head;
    // Sub-Q8 on an InternVL2-family decoder is checkpoint-dependent: WARN, do
    // not refuse. Measured on h2ovl-mississippi-2b (2560d, 32H/8KV, head_dim 80)
    // against a Python-blueprint reference, 7 real stages, same binary:
    //
    //   precision              llm_layer_0   llm_layer_2   verdict
    //   f16                    0.999972      0.999972      transcribes
    //   q8_0                   0.998033      0.995498      transcribes
    //   q4_k (attn held Q8_0)  0.922039      0.543576      still wrong
    //   q4_k                   0.594995     -0.268615      anti-correlated
    //   q6_k                   -- fails to load at all --
    //
    // Not a shape problem: every ne[0] involved (2560, 6912) is 256-divisible,
    // so Q4_K applies cleanly and still wrecks that decoder.
    //
    // But it does NOT generalise, and an earlier version of this guard that
    // refused for the whole arch was wrong: internvl2-1b ships q4_k in the
    // registry and OCRs correctly, and h2ovl-800m is recorded verified at q4_k.
    // One measured checkpoint does not license blocking two working ones, so
    // this warns and points at the gate that actually decides — the decoded
    // output. Attention is still held at Q8_0 below, which recovers about half
    // and costs little.
    if (is_internvl2 && ftype != GGML_FTYPE_MOSTLY_Q8_0 && ftype != GGML_FTYPE_MOSTLY_F16 &&
        ftype != GGML_FTYPE_ALL_F32) {
        fprintf(stderr, "Warning: internvl2 below Q8_0 is checkpoint-dependent. h2ovl-mississippi-2b's\n"
                        "         decoder diverges badly here (llm_layer_0 cos 0.594995, anti-correlated\n"
                        "         by layer 2) while internvl2-1b and h2ovl-800m are fine. Verify the\n"
                        "         DECODED OUTPUT before shipping this file.\n");
    }

    if (is_ppocrv6) {
        fprintf(stderr, "PP-OCRv6 precision policy: biases/SE/depthwise/early-head tensors stay F16/F32; CTC logits "
                        "head is Q8_0 minimum\n");
    }

    // CNN/face model detection: scan tensor names for known prefixes
    {
        bool is_cnn_model = false;
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx_in, i);
            std::string sn(name);
            if (sn.rfind("cnn.", 0) == 0 || sn.rfind("scrfd.", 0) == 0 || sn.rfind("arcface.", 0) == 0 ||
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

    std::vector<float> f32_data;
    std::vector<uint8_t> q_data;

    int n_quantized = 0, n_kept = 0, n_imatrix = 0;
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
        // SMT ConvNext: dwconv/downsampling are conv2d kernels; smt.positional_1d
        // is a baked sinusoidal PE consumed by ggml_add. All must stay F32
        // (quantizing a conv kernel or an add-operand PE breaks the graph).
        // TrOMR: the ResNetV2 backbone (enc.bb.*, incl. stem.conv/conv1/2/3) and the
        // HybridEmbed projector (enc.proj) are conv2d kernels flattened to 2D above;
        // quantizing them makes the in-engine reshape-to-4D produce an ne[0] that is
        // not a multiple of the block size → ggml_dup abort. Keep them as-is.
        // Transcoda ConvNeXt-V2 conv2d kernels (short-named by the converter): the
        // stem patch conv (enc.embed.patch), the downsample convs (enc.st*.ds.conv)
        // and the depthwise convs (enc.st*.l*.dw) are reshaped to 4D in-engine, so
        // quantizing them yields an ne[0] not %32 → ggml_dup/cast abort. The
        // pointwise convs (pw1/pw2) are matmuls in-engine and quantize fine.
        const bool ppocr_keep =
            is_ppocrv6 &&
            // PP-OCRv6 is a compact CNN/CTC graph: quantization error compounds
            // through every block and changes greedy CTC ties. Keep both the
            // detector and recognizer paths in F16; the policy-q4 container then
            // quantizes only non-critical metadata/auxiliary tensors and remains
            // a quality-preserving deployment variant.
            (!ppocr_q8_late || (sname.find("rec.head.encoder.") == std::string::npos &&
                                sname.find("rec.head.head.") == std::string::npos)) &&
            (sname.rfind("det.", 0) == 0 || sname.rfind("rec.", 0) == 0 || sname.find(".bias") != std::string::npos ||
             sname.find("normalization") != std::string::npos ||
             sname.find("squeeze_excitation") != std::string::npos ||
             sname.find("token_squeeze") != std::string::npos || sname.find("token_conv") != std::string::npos ||
             sname.find(".se1.") != std::string::npos || sname.find(".se2.") != std::string::npos ||
             sname.find(".dw.") != std::string::npos || sname.find("det.bb.stem") != std::string::npos ||
             sname.find("det.neck.") != std::string::npos ||
             (sname.find("head.") != std::string::npos && sname.find("head.fc2.weight") == std::string::npos));
        // Tesseract's output projection is the CTC decision boundary. Keeping
        // it at F32 avoids changing near-tied character logits even when the
        // recurrent matrices are quantized. The output bias is already kept
        // by the generic bias rule below.
        // Tesseract's native int-mode uses the separately preserved source
        // int8 matrices. Keep every float matrix lossless in the container so
        // the F32 path and the bias/activation reference do not acquire a
        // second, unrelated ggml quantization error.
        const bool tesseract_keep =
            is_tesseract_lstm && (sname == "output.weight" || sname.find(".weight") != std::string::npos);
        if (ppocr_keep || tesseract_keep || sname.find("patch_embed") != std::string::npos ||
            sname.find("downsample") != std::string::npos || sname.find("downsampling") != std::string::npos ||
            sname.find("dwconv") != std::string::npos || sname.find("enc.bb") != std::string::npos ||
            sname.find("enc.proj") != std::string::npos || sname.find("enc.embed.patch") != std::string::npos ||
            sname.find(".ds.conv") != std::string::npos || sname.find(".dw.") != std::string::npos ||
            sname.find("positional") != std::string::npos || sname.find("merger") != std::string::npos) {
            size_t off = data_offset_in + gguf_get_tensor_offset(ctx_in, i);
#ifdef _WIN32
            _fseeki64(fin, (int64_t)off, SEEK_SET);
#else
            fseeko(fin, (off_t)off, SEEK_SET);
#endif
            // TrOMR (`enc.bb.*`) backbone conv *weights* are cast to F16 in-engine
            // (tromr_ocr.cpp prep_conv) regardless of storage precision, so an F16
            // GGUF is LOSSLESS to the computation and ~halves the kept-conv bytes.
            // Gated on the TrOMR-only `enc.bb` prefix, so no other model is affected.
            bool conv_to_f16 = (type == GGML_TYPE_F32) && sname.find("enc.bb") != std::string::npos &&
                               sname.find("conv") != std::string::npos && sname.find(".weight") != std::string::npos;
            if (conv_to_f16) {
                const int64_t n = ggml_nelements(t);
                std::vector<float> f32buf(n);
                if (fread(f32buf.data(), sizeof(float), n, fin) != (size_t)n) {
                    fprintf(stderr, "failed to read conv tensor for F16 conversion\n");
                    fclose(fin);
                    fclose(fout);
                    return false;
                }
                std::vector<ggml_fp16_t> f16buf(n);
                for (int64_t j = 0; j < n; j++) f16buf[j] = ggml_fp32_to_fp16(f32buf[j]);
                const size_t sz16 = (size_t)n * sizeof(ggml_fp16_t);
                fwrite(f16buf.data(), 1, sz16, fout);
                gguf_set_tensor_type(ctx_out, name, GGML_TYPE_F16);
                size_t pad16 = GGML_PAD(sz16, GGUF_DEFAULT_ALIGNMENT) - sz16;
                for (size_t j = 0; j < pad16; j++) fputc(0, fout);
                printf("note: %s — F16 (backbone conv, engine casts to F16)\n", name);
                total_orig += ggml_nbytes(t);
                total_new += sz16;
                n_kept++;
                continue;
            }
            printf("note: %s — copying as-is (host-side computation)\n", name);
            size_t sz = ggml_nbytes(t);
            std::vector<uint8_t> raw(sz);
            if (fread(raw.data(), 1, sz, fin) != sz) {
                fprintf(stderr, "failed to read raw data for patch_embed tensor\n");
                fclose(fin);
                fclose(fout);
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
                fclose(fin);
                fclose(fout);
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
                fclose(fin);
                fclose(fout);
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

        bool is_embd = sname.find("embd") != std::string::npos || sname.find("embed") != std::string::npos ||
                       sname.find("token_types") != std::string::npos;
        // Skip tiny embedding tables (token_types has only 2 rows)
        // — quantizing these breaks Ollama's binary ops (f32 + q8_0)
        bool is_tiny_embd = (t->ne[1] <= 4) && (sname.find("token_types") != std::string::npos ||
                                                sname.find("type_embd") != std::string::npos);
        // Position/class embeddings, LayerScale, and NAFNet beta/gamma
        // used in ggml_add/ggml_mul — must stay F32 (binary ops don't
        // support F32 + Q8_0/F16 operands, and these are tiny scale factors)
        bool is_add_operand = sname.find("position_embedding") != std::string::npos ||
                              sname.find("class_embedding") != std::string::npos ||
                              sname.find(".ls1") != std::string::npos || sname.find(".ls2") != std::string::npos ||
                              sname.find(".beta") != std::string::npos || sname.find(".gamma") != std::string::npos;
        if (is_add_operand) {
            is_tiny_embd = true; // force copy-as-is
        }
        // SentenceTransformer Dense / Matryoshka projection heads (dense.0/dense.1):
        // the decoder_embed loader reads these as F32, so quantizing them makes the
        // output GGUF fail to load ("tensor read out of bounds"). Keep at original
        // precision. Verified on embeddinggemma-300m: only dense.* differed from the
        // working reference q8_0 (F32 there, Q8_0 here → unloadable), and re-quantizing
        // with this guard loads + embeds cleanly.
        if (sname.rfind("dense.", 0) == 0 || sname.find(".dense.") != std::string::npos) {
            is_tiny_embd = true; // force copy-as-is (keep F32)
        }
        // Audio mel filterbank / window: host-read constants for the log-mel front
        // end, NOT graph matmul weights. Quantizing them injects error into every
        // spectrogram (and a raw host read used to crash — now dequant-safe). Tiny
        // (~100 KB), so keep F32.
        if (sname.find("mel_filters") != std::string::npos || sname.find("mel_window") != std::string::npos) {
            is_tiny_embd = true; // force copy-as-is (keep F32)
        }
        // Source may be F32/F16 OR already quantized (we dequantize it to F32
        // first — see the read block below). Re-quantizing from q8_0 lets us skip
        // the huge f32 base for large models: q8_0 is ~lossless (cos ~0.9998) so
        // q8→f32→q4 ≈ f32→q4, and the q8_0 is a fraction of the f32 download.
        bool src_ok = (type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || ggml_is_quantized(type));
        bool quantize =
            (ggml_is_quantized(qtype) || qtype == GGML_TYPE_F16) && src_ok && (ggml_n_dims(t) >= 2) && !is_tiny_embd;
        const int64_t ncols = t->ne[0];
        ggml_type qtype_used = qtype;

        // Embedding tables: use Q8_0 for aggressive quants to preserve quality
        // while still compressing (embedding tables are huge, ~50% of model)
        if (quantize && is_embd && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16) {
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
        bool is_vision_weight = sname.rfind("v.", 0) == 0 || sname.rfind("c.", 0) == 0 || sname.rfind("qe.", 0) == 0 ||
                                sname.rfind("vis.", 0) == 0 || sname.rfind("proj.", 0) == 0;
        if (quantize && is_vision_weight && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(vision→Q8_0) ");
        }

        // InternVL2/H2OVL decoder attention: Q4_K destroys it. Measured on
        // h2ovl-mississippi-2b (2560d, 32H/8KV, head_dim 80) against a
        // Python-blueprint reference, same binary, same 7 real stages:
        //
        //   stage         f16        q8_0       q4_k
        //   llm_layer_0   0.999972   0.998033   0.594995
        //   llm_layer_2   0.999972   0.995498  -0.268615   <- anti-correlated
        //
        // That is not quantization drift, and it is not a shape problem: every
        // ne[0] here (2560, 6912) is 256-divisible, so Q4_K applies cleanly and
        // still wrecks the decoder. The error is already 14x larger than Q8_0 at
        // layer 0 (max_abs 0.0559 vs 0.0041) and compounds to 0.95 by layer 3.
        // Attention is the sensitive part -- it is what mixes the spliced image
        // tokens into the text stream -- so hold q/k/v/o at Q8_0 and let the FFN
        // take the requested type.
        bool is_internvl2_attn =
            is_internvl2 && sname.rfind("l.blk.", 0) == 0 &&
            (sname.find(".attn_q.") != std::string::npos || sname.find(".attn_k.") != std::string::npos ||
             sname.find(".attn_v.") != std::string::npos || sname.find(".attn_o.") != std::string::npos);
        if (quantize && is_internvl2_attn && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(internvl2-attn→Q8_0) ");
        }

        // InternVL2/H2OVL vision tower: keep F16, do not drop it to Q8_0.
        // Bisected the whole encoder against the blueprint reference (the
        // reference carried vis_layer_0..23 and vis_pixel_unshuffle all along;
        // nothing read them until now). At f16 every stage passes -- vis_layer_*
        // 1.000000..0.999902, vis_pixel_unshuffle 0.999691, vis_proj_output
        // 0.999974 -- so the port is exact and there is no code defect. With the
        // tower at Q8_0 the same stages decay monotonically to 0.90 by layer 11
        // and 0.64 by layer 12, because 24 residual blocks compound the
        // per-weight error. The projector then reads 0.998630 on Metal and
        // 0.912992 on CPU.
        //
        // The generic is_vision_weight rule below already refuses to go below
        // Q8_0; for this arch Q8_0 is itself the ceiling on vision parity, and
        // the tower is small next to the decoder, so buy the accuracy.
        // Scoped to the Q8_0 target on purpose. Q8_0 is the quality tier, where
        // +285 MB on a 2.2 GB file buys full vision parity. Q4_K is the size
        // tier: applying this there took internvl2-1b -- the edge/WASM model --
        // from 758 MB to 1135 MB, so the "quantize" step INFLATED it 1.5x,
        // defeating the only reason that artifact exists. Same shape of mistake
        // as the sub-Q8 refusal: a rule measured on one model, applied across a
        // family whose members have different goals.
        bool is_internvl2_vision = is_internvl2 && ftype == GGML_FTYPE_MOSTLY_Q8_0 &&
                                   (sname.rfind("v.", 0) == 0 || sname.rfind("proj.", 0) == 0);
        if (const char * off = getenv("CRISPEMBED_QUANTIZE_NO_VISION_F16")) {
            if (atoi(off)) is_internvl2_vision = false;
        }
        if (quantize && is_internvl2_vision && qtype != GGML_TYPE_F16 && qtype != GGML_TYPE_F32) {
            qtype_used = GGML_TYPE_F16;
            printf("(internvl2-vision→F16) ");
        }

        // MoE router / gating weights (DeepSeek-V2: "*.mlp_gate.weight", also
        // the generic "ffn_gate_inp"): these pick which experts run, so even
        // small quant error flips the top-k selection and corrupts the output.
        // Keep them at Q8_0 minimum (they are tiny: n_experts × hidden).
        bool is_moe_router =
            sname.find("mlp_gate.weight") != std::string::npos || sname.find("ffn_gate_inp") != std::string::npos;
        if (quantize && is_moe_router && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16) {
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
        // SMT OMR LM head is a squeezed 1×1 Conv1d named "decoder.out_layer.weight"
        // (not lm_head/output) — its logits drive a near-tie AR bekern decode that
        // Q4_K flips into repetition, so include it in the head guard.
        bool is_lm_head = sname.find("lm_head.weight") != std::string::npos || sname == "output.weight" ||
                          sname.find(".output.weight") != std::string::npos || sname == "decoder.out_layer.weight";
        if (quantize && is_lm_head && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 && qtype != GGML_TYPE_Q6_K &&
            qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(lm-head→Q8_0) ");
        }

        // PP-OCRv6 CTC fc2 is the direct per-timestep logit projection.  It is
        // the OCR equivalent of an LM head: Q4_K error can flip adjacent
        // character ties and then CTC collapse turns that into visible text
        // errors.  Q8_0 is still a substantial reduction and is the minimum
        // precision accepted for the shipped aggressive quantizations.
        const bool is_ppocr_logits = is_ppocrv6 && sname.find("head.fc2.weight") != std::string::npos;
        if (quantize && is_ppocr_logits && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(ppocrv6-ctc-head→Q8_0) ");
        }

        // SMT OMR ConvNext encoder (encoder.encoder.stages.*.pwconv{1,2}) is the
        // "reading" half — its output directly determines the transcription, and
        // Q4_K drops enc_output cos to ~0.95 and derails the decode. Keep the
        // encoder matmul weights at Q8_0 minimum (the ConvNext pointwise convs;
        // dwconv/downsample are already copied as-is by Guard 1 above).
        bool is_smt_encoder = sname.rfind("encoder.encoder.stages.", 0) == 0 &&
                              sname.find("pwconv") != std::string::npos && sname.find(".weight") != std::string::npos;
        if (quantize && is_smt_encoder && qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16 &&
            qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q5_K) {
            qtype_used = GGML_TYPE_Q8_0;
            printf("(smt-enc→Q8_0) ");
        }

        // LLM decoder weights (prefix "l.": attn_*, ffn_*, embed_tokens): keep at
        // F16 when --decoder-f16 is set. This is OPTIONAL and not needed for
        // correctness — the 0.5B Qwen2 decoder quantizes cleanly to q8_0/q4_k
        // (llm_layer_0 cos ≥ 0.99996 vs f32, identical OCR). The earlier "cos
        // 0.936 → garbage" that motivated this flag was a per-row diff-harness
        // artifact, not real sensitivity (see #25). Flag kept for diagnostics.
        if (quantize && g_decoder_f16 && sname.rfind("l.", 0) == 0 && qtype_used != GGML_TYPE_F16 &&
            qtype_used != GGML_TYPE_F32) {
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
            case GGML_TYPE_Q4_K:
                fallback = GGML_TYPE_Q4_0;
                break;
            case GGML_TYPE_Q5_K:
                fallback = GGML_TYPE_Q5_0;
                break;
            case GGML_TYPE_Q6_K:
                fallback = GGML_TYPE_Q8_0;
                break;
            // IQ4_XS uses 256-wide super-blocks; fall back to IQ4_NL (32-wide,
            // same 4-bit non-linear codebook) when the row isn't 256-aligned.
            case GGML_TYPE_IQ4_XS:
                fallback = GGML_TYPE_IQ4_NL;
                break;
            case GGML_TYPE_IQ4_NL:
                fallback = GGML_TYPE_Q4_0;
                break;
            default:
                break;
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

        // Source already at the target type (e.g. a q8_0 embedding kept at q8_0):
        // copy the raw bytes as-is, no dequant/requant roundtrip.
        if (quantize && type == qtype_used) {
            quantize = false;
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
                    fclose(fin);
                    fclose(fout);
                    return false;
                }
            } else if (type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> f16_data(nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    fclose(fin);
                    fclose(fout);
                    return false;
                }
                for (int64_t j = 0; j < nelements; j++) {
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
                }
            } else {
                // Quantized source: read the raw quantized bytes and dequantize to
                // F32 via the type's traits, then re-quantize to the target below.
                const size_t src_bytes = ggml_nbytes(t);
                std::vector<uint8_t> qbuf(src_bytes);
                if (fread(qbuf.data(), 1, src_bytes, fin) != src_bytes) {
                    fprintf(stderr, "failed to read quantized source data\n");
                    fclose(fin);
                    fclose(fout);
                    return false;
                }
                const ggml_type_traits * tr = ggml_get_type_traits(type);
                if (!tr || !tr->to_float) {
                    fprintf(stderr, "no dequantizer for source type %s\n", ggml_type_name(type));
                    fclose(fin);
                    fclose(fout);
                    return false;
                }
                tr->to_float(qbuf.data(), f32_data.data(), nelements);
            }

            const size_t max_q_size = ggml_row_size(qtype_used, t->ne[0]) * (nelements / t->ne[0]);
            q_data.resize(max_q_size);

            // Importance matrix (if loaded and shape-matched): steers k-quant/IQ
            // precision toward the columns the calibration data actually exercised.
            const float * imatrix = nullptr;
            std::vector<float> imatrix_sum; // storage for the `sum` mode; must outlive the call below
            if (!g_imatrix.empty()) {
                auto it = g_imatrix.find(sname);
                // BERT-family attn q/k/v statistics live under the runtime's
                // merged-QKV name (same input width). Usually there is no
                // direct entry at all and the alias is the only source; when
                // both exist, g_qkv_mode decides.
                const std::string alias = qkv_merged_alias(sname);
                auto ait = alias.empty() ? g_imatrix.end() : g_imatrix.find(alias);
                const bool both = it != g_imatrix.end() && ait != g_imatrix.end();
                if (it == g_imatrix.end()) {
                    it = ait;
                } else if (both && g_qkv_mode == qkv_imatrix_mode::merged) {
                    it = ait;
                }
                if (it != g_imatrix.end()) {
                    if ((int64_t)it->second.size() == t->ne[0]) {
                        imatrix = it->second.data();
                        if (both && g_qkv_mode == qkv_imatrix_mode::sum && ait->second.size() == it->second.size()) {
                            // Both vectors are per-input-column mean squared
                            // activations over the same column axis, so an
                            // element-wise sum keeps every column exercised by
                            // either provenance visible to the quantizer.
                            imatrix_sum = it->second;
                            for (size_t c = 0; c < imatrix_sum.size(); c++) imatrix_sum[c] += ait->second[c];
                            imatrix = imatrix_sum.data();
                        }
                        n_imatrix++;
                        printf("(imatrix) ");
                    } else {
                        printf("(imatrix shape %zu!=%lld, skipped) ", it->second.size(), (long long)t->ne[0]);
                    }
                }
            }

            size_t q_size = ggml_quantize_chunk(qtype_used, f32_data.data(), q_data.data(), 0, nelements / t->ne[0],
                                                t->ne[0], imatrix);

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
                fclose(fin);
                fclose(fout);
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

    printf("\n%d quantized, %d kept", n_quantized, n_kept);
    if (!g_imatrix.empty()) printf(", %d with imatrix", n_imatrix);
    printf("\n");
    printf("%.0f MB -> %.0f MB (%.1fx compression)\n", total_orig / 1048576.0, total_new / 1048576.0,
           (double)total_orig / (total_new > 0 ? (double)total_new : 1.0));

    return true;
}

int main(int argc, char ** argv) {
    // Collect positional args, allowing policy flags anywhere.
    std::vector<std::string> pos;
    std::string imatrix_path;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--decoder-f16")
            g_decoder_f16 = true;
        else if (a == "--ppocrv6-q8-head")
            g_ppocrv6_q8_head = true;
        else if (a == "--imatrix") {
            if (i + 1 >= argc) {
                fprintf(stderr, "--imatrix requires a file path\n");
                return 1;
            }
            imatrix_path = argv[++i];
        } else
            pos.push_back(a);
    }
    if (pos.size() != 3) {
        fprintf(
            stderr,
            "usage: %s <input.gguf> <output.gguf> <type> [--decoder-f16] [--ppocrv6-q8-head] [--imatrix <file>]\n\n",
            argv[0]);
        fprintf(stderr, "  --imatrix <f> use a CrispEmbed importance matrix (from a calibration run\n");
        fprintf(stderr, "                with CRISPEMBED_IMATRIX_OUT set) to improve k-quant/IQ accuracy\n");
        fprintf(stderr, "  --decoder-f16  keep LLM decoder weights (prefix 'l.') at F16\n");
        fprintf(stderr, "                 (optional; NOT required for correctness — small decoders\n");
        fprintf(stderr, "                  like GOT-OCR2's 0.5B quantize cleanly to q4_k/q8_0.\n");
        fprintf(stderr, "                  Retained for diagnostic/comparison use; see #25)\n\n");
        fprintf(stderr, "  --ppocrv6-q8-head  for PP-OCRv6 Q8_0, keep the CNN/backbone F32 and\n");
        fprintf(stderr, "                     quantize only the final SVTR/CTC head\n\n");
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

    if (!imatrix_path.empty()) {
        load_imatrix(imatrix_path); // non-fatal: falls back to unweighted if empty
        init_qkv_imatrix_mode();
    }

    if (!quantize_model(fname_inp, fname_out, it->second)) {
        fprintf(stderr, "Failed to quantize model\n");
        return 1;
    }

    return 0;
}
