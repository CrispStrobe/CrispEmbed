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

static bool quantize_model(const std::string & fname_inp, const std::string & fname_out, ggml_ftype ftype) {
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
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
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx_in_ggml, name);
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
        bool is_embd = sname.find("embd") != std::string::npos ||
                        sname.find("embed") != std::string::npos ||
                        sname.find("token_types") != std::string::npos;
        // Skip tiny embedding tables (token_types has only 2 rows)
        // — quantizing these breaks Ollama's binary ops (f32 + q8_0)
        bool is_tiny_embd = (t->ne[1] <= 4) &&
                            (sname.find("token_types") != std::string::npos ||
                             sname.find("type_embd") != std::string::npos);
        bool quantize = ggml_is_quantized(qtype) &&
                        (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) &&
                        (ggml_n_dims(t) == 2) &&
                        (sname.find("weight") != std::string::npos ||
                         sname.find(".w") != std::string::npos ||
                         sname.find("_w") != std::string::npos) &&
                        (sname.find("norm") == std::string::npos) &&
                        !is_tiny_embd;
        const int64_t ncols = t->ne[0];
        ggml_type qtype_used = qtype;

        // Embedding tables: use Q8_0 for aggressive quants to preserve quality
        // while still compressing (embedding tables are huge, ~50% of model)
        if (quantize && is_embd &&
            qtype != GGML_TYPE_Q8_0 && qtype != GGML_TYPE_F16) {
            qtype_used = GGML_TYPE_Q8_0;
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
    if (argc != 4) {
        fprintf(stderr, "usage: %s <input.gguf> <output.gguf> <type>\n\n", argv[0]);
        fprintf(stderr, "Supported types:\n");
        for (auto & [name, _] : FTYPE_MAP) {
            fprintf(stderr, "  %s\n", name.c_str());
        }
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const char * type_str = argv[3];

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
