// Standalone audit against pinned upstream llama.cpp; not linked into CrispEmbed.
// Build with upstream include/ and libllama. Input archive is written by
// tests/lfm2_runtime_comparison.py. Output files contain raw F32 token features.
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

template <class T> static bool read(std::ifstream & in, T & value) {
    return bool(in.read(reinterpret_cast<char *>(&value), sizeof(value)));
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr, "Usage: %s MODEL.gguf INPUT.bin OUTPUT_PREFIX [plain-f32-kv]\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    const bool control = argc == 5 && std::string(argv[4]) == "plain-f32-kv";
    if (argc == 5 && !control) return 1;
    if (control) mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 2048;
    cp.n_seq_max = 1;
    if (control) cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 1;
    }
    std::ifstream input(argv[2], std::ios::binary);
    uint32_t cases = 0;
    if (!read(input, cases)) return 1;
    const auto * vocab = llama_model_get_vocab(model);
    const int dim = llama_model_n_embd(model);
    for (uint32_t case_id = 0; case_id < cases; ++case_id) {
        uint32_t length = 0, n = 0;
        if (!read(input, length) || length > 1000000) return 1;
        std::string text(length, '\0');
        if (!input.read(text.data(), length) || !read(input, n) || n > cp.n_batch) return 1;
        std::vector<llama_token> ids(n), actual(n + 16);
        if (!input.read(reinterpret_cast<char *>(ids.data()), n * sizeof(llama_token))) return 1;
        const int count = llama_tokenize(vocab, text.data(), length, actual.data(), actual.size(), true, true);
        actual.resize(count > 0 ? count : 0);
        if (ids != actual) {
            std::fprintf(stderr, "Tokenizer mismatch case %u: %u reference, %d upstream\n", case_id, n, count);
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        auto batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (uint32_t i = 0; i < n; ++i) {
            batch.token[i] = ids[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;
        }
        if (llama_decode(ctx, batch) != 0) return 1;
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "-%03u.f32", case_id);
        std::ofstream output(std::string(argv[3]) + suffix, std::ios::binary);
        for (uint32_t i = 0; i < n; ++i) {
            const float * row = llama_get_embeddings_ith(ctx, i);
            if (!row || !output.write(reinterpret_cast<const char *>(row), dim * sizeof(float))) return 1;
        }
        llama_batch_free(batch);
        std::printf("PASS upstream case %u: %u exact IDs, %d raw features/token\n", case_id, n, dim);
        std::fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
