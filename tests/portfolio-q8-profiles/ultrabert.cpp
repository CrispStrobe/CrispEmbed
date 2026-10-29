// Reuses the pinned metadata-only fixture; no model payloads or inference.
#include "ultrabert_profile.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <memory>
namespace {
struct Fixture {
    ggml_context * tensors = ggml_init({ 256 * 1024, nullptr, true });
    gguf_context * meta = gguf_init_empty();
    explicit Fixture(int tensor_variant = 0) {
        auto s = [&](const char * key, const char * value) { gguf_set_val_str(meta, key, value); };
        auto u = [&](const char * key, uint32_t value) { gguf_set_val_u32(meta, key, value); };
        s("general.architecture", "ultrabert");
        s("ultrabert.source.revision", "367d5343b5a0164d57b810cedf5de4b8983d5984");
        s("ultrabert.source.weight_sha256", "da8be29564b69e4f90fc8f2683e1f369c3d3575e75f7fe798643e28de02e2b9c");
        u("general.file_type", tensor_variant >= 4 ? GGML_FTYPE_MOSTLY_Q4_K : GGML_FTYPE_ALL_F32);
        u("ultrabert.context_length", 32768);
        u("ultrabert.embedding_length", 768);
        u("ultrabert.block_count", 24);
        u("ultrabert.feed_forward_length", 1024);
        u("ultrabert.attention.head_count", 6);
        u("ultrabert.attention.head_count_kv", 6);
        u("ultrabert.attention.key_length", 128);
        u("ultrabert.attention.value_length", 128);
        gguf_set_val_bool(meta, "ultrabert.attention.causal", false);
        gguf_set_val_f32(meta, "ultrabert.attention.layer_norm_rms_epsilon", 1e-6f);
        s("ultrabert.rope.layout", "split-half");
        s("ultrabert.pooling", "cls");
        gguf_set_val_bool(meta, "ultrabert.pooling.include_prompt", true);
        gguf_set_val_bool(meta, "ultrabert.pooling.normalize_default", false);
        s("ultrabert.prompt.query", ultrabert_profile::query_prefix());
        s("ultrabert.prompt.document", "");
        s("ultrabert.prompt.default", "");
        std::vector<int32_t> windows(24);
        std::vector<float> thetas(24);
        for (int i = 0; i < 24; ++i) {
            windows[i] = i % 3 == 2 ? -1 : 128;
            thetas[i] = i % 3 == 2 ? 2560000.f : 10000.f;
        }
        gguf_set_arr_data(meta, "ultrabert.attention.window_radius", GGUF_TYPE_INT32, windows.data(), 24);
        gguf_set_arr_data(meta, "ultrabert.rope.freq_base_per_layer", GGUF_TYPE_FLOAT32, thetas.data(), 24);
        s("tokenizer.ggml.model", "gpt2");
        s("tokenizer.ggml.pre", "gpt2");
        s("ultrabert.tokenizer.single_template", "<|endoftext|> $A");
        s("ultrabert.tokenizer.pair_template", "<|endoftext|> $A <|sep|> $B");
        gguf_set_val_bool(meta, "tokenizer.ggml.add_bos_token", true);
        gguf_set_val_bool(meta, "tokenizer.ggml.add_eos_token", false);
        for (const auto * key :
             { "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id" })
            u(key, 0);
        u("tokenizer.ggml.separator_token_id", 15);
        u("tokenizer.ggml.mask_token_id", 16);
        for (const auto & item : ultrabert_profile::schema()) {
            if (tensor_variant == 1 && item.first == "output_norm.weight") continue;
            auto type = tensor_variant == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            const bool critical =
                item.first == "token_embd.weight" || item.first.find("norm.weight") != std::string::npos;
            if (tensor_variant >= 4 && !critical) type = GGML_TYPE_Q4_K;
            if (tensor_variant == 5 && item.first == "output_norm.weight") type = GGML_TYPE_Q4_K;
            if (tensor_variant == 6 && item.first == "token_embd.weight") type = GGML_TYPE_Q8_0;
            auto * t = ggml_new_tensor_2d(tensors, type, item.second.first + (tensor_variant == 3 ? 1 : 0),
                                          item.second.second);
            ggml_set_name(t, item.first.c_str());
            gguf_add_tensor(meta, t);
        }
    }
    ~Fixture() {
        gguf_free(meta);
        ggml_free(tensors);
    }
};
void check(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void q8(Fixture & f) {
    gguf_set_val_u32(f.meta, "general.file_type", 7);
    for (const auto & item : ultrabert_profile::schema()) {
        const bool critical = item.first == "token_embd.weight" || item.first.find("norm.weight") != std::string::npos;
        if (!critical) gguf_set_tensor_type(f.meta, item.first.c_str(), GGML_TYPE_Q8_0);
    }
}
bool accepted(Fixture & f, bool quantized) {
    try {
        ultrabert_profile::validate(f.meta, quantized);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}
int run() {
    Fixture f32;
    check(accepted(f32, false), "F32 profile changed");
    Fixture valid;
    q8(valid);
    check(accepted(valid, true), "actual Q8 profile rejected");
    check(!accepted(valid, false), "Q8 passed legacy F32 admission");
    gguf_set_val_i32(valid.meta, "general.file_type", 7);
    check(!accepted(valid, true), "signed Q8 header accepted");
    gguf_set_val_f32(valid.meta, "general.file_type", 7.f);
    check(!accepted(valid, true), "floating Q8 header accepted");
    for (auto header : { 12u, 14u }) {
        gguf_set_val_u32(valid.meta, "general.file_type", header);
        check(!accepted(valid, true), "Q4 header accepted Q8-only storage");
        Fixture old(4);
        gguf_set_val_u32(old.meta, "general.file_type", header);
        check(accepted(old, true), "legacy Q4 profile changed");
    }
    gguf_set_val_u32(valid.meta, "general.file_type", 7);
    gguf_set_tensor_type(valid.meta, "blk.0.attn_q.weight", GGML_TYPE_Q4_K);
    check(!accepted(valid, true), "Q8 header accepted Q4 storage");
    Fixture mislabeled;
    gguf_set_val_u32(mislabeled.meta, "general.file_type", 7);
    check(!accepted(mislabeled, true), "floating model mislabeled Q8");
    Fixture critical;
    q8(critical);
    gguf_set_tensor_type(critical.meta, "output_norm.weight", GGML_TYPE_Q8_0);
    check(!accepted(critical, true), "critical Q8 norm accepted");
    std::puts("PASS: UltraBERT Q8/Q4/F32 metadata and critical storage controls; no inference");
    return 0;
}
} // namespace
int main() {
    try {
        core_util::clean_exit(run());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
