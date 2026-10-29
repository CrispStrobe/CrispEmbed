#include "ultrabert_profile.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <functional>
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
int run() {
    Fixture valid;
    ultrabert_profile::validate(valid.meta);
    int controls = 0;
    auto reject = [&](const std::function<void(gguf_context *)> & change, int tensor_variant = 0) {
        Fixture f(tensor_variant);
        change(f.meta);
        bool failed = false;
        try {
            ultrabert_profile::validate(f.meta);
        } catch (const std::exception &) {
            failed = true;
        }
        ultrabert_profile::require(failed, "invalid profile accepted");
        ++controls;
    };
    reject([](auto * g) { gguf_set_val_str(g, "general.architecture", "bert"); });
    reject([](auto * g) { gguf_set_val_str(g, "ultrabert.source.revision", "other"); });
    reject([](auto * g) { gguf_set_val_u32(g, "general.file_type", 1); });
    reject([](auto * g) { gguf_set_val_bool(g, "ultrabert.attention.causal", true); });
    reject([](auto * g) { gguf_set_val_str(g, "ultrabert.pooling", "mean"); });
    reject([](auto * g) { gguf_set_val_str(g, "ultrabert.prompt.query", "query: "); });
    reject([](auto * g) { gguf_set_val_str(g, "tokenizer.ggml.pre", "qwen2"); });
    reject([](auto * g) { gguf_set_val_bool(g, "tokenizer.ggml.add_eos_token", true); });
    reject([](auto * g) { gguf_set_val_f32(g, "ultrabert.attention.layer_norm_rms_epsilon", 1e-5f); });
    reject([](auto * g) { gguf_remove_key(g, "ultrabert.attention.window_radius"); });
    reject([](auto * g) {
        std::vector<int32_t> windows(24, 128); // missing global layers
        gguf_set_arr_data(g, "ultrabert.attention.window_radius", GGUF_TYPE_INT32, windows.data(), 24);
    });
    reject([](auto *) {}, 1);
    reject([](auto *) {}, 2);
    reject([](auto *) {}, 3);
    Fixture q4(4);
    ultrabert_profile::validate(q4.meta, true);
    bool old_rejected = false;
    try {
        ultrabert_profile::validate(q4.meta);
    } catch (const std::exception &) {
        old_rejected = true;
    }
    ultrabert_profile::require(old_rejected, "legacy F32 validator accepted Q4_K");
    for (int variant : { 5, 6 }) {
        Fixture bad(variant);
        bool failed = false;
        try {
            ultrabert_profile::validate(bad.meta, true);
        } catch (const std::exception &) {
            failed = true;
        }
        ultrabert_profile::require(failed, "quantized critical tensor accepted");
        ++controls;
    }
    Fixture mislabeled;
    gguf_set_val_u32(mislabeled.meta, "general.file_type", GGML_FTYPE_MOSTLY_Q4_K);
    bool no_q4_rejected = false;
    try {
        ultrabert_profile::validate(mislabeled.meta, true);
    } catch (const std::exception &) {
        no_q4_rejected = true;
    }
    ultrabert_profile::require(no_q4_rejected, "mislabeled Q4_K without Q4_K tensors accepted");
    std::vector<int32_t> ids;
    BPETokenizer unused;
    ultrabert_profile::require(!ultrabert_profile::tokenize(unused, "<|mask|>", ids), "reserved text accepted");
    ultrabert_profile::require(!ultrabert_profile::tokenize(unused, std::string(65537, 'x'), ids), "oversize accepted");
    printf("PASS: valid CPU/F32 profile, %d malformed metadata/schema controls and two text preflights\n", controls);
    return 0;
}
} // namespace
static int crispembed_test_main() {
    try {
        return run();
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
