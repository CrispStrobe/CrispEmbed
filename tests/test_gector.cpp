#include "gector.h"
#include "core/clean_exit.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
int checks = 0, failed = 0;
void check(bool condition, const char * text) {
    ++checks;
    if (!condition) { ++failed; std::fprintf(stderr, "FAIL: %s\n", text); }
}
void gate(const char * value) {
#if defined(_WIN32)
    _putenv_s("CRISPEMBED_GECTOR", value ? value : "");
#else
    if (value) setenv("CRISPEMBED_GECTOR", value, 1);
    else unsetenv("CRISPEMBED_GECTOR");
#endif
}
void metadata(gguf_context * g) {
    gguf_set_val_str(g, "general.architecture", "gector");
    gguf_set_val_u32(g, "gector.format_version", 1);
    gguf_set_val_u32(g, "gector.block_count", 12);
    gguf_set_val_u32(g, "gector.embedding_length", 768);
    gguf_set_val_u32(g, "gector.feed_forward_length", 3072);
    gguf_set_val_u32(g, "gector.attention.head_count", 12);
    gguf_set_val_u32(g, "gector.context_length", 512);
    gguf_set_val_u32(g, "gector.vocab_size", 31102);
    gguf_set_val_u32(g, "gector.type_vocab_size", 2);
    gguf_set_val_u32(g, "gector.label_count", 5000);
    gguf_set_val_u32(g, "gector.detect_count", 2);
    gguf_set_val_f32(g, "gector.layer_norm_epsilon", 1e-12f);
    gguf_set_val_f32(g, "gector.attention.qk_scale", 0.3535533845424652f);
    gguf_set_val_str(g, "gector.activation", "gelu_erf");
    gguf_set_val_str(g, "gector.mask_policy", "unpadded_all_ones");
    gguf_set_val_str(g, "gector.source.repository", "proofbird/gector-bert-de");
    gguf_set_val_str(g, "gector.source.revision", "8e3b50beef7d775cedb7b876009e88bcf2e8d6af");
    gguf_set_val_str(g, "gector.source.sha256", "6d7b43b4bd95c7a00c02c8babc7569d28558df894afe4874d96c202e6ab9a464");
}
void model_admission_tests() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string path = "gector-contract-" + std::to_string(nonce) + ".gguf";
    auto * gc = ggml_init({ 1024 * 1024, nullptr, false });
    auto * t = ggml_new_tensor_1d(gc, GGML_TYPE_F32, 1);
    ggml_set_name(t, "tiny-invalid-model");
    *static_cast<float *>(t->data) = 0;
    auto * g = gguf_init_empty();
    metadata(g);
    gguf_add_tensor(g, t);
    char error[256] = {};
    auto rejected = [&](const char * wanted) {
        check(gguf_write_to_file(g, path.c_str(), false), "write tiny invalid model");
        auto * c = crispembed_gector_create(path.c_str(), 1, error, sizeof(error));
        check(c == nullptr, "invalid model is rejected before inference");
        check(std::strstr(error, wanted) != nullptr, wanted);
        crispembed_gector_free(c);
    };
    rejected("201");
    gguf_set_val_str(g, "general.architecture", "bert");
    rejected("architecture");
    gguf_set_val_str(g, "general.architecture", "gector");
    gguf_set_val_u64(g, "gector.block_count", (uint64_t(1) << 32) + 12);
    rejected("block_count");
    gguf_set_val_u32(g, "gector.block_count", 12);
    gguf_set_val_f32(g, "gector.layer_norm_epsilon", 1e-5f);
    rejected("layer_norm_epsilon");
    gguf_set_val_f32(g, "gector.layer_norm_epsilon", 1e-12f);
    gguf_set_val_str(g, "gector.source.sha256", "substituted-base-model");
    rejected("source.sha256");
    gguf_free(g);
    ggml_free(gc);
    check(std::remove(path.c_str()) == 0, "remove task-owned temporary fixture");
}
int run() {
    char error[256] = {};
    int64_t ids[] = { 0, 101, 31101 }, mask[] = { 1, 1, 1 };
    check(crispembed_gector_validate_inputs(ids, 3, mask, 3, 1, error, sizeof(error)) == 0,
          "valid boundary token IDs");
    check(error[0] == '\0', "success clears diagnostic");
    for (size_t batch : { size_t(0), size_t(2), std::numeric_limits<size_t>::max() })
        check(crispembed_gector_validate_inputs(ids, 3, mask, 3, batch, error, sizeof(error)) == 1,
              "invalid batch is rejected");
    for (size_t count : { size_t(0), size_t(513), std::numeric_limits<size_t>::max() })
        check(crispembed_gector_validate_inputs(ids, count, mask, count, 1, error, sizeof(error)) == 1,
              "sequence bounds checked before pointer reads");
    check(crispembed_gector_validate_inputs(ids, 3, mask, 2, 1, error, sizeof(error)) == 1,
          "mask length mismatch");
    check(crispembed_gector_validate_inputs(nullptr, 3, mask, 3, 1, error, sizeof(error)) == 1, "null IDs");
    check(crispembed_gector_validate_inputs(ids, 3, nullptr, 3, 1, error, sizeof(error)) == 1, "null mask");
    for (int64_t bad : { int64_t(-1), int64_t(31102), std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min() }) {
        ids[1] = bad;
        check(crispembed_gector_validate_inputs(ids, 3, mask, 3, 1, error, sizeof(error)) == 1,
              "invalid int64 ID cannot narrow to a valid int32");
    }
    ids[1] = 101;
    for (int64_t bad : { int64_t(0), int64_t(-1), int64_t(2), std::numeric_limits<int64_t>::max() }) {
        mask[1] = bad;
        check(crispembed_gector_validate_inputs(ids, 3, mask, 3, 1, error, sizeof(error)) == 1,
              "only all-ones masks accepted");
    }
    mask[1] = 1;
    std::vector<int64_t> max_ids(512, 1), max_mask(512, 1);
    check(crispembed_gector_validate_inputs(max_ids.data(), 512, max_mask.data(), 512, 1, nullptr, 0) == 0,
          "512 tokens accepted without an error buffer");
    char short_error[2] = { 'X', 'Y' };
    check(crispembed_gector_validate_inputs(ids, 0, mask, 0, 1, short_error, sizeof(short_error)) == 1 && short_error[1] == '\0',
          "truncated error is nul terminated");
    check(crispembed_gector_threads(nullptr) == 0, "null thread query");
    crispembed_gector_free(nullptr);
    float sentinel = 123;
    check(crispembed_gector_forward(nullptr, ids, 3, mask, 3, 1, &sentinel, 1, &sentinel, 1,
                                    nullptr, 0, error, sizeof(error)) == 1 && sentinel == 123,
          "null context fails without writing output");
    for (const char * value : { "", "0", "false", "OFF", "No" }) {
        gate(value);
        check(!crispembed_gector_create("never-open-this-file.gguf", 1, error, sizeof(error)) && std::strstr(error, "disabled"),
              "off spellings reject before model access");
    }
    gate(nullptr);
    check(!crispembed_gector_create("never-open-this-file.gguf", 1, error, sizeof(error)) && std::strstr(error, "disabled"),
          "unset gate is disabled");
    gate("1");
    check(!crispembed_gector_create(nullptr, 1, error, sizeof(error)), "null model path rejected");
    for (int threads : { -1, 0, 65, std::numeric_limits<int>::max() })
        check(!crispembed_gector_create("never-open-this-file.gguf", threads, error, sizeof(error)) && std::strstr(error, "thread"),
              "thread count checked before model access");
    model_admission_tests();
    gate(nullptr);
    std::printf("GECToR contract: %d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
} // namespace
int main() { core_util::clean_exit(run()); }
