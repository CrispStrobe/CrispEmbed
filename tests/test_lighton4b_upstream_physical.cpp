#include "lighton4b_native_context.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
using namespace qwen35_ocr;
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static bool bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (!a.size() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "physical control%d failed\n", checks);
            std::exit(1);
        }
    };
    for (auto * name :
         { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS", "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER",
           "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
           "CRISPEMBED_LIGHTON4B_Q6_K" })
        gate(name, "1");
    for (auto * name : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                         "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4",
                         "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8", "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6",
                         "CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD" })
        gate(name, "0");
    auto * ctx = ggml_init({ 427 * ggml_tensor_overhead() + 8192, nullptr, true });
    check(ctx != nullptr);
    core_gguf::tensor_map tensors;
    std::ifstream file(argv[1]);
    std::string name;
    int type;
    int64_t ne[4];
    while (file >> name >> type >> ne[0] >> ne[1] >> ne[2] >> ne[3]) {
        auto * tensor = ggml_new_tensor(ctx, ggml_type(type), 4, ne);
        check(tensor && !tensor->data && tensors.emplace(name, tensor).second);
    }
    check(file.eof() && tensors.size() == 426);
    lighton4b_text_owner owner;
    check(map_lighton4b_text(tensors, owner));
    check(owner.decoder.head == owner.decoder.embedding && owner.decoder.head->type == GGML_TYPE_Q6_K);
    lighton4b_quant_recipe recipe;
    for (auto * value : { static_cast<const char *>(nullptr), "", "0", "false", "off", "no" }) {
        gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", value);
        check(!lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
        check(lighton4b_upstream_physical_mode_valid(false));
    }
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1");
    check(lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe) &&
          recipe == lighton4b_quant_recipe::upstream_physical_q4km);
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", "1");
    for (const char * mode : { "all", "quant", "rms", "layers", "head", "off" }) {
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", mode);
        check(lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
    }
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", "unknown");
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", "0");
    check(lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", nullptr);
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, recipe));
    check(!lighton4b_select_recipe(owner.decoder, "", recipe));
    for (auto & pair : tensors)
        if (pair.second->type != GGML_TYPE_F32) {
            const auto saved = pair.second->type;
            pair.second->type = saved == GGML_TYPE_Q6_K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
            check(!lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
            pair.second->type = saved;
        }
    for (auto * flag : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                         "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4",
                         "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8", "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6",
                         "CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD" }) {
        gate(flag, "1");
        check(!lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
        gate(flag, "0");
    }
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "0");
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_upstream_physical_recipe, recipe));
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "1");
    // Complete geometry-wide lane proof for all admitted chunk lengths, not an image.
    for (int count = 1; count <= 16; ++count) {
        std::vector<float> input(size_t(count) * 4096), tiled;
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(i);
        check(lighton4b_upstream_v_to_tiled(input, count, tiled));
        for (int t = 0; t < count; ++t)
            for (int h = 0; h < 32; ++h)
                for (int c = 0; c < 128; ++c)
                    check(tiled[size_t(t) * 4096 + size_t(16 * (h % 2) + h / 2) * 128 + c] ==
                          input[size_t(t) * 4096 + size_t(h) * 128 + c]);
        auto aliased = input;
        check(lighton4b_upstream_v_to_tiled(aliased, count, aliased));
        check(bits(aliased, tiled));
    }
    std::vector<float> malformed(4095), output;
    check(!lighton4b_upstream_v_to_tiled(malformed, 1, output));
    std::vector<float> correct(4096);
    check(!lighton4b_upstream_v_to_tiled(correct, 0, output));
    check(!lighton4b_upstream_v_to_tiled(correct, 17, output));
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "0");
    check(!lighton4b_upstream_v_to_tiled(correct, 1, output));
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1");
    check(lighton4b_upstream_gamma(1.25f, true) == 1.25f);
    check(lighton4b_upstream_gamma(0.25f, false) == 1.25f);
    // Actual grouped core: identical −exp(0)=−1 and direct physical−1 must have
    // identical values/state; interpreting the physical number as A_log must differ.
    std::vector<float> qkv(2 * 8192, 0.02f), z(2 * 4096, 1.0f), a(64, 0.0f), b(64, 0.0f), conv(8192 * 4, 0.0f),
        alog(32, 0.0f), direct(32, -1.0f), dt(32, 0.0f), norm(128, 1.0f);
    for (int i = 0; i < 8192; ++i) conv[size_t(i) * 4 + 3] = 1.0f;
    linear_inputs in{ 1,           2,          32,          128,      4,        1,
                      1e-6f,       qkv.data(), z.data(),    a.data(), b.data(), conv.data(),
                      alog.data(), dt.data(),  norm.data(), nullptr,  nullptr };
    linear_outputs legacy, physical, wrong, old_repeat;
    check(linear_grouped_core_cpu(in, 16, legacy));
    in.a_log = direct.data();
    in.a_is_pretransformed = true;
    check(linear_grouped_core_cpu(in, 16, physical));
    check(bits(legacy.values, physical.values) && bits(legacy.state, physical.state) &&
          bits(legacy.conv_history, physical.conv_history));
    in.a_is_pretransformed = false;
    check(linear_grouped_core_cpu(in, 16, wrong));
    check(!bits(wrong.state, physical.state));
    in.a_log = alog.data();
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "0");
    check(linear_grouped_core_cpu(in, 16, old_repeat));
    check(bits(old_repeat.values, legacy.values) && bits(old_repeat.state, legacy.state));
    in.a_is_pretransformed = true;
    in.a_log = direct.data();
    check(!linear_grouped_core_cpu(in, 16, wrong));
    gate("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1");
    check(!linear_core_cpu(in, wrong));
    in.width = 64;
    check(!linear_grouped_core_cpu(in, 16, wrong));
    in.width = 128;
    direct[0] = 0.0f;
    check(!linear_grouped_core_cpu(in, 16, wrong));
    direct[0] = -1.0f;
    auto * metadata = gguf_init_empty();
    check(metadata != nullptr);
    check(!lighton4b_upstream_physical_metadata_matches(metadata));
    const std::pair<const char *, const char *> strings[]{
        { "qwen35_ocr.mixed_quant_recipe", "lighton4b-upstream-q4km-physical-v1" },
        { "qwen35_ocr.upstream_physical.repository", "richardyoung/LightOnOCR-3-4B-GGUF" },
        { "qwen35_ocr.upstream_physical.revision", "8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01" },
        { "qwen35_ocr.upstream_physical.sha256", "b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26" },
        { "qwen35_ocr.upstream_physical.protected_source_sha256",
          "f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91" },
        { "qwen35_ocr.upstream_physical.mapping_sha256",
          "cf3af1c7f7893a06cf1660358e3ec7084dd6352730e8bcfd384e749e4f11edb1" },
        { "qwen35_ocr.upstream_physical.source_code", "6c01da6278ab6c69ed67d9124806b50d2b299ba7" },
        { "qwen35_ocr.upstream_physical.semantics",
          "gamma-direct,a-direct,hf-state,tiled-out-input,upstream-control-types" },
    };
    for (const auto & s : strings) gguf_set_val_str(metadata, s.first, s.second);
    const std::pair<const char *, uint32_t> integers[]{
        { "general.file_type", 15 },
        { "qwen35_ocr.format_version", 6 },
        { "qwen35_ocr.upstream_physical.version", 1 },
        { "qwen35_ocr.upstream_physical.row_permuted_tensors", 96 },
        { "qwen35_ocr.upstream_physical.retained_out_tensors", 24 },
        { "qwen35_ocr.upstream_physical.protected_vision_tensors", 297 },
    };
    for (const auto & n : integers) gguf_set_val_u32(metadata, n.first, n.second);
    gguf_set_val_bool(metadata, "qwen35_ocr.zero_centered_text_norm", false);
    gguf_set_val_bool(metadata, "qwen35_ocr.tie_word_embeddings", true);
    const int32_t eog[]{ 248044, 248046 };
    gguf_set_arr_data(metadata, "qwen35_ocr.upstream_physical.eog_ids", GGUF_TYPE_INT32, eog, 2);
    check(lighton4b_upstream_physical_metadata_matches(metadata));
    for (const auto & s : strings) {
        gguf_set_val_str(metadata, s.first, "wrong");
        check(!lighton4b_upstream_physical_metadata_matches(metadata));
        gguf_set_val_u32(metadata, s.first, 1);
        check(!lighton4b_upstream_physical_metadata_matches(metadata));
        gguf_set_val_str(metadata, s.first, s.second);
    }
    for (const auto & n : integers) {
        gguf_set_val_u32(metadata, n.first, n.second + 1);
        check(!lighton4b_upstream_physical_metadata_matches(metadata));
        gguf_set_val_str(metadata, n.first, "6");
        check(!lighton4b_upstream_physical_metadata_matches(metadata));
        gguf_set_val_u32(metadata, n.first, n.second);
    }
    gguf_set_val_bool(metadata, "qwen35_ocr.zero_centered_text_norm", true);
    check(!lighton4b_upstream_physical_metadata_matches(metadata));
    gguf_set_val_bool(metadata, "qwen35_ocr.zero_centered_text_norm", false);
    const int32_t short_eog[]{ 248046 };
    gguf_set_arr_data(metadata, "qwen35_ocr.upstream_physical.eog_ids", GGUF_TYPE_INT32, short_eog, 1);
    check(!lighton4b_upstream_physical_metadata_matches(metadata));
    gguf_set_arr_data(metadata, "qwen35_ocr.upstream_physical.eog_ids", GGUF_TYPE_INT32, eog, 2);
    check(lighton4b_upstream_physical_metadata_matches(metadata));
    gguf_free(metadata);
    ggml_free(ctx);
    std::printf("PHYSICAL controls=%d passed=1 model_payload_bytes=0 learned_model_inference=0\n", checks);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
