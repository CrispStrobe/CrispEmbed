#include "lighton4b_native_context.h"
#include "core/lighton4b_imatrix_q4km.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
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
static int run(int argc, char ** argv) {
    if (argc != 2) return 2;
    int checks = 0;
    auto check = [&](bool ok) {
        if (!ok) {
            std::fprintf(stderr, "imatrix Q4KM control%d failed\n", checks);
            std::exit(1);
        }
        ++checks;
    };
    for (auto * name : { "CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
                         "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_QWEN35_QUANT" })
        gate(name, "1");
    for (auto * name : { "CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4", "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8",
                         "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6" })
        gate(name, "0");
    check(lighton4b_imatrix_q4km::manifest_valid());
    auto * ctx = ggml_init({ 426 * ggml_tensor_overhead() + 8192, nullptr, true });
    check(ctx != nullptr);
    core_gguf::tensor_map tensors;
    std::ifstream input(argv[1]);
    std::string name;
    int kind;
    int64_t ne[4];
    int q6 = 0, q4 = 0, f32 = 0;
    while (input >> name >> kind >> ne[0] >> ne[1] >> ne[2] >> ne[3]) {
        auto * t = ggml_new_tensor(ctx, ggml_type(kind), 4, ne);
        check(t && !t->data && tensors.emplace(name, t).second);
        check(t->type == lighton4b_imatrix_q4km::type(name));
        if (kind == 14)
            ++q6;
        else if (kind == 12)
            ++q4;
        else if (kind == 0)
            ++f32;
    }
    check(input.eof() && tensors.size() == 426 && q6 == 33 && q4 == 216 && f32 == 177);
    lighton4b_text_owner owner;
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "0");
    check(!map_lighton4b_text(tensors, owner));
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "1");
    check(map_lighton4b_text(tensors, owner));
    check(owner.decoder.head == owner.decoder.embedding && owner.decoder.head->type == GGML_TYPE_Q6_K);
    check(lighton4b_q4km_types_match(owner.decoder, GGML_TYPE_Q6_K));
    lighton4b_quant_recipe chosen;
    gate("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD", "0");
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, chosen));
    gate("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD", "1");
    check(lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, chosen) &&
          chosen == lighton4b_quant_recipe::imatrix_q4km_q6head);
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_q6_recipe, chosen));
    check(!lighton4b_select_recipe(owner.decoder, "", chosen));
    for (auto & row : tensors)
        if (row.second->type != GGML_TYPE_F32) {
            const auto saved = row.second->type;
            row.second->type = saved == GGML_TYPE_Q6_K ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
            check(!lighton4b_quant_recipe_matches(owner.decoder, lighton4b_quant_recipe::imatrix_q4km_q6head));
            row.second->type = saved;
        }
    auto * alias = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 2560, 248320);
    owner.decoder.head = alias;
    check(!lighton4b_quant_recipe_matches(owner.decoder, lighton4b_quant_recipe::imatrix_q4km_q6head));
    owner.decoder.head = owner.decoder.embedding;
    auto * meta = gguf_init_empty();
    const std::pair<const char *, const char *> strings[] = {
        { "qwen35_ocr.mixed_quant_recipe", lighton4b_imatrix_q4km_recipe },
        { "qwen35_ocr.q6_names_sha256", lighton4b_imatrix_q4km::names_sha },
        { "qwen35_ocr.quant_source_f16_sha256", lighton4b_imatrix_q4km::source_sha },
        { "qwen35_ocr.quant_coordinate_profile", "hf-grouped-v1" },
        { "crispembed.quantization.imatrix_source_sha256", lighton4b_imatrix::source_sha },
        { "crispembed.quantization.imatrix_profile", lighton4b_imatrix::profile }
    };
    for (const auto & s : strings) gguf_set_val_str(meta, s.first, s.second);
    const std::pair<const char *, uint32_t> numbers[] = { { "general.file_type", 15 },
                                                          { "qwen35_ocr.q6_tensor_count", 33 },
                                                          { "crispembed.quantization.imatrix_expected_entries", 248 },
                                                          { "crispembed.quantization.imatrix_applied_entries", 248 } };
    for (const auto & n : numbers) gguf_set_val_u32(meta, n.first, n.second);
    check(lighton4b_imatrix_q4km_metadata(meta));
    for (const auto & s : strings) {
        gguf_set_val_str(meta, s.first, "wrong");
        check(!lighton4b_imatrix_q4km_metadata(meta));
        gguf_set_val_u32(meta, s.first, 1);
        check(!lighton4b_imatrix_q4km_metadata(meta));
        gguf_set_val_str(meta, s.first, s.second);
    }
    for (const auto & n : numbers) {
        gguf_set_val_u32(meta, n.first, n.second + 1);
        check(!lighton4b_imatrix_q4km_metadata(meta));
        gguf_set_val_str(meta, n.first, "248");
        check(!lighton4b_imatrix_q4km_metadata(meta));
        gguf_set_val_u32(meta, n.first, n.second);
    }
    for (auto * value : { static_cast<const char *>(nullptr), "", "0", "false" }) {
        gate("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD", value);
        check(!lighton4b_imatrix_q4km_metadata(meta));
        check(!lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, chosen));
    }
    gate("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD", "1");
    for (auto * flag : { "CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4", "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8",
                         "CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6" }) {
        gate(flag, "1");
        check(!lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, chosen));
        gate(flag, "0");
    }
    gate("CRISPEMBED_LIGHTON4B_Q6_K", "0");
    check(!lighton4b_imatrix_q4km_metadata(meta));
    check(!lighton4b_select_recipe(owner.decoder, lighton4b_imatrix_q4km_recipe, chosen));
    gguf_free(meta);
    ggml_free(ctx);
    std::printf("checks=%d pass=1 model_payload_bytes=0\n", checks);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
