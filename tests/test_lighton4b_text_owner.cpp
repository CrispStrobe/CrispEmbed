#include "qwen35_lighton4b_text.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER", value, 1);
    else
        unsetenv("CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER");
#endif
}
bool empty(const qwen35_ocr::lighton4b_text_owner & owner) {
    return owner.decoder.layers.empty() && !owner.decoder.embedding && !owner.decoder.norm && !owner.decoder.head &&
           !owner.linear_key_heads && !owner.linear_value_heads && !owner.linear_head_dim;
}
struct row {
    std::string name;
    int64_t ne[4];
    bool matrix;
};
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    std::vector<row> rows;
    row item;
    while (input >> item.name >> item.ne[0] >> item.ne[1] >> item.ne[2] >> item.ne[3] >> item.matrix)
        rows.push_back(item);
    if (rows.size() != 426 || !input.eof()) return 2;
    ggml_init_params params{ 426 * ggml_tensor_overhead() + 4096, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (!context) return 2;
    core_gguf::tensor_map tensors;
    for (const auto & row : rows) {
        auto * t = ggml_new_tensor(context, row.matrix ? GGML_TYPE_F16 : GGML_TYPE_F32, 4, row.ne);
        if (!t || t->data || !tensors.emplace(row.name, t).second) return 2;
    }
    size_t controls = 0;
    bool passed = true;
    auto check = [&](bool condition, const char * label) {
        ++controls;
        if (!condition) {
            std::fprintf(stderr, "FAIL %s control=%zu\n", label, controls);
            passed = false;
        }
    };
    qwen35_ocr::lighton4b_text_owner owner;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        gate(value);
        owner.linear_key_heads = 99;
        check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "gate rejects and resets");
    }
    gate("1");
    check(qwen35_ocr::map_lighton4b_text(tensors, owner), "original header F16 schema accepted");
    check(owner.decoder.layers.size() == 32 && owner.decoder.head == owner.decoder.embedding &&
              owner.decoder.embedding == tensors.at("l.embed_tokens.weight") &&
              owner.decoder.norm == tensors.at("l.norm.weight"),
          "tied head and owner identities");
    check(owner.attention.query_heads == 16 && owner.attention.kv_heads == 4 && owner.attention.head_dim == 256 &&
              owner.attention.rotary_dim == 64 && owner.linear_key_heads == 16 && owner.linear_value_heads == 32 &&
              owner.linear_head_dim == 128,
          "original full and grouped geometry");
    size_t owned = 2, full = 0;
    for (size_t i = 0; i < owner.decoder.layers.size(); ++i) {
        const auto & layer = owner.decoder.layers[i];
        const std::string prefix = "l.layers." + std::to_string(i) + ".";
        check(layer.full_attention == (i % 4 == 3), "exact attention schedule");
        check(layer.input_norm == tensors.at(prefix + "input_layernorm.weight") &&
                  layer.post_norm == tensors.at(prefix + "post_attention_layernorm.weight") &&
                  layer.gate == tensors.at(prefix + "mlp.gate_proj.weight") &&
                  layer.up == tensors.at(prefix + "mlp.up_proj.weight") &&
                  layer.down == tensors.at(prefix + "mlp.down_proj.weight"),
              "all FFN owner identities");
        owned += 5;
        if (layer.full_attention) {
            ++full;
            const std::vector<ggml_tensor *> actual{
                layer.attention.q,   layer.attention.k,      layer.attention.v,
                layer.attention.out, layer.attention.q_norm, layer.attention.k_norm
            };
            const char * names[]{ "q_proj.weight", "k_proj.weight", "v_proj.weight",
                                  "o_proj.weight", "q_norm.weight", "k_norm.weight" };
            for (size_t j = 0; j < actual.size(); ++j)
                check(actual[j] == tensors.at(prefix + "self_attn." + names[j]), "all full-attention owner identities");
            owned += actual.size();
        } else {
            const std::vector<ggml_tensor *> actual{ layer.linear.qkv,   layer.linear.z,       layer.linear.a,
                                                     layer.linear.b,     layer.linear.out,     layer.linear.conv,
                                                     layer.linear.a_log, layer.linear.dt_bias, layer.linear.norm };
            const char * names[]{ "in_proj_qkv.weight",
                                  "in_proj_z.weight",
                                  "in_proj_a.weight",
                                  "in_proj_b.weight",
                                  "out_proj.weight",
                                  "conv1d.weight",
                                  "A_log",
                                  "dt_bias",
                                  "norm.weight" };
            for (size_t j = 0; j < actual.size(); ++j)
                check(actual[j] == tensors.at(prefix + "linear_attn." + names[j]),
                      "all grouped-attention owner identities");
            owned += actual.size();
        }
    }
    check(owned == 426 && full == 8, "complete unique owner census");
    // Every source entry is individually mandatory and shape/type checked.
    for (const auto & row : rows) {
        auto * t = tensors.at(row.name);
        tensors.erase(row.name);
        check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "missing tensor");
        tensors.emplace(row.name, t);
        for (int axis = 0; axis < 4; ++axis) {
            ++t->ne[axis];
            check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "axis drift");
            --t->ne[axis];
        }
        const auto type = t->type;
        t->type = GGML_TYPE_Q8_0;
        check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "quantized owner rejected");
        t->type = type;
        const auto stride = t->nb[0];
        ++t->nb[0];
        check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "noncontiguous owner rejected");
        t->nb[0] = stride;
        if (!row.matrix) {
            t->type = GGML_TYPE_F16;
            check(!qwen35_ocr::map_lighton4b_text(tensors, owner) && empty(owner), "F16 recurrent/norm rejected");
            t->type = type;
        }
    }
    auto extra = tensors;
    extra.emplace("lm_head.weight", tensors.at("l.embed_tokens.weight"));
    check(!qwen35_ocr::map_lighton4b_text(extra, owner) && empty(owner), "stored tied alias forbidden");
    extra = tensors;
    extra.erase("l.norm.weight");
    extra.emplace("unknown", tensors.at("l.norm.weight"));
    check(!qwen35_ocr::map_lighton4b_text(extra, owner) && empty(owner), "same-count unknown forbidden");
    extra = tensors;
    extra["l.layers.0.post_attention_layernorm.weight"] = extra.at("l.layers.0.input_layernorm.weight");
    check(!qwen35_ocr::map_lighton4b_text(extra, owner) && empty(owner), "same-shape pointer alias forbidden");
    for (const auto & row : rows)
        if (row.matrix) {
            auto * t = tensors.at(row.name);
            t->type = GGML_TYPE_F32;
            t->nb[0] = 4;
            t->nb[1] = 4 * t->ne[0];
            t->nb[2] = t->nb[1] * t->ne[1];
            t->nb[3] = t->nb[2] * t->ne[2];
        }
    check(qwen35_ocr::map_lighton4b_text(tensors, owner), "all-F32 text schema accepted");
    ggml_free(context);
    gate(nullptr);
    std::printf("LIGHTON4B_TEXT_OWNER controls=%zu passed=%d payload_bytes=0 model_inference=0\n", controls,
                passed ? 1 : 0);
    return passed ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
