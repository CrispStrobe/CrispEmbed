#include "lighton4b_native_context.h"
#include <cstdlib>
#include <iostream>
using namespace qwen35_ocr;
static void gate(const char * n, const char * v) {
#ifdef _WIN32
    _putenv_s(n, v ? v : "");
#else
    if (v)
        setenv(n, v, 1);
    else
        unsetenv(n);
#endif
}
int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--metadata") {
        auto * meta = core_gguf::open_metadata(argv[2]);
        bool ok = lighton4b_gdn_out_metadata_matches(meta);
        core_gguf::free_metadata(meta);
        std::cout << "GDN_OUT_METADATA passed=" << ok << " payload_bytes=0\n";
        return ok ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--closed-context") {
        auto context = lighton4b_native_context::open(argv[2]);
        if (context) return 2;
        std::cout << "GDN_OUT_CONTEXT_CLOSED passed=1 payload_bytes=0\n";
        return 0;
    }
    int checks = 0;
    auto check = [&](bool x) {
        ++checks;
        if (!x) {
            std::cerr << "GDN out control failed " << checks << '\n';
            std::exit(1);
        }
    };
    ggml_tensor q4{}, q8{};
    q4.type = GGML_TYPE_Q4_K;
    q8.type = GGML_TYPE_Q8_0;
    decoder_weights w;
    w.head = w.embedding = &q8;
    w.layers.resize(32);
    for (int i = 0; i < 32; ++i) {
        auto & l = w.layers[i];
        l.full_attention = i % 4 == 3;
        l.gate = l.up = l.down = &q4;
        if (l.full_attention)
            l.attention.q = l.attention.k = l.attention.v = l.attention.out = &q4;
        else {
            l.linear.qkv = l.linear.z = l.linear.a = l.linear.b = &q4;
            l.linear.out = &q8;
        }
    }
    auto kind = lighton4b_quant_recipe::gdn_out_q8;
    check(lighton4b_quant_recipe_matches(w, kind));
    check(!lighton4b_quant_recipe_matches(w, lighton4b_quant_recipe::q4_k));
    check(!lighton4b_quant_recipe_matches(w, lighton4b_quant_recipe::q8_0));
    check(!lighton4b_quant_recipe_matches(w, lighton4b_quant_recipe::attention_q8_ffn_q4));
    for (auto & l : w.layers) {
        auto * saved = l.down;
        l.down = &q8;
        check(!lighton4b_quant_recipe_matches(w, kind));
        l.down = saved;
        auto *& p = l.full_attention ? l.attention.v : l.linear.out;
        saved = p;
        p = l.full_attention ? &q8 : &q4;
        check(!lighton4b_quant_recipe_matches(w, kind));
        p = saved;
        if (!l.full_attention) {
            saved = l.linear.qkv;
            l.linear.qkv = &q8;
            check(!lighton4b_quant_recipe_matches(w, kind));
            l.linear.qkv = saved;
        }
    }
    lighton4b_quant_recipe chosen;
    gate("CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4", "0");
    for (const char * value : { static_cast<const char *>(nullptr), "", "0", "false", "no", "off" }) {
        gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8", value);
        check(!lighton4b_select_recipe(w, "lighton4b-q4k-gdn-out-q8-v1", chosen));
    }
    gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8", "1");
    check(lighton4b_select_recipe(w, "lighton4b-q4k-gdn-out-q8-v1", chosen) && chosen == kind);
    check(!lighton4b_select_recipe(w, "", chosen));
    check(!lighton4b_select_recipe(w, "Q4_K_M", chosen));
    gate("CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4", "1");
    check(!lighton4b_select_recipe(w, "lighton4b-q4k-gdn-out-q8-v1", chosen));
    gate("CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4", "0");
    gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8", "0");
    for (auto & l : w.layers)
        if (!l.full_attention) l.linear.out = &q4;
    check(lighton4b_select_recipe(w, "", chosen) && chosen == lighton4b_quant_recipe::q4_k);
    auto * meta = gguf_init_empty();
    check(!lighton4b_gdn_out_metadata_matches(meta));
    const std::pair<const char *, const char *> strings[] = {
        { "qwen35_ocr.mixed_quant_recipe", "lighton4b-q4k-gdn-out-q8-v1" },
        { "qwen35_ocr.mixed_quant_base_sha256", "f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91" },
        { "qwen35_ocr.mixed_quant_donor_sha256", "5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1" },
        { "qwen35_ocr.mixed_quant_manifest_sha256",
          "895d1f509006c3f8ed599555e85033b7cec1064ca33699ef4ca1ac151f60c663" }
    };
    for (auto s : strings) gguf_set_val_str(meta, s.first, s.second);
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_selected_count", 24);
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_payload_delta", 125829120);
    check(lighton4b_gdn_out_metadata_matches(meta));
    gguf_set_val_u64(meta, "qwen35_ocr.mixed_quant_selected_count", 24);
    check(!lighton4b_gdn_out_metadata_matches(meta));
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_selected_count", 24);
    for (auto s : strings) {
        gguf_set_val_str(meta, s.first, "wrong");
        check(!lighton4b_gdn_out_metadata_matches(meta));
        gguf_set_val_str(meta, s.first, s.second);
    }
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_selected_count", 23);
    check(!lighton4b_gdn_out_metadata_matches(meta));
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_selected_count", 24);
    gguf_set_val_u32(meta, "qwen35_ocr.mixed_quant_payload_delta", 125829119);
    check(!lighton4b_gdn_out_metadata_matches(meta));
    gguf_free(meta);
    std::cout << "GDN_OUT_RECIPE passed=1 checks=" << checks
              << " payload_bytes=0 default_off=1 recipe=lighton4b-q4k-gdn-out-q8-v1\n";
}
