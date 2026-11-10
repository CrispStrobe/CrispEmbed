#pragma once
// Distinct HF-scalar profile: only 24 recurrent output matrices/activation
// coordinates change. Never enables upstream scalar or native-op semantics.
#include "core/env_gate.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace qwen35_ocr {
inline constexpr const char * lighton4b_hf_tiled_out_recipe = "lighton4b-weighted-q4km-hf-tiled-out-v1";
inline bool lighton4b_hf_tiled_out_enabled() {
#if defined(__EMSCRIPTEN__)
    return false;
#else
    return core_env::on("CRISPEMBED_LIGHTON4B_HF_TILED_OUT");
#endif
}
inline bool lighton4b_hf_tiled_out_mode_valid(bool tiled, bool physical) {
    return !tiled || (lighton4b_hf_tiled_out_enabled() && !physical && core_env::on("CRISPEMBED_LIGHTON4B_Q6_K") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_BF16_LINEAR") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8") &&
                      !core_env::on("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6"));
}
inline bool lighton4b_hf_tiled_out_metadata(gguf_context * g) {
    if (!g || !lighton4b_hf_tiled_out_mode_valid(true, false)) return false;
    auto text = [&](const char * k, const char * v) {
        const int i = gguf_find_key(g, k);
        return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_STRING && !std::strcmp(gguf_get_val_str(g, i), v);
    };
    auto number = [&](const char * k, uint32_t v) {
        const int i = gguf_find_key(g, k);
        return i >= 0 && gguf_get_kv_type(g, i) == GGUF_TYPE_UINT32 && gguf_get_val_u32(g, i) == v;
    };
    return text("qwen35_ocr.mixed_quant_recipe", lighton4b_hf_tiled_out_recipe) &&
           text("qwen35_ocr.q6_names_sha256", "0ebc0b6104238b46de946f940f61cdc90e04a0eb187553026714fc1270cdb44c") &&
           text("qwen35_ocr.quant_source_f16_sha256",
                "757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606") &&
           text("qwen35_ocr.quant_coordinate_profile", "hf-scalars-tiled-gdn-out-v1") &&
           text("crispembed.quantization.imatrix_source_sha256",
                "10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7") &&
           text("crispembed.quantization.imatrix_profile", "lighton4b-text-inputs-hf-with-tiled-gdn-out-v1") &&
           text("qwen35_ocr.hf_tiled_out.base_candidate_sha256",
                "5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29") &&
           text("qwen35_ocr.hf_tiled_out.proof_protocol_sha256",
                "c21003e25e95509ecf71e386017b813d410490a696737eb31d45b3c08caa310f") &&
           text("qwen35_ocr.hf_tiled_out.coordinate_map_sha256",
                "d85c5067c02735fb9f5701d64a1f04ff43d45d20d57612f0a015a57b7ce72b5c") &&
           text("qwen35_ocr.hf_tiled_out.paired_receipt_sha256",
                "e87e6b28936e1022b6207cb942ba5e1d468b309bf731c24bf44f136dbd20b215") &&
           number("qwen35_ocr.hf_tiled_out.tensor_count", 24) &&
           number("qwen35_ocr.hf_tiled_out.protected_tensor_count", 699) &&
           number("crispembed.quantization.imatrix_expected_entries", 248) &&
           number("crispembed.quantization.imatrix_applied_entries", 248) &&
           number("crispembed.quantization.imatrix_hf_grouped_entries", 224) &&
           number("crispembed.quantization.imatrix_upstream_tiled_entries", 24) &&
           number("qwen35_ocr.q6_tensor_count", 33) && number("general.file_type", 15);
}
inline bool lighton4b_hf_v_to_tiled(const std::vector<float> & input, int tokens, std::vector<float> & output) {
    if (!lighton4b_hf_tiled_out_mode_valid(true, false) || tokens < 1 || tokens > 16 ||
        input.size() != size_t(tokens) * 4096 ||
        !std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
        return false;
    std::vector<float> candidate(input.size());
    for (int t = 0; t < tokens; ++t)
        for (int h = 0; h < 32; ++h)
            for (int c = 0; c < 128; ++c)
                candidate[size_t(t) * 4096 + size_t(16 * (h % 2) + h / 2) * 128 + c] =
                    input[size_t(t) * 4096 + size_t(h) * 128 + c];
    output = std::move(candidate);
    return true;
}
} // namespace qwen35_ocr
