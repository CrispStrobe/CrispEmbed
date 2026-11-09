#pragma once
// Tool-only strict domain for one authenticated external dense imatrix adapter.
#include "ggml.h"
#include "gguf.h"
#include "sha256.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <filesystem>
#include <string>
#include <vector>
namespace lighton4b_imatrix {
inline constexpr const char * profile = "lighton4b-text-inputs-hf-grouped-v1";
inline constexpr const char * source_sha = "10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7";
inline const std::map<std::string, std::pair<int, int>> & expected() {
    static const auto result = [] {
        std::map<std::string, std::pair<int, int>> out;
        for (int i = 0; i < 32; ++i) {
            const auto p = "l.layers." + std::to_string(i) + ".";
            out[p + "mlp.gate_proj.weight"] = { 2560, 9216 };
            out[p + "mlp.up_proj.weight"] = { 2560, 9216 };
            out[p + "mlp.down_proj.weight"] = { 9216, 2560 };
            if (i % 4 == 3) {
                for (auto * name : { "q", "k", "v" })
                    out[p + "self_attn." + name + "_proj.weight"] = { 2560, std::string(name) == "q" ? 8192 : 1024 };
                out[p + "self_attn.o_proj.weight"] = { 4096, 2560 };
            } else {
                out[p + "linear_attn.in_proj_qkv.weight"] = { 2560, 8192 };
                out[p + "linear_attn.in_proj_z.weight"] = { 2560, 4096 };
                out[p + "linear_attn.in_proj_a.weight"] = { 2560, 32 };
                out[p + "linear_attn.in_proj_b.weight"] = { 2560, 32 };
                out[p + "linear_attn.out_proj.weight"] = { 4096, 2560 };
            }
        }
        return out;
    }();
    return result;
}
inline bool string_is(gguf_context * g, const char * key, const char * value) {
    const auto k = gguf_find_key(g, key);
    return k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING && std::string(gguf_get_val_str(g, k)) == value;
}
inline bool u32_is(gguf_context * g, const char * key, uint32_t value) {
    const auto k = gguf_find_key(g, key);
    return k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_UINT32 && gguf_get_val_u32(g, k) == value;
}
inline bool load(const std::string & path, std::map<std::string, std::vector<float>> & means) {
    means.clear();
    // Deterministic adapter v1 is bound to the approved source and inverse permutation.
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (ec || bytes != 3621184) return false;
    if (core_sha256::file(path) != "105aba299aff48c63b43323bf6ec4f02d20f2fbb726b1ece0728e9e7fbb24663") return false;
    ggml_context * ctx = nullptr;
    gguf_init_params p{ false, &ctx };
    auto * g = gguf_init_from_file(path.c_str(), p);
    if (!g || !ctx) {
        if (g) gguf_free(g);
        if (ctx) ggml_free(ctx);
        return false;
    }
    bool ok = string_is(g, "general.architecture", "crispembed-imatrix") &&
              string_is(g, "crispembed.imatrix.profile", profile) &&
              string_is(g, "crispembed.imatrix.source_sha256", source_sha) && u32_is(g, "imatrix.version", 1) &&
              u32_is(g, "crispembed.imatrix.expected_entries", 248) && gguf_get_n_tensors(g) == 248;
    const auto & domain = expected();
    for (const auto & row : domain) {
        if (!ok) break;
        auto * t = ggml_get_tensor(ctx, row.first.c_str());
        const auto k = gguf_find_key(g, ("count." + row.first).c_str());
        ok = t && t->type == GGML_TYPE_F32 && t->ne[0] == row.second.first && ggml_nelements(t) == t->ne[0] && k >= 0 &&
             gguf_get_kv_type(g, k) == GGUF_TYPE_UINT64;
        if (!ok) break;
        const auto count = gguf_get_val_u64(g, k);
        ok = count == 66048;
        if (!ok) break;
        auto & v = means[row.first];
        v.resize(t->ne[0]);
        const auto * raw = static_cast<const float *>(t->data);
        bool energy = false;
        for (size_t i = 0; i < v.size(); ++i) {
            ok = std::isfinite(raw[i]) && raw[i] >= 0;
            if (!ok) break;
            v[i] = float(double(raw[i]) / double(count));
            const float upstream = raw[i] / float(count);
            ok = std::isfinite(v[i]) && std::memcmp(&upstream, &v[i], sizeof(float)) == 0;
            energy |= raw[i] > 0;
            if (!ok) break;
        }
        ok = ok && energy;
    }
    gguf_free(g);
    ggml_free(ctx);
    if (!ok || means.size() != 248) {
        means.clear();
        return false;
    }
    return true;
}
inline bool target(gguf_context * g, ggml_context * ctx) {
    if (!g || !ctx || gguf_get_n_tensors(g) != 723 || !string_is(g, "general.architecture", "qwen35_ocr") ||
        !string_is(g, "qwen35_ocr.profile", "lightonocr3-4b"))
        return false;
    int vision = 0, other = 0, head = 0;
    for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        const std::string name = ggml_get_name(t);
        if (expected().count(name)) continue;
        if (name == "l.embed_tokens.weight") {
            if (t->ne[0] != 2560 || t->ne[1] != 248320 || t->ne[2] != 1 || t->ne[3] != 1) return false;
            ++head;
        } else {
            if (t->type != GGML_TYPE_F32) return false;
            if (name.rfind("v.", 0) == 0)
                ++vision;
            else if (name.rfind("l.", 0) == 0)
                ++other;
            else
                return false;
        }
    }
    if (vision != 297 || other != 177 || head != 1) return false;
    for (const auto & row : expected()) {
        auto * t = ggml_get_tensor(ctx, row.first.c_str());
        if (!t || t->ne[0] != row.second.first || t->ne[1] != row.second.second || t->ne[2] != 1 || t->ne[3] != 1 ||
            (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16))
            return false;
    }
    return true;
}
inline std::string audit(const std::map<std::string, std::vector<float>> & means) {
    std::string out =
        "{\"schema\":\"lighton4b-imatrix-consumer-check/"
        "v1\",\"quantization_executed\":false,\"model_payload_bytes\":0,\"entries\":248,\"normalized_f32_sha256\":{";
    bool first = true;
    for (const auto & row : means) {
        core_sha256::sha256_state h;
        core_sha256::sha256_update(h, reinterpret_cast<const uint8_t *>(row.second.data()),
                                   row.second.size() * sizeof(float));
        if (!first) out += ",";
        first = false;
        out += "\"" + row.first + "\":\"" + core_sha256::sha256_finish(h) + "\"";
    }
    return out + "}}";
}
} // namespace lighton4b_imatrix
