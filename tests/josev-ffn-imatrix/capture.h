// TEST OVERLAY ONLY. A const view of the actual normalized CPU operand; no math/replay injection.
#pragma once
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
namespace josev_ffn_imatrix_test {
static std::string directory;
static int calls = 0, rows_seen = 0;
static uint32_t epsilon_bits = 0;
static bool failed = false;
static bool shape(const ggml_tensor * t, int cols, int rows) {
    return t && t->type == GGML_TYPE_F32 && t->ne[0] == cols && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1;
}
static bool capture(const float * input, int columns, int rows, const ggml_tensor * gate, const ggml_tensor * up,
                    const ggml_tensor * norm, float epsilon, int batch, int past) {
    if (directory.empty()) return true;
    auto reject = []() {
        failed = true;
        return false;
    };
    if (calls++ != 0 || !input || columns != 2048 || rows < 1 || rows > 512 || batch != 1 || past != 0 ||
        !std::isfinite(epsilon) || epsilon <= 0 || !shape(gate, 2048, 6144) || !shape(up, 2048, 6144) ||
        !shape(norm, 2048, 1) || std::strcmp(gate->name, "l.layers.23.mlp.gate_proj.weight") != 0 ||
        std::strcmp(up->name, "l.layers.23.mlp.up_proj.weight") != 0 ||
        std::strcmp(norm->name, "l.layers.23.post_attention_layernorm.weight") != 0)
        return reject();
    for (size_t i = 0; i < size_t(rows) * 2048; ++i)
        if (!std::isfinite(input[i])) return reject();
    const auto temporary = std::filesystem::path(directory) / "pre_ffn23.f32.partial";
    std::ofstream f(temporary, std::ios::binary);
    f.write(reinterpret_cast<const char *>(input), size_t(rows) * 2048 * 4);
    f.close();
    if (!f) return reject();
    std::filesystem::rename(temporary, std::filesystem::path(directory) / "pre_ffn23.f32");
    rows_seen = rows;
    std::memcpy(&epsilon_bits, &epsilon, 4);
    return true;
}
} // namespace josev_ffn_imatrix_test
extern "C" int josev_ffn_imatrix_begin(const char * path) {
    using namespace josev_ffn_imatrix_test;
    directory.clear();
    calls = 0;
    rows_seen = 0;
    failed = false;
    if (!path || !*path || !std::filesystem::is_directory(path)) return 0;
    directory = path;
    return 1;
}
extern "C" int josev_ffn_imatrix_finish() {
    using namespace josev_ffn_imatrix_test;
    if (directory.empty() || failed || calls != 1 || rows_seen < 1) return 0;
    const auto temporary = std::filesystem::path(directory) / "pre_ffn23.json.partial";
    std::ofstream f(temporary);
    f << "{\"schema\":\"crispembed-josev-native-f32-pre-ffn23-v1\",\"calls\":1,\"layer\":23,\"batch\":1,\"past\":0,"
         "\"rows\":"
      << rows_seen << ",\"columns\":2048,\"epsilon_bits\":" << epsilon_bits
      << ",\"gate\":\"l.layers.23.mlp.gate_proj.weight\",\"up\":\"l.layers.23.mlp.up_proj.weight\","
         "\"post_norm\":\"l.layers.23.post_attention_layernorm.weight\",\"stored_types\":[0,0,0],"
         "\"actual_native_f32_operand\":true,\"python_reference_activation\":false}\n";
    f.close();
    if (!f) return 0;
    std::filesystem::rename(temporary, std::filesystem::path(directory) / "pre_ffn23.json");
    directory.clear();
    return 1;
}
