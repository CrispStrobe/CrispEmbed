#include "lighton4b_bf16_dequant_weights.h"
#include "qwen35_decoder.h"
#include "qwen35_math.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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
// Independent integer RNE reference for finite quantized dequantization output.
static float round_ref(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    b += 0x7fff + ((b >> 16) & 1);
    b &= 0xffff0000;
    std::memcpy(&x, &b, 4);
    return x;
}
static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4);
}
static int run() {
    using namespace qwen35_ocr;
    const char * flag = "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS";
    for (auto * g : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_QUANT",
                      "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_QWEN35_STREAM_PROJECTION" })
        gate(g, "1");
    auto * ctx = ggml_init({ 65536, nullptr, true });
    auto * backend = ggml_backend_cpu_init();
    if (!ctx || !backend) return 1;
    auto * q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 257);
    auto * q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 257);
    auto * f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 257);
    auto * q6 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 256, 257);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 2;
    std::vector<float> source(256 * 257), raw(source.size()), rounded(source.size()), got(source.size()),
        hidden(256 * 3);
    for (size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 4001) - 2000) / 127;
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(int(i % 37) - 18) / 32;
    int checks = 0;
    for (auto * t : { q8, q4 }) {
        if (ggml_quantize_chunk(t->type, source.data(), t->data, 0, 257, 256, nullptr) != ggml_nbytes(t)) return 3;
        std::vector<uint8_t> storage(ggml_nbytes(t));
        ggml_backend_tensor_get(t, storage.data(), 0, storage.size());
        ggml_get_type_traits(t->type)->to_float(storage.data(), raw.data(), raw.size());
        rounded = raw;
        for (auto & x : rounded) x = round_ref(x);
        if (same(raw, rounded)) return 14;
        std::memcpy(f32->data, rounded.data(), rounded.size() * 4);
        ggml_set_name(t, "l.layers.0.mlp.gate_proj.weight");
        for (auto * value : { static_cast<const char *>(nullptr), "", "0" }) {
            gate(flag, value);
            if (!tensor_floats(t, got.data(), 2) || !same(got, raw)) return 4;
            ++checks;
        }
        gate(flag, "1");
        if (!tensor_floats(t, got.data(), 2) || !same(got, rounded)) return 5;
        std::vector<std::vector<float>> actual, expected;
        if (!project_cpu({ t }, hidden.data(), 256, 3, 2, actual) ||
            !project_cpu({ f32 }, hidden.data(), 256, 3, 2, expected) || !same(actual[0], expected[0]))
            return 6;
        ggml_set_name(t, "l.embed_tokens.weight");
        std::vector<float> embedded;
        if (!embed_cpu(t, { 0, 256, 1 }, embedded)) return 7;
        for (size_t k = 0; k < 3; ++k) {
            const int row = k == 0 ? 0 : k == 1 ? 256 : 1;
            if (std::memcmp(embedded.data() + k * 256, rounded.data() + row * 256, 256 * 4)) return 8;
        }
        std::vector<float> head, head_ref;
        if (!project_rows_cpu(t, hidden.data(), 256, 3, 2, 31, head) ||
            !project_rows_cpu(f32, hidden.data(), 256, 3, 2, 31, head_ref) || !same(head, head_ref))
            return 9;
        for (auto * name : { "v.blocks.0.weight", "l.layers.32.mlp.gate_proj.weight", "other.weight" }) {
            ggml_set_name(t, name);
            if (lighton4b_bf16_dequant_weights_enabled(t)) return 10;
        }
        ggml_set_name(t, "l.layers.0.mlp.gate_proj.weight");
        for (auto * g :
             { "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT" }) {
            gate(g, "0");
            if (lighton4b_bf16_dequant_weights_enabled(t)) return 11;
            gate(g, "1");
        }
        std::vector<uint8_t> after(storage.size());
        ggml_backend_tensor_get(t, after.data(), 0, after.size());
        if (after != storage) return 12;
        checks += 5;
    }
    ggml_set_name(f32, "l.layers.0.mlp.gate_proj.weight");
    ggml_set_name(q6, "l.layers.0.mlp.gate_proj.weight");
    if (lighton4b_bf16_dequant_weights_enabled(f32) || lighton4b_bf16_dequant_weights_enabled(q6)) return 13;
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("BF16_DEQUANT_WEIGHTS passed=1 checks=%d embedding_head=1 storage_unchanged=1 gate_off_byte_identity=1 "
                "model_payload=0\n",
                checks);
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
