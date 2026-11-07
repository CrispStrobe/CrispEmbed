// Diagnostic only: actual native input -> duplicate unchanged block -> strict main-trace closure.
#include "lighton4b_attention_block.h"
#include "ggml.h"
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
namespace fs = std::filesystem;
static void require(bool ok, const std::string & s) { if (!ok) throw std::runtime_error(s); }
static void read(const fs::path & p, void * data, size_t n) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f.read(static_cast<char *>(data), n)) && f.peek() == EOF, "read: " + p.string());
}
static void write(const fs::path & p, const std::vector<float> & x) {
    for (float v : x) require(std::isfinite(v), "nonfinite: " + p.string());
    std::ofstream f(p, std::ios::binary);
    require(bool(f.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(float))), "write: " + p.string());
}
int main(int argc, char ** argv) try {
    require(argc == 4, "usage: native WEIGHTS CAPTURE_CASE OUTPUT");
    const fs::path weights = argv[1], capture = fs::path(argv[2]) / "upstream_features", out = argv[3];
    fs::create_directories(out);
    for (const char * name : {"CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_LIGHTON4B_Q6_K"}) setenv(name, "1", 1);
    for (const char * name : {"CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_QWEN35_PRECISE_RMS"}) setenv(name, "0", 1);
    // All borrowed tensors together are <65MiB. No model/head or whole-model F32 cache.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({80ull << 20, nullptr, false}), ggml_free);
    require(bool(ctx), "context");
    auto tensor = [&](const char * name, ggml_type type, int width, int rows = 1) {
        auto * t = ggml_new_tensor_2d(ctx.get(), type, width, rows);
        read(weights / (std::string(name) + ".bin"), t->data, ggml_nbytes(t)); return t;
    };
    decoder_layer_weights w{}; w.full_attention = true; w.upstream_physical = true; w.attention.upstream_physical = true;
    w.input_norm = tensor("blk.3.attn_norm.weight", GGML_TYPE_F32, 2560);
    w.post_norm = tensor("blk.3.post_attention_norm.weight", GGML_TYPE_F32, 2560);
    w.gate = tensor("blk.3.ffn_gate.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.up = tensor("blk.3.ffn_up.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.down = tensor("blk.3.ffn_down.weight", GGML_TYPE_Q6_K, 9216, 2560);
    w.attention.q = tensor("blk.3.attn_q.weight", GGML_TYPE_Q4_K, 2560, 8192);
    w.attention.k = tensor("blk.3.attn_k.weight", GGML_TYPE_Q4_K, 2560, 1024);
    w.attention.v = tensor("blk.3.attn_v.weight", GGML_TYPE_Q6_K, 2560, 1024);
    w.attention.out = tensor("blk.3.attn_output.weight", GGML_TYPE_Q4_K, 4096, 2560);
    w.attention.q_norm = tensor("blk.3.attn_q_norm.weight", GGML_TYPE_F32, 256);
    w.attention.k_norm = tensor("blk.3.attn_k_norm.weight", GGML_TYPE_F32, 256);
    std::vector<float> input(154 * 2560), expected(input.size()), allpos(3 * 154), positions(3 * 16);
    read(capture / "prefill.layer.2.f32", input.data(), input.size() * 4);
    read(capture / "prefill.layer.3.f32", expected.data(), expected.size() * 4);
    read(capture / "prefill.positions.f32", allpos.data(), allpos.size() * 4);
    for (int a = 0; a < 3; ++a) for (int t = 0; t < 16; ++t) positions[a * 16 + t] = allpos[a * 154 + t];
    lighton4b_attention_block_outputs block;
    require(lighton4b_attention_block_cpu(w, input.data(), positions.data(), 16, 4, nullptr, block), "native block rejected");
    require(block.hidden.size() == 16 * 2560 && !std::memcmp(block.hidden.data(), expected.data(), block.hidden.size() * 4), "actual native layer3 hidden byte closure FAILED");
    // Reproduce the unchanged native rotation, then demand scalar context closure in consumer.
    auto q = block.q_norm;
    for (int t = 0; t < 16; ++t) for (int h = 0; h < 16; ++h) for (int i = 0; i < 32; ++i) {
        int axis = 0; if (i % 3 == 1 && i < 33) axis = 1; if (i % 3 == 2 && i < 30) axis = 2;
        const float frequency = 1.f / std::pow(10000000.f, float(2 * i) / 64);
        const float angle = frequency * positions[axis * 16 + t];
        const float c = std::cos(angle), s = std::sin(angle);
        float * x = q.data() + (t * 16 + h) * 256; const float a = x[i], b = x[i + 32];
        x[i] = a * c - b * s; x[i + 32] = b * c + a * s;
    }
    std::vector<float> k(16 * 4 * 256), gate(16 * 16 * 256);
    for (int t = 0; t < 16; ++t) for (int h = 0; h < 4; ++h)
        std::copy_n(block.cache.keys.data() + (h * 16 + t) * 256, 256, k.data() + (t * 4 + h) * 256);
    for (int t = 0; t < 16; ++t) for (int h = 0; h < 16; ++h)
        std::copy_n(block.q.data() + (t * 16 + h) * 512 + 256, 256, gate.data() + (t * 16 + h) * 256);
    write(out / "q.f32", q); write(out / "k.f32", k); write(out / "v.f32", block.v);
    write(out / "gate.f32", gate); write(out / "pooled.f32", block.pooled);
    write(out / "hidden.f32", block.hidden); write(out / "positions.f32", positions);
    std::cout << "ACTUAL_LAYER3_FIRST16_BYTE_CLOSURE PASS\n";
    return 0;
} catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
