// Diagnostic only: actual native input -> duplicate unchanged block -> strict main-trace closure.
#include "core/clean_exit.h"
#include "lighton4b_linear_block.h"
#include "lighton4b_source_prefill_chunks.h"
#include "qwen35_weight_cache.h"
#include <map>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
namespace fs = std::filesystem;
static void require(bool ok, const std::string & s) {
    if (!ok) throw std::runtime_error(s);
}
static void read(const fs::path & p, void * data, size_t n) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f.read(static_cast<char *>(data), n)) && f.peek() == EOF, "read: " + p.string());
}
static void write(const fs::path & p, const std::vector<float> & x) {
    for (float v : x) require(std::isfinite(v), "nonfinite: " + p.string());
    std::ofstream f(p, std::ios::binary);
    require(bool(f.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(float))), "write: " + p.string());
}
static int run(int argc, char ** argv) try {
    require(argc == 5, "component WEIGHTS ACTUAL OUTPUT {off,on}");
    const bool aligned = std::string(argv[4]) == "on";
    require(aligned || std::string(argv[4]) == "off", "closed aligned mode");
    setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL_ALIGNED", aligned ? "1" : "0", 1);
    const fs::path weights = argv[1], capture = argv[2], out = argv[3];
    fs::create_directories(out);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 128ull << 20, nullptr, false }), ggml_free);
    require(bool(ctx), "bounded component context");
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    std::vector<buffer_ptr> storage;
    auto tensor = [&](const char * name, ggml_type type, int a, int b = 1, int c = 1) {
        auto * t = ggml_new_tensor_3d(ctx.get(), type, a, b, c);
        read(weights / (std::string(name) + ".bin"), t->data, ggml_nbytes(t));
        auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(t));
        require(buffer, "weight backend");
        storage.emplace_back(buffer, ggml_backend_buffer_free);
        void * base = ggml_backend_buffer_get_base(buffer);
        require(base, "host storage");
        std::memcpy(base, t->data, ggml_nbytes(t));
        t->data = base;
        t->buffer = buffer;
        return t;
    };
    decoder_layer_weights w{};
    w.full_attention = false;
    w.upstream_physical = w.linear.upstream_physical = true;
    w.input_norm = tensor("blk.0.attn_norm.weight", GGML_TYPE_F32, 2560);
    w.post_norm = tensor("blk.0.post_attention_norm.weight", GGML_TYPE_F32, 2560);
    w.gate = tensor("blk.0.ffn_gate.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.up = tensor("blk.0.ffn_up.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.down = tensor("blk.0.ffn_down.weight", GGML_TYPE_Q6_K, 9216, 2560);
    w.linear.qkv = tensor("blk.0.attn_qkv.weight", GGML_TYPE_Q6_K, 2560, 8192);
    w.linear.z = tensor("blk.0.attn_gate.weight", GGML_TYPE_Q4_K, 2560, 4096);
    w.linear.a = tensor("blk.0.ssm_alpha.weight", GGML_TYPE_Q4_K, 2560, 32);
    w.linear.b = tensor("blk.0.ssm_beta.weight", GGML_TYPE_Q4_K, 2560, 32);
    w.linear.out = tensor("blk.0.ssm_out.weight", GGML_TYPE_Q4_K, 4096, 2560);
    w.linear.conv = tensor("blk.0.ssm_conv1d.weight", GGML_TYPE_F32, 4, 1, 8192);
    w.linear.a_log = tensor("blk.0.ssm_a", GGML_TYPE_F32, 32);
    w.linear.dt_bias = tensor("blk.0.ssm_dt.bias", GGML_TYPE_F32, 32);
    w.linear.norm = tensor("blk.0.ssm_norm.weight", GGML_TYPE_F32, 128);
    converted_weight_cache cache_owner(0);
    require(cache_owner.enable_physical_repack(true), "quantized-only cache");
    w.physical_q4_repack = w.linear.physical_q4_repack = true;
    w.physical_weights_cache = &cache_owner;
    std::vector<float> input(256 * 2560), expected(input.size());
    read(capture / "input.f32", input.data(), input.size() * 4);
    read(capture / "hidden.f32", expected.data(), expected.size() * 4);
    std::vector<int> mods(256, 0);
    std::fill(mods.begin() + 4, mods.begin() + 246, 1);
    std::vector<lighton4b_prefill_chunk> plan;
    require(lighton4b_source_prefill_plan(mods, plan, lighton4b_source_prefill_aligned_mode(true)),
            "unchanged actual chunk plan");
    std::map<std::string, std::vector<float>> whole;
    lighton4b_linear_cache previous;
    for (const auto & part : plan) {
        lighton4b_linear_block_outputs b;
        const int n = part.tokens, offset = part.offset;
        require(
            lighton4b_linear_block_cpu(w, input.data() + size_t(offset) * 2560, n, 4, offset ? &previous : nullptr, b),
            "actual production layer0 block");
        require(aligned || !std::memcmp(b.hidden.data(), expected.data() + size_t(offset) * 2560, b.hidden.size() * 4),
                "OFF ALL256 actual native hidden0 byte closure");
        require(b.cache.physical_source_prefill_aligned == aligned, "actual aligned production cache identity");
        projected_linear_outputs l;
        require(linear_grouped_cpu(w.linear, 16, b.input_norm.data(), 1, n, 4, offset ? previous.conv.data() : nullptr,
                                   offset ? previous.recurrent.data() : nullptr, l, &cache_owner),
                "same-input production projected GDN replay");
        auto bits = [](const std::vector<float> & a, const std::vector<float> & b) {
            return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
        };
        require(bits(l.attention, b.attention) && bits(l.core.state, b.cache.recurrent) &&
                    bits(l.core.conv_history, b.cache.conv),
                "duplicate GDN actual block/cache exact");
        auto append = [&](const char * name, const std::vector<float> & x) {
            auto & v = whole[name];
            v.insert(v.end(), x.begin(), x.end());
        };
        append("input_norm", b.input_norm);
        append("qkv", l.qkv);
        append("z", l.z);
        append("a", l.a);
        append("b", l.b);
        append("postgdn", l.core.values);
        append("attention", b.attention);
        append("attention_residual", b.attention_residual);
        append("post_norm", b.post_norm);
        append("gate", b.gate);
        append("up", b.up);
        append("activation", b.activation);
        append("down", b.down);
        append("hidden", b.hidden);
        previous = std::move(b.cache);
        if (offset == 0 || offset + n == 228 || offset + n == 238 || offset + n == 246) {
            write(out / ("state" + std::to_string(offset + n) + ".f32"), previous.recurrent);
            write(out / ("fifo" + std::to_string(offset + n) + ".f32"), previous.conv);
        }
    }
    for (const auto & kv : whole) write(out / (kv.first + ".f32"), kv.second);
    require(cache_owner.bytes() == 0, "no F32 weight cache");
    std::cout << "ALIGNED_RECEIPT_COMPONENT PASS duplicate_GDN_cache_exact=1 quantized_only=1\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
