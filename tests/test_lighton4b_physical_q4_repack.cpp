#include "lighton4b_native_context.h"
#include "core/clean_exit.h"
#include "qwen35_weight_cache.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
static int checks = 0;
static void check(bool ok, const char * why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
static void flag(const char * key, const char * value) {
    setenv(key, value, 1);
}
static void gates() {
    const char * on[] = { "CRISPEMBED_QWEN35_OCR",
                          "CRISPEMBED_QWEN35_QUANT",
                          "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS",
                          "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
                          "CRISPEMBED_LIGHTON4B_QUANT",
                          "CRISPEMBED_LIGHTON4B_Q6_K",
                          "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL",
                          "CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS",
                          "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK",
                          "CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK",
                          "CRISPEMBED_LIGHTON4B_DECODER_COMPOSE",
                          "CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION" };
    for (auto * key : on) flag(key, "1");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", "all");
    for (auto * key : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                        "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK" })
        flag(key, "0");
}
struct storage {
    ggml_context * ctx = ggml_init({ 1024 * 1024, nullptr, true });
    std::vector<ggml_backend_buffer_t> buffers;
    ~storage() {
        for (auto * b : buffers) ggml_backend_buffer_free(b);
        ggml_free(ctx);
    }
    ggml_backend_buffer_t bytes(size_t count) {
        auto * b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), count);
        check(b, "host buffer");
        buffers.push_back(b);
        ggml_backend_buffer_clear(b, 0);
        return b;
    }
    ggml_tensor * tensor(ggml_type type, int a, int b, ggml_backend_buffer_t buffer, int c = 1) {
        auto * t = ggml_new_tensor_3d(ctx, type, a, b, c);
        t->data = ggml_backend_buffer_get_base(buffer);
        t->buffer = buffer;
        return t;
    }
};
static std::vector<float> calculate(ggml_backend_t backend, ggml_tensor * weight, const std::vector<float> & x,
                                    int tokens) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 2 * 1024 * 1024, nullptr, false }), ggml_free);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, weight->ne[0], tokens);
    std::memcpy(input->data, x.data(), x.size() * 4);
    auto * y = ggml_mul_mat(ctx.get(), weight, input);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, y);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "lease matmul");
    auto * p = static_cast<float *>(y->data);
    return { p, p + ggml_nelements(y) };
}
static void lifecycle() {
    storage s;
    auto * raw = s.bytes(ggml_row_size(GGML_TYPE_Q4_K, 256) * 8);
    auto * source = s.tensor(GGML_TYPE_Q4_K, 256, 8, raw);
    auto * second = s.tensor(GGML_TYPE_Q4_K, 256, 8, raw);
    std::vector<float> full(256 * 8), x(256 * 4);
    for (size_t i = 0; i < full.size(); ++i) full[i] = std::sin(float(i) * .13f) * .3f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(float(i) * .07f);
    check(ggml_quantize_chunk(GGML_TYPE_Q4_K, full.data(), source->data, 0, 8, 256, nullptr) == ggml_nbytes(source),
          "quantized fixture");
    std::vector<char> original(ggml_nbytes(source));
    std::memcpy(original.data(), source->data, original.size());
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    ggml_backend_cpu_set_n_threads(backend.get(), 4);
    auto * device = ggml_backend_get_device(backend.get());
    std::vector<std::vector<float>> old, again, nonphysical;
    check(project_cpu({ source }, x.data(), 256, 4, 4, old, nullptr, true), "original borrowed path");
    converted_weight_cache disabled(0);
    check(!disabled.enable_physical_repack(true), "default-off cache does not initialize");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "1");
    converted_weight_cache cache(0);
    check(!cache.enable_physical_repack(false), "nonphysical owner cannot initialize");
    check(!cache.enable_physical_repack(true, 0), "zero budget");
    check(!cache.enable_physical_repack(true, converted_weight_cache::maximum_physical_repack_budget + 1),
          "excess budget");
    const auto budget = ggml_nbytes(source) + converted_weight_cache::physical_repack_entry_overhead;
    check(cache.enable_physical_repack(true, budget), "bounded initialization");
    check(!cache.enable_physical_repack(true, budget + 1), "cannot replace live accounting");
    auto a = cache.get_physical_repacked(source, device);
    check(bool(a), "first repack");
    check(a->type == GGML_TYPE_Q4_K && a->extra && std::strcmp(ggml_backend_buffer_name(a->buffer), "CPU_REPACK") == 0,
          "quantized preferred buffer");
    const auto bytes = cache.physical_repack_bytes();
    check(bytes > ggml_nbytes(source) && bytes <= budget, "actual bounded charge");
    check(cache.bytes() == 0 && cache.entries() == 0, "no F32 weight cache");
    auto b = cache.get_physical_repacked(source, device);
    check(a.get() == b.get() && cache.physical_repack_hits() == 1, "immutable reuse");
    check(!cache.get_physical_repacked(second, device), "second allocation exceeds budget");
    check(cache.physical_repack_entries() == 1, "one indexed allocation");
    const auto expected = calculate(backend.get(), a.get(), x, 4);
    std::vector<std::vector<float>> actual;
    check(project_cpu({ source }, x.data(), 256, 4, 4, actual, &cache, true) && actual[0] == expected,
          "project path consumes preferred tensor");
    auto * saved = source->data;
    source->data = static_cast<char *>(saved) + 16;
    check(!cache.get_physical_repacked(source, device), "source pointer mutation rejected");
    source->data = saved;
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "0");
    check(!cache.get_physical_repacked(source, device), "disabled cache never serves existing entries");
    check(!project_cpu({ source }, x.data(), 256, 4, 4, actual, &cache, true), "mode toggle rejects cached project");
    check(project_cpu({ source }, x.data(), 256, 4, 4, again, nullptr, true) && old == again,
          "gate-off borrowed byte identity");
    check(project_cpu({ source }, x.data(), 256, 4, 4, nonphysical, nullptr, false), "nonphysical baseline");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "1");
    check(project_cpu({ source }, x.data(), 256, 4, 4, actual, nullptr, false) && actual == nonphysical,
          "nonphysical gate inert");
    cache.clear();
    check(cache.physical_repack_entries() == 0 && cache.physical_repack_bytes() == bytes,
          "clear retains charge for leased buffers");
    check(!cache.get_physical_repacked(source, device), "leases cannot overcommit cleared cache");
    check(calculate(backend.get(), a.get(), x, 4) == expected, "lease survives clear");
    a.reset();
    check(cache.physical_repack_bytes() == bytes, "second lease retains charge");
    b.reset();
    check(cache.physical_repack_bytes() == 0, "last lease refunds bytes");
    std::shared_ptr<ggml_tensor> survivor;
    {
        converted_weight_cache temporary(0);
        check(temporary.enable_physical_repack(true, budget), "temporary cache");
        survivor = temporary.get_physical_repacked(source, device);
        check(bool(survivor), "temporary lease");
    }
    check(calculate(backend.get(), survivor.get(), x, 4) == expected, "lease survives cache destruction");
    std::weak_ptr<ggml_tensor> weak = survivor;
    survivor.reset();
    check(weak.expired(), "no ownership cycle");
    check(std::memcmp(original.data(), source->data, original.size()) == 0 && !source->extra,
          "source quant storage and metadata unchanged");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", "unknown");
    check(!cache.get_physical_repacked(source, device), "unknown selector closed");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", "all");
    flag("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "1");
    check(!cache.get_physical_repacked(source, device), "incompatible precision closed");
    flag("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "0");
}
struct blocks {
    storage s;
    ggml_backend_buffer_t zero = s.bytes(size_t(2560) * 9216 * 4), ones = s.bytes(2560 * 4), negative = s.bytes(32 * 4);
    decoder_layer_weights linear, attention;
    ggml_tensor * final_norm = nullptr;
    blocks() {
        std::fill_n(static_cast<float *>(ggml_backend_buffer_get_base(ones)), 2560, 1.0f);
        std::fill_n(static_cast<float *>(ggml_backend_buffer_get_base(negative)), 32, -1.0f);
        auto f = [&](int a, int b = 1) { return s.tensor(GGML_TYPE_F32, a, b, zero); };
        auto n = [&](int a) { return s.tensor(GGML_TYPE_F32, a, 1, ones); };
        linear.upstream_physical = true;
        linear.input_norm = n(2560);
        linear.post_norm = n(2560);
        linear.gate = f(2560, 9216);
        linear.up = f(2560, 9216);
        linear.down = f(9216, 2560);
        linear.linear = { f(2560, 8192),
                          f(2560, 4096),
                          f(2560, 32),
                          f(2560, 32),
                          f(4096, 2560),
                          s.tensor(GGML_TYPE_F32, 4, 1, zero, 8192),
                          s.tensor(GGML_TYPE_F32, 32, 1, negative),
                          f(32),
                          n(128) };
        linear.linear.upstream_physical = true;
        attention.full_attention = true;
        attention.upstream_physical = true;
        attention.input_norm = n(2560);
        attention.post_norm = n(2560);
        attention.gate = f(2560, 9216);
        attention.up = f(2560, 9216);
        attention.down = f(9216, 2560);
        attention.attention = { f(2560, 8192), f(2560, 1024), f(2560, 1024), f(4096, 2560), n(256), n(256) };
        attention.attention.upstream_physical = true;
        final_norm = n(2560);
    }
    decoder_weights weights(bool enabled, const std::shared_ptr<converted_weight_cache> & cache) {
        decoder_weights w;
        w.upstream_physical = true;
        w.physical_q4_repack = enabled;
        w.physical_weights_cache = cache;
        w.norm = final_norm;
        for (int i = 0; i < 32; ++i) {
            auto layer = i % 4 == 3 ? attention : linear;
            layer.physical_q4_repack = enabled;
            layer.physical_weights_cache = cache.get();
            if (layer.full_attention)
                layer.attention.physical_q4_repack = enabled;
            else
                layer.linear.physical_q4_repack = enabled;
            w.layers.push_back(layer);
        }
        return w;
    }
};
static void cache_modes() {
    blocks fixture;
    std::vector<float> x(2560, 1.0f), pos(3, 0.0f);
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "0");
    auto off = fixture.weights(false, {});
    lighton4b_decoder_outputs old;
    check(lighton4b_decoder_hidden_cpu(off, x.data(), pos.data(), 1, 4, nullptr, old), "valid off decoder fixture");
    check(old.cache.tokens == 1 && !old.cache.physical_q4_repack, "off mode cache");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "1");
    auto cache = std::make_shared<converted_weight_cache>(0);
    check(cache->enable_physical_repack(true), "on manager");
    auto on = fixture.weights(true, cache);
    lighton4b_decoder_outputs current;
    check(lighton4b_decoder_hidden_cpu(on, x.data(), pos.data(), 1, 4, nullptr, current), "valid on decoder fixture");
    check(current.cache.physical_q4_repack && current.hidden == old.hidden && current.norm == old.norm,
          "F32 components unchanged by quant-only mode");
    lighton4b_decoder_outputs rejected;
    check(!lighton4b_decoder_hidden_cpu(on, x.data(), pos.data(), 1, 4, &old.cache, rejected) &&
              rejected.hidden.empty(),
          "decoder off-to-on cache rejected");
    lighton4b_linear_block_outputs l;
    check(!lighton4b_linear_block_cpu(on.layers[0], x.data(), 1, 4, &old.cache.layers[0].linear, l),
          "linear off-to-on rejected");
    lighton4b_attention_block_outputs a;
    check(!lighton4b_attention_block_cpu(on.layers[3], x.data(), pos.data(), 1, 4, &old.cache.layers[3].attention, a),
          "attention off-to-on rejected");
    attention_config config;
    config.query_heads = 16;
    config.kv_heads = 4;
    attention_outputs direct;
    check(!attention_cpu(on.layers[3].attention, config, x.data(), pos.data(), 1, 1, 4, &old.cache.layers[3].attention,
                         direct, cache.get()),
          "low-level attention off-to-on rejected");
    flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "0");
    check(!lighton4b_decoder_hidden_cpu(off, x.data(), pos.data(), 1, 4, &current.cache, rejected),
          "decoder on-to-off rejected");
    check(!lighton4b_decoder_hidden_cpu(on, x.data(), pos.data(), 1, 4, nullptr, rejected),
          "loaded owner cannot change mode before first token");
    check(!lighton4b_linear_block_cpu(off.layers[0], x.data(), 1, 4, &current.cache.layers[0].linear, l),
          "linear on-to-off rejected");
    check(!lighton4b_attention_block_cpu(off.layers[3], x.data(), pos.data(), 1, 4, &current.cache.layers[3].attention,
                                         a),
          "attention on-to-off rejected");
    check(!attention_cpu(off.layers[3].attention, config, x.data(), pos.data(), 1, 1, 4,
                         &current.cache.layers[3].attention, direct, nullptr),
          "low-level attention on-to-off rejected");
    check(cache->bytes() == 0 && cache->entries() == 0, "zero converted F32 allocations throughout");
}
static int crispembed_test_main() {
    try {
        gates();
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "1");
        check(!lighton4b_physical_q4_repack_mode(false) && lighton4b_physical_q4_repack_valid(false),
              "nonphysical mode inert");
        if (!lighton4b_physical_f16_attention_available()) {
            converted_weight_cache cache(0);
            check(!lighton4b_physical_q4_repack_valid(true) && !cache.enable_physical_repack(true) &&
                      !cache.physical_repack_enabled() && cache.physical_repack_bytes() == 0,
                  "unsupported closed");
            std::cout << "PHYSICAL_Q4_REPACK_UNSUPPORTED_CLOSED PASS " << checks << "\n";
            return 0;
        }
        flag("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK", "0");
        lifecycle();
        cache_modes();
        std::cout << "PHYSICAL_Q4_REPACK PASS " << checks << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
