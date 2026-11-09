#include "lighton4b_native_context.h"
#include "core/clean_exit.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace qwen35_ocr;
static void gate(const char * key, const char * value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}
static int run() {
    int checks = 0;
    auto check = [&](bool ok) {
        if (!ok) {
            std::fprintf(stderr, "HF tiled operator check%d failed\n", checks);
            std::exit(1);
        }
        ++checks;
    };
    for (auto * k : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS", "CRISPEMBED_LIGHTON4B_Q6_K",
                      "CRISPEMBED_LIGHTON4B_HF_TILED_OUT", "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK",
                      "CRISPEMBED_LIGHTON4B_DECODER_COMPOSE", "CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK" })
        gate(k, "1");
    for (auto * k : { "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD",
                      "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                      "CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT" })
        gate(k, "0");
    auto * ctx = ggml_init({ 65536, nullptr, true });
    check(ctx != nullptr);
    auto matrix = [&](int width, int rows) {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
        check(t != nullptr);
        return t;
    };
    linear_weights w{};
    w.qkv = matrix(2560, 8192);
    w.z = matrix(2560, 4096);
    w.a = matrix(2560, 32);
    w.b = matrix(2560, 32);
    w.out = matrix(4096, 2560);
    w.conv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 8192);
    check(w.conv != nullptr);
    w.a_log = matrix(32, 1);
    w.dt_bias = matrix(32, 1);
    w.norm = matrix(128, 1);
    auto * tiled_out = matrix(4096, 2560);
    auto * decoder_norm = matrix(2560, 1);
    auto * backend = ggml_backend_cpu_init();
    check(backend != nullptr);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr);
    for (auto * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
        check(tensor->data != nullptr);
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
    }
    for (auto * t : { w.qkv, w.z })
        for (int64_t row = 0; row < t->ne[1]; ++row)
            static_cast<float *>(t->data)[row * 2560 + (row * 17) % 2560] = float(1 + row % 7) / 32;
    for (int c = 0; c < 8192; ++c) static_cast<float *>(w.conv->data)[c * 4 + 3] = 1;
    std::fill_n(static_cast<float *>(w.norm->data), 128, 1.0f);
    for (int row = 0; row < 2560; ++row) {
        const int col = (row * 37) % 4096;
        const int tiled = (16 * ((col / 128) % 2) + (col / 128) / 2) * 128 + col % 128;
        const float value = row % 2 ? -.5f : .5f;
        static_cast<float *>(w.out->data)[row * 4096 + col] = value;
        static_cast<float *>(tiled_out->data)[row * 4096 + tiled] = value;
    }
    std::vector<float> hidden(2560);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(1 + i % 11) / 8;
    projected_linear_outputs baseline, candidate, repeat;
    check(linear_grouped_cpu(w, 16, hidden.data(), 1, 1, 2, nullptr, nullptr, baseline));
    check(baseline.attention.size() == 2560);
    auto tiled = w;
    tiled.out = tiled_out;
    tiled.hf_tiled_out = true;
    check(linear_grouped_cpu(tiled, 16, hidden.data(), 1, 1, 2, nullptr, nullptr, candidate));
    auto equal = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4);
    };
    check(equal(baseline.qkv, candidate.qkv) && equal(baseline.z, candidate.z) && equal(baseline.a, candidate.a) &&
          equal(baseline.b, candidate.b));
    check(equal(baseline.core.values, candidate.core.values) && equal(baseline.core.state, candidate.core.state) &&
          equal(baseline.core.conv_history, candidate.core.conv_history));
    check(std::any_of(baseline.attention.begin(), baseline.attention.end(), [](float x) { return x != 0; }));
    check(equal(baseline.attention, candidate.attention));
    check(linear_grouped_cpu(w, 16, hidden.data(), 1, 1, 2, nullptr, nullptr, repeat) &&
          equal(repeat.attention, baseline.attention));
    gate("CRISPEMBED_LIGHTON4B_HF_TILED_OUT", "0");
    check(!linear_grouped_cpu(tiled, 16, hidden.data(), 1, 1, 2, nullptr, nullptr, candidate) &&
          candidate.attention.empty());
    gate("CRISPEMBED_LIGHTON4B_HF_TILED_OUT", "1");
    tiled.upstream_physical = true;
    check(!linear_grouped_cpu(tiled, 16, hidden.data(), 1, 1, 2, nullptr, nullptr, candidate));
    tiled.upstream_physical = false;
    gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "1");
    check(!lighton4b_bf16_grouped_cpu(tiled, hidden.data(), 1, 2, nullptr, nullptr, candidate));
    gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "0");
    // Cache mode mismatches must stop before any full block/decoder computation.
    decoder_layer_weights layer{};
    layer.linear = tiled;
    lighton4b_linear_cache old;
    old.hf_tiled_out = false;
    lighton4b_linear_block_outputs block;
    block.hidden = { 1 };
    check(!lighton4b_linear_block_cpu(layer, hidden.data(), 1, 2, &old, block) && block.hidden.empty());
    decoder_weights decoder{};
    decoder.layers.resize(32);
    decoder.norm = decoder_norm;
    decoder.hf_tiled_out = true;
    lighton4b_decoder_cache previous;
    previous.hf_tiled_out = false;
    lighton4b_decoder_outputs output;
    output.norm = { 1 };
    float positions[3] = { 0, 0, 0 };
    check(!lighton4b_decoder_hidden_cpu(decoder, hidden.data(), positions, 1, 2, &previous, output) &&
          output.norm.empty());
    decoder.hf_tiled_out = false;
    previous.hf_tiled_out = true;
    check(!lighton4b_decoder_hidden_cpu(decoder, hidden.data(), positions, 1, 2, &previous, output));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf(
        "checks=%d pass=1 original_model_payload_bytes=0 synthetic_fullwidth_projection=1 cache_mode_controls=1\n",
        checks);
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
