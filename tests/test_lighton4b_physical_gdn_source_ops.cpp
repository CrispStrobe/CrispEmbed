// Model-free lifecycle/gate/cache controls. Source-fed numerical proof is separate.
#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace qwen35_ocr;
static void need(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
static int crispembed_test_main() {
    try {
        for (const char * key :
             { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_QUANT",
               "CRISPEMBED_LIGHTON4B_Q6_K", "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK", "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS",
               "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS" })
            setenv(key, "1", 1);
        for (const char * key : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                                  "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK",
                                  "CRISPEMBED_QWEN35_PRECISE_RMS", "CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU" })
            setenv(key, "0", 1);
        setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION", "all", 1);
        const char * flag = "CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS";
        size_t checks = 0;
        unsetenv(flag);
        need(!lighton4b_physical_gdn_source_ops_mode(true), "default source operators off");
        ++checks;
        for (const char * value : { "0", "false", "off", "no", "FaLsE", "OFF" }) {
            setenv(flag, value, 1);
            need(!lighton4b_physical_gdn_source_ops_mode(true), "false alias");
            ++checks;
        }
        for (const char * value : { "1", "true", "on", "yes" }) {
            setenv(flag, value, 1);
            need(lighton4b_physical_gdn_source_ops_mode(true) && !lighton4b_physical_gdn_source_ops_mode(false),
                 "physical owner scope");
            ++checks;
        }
        linear_inputs bad{};
        bad.a_is_pretransformed = true;
        linear_outputs stale;
        stale.values = { 123 };
        need(!lighton4b_physical_gdn_source_ops_cpu(bad, 16, stale) && stale.values.empty() && stale.state.empty(),
             "bad geometry clears stale result");
        ++checks;
        auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({ 65536, nullptr, true }), ggml_free);
        need(bool(ctx), "descriptor context");
        auto buffer = [](size_t bytes) {
            return std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
                ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes), ggml_backend_buffer_free);
        };
        auto zeros = buffer(20ull * 1024 * 1024), ones = buffer(4096 * sizeof(float)),
             negative = buffer(128 * sizeof(float));
        need(zeros && ones && negative, "aligned synthetic storage");
        std::memset(ggml_backend_buffer_get_base(zeros.get()), 0, 20ull * 1024 * 1024);
        std::fill_n(static_cast<float *>(ggml_backend_buffer_get_base(ones.get())), 4096, 1.0f);
        std::fill_n(static_cast<float *>(ggml_backend_buffer_get_base(negative.get())), 128, -1.0f);
        auto tensor = [&](ggml_type type, int64_t a, int64_t b, int64_t c, ggml_backend_buffer_t storage) {
            auto * t = ggml_new_tensor_3d(ctx.get(), type, a, b, c);
            t->buffer = storage;
            t->data = ggml_backend_buffer_get_base(storage);
            need(ggml_nbytes(t) <= ggml_backend_buffer_get_size(storage), "synthetic tensor bound");
            return t;
        };
        decoder_layer_weights w{};
        w.upstream_physical = true;
        w.linear.upstream_physical = true;
        w.input_norm = tensor(GGML_TYPE_F32, 2560, 1, 1, ones.get());
        w.post_norm = tensor(GGML_TYPE_F32, 2560, 1, 1, ones.get());
        w.linear.qkv = tensor(GGML_TYPE_Q6_K, 2560, 8192, 1, zeros.get());
        w.linear.z = tensor(GGML_TYPE_Q4_K, 2560, 4096, 1, zeros.get());
        w.linear.a = tensor(GGML_TYPE_Q4_K, 2560, 32, 1, zeros.get());
        w.linear.b = tensor(GGML_TYPE_Q4_K, 2560, 32, 1, zeros.get());
        w.linear.out = tensor(GGML_TYPE_Q4_K, 4096, 2560, 1, zeros.get());
        w.linear.conv = tensor(GGML_TYPE_F32, 4, 1, 8192, zeros.get());
        w.linear.a_log = tensor(GGML_TYPE_F32, 32, 1, 1, negative.get());
        w.linear.dt_bias = tensor(GGML_TYPE_F32, 32, 1, 1, zeros.get());
        w.linear.norm = tensor(GGML_TYPE_F32, 128, 1, 1, ones.get());
        w.gate = tensor(GGML_TYPE_Q4_K, 2560, 9216, 1, zeros.get());
        w.up = tensor(GGML_TYPE_Q4_K, 2560, 9216, 1, zeros.get());
        w.down = tensor(GGML_TYPE_Q4_K, 9216, 2560, 1, zeros.get());
        std::vector<float> input(2560, 1.0f / 32);
        lighton4b_linear_block_outputs off, on, next, rejected;
        auto good = [&](const lighton4b_linear_block_outputs & o, bool mode, int tokens) {
            need(o.hidden.size() == input.size() && !std::memcmp(o.hidden.data(), input.data(), input.size() * 4),
                 "analytic zero-matrix residual preserves input");
            need(o.cache.tokens == tokens && o.cache.physical_gdn_source_ops == mode && o.cache.conv.size() == 32768 &&
                     o.cache.recurrent.size() == 524288,
                 "actual cache mode and unchanged layouts");
            need(std::all_of(o.cache.recurrent.begin(), o.cache.recurrent.end(), [](float x) { return x == 0; }) &&
                     std::all_of(o.cache.conv.begin(), o.cache.conv.end(), [](float x) { return x == 0; }),
                 "zero matrix actual cache");
            ++checks;
        };
        setenv(flag, "0", 1);
        setenv("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "0", 1);
        need(!lighton4b_linear_block_cpu(w, input.data(), 1, 4, nullptr, rejected) && rejected.hidden.empty(),
             "Q6 fixture rejects missing native context");
        ++checks;
        setenv("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "1", 1);
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, nullptr, off),
             "old synthetic full block with required Q6 context gate");
        good(off, false, 1);
        setenv(flag, "1", 1);
        rejected.hidden = { 123 };
        need(!lighton4b_linear_block_cpu(w, input.data(), 1, 4, &off.cache, rejected) && rejected.hidden.empty() &&
                 rejected.cache.recurrent.empty(),
             "off-to-on cache rejected before use");
        ++checks;
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, nullptr, on), "new synthetic full block");
        good(on, true, 1);
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, &on.cache, next), "same-mode source cache continues");
        good(next, true, 2);
        setenv(flag, "off", 1);
        rejected.hidden = { 123 };
        need(!lighton4b_linear_block_cpu(w, input.data(), 1, 4, &on.cache, rejected) && rejected.hidden.empty(),
             "on-to-off cache rejected");
        ++checks;
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, &off.cache, next), "same-mode old cache continues");
        good(next, false, 2);
        need(off.cache.tokens == 1 && !off.cache.physical_gdn_source_ops && on.cache.tokens == 1 &&
                 on.cache.physical_gdn_source_ops,
             "previous caches remain immutable");
        ++checks;
        // The independently gated FFN arithmetic must bind the actual recurrent
        // stream even when its zero-matrix analytic hidden output is unchanged.
        const char * swiglu = "CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU";
        setenv(swiglu, "1", 1);
        rejected.hidden = { 123 };
        need(!lighton4b_linear_block_cpu(w, input.data(), 1, 4, &off.cache, rejected) && rejected.hidden.empty(),
             "SwiGLU off-to-on cache rejected");
        ++checks;
        lighton4b_linear_block_outputs fused, fused_next;
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, nullptr, fused) && fused.cache.physical_swiglu,
             "fused synthetic full block binds cache");
        good(fused, false, 1);
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, &fused.cache, fused_next) &&
                 fused_next.cache.physical_swiglu,
             "same-mode fused cache continues");
        good(fused_next, false, 2);
        setenv(swiglu, "0", 1);
        rejected.hidden = { 123 };
        need(!lighton4b_linear_block_cpu(w, input.data(), 1, 4, &fused.cache, rejected) && rejected.hidden.empty(),
             "SwiGLU on-to-off cache rejected");
        ++checks;
        need(lighton4b_linear_block_cpu(w, input.data(), 1, 4, &off.cache, next) && !next.cache.physical_swiglu,
             "SwiGLU original cache recovers");
        good(next, false, 2);
        need(!off.cache.physical_swiglu && fused.cache.physical_swiglu && fused.cache.tokens == 1,
             "SwiGLU previous caches immutable");
        ++checks;
        std::printf("PHYSICAL_GDN_SOURCE_OPS_CONTROLS PASS checks=%zu model_payload=0 decoded=0\n", checks);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
