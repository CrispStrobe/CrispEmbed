#include "lightonocr3_profile.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int crispembed_test_main() {
    int failures = 0;
    auto check = [&](bool result, const char * message) {
        if (!result) {
            ++failures;
            fprintf(stderr, "%s\n", message);
        }
    };
    const char * gate = "CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS";
    auto set_gate = [&](const char * value) {
#ifdef _WIN32
        _putenv_s(gate, value ? value : "");
#else
        if (value)
            setenv(gate, value, 1);
        else
            unsetenv(gate);
#endif
    };
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        set_gate(value);
        check(lightonocr3::vision_rms_epsilon(true) == 1e-6f, "legacy epsilon/default changed");
    }
    set_gate("1");
    check(lightonocr3::vision_rms_epsilon(false) == 1e-6f, "legacy model affected by LightOn3 gate");
    // Actual GGML RMSNorm on a low-variance input makes the epsilon difference
    // observable; compare the opt-in result to the independent scalar formula.
    auto * gg = ggml_init({ 2 * 1024 * 1024, nullptr, false });
    if (!gg) return 2;
    const float values[] = { 0.001f, -0.002f, 0.003f, -0.004f };
    auto * input = ggml_new_tensor_1d(gg, GGML_TYPE_F32, 4);
    std::memcpy(ggml_get_data(input), values, sizeof(values));
    auto * norm = ggml_rms_norm(gg, input, lightonocr3::vision_rms_epsilon(true));
    auto * graph = ggml_new_graph(gg);
    ggml_build_forward_expand(graph, norm);
    check(ggml_graph_compute_with_ctx(gg, graph, 1) == GGML_STATUS_SUCCESS, "norm component execution failed");
    const auto * actual = static_cast<const float *>(ggml_get_data(norm));
    double mean_square = 0;
    for (float value : values) mean_square += double(value) * value / 4;
    for (int i = 0; i < 4; ++i)
        check(std::abs(actual[i] - values[i] / std::sqrt(mean_square + 1e-5)) < 1e-6,
              "Pixtral epsilon differs from source RMSNorm");
    ggml_free(gg);
    set_gate(nullptr);
    gate = "CRISPEMBED_LIGHTONOCR3_KV_F32";
    for (const char * value : { static_cast<const char *>(nullptr), "", "0", "1" }) {
        set_gate(value);
        const bool enabled = value && std::strcmp(value, "1") == 0;
        check(lightonocr3::kv_f32_enabled(true, true) == enabled, "KV F32 gate parsing changed");
        check(!lightonocr3::kv_f32_enabled(false, true), "KV F32 gate affects legacy CPU model");
        check(!lightonocr3::kv_f32_enabled(true, false), "KV F32 gate affects GPU backend");
        check(!lightonocr3::kv_f32_enabled(false, false), "KV F32 gate affects legacy GPU model");
    }
    set_gate(nullptr);
    // Exercise the real cache layout and ggml_cpy conversions, independently of
    // a model: prefill writes two positions, then decode appends a third. Each
    // layer/head/value differs so a wrong byte stride cannot pass by coincidence.
    for (ggml_type cache_type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        auto * cache_gg = ggml_init({ 8 * 1024 * 1024, nullptr, false });
        if (!cache_gg) return 2;
        constexpr int hd = 4, max_seq = 5, heads = 2, layers = 2;
        ggml_tensor * caches[] = { ggml_new_tensor_4d(cache_gg, cache_type, hd, max_seq, heads, layers),
                                   ggml_new_tensor_4d(cache_gg, cache_type, hd, max_seq, heads, layers) };
        for (int kind = 0; kind < 2; ++kind) {
            auto * cache = caches[kind];
            std::memset(ggml_get_data(cache), 0, ggml_nbytes(cache));
            check(cache->nb[1] == hd * ggml_type_size(cache_type) && cache->nb[2] == max_seq * cache->nb[1] &&
                      cache->nb[3] == heads * cache->nb[2],
                  "KV cache layout strides differ");
            auto authored = [=](int layer, int position, int head, int channel) {
                return 1.0001f + kind * 8.0f + layer * 2.0f + head * 0.25f + position * 0.03125f +
                       channel * 0.00390625f;
            };
            for (int layer = 0; layer < layers; ++layer) {
                for (int start : { 0, 2 }) {
                    const int count = start == 0 ? 2 : 1;
                    auto * source = ggml_new_tensor_3d(cache_gg, GGML_TYPE_F32, hd, count, heads);
                    auto * data = static_cast<float *>(ggml_get_data(source));
                    for (int head = 0; head < heads; ++head)
                        for (int pos = 0; pos < count; ++pos)
                            for (int channel = 0; channel < hd; ++channel)
                                data[channel + hd * (pos + count * head)] = authored(layer, start + pos, head, channel);
                    auto * view = ggml_view_4d(cache_gg, cache, hd, count, heads, 1, cache->nb[1], cache->nb[2],
                                               cache->nb[3], layer * cache->nb[3] + start * cache->nb[1]);
                    auto * writes = ggml_new_graph(cache_gg);
                    ggml_build_forward_expand(writes, ggml_cpy(cache_gg, source, view));
                    check(ggml_graph_compute_with_ctx(cache_gg, writes, 1) == GGML_STATUS_SUCCESS,
                          "KV cache write failed");
                }
            }
            for (int layer = 0; layer < layers; ++layer) {
                auto * view =
                    ggml_view_3d(cache_gg, cache, hd, max_seq, heads, cache->nb[1], cache->nb[2], layer * cache->nb[3]);
                auto * restored = ggml_new_tensor_3d(cache_gg, GGML_TYPE_F32, hd, max_seq, heads);
                auto * reads = ggml_new_graph(cache_gg);
                ggml_build_forward_expand(reads, ggml_cpy(cache_gg, view, restored));
                check(ggml_graph_compute_with_ctx(cache_gg, reads, 1) == GGML_STATUS_SUCCESS, "KV cache read failed");
                const auto * data = static_cast<const float *>(ggml_get_data(restored));
                for (int head = 0; head < heads; ++head)
                    for (int pos = 0; pos < max_seq; ++pos)
                        for (int channel = 0; channel < hd; ++channel) {
                            const float value = data[channel + hd * (pos + max_seq * head)];
                            if (pos >= 3) {
                                check(value == 0.0f, "KV write changed an unused position");
                            } else {
                                const float expected = authored(layer, pos, head, channel);
                                if (cache_type == GGML_TYPE_F32) {
                                    check(value == expected, "F32 cache lost authored precision or stride");
                                } else {
                                    check(value == ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected)),
                                          "F16 cache conversion or stride differs");
                                    check(value != expected, "F16 control unexpectedly preserved authored F32 value");
                                }
                            }
                        }
            }
        }
        ggml_free(cache_gg);
    }
    // Official floor/ceil policy: odd patch grids must not become even grids.
    check(lightonocr3::image_size(317, 452) == std::make_pair(322, 462), "odd patch grid changed");
    check(lightonocr3::image_size(3080, 1540) == std::make_pair(1540, 770), "aspect downscale changed");
    check(lightonocr3::image_size(28, 28) == std::make_pair(28, 28), "aligned input changed");
    bool rejected = false;
    try {
        lightonocr3::image_size(0, 28);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "zero image accepted");
    rejected = false;
    try {
        lightonocr3::image_size(1, 100);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "empty merge accepted");
    std::vector<uint8_t> rgb(28 * 28 * 3, 255);
    int ph, pw;
    const auto normalized = lightonocr3::preprocess(rgb.data(), 28, 28, ph, pw);
    check(ph == 2 && pw == 2 && normalized.size() == rgb.size(), "processor shape mismatch");
    check(std::abs(normalized[0] - 1.9303362f) < 1e-6f, "white/red normalization drift");
    check(std::abs(normalized[784] - 2.0748837f) < 1e-6f, "white/green normalization drift");
    check(std::abs(normalized[1568] - 2.145897f) < 1e-6f, "white/blue normalization drift");
    std::vector<int32_t> prefix{ 151644, 8948, 151645, 198, 151644, 872, 198 };
    const std::vector<int32_t> suffix{ 151645, 198, 151644, 77091, 198 };
    const std::vector<int32_t> grounding{ 1951, 287, 198, 151645, 198, 151644, 77091, 198 };
    const std::vector<int32_t> stops{ 151645, 151643 };
    std::vector<int32_t> special;
    for (int id = 151643; id <= 151656; ++id) special.push_back(id);
    const char * model = "lightonai/LightOnOCR-3-1B";
    const char * revision = "23d544085c886bbdace647bcfab91203299417e2";
    check(lightonocr3::token_contract_matches(model, revision, prefix, suffix, prefix, grounding, stops, special),
          "pinned token contract rejected");
    check(!lightonocr3::token_contract_matches("different/repo", revision, prefix, suffix, prefix, grounding, stops,
                                               special),
          "substituted model repo accepted");
    auto changed = prefix;
    changed[1] += 1;
    check(!lightonocr3::token_contract_matches(model, revision, changed, suffix, prefix, grounding, stops, special),
          "in-range but substituted prompt accepted");
    special.pop_back();
    check(!lightonocr3::token_contract_matches(model, revision, prefix, suffix, prefix, grounding, stops, special),
          "incomplete special-ID set accepted");
    return failures ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
