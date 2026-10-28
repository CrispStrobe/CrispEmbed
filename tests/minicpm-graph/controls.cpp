#include "qwen35_graph_decoder.h"
#include "qwen35_math.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace {
void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
bool near(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j)
            if (!std::isfinite(b[i][j]) || std::abs(a[i][j] - b[i][j]) > 0.002f) return false;
    }
    return true;
}
int run() {
    using namespace qwen35_ocr;
    env("CRISPEMBED_QWEN35_OCR", "1");
    env("CRISPEMBED_QWEN35_QUANT", "1");
    using ctx_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    ctx_owner ctx(ggml_init({ 65536, nullptr, true }), ggml_free);
    backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(ctx) && bool(backend), "fixture owners");
    std::vector<ggml_tensor *> weights;
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K })
        weights.push_back(ggml_new_tensor_2d(ctx.get(), type, 256, 4));
    buffer_owner buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "real backend weight storage");
    std::vector<float> source(1024), hidden(256 * 3);
    for (size_t i = 0; i < source.size(); ++i) source[i] = float(int(i % 17) - 8) / 128;
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(int(i % 9) - 4) / 256;
    std::vector<std::vector<uint8_t>> originals;
    for (auto * weight : weights) {
        originals.emplace_back(ggml_nbytes(weight));
        auto & bytes = originals.back();
        if (weight->type == GGML_TYPE_F32)
            std::memcpy(bytes.data(), source.data(), bytes.size());
        else if (weight->type == GGML_TYPE_F16)
            ggml_fp32_to_fp16_row(source.data(), reinterpret_cast<ggml_fp16_t *>(bytes.data()), source.size());
        else
            require(ggml_quantize_chunk(weight->type, source.data(), bytes.data(), 0, 4, 256, nullptr) == bytes.size(),
                    "quant fixture bytes");
        ggml_backend_tensor_set(weight, bytes.data(), 0, bytes.size());
    }
    graph_decoder one, two;
    std::vector<std::vector<float>> actual = { { 42 } }, expected, repeat;
    for (const char * mode : { static_cast<const char *>(nullptr), "", "0" }) {
        env("CRISPEMBED_MINICPM_OCR_GRAPH", mode);
        require(!one.project({ weights[0] }, hidden.data(), 256, 3, 2, actual) && actual.empty(),
                "off is rejected and clears");
    }
    env("CRISPEMBED_MINICPM_OCR_GRAPH", "1");
    require(project_cpu(weights, hidden.data(), 256, 3, 2, expected), "scalar storage control");
    require(one.project(weights, hidden.data(), 256, 3, 2, actual) && near(expected, actual),
            "all stored-type projections");
    require(two.project(weights, hidden.data(), 256, 3, 1, repeat) && near(actual, repeat), "separate backend context");
    require(one.project(weights, hidden.data(), 256, 3, 2, repeat) && repeat == actual, "same-context repeat exact");
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<uint8_t> bytes(ggml_nbytes(weights[i]));
        ggml_backend_tensor_get(weights[i], bytes.data(), 0, bytes.size());
        require(bytes == originals[i], "borrowed weights remain byte exact");
    }
    env("CRISPEMBED_QWEN35_QUANT", "0");
    require(!one.project({ weights[2] }, hidden.data(), 256, 3, 1, actual) && actual.empty(),
            "quant needs its existing gate");
    env("CRISPEMBED_QWEN35_QUANT", "1");
    hidden[17] = INFINITY;
    require(!one.project({ weights[0] }, hidden.data(), 256, 3, 1, actual) && actual.empty(), "nonfinite input");
    hidden[17] = 0;
    require(!one.project({ weights[0] }, hidden.data(), 255, 3, 1, actual) && actual.empty(), "wrong shape");
    require(!one.project({}, hidden.data(), 256, 3, 1, actual) && actual.empty(), "no matrices");
    require(!one.project({ weights[0] }, hidden.data(), 256, 4097, 1, actual) && actual.empty(),
            "token bound before input read");
    decoder_weights invalid;
    decoder_outputs stale;
    stale.logits = { 42 };
    const float positions[] = { 0, 0, 0 };
    require(!one.run(invalid, {}, hidden.data(), positions, 1, 1, nullptr, stale) && stale.logits.empty(),
            "invalid model clears outputs");
    one.clear();
    require(one.project({ weights[0] }, hidden.data(), 256, 3, 1, actual), "recovery after explicit clear");
    env("CRISPEMBED_MINICPM_OCR_GRAPH", "0");
    require(!one.project({ weights[0] }, hidden.data(), 256, 3, 1, actual) && actual.empty(), "disable after success");
    env("CRISPEMBED_MINICPM_OCR_GRAPH", "1");
    require(one.project({ weights[0] }, hidden.data(), 256, 3, 1, actual), "re-enable existing context");
    std::puts("PASS stored F32/F16/Q8_0/Q4_K, owners, gates, repeat, isolation, errors/recovery");
    return 0;
}
int entry(int, char **) {
    try {
        return run();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(entry(argc, argv));
}
