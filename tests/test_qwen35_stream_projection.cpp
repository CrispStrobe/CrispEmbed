#include "core/clean_exit.h"
#include "qwen35_math.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static int calls = 0;
static bool observe(const char *, const float *, int, int, void *) {
    ++calls;
    return true;
}
static bool throwing_observer(const char *, const float *, int, int, void *) {
    throw 7;
}
static int run() {
    // Both gate modes are executed in one process; counters are per run.
    calls = 0;
    auto fail = [](int line) {
        std::fprintf(stderr, "STREAM_PROJECTION_FAILURE mode=%s line=%d observer_calls=%d\n",
                     std::getenv("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE"), line, calls);
        return 1;
    };
    constexpr int width = 256, rows = 257, tokens = 3;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1024 * 1024, nullptr, true }), ggml_free);
    if (!ctx) return fail(__LINE__);
    auto * f32 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    auto * f16 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, width, rows);
    auto * q8 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, width, rows);
    auto * q4 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, width, rows);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) return fail(__LINE__);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) return fail(__LINE__);
    std::vector<float> hidden(width * tokens), output;
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(int(i % 17) - 8) / 16;
    for (int i = 0; i < width * rows; ++i) {
        const float value = float(i % 71 - 35) / 32;
        static_cast<float *>(f32->data)[i] = value;
        static_cast<ggml_fp16_t *>(f16->data)[i] = ggml_fp32_to_fp16(value);
    }
    for (auto * matrix : { q8, q4 }) {
        const size_t written =
            ggml_quantize_chunk(matrix->type, static_cast<float *>(f32->data), matrix->data, 0, rows, width, nullptr);
        if (written != ggml_nbytes(matrix)) return fail(__LINE__);
    }
    env("CRISPEMBED_QWEN35_OCR", "1");
    const char * gate = "CRISPEMBED_QWEN35_STREAM_PROJECTION";
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        env(gate, value);
        output = { 42 };
        if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
            return fail(__LINE__);
    }
    env(gate, "1");
    for (auto * matrix : { f32, f16 }) {
        std::vector<std::vector<float>> baseline;
        if (!qwen35_ocr::project_cpu({ matrix }, hidden.data(), width, tokens, 2, baseline)) return fail(__LINE__);
        // Different seams, including one-row tiles and a partial final tile.
        for (int tile : { 1, 31, 128, 1024 }) {
            if (!qwen35_ocr::project_rows_cpu(matrix, hidden.data(), width, tokens, 2, tile, output) ||
                output.size() != size_t(rows * tokens))
                return fail(__LINE__);
            const bool reuse_on = std::getenv("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE")[0] == '1';
            if (reuse_on) {
                std::vector<float> legacy;
                env("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", "0");
                const bool ok = qwen35_ocr::project_rows_cpu(matrix, hidden.data(), width, tokens, 2, tile, legacy);
                env("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", "1");
                if (!ok || legacy.size() != output.size() ||
                    std::memcmp(legacy.data(), output.data(), output.size() * sizeof(float))) {
                    std::fprintf(stderr, "STREAM_AB_MISMATCH type=%d tile=%d rows=%d tokens=%d legacy_ok=%d\n",
                                 int(matrix->type), tile, rows, tokens, int(ok));
                    return fail(__LINE__);
                }
            }
            for (int t = 0; t < tokens; ++t)
                for (int row = 0; row < rows; ++row) {
                    double expected = 0;
                    for (int column = 0; column < width; ++column) {
                        const int index = row * width + column;
                        const float weight = matrix == f32
                                                 ? static_cast<float *>(matrix->data)[index]
                                                 : ggml_fp16_to_fp32(static_cast<ggml_fp16_t *>(matrix->data)[index]);
                        expected += double(weight) * hidden[t * width + column];
                    }
                    const size_t index = size_t(t) * rows + row;
                    if (std::abs(output[index] - expected) > 1e-5 ||
                        std::abs(output[index] - baseline[0][index]) > 1e-5)
                        return fail(__LINE__);
                }
        }
    }
    // Quantized rows must obey the existing quant gate and reproduce an
    // independently dequantized dot product across tile seams.
    for (auto * matrix : { q8, q4 }) {
        env("CRISPEMBED_QWEN35_QUANT", "0");
        output = { 42 };
        if (qwen35_ocr::project_rows_cpu(matrix, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
            return fail(__LINE__);
        env("CRISPEMBED_QWEN35_QUANT", "1");
        std::vector<float> dequantized(width * rows);
        ggml_get_type_traits(matrix->type)->to_float(matrix->data, dequantized.data(), width * rows);
        std::vector<std::vector<float>> baseline;
        if (!qwen35_ocr::project_cpu({ matrix }, hidden.data(), width, tokens, 2, baseline)) return fail(__LINE__);
        for (int tile : { 1, 31, 128, 1024 }) {
            if (!qwen35_ocr::project_rows_cpu(matrix, hidden.data(), width, tokens, 2, tile, output) ||
                output.size() != size_t(rows * tokens))
                return fail(__LINE__);
            const bool reuse_on = std::getenv("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE")[0] == '1';
            if (reuse_on) {
                std::vector<float> legacy;
                env("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", "0");
                const bool ok = qwen35_ocr::project_rows_cpu(matrix, hidden.data(), width, tokens, 2, tile, legacy);
                env("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", "1");
                if (!ok || legacy.size() != output.size() ||
                    std::memcmp(legacy.data(), output.data(), output.size() * sizeof(float))) {
                    std::fprintf(stderr, "STREAM_AB_MISMATCH type=%d tile=%d rows=%d tokens=%d legacy_ok=%d\n",
                                 int(matrix->type), tile, rows, tokens, int(ok));
                    return fail(__LINE__);
                }
            }
            for (int t = 0; t < tokens; ++t)
                for (int row = 0; row < rows; ++row) {
                    double expected = 0;
                    for (int column = 0; column < width; ++column)
                        expected += double(dequantized[row * width + column]) * hidden[t * width + column];
                    const size_t index = size_t(t) * rows + row;
                    if (std::abs(output[index] - expected) > 1e-4 ||
                        std::abs(output[index] - baseline[0][index]) > 1e-4)
                        return fail(__LINE__);
                }
        }
    }
    for (int tile : { 0, 1025 }) {
        output = { 42 };
        if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, tile, output) || !output.empty())
            return fail(__LINE__);
    }
    output = { 42 };
    if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, 17, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    ggml_tensor invalid = *f32;
    invalid.buffer = nullptr;
    if (qwen35_ocr::project_rows_cpu(&invalid, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    invalid = *f32;
    invalid.type = GGML_TYPE_Q8_0;
    env("CRISPEMBED_QWEN35_QUANT", "1");
    if (qwen35_ocr::project_rows_cpu(&invalid, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    // Failure after earlier valid tiles must not expose partial vocabulary.
    static_cast<float *>(f32->data)[width * 200] = std::numeric_limits<float>::quiet_NaN();
    output = { 42 };
    if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    static_cast<float *>(f32->data)[width * 200] = 0;
    env("CRISPEMBED_QWEN35_IMATRIX", "1");
    if (!qwen35_ocr::set_projection_observer(observe, nullptr) ||
        !qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, 31, output) || calls != 1)
        return fail(__LINE__);
    std::vector<std::vector<float>> ordinary;
    if (!qwen35_ocr::project_cpu({ f32 }, hidden.data(), width, tokens, 2, ordinary) || calls != 2)
        return fail(__LINE__);
    qwen35_ocr::set_projection_observer(throwing_observer, nullptr);
    output = { 42 };
    if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    qwen35_ocr::set_projection_observer(nullptr, nullptr);
    env("CRISPEMBED_QWEN35_OCR", "0");
    if (qwen35_ocr::project_rows_cpu(f32, hidden.data(), width, tokens, 2, 31, output) || !output.empty())
        return fail(__LINE__);
    std::puts("STREAM_PROJECTION_CONTROLS pass=1 f32=1 f16=1 seams=1 independent_dot=1 legacy=1 gates=1 "
              "atomic_failure=1 observer=1");
    return 0;
}
int main() {
    for (const char * mode : { "0", "1" }) {
        env("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", mode);
        const auto started = std::chrono::steady_clock::now();
        if (run()) core_util::clean_exit(1);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::printf("STREAM_GRAPH_REUSE_CONTROLS mode=%s seconds=%.9f timing_is_operator_controls_only=1\n", mode,
                    elapsed);
    }
    core_util::clean_exit(0);
}
