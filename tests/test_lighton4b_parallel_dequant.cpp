#include "lighton4b_parallel_dequant.h"
#include "qwen35_math.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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
static int run() {
    using namespace qwen35_ocr;
    const char * flag = "CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT";
    gate("CRISPEMBED_QWEN35_QUANT", "1");
    auto * ctx = ggml_init({ 16384, nullptr, true });
    auto * backend = ggml_backend_cpu_init();
    if (!ctx || !backend) return 1;
    auto * q8 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 1025);
    auto * q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, 256, 1025);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) return 1;
    const size_t count = 256 * 1025;
    std::vector<float> source(count), serial(count), parallel(count + 2, -12345);
    for (size_t i = 0; i < count; ++i) source[i] = float(int(i % 4001) - 2000) / 127;
    int checks = 0;
    for (auto * tensor : { q8, q4 }) {
        if (ggml_quantize_chunk(tensor->type, source.data(), tensor->data, 0, 1025, 256, nullptr) !=
            ggml_nbytes(tensor))
            return 2;
        gate(flag, "0");
        if (!tensor_floats(tensor, serial.data(), 1)) return 3;
        for (const char * value : { static_cast<const char *>(nullptr), "", "0", "1" }) {
            gate(flag, value);
            for (int threads : { 1, 2, 3, 7, 8, 16 }) {
                if (!tensor_floats(tensor, parallel.data() + 1, threads) ||
                    std::memcmp(serial.data(), parallel.data() + 1, count * sizeof(float)) ||
                    parallel.front() != -12345 || parallel.back() != -12345)
                    return 4;
                const int workers = lighton4b_dequant_workers(tensor->type, count, threads);
                if ((!value || !*value || *value == '0' || threads == 1) ? workers != 1 : workers < 2 || workers > 8)
                    return 5;
                ++checks;
            }
        }
    }
    gate(flag, "1");
    gate("CRISPEMBED_QWEN35_QUANT", "0");
    if (lighton4b_dequant_workers(GGML_TYPE_Q8_0, count, 8) != 1 || tensor_floats(q8, parallel.data(), 8)) return 6;
    gate("CRISPEMBED_QWEN35_QUANT", "1");
    for (auto type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q6_K })
        if (lighton4b_dequant_workers(type, count, 8) != 1) return 7;
    if (lighton4b_dequant_workers(GGML_TYPE_Q4_K, 65536, 8) != 1 ||
        lighton4b_parallel_dequant(GGML_TYPE_Q4_K, static_cast<uint8_t *>(q4->data), count - 1, 8, parallel.data()))
        return 8;
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::printf("PARALLEL_DEQUANT passed=1 bit_exact_cases=%d boundary_guards=1 legacy_quant_gate=1 model_payload=0\n",
                checks);
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
