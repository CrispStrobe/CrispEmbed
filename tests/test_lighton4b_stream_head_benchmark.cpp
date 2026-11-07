#include "core/clean_exit.h"
#include "qwen35_math.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static bool read_exact(const char * path, void * data, size_t bytes) {
    std::ifstream in(path, std::ios::binary);
    return bool(in.read(static_cast<char *>(data), bytes)) && in.peek() == EOF;
}
static int run(int argc, char ** argv) {
    // Raw quantized head only, never a full model or a converted F32 head.
    if (argc != 7) return 2;
    const std::string type_name = argv[1];
    const ggml_type type = type_name == "q8_0" ? GGML_TYPE_Q8_0 : type_name == "q4_k" ? GGML_TYPE_Q4_K : GGML_TYPE_Q6_K;
    if (type_name != "q8_0" && type_name != "q4_k" && type_name != "q6_k") return 2;
    const int width = std::stoi(argv[2]), rows = std::stoi(argv[3]);
    if (width < 256 || width > 4096 || width % 256 || rows < 257 || rows > 248320) return 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1024 * 1024, nullptr, true }), ggml_free);
    if (!ctx) return 1;
    auto * weight = ggml_new_tensor_2d(ctx.get(), type, width, rows);
    if (ggml_nbytes(weight) > 1024ull * 1024 * 1024) return 2;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) return 1;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer || !read_exact(argv[4], weight->data, ggml_nbytes(weight))) return 1;
    std::vector<float> hidden(width), output, baseline;
    if (!read_exact(argv[5], hidden.data(), width * sizeof(float))) return 1;
    for (const char * name :
         { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_QWEN35_STREAM_PROJECTION",
           "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_LIGHTON4B_Q6_K" })
        gate(name, "1");
    int index = 0;
    for (const char * mode : { "0", "1", "1", "0" }) {
        gate("CRISPEMBED_LIGHTON4B_STREAM_GRAPH_REUSE", mode);
        const auto start = std::chrono::steady_clock::now();
        if (!qwen35_ocr::project_rows_cpu(weight, hidden.data(), width, 1, 2, 256, output)) return 1;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (index == 0) baseline = output;
        if (output.size() != baseline.size() ||
            std::memcmp(output.data(), baseline.data(), output.size() * sizeof(float)))
            return 1;
        const auto path = std::string(argv[6]) + "." + std::to_string(index) + ".f32";
        std::ofstream out(path, std::ios::binary);
        if (!out.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float))) return 1;
        std::printf(
            "STREAM_HEAD_AB index=%d mode=%s seconds=%.9f rows=%d width=%d type=%s bit_exact=1 threads=2 tile=256\n",
            index++, mode, seconds, rows, width, type_name.c_str());
        std::fflush(stdout);
    }
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
