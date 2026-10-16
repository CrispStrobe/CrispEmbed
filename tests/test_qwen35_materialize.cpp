#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "qwen35_math.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>

using namespace qwen35_ocr;
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_PARALLEL_F16", value);
#else
    setenv("CRISPEMBED_QWEN35_PARALLEL_F16", value, 1);
#endif
}
static bool equal(ggml_tensor * tensor, int threads, bool finite) {
    std::vector<float> serial(size_t(ggml_nelements(tensor))), parallel(serial.size());
    gate("0");
    const bool a = tensor_floats(tensor, serial.data(), threads);
    gate("1");
    const bool b = tensor_floats(tensor, parallel.data(), threads);
    return a == finite && b == finite && !std::memcmp(serial.data(), parallel.data(), serial.size() * sizeof(float));
}
static int controls() {
    gate("0");
    if (f16_conversion_workers(524289, 4) != 1) return 1;
    gate("");
    if (f16_conversion_workers(524289, 4) != 1) return 1;
    gate("1");
    if (f16_conversion_workers(262143, 4) != 1 || f16_conversion_workers(524289, 1) != 1 ||
        f16_conversion_workers(524289, 0) != 1 || f16_conversion_workers(524289, 4) != 4 ||
        f16_conversion_workers(524289, 256) != 8)
        return 1;
    using context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    context_owner ctx(ggml_init({ 16 * 1024, nullptr, true }), ggml_free);
    if (!ctx) return 1;
    auto * finite = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 524289);
    auto * all_bits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 262161);
    auto * small = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 65536);
    auto * f32 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 262161);
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) return 1;
    buffer_owner buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) return 1;
    for (size_t i = 0; i < size_t(ggml_nelements(finite)); ++i)
        static_cast<uint16_t *>(finite->data)[i] = uint16_t((i % 0x7c00) | ((i & 1) ? 0x8000 : 0));
    for (size_t i = 0; i < size_t(ggml_nelements(all_bits)); ++i)
        static_cast<uint16_t *>(all_bits->data)[i] = uint16_t(i);
    for (size_t i = 0; i < size_t(ggml_nelements(small)); ++i)
        static_cast<uint16_t *>(small->data)[i] = uint16_t(i % 0x7c00);
    for (size_t i = 0; i < size_t(ggml_nelements(f32)); ++i)
        static_cast<float *>(f32->data)[i] = float(int(i % 19) - 9) / 8;
    for (int threads : { 1, 2, 4, 256 })
        if (!equal(finite, threads, true) || !equal(all_bits, threads, false) || !equal(small, threads, true) ||
            !equal(f32, threads, true))
            return 1;
    static_cast<float *>(f32->data)[262160] = INFINITY;
    if (!equal(f32, 4, false) || tensor_floats(nullptr, nullptr, 4) || tensor_floats(finite, nullptr, 4)) return 1;
    gate("0");
    std::puts("MATERIALIZE_CONTROLS pass=1 literal0=1 all_half_bits=1 finite_rejection=1 odd_tail=1 worker_bound=1 "
              "small=1 single=1 f32=1");
    return 0;
}
static int checkpoint(const char * model, const char * destination) {
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights(model, backend.get(), "qwen35-materialize", loaded)) return 2;
    struct release {
        core_gguf::WeightLoad & weights;
        ~release() { core_gguf::free_weights(weights); }
    } cleanup{ loaded };
    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(destination, "wb"), std::fclose);
    if (!file) return 2;
    std::map<std::string, ggml_tensor *> ordered(loaded.tensors.begin(), loaded.tensors.end());
    uint64_t offset = 0;
    for (const auto & entry : ordered) {
        auto * tensor = entry.second;
        const size_t count = size_t(ggml_nelements(tensor));
        std::vector<float> serial(count), parallel(count);
        gate("0");
        const bool a = tensor_floats(tensor, serial.data(), 4);
        gate("1");
        const bool b = tensor_floats(tensor, parallel.data(), 4);
        const bool exact = !std::memcmp(serial.data(), parallel.data(), count * sizeof(float));
        double square = 0;
        float magnitude = 0, error = 0;
        for (size_t i = 0; i < count; ++i) {
            magnitude = std::max(magnitude, std::fabs(serial[i]));
            error = std::max(error, std::fabs(serial[i] - parallel[i]));
            square += double(serial[i]) * serial[i];
        }
        const int workers = tensor->type == GGML_TYPE_F16 ? f16_conversion_workers(count, 4) : 1;
        std::printf("MATERIALIZE_MATRIX tensor=%s elements=%zu exact=%d finite=%d workers=%d max_abs=%.9g "
                    "magnitude=%.9g norm=%.17g offset=%llu\n",
                    entry.first.c_str(), count, int(exact), int(a && b), workers, error, magnitude, std::sqrt(square),
                    static_cast<unsigned long long>(offset));
        if (!a || !b || !exact || std::fwrite(parallel.data(), sizeof(float), count, file.get()) != count) return 1;
        offset += count * sizeof(float);
    }
    gate("0");
    if (std::fflush(file.get())) return 1;
    std::printf("MATERIALIZE_COMPLETE tensors=%zu bytes=%llu\n", ordered.size(),
                static_cast<unsigned long long>(offset));
    return 0;
}
static int run(int argc, char ** argv) {
    if (argc == 1) return controls();
    if (argc == 3) return checkpoint(argv[1], argv[2]);
    return 2;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
