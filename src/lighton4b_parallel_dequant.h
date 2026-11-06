#pragma once
#include "core/env_gate.h"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <thread>

namespace qwen35_ocr {
// Only scheduling changes: each converter receives complete independent blocks.
// False leaves the existing serial conversion path responsible for the output.
inline int lighton4b_dequant_workers(ggml_type type, size_t count, int threads) {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    return 1;
#else
    if ((type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q4_K) || threads < 2 || count < 262144 ||
        !core_env::on("CRISPEMBED_QWEN35_QUANT") || !core_env::on("CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT"))
        return 1;
    return int(std::min({ size_t(threads), size_t(8), count / 65536 }));
#endif
}
inline bool lighton4b_parallel_dequant(ggml_type type, const uint8_t * raw, size_t count, int threads, float * target) {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    (void)type;
    (void)raw;
    (void)count;
    (void)threads;
    (void)target;
    return false;
#else
    const int workers = lighton4b_dequant_workers(type, count, threads);
    if (workers < 2 || !raw || !target) return false;
    const size_t block = size_t(ggml_blck_size(type)), bytes = ggml_type_size(type);
    if (count % block) return false;
    const auto convert = ggml_get_type_traits(type)->to_float;
    if (!convert) return false;
    const size_t blocks = count / block;
    std::array<std::thread, 7> jobs;
    const auto chunk = [&](int index) {
        const size_t begin = blocks * size_t(index) / size_t(workers);
        const size_t end = blocks * size_t(index + 1) / size_t(workers);
        convert(raw + begin * bytes, target + begin * block, int64_t((end - begin) * block));
    };
    bool fallback = false;
    try {
        for (int i = 1; i < workers; ++i) jobs[size_t(i - 1)] = std::thread(chunk, i);
    } catch (...) {
        fallback = true;
    }
    if (!fallback) chunk(0);
    for (auto & job : jobs)
        if (job.joinable()) job.join();
    // No launched writer survives before serial overwrite after resource failure.
    if (fallback) convert(raw, target, int64_t(count));
    return true;
#endif
}
} // namespace qwen35_ocr
