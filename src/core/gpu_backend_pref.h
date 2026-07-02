#pragma once
// gpu_backend_pref.h — process-global GPU backend preference
//
// Issue #214: `--gpu-backend vulkan` was ignored because every backend
// called `ggml_backend_init_best()` which unconditionally picks CUDA
// over Vulkan when both are compiled in. This header provides:
//
//   crispasr_set_gpu_backend_pref("vulkan")  — set once at startup
//   crispasr_init_gpu_backend()              — drop-in replacement for
//                                              ggml_backend_init_best()
//
// The preference is matched against ggml backend registry names
// (case-insensitive). Common values: "cuda", "vulkan", "metal", "cpu".
// Empty or null = auto (same as ggml_backend_init_best).

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace crispasr_gpu_pref {

// The preference is a simple global — set once at process start,
// read many times. Thread-safe via the string copy in get().
inline std::string& pref_storage() {
    static std::string s;
    return s;
}

inline std::mutex& pref_mutex() {
    static std::mutex m;
    return m;
}

} // namespace crispasr_gpu_pref

// Set the GPU backend preference. Call once at startup, before any
// backend init_from_file. Empty string = auto.
inline void crispasr_set_gpu_backend_pref(const char* name) {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    crispasr_gpu_pref::pref_storage() = name ? name : "";
}

inline std::string crispasr_get_gpu_backend_pref() {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    return crispasr_gpu_pref::pref_storage();
}

// Case-insensitive prefix check: does `haystack` start with `needle`?
inline bool ci_starts_with(const char* haystack, const char* needle) {
    for (; *needle; ++haystack, ++needle) {
        if (!*haystack)
            return false;
        if (tolower((unsigned char)*haystack) != tolower((unsigned char)*needle))
            return false;
    }
    return true;
}

// Drop-in replacement for ggml_backend_init_best().
// If a gpu_backend preference is set, iterate registered devices and
// pick the first GPU/iGPU device whose name starts with the preference
// (e.g. "vulkan" matches "Vulkan0", "Vulkan1", …).
// Falls back to ggml_backend_init_best() when no preference is set or
// the preferred backend isn't found.
inline ggml_backend_t crispasr_init_gpu_backend() {
    std::string pref = crispasr_get_gpu_backend_pref();

    if (!pref.empty()) {
        // Iterate all registered devices and find the first GPU whose
        // name starts with the preference string.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
            if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU)
                continue;
            const char* dev_name = ggml_backend_dev_name(dev);
            if (ci_starts_with(dev_name, pref.c_str())) {
                ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
                if (result) {
                    fprintf(stderr, "%s: using preferred GPU backend: %s\n", __func__, dev_name);
                    return result;
                }
                fprintf(stderr, "%s: preferred GPU device '%s' failed to init, trying fallback\n", __func__, dev_name);
            }
        }

        // Also try matching against the registry (backend library) name,
        // e.g. the user writes "vulkan" and the registry name is "Vulkan".
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            const char* reg_name = ggml_backend_reg_name(reg);
            if (!ci_starts_with(reg_name, pref.c_str()))
                continue;
            // Found the registry — pick the first GPU device from it.
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
                enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
                if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU)
                    continue;
                ggml_backend_t result = ggml_backend_dev_init(dev, nullptr);
                if (result) {
                    fprintf(stderr, "%s: using preferred GPU backend: %s (via registry '%s')\n", __func__,
                            ggml_backend_dev_name(dev), reg_name);
                    return result;
                }
            }
        }

        fprintf(stderr,
                "%s: WARNING: --gpu-backend '%s' requested but no matching "
                "GPU device found, falling back to auto\n",
                __func__, pref.c_str());
    }

    return ggml_backend_init_best();
}
