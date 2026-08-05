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
#include "ggml-cpu.h"                    // ggml_backend_cpu_init() for the `--gpu-backend cpu` short-circuit
#include "metal_pipeline_cache_policy.h" // T18 cap on the ggml-metal MTLBinaryArchive open cost

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace crispasr_gpu_pref {

// The preference is a simple global — set once at process start,
// read many times. Thread-safe via the string copy in get().
inline std::string & pref_storage() {
    static std::string s;
    return s;
}

inline std::mutex & pref_mutex() {
    static std::mutex m;
    return m;
}

} // namespace crispasr_gpu_pref

// Set the GPU backend preference. Call once at startup, before any
// backend init_from_file. Empty string = auto.
inline void crispasr_set_gpu_backend_pref(const char * name) {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    crispasr_gpu_pref::pref_storage() = name ? name : "";
}

inline std::string crispasr_get_gpu_backend_pref() {
    std::lock_guard<std::mutex> lock(crispasr_gpu_pref::pref_mutex());
    return crispasr_gpu_pref::pref_storage();
}

// Case-insensitive prefix check: does `haystack` start with `needle`?
inline bool ci_starts_with(const char * haystack, const char * needle) {
    for (; *needle; ++haystack, ++needle) {
        if (!*haystack) return false;
        if (tolower((unsigned char)*haystack) != tolower((unsigned char)*needle)) return false;
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

    // G4 (extends T18): bound the ggml-metal MTLBinaryArchive open cost for
    // EVERY lane that reaches a GPU device through this helper — OCR, VLM, SR,
    // NER, denoise — not just the embed CLI. The archive open costs ~1 ms/MB
    // (683 MB observed = ~680 ms of fixed init) and a one-shot CLI never
    // writes an entry back (clean_exit skips the serialising destructor).
    // Same guard as the embed path: skipped when the preference is "cpu",
    // because then no GPU device is created and the diagnostic would fire
    // spuriously. apply() is idempotent, so the embed path's own earlier call
    // is unaffected. CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB=0 restores the
    // pre-T18 behaviour for every lane at once.
    {
        const bool pref_is_cpu = !pref.empty() && pref.size() <= 3 && ci_starts_with("cpu", pref.c_str());
        if (!pref_is_cpu) core_metal_cache::apply();
    }

    // T18: `--gpu-backend cpu` used to fall THROUGH to the GPU. The loops below
    // only ever consider GPU/iGPU devices, so "cpu" matched nothing, printed the
    // "no matching GPU device found" warning and handed back
    // ggml_backend_init_best() — i.e. Metal. On an M1 that silently cost ~680 ms
    // of Metal device init on a flag whose whole purpose was to avoid it (found
    // and reported by T14, which had to add its own engine-local DS2_FORCE_CPU=1
    // because this flag did not work).
    // CRISPEMBED_GPU_PREF_CPU_LEGACY=1 restores the old fall-through for A/B.
    if (!pref.empty() && ci_starts_with("cpu", pref.c_str()) && pref.size() <= 3) {
        const char * legacy = std::getenv("CRISPEMBED_GPU_PREF_CPU_LEGACY");
        if (!(legacy && legacy[0] && legacy[0] != '0')) {
            // ggml_backend_cpu_init(), NOT ggml_backend_dev_by_type(...CPU):
            // the registry lookup enumerates every registered device, which
            // constructs the Metal device as a side effect (measured: it still
            // ran ggml_metal_device_init and cost ~29 ms). The direct
            // constructor touches no registry, so nothing GPU is created.
            ggml_backend_t cpu = ggml_backend_cpu_init();
            if (cpu) {
                fprintf(stderr, "%s: --gpu-backend cpu — using the CPU backend, no GPU device initialised\n", __func__);
                return cpu;
            }
        }
    }

    // ggml names the Apple backend "MTL" (registry) / "MTL0" (device), so the
    // natural `--gpu-backend metal` would never prefix-match. Alias it.
    if (pref.size() >= 3 && ci_starts_with("metal", pref.c_str())) pref = "mtl";

    if (!pref.empty()) {
        // Iterate all registered devices and find the first GPU whose
        // name starts with the preference string.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
            if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            const char * dev_name = ggml_backend_dev_name(dev);
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
            const char * reg_name = ggml_backend_reg_name(reg);
            if (!ci_starts_with(reg_name, pref.c_str())) continue;
            // Found the registry — pick the first GPU device from it.
            for (size_t j = 0; j < ggml_backend_reg_dev_count(reg); ++j) {
                ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, j);
                enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
                if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
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

// ---------------------------------------------------------------------------
// Optional process-shared GPU backend (opt-in: CRISPEMBED_SHARED_GPU_BACKEND=1)
//
// Each engine calling crispasr_init_gpu_backend() gets its own
// ggml_backend_dev_init, so N GPU engines in one process pay N Metal inits.
// That init is largely blocking wait rather than compute and degrades badly
// under contention (an EasyOCR stage measured load=36 s then 87 s on two
// consecutive runs of the same command while the process used 4 s of CPU).
//
// MEASURED STATUS 2026-08-02: this currently buys nothing, because after the
// detector loaders moved to the CPU backend no OCR lane initialises Metal more
// than once (tesseract 0, EasyOCR 1, PP-OCRv6 1 -- DBNet is CPU by default
// since Metal conv is slower for it). It is kept, working and gated off,
// because the duplication returns the moment two GPU-resident engines run in
// one process -- a VLM stage beside a recognizer, the detector graph being
// promoted, or batch/server use where several engines are resident at once.
//
// HAZARD if you enable it: a ggml_backend_t is a device+queue handle. Paths
// that recognize lines in parallel (CRISPEMBED_TESSERACT_WORKERS) would drive
// one handle from several threads, which ggml does not promise is safe. Only
// engines whose free path also goes through crispasr_free_gpu_backend() may use
// this, or the first engine to shut down frees the backend out from under its
// peers.
inline bool crispasr_shared_gpu_backend_enabled() {
    static const bool on = std::getenv("CRISPEMBED_SHARED_GPU_BACKEND") != nullptr;
    return on;
}

inline ggml_backend_t & crispasr_shared_gpu_backend_slot() {
    static ggml_backend_t shared = nullptr;
    return shared;
}

// Shared backend when enabled, otherwise a fresh one exactly as
// crispasr_init_gpu_backend() would return.
inline ggml_backend_t crispasr_init_gpu_backend_shared() {
    if (!crispasr_shared_gpu_backend_enabled()) return crispasr_init_gpu_backend();
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    ggml_backend_t & slot = crispasr_shared_gpu_backend_slot();
    if (!slot) slot = crispasr_init_gpu_backend();
    return slot;
}

// Free a backend from either helper. The shared instance is deliberately never
// freed -- it outlives every engine and process teardown reclaims it.
inline void crispasr_free_gpu_backend(ggml_backend_t backend) {
    if (!backend) return;
    if (backend == crispasr_shared_gpu_backend_slot()) return;
    ggml_backend_free(backend);
}
