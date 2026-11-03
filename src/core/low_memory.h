// low_memory.h — automatic fallback for optional F32 weight copies.
//
// Some engines make F32 copies of F16/quantized matrices once at init because
// F32 x F32 products are faster on CPU (MiniCPM-OCR vision + graph decoder,
// Unlimited-OCR-2 SAM/CLIP blocks). The copies are pure speed: the engines
// also have a measured non-copy path. On a host that cannot hold the extra
// bytes, the copies would push the process into swap, so the engine checks the
// available physical memory after loading its weights and skips them when
//
//     available < extra_bytes + margin
//
// Per engine, one tri-state environment variable controls this:
//
//     <ENGINE>_LOW_MEMORY unset  automatic check (default)
//     <ENGINE>_LOW_MEMORY=1      always take the low-memory path
//     <ENGINE>_LOW_MEMORY=0      never run the check (copies as configured)
//
// Unknown available memory (unsupported platform, failed query) keeps the
// configured behavior: the check never acts on ignorance.
// CRISPEMBED_RAM_GUARD_AVAILABLE_MB=N (the core/ram_guard.h test hook) also
// overrides the measurement here.
//
// The decision itself is a pure function (decide) and is unit-tested by
// tests/test_low_memory.cpp.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace core_lowmem {

// Headroom kept free beyond the copies themselves (activations, KV, OS).
constexpr uint64_t kDefaultMarginBytes = 1024ull * 1024ull * 1024ull;

enum class mode { automatic, forced_on, disabled };

enum class decision {
    keep_copies,     // enough memory, check disabled, or memory unknown
    skip_forced,     // <ENGINE>_LOW_MEMORY=1
    skip_low_memory, // automatic check: available < extra + margin
};

// Tri-state parse with core_env semantics: unset/empty = automatic, "0" =
// disabled, any other non-empty value = forced on.
inline mode parse_mode(const char * value) {
    if (!value || !*value) return mode::automatic;
    if (std::strcmp(value, "0") == 0) return mode::disabled;
    return mode::forced_on;
}

// available_bytes < 0 means unknown.
inline decision decide(mode m, uint64_t extra_bytes, int64_t available_bytes,
                       uint64_t margin_bytes = kDefaultMarginBytes) {
    if (m == mode::forced_on) return decision::skip_forced;
    if (m == mode::disabled || extra_bytes == 0 || available_bytes < 0) return decision::keep_copies;
    const uint64_t need = extra_bytes + margin_bytes;
    return uint64_t(available_bytes) < need ? decision::skip_low_memory : decision::keep_copies;
}

// Available physical memory in bytes, or -1 when unknown.
inline int64_t available_physical_bytes() {
    if (const char * e = std::getenv("CRISPEMBED_RAM_GUARD_AVAILABLE_MB")) {
        const long long v = std::atoll(e);
        if (v > 0) return int64_t(v) * 1024 * 1024;
    }
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return int64_t(status.ullAvailPhys);
    return -1;
#elif defined(__APPLE__)
    // free + inactive + purgeable: memory the kernel hands out without swapping.
    mach_port_t host = mach_host_self();
    vm_size_t page = 0;
    if (host_page_size(host, &page) != KERN_SUCCESS || page == 0) return -1;
    vm_statistics64_data_t vs;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t) &vs, &count) != KERN_SUCCESS) return -1;
    const uint64_t pages = uint64_t(vs.free_count) + vs.inactive_count + vs.purgeable_count;
    return int64_t(pages * uint64_t(page));
#else
#if defined(__linux__)
    // MemAvailable counts reclaimable page cache; _SC_AVPHYS_PAGES (MemFree) does
    // not and would trigger the fallback on any host with a warm file cache.
    if (FILE * f = std::fopen("/proc/meminfo", "r")) {
        char line[256];
        long long kb = -1;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "MemAvailable:", 13) == 0) {
                kb = std::atoll(line + 13);
                break;
            }
        }
        std::fclose(f);
        if (kb > 0) return int64_t(kb) * 1024;
    }
#endif
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) return int64_t(pages) * int64_t(page);
#endif
    return -1;
#endif
}

// Reads `env_name`, measures memory, and prints one stderr notice when the
// copies are skipped. Returns true when the caller must take the low-memory path.
inline bool skip_f32_copies(const char * engine, const char * env_name, uint64_t extra_bytes,
                            uint64_t margin_bytes = kDefaultMarginBytes) {
    const mode m = parse_mode(env_name ? std::getenv(env_name) : nullptr);
    if (m == mode::forced_on) {
        std::fprintf(stderr, "%s: %s=1, skipping %.0f MiB of F32 weight copies (low-memory path)\n", engine,
                     env_name, double(extra_bytes) / 1048576.0);
        return true;
    }
    if (m == mode::disabled || extra_bytes == 0) return false;
    const int64_t avail = available_physical_bytes();
    if (decide(m, extra_bytes, avail, margin_bytes) != decision::skip_low_memory) return false;
    std::fprintf(stderr,
                 "%s: ~%.0f MiB of memory available, F32 weight copies need ~%.0f MiB + %.0f MiB margin; "
                 "using the low-memory path (same weights, slower). %s=0 disables this check.\n",
                 engine, double(avail) / 1048576.0, double(extra_bytes) / 1048576.0,
                 double(margin_bytes) / 1048576.0, env_name);
    return true;
}

} // namespace core_lowmem
