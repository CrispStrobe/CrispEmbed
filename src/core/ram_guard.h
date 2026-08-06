// RAM preflight for the large-VLM engines (SubtitleEdit PR-13238 field
// report, 2026-08-06): deepseek-ocr2 on a 4 GB host froze the desktop 3/3
// before producing ANY output — the model + activations exceed physical
// memory and the host swap-thrashes instead of failing. The engines had no
// memory check at all. This helper refuses (default) or warns before the
// weight load, with the numbers printed so the user knows WHY and how to
// override.
//
//   CRISPEMBED_RAM_GUARD=0     disable entirely (old behavior)
//   CRISPEMBED_RAM_GUARD=warn  print the warning but continue
//   CRISPEMBED_RAM_GUARD_AVAILABLE_MB=N   test hook: pretend N MiB available
//
// "Required" is estimated as file_size * factor + headroom: the weights are
// resident in full, and factor/headroom cover dequant copies, activations
// and KV. Estimates err permissive — the guard exists to catch the
// hopeless-by-2x case that freezes a host, not to police tight fits.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

namespace core_ram {

inline long long available_mb() {
    if (const char * e = std::getenv("CRISPEMBED_RAM_GUARD_AVAILABLE_MB")) {
        const long long v = atoll(e);
        if (v > 0) return v;
    }
#if defined(__APPLE__)
    // free + inactive + purgeable approximates reclaimable-on-demand memory.
    mach_port_t host = mach_host_self();
    vm_size_t page = 0;
    host_page_size(host, &page);
    vm_statistics64_data_t vs;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vs, &count) == KERN_SUCCESS) {
        const unsigned long long pages = (unsigned long long)vs.free_count + vs.inactive_count + vs.purgeable_count;
        return (long long)(pages * (unsigned long long)page / (1024ull * 1024ull));
    }
    return -1;
#elif defined(__linux__)
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long long kb = -1;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = atoll(line + 13);
            break;
        }
    }
    std::fclose(f);
    return kb > 0 ? kb / 1024 : -1;
#else
    return -1; // unknown platform: guard becomes a no-op
#endif
}

// Returns true when loading may proceed. `engine` names the caller in the
// message; factor/headroom_mb size the estimate for that engine's runtime.
inline bool preflight(const char * engine, const char * model_path, double factor = 1.3, long long headroom_mb = 1024) {
    const char * mode = std::getenv("CRISPEMBED_RAM_GUARD");
    if (mode && mode[0] == '0' && mode[1] == '\0') return true;
    struct stat st {};
    if (!model_path || stat(model_path, &st) != 0 || st.st_size <= 0) return true;
    const long long avail = available_mb();
    if (avail < 0) return true; // could not measure: never block on ignorance
    const long long file_mb = (long long)(st.st_size / (1024ll * 1024ll));
    const long long need_mb = (long long)(file_mb * factor) + headroom_mb;
    if (avail >= need_mb) return true;
    const bool warn_only = mode && std::strncmp(mode, "warn", 4) == 0;
    std::fprintf(stderr,
                 "%s: %s memory preflight: model %s is %lld MiB; estimated requirement ~%lld MiB "
                 "(weights x%.1f + %lld MiB activations/KV) but only ~%lld MiB of RAM is available. "
                 "Loading anyway would likely swap-thrash or freeze this machine. "
                 "%s (CRISPEMBED_RAM_GUARD=0 disables this check, =warn continues with a warning.)\n",
                 engine, warn_only ? "WARNING —" : "REFUSING to load —", model_path, file_mb, need_mb, factor,
                 headroom_mb, avail, warn_only ? "Continuing because CRISPEMBED_RAM_GUARD=warn." : "");
    return warn_only;
}

} // namespace core_ram
