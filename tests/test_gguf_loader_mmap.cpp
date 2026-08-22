// Unit test: core_gguf::load_weights copy path vs the opt-in no-copy mmap path.
// Builds a small GGUF, loads it both ways on the CPU backend (which advertises
// buffer_from_host_ptr), and asserts the tensors are byte-identical and match
// the values written. Validates the no-copy path is actually taken and that
// free_weights() releases the mapping — asked of the kernel, not inferred from
// the call returning.

#include "core/gguf_loader.h"
#include "core/clean_exit.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <climits>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/proc_info.h>
#include <unistd.h>
#elif defined(__linux__)
#include <fstream>
#endif

static const char * kPath = "/tmp/crispembed_test_loader_mmap.gguf";

// The kernel names a region by its resolved path, so the comparison has to be
// made against the same. On macOS /tmp is a symlink to /private/tmp, which is
// enough on its own to make every region look like a stranger's.
static std::string resolved_path(const char * path) {
#if defined(_WIN32)
    return std::string(path);
#else
    char buf[PATH_MAX];
    return realpath(path, buf) ? std::string(buf) : std::string(path);
#endif
}

// How many regions of this process are backed by `path`. The no-copy path keeps
// the weight file mapped for the buffer's lifetime, so this is the exact way to
// ask whether the mapping went away — no footprint threshold, nothing to settle.
// Returns (size_t)-1 where the platform offers no region enumeration, which the
// caller treats as "cannot assert here".
static size_t count_regions_backed_by(const std::string & path) {
#if defined(__APPLE__)
    // PROC_PIDREGIONPATHINFO returns the region and its backing path in one
    // record. proc_regionfilename() alone is not usable: asked about the base of
    // an anonymous region it answers with the file of the next region above,
    // counting an unrelated neighbour as a mapping of this file.
    size_t n = 0;
    uint64_t addr = 0;
    for (;;) {
        struct proc_regionwithpathinfo rpi;
        if (proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, addr, &rpi, sizeof(rpi)) != (int)sizeof(rpi)) break;
        if (rpi.prp_vip.vip_path[0] != '\0' && path == rpi.prp_vip.vip_path) n++;
        const uint64_t next = rpi.prp_prinfo.pri_address + rpi.prp_prinfo.pri_size;
        if (next <= addr) break; // no forward progress; stop rather than spin
        addr = next;
    }
    return n;
#elif defined(__linux__)
    size_t n = 0;
    const size_t len = path.size();
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line))
        if (line.size() > len && line.compare(line.size() - len, len, path) == 0) n++;
    return n;
#else
    (void)path;
    return (size_t)-1;
#endif
}

static float expected(int i) {
    return sinf((float)i * 0.013f) + 0.5f;
}

static bool write_test_gguf(const char * path) {
    ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);
    struct Def {
        const char * name;
        int n0, n1;
    };
    Def defs[] = { { "alpha", 37, 5 }, { "beta.weight", 256, 3 }, { "gamma", 1, 1 } };
    for (auto & d : defs) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.n0, d.n1);
        ggml_set_name(t, d.name);
        float * data = (float *)t->data;
        int64_t n = (int64_t)d.n0 * d.n1;
        for (int64_t i = 0; i < n; i++) data[i] = expected((int)i);
    }
    gguf_context * g = gguf_init_empty();
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) gguf_add_tensor(g, t);
    bool ok = gguf_write_to_file(g, path, /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static std::vector<float> read_tensor(core_gguf::WeightLoad & wl, const char * name) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end() || !it->second) return {};
    ggml_tensor * t = it->second;
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

static int crispembed_test_main() {
    if (!write_test_gguf(kPath)) {
        fprintf(stderr, "FAIL: could not write %s\n", kPath);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "FAIL: cpu backend\n");
        return 1;
    }

    core_gguf::WeightLoad cw, mw;
    if (!core_gguf::load_weights(kPath, backend, "test", cw, /*try_mmap=*/false)) {
        fprintf(stderr, "FAIL: copy load\n");
        return 1;
    }
    if (!core_gguf::load_weights(kPath, backend, "test", mw, /*try_mmap=*/true)) {
        fprintf(stderr, "FAIL: mmap load\n");
        return 1;
    }
    if (!mw.used_mmap) {
        fprintf(stderr, "FAIL: no-copy path not taken (CPU advertises buffer_from_host_ptr)\n");
        return 1;
    }
    printf("no-copy mmap path taken (used_mmap=1)\n");

    const char * names[] = { "alpha", "beta.weight", "gamma" };
    int fails = 0;
    for (const char * nm : names) {
        std::vector<float> a = read_tensor(cw, nm), b = read_tensor(mw, nm);
        if (a.empty() || a.size() != b.size()) {
            fprintf(stderr, "FAIL: %s size\n", nm);
            fails++;
            continue;
        }
        bool ok = true;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i] || fabsf(a[i] - expected((int)i)) > 1e-6f) {
                ok = false;
                break;
            }
        printf("  %-12s %4zu elems  copy==mmap==written: %s\n", nm, a.size(), ok ? "OK" : "MISMATCH");
        if (!ok) fails++;
    }

    // The mapping is keyed to the backend buffer, so free_weights reaches it
    // through release_weight_buffer rather than through mw.mmap_addr. Check the
    // kernel, not the field: a desync between where the region is registered
    // and where it is taken would leave the file mapped and clear the field
    // anyway.
    const std::string kAbs = resolved_path(kPath);
    const size_t mapped_before = count_regions_backed_by(kAbs);
    core_gguf::free_weights(cw);
    core_gguf::free_weights(mw);
    if (mapped_before == (size_t)-1) {
        printf("region enumeration unavailable on this platform — mapping release not asserted\n");
    } else if (mapped_before < 1) {
        // Positive control. Without it the zero below would also be satisfied
        // by a load that never mapped the file in the first place.
        fprintf(stderr, "FAIL: no-copy load left no mapping to release (found %zu regions)\n", mapped_before);
        fails++;
    } else {
        const size_t mapped_after = count_regions_backed_by(kAbs);
        printf("  weight file regions: %zu while loaded, %zu after free_weights\n", mapped_before, mapped_after);
        if (mapped_after != 0) {
            fprintf(stderr, "FAIL: free_weights left %zu mapping(s) of %s\n", mapped_after, kPath);
            fails++;
        }
    }

    // --- Test: release_weight_buffer on a non-mmap (copy-path) buffer ---
    // The copy path maps nothing that outlives the load, so the side map has no
    // entry for this buffer. release_weight_buffer must free it like any other
    // backend buffer, without error.
    {
        core_gguf::WeightLoad copy_wl;
        if (!core_gguf::load_weights(kPath, backend, "test", copy_wl, /*try_mmap=*/false)) {
            fprintf(stderr, "FAIL: copy load for release_weight_buffer test\n");
            fails++;
        } else {
            // Move the buffer out of the WeightLoad, as the 11 model sites do.
            ggml_backend_buffer_t moved_buf = copy_wl.buf;
            copy_wl.buf = nullptr;
            core_gguf::release_weight_buffer(moved_buf);
            if (moved_buf != nullptr) {
                fprintf(stderr, "FAIL: release_weight_buffer did not null the handle (copy path)\n");
                fails++;
            } else {
                printf("  release_weight_buffer(copy-path buf): OK (freed, handle nulled)\n");
            }
            // Clean up the rest of the WeightLoad (ctx, etc.)
            core_gguf::free_weights(copy_wl);
        }
    }

    // --- Test: double-call safety (null handle → no-op) ---
    {
        ggml_backend_buffer_t null_buf = nullptr;
        core_gguf::release_weight_buffer(null_buf);
        if (null_buf != nullptr) {
            fprintf(stderr, "FAIL: release_weight_buffer(nullptr) changed the handle\n");
            fails++;
        } else {
            printf("  release_weight_buffer(nullptr): OK (no-op)\n");
        }
    }

    // --- Test: release_weight_buffer on an mmap buffer moved out of WeightLoad ---
    // Simulates the pattern used by the 11 model sites: move wl.buf into a
    // model struct, let the WeightLoad die, then free via release_weight_buffer.
    {
        core_gguf::WeightLoad mmap_wl;
        if (!core_gguf::load_weights(kPath, backend, "test", mmap_wl, /*try_mmap=*/true)) {
            fprintf(stderr, "FAIL: mmap load for moved-buffer test\n");
            fails++;
        } else if (!mmap_wl.used_mmap) {
            fprintf(stderr, "FAIL: mmap path not taken for moved-buffer test\n");
            fails++;
        } else {
            // Move the buffer out, as models do.
            ggml_backend_buffer_t moved_buf = mmap_wl.buf;
            mmap_wl.buf = nullptr;
            // Clear the WeightLoad's mmap fields to simulate scope exit.
            mmap_wl.mmap_addr = nullptr;
            mmap_wl.mmap_len = 0;
            mmap_wl.used_mmap = false;
            core_gguf::free_weights(mmap_wl); // frees ctx only, buf already moved

            // The mapping should still exist (owned by the buffer via the side map).
            const size_t before = count_regions_backed_by(kAbs);
            core_gguf::release_weight_buffer(moved_buf);
            const size_t after = count_regions_backed_by(kAbs);
            if (moved_buf != nullptr) {
                fprintf(stderr, "FAIL: release_weight_buffer did not null the handle (mmap moved)\n");
                fails++;
            } else if (before == (size_t)-1) {
                printf("  release_weight_buffer(mmap moved buf): OK (freed, region check unavailable)\n");
            } else if (before < 1) {
                fprintf(stderr, "FAIL: moved mmap buffer had no mapping to release (%zu regions)\n", before);
                fails++;
            } else if (after != 0) {
                fprintf(stderr, "FAIL: release_weight_buffer(moved) left %zu mapping(s)\n", after);
                fails++;
            } else {
                printf("  release_weight_buffer(mmap moved buf): OK (regions %zu→%zu)\n", before, after);
            }
        }
    }

    ggml_backend_free(backend);
    remove(kPath);

    if (fails) {
        fprintf(stderr, "FAILED (%d check(s))\n", fails);
        return 1;
    }
    printf("PASS: gguf_loader no-copy mmap == copy\n");
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
