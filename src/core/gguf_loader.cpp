// src/core/gguf_loader.cpp — implementation of core_gguf:: helpers.
// See gguf_loader.h for the interface contract.

#include "gguf_loader.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ── Metal residency-set teardown guard (ggml v0.10.0 regression) ────────────
// ggml v0.10.0 (submodule bump 8be60f83) added Metal "residency sets": a GPU
// keep-alive cache (default 180 s, background heartbeat thread) plus a hard
// teardown assert `GGML_ASSERT([rsets->data count] == 0)` in
// ggml_metal_device_free(). The device is a process-global freed by a C++ static
// destructor at exit; any Metal buffer still registered then aborts the process
// (SIGABRT / exit 134) AFTER results are already printed — turning passing
// one-shot CLI and test-*-diff runs into spurious "signal 6" failures.
//
// The residency cache only benefits a long-lived process (the server); a
// one-shot CLI/test run is a fresh process, so it buys nothing there while
// adding this crash. Disable it by default via ggml's own kill-switch so no
// entry point can abort at exit; a long-lived host opts back in with
// CRISPEMBED_METAL_RESIDENCY=1. This constructor runs at library load, before
// main() and thus before any ggml Metal device init. (The complementary
// "free every Metal backend before exit" fixes let residency be re-enabled
// safely.) See memory ggml-8be60f-sched-teardown-asserts.
#if defined(__APPLE__)
__attribute__((constructor)) static void crispembed_metal_residency_default(void) {
    if (!getenv("CRISPEMBED_METAL_RESIDENCY")) {
        setenv("GGML_METAL_NO_RESIDENCY", "1", /*overwrite=*/0);
    }
}
#endif

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata
// ---------------------------------------------------------------------------

gguf_context * open_metadata(const char * path) {
    gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "core_gguf: failed to open '%s' for metadata read\n", path);
    }
    return g;
}

void free_metadata(gguf_context * gctx) {
    if (gctx) gguf_free(gctx);
}

// Type-checked scalar readers. The GGUF format stores types explicitly so
// we can validate; if the file has a mismatched type the reader silently
// returns the default rather than crashing, matching the existing inline
// helpers in each model.

uint32_t kv_u32(gguf_context * gctx, const char * key, uint32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_UINT32:
        return gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT32:
        return (uint32_t)gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT64:
        return (uint32_t)gguf_get_val_u64(gctx, k);
    case GGUF_TYPE_INT64:
        return (uint32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT16:
        return gguf_get_val_u16(gctx, k);
    case GGUF_TYPE_INT16:
        return (uint32_t)gguf_get_val_i16(gctx, k);
    case GGUF_TYPE_UINT8:
        return gguf_get_val_u8(gctx, k);
    case GGUF_TYPE_INT8:
        return (uint32_t)gguf_get_val_i8(gctx, k);
    default:
        return default_val;
    }
}

int32_t kv_i32(gguf_context * gctx, const char * key, int32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    switch (t) {
    case GGUF_TYPE_INT32:
        return gguf_get_val_i32(gctx, k);
    case GGUF_TYPE_UINT32:
        return (int32_t)gguf_get_val_u32(gctx, k);
    case GGUF_TYPE_INT64:
        return (int32_t)gguf_get_val_i64(gctx, k);
    case GGUF_TYPE_UINT64:
        return (int32_t)gguf_get_val_u64(gctx, k);
    default:
        return default_val;
    }
}

float kv_f32(gguf_context * gctx, const char * key, float default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    if (t == GGUF_TYPE_FLOAT32) return gguf_get_val_f32(gctx, k);
    if (t == GGUF_TYPE_FLOAT64) return (float)gguf_get_val_f64(gctx, k);
    return default_val;
}

bool kv_bool(gguf_context * gctx, const char * key, bool default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL) return default_val;
    return gguf_get_val_bool(gctx, k);
}

std::string kv_str(gguf_context * gctx, const char * key, const char * default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return default_val ? default_val : "";
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING) return default_val ? default_val : "";
    const char * s = gguf_get_val_str(gctx, k);
    return s ? std::string(s) : std::string(default_val ? default_val : "");
}

std::vector<std::string> kv_str_array(gguf_context * gctx, const char * key) {
    std::vector<std::string> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING) return out;
    const int n = gguf_get_arr_n(gctx, k);
    out.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(gctx, k, i));
    }
    return out;
}

std::vector<float> kv_f32_array(gguf_context * gctx, const char * key) {
    std::vector<float> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_FLOAT32) return out;
    const size_t n = gguf_get_arr_n(gctx, k);
    const float * data = (const float *)gguf_get_arr_data(gctx, k);
    out.assign(data, data + n);
    return out;
}

std::vector<int> kv_i32_array(gguf_context * gctx, const char * key) {
    std::vector<int> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0) return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY) return out;
    const int n = gguf_get_arr_n(gctx, k);
    const void * data = gguf_get_arr_data(gctx, k);
    auto arr_type = gguf_get_arr_type(gctx, k);
    out.resize(n);
    if (arr_type == GGUF_TYPE_INT32) {
        memcpy(out.data(), data, n * sizeof(int32_t));
    } else if (arr_type == GGUF_TYPE_UINT32) {
        const uint32_t * p = (const uint32_t *)data;
        for (int i = 0; i < n; i++) out[i] = (int)p[i];
    } else if (arr_type == GGUF_TYPE_INT64) {
        const int64_t * p = (const int64_t *)data;
        for (int i = 0; i < n; i++) out[i] = (int)p[i];
    } else {
        out.clear();
    }
    return out;
}

std::vector<uint8_t> kv_u8_array(gguf_context * gctx, const char * key) {
    std::vector<uint8_t> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0 || gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gctx, k) != GGUF_TYPE_UINT8)
        return out;
    const size_t n = gguf_get_arr_n(gctx, k);
    const auto * data = (const uint8_t *)gguf_get_arr_data(gctx, k);
    out.assign(data, data + n);
    return out;
}

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

namespace {

// Platform unmap, shared by MappedFile's destructor and release_weight_buffer()
// (the no-copy path transfers the mapping to the backend buffer, which unmaps
// when that buffer is released).
void core_unmap(void * base, size_t size) {
    if (!base) return;
#if defined(__EMSCRIPTEN__)
    (void)size;
#elif defined(_WIN32)
    (void)size;
    UnmapViewOfFile(base);
#else
    ::munmap(base, size);
#endif
}

// Which host mapping belongs to which backend buffer.
//
// On the no-copy path the backend buffer is a view onto pages the backend does
// not own: ggml_backend_dev_buffer_from_host_ptr has no deallocator parameter,
// so freeing the buffer releases the view and nothing else. WeightLoad carries
// mmap_addr/mmap_len for the caller that keeps the whole struct, but a caller
// that moves `buf` into its model and lets the WeightLoad die drops the only
// record of the mapping — and eleven of the models in src/ tear down that way.
// Keying the region to the buffer instead means the mapping is released by
// whoever releases the buffer, whichever of the two shapes the caller uses.
struct mmap_region {
    void * base = nullptr;
    size_t size = 0;
};

std::mutex g_buf_mmap_mu;
std::map<ggml_backend_buffer_t, mmap_region> g_buf_mmap;

void register_buf_mmap(ggml_backend_buffer_t buf, void * base, size_t size) {
    std::lock_guard<std::mutex> lk(g_buf_mmap_mu);
    g_buf_mmap[buf] = { base, size };
}

// Look the region up and remove the entry in one critical section. A lookup
// followed by a separate erase would let a second release of the same buffer
// read the entry before the first erased it and unmap twice; it would also
// race a concurrent load whose fresh buffer landed on the same address after
// the free. A default-constructed region means no entry, which is the ordinary
// case — the copy path maps nothing that outlives the load.
mmap_region take_buf_mmap(ggml_backend_buffer_t buf) {
    std::lock_guard<std::mutex> lk(g_buf_mmap_mu);
    auto it = g_buf_mmap.find(buf);
    if (it == g_buf_mmap.end()) return mmap_region{};
    const mmap_region r = it->second;
    g_buf_mmap.erase(it);
    return r;
}

// Read a file slice into a backend tensor. Uses mmap on POSIX; falls back
// to pread/lseek+read when mmap is unavailable (rare in practice).
//
// On POSIX the mmap lives for the duration of one load call — we copy via
// ggml_backend_tensor_set then unmap. No mmap persists past load_weights()
// UNLESS release() is called (the no-copy path keeps it alive in WeightLoad).
struct MappedFile {
    int fd = -1;
    void * base = nullptr;
    size_t size = 0;
    bool ok = false;

    explicit MappedFile(const char * path) {
#if defined(__EMSCRIPTEN__)
        // Emscripten MEMFS: skip mmap, fall through to fread path.
        (void)path;
        return;
#elif defined(_WIN32)
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) return;
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hFile, &fsize)) {
            CloseHandle(hFile);
            return;
        }
        size = (size_t)fsize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(hFile);
        if (!hMap) return;
        base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
        if (!base) return;
        ok = true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return;
        struct stat st;
        if (fstat(fd, &st) != 0) {
            ::close(fd);
            fd = -1;
            return;
        }
        size = (size_t)st.st_size;
        base = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        fd = -1;
        if (base == MAP_FAILED) {
            base = nullptr;
            return;
        }
        // Cold load is dominated by per-tensor page faults (2000+ small reads
        // with no read-ahead). Hint sequential access and kick off an async
        // read-ahead of the whole file so the copy loop streams instead of
        // stalling page-by-page. Advisory — ignore failures.
#if defined(MADV_SEQUENTIAL)
        ::madvise(base, size, MADV_SEQUENTIAL);
#endif
#if defined(MADV_WILLNEED)
        ::madvise(base, size, MADV_WILLNEED);
#endif
        ok = true;
#endif
    }
    ~MappedFile() { core_unmap(base, size); }
    // Transfer ownership of the mapping out (the no-copy path stores it in
    // WeightLoad). After release() the destructor will not unmap.
    void release() {
        base = nullptr;
        size = 0;
    }
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
};

} // namespace

bool load_weights(const char * path, ggml_backend_t backend, const char * model_tag, WeightLoad & out, bool try_mmap) {
    const char * tag = model_tag ? model_tag : "core_gguf";

    gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&out.ctx };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx) gguf_free(gctx);
        return false;
    }

    const size_t data_off = gguf_get_data_offset(gctx);

    // --- No-copy mmap path (opt-in) ----------------------------------------
    // Point the backend buffer directly at the mmap'd file (no 2.x GB copy,
    // half the resident memory). Only when the device advertises
    // buffer_from_host_ptr (Metal/CPU unified memory); otherwise fall through.
    if (try_mmap) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        ggml_backend_dev_props props{};
        if (dev) ggml_backend_dev_get_props(dev, &props);
        if (dev && props.caps.buffer_from_host_ptr) {
            MappedFile mf(path);
            if (mf.ok && mf.size > data_off) {
                size_t max_ts = 0;
                for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t))
                    max_ts = (std::max)(max_ts, ggml_nbytes(t));
                void * host_base = (char *)mf.base + data_off;
                ggml_backend_buffer_t buf =
                    ggml_backend_dev_buffer_from_host_ptr(dev, host_base, mf.size - data_off, max_ts);
                bool ok = (buf != nullptr);
                if (ok) {
                    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                        out.tensors[ggml_get_name(t)] = t;
                        const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                        if (tid < 0) continue;
                        const size_t off = gguf_get_tensor_offset(gctx, tid);
                        // Guard against a truncated/corrupt file: the tensor data
                        // must lie fully within the mapping, else host_base+off is
                        // out of bounds and the backend read segfaults.
                        if (data_off + off + ggml_nbytes(t) > mf.size) {
                            fprintf(stderr,
                                    "%s: truncated/corrupt GGUF '%s' — tensor '%s' "
                                    "extends past EOF (need %zu, file %zu)\n",
                                    tag, path, ggml_get_name(t), (size_t)(data_off + off + ggml_nbytes(t)), mf.size);
                            ok = false;
                            break;
                        }
                        if (ggml_backend_tensor_alloc(buf, t, (char *)host_base + off) != GGML_STATUS_SUCCESS) {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok) {
                    out.buf = buf;
                    out.mmap_addr = mf.base;
                    out.mmap_len = mf.size;
                    out.used_mmap = true;
                    // The buffer owns the mapping from here; the WeightLoad
                    // fields above are a record of it, not a second owner.
                    register_buf_mmap(buf, mf.base, mf.size);
                    mf.release();
                    gguf_free(gctx);
                    return true;
                }
                if (buf) ggml_backend_buffer_free(buf);
                out.tensors.clear(); // discard partial; mf dtor unmaps
            }
            // mmap unsupported here / failed → fall through to the copy path
        }
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        fprintf(stderr, "%s: failed to allocate backend buffer\n", tag);
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    MappedFile mf(path);
    if (!mf.ok) {
        // Fallback: read via FILE* pread/fseek. This is the rare path —
        // most systems have working mmap. We implement it inline here so
        // models don't have to.
        FILE * fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            return false;
        }
        std::vector<uint8_t> tbuf;
        bool trunc = false;
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes) tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0) {
                trunc = true;
                break;
            }
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0) {
                trunc = true;
                break;
            }
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes) {
                trunc = true;
                break;
            }
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
        if (trunc) {
            fprintf(stderr, "%s: truncated/corrupt GGUF '%s' — tensor data past EOF\n", tag, path);
            gguf_free(gctx);
            free_weights(out);
            return false;
        }
    } else {
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            // Guard against a truncated/corrupt file: reading past the mapping
            // segfaults inside the backend memmove (the historical crash mode for
            // partially-downloaded models). Fail cleanly instead.
            if (data_off + off + nbytes > mf.size) {
                fprintf(stderr,
                        "%s: truncated/corrupt GGUF '%s' — tensor '%s' extends "
                        "past EOF (need %zu, file %zu)\n",
                        tag, path, ggml_get_name(t), (size_t)(data_off + off + nbytes), mf.size);
                gguf_free(gctx);
                free_weights(out);
                return false;
            }
            ggml_backend_tensor_set(t, (const char *)mf.base + data_off + off, 0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

bool load_weights_split(const char * path, ggml_backend_t gpu_backend, ggml_backend_t cpu_backend, IsGpuTensor is_gpu,
                        void * user, const char * model_tag, WeightLoad & out) {
    const char * tag = model_tag ? model_tag : "core_gguf";

    if (!gpu_backend || !cpu_backend) {
        fprintf(stderr, "%s: load_weights_split requires both gpu and cpu backends\n", tag);
        return false;
    }
    if (!is_gpu) {
        fprintf(stderr, "%s: load_weights_split requires a non-null is_gpu predicate\n", tag);
        return false;
    }

    gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&out.ctx };
    gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx) gguf_free(gctx);
        return false;
    }

    // Pass 1: partition tensors by predicate, sum sizes per partition.
    std::vector<ggml_tensor *> gpu_tensors, cpu_tensors;
    size_t gpu_size = 0, cpu_size = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
        const char * tname = ggml_get_name(t);
        if (is_gpu(tname, user)) {
            gpu_tensors.push_back(t);
            gpu_size += ggml_nbytes(t);
        } else {
            cpu_tensors.push_back(t);
            cpu_size += ggml_nbytes(t);
        }
        out.tensors[tname] = t;
    }

    // Some drivers cap a single device allocation (AMD Vulkan proprietary:
    // 2 GiB). Chunk each partition into <= 1.5 GiB buffers; the headroom
    // absorbs alignment padding.
    static constexpr size_t max_alloc_chunk = (size_t)1536 * 1024 * 1024;

    auto round_up = [](size_t n, size_t a) { return (n + a - 1) & ~(a - 1); };
    auto bind_partition = [&](ggml_backend_t be, const std::vector<ggml_tensor *> & tensors,
                              std::vector<ggml_backend_buffer_t> & out_bufs) -> bool {
        if (tensors.empty()) return true;
        const size_t align = ggml_backend_get_alignment(be);

        // Size tensors by the BUFFER TYPE's alloc size, not ggml_nbytes():
        // CUDA pads quantized rows up to MATRIX_ROW_PADDING so MMQ can
        // over-read; sizing with ggml_nbytes leaves the last tensor of a
        // chunk short and the tensor_alloc write runs past the buffer end.
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(be);
        auto tensor_size = [&](const ggml_tensor * t) { return ggml_backend_buft_get_alloc_size(buft, t); };

        struct Chunk {
            std::vector<ggml_tensor *> ts;
            size_t aligned_total = 0;
        };
        std::vector<Chunk> chunks(1);
        for (ggml_tensor * t : tensors) {
            const size_t nb = tensor_size(t);
            const size_t next = round_up(chunks.back().aligned_total, align) + nb;
            if (next > max_alloc_chunk && !chunks.back().ts.empty()) {
                chunks.push_back({});
                chunks.back().ts.push_back(t);
                chunks.back().aligned_total = nb;
            } else {
                chunks.back().ts.push_back(t);
                chunks.back().aligned_total = next;
            }
        }

        for (auto & chunk : chunks) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, chunk.aligned_total);
            if (!buf) {
                fprintf(stderr, "%s: failed to allocate %zu MiB backend buffer\n", tag, chunk.aligned_total / 1048576);
                for (auto * b : out_bufs) ggml_backend_buffer_free(b);
                out_bufs.clear();
                return false;
            }
            char * base = (char *)ggml_backend_buffer_get_base(buf);
            size_t cursor = 0;
            for (ggml_tensor * t : chunk.ts) {
                cursor = round_up(cursor, align);
                ggml_backend_tensor_alloc(buf, t, base + cursor);
                cursor += tensor_size(t);
            }
            out_bufs.push_back(buf);
        }
        return true;
    };

    std::vector<ggml_backend_buffer_t> gpu_bufs, cpu_bufs;
    if (!bind_partition(gpu_backend, gpu_tensors, gpu_bufs)) {
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        out.tensors.clear();
        return false;
    }
    if (!bind_partition(cpu_backend, cpu_tensors, cpu_bufs)) {
        for (auto * b : gpu_bufs) ggml_backend_buffer_free(b);
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        out.tensors.clear();
        return false;
    }

    if (!gpu_bufs.empty()) {
        out.buf = gpu_bufs[0];
        for (size_t i = 1; i < gpu_bufs.size(); i++) out.split_bufs.push_back(gpu_bufs[i]);
    }
    if (!cpu_bufs.empty()) {
        out.buf_cpu = cpu_bufs[0];
        for (size_t i = 1; i < cpu_bufs.size(); i++) out.split_bufs.push_back(cpu_bufs[i]);
    }

    // Copy tensor data from the file (mmap when available, pread fallback).
    const size_t data_off = gguf_get_data_offset(gctx);
    MappedFile mf(path);
    if (!mf.ok) {
        FILE * fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            free_weights(out);
            gguf_free(gctx);
            return false;
        }
        std::vector<uint8_t> tbuf;
        bool trunc = false;
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes) tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0) {
                trunc = true;
                break;
            }
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0) {
                trunc = true;
                break;
            }
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes) {
                trunc = true;
                break;
            }
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
        if (trunc) {
            fprintf(stderr, "%s: truncated/corrupt GGUF '%s' — tensor data past EOF\n", tag, path);
            free_weights(out);
            gguf_free(gctx);
            return false;
        }
    } else {
        for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0) continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            // Overflow-safe bounds check: a crafted GGUF can put off/nbytes near
            // SIZE_MAX so the additive form wraps. Compare subtractively.
            if (data_off > mf.size || off > mf.size - data_off || nbytes > mf.size - data_off - off) {
                fprintf(stderr,
                        "%s: truncated/corrupt GGUF '%s' — tensor '%s' extends "
                        "past EOF (need %zu, file %zu)\n",
                        tag, path, ggml_get_name(t), data_off + off + nbytes, mf.size);
                free_weights(out);
                gguf_free(gctx);
                return false;
            }
            ggml_backend_tensor_set(t, (const char *)mf.base + data_off + off, 0, nbytes);
        }
    }

    fprintf(stderr, "%s: weight residency: gpu=%zu MiB (%zu tensors), cpu=%zu MiB (%zu tensors)\n", tag,
            gpu_size / 1048576, gpu_tensors.size(), cpu_size / 1048576, cpu_tensors.size());

    gguf_free(gctx);
    return true;
}

void release_weight_buffer(ggml_backend_buffer_t & buf) {
    if (!buf) return;
    // Take the entry before the free, not after: between a free and a later
    // erase, a concurrent load could receive a new buffer at the same address
    // and register it, and the erase would then drop a live mapping's record.
    const mmap_region r = take_buf_mmap(buf);
    // Free the buffer first. On the no-copy path it is a view onto these pages,
    // so unmapping them while it is alive would leave the device addressing
    // unmapped memory.
    ggml_backend_buffer_free(buf);
    buf = nullptr;
    core_unmap(r.base, r.size);
}

void free_weights(WeightLoad & wl) {
    release_weight_buffer(wl.buf);
    release_weight_buffer(wl.buf_cpu);
    for (auto * b : wl.split_bufs) release_weight_buffer(b);
    wl.split_bufs.clear();
    // The mapping was released with its buffer above; these fields are the
    // caller-visible record of the load, so clear them without unmapping again.
    wl.mmap_addr = nullptr;
    wl.mmap_len = 0;
    wl.used_mmap = false;
    if (wl.ctx) {
        ggml_free(wl.ctx);
        wl.ctx = nullptr;
    }
    wl.tensors.clear();
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

// Signatures use `core_gguf::tensor_map` (see gguf_loader.h cross-repo contract).
ggml_tensor * try_get(const tensor_map & tensors, const char * name) {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor * require(const tensor_map & tensors, const char * name, const char * model_tag) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: required tensor '%s' not found in GGUF\n", model_tag ? model_tag : "core_gguf", name);
        return nullptr;
    }
    return it->second;
}


std::string format_layer_name(const char * fmt, int i) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i);
    return std::string(buf);
}

std::string format_layer_name(const char * fmt, int i, int j) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i, j);
    return std::string(buf);
}

} // namespace core_gguf
