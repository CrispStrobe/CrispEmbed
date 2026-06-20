// src/core/gguf_loader.cpp — implementation of core_gguf:: helpers.
// See gguf_loader.h for the interface contract.

#include "gguf_loader.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#elif !defined(__EMSCRIPTEN__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace core_gguf {

// ---------------------------------------------------------------------------
// Pass 1: metadata
// ---------------------------------------------------------------------------

gguf_context* open_metadata(const char* path) {
    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) {
        fprintf(stderr, "core_gguf: failed to open '%s' for metadata read\n", path);
    }
    return g;
}

void free_metadata(gguf_context* gctx) {
    if (gctx)
        gguf_free(gctx);
}

// Type-checked scalar readers. The GGUF format stores types explicitly so
// we can validate; if the file has a mismatched type the reader silently
// returns the default rather than crashing, matching the existing inline
// helpers in each model.

uint32_t kv_u32(gguf_context* gctx, const char* key, uint32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
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

int32_t kv_i32(gguf_context* gctx, const char* key, int32_t default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
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

float kv_f32(gguf_context* gctx, const char* key, float default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    const gguf_type t = gguf_get_kv_type(gctx, k);
    if (t == GGUF_TYPE_FLOAT32)
        return gguf_get_val_f32(gctx, k);
    if (t == GGUF_TYPE_FLOAT64)
        return (float)gguf_get_val_f64(gctx, k);
    return default_val;
}

bool kv_bool(gguf_context* gctx, const char* key, bool default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_BOOL)
        return default_val;
    return gguf_get_val_bool(gctx, k);
}

std::string kv_str(gguf_context* gctx, const char* key, const char* default_val) {
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return default_val ? default_val : "";
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_STRING)
        return default_val ? default_val : "";
    const char* s = gguf_get_val_str(gctx, k);
    return s ? std::string(s) : std::string(default_val ? default_val : "");
}

std::vector<std::string> kv_str_array(gguf_context* gctx, const char* key) {
    std::vector<std::string> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY)
        return out;
    if (gguf_get_arr_type(gctx, k) != GGUF_TYPE_STRING)
        return out;
    const int n = gguf_get_arr_n(gctx, k);
    out.reserve((size_t)n);
    for (int i = 0; i < n; i++) {
        out.emplace_back(gguf_get_arr_str(gctx, k, i));
    }
    return out;
}

std::vector<int> kv_i32_array(gguf_context* gctx, const char* key) {
    std::vector<int> out;
    const int k = gguf_find_key(gctx, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(gctx, k) != GGUF_TYPE_ARRAY)
        return out;
    const int n = gguf_get_arr_n(gctx, k);
    const void* data = gguf_get_arr_data(gctx, k);
    auto arr_type = gguf_get_arr_type(gctx, k);
    out.resize(n);
    if (arr_type == GGUF_TYPE_INT32) {
        memcpy(out.data(), data, n * sizeof(int32_t));
    } else if (arr_type == GGUF_TYPE_UINT32) {
        const uint32_t* p = (const uint32_t*)data;
        for (int i = 0; i < n; i++) out[i] = (int)p[i];
    } else if (arr_type == GGUF_TYPE_INT64) {
        const int64_t* p = (const int64_t*)data;
        for (int i = 0; i < n; i++) out[i] = (int)p[i];
    } else {
        out.clear();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pass 2: tensor allocation + weight data copy.
// ---------------------------------------------------------------------------

namespace {

// Platform unmap, shared by MappedFile's destructor and free_weights() (the
// no-copy path transfers the mapping into WeightLoad, which unmaps on free).
void core_unmap(void* base, size_t size) {
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

// Read a file slice into a backend tensor. Uses mmap on POSIX; falls back
// to pread/lseek+read when mmap is unavailable (rare in practice).
//
// On POSIX the mmap lives for the duration of one load call — we copy via
// ggml_backend_tensor_set then unmap. No mmap persists past load_weights()
// UNLESS release() is called (the no-copy path keeps it alive in WeightLoad).
struct MappedFile {
    int fd = -1;
    void* base = nullptr;
    size_t size = 0;
    bool ok = false;

    explicit MappedFile(const char* path) {
#if defined(__EMSCRIPTEN__)
        // Emscripten MEMFS: skip mmap, fall through to fread path.
        (void)path;
        return;
#elif defined(_WIN32)
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (hFile == INVALID_HANDLE_VALUE)
            return;
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hFile, &fsize)) {
            CloseHandle(hFile);
            return;
        }
        size = (size_t)fsize.QuadPart;
        HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(hFile);
        if (!hMap)
            return;
        base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
        if (!base)
            return;
        ok = true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0)
            return;
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
    void release() { base = nullptr; size = 0; }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

} // namespace

bool load_weights(const char* path, ggml_backend_t backend, const char* model_tag,
                  WeightLoad& out, bool try_mmap) {
    const char* tag = model_tag ? model_tag : "core_gguf";

    gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&out.ctx};
    gguf_context* gctx = gguf_init_from_file(path, gp);
    if (!gctx || !out.ctx) {
        fprintf(stderr, "%s: failed to load tensor metadata from '%s'\n", tag, path);
        if (gctx)
            gguf_free(gctx);
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
                for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t))
                    max_ts = std::max(max_ts, ggml_nbytes(t));
                void* host_base = (char*)mf.base + data_off;
                ggml_backend_buffer_t buf =
                    ggml_backend_dev_buffer_from_host_ptr(dev, host_base, mf.size - data_off, max_ts);
                bool ok = (buf != nullptr);
                if (ok) {
                    for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
                        out.tensors[ggml_get_name(t)] = t;
                        const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
                        if (tid < 0) continue;
                        const size_t off = gguf_get_tensor_offset(gctx, tid);
                        if (ggml_backend_tensor_alloc(buf, t, (char*)host_base + off) != GGML_STATUS_SUCCESS) {
                            ok = false; break;
                        }
                    }
                }
                if (ok) {
                    out.buf = buf;
                    out.mmap_addr = mf.base;
                    out.mmap_len = mf.size;
                    out.used_mmap = true;
                    mf.release();          // WeightLoad now owns the mapping
                    gguf_free(gctx);
                    return true;
                }
                if (buf) ggml_backend_buffer_free(buf);
                out.tensors.clear();       // discard partial; mf dtor unmaps
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
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "%s: cannot open '%s' for fread fallback\n", tag, path);
            gguf_free(gctx);
            return false;
        }
        std::vector<uint8_t> tbuf;
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            if (tbuf.size() < nbytes)
                tbuf.resize(nbytes);
#if defined(_WIN32)
            if (_fseeki64(fp, (int64_t)(data_off + off), SEEK_SET) != 0)
                break;
#else
            if (fseeko(fp, (off_t)(data_off + off), SEEK_SET) != 0)
                break;
#endif
            if (fread(tbuf.data(), 1, nbytes, fp) != nbytes)
                break;
            ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
        }
        fclose(fp);
    } else {
        for (ggml_tensor* t = ggml_get_first_tensor(out.ctx); t; t = ggml_get_next_tensor(out.ctx, t)) {
            out.tensors[ggml_get_name(t)] = t;
            const int64_t tid = gguf_find_tensor(gctx, ggml_get_name(t));
            if (tid < 0)
                continue;
            const size_t off = gguf_get_tensor_offset(gctx, tid);
            const size_t nbytes = ggml_nbytes(t);
            ggml_backend_tensor_set(t, (const char*)mf.base + data_off + off, 0, nbytes);
        }
    }

    gguf_free(gctx);
    return true;
}

void free_weights(WeightLoad& wl) {
    if (wl.buf) {
        ggml_backend_buffer_free(wl.buf);   // no-copy buffer doesn't own the pages
        wl.buf = nullptr;
    }
    if (wl.mmap_addr) {                      // unmap after the buffer is freed
        core_unmap(wl.mmap_addr, wl.mmap_len);
        wl.mmap_addr = nullptr;
        wl.mmap_len = 0;
        wl.used_mmap = false;
    }
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
ggml_tensor* try_get(const tensor_map& tensors, const char* name) {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

ggml_tensor* require(const tensor_map& tensors, const char* name, const char* model_tag) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        fprintf(stderr, "%s: required tensor '%s' not found in GGUF\n", model_tag ? model_tag : "core_gguf", name);
        return nullptr;
    }
    return it->second;
}


std::string format_layer_name(const char* fmt, int i) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i);
    return std::string(buf);
}

std::string format_layer_name(const char* fmt, int i, int j) {
    char buf[256];
    snprintf(buf, sizeof(buf), fmt, i, j);
    return std::string(buf);
}

} // namespace core_gguf
