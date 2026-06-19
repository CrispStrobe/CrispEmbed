// Unit test: core_gguf::load_weights copy path vs the opt-in no-copy mmap path.
// Builds a small GGUF, loads it both ways on the CPU backend (which advertises
// buffer_from_host_ptr), and asserts the tensors are byte-identical and match
// the values written. Validates the no-copy path is actually taken and that
// free_weights() cleans up (incl. the mmap) without crashing.

#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char* kPath = "/tmp/crispembed_test_loader_mmap.gguf";

static float expected(int i) { return sinf((float)i * 0.013f) + 0.5f; }

static bool write_test_gguf(const char* path) {
    ggml_init_params ip = {32 * 1024 * 1024, nullptr, /*no_alloc=*/false};
    ggml_context* ctx = ggml_init(ip);
    struct Def { const char* name; int n0, n1; };
    Def defs[] = {{"alpha", 37, 5}, {"beta.weight", 256, 3}, {"gamma", 1, 1}};
    for (auto& d : defs) {
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.n0, d.n1);
        ggml_set_name(t, d.name);
        float* data = (float*)t->data;
        int64_t n = (int64_t)d.n0 * d.n1;
        for (int64_t i = 0; i < n; i++) data[i] = expected((int)i);
    }
    gguf_context* g = gguf_init_empty();
    for (ggml_tensor* t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t))
        gguf_add_tensor(g, t);
    bool ok = gguf_write_to_file(g, path, /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static std::vector<float> read_tensor(core_gguf::WeightLoad& wl, const char* name) {
    auto it = wl.tensors.find(name);
    if (it == wl.tensors.end() || !it->second) return {};
    ggml_tensor* t = it->second;
    std::vector<float> out(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

int main() {
    if (!write_test_gguf(kPath)) { fprintf(stderr, "FAIL: could not write %s\n", kPath); return 1; }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "FAIL: cpu backend\n"); return 1; }

    core_gguf::WeightLoad cw, mw;
    if (!core_gguf::load_weights(kPath, backend, "test", cw, /*try_mmap=*/false)) {
        fprintf(stderr, "FAIL: copy load\n"); return 1;
    }
    if (!core_gguf::load_weights(kPath, backend, "test", mw, /*try_mmap=*/true)) {
        fprintf(stderr, "FAIL: mmap load\n"); return 1;
    }
    if (!mw.used_mmap) {
        fprintf(stderr, "FAIL: no-copy path not taken (CPU advertises buffer_from_host_ptr)\n");
        return 1;
    }
    printf("no-copy mmap path taken (used_mmap=1)\n");

    const char* names[] = {"alpha", "beta.weight", "gamma"};
    int fails = 0;
    for (const char* nm : names) {
        std::vector<float> a = read_tensor(cw, nm), b = read_tensor(mw, nm);
        if (a.empty() || a.size() != b.size()) { fprintf(stderr, "FAIL: %s size\n", nm); fails++; continue; }
        bool ok = true;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i] || fabsf(a[i] - expected((int)i)) > 1e-6f) { ok = false; break; }
        printf("  %-12s %4zu elems  copy==mmap==written: %s\n", nm, a.size(), ok ? "OK" : "MISMATCH");
        if (!ok) fails++;
    }

    core_gguf::free_weights(cw);
    core_gguf::free_weights(mw);  // also unmaps
    ggml_backend_free(backend);
    remove(kPath);

    if (fails) { fprintf(stderr, "FAILED (%d tensor(s))\n", fails); return 1; }
    printf("PASS: gguf_loader no-copy mmap == copy\n");
    return 0;
}
