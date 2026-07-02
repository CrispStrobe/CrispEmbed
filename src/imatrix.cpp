// imatrix.cpp — see imatrix.h.
//
// File format (private to CrispEmbed, GGUF-based):
//   KV  general.architecture = "crispembed-imatrix"
//   KV  imatrix.version       = 1  (u32)
//   KV  count.<weight-name>   = u64   number of activation rows accumulated
//   tensor <weight-name>      = F32[n_per_row]  running sum of squares per column
// The quantizer reads importance[c] = sum_sq[c] / count.

#include "imatrix.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

std::mutex g_mu;
bool g_checked = false;
bool g_active = false;
bool g_atexit = false;
std::string g_path;
// weight name -> running sum of squares (double for numerical stability)
std::map<std::string, std::vector<double>> g_sumsq;
// weight name -> running row count
std::map<std::string, uint64_t> g_count;

bool is_active() {
    if (!g_checked) {
        const char * p = getenv("CRISPEMBED_IMATRIX_OUT");
        if (p && p[0]) {
            g_active = true;
            g_path = p;
        }
        g_checked = true;
    }
    return g_active;
}

// Chase view chains to the underlying leaf tensor (the actual weight).
const ggml_tensor * underlying(const ggml_tensor * t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}

// eval callback: ask==true  -> return whether we want a post-compute callback
//                ask==false -> the node's data is ready; collect it
bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (t->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor * w = underlying(t->src[0]);
    // src0 must be a named leaf weight (activations are computed nodes, op != NONE)
    if (!w || w->op != GGML_OP_NONE || w->name[0] == '\0') return false;

    if (ask) return true; // request the post-compute callback

    struct ggml_tensor * a = t->src[1]; // the activation input
    if (!a || !ggml_is_contiguous(a)) return true;
    const int64_t ne0 = a->ne[0];
    if (ne0 <= 0) return true;
    const int64_t nrows = ggml_nelements(a) / ne0;
    if (nrows <= 0) return true;

    const size_t nbytes = ggml_nbytes(a);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(a, buf.data(), 0, nbytes);

    std::lock_guard<std::mutex> lk(g_mu);
    auto & acc = g_sumsq[w->name];
    if (acc.empty()) acc.assign((size_t)ne0, 0.0);
    if ((int64_t)acc.size() != ne0) return true; // shape drift; skip defensively

    if (a->type == GGML_TYPE_F32) {
        const float * x = (const float *)buf.data();
        for (int64_t r = 0; r < nrows; r++) {
            const float * row = x + r * ne0;
            for (int64_t c = 0; c < ne0; c++) acc[c] += (double)row[c] * (double)row[c];
        }
    } else if (a->type == GGML_TYPE_F16) {
        const ggml_fp16_t * x = (const ggml_fp16_t *)buf.data();
        for (int64_t r = 0; r < nrows; r++) {
            const ggml_fp16_t * row = x + r * ne0;
            for (int64_t c = 0; c < ne0; c++) {
                const double v = ggml_fp16_to_fp32(row[c]);
                acc[c] += v * v;
            }
        }
    } else {
        return true; // unsupported activation type; skip
    }
    g_count[w->name] += (uint64_t)nrows;
    return true;
}

// Merge any pre-existing imatrix file into the in-memory accumulators.
void merge_existing() {
    struct ggml_context * ctx = nullptr;
    struct gguf_init_params p = { /*no_alloc*/ false, /*ctx*/ &ctx };
    struct gguf_context * g = gguf_init_from_file(g_path.c_str(), p);
    if (!g) return; // no prior file (first run)
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; i++) {
        const char * name = gguf_get_tensor_name(g, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t || t->type != GGML_TYPE_F32) continue;
        const int64_t ne0 = t->ne[0];
        const float * d = (const float *)t->data;
        auto & acc = g_sumsq[name];
        if (acc.empty()) acc.assign((size_t)ne0, 0.0);
        if ((int64_t)acc.size() != ne0) continue;
        for (int64_t c = 0; c < ne0; c++) acc[c] += (double)d[c];
        std::string ck = std::string("count.") + name;
        int64_t kid = gguf_find_key(g, ck.c_str());
        if (kid >= 0) g_count[name] += gguf_get_val_u64(g, kid);
    }
    gguf_free(g);
    ggml_free(ctx);
}

} // namespace

void crispembed_imatrix_install(ggml_backend_sched_t sched) {
    if (!is_active() || !sched) return;
    ggml_backend_sched_set_eval_callback(sched, eval_cb, nullptr);
    if (!g_atexit) {
        atexit(crispembed_imatrix_flush);
        g_atexit = true;
    }
}

void crispembed_imatrix_flush(void) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_active || g_sumsq.empty()) return;

    merge_existing();

    // Allocate a ggml context large enough to hold all sum-of-squares tensors.
    size_t total = 0;
    for (auto & kv : g_sumsq) total += kv.second.size();
    struct ggml_init_params ip = {
        /*mem_size*/ total * sizeof(float) + g_sumsq.size() * ggml_tensor_overhead() + (1u << 20),
        /*mem_buffer*/ nullptr,
        /*no_alloc*/ false,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fprintf(stderr, "imatrix: ggml_init failed\n");
        return;
    }

    struct gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "crispembed-imatrix");
    gguf_set_val_u32(g, "imatrix.version", 1);

    for (auto & kv : g_sumsq) {
        const std::string & name = kv.first;
        const std::vector<double> & acc = kv.second;
        struct ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)acc.size());
        ggml_set_name(t, name.c_str());
        float * d = (float *)t->data;
        for (size_t c = 0; c < acc.size(); c++) d[c] = (float)acc[c];
        gguf_add_tensor(g, t);
        std::string ck = std::string("count.") + name;
        gguf_set_val_u64(g, ck.c_str(), g_count[name]);
    }

    if (!gguf_write_to_file(g, g_path.c_str(), /*only_meta*/ false)) {
        fprintf(stderr, "imatrix: failed to write '%s'\n", g_path.c_str());
    } else {
        fprintf(stderr, "imatrix: wrote %zu tensors to '%s'\n", g_sumsq.size(), g_path.c_str());
    }
    gguf_free(g);
    ggml_free(ctx);
}
