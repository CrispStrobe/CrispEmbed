// Layer0 attention op-by-op probe (test-only). Same source built native and WASM.
//
//   probe chain <model|authored> <outdir> <x.f32>        full production op chain, every intermediate written
//   probe ops   <model|authored> <outdir> <indir>        each op once, inputs read from <indir> (another chain)
//
// Ops (production crispembed post-LN BERT layer0, MiniLM: no rope, no relative bias,
// single unpadded sentence so flash_attn_ext has no mask):
//   ln_mul = x*embd_ln_w; ln_add = ln_mul+embd_ln_b; {q,k,v}_raw = mul_mat(W, ln_add);
//   {q,k,v} = raw+bias; flash = flash_attn_ext(q,k,v, null, 1/sqrt(32)) as [384,12];
//   o_raw = mul_mat(o_w, flash); o = o_raw+o_b; residual = ln_add+o   (= norm01 input)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

static const int H = 384, T = 12, NH = 12, HD = 32;

static std::string host(const std::string & p) {
#ifdef __EMSCRIPTEN__
    return "/host" + p;
#else
    return p;
#endif
}
static void write_file(const std::string & path, const std::vector<float> & v) {
    FILE * f = std::fopen(host(path).c_str(), "wb");
    if (!f || std::fwrite(v.data(), 4, v.size(), f) != v.size() || std::fclose(f) != 0) { std::fprintf(stderr, "write failed %s\n", path.c_str()); std::exit(3); }
}
static std::vector<float> read_f32(const std::string & path, size_t n) {
    std::vector<float> v(n);
    FILE * f = std::fopen(host(path).c_str(), "rb");
    if (!f || std::fread(v.data(), 4, n, f) != n || std::fgetc(f) != EOF) { std::fprintf(stderr, "read failed %s\n", path.c_str()); std::exit(3); }
    std::fclose(f);
    return v;
}

struct Weights {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::map<std::string, ggml_tensor *> t;
    std::string buffer_name, json;
};

static const char * NAMES[][2] = {
    {"ln_w", "token_embd_norm.weight"}, {"ln_b", "token_embd_norm.bias"},
    {"q_w", "blk.0.attn_q.weight"}, {"q_b", "blk.0.attn_q.bias"},
    {"k_w", "blk.0.attn_k.weight"}, {"k_b", "blk.0.attn_k.bias"},
    {"v_w", "blk.0.attn_v.weight"}, {"v_b", "blk.0.attn_v.bias"},
    {"o_w", "blk.0.attn_output.weight"}, {"o_b", "blk.0.attn_output.bias"},
};
static const char * ALT[][2] = {
    {"ln_w", "embd_ln.weight"}, {"ln_b", "embd_ln.bias"},
    {"q_w", "enc.0.attn.q.weight"}, {"q_b", "enc.0.attn.q.bias"},
    {"k_w", "enc.0.attn.k.weight"}, {"k_b", "enc.0.attn.k.bias"},
    {"v_w", "enc.0.attn.v.weight"}, {"v_b", "enc.0.attn.v.bias"},
    {"o_w", "enc.0.attn.o.weight"}, {"o_b", "enc.0.attn.o.bias"},
};

static uint32_t lcg(uint32_t & s) { s = s * 1664525u + 1013904223u; return s >> 8; }

// Weights land in a plain CPU backend buffer, as in production.
static Weights load_weights(ggml_backend_t backend, const std::string & model) {
    Weights w;
    ggml_init_params p = { ggml_tensor_overhead() * 16, nullptr, true };
    w.ctx = ggml_init(p);
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> data;
    if (model == "authored") {  // weight-free controls: random IQ4_NL weights, F32 vectors
        uint32_t s = 777;
        for (auto & n : NAMES) {
            const std::string key = n[0];
            const bool mat = key.size() > 2 && key.substr(key.size() - 2) == "_w" && key != "ln_w";
            ggml_tensor * t = mat ? ggml_new_tensor_2d(w.ctx, GGML_TYPE_IQ4_NL, H, H) : ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, H);
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            if (mat) {
                for (auto & b : bytes) b = (uint8_t) lcg(s);
                for (size_t i = 0; i < bytes.size(); i += 18) {  // block_iq4_nl: fp16 d + 16 qs
                    const ggml_fp16_t d = ggml_fp32_to_fp16(((int) (lcg(s) % 2001) - 1000) / 20000.0f);
                    std::memcpy(&bytes[i], &d, 2);
                }
            } else {
                for (size_t i = 0; i < bytes.size(); i += 4) {
                    const float v = key == "ln_w" ? 1.0f + ((int) (lcg(s) % 2001) - 1000) / 4000.0f : ((int) (lcg(s) % 2001) - 1000) / 8000.0f;
                    std::memcpy(&bytes[i], &v, 4);
                }
            }
            w.t[key] = t;
            data.push_back({t, bytes});
        }
    } else {
        ggml_context * mctx = nullptr;
        gguf_init_params gp = { false, &mctx };
        gguf_context * g = gguf_init_from_file(host(model).c_str(), gp);
        if (!g) { std::fprintf(stderr, "gguf load failed\n"); std::exit(3); }
        for (size_t i = 0; i < sizeof(NAMES) / sizeof(NAMES[0]); ++i) {
            ggml_tensor * src = ggml_get_tensor(mctx, NAMES[i][1]);
            if (!src) src = ggml_get_tensor(mctx, ALT[i][1]);
            if (!src) { std::fprintf(stderr, "missing %s\n", NAMES[i][1]); std::exit(5); }
            ggml_tensor * t = ggml_dup_tensor(w.ctx, src);
            w.t[NAMES[i][0]] = t;
            data.push_back({t, std::vector<uint8_t>((uint8_t *) src->data, (uint8_t *) src->data + ggml_nbytes(src))});
            char row[192];
            std::snprintf(row, sizeof(row), "%s\"%s\":{\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld]}", w.json.empty() ? "" : ",",
                          NAMES[i][0], ggml_get_name(src), ggml_type_name(src->type), (long long) src->ne[0], (long long) src->ne[1]);
            w.json += row;
        }
        gguf_free(g);
        ggml_free(mctx);
    }
    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    for (auto & d : data) ggml_backend_tensor_set(d.first, d.second.data(), 0, d.second.size());
    w.buffer_name = ggml_backend_buffer_name(w.buf);
    return w;
}

// One production op on the CPU backend; inputs are [H,T] F32 host arrays.
using Build = std::function<ggml_tensor *(ggml_context *, const std::vector<ggml_tensor *> &)>;
static std::vector<float> run(ggml_backend_t backend, const std::vector<const std::vector<float> *> & inputs, const Build & build) {
    ggml_init_params gp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * g = ggml_init(gp);
    std::vector<ggml_tensor *> in;
    for (size_t i = 0; i < inputs.size(); ++i) in.push_back(ggml_new_tensor_2d(g, GGML_TYPE_F32, H, T));
    ggml_tensor * y = build(g, in);
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) { std::fprintf(stderr, "alloc failed\n"); std::exit(4); }
    for (size_t i = 0; i < inputs.size(); ++i) ggml_backend_tensor_set(in[i], inputs[i]->data(), 0, ggml_nbytes(in[i]));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); std::exit(4); }
    if (ggml_nelements(y) != H * T || y->type != GGML_TYPE_F32) { std::fprintf(stderr, "unexpected output geometry\n"); std::exit(4); }
    std::vector<float> out(H * T);
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(ga);
    ggml_free(g);
    return out;
}

struct Op { const char * name; std::vector<const char *> deps; Build build; };

static std::vector<Op> ops(Weights & w) {
    auto W = [&](const char * k) { return w.t.at(k); };
    const float scale = 1.0f / sqrtf((float) HD);
    auto heads = [](ggml_context * g, ggml_tensor * x) {  // production reshape_4d + permute(0,2,1,3)
        return ggml_permute(g, ggml_reshape_4d(g, x, HD, NH, T, 1), 0, 2, 1, 3);
    };
    return {
        {"ln_mul", {"x"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_mul(g, i[0], W("ln_w")); }},
        {"ln_add", {"ln_mul"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], W("ln_b")); }},
        {"q_raw", {"ln_add"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_mul_mat(g, W("q_w"), i[0]); }},
        {"k_raw", {"ln_add"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_mul_mat(g, W("k_w"), i[0]); }},
        {"v_raw", {"ln_add"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_mul_mat(g, W("v_w"), i[0]); }},
        {"q", {"q_raw"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], W("q_b")); }},
        {"k", {"k_raw"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], W("k_b")); }},
        {"v", {"v_raw"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], W("v_b")); }},
        {"flash", {"q", "k", "v"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) {
             ggml_tensor * a = ggml_flash_attn_ext(g, heads(g, i[0]), heads(g, i[1]), heads(g, i[2]), nullptr, scale, 0.0f, 0.0f);
             return ggml_reshape_2d(g, a, H, T);
         }},
        {"o_raw", {"flash"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_mul_mat(g, W("o_w"), i[0]); }},
        {"o", {"o_raw"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], W("o_b")); }},
        {"residual", {"ln_add", "o"}, [=](ggml_context * g, const std::vector<ggml_tensor *> & i) { return ggml_add(g, i[0], i[1]); }},
    };
}

static ggml_backend_t make_backend() {
    ggml_backend_t b = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(b, 1);
#ifdef __EMSCRIPTEN__
    ggml_backend_cpu_set_wasm_q8_0_rounding(b, true);  // production WASM holds CRISPEMBED_WASM_Q8_0_ROUND=1
#endif
    return b;
}

int main(int argc, char ** argv) {
#ifdef __EMSCRIPTEN__
    EM_ASM({ FS.mkdir('/host'); FS.mount(NODEFS, { root: '/' }, '/host'); });
#endif
    if (argc != 5 || (std::string(argv[1]) != "chain" && std::string(argv[1]) != "ops")) {
        std::fprintf(stderr, "usage: probe chain|ops <model|authored> <outdir> <x.f32|indir>\n");
        return 2;
    }
    const bool chain = std::string(argv[1]) == "chain";
    const std::string out = argv[3], src = argv[4];
    ggml_backend_t backend = make_backend();
    Weights w = load_weights(backend, argv[2]);
    std::map<std::string, std::vector<float>> v;
    if (chain) v["x"] = read_f32(src, H * T);
    std::printf("{\"mode\":\"%s\",\"weights\":{%s},\"weight_buffer\":\"%s\",\"ops\":[", argv[1], w.json.c_str(), w.buffer_name.c_str());
    bool first = true;
    for (auto & op : ops(w)) {
        std::vector<std::vector<float>> owned;
        std::vector<const std::vector<float> *> in;
        for (auto d : op.deps) {
            if (chain) in.push_back(&v.at(d));
            else { owned.push_back(read_f32(src + "/" + (std::string(d) == "x" ? std::string("x") : std::string(d)) + ".f32", H * T)); }
        }
        if (!chain) for (auto & o : owned) in.push_back(&o);
        v[op.name] = run(backend, in, op.build);
        write_file(out + "/" + op.name + ".f32", v[op.name]);
        std::printf("%s\"%s\"", first ? "" : ",", op.name);
        first = false;
    }
    if (chain) write_file(out + "/x.f32", v["x"]);
    std::printf("]}\n");
    ggml_backend_buffer_free(w.buf);
    ggml_free(w.ctx);
    ggml_backend_free(backend);
    return 0;
}
