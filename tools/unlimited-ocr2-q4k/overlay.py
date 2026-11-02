"""Exact-source, hosted-only observational overlay. Never a production source."""
import hashlib
from pathlib import Path

SOURCE_SHA = '57df29c381a9438667c9dae87b61ece42e5c4be29001966db6608de7434cdac5'

HELPER = r'''
// Generated test-only overlay: absent from every ordinary/root source file.
namespace uocr_q4_audit {
static const char * directory() {
    const char * p = getenv("UOCR_Q4_AUDIT_DIR");
    return p && *p ? p : nullptr;
}
static void raw(const std::string & key, const void * data, size_t bytes,
                int width, int middle, int rows, bool integer = false) {
    const char * dir = directory();
    if (!dir) return;
    if (!data || width < 1 || middle < 1 || rows < 1 || bytes > 64ull * 1024 * 1024 ||
        bytes != size_t(width) * middle * rows * 4) throw std::runtime_error("audit extent");
    if (!integer) {
        const float * f = static_cast<const float *>(data);
        for (size_t i = 0; i < bytes / 4; ++i)
            if (!std::isfinite(f[i])) throw std::runtime_error("audit nonfinite");
    }
    const std::string file = key + (integer ? ".i32" : ".f32");
    const std::string path = std::string(dir) + "/" + file;
    FILE * out = fopen(path.c_str(), "wb");
    if (!out) throw std::runtime_error("audit raw open");
    const bool ok = fwrite(data, 1, bytes, out) == bytes;
    const bool closed = fclose(out) == 0;
    if (!ok || !closed) throw std::runtime_error("audit raw write");
    FILE * receipt = fopen((std::string(dir) + "/stages.jsonl").c_str(), "ab");
    if (!receipt) throw std::runtime_error("audit journal open");
    const int result = fprintf(receipt,
        "{\"name\":\"%s\",\"file\":\"%s\",\"type\":\"%s\",\"shape\":[%d,%d,%d]}\n",
        key.c_str(), file.c_str(), integer ? "i32" : "f32", rows, middle, width);
    const bool done = fclose(receipt) == 0;
    if (result < 0 || !done) throw std::runtime_error("audit journal write");
}
static void tensor(const std::string & key, ggml_tensor * value, int rows) {
    if (!directory()) return;
    if (!value || (value->type != GGML_TYPE_F32 && value->type != GGML_TYPE_I32) ||
        !ggml_is_contiguous(value) || value->ne[3] != 1 || rows < 1 || rows > value->ne[2])
        throw std::runtime_error("audit tensor layout");
    const size_t bytes = size_t(value->ne[0]) * value->ne[1] * rows * 4;
    if (value->type == GGML_TYPE_I32) {
        std::vector<int32_t> data(bytes / 4);
        ggml_backend_tensor_get(value, data.data(), 0, bytes);
        raw(key, data.data(), bytes, int(value->ne[0]), int(value->ne[1]), rows, true);
    } else {
        std::vector<float> data(bytes / 4);
        ggml_backend_tensor_get(value, data.data(), 0, bytes);
        raw(key, data.data(), bytes, int(value->ne[0]), int(value->ne[1]), rows);
    }
}
}
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Exact unique overlay source anchor differs')
    return text.replace(old, new, 1)


def transform(data):
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA:
        raise ValueError('Frozen Unlimited source SHA differs')
    text = '#include <stdexcept>\n' + data.decode('utf-8')
    text = replace_once(text, 'static bool run_llm_decoder_v2_fast(',
                        HELPER + '\nstatic bool run_llm_decoder_v2_fast(')
    text = replace_once(text, '                ggml_tensor * ff;\n                if (li == 0) {',
                        '                ggml_tensor * audit_hn = nullptr, * audit_ids = nullptr, * audit_gate = nullptr, * audit_up = nullptr;\n'
                        '                ggml_tensor * ff;\n                if (li == 0) {')
    text = replace_once(text,
        '                    ggml_tensor * gt = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));',
        '                    ggml_tensor * gate_pre = ggml_mul_mat_id(g, ly.gate_exps, hnK, ids);\n'
        '                    ggml_tensor * gt = ggml_silu(g, gate_pre);')
    text = replace_once(text,
        '                    ggml_tensor * dn = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gt, up), ids);',
        '                    if (p0 == 0 && r0 == 0 && uocr_q4_audit::directory()) {\n'
        '                        audit_hn = hn; audit_ids = ids; audit_gate = gate_pre; audit_up = up;\n'
        '                        for (ggml_tensor * retained : { audit_hn, audit_ids, audit_gate, audit_up }) ggml_set_output(retained);\n'
        '                    }\n'
        '                    ggml_tensor * dn = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gt, up), ids);')
    anchor = '                ggml_backend_tensor_get(out, h.data() + (size_t)r0 * D, 0, (size_t)Tc * D * sizeof(float));'
    text = replace_once(text, anchor, anchor + r'''
                if (p0 == 0 && uocr_q4_audit::directory()) {
                    const std::string key = "layer_" + std::to_string(li) + "_chunk_" + std::to_string(r0);
                    uocr_q4_audit::raw(key, h.data() + size_t(r0) * D, size_t(Tc) * D * 4, D, 1, Tc);
                    if (audit_hn) {
                        const int n = std::min(Tc, 16);
                        std::vector<float> hn_rows(size_t(n) * D);
                        ggml_backend_tensor_get(audit_hn, hn_rows.data(), 0, hn_rows.size() * 4);
                        uocr_q4_audit::raw(key + "_hn", hn_rows.data(), hn_rows.size() * 4, D, 1, n);
                        std::vector<int32_t> id_rows(size_t(n) * lhp.n_experts_top);
                        if (audit_ids->type != GGML_TYPE_I32 || audit_ids->ne[0] != lhp.n_experts_top ||
                            audit_ids->ne[1] != Tc || audit_ids->ne[2] != 1 || audit_ids->nb[0] != 4)
                            throw std::runtime_error("audit router-ID layout");
                        // Honor the actual top_k row stride; do not assume
                        // either a compact result or a full-argsort view.
                        for (int i = 0; i < n; ++i)
                            ggml_backend_tensor_get(audit_ids, id_rows.data() + size_t(i) * lhp.n_experts_top,
                                                    size_t(i) * audit_ids->nb[1], size_t(lhp.n_experts_top) * 4);
                        uocr_q4_audit::raw(key + "_ids", id_rows.data(), id_rows.size() * 4, lhp.n_experts_top, 1, n, true);
                        uocr_q4_audit::tensor(key + "_gate", audit_gate, n);
                        uocr_q4_audit::tensor(key + "_up", audit_up, n);
                    }
                }
''')
    anchor = '            ggml_backend_tensor_get(lg, logits.data(), 0, (size_t)V * sizeof(float));'
    # This line occurs only on the fast route; observe before page-guard -inf.
    text = replace_once(text, anchor, anchor + r'''
            if (step == 0 && uocr_q4_audit::directory()) {
                std::vector<float> final_norm(D);
                ggml_backend_tensor_get(xn, final_norm.data(), 0, size_t(D) * 4);
                uocr_q4_audit::raw("final_norm", final_norm.data(), size_t(D) * 4, D, 1, 1);
                uocr_q4_audit::raw("logits", logits.data(), size_t(V) * 4, V, 1, 1);
            }
''')
    # Retain norm storage through the head projection, with no arithmetic change.
    text = replace_once(text, '            ggml_tensor * lg = ggml_mul_mat(g, lm_w, xn);',
        '            if (step == 0 && uocr_q4_audit::directory()) ggml_set_output(xn);\n'
        '            ggml_tensor * lg = ggml_mul_mat(g, lm_w, xn);')
    return text.encode('utf-8')


if __name__ == '__main__':
    import sys
    Path(sys.argv[2]).write_bytes(transform(Path(sys.argv[1]).read_bytes()))
