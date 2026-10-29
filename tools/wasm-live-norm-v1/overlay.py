"""Exact disposable-source observer overlay; never edits the guarded checkout."""
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from successor_wasm_trace_contract import replace_one
from wasm_q8_rounding_ab_v1 import overlay_ab
from wasm_layer3_attribution_v1 import NODES

def encoder(text):
 text=replace_one(text,'#include "core/wasm_native_order_norm.h"',
                  '#include "core/wasm_native_order_norm.h"\n#include "live_norm_observer.h"')
 text=replace_one(text,'    bool bench = false;', '    live_norm_observer live_norm;\n    bool bench = false;')
 anchor='''    ctx->wasm_norm_parameters.epsilon = hp.layer_norm_eps;
#endif
    return true;
}'''
 text=replace_one(text,anchor,anchor.replace('    return true;', '''    bool live_enabled = false;
#if defined(__EMSCRIPTEN__)
    live_enabled = ctx->wasm_native_order_norm_enabled;
#endif
    ctx->live_norm.configure(gguf_arch.c_str(), hp.n_embd, hp.n_layer, hp.n_head, hp.n_intermediate,
        ctx->pre_ln, hp.layer_norm_eps, ggml_backend_is_cpu(ctx->backend), live_enabled);
    return true;'''))
 text=replace_one(text,'    const int TB = T * B; // total tokens in batch',
                  '    ctx->live_norm.begin(ctx->sched);\n    const int TB = T * B; // total tokens in batch')
 anchor='''    auto bert_norm = [&](ggml_tensor * input) {'''
 end='''        return ggml_norm(gctx, input, ln_eps);
    };'''
 at=text.index(anchor);stop=text.index(end,at)+len(end)
 old=text[at:stop]
 assert old.count('wasm_native_order_norm::make')==1
 new='''    auto bert_norm = [&](ggml_tensor * input) {
        ggml_tensor * result = nullptr;
#if defined(__EMSCRIPTEN__)
        if (ctx->wasm_native_order_norm_enabled) {
            result = wasm_native_order_norm::make(gctx, input, &ctx->wasm_norm_parameters);
            GGML_ASSERT(result != nullptr);
        } else
#endif
        {
            result = ggml_norm(gctx, input, ln_eps);
        }
        GGML_ASSERT(ctx->live_norm.add_norm(input, result));
        return result;
    };'''
 text=text[:at]+text[at:].replace(old,new,1)
 anchor='        ggml_tensor * inp = cur; // save for residual connection'
 text=replace_one(text,anchor,'''        auto layer3 = [&](const char * name, ggml_tensor * tensor) {
            if (il == 3) GGML_ASSERT(ctx->live_norm.add(std::string("layer3_") + name, tensor));
        };
        layer3("input", cur);
'''+anchor)
 text=replace_one(text,'            V = ggml_mul_mat(gctx, L.v_w, cur);\n            if (L.q_b)',
                  '            V = ggml_mul_mat(gctx, L.v_w, cur);\n            layer3("q_raw", Q); layer3("k_raw", K); layer3("v_raw", V);\n            if (L.q_b)')
 text=replace_one(text,'        // Reshape for attention: [H, T*B]',
                  '        layer3("q", Q); layer3("k", K); layer3("v", V);\n        // Reshape for attention: [H, T*B]')
 old='''        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);'''
 text=replace_one(text,old,'''        layer3("flash", attn);
        attn = ggml_mul_mat(gctx, L.o_w, attn);
        layer3("o_raw", attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);
        layer3("o", attn);''')
 for kind,residual,ln in [('attention','inp, attn','ln1'),('ffn','cur, ffn','ln2')]:
  old=f'''            cur = ggml_add(gctx, {residual});
            cur = bert_norm(cur);
            cur = ggml_mul(gctx, cur, L.{ln}_w);
            if (L.{ln}_b) cur = ggml_add(gctx, cur, L.{ln}_b);'''
  new=old.replace('            cur = bert_norm',f'            layer3("{kind}_residual", cur);\n            cur = bert_norm')
  new=new.replace('            cur = ggml_mul',f'            layer3("{kind}_norm", cur);\n            cur = ggml_mul')
  new=new.replace(f'            if (L.{ln}_b)',f'            layer3("{kind}_scaled", cur);\n            if (L.{ln}_b)')
  new+='\n            layer3("'+('attention_output'if kind=='attention'else'output')+'", cur);'
  text=replace_one(text,old,new)
 old='''            ffn = ggml_mul_mat(gctx, L.fc1_w, cur);
            if (L.fc1_b) ffn = ggml_add(gctx, ffn, L.fc1_b);'''
 text=replace_one(text,old,old.replace('            if (L.fc1_b)','            layer3("fc1_raw", ffn);\n            if (L.fc1_b)')+'\n            layer3("fc1", ffn);')
 old='''            ffn = ggml_gelu_erf(gctx, ffn);
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
            if (L.fc2_b) ffn = ggml_add(gctx, ffn, L.fc2_b);'''
 text=replace_one(text,old,'''            ffn = ggml_gelu_erf(gctx, ffn);
            layer3("gelu", ffn);
            ffn = ggml_mul_mat(gctx, L.fc2_w, ffn);
            layer3("fc2_raw", ffn);
            if (L.fc2_b) ffn = ggml_add(gctx, ffn, L.fc2_b);
            layer3("fc2", ffn);''')
 text+='''
// Test-only completion export; no source/output injection.
extern "C" int crispembed_live_norm_finish(crispembed_context * ctx) {
    return ctx && ctx->live_norm.finish() ? 1 : 0;
}
'''
 assert all(('layer3("'+n+'"')in text for n in NODES)
 return text

def apply(source):
 source=Path(source);changes=overlay_ab(source)
 p=source/'src/crispembed.cpp';p.write_text(encoder(p.read_text()))
 changes['src/crispembed.cpp']['after']=hashlib.sha256(p.read_bytes()).hexdigest()
 target=source/'src/live_norm_observer.h';target.write_bytes((ROOT/'tests/wasm-live-norm/observer.h').read_bytes())
 changes['src/live_norm_observer.h']={'before':None,'after':hashlib.sha256(target.read_bytes()).hexdigest()}
 p=source/'build-embed-wasm.sh';text=p.read_text();text=replace_one(text,"'_wasm_trace_setenv',", "'_crispembed_live_norm_finish',\\\n'_wasm_trace_setenv',")
 p.write_text(text);changes['build-embed-wasm.sh']['after']=hashlib.sha256(p.read_bytes()).hexdigest()
 return changes
