#!/usr/bin/env python3
"""Frozen negative consumer and exact disposable-source layer3 overlays (stdlib)."""
import json
import math
from pathlib import Path
import re
import shlex
import struct
from successor_wasm_trace_contract import (MODEL_SHA, STAGES, TEXT, digest, f32_bytes,
    floats, metrics, tokens, gguf, replace_one)

BASE='580e3ae3be3671ddce5e945b0725d0583ec34110'
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
ENCODER='99bb8a2308021ce997e1570927322d749155c89155dd08af11fdcac6b99e7be1'
AB_RUN=40391544739
AB_REVIEW={'id':12692262412,'bytes':503633,'sha256':'6185d3a2b0776a157ab0dc10f25ddcc22406ba847dbb70b0b2638d43483bb047'}
AB_RAW={'id':12692387342,'bytes':2357641,'sha256':'0ac59ef65d00c4f267791b47323dc5705a958bdfab4805e1f8337d2eb6da65da'}
AB_REPORT='85102ccff26f3579f050a62a2747ef3f423b2546fc4dd9454e5b12663be76526'
NODES=('input','q_raw','k_raw','v_raw','q','k','v','flash','o_raw','o',
       'attention_residual','attention_norm','attention_scaled','attention_output',
       'fc1_raw','fc1','gelu','fc2_raw','fc2','ffn_residual','ffn_norm','ffn_scaled','output')
NAMES=tuple('layer3_'+n for n in NODES)
WEIGHTS={
 'blk.3.attn_q.weight':('input','q_raw'), 'blk.3.attn_k.weight':('input','k_raw'),
 'blk.3.attn_v.weight':('input','v_raw'), 'blk.3.attn_output.weight':('flash','o_raw'),
 'blk.3.ffn_up.weight':('attention_output','fc1_raw'),
 'blk.3.ffn_down.weight':('gelu','fc2_raw')}


def trusted_ab(report):
    expected=[c+'/1 final gates' for c in ('original','empty','punctuation-currency',
        'mixed-scripts','whitespace','numbers','near-limit','truncating')]
    for thread in ('1','2'):
        expected += ['original/1/stage/'+thread+'/'+s for s in ('layer_3','layer_4','encoder_out')]
        expected += ['original/1/recomputed-mean/'+thread]
    if (report.get('schema')!='crispembed-wasm-q8-rounding-ab-result-v1' or
            report.get('source')!=BASE or report.get('state')!='completed-rejected' or
            report.get('qualification') is not False or report.get('passed') is not False or
            report.get('failures')!=expected):
        raise ValueError('Reviewed negative A/B provenance differs')
    for case in report['cases'].values():
        if set(case)!=set(('unset','empty','0','1')):raise ValueError('Gate coverage differs')
        for gate,row in case.items():
            if row['ordinary_dump_bit_exact'] is not True or gate!='1' and row['legacy_bit_exact'] is not True:
                raise ValueError('Negative A/B neutrality/legacy controls differ')
    if len(report['cases'])!=10:raise ValueError('Panel coverage differs')
    for gate in ('unset','empty','0'):
        for thread in ('1','2'):
            rows=report['first_sentence_stages'][gate][thread]
            if [r['stage'] for r in rows]!=list(STAGES) or not all(r['old_wasm_bit_exact'] for r in rows):
                raise ValueError('Frozen legacy stage anchors differ')
    return report


def layer_rows(data):
    meta,rows=gguf(data,True)
    if meta.get('general.architecture')!='crispembed-encoder-dump' or meta.get('dump.n_layer')!=6 or meta.get('dump.n_embd')!=384:
        raise ValueError('Layer3 graph architecture differs')
    result={r['name']:r for r in rows}
    if set(result)!=set(STAGES+NAMES):raise ValueError('Missing/extra layer3 stage')
    for name,row in result.items():
        if row['shape']!=[1536 if name in ('layer3_fc1_raw','layer3_fc1','layer3_gelu') else 384,12]:
            raise ValueError('Layer3 graph shape differs')
    return result


def row_metrics(a,b,width):
    if width<=0 or len(a)%width or len(a)!=len(b):raise ValueError('Projection row shape differs')
    result=metrics(a,b)
    result['min_row_cosine']=min(metrics(a[i:i+width],b[i:i+width])['cosine'] for i in range(0,len(a),width))
    result['gate_pass']=result['max_abs']<=.002 and result['min_row_cosine']>=.99999
    return result


def conversion_blocks(data,kind,width,rows=12):
    """Actual workspace scales/quant bytes; Q8_K sums are preserved separately."""
    if kind==8: block,step=32,34
    elif kind==15: block,step=256,292
    else:raise ValueError('Unexpected actual activation conversion type')
    if width%block or len(data)!=rows*(width//block)*step:raise ValueError('Converted workspace shape differs')
    result=[]
    for row in range(rows):
        blocks=[]
        for b in range(width//block):
            piece=data[(row*width//block+b)*step:(row*width//block+b+1)*step]
            scale=struct.unpack('<e' if kind==8 else '<f',piece[:2 if kind==8 else 4])[0]
            if not math.isfinite(scale):raise ValueError('Nonfinite activation scale')
            offset=2 if kind==8 else 4
            blocks.append({'scale':scale,'quant_sha256':digest(piece[offset:offset+block]),
                'signed_min':min(struct.unpack('<'+'b'*block,piece[offset:offset+block])),
                'signed_max':max(struct.unpack('<'+'b'*block,piece[offset:offset+block])),
                'block_sha256':digest(piece),
                'sum_bytes_sha256':digest(piece[offset+block:]) if kind==15 else None})
        result.append(blocks)
    return result


def check_wasm_commands(rows):
    required=('src/crispembed.cpp','ggml/src/ggml-cpu/ggml-cpu.c',
              'ggml/src/ggml-cpu/arch/wasm/quants.c','wasm/embed_wrapper.c')
    observations={name:[] for name in required}
    for row in rows:
        matches=[name for name in required if row.get('file','').replace('\\','/').endswith('/'+name)]
        if not matches:continue
        args=row.get('arguments') or shlex.split(row['command'])
        simd_required=matches[0].startswith('ggml/')
        if ('-pthread' not in args or ('-msimd128' in args)!=simd_required or
                any(flag in args for flag in ('-mno-simd128','-mno-atomics','-sUSE_PTHREADS=0','-sPTHREADS=0')) or
                not any(Path(arg).name in ('emcc','em++','emcc.py','em++.py') for arg in args)):
            raise ValueError('Actual Emscripten SIMD/pthread compile flags differ: '+matches[0])
        for name in matches:observations[name].append(args)
    if any(not commands for commands in observations.values()):
        raise ValueError('Actual WASM compile-command coverage differs')
    return {'translation_units':{name:{'commands':len(commands),'pthread':True,
                'simd128':name.startswith('ggml/')} for name,commands in observations.items()},'passed':True}


def check_native_export(symbols):
    rows=[line.split() for line in symbols.splitlines() if line.split() and line.split()[-1]=='crispembed_layer3_replay']
    if len(rows)!=1 or len(rows[0])<3 or rows[0][-2]!='T':
        raise ValueError('Diagnostic native C replay export missing/ambiguous')
    return {'symbol':'crispembed_layer3_replay','dynamic_type':'T','passed':True}


def overlay_encoder(text):
    if digest(text.encode())!=ENCODER:raise ValueError('Frozen encoder bytes differ')
    anchor='''        ggml_tensor * inp = cur; // save for residual connection'''
    hook='''        auto layer3 = [&](const char * name, ggml_tensor * tensor) {
            if (il != 3 || !ctx->dump_layers || !core_env::on("CRISPEMBED_LAYER3_TRACE")) return;
            // Independent identity view names never rename an ordinary graph node.
            // ggml_view_tensor creates a metadata alias with no graph edge.
            // Use GGML_OP_VIEW/src[0] so scheduling and reservation see the owner.
            ggml_tensor * view = ggml_view_4d(gctx, tensor,
                tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
            for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
                GGML_ASSERT(view->ne[axis] == tensor->ne[axis]);
                GGML_ASSERT(view->nb[axis] == tensor->nb[axis]);
            }
            GGML_ASSERT(view->op == GGML_OP_VIEW && view->src[0] == tensor);
            ggml_set_name(view, (std::string("layer3_") + name).c_str());
            for (ggml_tensor * owner = view; owner; owner = owner->view_src) ggml_set_output(owner);
            ggml_build_forward_expand(gf, view);
        };
        layer3("input", cur);
'''
    text=replace_one(text,anchor,hook+anchor)
    old='''            V = ggml_mul_mat(gctx, L.v_w, cur);
            if (L.q_b)'''
    text=replace_one(text,old,'''            V = ggml_mul_mat(gctx, L.v_w, cur);
            layer3("q_raw", Q); layer3("k_raw", K); layer3("v_raw", V);
            if (L.q_b)''')
    text=replace_one(text,'        // Reshape for attention: [H, T*B]',
        '        layer3("q", Q); layer3("k", K); layer3("v", V);\n\n        // Reshape for attention: [H, T*B]')
    old='''        attn = ggml_mul_mat(gctx, L.o_w, attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);'''
    text=replace_one(text,old,'''        layer3("flash", attn);
        attn = ggml_mul_mat(gctx, L.o_w, attn);
        layer3("o_raw", attn);
        if (L.o_b) attn = ggml_add(gctx, attn, L.o_b);
        layer3("o", attn);''')
    old='''            cur = ggml_add(gctx, inp, attn);
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln1_w);
            if (L.ln1_b) cur = ggml_add(gctx, cur, L.ln1_b);'''
    new=old.replace('            cur = ggml_norm', '            layer3("attention_residual", cur);\n            cur = ggml_norm')
    new=new.replace('            cur = ggml_mul', '            layer3("attention_norm", cur);\n            cur = ggml_mul')
    new=new.replace('            if (L.ln1_b)', '            layer3("attention_scaled", cur);\n            if (L.ln1_b)')+'\n            layer3("attention_output", cur);'
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
    old='''            cur = ggml_add(gctx, cur, ffn);
            cur = ggml_norm(gctx, cur, ln_eps);
            cur = ggml_mul(gctx, cur, L.ln2_w);
            if (L.ln2_b) cur = ggml_add(gctx, cur, L.ln2_b);'''
    new=old.replace('            cur = ggml_norm','            layer3("ffn_residual", cur);\n            cur = ggml_norm')
    new=new.replace('            cur = ggml_mul','            layer3("ffn_norm", cur);\n            cur = ggml_mul')
    new=new.replace('            if (L.ln2_b)','            layer3("ffn_scaled", cur);\n            if (L.ln2_b)')+'\n            layer3("output", cur);'
    text=replace_one(text,old,new)
    anchor='        names.push_back("encoder_out");'
    text=replace_one(text,anchor,anchor+'\n        if (core_env::on("CRISPEMBED_LAYER3_TRACE")) {\n'+
        ''.join('            names.push_back("'+name+'");\n' for name in NAMES)+'        }')
    return text


def overlay(source,probe_root):
    source,probe_root=Path(source),Path(probe_root)
    changes={}
    def write(name,content):
        p=source/name;before=p.read_bytes();p.write_text(content)
        changes[name]={'before':digest(before),'after':digest(p.read_bytes())}
    p=source/'src/crispembed.cpp'
    text=overlay_encoder(p.read_text())
    text+='\n'+(probe_root/'tests/wasm-layer3/replay.inc').read_text()
    write('src/crispembed.cpp',text)
    name='ggml/src/ggml-cpu/ggml-cpu.c';text=(source/name).read_text()
    if digest(text.encode())!='b8c04a65c4ee295428501c0951c52b334920fef9101cb100061731b1b9fbee94':
        raise ValueError('Frozen CPU kernel source differs')
    # First barrier occurs in mul_mat; require complete following anchor so a
    # different operation/earlier synchronization cannot accidentally match.
    anchor='''    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {'''
    text=replace_one(text,anchor,'    ggml_barrier(params->threadpool);\n\n'+
        (probe_root/'tests/wasm-layer3/kernel_capture.inc').read_text()+
        '\n#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {')
    write(name,text)
    name='wasm/embed_wrapper.c';text=(source/name).read_text()
    text+='\n/* Test-only overlay. */\nWASM_EXPORT\nint wasm_trace_setenv(const char * k, const char * v) { return v ? setenv(k,v,1) : unsetenv(k); }\n'
    write(name,text)
    name='build-embed-wasm.sh';text=(source/name).read_text()
    text=replace_one(text,"'_wasm_embed_version',","'_wasm_trace_setenv',\\\n'_crispembed_layer3_replay',\\\n'_wasm_embed_version',")
    text=replace_one(text,'-j$(nproc 2>/dev/null || echo 4)','-j2')
    write(name,text)
    return changes
