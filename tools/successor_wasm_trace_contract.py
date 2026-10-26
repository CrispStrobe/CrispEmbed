#!/usr/bin/env python3
"""Small stdlib parsers and exact test-only overlay for the frozen WASM trace."""
import hashlib
import math
from pathlib import Path
import re
import struct

PRODUCER='40a82158146faf0cd44de2a4ee7056e7b0c6a58c'
INITIAL_RUN=40354251257
INITIAL_SOURCE='49161f608af811154bfeb816ac38caea34224cae'
MODEL_SHA='77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2'
NATIVE_REPORT_SHA='6cdd426e0b71fcbe80000494645deeb75fa75f9d5028e06d29abcdb9ff8cf435'
FIXTURE_ID=12671621333
FIXTURE_BYTES=18259261
FIXTURE_SHA='a67abb8439d2af274c175dad9cefb8ba9606bec9c4ba024cf568a7359f41c8bf'
STAGES=('emb_ln_out','attn_out_0',*(f'layer_{i}' for i in range(5)),'encoder_out')
TEXT='A small reproducible embedding consumer.'

def digest(data):return hashlib.sha256(data).hexdigest()

def f32_bytes(values):return struct.pack('<'+'f'*len(values),*values)

def floats(data):
    if not data or len(data)%4:raise ValueError('F32 byte length invalid')
    values=list(struct.unpack('<'+'f'*(len(data)//4),data))
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite F32')
    return values

def metrics(a,b):
    if not a or len(a)!=len(b) or not all(math.isfinite(x) for x in a+b):raise ValueError('Shape/finite mismatch')
    errors=[abs(x-y) for x,y in zip(a,b)]
    aa=sum(x*x for x in a);bb=sum(x*x for x in b)
    if aa==0 or bb==0:raise ValueError('Zero magnitude')
    return {'n':len(a),'max_abs':max(errors),'mean_abs':sum(errors)/len(a),
            'rms':math.sqrt(sum(x*x for x in errors)/len(a)),
            'native_l2':math.sqrt(aa),'wasm_l2':math.sqrt(bb),
            'cosine':sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb),
            'native_min':min(a),'native_max':max(a),'wasm_min':min(b),'wasm_max':max(b)}

def tokens(log):
    matches=re.findall(r'^crispembed: token_ids \(n=(\d+)\):([ 0-9]+)$',log,re.M)
    if len(matches)!=1:raise ValueError('Expected one actual token-ID line')
    n,text=matches[0];ids=[int(x) for x in text.split()]
    if len(ids)!=int(n) or not ids or any(x<0 or x>=30522 for x in ids):raise ValueError('Token count/range differs')
    return ids

class Reader:
    def __init__(self,data):self.data=data;self.pos=0
    def take(self,n):
        if n<0 or self.pos+n>len(self.data):raise ValueError('Truncated GGUF')
        b=self.data[self.pos:self.pos+n];self.pos+=n;return b
    def number(self,fmt):return struct.unpack('<'+fmt,self.take(struct.calcsize('<'+fmt)))[0]
    def string(self):
        n=self.number('Q')
        if n>1024*1024:raise ValueError('Oversize string')
        return self.take(n).decode('utf-8')
    def value(self,t,depth=0):
        if depth>2:raise ValueError('Nested GGUF arrays')
        if t==8:return self.string()
        if t==9:
            subtype=self.number('I');n=self.number('Q')
            if n>2_000_000:raise ValueError('Oversize GGUF array')
            # Metadata is parsed, never trusted as an allocation size.
            return [self.value(subtype,depth+1) for _ in range(n)]
        fmt={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}.get(t)
        if fmt is None:raise ValueError('Unknown GGUF metadata type')
        return self.number(fmt)

def gguf(data,with_floats=False):
    r=Reader(data)
    if r.take(4)!=b'GGUF' or r.number('I')!=3:raise ValueError('Expected GGUF v3')
    nt,nk=r.number('Q'),r.number('Q')
    if nt>10000 or nk>10000:raise ValueError('Oversize GGUF inventory')
    meta={}
    for _ in range(nk):
        key=r.string()
        if key in meta:raise ValueError('Duplicate metadata')
        meta[key]=r.value(r.number('I'))
    info=[];names=set()
    for _ in range(nt):
        name=r.string();nd=r.number('I')
        if name in names or nd<1 or nd>4:raise ValueError('Duplicate tensor/invalid rank')
        names.add(name);shape=[r.number('Q') for _ in range(nd)]
        if not all(0<x<=1_000_000 for x in shape):raise ValueError('Invalid shape')
        info.append({'name':name,'shape':shape,'type':r.number('I'),'offset':r.number('Q')})
    align=meta.get('general.alignment',32)
    if not isinstance(align,int) or align<1 or align>4096 or align&(align-1):raise ValueError('Invalid alignment')
    base=(r.pos+align-1)//align*align
    if base>len(data):raise ValueError('Missing GGUF tensor payload')
    for row in info:
        if row['offset']%align or base+row['offset']>=len(data):raise ValueError('Tensor offset invalid')
        row['data_offset']=base+row['offset']
        if with_floats:
            if row['type']!=0:raise ValueError('Trace tensor is not F32')
            n=math.prod(row['shape']);start=base+row['offset'];end=start+n*4
            if end>len(data):raise ValueError('Truncated tensor')
            row['values']=floats(data[start:end]);row['sha256']=digest(data[start:end])
    return meta,info

def stage_rows(data,token_count):
    meta,rows=gguf(data,True)
    if meta.get('general.architecture')!='crispembed-encoder-dump' or meta.get('dump.n_layer')!=6 or meta.get('dump.n_embd')!=384:
        raise ValueError('Unexpected trace architecture')
    result={r['name']:r for r in rows}
    if set(result)!=set(STAGES) or any(r['shape']!=[384,token_count] for r in rows):raise ValueError('Missing/wrong named stage')
    return result

def replace_one(text,old,new):
    if text.count(old)!=1:raise ValueError('Overlay source anchor differs')
    return text.replace(old,new)

def overlay(source):
    """Write only into a disposable hosted source copy. Arithmetic stays identical."""
    source=Path(source);changes={}
    p=source/'wasm/embed_wrapper.c';before=p.read_bytes()
    p.write_bytes(before+br'''
/* Test-only environment setter and operator controls; never shipped. */
#include "../ggml/src/ggml-quants.h"
#include "ggml-cpu.h"
WASM_EXPORT
int wasm_trace_setenv(const char * key, const char * value) {
    return value ? setenv(key, value, 1) : unsetenv(key);
}
WASM_EXPORT
int wasm_trace_q8(const float * x, int n, void * arch, void * ref) {
    if (n <= 0 || n > 384 || n % 32) return -1;
    const struct ggml_type_traits_cpu * trait = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!trait->from_float) return -2;
    trait->from_float(x, arch, n);
    quantize_row_q8_0_ref(x, (block_q8_0 *) ref, n);
    return (n / 32) * sizeof(block_q8_0);
}
WASM_EXPORT
int wasm_trace_simd(void) { return ggml_cpu_has_wasm_simd(); }
WASM_EXPORT
int wasm_trace_dot(int type, int n, const void * weight, const void * q8, float * out) {
    if (type != GGML_TYPE_IQ4_NL || n != 384) return -1;
    const struct ggml_type_traits_cpu * trait = ggml_get_type_traits_cpu((enum ggml_type) type);
    if (trait->vec_dot_type != GGML_TYPE_Q8_0 || !trait->vec_dot) return -2;
    trait->vec_dot(n, out, 0, weight, 0, q8, 0, 1);
    return 0;
}
''')
    changes['wasm/embed_wrapper.c']={'before':digest(before),'after':digest(p.read_bytes())}
    p=source/'src/crispembed.cpp';before=p.read_bytes();text=before.decode()
    statement='''    // Test-only actual pooling buffer capture, enabled only by this hosted overlay.
    if (const char * path = std::getenv("CRISPEMBED_TRACE_POOL")) {
        const std::string file = std::string(path) + "%s";
        FILE * f = fopen(file.c_str(), "wb");
        if (!f || fwrite(pooled.data(), sizeof(float), pooled.size(), f) != pooled.size()) std::abort();
        if (fclose(f) != 0) std::abort();
    }
'''
    anchor='    // L2 normalize\n    auto t_pool = std::chrono::steady_clock::now();'
    text=replace_one(text,anchor,(statement % '.mean.f32')+anchor)
    anchor='\n    return pooled;\n}\n\n// C3:'
    text=replace_one(text,anchor,'\n'+(statement % '.final.f32')+anchor)
    p.write_text(text,encoding='utf-8')
    changes['src/crispembed.cpp']={'before':digest(before),'after':digest(p.read_bytes())}
    p=source/'build-embed-wasm.sh';before=p.read_bytes();text=before.decode()
    text=replace_one(text,"'_wasm_embed_version',", "'_wasm_trace_setenv',\\\n'_wasm_trace_q8',\\\n'_wasm_trace_simd',\\\n'_wasm_trace_dot',\\\n'_wasm_embed_version',")
    text=replace_one(text,'-j$(nproc 2>/dev/null || echo 4)','-j2')
    p.write_text(text,encoding='utf-8')
    changes['build-embed-wasm.sh']={'before':digest(before),'after':digest(p.read_bytes())}
    return changes
