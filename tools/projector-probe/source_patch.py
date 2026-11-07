"""Pinned reference-only output observer. No product/vendor changes."""
import hashlib
from pathlib import Path

STAGES = ['inp_raw','patch_bias','learned_pos','inp_pos_emb','ln1-0','ffn_inp_normed-0'] + ['layer_out-'+str(i) for i in range(24)]


def patch(root):
    root=Path(root);rows={}
    for path,digest in PINNED.items():
        file=root/path;raw=file.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('pinned reference source drift '+path)
        text=raw.decode()
        if path.endswith('qwen3vl.cpp'):
            anchor='    inp = ggml_add(ctx0, inp, learned_pos_embd);'
            if text.count(anchor)!=1:raise ValueError('position observer anchor')
            text=text.replace(anchor,'    cb(learned_pos_embd, "learned_pos", -1);\n'+anchor)
        else:
            names=', '.join('"'+s+'"'for s in STAGES)
            helper='''
#include <fstream>
#include <cstdlib>
#include <stdexcept>
static std::vector<ggml_tensor*> crisp_projector_stages(ggml_cgraph* gf, bool mark) {
    if (!std::getenv("CRISP_PROJECTOR_CAPTURE")) return {};
    const char* names[] = {NAMES};
    std::vector<ggml_tensor*> selected;
    for (const char* name : names) {
        auto* t=ggml_graph_get_tensor(gf,name);
        if (!t || t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nbytes(t)>32*1024*1024)
            throw std::runtime_error("unsupported materialized projector stage");
        if(mark) ggml_set_output(t);
        selected.push_back(t);
    }
    return selected;
}
static void crisp_projector_dump(ggml_cgraph* gf) {
    const char* dir=std::getenv("CRISP_PROJECTOR_CAPTURE");if(!dir)return;
    for(auto* t:crisp_projector_stages(gf,false)) {
        std::vector<float> v(ggml_nelements(t));
        ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));
        std::ofstream f(std::string(dir)+"/"+t->name+".f32",std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()),ggml_nbytes(t));
        if(!f)throw std::runtime_error("projector stage write");
        std::ofstream m(std::string(dir)+"/"+t->name+".shape");
        for(int i=0;i<4;i++)m<<t->ne[i]<<" "<<t->nb[i]<<"\\n";
        if(!m)throw std::runtime_error("projector shape write");
    }
}
'''.replace('NAMES',names)
            anchor='    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), gf)) {'
            if text.count(anchor)!=1:raise ValueError('allocation observer anchor')
            text=text.replace(anchor,'    crisp_projector_stages(gf,true);\n'+anchor)
            anchor='    if (embeddings != nullptr) {\n        // sanity check'
            if text.count(anchor)!=1:raise ValueError('dump observer anchor')
            text=text.replace(anchor,'    crisp_projector_dump(gf);\n\n'+anchor)
            # All ggml/standard includes are already in scope before this implementation.
            anchor='ggml_tensor * clip_graph::build_mm('
            if text.count(anchor)!=1:raise ValueError('helper observer anchor')
            text=text.replace(anchor,helper+'\n'+anchor)
        file.write_text(text);rows[path]={'original_sha256':digest,'patched_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    return rows

PINNED = {'tools/mtmd/clip.cpp': 'c43db7f76964cb3855c09851156400e4a6cab7ff41dc0972dc4ffb9a9dd31f5e', 'tools/mtmd/models/qwen3vl.cpp': '26e0da9db8258d5dcb6c3b75edea86caeeb9bad3ac4f378c5d508f7304d74781'}
