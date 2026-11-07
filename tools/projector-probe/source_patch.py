"""Pinned reference-only LIVE observer; no product/vendor arithmetic changes."""
import hashlib
from pathlib import Path
STAGES=['inp_raw','patch_bias','learned_pos','inp_pos_emb','ln1-0','ffn_inp_normed-0','ffn_inp-0','ffn_out-0']+['layer_out-'+str(i)for i in range(24)]


def patch(root):
 root=Path(root);rows={}
 for name,digest in PINNED.items():
  file=root/name;raw=file.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('pinned source drift '+name)
  text=raw.decode()
  if name.endswith('qwen3vl.cpp'):
   anchor='    inp = ggml_add(ctx0, inp, learned_pos_embd);'
   if text.count(anchor)!=1:raise ValueError('position observer anchor')
   text=text.replace(anchor,'    cb(learned_pos_embd, "learned_pos", -1);\n'+anchor)
  elif name.endswith('mtmd-image.cpp'):
   helper='''
#include <fstream>
#include <cstdlib>
#include <stdexcept>
static void crisp_projector_resized(const clip_image_u8& image) {
 const char*dir=std::getenv("CRISP_PROJECTOR_CAPTURE");if(!dir)return;
 const auto&rgb=image.get_ro_buf();std::ofstream f(std::string(dir)+"/resized.rgb",std::ios::binary);
 f.write(reinterpret_cast<const char*>(rgb.data()),rgb.size());if(!f)throw std::runtime_error("resized RGB capture");
}
'''
   anchor='void mtmd_image_preproc_out::append('
   at=text.index(anchor);text=text[:at]+helper+'\n'+text[at:]
   at=text.index('mtmd_image_preproc_out mtmd_image_preprocessor_dyn_size::preprocess(');end=text.index('\n}\n',at)
   part=text[at:end];anchor='    mtmd_image_preproc_out output;'
   if part.count(anchor)!=1:raise ValueError('resize observer anchor')
   text=text[:at]+part.replace(anchor,'    crisp_projector_resized(resized_image);\n'+anchor)+text[end:]
  else:
   names=', '.join('"'+s+'"'for s in STAGES)
   helper='''
#include <fstream>
#include <cstdlib>
#include <stdexcept>
static std::map<std::string,int> crisp_projector_counts;
static bool crisp_projector_wanted(const char*name) {
 const char* names[]={NAMES};for(const char*x:names)if(std::strcmp(name,x)==0)return true;return false;
}
static void crisp_projector_dump_live(ggml_tensor*t) {
 const char*dir=std::getenv("CRISP_PROJECTOR_CAPTURE");if(!dir)return;
 if(!t||t->type!=GGML_TYPE_F32||!ggml_is_contiguous(t)||ggml_nbytes(t)>32*1024*1024)
  throw std::runtime_error("unsupported live projector tensor");
 if(++crisp_projector_counts[t->name]!=1)throw std::runtime_error("duplicate live projector tensor");
 std::vector<float> v(ggml_nelements(t));ggml_backend_tensor_get(t,v.data(),0,ggml_nbytes(t));
 std::ofstream f(std::string(dir)+"/"+t->name+".f32",std::ios::binary);
 f.write(reinterpret_cast<const char*>(v.data()),ggml_nbytes(t));if(!f)throw std::runtime_error("live projector capture");
 std::ofstream m(std::string(dir)+"/"+t->name+".shape");for(int i=0;i<4;i++)m<<t->ne[i]<<" "<<t->nb[i]<<"\\n";
 if(!m)throw std::runtime_error("live projector shape");
}
static bool crisp_projector_eval(ggml_tensor*t,bool ask,void*) {
 bool want=crisp_projector_wanted(t->name)&&std::strcmp(t->name,"inp_raw")!=0;
 if(ask)return want;if(want)crisp_projector_dump_live(t);return true;
}
static void crisp_projector_begin(clip_ctx*ctx,ggml_cgraph*gf) {
 if(!std::getenv("CRISP_PROJECTOR_CAPTURE"))return;
 crisp_projector_counts.clear();auto*t=ggml_graph_get_tensor(gf,"inp_raw");crisp_projector_dump_live(t);
 ggml_backend_sched_set_eval_callback(ctx->sched.get(),crisp_projector_eval,nullptr);
}
static void crisp_projector_end(clip_ctx*ctx) {
 const char*dir=std::getenv("CRISP_PROJECTOR_CAPTURE");if(!dir)return;
 ggml_backend_sched_set_eval_callback(ctx->sched.get(),nullptr,nullptr);
 const char*names[]={NAMES};std::ofstream f(std::string(dir)+"/live-counts.tsv");
 for(const char*name:names){if(crisp_projector_counts[name]!=1)throw std::runtime_error("missing live projector tensor");f<<name<<"\\t"<<crisp_projector_counts[name]<<"\\n";}
 if(!f)throw std::runtime_error("live projector counts");
}
'''.replace('NAMES',names)
   anchor='ggml_tensor * clip_graph::build_mm('
   if text.count(anchor)!=1:raise ValueError('helper observer anchor')
   text=text.replace(anchor,helper+'\n'+anchor)
   anchor='    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);'
   if text.count(anchor)!=1:raise ValueError('live observer anchor')
   text=text.replace(anchor,'    crisp_projector_begin(ctx,gf);\n'+anchor)
   anchor='    if (embeddings != nullptr) {\n        // sanity check'
   if text.count(anchor)!=1:raise ValueError('live end observer anchor')
   text=text.replace(anchor,'    crisp_projector_end(ctx);\n\n'+anchor)
  file.write_text(text);rows[name]={'original_sha256':digest,'patched_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
 return rows
PINNED={'tools/mtmd/clip.cpp': 'c43db7f76964cb3855c09851156400e4a6cab7ff41dc0972dc4ffb9a9dd31f5e', 'tools/mtmd/models/qwen3vl.cpp': '26e0da9db8258d5dcb6c3b75edea86caeeb9bad3ac4f378c5d508f7304d74781', 'tools/mtmd/mtmd-image.cpp': '9f752e7d26cd607f2f0ffafd164859d76180299969de9b3bb30fccb75639bc63'}
