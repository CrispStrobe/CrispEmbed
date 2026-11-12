"""Test-only source layer3 observer. Exact build/firstscore neutrality required."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ENV='CRISPEMBED_TEST_UPSTREAM_ATTENTION252_DIR'
DONE='CRISPEMBED_TEST_UPSTREAM_ATTENTION252_SAMPLES'
SOURCE='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
STAGES=['model.input_embed','result_norm','result_output']+['l_out-'+str(i)for i in range(32)]+[
 x+'-3'for x in ['attn_norm','Qcur_full','Qcur_reshaped','Qcur_normed','Kcur_normed','Kcur_raw','Vcur_raw','Qcur','Kcur','Vcur','gate_reshaped','kq','kq_soft_max','kqv','kqv_out','attn_pregate','gate_sigmoid','attn_gated','attn_output','attn_residual','attn_post_norm','ffn_gate','ffn_up','ffn_swiglu','ffn_out']]

def configured():
 spec=importlib.util.spec_from_file_location('attention252_base',ROOT/'tools/form-text-trace-probe/source_patch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.ENV,m.DONE,m.STAGES=ENV,DONE,STAGES
 h=m.HELPER
 assert h.count('n==0||n==31')==1
 h=h.replace('n==0||n==31','n==0').replace('else if(!ggml_is_contiguous(t))status="unsupported-strides";','')
 h=h.replace('t->type!=GGML_TYPE_F32&&t->type!=GGML_TYPE_I32','t->type!=GGML_TYPE_F32&&t->type!=GGML_TYPE_I32&&t->type!=GGML_TYPE_F16')
 h=h.replace('bytes>(8u<<20)','bytes>(16u<<20)').replace('128u<<20','512u<<20')
 old='bool wanted=crisp_text_trace_wanted(t->name);if(ask)return wanted;if(wanted)crisp_text_trace_dump(t);return true;'
 new='''bool wanted=crisp_text_trace_wanted(t->name);if(ask)return wanted;if(wanted){
 crisp_text_trace_dump(t);
 if(std::strcmp(t->name,"kq-3")==0||std::strcmp(t->name,"kqv-3")==0){
  for(int i=0;i<2;++i){ggml_tensor operand=*t->src[i];
   std::string name=std::string("operand-")+t->name+"-"+std::to_string(i);ggml_set_name(&operand,name.c_str());crisp_text_trace_dump(&operand);}
 }
 }return true;'''
 assert h.count(old)==1;h=h.replace(old,new);m.HELPER=h
 return m

def disambiguate(text):
 anchor='    cb(Qcur_full, "Qcur_full", il);\n    cb(Kcur, "Kcur", il);\n    cb(Vcur, "Vcur", il);'
 if text.count(anchor)!=1 or text.count('cb(Kcur, "Kcur", il)')!=2 or text.count('cb(Vcur, "Vcur", il)')!=2:
  raise ValueError('exact raw/post-normalized source stage anchors')
 return text.replace(anchor,anchor.replace('"Kcur"','"Kcur_raw"').replace('"Vcur"','"Vcur_raw"'))

def patch(root):
 import hashlib
 root=Path(root);p=root/'src/models/qwen35.cpp';before=p.read_bytes()
 expected='3387cf23a95ca71d2a38fac4b32708d6424fc9888273148ff6b4db5e510c9406'
 if hashlib.sha256(before).hexdigest()!=expected:raise ValueError('pinned actual model blueprint')
 result=configured().patch(root);p.write_text(disambiguate(before.decode()))
 result['src/models/qwen35.cpp']={'before_sha256':expected,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'Two raw callback labels only; computation unchanged.'}
 return result
