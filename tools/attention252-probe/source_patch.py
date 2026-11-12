"""Test-only source layer3 observer. Exact build/firstscore neutrality required."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ENV='CRISPEMBED_TEST_UPSTREAM_ATTENTION252_DIR'
DONE='CRISPEMBED_TEST_UPSTREAM_ATTENTION252_SAMPLES'
SOURCE='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
STAGES=['model.input_embed','result_norm','result_output']+['l_out-'+str(i)for i in range(32)]+[
 x+'-3'for x in ['attn_norm','Qcur_full','Qcur_reshaped','Qcur_normed','Kcur_normed','Qcur','Kcur','Vcur','gate_reshaped','kq','kq_soft_max','kqv','kqv_out','attn_pregate','gate_sigmoid','attn_gated','attn_output','attn_residual','attn_post_norm','ffn_gate','ffn_up','ffn_swiglu','ffn_out']]

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

def patch(root):return configured().patch(root)
