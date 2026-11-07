"""Live, prefill and one same-prefix divergence reference observer for pinned upstream quantized text math."""
import hashlib
from pathlib import Path
SOURCE='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
ENV='CRISPEMBED_TEST_UPSTREAM_TEXT_FORM_TRACE_DIR'
DONE='CRISPEMBED_TEST_UPSTREAM_TEXT_FORM_TRACE_SAMPLES'
CONTEXT_SHA='9502cb706594752543975730f0b769b043c430dda4b62b7a172e35b4150ff43c'
# Existing immutable feature/sampler observer must already have been applied.
SAMPLING_SHA='b01b9068378e045898607f46bfd99ab7d2096e53cb845d7d1507d5e399f94fa6'
STAGES=['model.input_embed','attn_norm-0','linear_attn_qkv_mixed-0','z-0','alpha-0','beta-0','conv_output_raw-0','conv_output_silu-0','state_predelta-0','final_output-0','linear_attn_out-0','attn_residual-0','attn_post_norm-0','ffn_gate-0','ffn_up-0','ffn_swiglu-0','ffn_out-0','result_norm','result_output']+['l_out-'+str(i)for i in range(32)]
INPUTS=['inp_tokens','inp_mixed_tokens','inp_pos','out_ids']
HELPER=r'''
#include <fstream>
#include <cstdlib>
#include <map>
#include <set>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <string>
namespace {
struct crisp_text_trace_state {
 int graph=-1, sample_count=0; size_t total=0; std::set<std::string> seen; std::ofstream index;
};
static crisp_text_trace_state crisp_text_trace;
static int crisp_text_sample_count(){const char*v=std::getenv("TRACE_DONE");return v?std::atoi(v):0;}
static bool crisp_text_trace_active(){const int n=crisp_text_sample_count();return std::getenv("TRACE_ENV")&&(n==0||n==31);}
static bool crisp_text_trace_wanted(const char*name){const char*names[]={STAGE_NAMES};for(const char*x:names)if(std::strcmp(name,x)==0)return true;return false;}
static void crisp_text_trace_dump(ggml_tensor*t) {
 if(!t||!crisp_text_trace.seen.insert(t->name).second)throw std::runtime_error("duplicate/null live text tensor");
 const size_t bytes=ggml_nbytes(t);const char*status="captured";
 if(!ggml_nelements(t))status="zero-sized";
 else if(!ggml_is_contiguous(t))status="unsupported-strides";
 else if(t->type!=GGML_TYPE_F32&&t->type!=GGML_TYPE_I32)status="unsupported-type";
 else if(bytes>(8u<<20))status="unsupported-size";
 else if(!t->buffer||!t->data)status="unallocated";
 std::string file="graph-"+std::to_string(crisp_text_trace.graph)+"."+t->name+".bin";
 if(std::strcmp(status,"captured")==0){
  if(bytes>(128u<<20)-crisp_text_trace.total)throw std::runtime_error("bounded text trace total bytes");
  std::vector<char> v(bytes);ggml_backend_tensor_get(t,v.data(),0,bytes);
  std::ofstream f(std::string(std::getenv("TRACE_ENV"))+"/"+file,std::ios::binary);f.write(v.data(),bytes);
  if(!f)throw std::runtime_error("live text tensor write");crisp_text_trace.total+=bytes;
 }
 auto&f=crisp_text_trace.index;f<<t->name<<"\t"<<crisp_text_trace.sample_count<<"\t"<<int(t->type)<<"\t"<<int(t->op);
 for(int i=0;i<4;i++)f<<"\t"<<t->ne[i];for(int i=0;i<4;i++)f<<"\t"<<t->nb[i];
 f<<"\t"<<bytes<<"\t"<<status<<"\t"<<file<<"\n";f.flush();if(!f)throw std::runtime_error("live text index write");
}
static bool crisp_text_trace_eval(ggml_tensor*t,bool ask,void*) {
 bool wanted=crisp_text_trace_wanted(t->name);if(ask)return wanted;if(wanted)crisp_text_trace_dump(t);return true;
}
static bool crisp_text_trace_begin(ggml_cgraph*gf,ggml_backend_sched_t sched) {
 if(!crisp_text_trace_active()||!ggml_graph_get_tensor(gf,"model.input_embed"))return false;
 if(++crisp_text_trace.graph>=16)throw std::runtime_error("bounded prefill graph count");
 crisp_text_trace.sample_count=crisp_text_sample_count();crisp_text_trace.seen.clear();crisp_text_trace.index.open(std::string(std::getenv("TRACE_ENV"))+"/graph-"+std::to_string(crisp_text_trace.graph)+".tsv");
 if(!crisp_text_trace.index)throw std::runtime_error("text graph index open");
 crisp_text_trace.index<<"name\tsample_count\ttype\top\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tbytes\tstatus\tfile\n";
 const char*inputs[]={INPUT_NAMES};for(const char*name:inputs){auto*t=ggml_graph_get_tensor(gf,name);if(t)crisp_text_trace_dump(t);}
 // A direct image-embedding input can be a leaf; token GET_ROWS is observed live.
 auto*t=ggml_graph_get_tensor(gf,"model.input_embed");if(t&&t->op==GGML_OP_NONE)crisp_text_trace_dump(t);
 ggml_backend_sched_set_eval_callback(sched,crisp_text_trace_eval,nullptr);return true;
}
static void crisp_text_trace_end(){crisp_text_trace.index.close();}
}
'''
def patch(root):
 root=Path(root);context=root/'src/llama-context.cpp';sampling=root/'common/sampling.cpp'
 for p,digest in [(context,CONTEXT_SHA),(sampling,SAMPLING_SHA)]:
  if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('pinned text trace source drift: '+p.name)
 helper=HELPER.replace('TRACE_ENV',ENV).replace('TRACE_DONE',DONE).replace('STAGE_NAMES',','.join('"'+x+'"'for x in STAGES)).replace('INPUT_NAMES',','.join('"'+x+'"'for x in INPUTS))
 text=context.read_text();anchor='ggml_status llama_context::graph_compute('
 if text.count(anchor)!=1:raise ValueError('context helper anchor')
 text=text.replace(anchor,helper+'\n'+anchor)
 anchor='    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);'
 if text.count(anchor)!=1:raise ValueError('actual text compute anchor')
 replacement='''    if (crisp_text_trace_active() && cparams.cb_eval) throw std::runtime_error("text callback already occupied");
    const bool crisp_observed = crisp_text_trace_begin(gf, sched.get());
'''+anchor+'''
    if (crisp_observed) {
        ggml_backend_sched_synchronize(sched.get());
        crisp_text_trace_end();
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);
    }'''
 context.write_text(text.replace(anchor,replacement))
 text=sampling.read_text();anchor='    llama_synchronize(ctx);'
 if text.count(anchor)!=1:raise ValueError('sampling prefill-stop anchor')
 text=text.replace(anchor,anchor+f'''\n    if (const char*dir=std::getenv("{ENV}")) {{
        static int samples=0;
        if(samples>=512)throw std::runtime_error("bounded observed sampling count");
        std::ofstream f(std::string(dir)+"/sampling-steps.tsv",std::ios::app);f<<samples<<"\\n";
        if(!f||setenv("{DONE}",std::to_string(++samples).c_str(),1))throw std::runtime_error("text sampling marker");
    }}''')
 sampling.write_text(text)
 return {str(p.relative_to(root)):{'before_sha256':digest,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p,digest in [(context,CONTEXT_SHA),(sampling,SAMPLING_SHA)]}
