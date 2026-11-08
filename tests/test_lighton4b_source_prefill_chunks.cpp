// Model-free lifecycle/gate/cache controls. Source-fed numerical proof is separate.
#include "lighton4b_linear_block.h"
#include "lighton4b_attention_block.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace qwen35_ocr;
static void need(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static void analytical(size_t & checks){
 for(int width:{256,768,9216})for(int tokens:{8,9})for(int threads:{1,4}){
  const int rows=17,blocks=width/256;std::vector<unsigned char> packed(size_t(rows)*blocks*210);std::vector<float> x(size_t(tokens)*width);
  for(int t=0;t<tokens;++t)for(int k=0;k<width;++k)x[size_t(t)*width+k]=float((k+t*17)%255-127)/128.0f;
  for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b){auto*p=packed.data()+(size_t(r)*blocks+b)*210;std::fill_n(p,128,0x11);std::fill_n(p+128,64,0xaa);for(int g=0;g<16;++g)p[192+g]=static_cast<unsigned char>(((r+g)%2)?-1:1);const auto d=ggml_fp32_to_fp16(1.0f/16);std::memcpy(p+208,&d,2);}
  std::vector<float> y;need(lighton4b_physical_q6_tiled_cpu(packed.data(),packed.size(),x.data(),width,rows,tokens,threads,y),"analytical multi-block Q6");
  for(int t=0;t<tokens;++t)for(int r=0;r<rows;++r){int sum=0;for(int k=0;k<width;++k)sum+=(((r+(k%256)/16)%2)?-1:1)*((k+t*17)%255-127);need(y[size_t(t)*rows+r]==float(sum)/2048.0f,"independent dyadic integer result");}++checks;
 }
}
static void actual_route(size_t & checks){
 const int width=768,rows=17,tokens=8;const size_t bytes=size_t(rows)*(width/256)*210;
 auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({65536,nullptr,true}),ggml_free);need(bool(ctx),"route context");
 auto storage=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),bytes),ggml_backend_buffer_free);need(bool(storage),"route aligned buffer");auto *p=static_cast<unsigned char*>(ggml_backend_buffer_get_base(storage.get()));
 for(size_t b=0;b<bytes/210;++b){for(int j=0;j<192;++j)p[b*210+j]=static_cast<unsigned char>((b*37+j*13+7)%256);for(int j=0;j<16;++j)p[b*210+192+j]=static_cast<unsigned char>(int((b+j*7)%31)-15);const auto d=ggml_fp32_to_fp16(.013f+float(b%7)*.0003f);std::memcpy(p+b*210+208,&d,2);}
 auto*w=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q6_K,width,rows);w->data=p;w->buffer=storage.get();std::vector<float>x(size_t(tokens)*width);for(size_t k=0;k<x.size();++k)x[k]=std::sin(float(k)*.17f)*.43f;
 std::vector<float> expected;need(lighton4b_physical_q6_tiled_cpu(p,bytes,x.data(),width,rows,tokens,4,expected),"nontrivial helper");std::vector<std::vector<float>> on,off,repeat;
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED","1",1);need(project_cpu({w},x.data(),width,tokens,4,on,nullptr,true)&&on.size()==1&&on[0].size()==expected.size()&&!std::memcmp(on[0].data(),expected.data(),expected.size()*4),"actual production gate invokes helper");++checks;
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED","0",1);need(project_cpu({w},x.data(),width,tokens,4,off,nullptr,true)&&project_cpu({w},x.data(),width,tokens,4,repeat,nullptr,true)&&off.size()==1&&repeat.size()==1&&off[0].size()==expected.size()&&!std::memcmp(off[0].data(),repeat[0].data(),expected.size()*4),"legacy gate-off repeat bytes");need(std::memcmp(off[0].data(),expected.data(),expected.size()*4)!=0,"nontrivial route witness differs from legacy");++checks;
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED","1",1);
}
static void planner_controls(size_t & checks){
 for(int past=0;past<=512;++past)for(int count=1;count<=512;++count)need(lighton4b_source_prefill_budget(count,past)==(past+count<=512),"cumulative source checkpoint bound");++checks;
 need(!lighton4b_source_prefill_budget(1,-1)&&!lighton4b_source_prefill_budget(500,100),"invalid cumulative prefix rejected");++checks;
 for(int total=1;total<=512;++total)for(int period:{1,2,5,7,8,15,16,17}){
  std::vector<int> mods(total);for(int j=0;j<total;++j)mods[j]=(j/period)%2;
  std::vector<lighton4b_prefill_chunk> plan;need(lighton4b_source_prefill_plan(mods,plan),"arbitrary modality plan");
  std::vector<float> positions(size_t(3)*total),reassembled(size_t(3)*total);
  for(int a=0;a<3;++a)for(int j=0;j<total;++j)positions[size_t(a)*total+j]=float(a*1000+j*(a+1));positions[0]=-0.0f;
  int offset=0;
  for(const auto & chunk:plan){
   need(chunk.offset==offset&&chunk.tokens>=1&&chunk.tokens<=16&&chunk.offset>=chunk.logical_offset&&chunk.offset+chunk.tokens<=chunk.logical_offset+chunk.logical_tokens,"bounded contiguous logical chunk");
   need(chunk.logical_tokens<8?chunk.tokens==chunk.logical_tokens:chunk.tokens>=8,"preserved source batch threshold");
   for(int j=chunk.offset;j<chunk.offset+chunk.tokens;++j)need(mods[j]==mods[chunk.offset],"no mixed modality chunk");
   std::vector<float> sliced;need(lighton4b_prefill_chunk_positions(positions.data(),total,chunk,sliced),"actual position slice");
   for(int a=0;a<3;++a)std::copy_n(sliced.data()+a*chunk.tokens,chunk.tokens,reassembled.data()+size_t(a)*total+offset);
   offset+=chunk.tokens;
  }
  need(offset==total&&!std::memcmp(positions.data(),reassembled.data(),positions.size()*4),"all original axes and token rows preserved");++checks;
 }
 for(int image:{20,72,140,256}){
  std::vector<int> mods(size_t(image)+14);std::fill_n(mods.begin()+4,image,1);std::vector<lighton4b_prefill_chunk> plan;need(lighton4b_source_prefill_plan(mods,plan),"source structural prompt plan");
  std::vector<int> logical;int prior=-1;for(const auto & x:plan)if(x.logical_offset!=prior){logical.push_back(x.logical_tokens);prior=x.logical_offset;}
  need(logical==std::vector<int>({4,image,6,4}),"source media and recurrent checkpoint boundaries");++checks;
 }
 for(const auto & mods:std::vector<std::vector<int>>{{},{0,2,0},std::vector<int>(513,0)}){
  std::vector<lighton4b_prefill_chunk> stale{{0,1,0,1}};need(!lighton4b_source_prefill_plan(mods,stale)&&stale.empty(),"invalid plan clears stale output");++checks;
 }
 std::vector<float> positions(24,1),out{123};positions[0]=std::numeric_limits<float>::quiet_NaN();
 need(!lighton4b_prefill_chunk_positions(positions.data(),8,{0,8,0,8},out)&&out.empty(),"nonfinite positions clear");++checks;
 need(!lighton4b_prefill_chunk_positions(positions.data(),8,{7,2,0,8},out)&&out.empty(),"out-of-range position slice");++checks;
}
int main(){try{
 for(const char * key:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_QWEN35_QUANT","CRISPEMBED_LIGHTON4B_QUANT","CRISPEMBED_LIGHTON4B_Q6_K","CRISPEMBED_LIGHTON4B_LINEAR_BLOCK","CRISPEMBED_QWEN35_GDN_GROUPED_HEADS","CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL","CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS"})setenv(key,"1",1);
 for(const char * key:{"CRISPEMBED_LIGHTON4B_BF16_LINEAR","CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS","CRISPEMBED_LIGHTON4B_HEAD_BF16","CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK","CRISPEMBED_QWEN35_PRECISE_RMS"})setenv(key,"0",1);
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION","all",1);
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS","0",1);
 setenv("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","1",1);
 const char * scheduler="CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL";unsetenv(scheduler);
 const char * flag="CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED";size_t checks=0;planner_controls(checks);
 unsetenv(flag);need(!lighton4b_physical_q6_tiled_mode(true),"default source operators off");++checks;
 for(const char * value:{"0","false","off","no","FaLsE","OFF"}){setenv(flag,value,1);need(!lighton4b_physical_q6_tiled_mode(true),"false alias");++checks;}
 for(const char * value:{"1","true","on","yes"}){setenv(flag,value,1);need(lighton4b_physical_q6_tiled_mode(true)&&!lighton4b_physical_q6_tiled_mode(false),"physical owner scope");++checks;}
 std::vector<float> stale{123};float zero_input[256*8]={};unsigned char zero_weight[210]={};
 need(!lighton4b_physical_q6_tiled_cpu(zero_weight,210,zero_input,256,1,7,4,stale)&&stale.empty(),"ineligible batch clears output");++checks;
 need(lighton4b_physical_q6_tiled_valid(true),"actual supported CPU build required");++checks;
 need(lighton4b_physical_q6_tiled_cpu(zero_weight,210,zero_input,256,1,8,4,stale)&&stale.size()==8&&std::all_of(stale.begin(),stale.end(),[](float x){return x==0;}),"eligible zero operator");++checks;
 zero_input[0]=std::numeric_limits<float>::quiet_NaN();need(!lighton4b_physical_q6_tiled_cpu(zero_weight,210,zero_input,256,1,8,4,stale)&&stale.empty(),"nonfinite rejects and clears");++checks;
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION","head",1);need(!lighton4b_physical_q6_tiled_valid(true),"head-only selector cannot enable layer route");++checks;
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION","all",1);
 analytical(checks);actual_route(checks);
 auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({65536,nullptr,true}),ggml_free);need(bool(ctx),"descriptor context");
 auto buffer=[](size_t bytes){return std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),bytes),ggml_backend_buffer_free);};
 auto zeros=buffer(20ull*1024*1024),ones=buffer(4096*sizeof(float)),negative=buffer(128*sizeof(float));need(zeros&&ones&&negative,"aligned synthetic storage");std::memset(ggml_backend_buffer_get_base(zeros.get()),0,20ull*1024*1024);std::fill_n(static_cast<float*>(ggml_backend_buffer_get_base(ones.get())),4096,1.0f);std::fill_n(static_cast<float*>(ggml_backend_buffer_get_base(negative.get())),128,-1.0f);
 auto tensor=[&](ggml_type type,int64_t a,int64_t b,int64_t c,ggml_backend_buffer_t storage){auto*t=ggml_new_tensor_3d(ctx.get(),type,a,b,c);t->buffer=storage;t->data=ggml_backend_buffer_get_base(storage);need(ggml_nbytes(t)<=ggml_backend_buffer_get_size(storage),"synthetic tensor bound");return t;};
 decoder_layer_weights w{};w.upstream_physical=true;w.linear.upstream_physical=true;
 w.input_norm=tensor(GGML_TYPE_F32,2560,1,1,ones.get());w.post_norm=tensor(GGML_TYPE_F32,2560,1,1,ones.get());
 w.linear.qkv=tensor(GGML_TYPE_Q6_K,2560,8192,1,zeros.get());w.linear.z=tensor(GGML_TYPE_Q4_K,2560,4096,1,zeros.get());w.linear.a=tensor(GGML_TYPE_Q4_K,2560,32,1,zeros.get());w.linear.b=tensor(GGML_TYPE_Q4_K,2560,32,1,zeros.get());w.linear.out=tensor(GGML_TYPE_Q4_K,4096,2560,1,zeros.get());
 w.linear.conv=tensor(GGML_TYPE_F32,4,1,8192,zeros.get());w.linear.a_log=tensor(GGML_TYPE_F32,32,1,1,negative.get());w.linear.dt_bias=tensor(GGML_TYPE_F32,32,1,1,zeros.get());w.linear.norm=tensor(GGML_TYPE_F32,128,1,1,ones.get());
 w.gate=tensor(GGML_TYPE_Q4_K,2560,9216,1,zeros.get());w.up=tensor(GGML_TYPE_Q4_K,2560,9216,1,zeros.get());w.down=tensor(GGML_TYPE_Q4_K,9216,2560,1,zeros.get());
 std::vector<float> input(2560,1.0f/32);lighton4b_linear_block_outputs off,on,next,rejected;
 auto good=[&](const lighton4b_linear_block_outputs&o,bool mode,int tokens){need(o.hidden.size()==input.size()&&!std::memcmp(o.hidden.data(),input.data(),input.size()*4),"analytic zero-matrix residual preserves input");need(o.cache.tokens==tokens&&o.cache.physical_q6_tiled==mode&&o.cache.conv.size()==32768&&o.cache.recurrent.size()==524288,"actual cache mode and unchanged layouts");need(std::all_of(o.cache.recurrent.begin(),o.cache.recurrent.end(),[](float x){return x==0;})&&std::all_of(o.cache.conv.begin(),o.cache.conv.end(),[](float x){return x==0;}),"zero matrix actual cache");++checks;};
 setenv(flag,"0",1);setenv("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","0",1);need(!lighton4b_linear_block_cpu(w,input.data(),1,4,nullptr,rejected)&&rejected.hidden.empty(),"Q6 fixture rejects missing native context");++checks;setenv("CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","1",1);need(lighton4b_linear_block_cpu(w,input.data(),1,4,nullptr,off),"old synthetic full block with required Q6 context gate");good(off,false,1);
 setenv(flag,"1",1);rejected.hidden={123};need(!lighton4b_linear_block_cpu(w,input.data(),1,4,&off.cache,rejected)&&rejected.hidden.empty()&&rejected.cache.recurrent.empty(),"off-to-on cache rejected before use");++checks;
 need(lighton4b_linear_block_cpu(w,input.data(),1,4,nullptr,on),"new synthetic full block");good(on,true,1);need(lighton4b_linear_block_cpu(w,input.data(),1,4,&on.cache,next),"same-mode source cache continues");good(next,true,2);
 setenv(flag,"off",1);rejected.hidden={123};need(!lighton4b_linear_block_cpu(w,input.data(),1,4,&on.cache,rejected)&&rejected.hidden.empty(),"on-to-off cache rejected");++checks;
 need(lighton4b_linear_block_cpu(w,input.data(),1,4,&off.cache,next),"same-mode old cache continues");good(next,false,2);
 need(off.cache.tokens==1&&!off.cache.physical_q6_tiled&&on.cache.tokens==1&&on.cache.physical_q6_tiled,"previous caches remain immutable");++checks;
 setenv(flag,"1",1);
 for(const char * value:{"0","false","off","no","FaLsE","OFF"}){setenv(scheduler,value,1);need(!lighton4b_source_prefill_chunks_mode(true),"planner false aliases");++checks;}
 setenv(scheduler,"1",1);need(lighton4b_source_prefill_chunks_mode(true)&&!lighton4b_source_prefill_chunks_mode(false)&&lighton4b_source_prefill_chunks_valid(true),"physical supported planner scope");++checks;
 setenv(flag,"0",1);need(!lighton4b_source_prefill_chunks_valid(true),"planner requires source Q6 route");++checks;setenv(flag,"1",1);
 lighton4b_linear_block_outputs schedule_off,schedule_on,schedule_next;
 setenv(scheduler,"0",1);need(lighton4b_linear_block_cpu(w,input.data(),1,4,nullptr,schedule_off)&&!schedule_off.cache.physical_source_prefill_chunks,"legacy cache tag");++checks;
 setenv(scheduler,"1",1);need(!lighton4b_linear_block_cpu(w,input.data(),1,4,&schedule_off.cache,rejected)&&rejected.hidden.empty(),"legacy-to-planned cache rejected");++checks;
 need(lighton4b_linear_block_cpu(w,input.data(),1,4,nullptr,schedule_on)&&schedule_on.cache.physical_source_prefill_chunks,"new actual cache tag");++checks;
 need(lighton4b_linear_block_cpu(w,input.data(),1,4,&schedule_on.cache,schedule_next)&&schedule_next.cache.tokens==2&&schedule_next.cache.physical_source_prefill_chunks,"planned cache continues");++checks;
 setenv(scheduler,"0",1);need(!lighton4b_linear_block_cpu(w,input.data(),1,4,&schedule_on.cache,rejected)&&rejected.hidden.empty(),"planned-to-legacy cache rejected");++checks;
 need(schedule_off.cache.tokens==1&&!schedule_off.cache.physical_source_prefill_chunks&&schedule_on.cache.tokens==1&&schedule_on.cache.physical_source_prefill_chunks,"previous scheduling caches immutable");++checks;
 setenv("CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK","1",1);setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION","1",1);
 decoder_layer_weights aw{};aw.full_attention=true;aw.upstream_physical=true;aw.input_norm=w.input_norm;aw.post_norm=w.post_norm;aw.gate=w.gate;aw.up=w.up;aw.down=w.down;aw.attention.upstream_physical=true;
 aw.attention.q=tensor(GGML_TYPE_Q6_K,2560,8192,1,zeros.get());aw.attention.k=tensor(GGML_TYPE_Q6_K,2560,1024,1,zeros.get());aw.attention.v=tensor(GGML_TYPE_Q6_K,2560,1024,1,zeros.get());aw.attention.out=tensor(GGML_TYPE_Q6_K,4096,2560,1,zeros.get());aw.attention.q_norm=tensor(GGML_TYPE_F32,256,1,1,ones.get());aw.attention.k_norm=tensor(GGML_TYPE_F32,256,1,1,ones.get());
 lighton4b_attention_block_outputs attention_off,attention_on,attention_next,attention_rejected;float position0[3]={0,0,0},position1[3]={1,1,1};
 setenv(scheduler,"0",1);need(lighton4b_attention_block_cpu(aw,input.data(),position0,1,4,nullptr,attention_off)&&!attention_off.cache.physical_source_prefill_chunks,"actual old attention cache");++checks;
 setenv(scheduler,"1",1);need(!lighton4b_attention_block_cpu(aw,input.data(),position1,1,4,&attention_off.cache,attention_rejected)&&attention_rejected.hidden.empty(),"attention old-to-planned rejected");++checks;
 need(lighton4b_attention_block_cpu(aw,input.data(),position0,1,4,nullptr,attention_on)&&attention_on.cache.physical_source_prefill_chunks,"actual planned attention cache");++checks;
 need(lighton4b_attention_block_cpu(aw,input.data(),position1,1,4,&attention_on.cache,attention_next)&&attention_next.cache.tokens==2&&attention_next.cache.physical_source_prefill_chunks&&attention_next.hidden.size()==input.size()&&!std::memcmp(attention_next.hidden.data(),input.data(),input.size()*4),"planned attention continuation preserves analytic input");++checks;
 setenv(scheduler,"0",1);need(!lighton4b_attention_block_cpu(aw,input.data(),position1,1,4,&attention_on.cache,attention_rejected)&&attention_rejected.hidden.empty(),"attention planned-to-old rejected");++checks;
 need(attention_off.cache.tokens==1&&!attention_off.cache.physical_source_prefill_chunks&&attention_on.cache.tokens==1&&attention_on.cache.physical_source_prefill_chunks,"previous attention caches immutable");++checks;
 std::printf("SOURCE_PREFILL_CHUNKS_CONTROLS PASS checks=%zu model_payload=0 decoded=0\n",checks);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
