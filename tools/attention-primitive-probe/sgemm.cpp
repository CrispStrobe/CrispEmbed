// Diagnostic-only F16/F16 SGEMM nodes. Private ABI is compiled from exact backend
// headers; no global traits, vendor source or production dispatch are changed.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "llamafile/sgemm.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;
static void need(bool x,const char * why){if(!x)throw std::runtime_error(why);}
static std::vector<float> read(const fs::path & p,size_t size){std::vector<float>x(size);std::ifstream f(p,std::ios::binary);need(bool(f.read(reinterpret_cast<char *>(x.data()),size*4))&&f.peek()==EOF,"input extent");for(float v:x)need(std::isfinite(v),"finite input");return x;}
struct operator_state { ggml_threadpool * pool; bool pv; std::atomic_bool accepted{true}; };
static void sgemm(ggml_tensor * dst,const ggml_tensor *,const ggml_tensor * cache,const ggml_tensor * rhs,int ith,int nth,void * user){
    auto & state=*static_cast<operator_state *>(user);
    // Keep an unsupported private dispatch finite until the outer fail-closed
    // check. Accepted SGEMM synchronizes all workers before writing.
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth)static_cast<float *>(dst->data)[i]=0.f;
    if(nth!=4 || rhs->type!=GGML_TYPE_F16 || cache->type!=GGML_TYPE_F16 || dst->type!=GGML_TYPE_F32){state.accepted=false;return;}
    ggml_compute_params params{};params.ith=ith;params.nth=nth;params.threadpool=state.pool;
    const int n=int(rhs->ne[1]);
    for(int h=0;h<16;++h){
        // Exact source get_k/get_v and converted rhs row strides, elements.
        const auto * a=static_cast<const ggml_fp16_t *>(cache->data)+(state.pv?(h/4)*256*2048:(h/4)*256);
        const auto * b=static_cast<const ggml_fp16_t *>(rhs->data)+h*n*256;
        auto * c=static_cast<float *>(dst->data)+h*n*256;
        const bool ok=llamafile_sgemm(&params,256,n,256,a,state.pv?2048:1024,b,256,c,256,GGML_TYPE_F16,GGML_TYPE_F16,GGML_TYPE_F32);
        if(!ok){state.accepted=false;return;} // Never substitute another kernel.
    }
}
static void dump(const fs::path & p,ggml_tensor * t){need(t->type==GGML_TYPE_F32&&ggml_is_contiguous(t),"output layout");auto * x=static_cast<float *>(t->data);bool nonzero=false;for(int64_t i=0;i<ggml_nelements(t);++i){need(std::isfinite(x[i]),"finite output");nonzero|=x[i]!=0;}need(nonzero,"materialized output");std::ofstream f(p,std::ios::binary);need(bool(f.write(reinterpret_cast<char *>(t->data),ggml_nbytes(t))),"output write");}
int main(int argc,char ** argv)try{
    need(argc==5||argc==6,"usage: sgemm INPUT OUTPUT {4,16} {composed,observed} [SOURCE_P]");
    const int n=std::stoi(argv[3]);need(n==4||n==16,"tokens");const std::string mode=argv[4];need(mode=="observed"||mode=="composed","mode");
    const fs::path root=argv[1],out=argv[2];fs::create_directories(out);
    const auto q=read(root/"q.f32",16*4096),k=read(root/"k.f32",16*1024),v=read(root/"v.f32",16*1024),g=read(root/"gate.f32",16*4096);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({64ull<<20,nullptr,false}),ggml_free);need(bool(ctx),"context");
    auto pp=ggml_threadpool_params_default(4);
    std::unique_ptr<ggml_threadpool,decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pp),ggml_threadpool_free);need(bool(pool),"threadpool");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);need(bool(backend),"backend");
    ggml_backend_cpu_set_n_threads(backend.get(),4);ggml_backend_cpu_set_threadpool(backend.get(),pool.get());
    auto * c=ctx.get();auto * key=ggml_new_tensor_2d(c,GGML_TYPE_F16,1024,2048);auto * value=ggml_new_tensor_2d(c,GGML_TYPE_F16,2048,1024);
    std::fill_n(static_cast<ggml_fp16_t *>(key->data),1024*2048,ggml_fp32_to_fp16(0));std::fill_n(static_cast<ggml_fp16_t *>(value->data),1024*2048,ggml_fp32_to_fp16(0));
    for(int t=0;t<n;++t)for(int h=0;h<4;++h)for(int i=0;i<256;++i){
        const size_t ix=(t*4+h)*256+i;auto kh=ggml_fp32_to_fp16(k[ix]),vh=ggml_fp32_to_fp16(v[ix]);need(std::isfinite(ggml_fp16_to_fp32(kh))&&std::isfinite(ggml_fp16_to_fp32(vh)),"KV overflow");
        static_cast<ggml_fp16_t *>(key->data)[t*1024+h*256+i]=kh;static_cast<ggml_fp16_t *>(value->data)[(h*256+i)*2048+t]=vh;
    }
    auto * query=ggml_new_tensor_3d(c,GGML_TYPE_F16,256,n,16);
    for(int h=0;h<16;++h)for(int t=0;t<n;++t)for(int i=0;i<256;++i){auto half=ggml_fp32_to_fp16(q[(t*16+h)*256+i]);need(std::isfinite(ggml_fp16_to_fp32(half)),"Q overflow");static_cast<ggml_fp16_t *>(query->data)[(h*n+t)*256+i]=half;}
    operator_state qk{pool.get(),false},pv{pool.get(),true};
    auto * placeholder=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,n,16);
    auto * logits=ggml_map_custom3(c,placeholder,key,query,sgemm,4,&qk);
    auto * mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,32);for(int t=0;t<32;++t)for(int s=0;s<256;++s)static_cast<float *>(mask->data)[t*256+s]=(t<n&&s<=t)?0.f:-INFINITY;
    auto * probs=ggml_soft_max_ext(c,logits,mask,.0625f,0.f);
    ggml_tensor * pin=probs;
    if(argc==6){const auto source=read(argv[5],16*n*256);pin=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,n,16);std::copy(source.begin(),source.end(),static_cast<float *>(pin->data));}
    auto * ph=ggml_cast(c,pin,GGML_TYPE_F16);
    auto * raw=ggml_map_custom3(c,placeholder,value,ph,sgemm,4,&pv);
    auto * context=ggml_cont_2d(c,ggml_permute(c,raw,0,2,1,3),4096,n);
    auto * gate=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,n);std::copy_n(g.data(),n*4096,static_cast<float *>(gate->data));
    auto * sig=ggml_sigmoid(c,gate);auto * gated=ggml_mul(c,context,sig);
    auto * graph=ggml_new_graph_custom(c,128,false);
    const std::vector<std::pair<const char *,ggml_tensor *>> stages={{"logits",logits},{"probs",probs},{"context",context},{"sigmoid",sig},{"gated",gated}};
    if(mode=="observed")for(const auto & s:stages){ggml_set_output(s.second);ggml_build_forward_expand(graph,s.second);}
    ggml_build_forward_expand(graph,gated);need(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"graph");
    need(qk.accepted&&pv.accepted,"private SGEMM rejected; no fallback");
    if(mode=="observed")for(const auto & s:stages)dump(out/(std::string(s.first)+".f32"),s.second);else dump(out/"gated.f32",gated);
    std::cout<<"PHYSICAL_F16_SGEMM_PRIVATE_OPERATOR PASS tokens="<<n<<" threads=4 observed="<<(mode=="observed")<<'\n';return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
