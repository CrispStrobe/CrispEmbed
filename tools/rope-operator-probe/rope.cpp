// Test-only IMROPE operator on retained normalized Q/K. No weights or model.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;
static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
static std::vector<char> read(const fs::path&p,size_t n){std::ifstream f(p,std::ios::binary);std::vector<char>b(n);need(bool(f.read(b.data(),n))&&f.peek()==EOF,"exact input");return b;}
int main(int argc,char**argv)try{
 need(argc==5,"INPUT OUTPUT TOKENS HEADS");const int n=std::stoi(argv[3]),heads=std::stoi(argv[4]);need((n==4||n==6||n==252)&&(heads==4||heads==16),"frozen geometry");
 const fs::path root=argv[1],out=argv[2];fs::create_directories(out);
 const auto values=read(root/"normalized.f32",size_t(n)*heads*256*4),positions=read(root/"positions.i32",size_t(n)*4*4);
 std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({32ull<<20,nullptr,false}),ggml_free);need(bool(ctx),"operator context");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_cpu_init(),ggml_backend_free);need(bool(backend),"CPU backend");ggml_backend_cpu_set_n_threads(backend.get(),4);
 auto*x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,256,heads,n);auto*p=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,4*n);
 std::memcpy(x->data,values.data(),values.size());std::memcpy(p->data,positions.data(),positions.size());
 for(size_t i=0;i<values.size()/4;++i)need(std::isfinite(static_cast<float*>(x->data)[i]),"finite normalized input");
 int sections[4]={11,11,10,0};
 // No extrapolation/scaling on this frozen short-context capture. The unused
 // YARN fields are a fixed preset, not claimed independently observed metadata.
 auto*y=ggml_rope_multi(ctx.get(),x,p,nullptr,64,sections,GGML_ROPE_TYPE_IMROPE,262144,10000000.f,1.f,0.f,1.f,32.f,1.f);
 auto*graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,y);need(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"RoPE graph");
 need(ggml_is_contiguous(y)&&y->type==GGML_TYPE_F32,"output layout");
 for(int64_t i=0;i<ggml_nelements(y);++i)need(std::isfinite(static_cast<float*>(y->data)[i]),"finite RoPE output");
 std::ofstream f(out/"rotated.f32",std::ios::binary);need(bool(f.write(static_cast<char*>(y->data),ggml_nbytes(y))),"complete output");
 std::cout<<"ROPE_OPERATOR PASS tokens="<<n<<" heads="<<heads<<" model=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
