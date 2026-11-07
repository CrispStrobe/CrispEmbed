#include "lighton4b_physical_f16_attention.h"
#include "qwen35_attention.h"
#include "core/clean_exit.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
using namespace qwen35_ocr;
namespace fs=std::filesystem;
static void need(bool x,const char * s){if(!x)throw std::runtime_error(s);}
static std::vector<float> read(const fs::path & p,size_t n){std::vector<float>x(n);std::ifstream f(p,std::ios::binary);need(bool(f.read(reinterpret_cast<char *>(x.data()),n*4))&&f.peek()==EOF,"read extent");return x;}
static void write(const fs::path & p,const std::vector<float> & x){fs::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);need(bool(f.write(reinterpret_cast<const char *>(x.data()),x.size()*4)),"write");}
static void gate(const char * name,const char * value){
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
static int run(int argc,char ** argv){
    need(argc==3,"usage: probe AUTHENTICATED_INPUT OUTPUT");
    for(const char * n:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL","CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION"})gate(n,"1");
    for(const char * n:{"CRISPEMBED_LIGHTON4B_BF16_LINEAR","CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS","CRISPEMBED_LIGHTON4B_HEAD_BF16"})gate(n,"0");
    need(lighton4b_physical_f16_attention_available()&&lighton4b_physical_f16_attention_valid(true),"supported ABI required");
    const fs::path input=argv[1],out=argv[2];const auto q=read(input/"q.f32",16*4096),k=read(input/"k.f32",16*1024),v=read(input/"v.f32",16*1024),g=read(input/"gate.f32",16*4096);
    auto invoke=[&](int past,int n,const attention_cache * previous,const fs::path & dest){
        attention_cache cache;cache.batch=1;cache.heads=4;cache.width=256;cache.tokens=past+n;cache.upstream_physical=true;cache.physical_f16_attention=true;
        cache.keys.resize(size_t(past+n)*1024);cache.values.resize(cache.keys.size());
        for(int h=0;h<4;++h){if(past){need(previous&&previous->tokens==past,"previous geometry");std::copy_n(previous->keys.data()+size_t(h)*past*256,size_t(past)*256,cache.keys.data()+size_t(h)*(past+n)*256);std::copy_n(previous->values.data()+size_t(h)*past*256,size_t(past)*256,cache.values.data()+size_t(h)*(past+n)*256);}
            for(int t=0;t<n;++t){std::copy_n(k.data()+(size_t(past+t)*4+h)*256,256,cache.keys.data()+(size_t(h)*(past+n)+past+t)*256);std::copy_n(v.data()+(size_t(past+t)*4+h)*256,256,cache.values.data()+(size_t(h)*(past+n)+past+t)*256);}}
        std::vector<float> query(q.begin()+size_t(past)*4096,q.begin()+size_t(past+n)*4096),fused(size_t(n)*8192,0),pooled;
        for(int t=0;t<n;++t)for(int h=0;h<16;++h)std::copy_n(g.data()+(size_t(past+t)*16+h)*256,256,fused.data()+(size_t(t)*16+h)*512+256);
        attention_cache canonical;need(lighton4b_physical_f16_attention_cpu(query,fused,cache,past,n,4,pooled,canonical),"runtime helper");
        write(dest/"gated.f32",pooled);write(dest/"keys.f32",canonical.keys);write(dest/"values.f32",canonical.values);return canonical;
    };
    invoke(0,4,nullptr,out/"prefill4");auto full=invoke(0,16,nullptr,out/"prefill16");attention_cache previous;
    for(int t=0;t<16;++t){char name[16];std::snprintf(name,sizeof(name),"step%02d",t);auto next=invoke(t,1,t?&previous:nullptr,out/name);previous=std::move(next);}
    need(previous.keys==full.keys&&previous.values==full.values,"prefill/step canonical cache bytes");
    std::puts("PHYSICAL_F16_RUNTIME_SOURCEFED PASS prefill4 prefill16 steps16 canonical_cache_equal");return 0;
}
int main(int argc,char ** argv){int rc=1;try{rc=run(argc,argv);}catch(const std::exception & e){std::fprintf(stderr,"%s\n",e.what());}core_util::clean_exit(rc);}
