// Diagnostic-only explicit original paired-F16 dot; never changes global traits.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
// Private backend ABI experiment, intentionally isolated from production. Link
// failure must be preserved; do not substitute a different accumulation kernel.
extern "C" void ggml_vec_dot_f16(int, float *, size_t, ggml_fp16_t *, size_t, ggml_fp16_t *, size_t, int);
namespace fs = std::filesystem;
static void need(bool ok, const char * s) { if (!ok) throw std::runtime_error(s); }
static std::vector<float> read(const fs::path & p, size_t n) {
    std::vector<float> v(n); std::ifstream f(p, std::ios::binary);
    need(bool(f.read(reinterpret_cast<char *>(v.data()), n * 4)) && f.peek() == EOF, "input extent");
    for (float x : v) need(std::isfinite(x), "nonfinite input"); return v;
}
static std::vector<ggml_fp16_t> half(const std::vector<float> & x) {
    std::vector<ggml_fp16_t> y(x.size()); ggml_fp32_to_fp16_row(x.data(), y.data(), int64_t(x.size()));
    for (auto v : y) need(std::isfinite(ggml_fp16_to_fp32(v)), "F16 overflow"); return y;
}
static void dump(const fs::path & p, const std::vector<float> & x) {
    for (float v : x) need(std::isfinite(v), "nonfinite stage");
    std::ofstream f(p, std::ios::binary); need(bool(f.write(reinterpret_cast<const char *>(x.data()), x.size() * 4)), "output write");
}
int main(int argc, char ** argv) try {
    if (argc == 2 && std::string(argv[1]) == "--selftest") {
        std::unique_ptr<ggml_context,decltype(&ggml_free)> init(ggml_init({1ull<<20,nullptr,false}),ggml_free);
        need(bool(init),"selftest init");
        std::vector<ggml_fp16_t> a(256,ggml_fp32_to_fp16(1.f)), b(256,ggml_fp32_to_fp16(.5f));
        std::fill_n(b.data(),128,ggml_fp32_to_fp16(1.f));float value=0;
        ggml_vec_dot_f16(256,&value,0,a.data(),0,b.data(),0,1);need(value==192.f,"paired dot symbol/control");
        bool overflow_rejected=false;try { half({65536.f}); } catch(const std::runtime_error &) { overflow_rejected=true; }
        need(overflow_rejected,"F16 overflow must reject");
        std::cout<<"PAIRED_F16_SYMBOL_OVERFLOW_CONTROLS PASS 2\n";return 0;
    }
    need(argc == 4, "usage: paired INPUT OUTPUT {4,16}"); const fs::path root=argv[1], out=argv[2];
    const int n=std::stoi(argv[3]); need(n==4 || n==16, "tokens"); fs::create_directories(out);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8ull<<20,nullptr,false}),ggml_free);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
    need(bool(ctx)&&bool(backend),"backend"); ggml_backend_cpu_set_n_threads(backend.get(),4);
    auto q=half(read(root/"q.f32",16*4096)), k=half(read(root/"k.f32",16*1024));
    auto v=half(read(root/"v.f32",16*1024)); const auto gate=read(root/"gate.f32",16*4096);
    std::vector<float> logits(16*n*256,0.f), probs(logits.size()), context(n*4096), gated(context.size()), sigmoid(context.size());
    for (int h=0;h<16;++h) for (int t=0;t<n;++t) for (int s=0;s<n;++s)
        ggml_vec_dot_f16(256,&logits[(h*n+t)*256+s],0,k.data()+(s*4+h/4)*256,0,q.data()+(t*16+h)*256,0,1);
    auto * c=ctx.get(); auto * score=ggml_new_tensor_3d(c,GGML_TYPE_F32,256,n,16);
    std::copy(logits.begin(),logits.end(),static_cast<float *>(score->data));
    auto * mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,256,32);
    for(int t=0;t<32;++t)for(int s=0;s<256;++s)static_cast<float *>(mask->data)[t*256+s]=(t<n&&s<=t)?0.f:-INFINITY;
    auto * probability=ggml_soft_max_ext(c,score,mask,.0625f,0.f);
    auto * rawgate=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,n);std::copy_n(gate.data(),n*4096,static_cast<float *>(rawgate->data));
    auto * sig=ggml_sigmoid(c,rawgate);
    auto * graph=ggml_new_graph_custom(c,32,false);ggml_set_output(probability);ggml_set_output(sig);
    ggml_build_forward_expand(graph,probability);ggml_build_forward_expand(graph,sig);
    need(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"softmax/sigmoid graph");
    std::copy_n(static_cast<float *>(probability->data),probs.size(),probs.data());
    std::copy_n(static_cast<float *>(sig->data),sigmoid.size(),sigmoid.data());
    auto ph=half(probs);
    auto rounded = [&](const char * name, const std::vector<ggml_fp16_t> & values) {
        std::vector<float> f(values.size());ggml_fp16_to_fp32_row(values.data(),f.data(),int64_t(values.size()));dump(out/name,f);
    };
    rounded("q.rounded.f32",q);rounded("k.rounded.f32",k);rounded("v.rounded.f32",v);rounded("probs.rounded.f32",ph);
std::vector<ggml_fp16_t> lane(256,ggml_fp32_to_fp16(0.f));
    for(int h=0;h<16;++h)for(int i=0;i<256;++i){
        std::fill(lane.begin(),lane.end(),ggml_fp32_to_fp16(0.f));
        for(int s=0;s<n;++s)lane[s]=v[(s*4+h/4)*256+i];
        for(int t=0;t<n;++t){
            const size_t ix=(t*16+h)*256+i;
            ggml_vec_dot_f16(256,&context[ix],0,lane.data(),0,ph.data()+(h*n+t)*256,0,1);
            gated[ix]=context[ix]*sigmoid[ix];
        }
    }
    dump(out/"logits.f32",logits);dump(out/"probs.f32",probs);dump(out/"context.f32",context);dump(out/"sigmoid.f32",sigmoid);dump(out/"gated.f32",gated);
    std::cout<<"PAIRED_F16_OPERATOR_ONLY PASS tokens="<<n<<'\n';return 0;
}catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
