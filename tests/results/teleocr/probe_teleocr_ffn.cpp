#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

static int run(int argc,char**argv) {
    if(argc!=3)return 2;
    ggml_context *weights=nullptr;
    gguf_context *file=gguf_init_from_file(argv[1],{false,&weights});
    if(!file||!weights)return 3;
    auto t=[&](const char*n){return ggml_get_tensor(weights,n);};
    ggml_context *g=ggml_init({256*1024*1024,nullptr,false});
    auto backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(backend,2);
    std::vector<std::pair<const char*,ggml_tensor*>> outputs;
    auto keep=[&](const char*n,ggml_tensor*x){outputs.push_back({n,x});return x;};
    auto norm=keep("norm",ggml_mul(g,ggml_rms_norm(g,t("input"),1e-6f),t("norm_w")));
    auto mul=[&](ggml_tensor*w,ggml_tensor*x){auto r=ggml_mul_mat(g,w,x);ggml_mul_mat_set_prec(r,GGML_PREC_F32);return r;};
    auto gate=keep("gate",ggml_add(g,mul(t("gate_w"),norm),t("gate_b")));
    auto up=keep("up",ggml_add(g,mul(t("up_w"),norm),t("up_b")));
    auto silu=keep("silu",ggml_silu(g,gate));
    auto prod=keep("product",ggml_mul(g,silu,up));
    auto down=keep("down",ggml_add(g,mul(t("down_w"),prod),t("down_b")));
    auto output=keep("output",ggml_add(g,t("input"),down));
    auto graph=ggml_new_graph(g);ggml_build_forward_expand(graph,output);
    if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return 4;
    for(auto &p:outputs){
        auto ref=t(p.first);auto actual=(const float*)p.second->data;auto expected=(const float*)ref->data;auto n=ggml_nelements(ref);
        double max=0;int64_t at=0;for(int64_t i=0;i<n;++i){double d=std::fabs(double(actual[i])-expected[i]);if(d>max){max=d;at=i;}}
        std::printf("%s max_abs=%.9g index=%lld row=%lld column=%lld native=%.9g ref=%.9g\n",p.first,max,(long long)at,(long long)(at/ref->ne[0]),(long long)(at%ref->ne[0]),actual[at],expected[at]);
        std::string path=std::string(argv[2])+"/native_"+p.first+".f32";auto f=std::fopen(path.c_str(),"wb");if(!f)return 5;auto wrote=std::fwrite(actual,sizeof(float),n,f);std::fclose(f);if(wrote!=(size_t)n)return 6;
    }
    ggml_backend_free(backend);ggml_free(g);ggml_free(weights);gguf_free(file);return 0;
}
int main(int argc,char**argv){core_util::clean_exit(run(argc,argv));}
