// TEST ONLY. Reproduce native AVX2 variance order; never a production export.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
namespace {
constexpr int H=384,T=12,N=H*T,M=1536*T,W=331776;
bool finite(const float * p,int n) { if(!p)return false;for(int i=0;i<n;++i)if(!std::isfinite(p[i]))return false;return true; }
float add(float a,float b){volatile float v=a+b;return v;}
float sub(float a,float b){volatile float v=a-b;return v;}
float mul(float a,float b){volatile float v=a*b;return v;}
float div(float a,float b){volatile float v=a/b;return v;}
// Exact a06 vec.h sum, vec.cpp AVX2+FMA cvar tree, ops.cpp norm scaling.
std::vector<float> native_order(const float * x) {
    std::vector<float> y(N);
    for(int row=0;row<T;++row){
        const float * a=x+row*H;float * b=y.data()+row*H;
        double sum=0;for(int i=0;i<H;++i)sum+=double(a[i]);
        const float mean=div(float(sum),float(H));double square_sum=0;
        for(int i=0;i<H;i+=8){
            float v[8];for(int j=0;j<8;++j){b[i+j]=sub(a[i+j],mean);v[j]=mul(b[i+j],b[i+j]);}
            // AVX high128+low128, movehl, then movehdup/add_ss.
            const float a0=add(v[4],v[0]),a1=add(v[5],v[1]),a2=add(v[6],v[2]),a3=add(v[7],v[3]);
            square_sum+=double(add(add(a0,a2),add(a1,a3)));
        }
        const float variance=float(square_sum/double(H));
        const float scale=div(1.f,std::sqrt(add(variance,1e-12f)));
        for(int i=0;i<H;++i)b[i]=mul(b[i],scale);
    }
    return y;
}
struct owner {
    ggml_context * ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ~owner(){if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
}
extern "C" int crispembed_norm_setenv(const char * directory){return directory&&*directory?setenv("CRISPEMBED_LAYER3_KERNEL",directory,1):unsetenv("CRISPEMBED_LAYER3_KERNEL");}
// No quantized/workspace/reference-output argument exists. mode0 runs original
// GGML norm. mode1 computes the native-order norm from x. kind1 carries that
// actual result through original affine parameters and IQ4_NL/Q8_0 matmul.
extern "C" int crispembed_norm_order(int mode,int kind,const float * x,int count,const float * gamma,const float * beta,
                                      const uint8_t * weight,int weight_bytes,float * output,int capacity,int threads){
    const int required=kind==1?3*N+M:N;
    if((mode!=0&&mode!=1)||(kind!=0&&kind!=1)||count!=N||!finite(x,N)||!output||capacity!=required||threads<1||threads>2)return -1;
    if(kind==0&&(gamma||beta||weight||weight_bytes))return -2;
    if(kind==1&&(!finite(gamma,H)||!finite(beta,H)||!weight||weight_bytes!=W))return -2;
    if(kind)for(int i=0;i<W;i+=18){uint16_t d;std::memcpy(&d,weight+i,2);if((d&0x7c00)==0x7c00)return -3;}
    try{
        std::vector<float> computed;if(mode)computed=native_order(x);
        if(mode&&!finite(computed.data(),N))return -4;
        if(mode&&kind==0){std::copy(computed.begin(),computed.end(),output);return N;}
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);
        if(!backend)return -5;ggml_backend_cpu_set_n_threads(backend.get(),threads);
#ifdef __EMSCRIPTEN__
        ggml_backend_cpu_set_wasm_q8_0_rounding(backend.get(),true);
#endif
        owner o;o.ctx=ggml_init({64*ggml_tensor_overhead()+ggml_graph_overhead_custom(64,false),nullptr,true});if(!o.ctx)return -6;
        auto * input=ggml_new_tensor_2d(o.ctx,GGML_TYPE_F32,H,T);ggml_set_input(input);
        auto * norm=mode?input:ggml_norm(o.ctx,input,1e-12f);ggml_set_output(norm);
        std::vector<ggml_tensor*> stages{norm};ggml_tensor * g=nullptr,*b=nullptr,*w=nullptr;
        if(kind){
            g=ggml_new_tensor_1d(o.ctx,GGML_TYPE_F32,H);b=ggml_new_tensor_1d(o.ctx,GGML_TYPE_F32,H);
            w=ggml_new_tensor_2d(o.ctx,GGML_TYPE_IQ4_NL,H,1536);ggml_set_name(w,"blk.3.ffn_up.weight");
            if(ggml_nbytes(w)!=W||ggml_get_type_traits_cpu(w->type)->vec_dot_type!=GGML_TYPE_Q8_0)return -7;
            auto * scaled=ggml_mul(o.ctx,norm,g);auto * affine=ggml_add(o.ctx,scaled,b);auto * projected=ggml_mul_mat(o.ctx,w,affine);
            stages.insert(stages.end(),{scaled,affine,projected});for(auto * v:stages)ggml_set_output(v);
        }
        auto * graph=ggml_new_graph_custom(o.ctx,64,false);for(auto * v:stages)ggml_build_forward_expand(graph,v);
        o.buffer=ggml_backend_alloc_ctx_tensors(o.ctx,backend.get());if(!o.buffer)return -8;
        ggml_backend_tensor_set(input,mode?computed.data():x,0,N*4);
        if(kind){ggml_backend_tensor_set(g,gamma,0,H*4);ggml_backend_tensor_set(b,beta,0,H*4);ggml_backend_tensor_set(w,weight,0,W);}
        if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)return -9;
        std::vector<float> pending(required);int offset=0;
        for(auto * v:stages){int n=int(ggml_nelements(v));ggml_backend_tensor_get(v,pending.data()+offset,0,n*4);offset+=n;}
        if(offset!=required||!finite(pending.data(),required))return -10;
        std::copy(pending.begin(),pending.end(),output);return required;
    }catch(...){return -11;}
}
