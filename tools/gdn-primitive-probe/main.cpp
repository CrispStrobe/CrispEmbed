// Reference-only same-input GDN operator probe, never a production dependency.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef CRISPEMBED_GDN_RUNTIME_PROBE
#include "qwen35_delta.h"
#include "qwen35_math.h"
#include "lighton4b_upstream_physical_semantics.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using V = std::vector<float>;
static void need(bool x, const char * s) { if (!x) throw std::runtime_error(s); }
static V load(const std::filesystem::path & p, size_t total, size_t keep) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    need(f && f.tellg() == std::streamoff(total * 4), "bounded exact F32 file");
    f.seekg(0); V v(keep); need(bool(f.read(reinterpret_cast<char *>(v.data()), keep * 4)), "complete F32 prefix");
    need(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }), "finite input"); return v;
}
static void dump(const std::filesystem::path & p, const V & v) {
    need(std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); }), "finite stage");
    std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char *>(v.data()), v.size() * 4);
    need(bool(f), "stage write");
}
static size_t ph(size_t h) { return 16 * (h % 2) + h / 2; }
static size_t pv(size_t i) { return ph(i / 128) * 128 + i % 128; }
static V to_tiled(const V & hf, int n, int width, int fixed = 0) {
    V out(hf.size());
    for (int t=0;t<n;++t) for (int i=0;i<width;++i) {
        const int j=i<fixed ? i : fixed + int(width-fixed==32 ? ph(i-fixed) : pv(i-fixed));
        out[size_t(t)*width+j]=hf[size_t(t)*width+i];
    }
    return out;
}
static V to_hf(const float * data, int n, int width, bool tiled, int fixed=0) {
    V out(size_t(n)*width);
    for(int t=0;t<n;++t)for(int i=0;i<width;++i){
        const int j=!tiled || i<fixed ? i : fixed+int(width-fixed==32 ? ph(i-fixed) : pv(i-fixed));
        out[size_t(t)*width+i]=data[size_t(t)*width+j];
    }
    return out;
}
static float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    const float e=std::exp(x); return e/(1.0f+e);
}
struct Inputs { V qkv,z,a,b,conv,A,dt,norm; };
static std::map<std::string,V> project_pair(const V & values,int n,const std::filesystem::path & weights) {
    auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({64ull*1024*1024,nullptr,false}),ggml_free);
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    need(bool(ctx)&&bool(backend),"projection CPU allocation");ggml_backend_cpu_set_n_threads(backend.get(),4);auto * c=ctx.get();
    auto input=[&](const V & v,int width,int tokens){auto * t=ggml_new_tensor_2d(c,GGML_TYPE_F32,width,tokens);std::memcpy(t->data,v.data(),v.size()*4);return t;};
    std::map<std::string,V> out;ggml_cgraph * graph=nullptr;
    V tiled=to_tiled(values,n,4096);auto * x=input(tiled,4096,n);auto * w=ggml_new_tensor_2d(c,GGML_TYPE_Q4_K,4096,2560);
    std::ifstream f(weights/"out.q4k",std::ios::binary|std::ios::ate);need(f&&f.tellg()==std::streamoff(ggml_nbytes(w)),"exact Q4_K out tensor");f.seekg(0);need(bool(f.read(static_cast<char *>(w->data),ggml_nbytes(w))),"out tensor read");
    auto * wf=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,2560);const auto * traits=ggml_get_type_traits(GGML_TYPE_Q4_K);traits->to_float(w->data,static_cast<float *>(wf->data),ggml_nelements(w));
    auto * direct=ggml_mul_mat(c,w,x),*dequant=ggml_mul_mat(c,wf,x);ggml_mul_mat_set_prec(direct,GGML_PREC_F32);ggml_mul_mat_set_prec(dequant,GGML_PREC_F32);
    graph=ggml_new_graph(c);ggml_build_forward_expand(graph,direct);ggml_build_forward_expand(graph,dequant);need(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"same-input out projections");
    out["out_direct"]=V(static_cast<float *>(direct->data),static_cast<float *>(direct->data)+size_t(n)*2560);
    out["out_dequant"]=V(static_cast<float *>(dequant->data),static_cast<float *>(dequant->data)+size_t(n)*2560);
    return out;
}
static std::map<std::string,V> compute(const Inputs & in, int n, bool source, bool observe,
                                      const std::filesystem::path & weights) {
    auto ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>(ggml_init({128ull*1024*1024,nullptr,false}),ggml_free);
    auto backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),ggml_backend_free);
    need(bool(ctx)&&bool(backend),"CPU graph allocation");ggml_backend_cpu_set_n_threads(backend.get(),4);
    auto c=ctx.get(); auto input=[&](const V & values,int a,int b=1,int d=1){
        auto * x=ggml_new_tensor_3d(c,GGML_TYPE_F32,a,b,d);need(size_t(ggml_nelements(x))==values.size(),"tensor input shape");
        std::memcpy(x->data,values.data(),values.size()*4);return x;
    };
    std::map<std::string,ggml_tensor *> stages;
    V qkv=source?to_tiled(in.qkv,n,8192,4096):in.qkv;
    V z=source?to_tiled(in.z,n,4096):in.z;
    V aa=source?to_tiled(in.a,n,32):in.a, bb=source?to_tiled(in.b,n,32):in.b;
    V convHF(8192*4),Ahf(32),dthf(32);
    for(int i=0;i<8192;++i)for(int tap=0;tap<4;++tap)convHF[size_t(i)*4+tap]=in.conv[size_t(i<4096?i:4096+pv(i-4096))*4+tap];
    for(int h=0;h<32;++h){Ahf[h]=in.A[ph(h)];dthf[h]=in.dt[ph(h)];}
    ggml_tensor * qr,*kr,*vr,*decay,*beta;
    V nativeConv,nativeSilu,gates,mixing;
    if(source){
        auto * mixed=input(qkv,8192,n);V zeros(3*8192,0);auto * hist=input(zeros,3,8192);
        auto * convInput=ggml_concat(c,hist,ggml_transpose(c,mixed),0);
        auto * raw=ggml_ssm_conv(c,convInput,input(in.conv,4,8192));stages["conv_raw"]=raw;
        auto * silu=ggml_silu(c,raw);stages["conv_silu"]=silu;
        const size_t row=8192*4;
        qr=ggml_view_4d(c,silu,128,16,n,1,128*4,row,row*n,0);
        kr=ggml_view_4d(c,silu,128,16,n,1,128*4,row,row*n,2048*4);
        vr=ggml_view_4d(c,silu,128,32,n,1,128*4,row,row*n,4096*4);
        qr=ggml_scale(c,ggml_rms_norm(c,qr,1e-6f/128),1.0f/std::sqrt(128.0f));
        kr=ggml_scale(c,ggml_rms_norm(c,kr,1e-6f/128),1.0f/std::sqrt(128.0f));
        auto * biased=ggml_add(c,input(aa,32,n),input(in.dt,32));stages["alpha_biased"]=biased;
        auto * soft=ggml_softplus(c,biased);stages["alpha_softplus"]=soft;
        decay=ggml_reshape_4d(c,ggml_mul(c,soft,input(in.A,32)),1,32,n,1);
        beta=ggml_reshape_4d(c,ggml_sigmoid(c,input(bb,32,n)),1,32,n,1);
    } else {
        nativeConv.resize(size_t(n)*8192);nativeSilu.resize(nativeConv.size());V q(size_t(n)*4096),k(q.size()),v(q.size());gates.resize(size_t(n)*32);mixing.resize(gates.size());
        V biased(gates.size()),soft(gates.size());
        for(int t=0;t<n;++t){
            for(int ch=0;ch<8192;++ch){float sum=0;
                for(int tap=0;tap<4;++tap){int token=t-3+tap;sum+=(token>=0?in.qkv[size_t(token)*8192+ch]:0.0f)*convHF[size_t(ch)*4+tap];}
                nativeConv[size_t(t)*8192+ch]=sum;const float value=sum*sigmoid(sum);nativeSilu[size_t(t)*8192+ch]=value;
                if(ch<4096){const int compact=ch%2048,head=compact/128,lane=compact%128;auto & dest=ch<2048?q:k;
                    for(int r=0;r<2;++r)dest[size_t(t)*4096+(head*2+r)*128+lane]=value;
                }else v[size_t(t)*4096+ch-4096]=value;
            }
            for(int h=0;h<32;++h){size_t i=size_t(t)*32+h;float time=in.a[i]+dthf[h];biased[i]=time;soft[i]=time>20?time:std::log1p(std::exp(time));gates[i]=Ahf[h]*soft[i];mixing[i]=sigmoid(in.b[i]);}
        }
        stages["conv_raw"]=input(nativeConv,8192,n);stages["conv_silu"]=input(nativeSilu,8192,n);
        stages["alpha_biased"]=input(biased,32,n);stages["alpha_softplus"]=input(soft,32,n);
        auto normalize=[&](ggml_tensor * x){auto * den=ggml_sqrt(c,ggml_add1(c,ggml_sum_rows(c,ggml_sqr(c,x)),ggml_new_f32(c,1e-6f)));return ggml_div(c,x,den);};
        qr=normalize(ggml_reshape_4d(c,input(q,128,32,n),128,32,n,1));
        kr=normalize(ggml_reshape_4d(c,input(k,128,32,n),128,32,n,1));
        vr=ggml_reshape_4d(c,input(v,128,32,n),128,32,n,1);
        decay=ggml_reshape_4d(c,input(gates,1,32,n),1,32,n,1);beta=ggml_reshape_4d(c,input(mixing,1,32,n),1,32,n,1);
    }
    stages["q_norm"]=qr;stages["k_norm"]=kr;stages["decay"]=decay;stages["beta"]=beta;
    auto * state=ggml_new_tensor_4d(c,GGML_TYPE_F32,128,128,32,1);std::memset(state->data,0,ggml_nbytes(state));
    auto * result=ggml_gated_delta_net(c,qr,kr,vr,decay,beta,state,1);
    need(size_t(ggml_nelements(result))==size_t(n)*4096+524288,"fused K1 result geometry");
    auto * delta=ggml_view_4d(c,result,128,32,n,1,128*4,4096*4,size_t(n)*4096*4,0);stages["delta"]=delta;
    auto * finalState=ggml_view_4d(c,result,128,128,32,1,128*4,128*128*4,524288*4,size_t(n)*4096*4);stages["state"]=finalState;
    ggml_tensor * sourcePost=nullptr,*sourceDirect=nullptr,*sourceDequant=nullptr;
    if(source){
        auto * rms=ggml_rms_norm(c,delta,1e-6f);auto * weighted=ggml_mul(c,rms,input(in.norm,128));
        auto * zs=ggml_silu(c,ggml_reshape_4d(c,input(z,128,32,n),128,32,n,1));
        sourcePost=ggml_mul(c,weighted,zs);stages["post_rms"]=rms;stages["post_weighted"]=weighted;stages["z_silu"]=zs;stages["postnorm"]=sourcePost;
        auto * flat=ggml_reshape_2d(c,sourcePost,4096,n);auto * w=ggml_new_tensor_2d(c,GGML_TYPE_Q4_K,4096,2560);
        std::ifstream f(weights/"out.q4k",std::ios::binary|std::ios::ate);need(f&&f.tellg()==std::streamoff(ggml_nbytes(w)),"exact source out bytes");f.seekg(0);need(bool(f.read(static_cast<char *>(w->data),ggml_nbytes(w))),"source out read");
        sourceDirect=ggml_mul_mat(c,w,flat);ggml_mul_mat_set_prec(sourceDirect,GGML_PREC_F32);
        if(observe){auto * wf=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,2560);ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(w->data,static_cast<float *>(wf->data),ggml_nelements(w));sourceDequant=ggml_mul_mat(c,wf,flat);ggml_mul_mat_set_prec(sourceDequant,GGML_PREC_F32);}
    }
    auto * graph=ggml_new_graph(c);
    // Stage observation adds sinks only; an independent composed run must match.
    if(observe)for(auto & entry:stages){
        // CPU RMS+MUL fusion may skip an intermediate unless explicitly requested.
        ggml_set_output(entry.second);ggml_build_forward_expand(graph,entry.second);
    }
    ggml_build_forward_expand(graph,source?sourceDirect:delta);if(sourceDequant)ggml_build_forward_expand(graph,sourceDequant);ggml_build_forward_expand(graph,finalState);
    need(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"fused GDN compute");
    std::map<std::string,V> out;
    for(auto & [name,t]:stages){
        if(name=="state"){
            V s(524288);const auto * p=static_cast<const float *>(t->data);
            for(int h=0;h<32;++h)for(int key=0;key<128;++key)for(int value=0;value<128;++value)
                s[(size_t(h)*128+key)*128+value]=p[(size_t(source?ph(h):h)*128+value)*128+key];out[name]=std::move(s);
        }else if(name=="q_norm"||name=="k_norm"){
            V compact(size_t(n)*2048);const auto * p=static_cast<const float *>(t->data);
            for(int token=0;token<n;++token)for(int h=0;h<16;++h)for(int d=0;d<128;++d)
                compact[size_t(token)*2048+h*128+d]=p[(size_t(token)*(source?16:32)+(source?h:2*h))*128+d];out[name]=std::move(compact);
        }else out[name]=to_hf(static_cast<const float *>(t->data),n,int(ggml_nelements(t))/n,source,name=="conv_raw"||name=="conv_silu"?4096:0);
    }
    V values(size_t(n)*4096);
    if(!source){
        out["post_rms"].resize(values.size());out["post_weighted"].resize(values.size());out["z_silu"].resize(values.size());
        for(size_t row=0;row<size_t(n)*32;++row){double sum=0;for(int d=0;d<128;++d){float x=out["delta"][row*128+d];sum+=double(x)*x;}
            const float scale=1.0f/std::sqrt(float(sum/128)+1e-6f);
            for(int d=0;d<128;++d){size_t i=row*128+d;out["post_rms"][i]=out["delta"][i]*scale;out["post_weighted"][i]=out["post_rms"][i]*in.norm[d];out["z_silu"][i]=in.z[i]*sigmoid(in.z[i]);values[i]=out["delta"][i]*scale*in.norm[d]*in.z[i]*sigmoid(in.z[i]);}
        }
    }else{
        values=to_hf(static_cast<const float *>(sourcePost->data),n,4096,true);
    }
    out["postnorm"]=values;
    if(source&&observe)need(std::any_of(out.at("post_rms").begin(),out.at("post_rms").end(),[](float x){return x!=0;}),"actual nonzero RMS stage must be materialized");
#ifdef CRISPEMBED_GDN_RUNTIME_PROBE
    if(!source){
        qwen35_ocr::linear_inputs li{};li.batch=1;li.tokens=n;li.heads=32;li.width=128;li.conv_kernel=4;li.threads=4;li.norm_eps=1e-6f;
        li.qkv=in.qkv.data();li.z=in.z.data();li.a=in.a.data();li.b=in.b.data();li.conv_weights=convHF.data();li.a_log=Ahf.data();li.dt_bias=dthf.data();li.norm_weights=in.norm.data();li.a_is_pretransformed=true;
        qwen35_ocr::linear_outputs actual;need(qwen35_ocr::linear_grouped_core_cpu(li,16,actual),"actual native grouped GDN");
        need(actual.values.size()==values.size()&&!std::memcmp(actual.values.data(),values.data(),values.size()*4),"reconstructed native GDN exact actual helper");
        need(actual.state.size()==out["state"].size()&&!std::memcmp(actual.state.data(),out["state"].data(),actual.state.size()*4),"reconstructed native state exact actual helper");
        out["actual_native_values"]=actual.values;
    }
#endif
    auto projected=project_pair(values,n,weights);
    if(source){const V raw(static_cast<float *>(sourceDirect->data),static_cast<float *>(sourceDirect->data)+size_t(n)*2560);need(!std::memcmp(raw.data(),projected.at("out_direct").data(),raw.size()*4),"source composed direct out equals same-input standalone projection");if(sourceDequant){const V deq(static_cast<float *>(sourceDequant->data),static_cast<float *>(sourceDequant->data)+size_t(n)*2560);need(!std::memcmp(deq.data(),projected.at("out_dequant").data(),deq.size()*4),"source auxiliary dequant out equals same-input standalone projection");}}
    out.insert(projected.begin(),projected.end());
    return out;
}
int main(int argc,char **argv){try{
    need(argc==6,"MODE TOKENS ACTUAL_INPUT_DIR WEIGHTS_DIR OUTPUT_DIR");
    const std::string mode=argv[1];const int n=std::stoi(argv[2]);need((mode=="source"||mode=="native"||mode=="out")&&(n==4||n==16),"closed mode/geometry");
#ifndef CRISPEMBED_GDN_RUNTIME_PROBE
    need(mode!="native","native helper requires exact runtime source");
#endif
    const std::filesystem::path inputs=argv[3],weights=argv[4],output=argv[5];need(std::filesystem::is_directory(output),"output directory");
    if(mode=="out"){
        V x=load(inputs/"postnorm.f32",size_t(n)*4096,size_t(n)*4096);auto result=project_pair(x,n,weights);
        for(auto & [name,v]:result)dump(output/(name+".f32"),v);return 0;
    }
    for(const char * name:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_QWEN35_QUANT","CRISPEMBED_QWEN35_GDN_GROUPED_HEADS","CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL","CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","CRISPEMBED_LIGHTON4B_QUANT","CRISPEMBED_LIGHTON4B_Q6_K"})setenv(name,"1",1);
    for(const char * name:{"CRISPEMBED_LIGHTON4B_BF16_LINEAR","CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS","CRISPEMBED_LIGHTON4B_HEAD_BF16","CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS","CRISPEMBED_QWEN35_PRECISE_RMS"})setenv(name,"0",1);
    Inputs in;
    auto stage=[&](const char * name,int width){return load(inputs/(std::string("prefill.first_chunk.layer0.")+name+".f32"),size_t(16)*width,size_t(n)*width);};
    in.qkv=stage("qkv",8192);in.z=stage("z",4096);in.a=stage("a",32);in.b=stage("b",32);
    in.conv=load(weights/"conv.f32",8192*4,8192*4);in.A=load(weights/"A.f32",32,32);in.dt=load(weights/"dt.f32",32,32);in.norm=load(weights/"norm.f32",128,128);
    const bool source=mode=="source";auto composed=compute(in,n,source,false,weights);auto observed=compute(in,n,source,true,weights);
    for(const char * name:{"delta","state","postnorm","out_direct","out_dequant"}){
        const auto & a=composed.at(name);const auto & b=observed.at(name);need(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"stage-observer composed control changed");
    }
    bool actual=false;
    if(!source&&n==16){const auto trace=load(inputs/"prefill.first_chunk.layer0.attention.f32",16*2560,16*2560);const auto & got=observed.at("out_dequant");need(trace.size()==got.size()&&!std::memcmp(trace.data(),got.data(),got.size()*4),"native full GDN/out pipeline differs actual main attention");actual=true;}
    for(auto & [name,v]:observed)dump(output/(name+".f32"),v);
    std::ofstream report(output/"probe-control.json");report<<"{\"tokens\":"<<n<<",\"source_graph_blueprint\":"<<(source?"true":"false")<<",\"observer_composed_bit_exact\":true,\"actual_main_attention_bit_exact\":"<<(actual?"true":"false")<<"}\n";
    return 0;
}catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
