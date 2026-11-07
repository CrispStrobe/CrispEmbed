// Reference-only actual-input FFN attribution. Never linked into a product.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using V=std::vector<float>;
using Bytes=std::vector<unsigned char>;
static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
static Bytes read(const std::filesystem::path&p,size_t bytes){
 std::ifstream f(p,std::ios::binary|std::ios::ate);need(f&&f.tellg()==std::streamoff(bytes),"exact bounded input bytes");f.seekg(0);Bytes b(bytes);need(bool(f.read(reinterpret_cast<char*>(b.data()),bytes)),"complete input read");return b;
}
static V floats(const std::filesystem::path&p,size_t elements){auto b=read(p,elements*4);V x(elements);std::memcpy(x.data(),b.data(),b.size());need(std::all_of(x.begin(),x.end(),[](float v){return std::isfinite(v);}),"finite actual inputs");return x;}
static void write(const std::filesystem::path&p,const void*data,size_t bytes){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),bytes);need(bool(f),"complete output write");}
static void dump(const std::filesystem::path&p,const V&x){need(std::all_of(x.begin(),x.end(),[](float v){return std::isfinite(v);}),"finite output");write(p,x.data(),x.size()*4);}
using Ctx=std::unique_ptr<ggml_context,decltype(&ggml_free)>;
using Backend=std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>;
using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>;
static Ctx context(size_t bytes,bool no_alloc=false){Ctx c(ggml_init({bytes,nullptr,no_alloc}),ggml_free);need(bool(c),"bounded context allocation");return c;}
static Backend backend(){Backend b(ggml_backend_cpu_init(),ggml_backend_free);need(bool(b),"CPU backend");ggml_backend_cpu_set_n_threads(b.get(),4);return b;}
static V activate(const V&gate,const V&up,int n,bool scalar){
 need(gate.size()==size_t(n)*9216&&up.size()==gate.size(),"closed FFN activation shape");
 V result(gate.size());
 if(scalar){for(size_t i=0;i<gate.size();++i)result[i]=(gate[i]/(1.0f+std::exp(-gate[i])))*up[i];return result;}
 auto c=context(8ull<<20);auto b=backend();auto *g=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,9216,n),*u=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,9216,n);std::memcpy(g->data,gate.data(),gate.size()*4);std::memcpy(u->data,up.data(),up.size()*4);
 auto *y=ggml_swiglu_split(c.get(),g,u);auto *graph=ggml_new_graph(c.get());ggml_build_forward_expand(graph,y);need(ggml_backend_graph_compute(b.get(),graph)==GGML_STATUS_SUCCESS,"actual fused SwiGLU compute");std::memcpy(result.data(),y->data,result.size()*4);return result;
}
static ggml_type activation_type(){const auto*t=ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);need(t&&t->vec_dot_type==GGML_TYPE_Q8_K&&ggml_type_size(t->vec_dot_type)==292,"actual Q6 activation type and declared Q8_K layout");return t->vec_dot_type;}
static Bytes codes(const V&x,int n){
 need(x.size()==size_t(n)*9216,"closed Q6 down input");const auto type=activation_type();const auto*t=ggml_get_type_traits_cpu(type);need(t&&t->from_float,"public actual activation converter");const size_t row=ggml_row_size(type,9216);Bytes b(row*size_t(n));for(int i=0;i<n;++i)t->from_float(x.data()+size_t(i)*9216,b.data()+size_t(i)*row,9216);return b;
}
static V project(const V&x,int n,const Bytes&weights,const std::string&role,bool preferred,bool prequantized){
 const bool down=role=="down";need(down||role=="gate"||role=="up","closed actual FFN role");need(!preferred||!down,"actual Q6 down is not relabeled as Q4 repack");need(!prequantized||down,"Q8_K intervention only on Q6 down");const int width=down?9216:2560,rows=down?2560:9216;const auto type=down?GGML_TYPE_Q6_K:GGML_TYPE_Q4_K;
 need(x.size()==size_t(n)*width&&weights.size()==ggml_row_size(type,width)*size_t(rows),"exact actual matrix/input extent");auto c=context(8ull<<20);auto wc=context(65536,true);auto b=backend();auto*w=ggml_new_tensor_2d(wc.get(),type,width,rows);Buffer storage(nullptr,ggml_backend_buffer_free);
 if(preferred){auto*dev=ggml_backend_get_device(b.get());auto fn=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_backend_dev_get_extra_bufts");need(fn,"public preferred buffer extension");auto**types=fn(dev);ggml_backend_buffer_type_t selected=nullptr;for(size_t i=0;types&&types[i]&&i<32;++i)if(std::string(ggml_backend_buft_name(types[i]))=="CPU_REPACK"){need(!selected,"unique selected buffer");selected=types[i];}need(selected,"actual Q4 preferred buffer");storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(wc.get(),selected));need(storage&&w->data&&w->buffer&&w->extra,"initialized actual preferred weight");ggml_backend_buffer_set_usage(storage.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(w,weights.data(),0,weights.size());}
 else{storage.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),weights.size()));need(bool(storage),"aligned raw original quantized storage");w->data=ggml_backend_buffer_get_base(storage.get());need(w->data,"raw base");std::memcpy(w->data,weights.data(),weights.size());}
 auto*input=ggml_new_tensor_2d(c.get(),prequantized?activation_type():GGML_TYPE_F32,width,n);if(prequantized){auto packed=codes(x,n);need(packed.size()==ggml_nbytes(input),"actual packed input geometry");std::memcpy(input->data,packed.data(),packed.size());}else std::memcpy(input->data,x.data(),x.size()*4);
 auto*y=ggml_mul_mat(c.get(),w,input);ggml_mul_mat_set_prec(y,GGML_PREC_F32);std::fill_n(static_cast<float*>(y->data),size_t(n)*rows,std::numeric_limits<float>::quiet_NaN());auto*graph=ggml_new_graph(c.get());ggml_build_forward_expand(graph,y);need(ggml_backend_graph_compute(b.get(),graph)==GGML_STATUS_SUCCESS,"actual same-weight FFN projection");return V(static_cast<float*>(y->data),static_cast<float*>(y->data)+size_t(n)*rows);
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--controls"){
  for(int n:{1,4,16})for(bool scalar:{false,true}){auto y=activate(V(size_t(n)*9216,0),V(size_t(n)*9216,1),n,scalar);need(std::all_of(y.begin(),y.end(),[](float x){return x==0;}),"analytic zero activation");}
  for(const std::string role:{"gate","down"}){const bool down=role=="down";const int width=down?9216:2560,rows=down?2560:9216;Bytes zeros(ggml_row_size(down?GGML_TYPE_Q6_K:GGML_TYPE_Q4_K,width)*size_t(rows),0);for(bool mode:{false,true}){auto y=project(V(width,1.0f/32),1,zeros,role,down?false:mode,down?mode:false);need(std::all_of(y.begin(),y.end(),[](float x){return x==0;}),"analytic zero matrix/backend control");}}
  std::puts("SWIGLU_SOURCE_CONTROLS PASS model=0 quant_weights=synthetic");return 0;
 }
 need(argc==7,"MODE ROLE TOKENS INPUT_DIR WEIGHTS_DIR OUTPUT_FILE");const std::string mode=argv[1],role=argv[2];const int n=std::stoi(argv[3]);need(n==1||n==4||n==16,"closed actual token extent");const std::filesystem::path input=argv[4],weights=argv[5],output=argv[6];
 if(mode=="scalar"||mode=="ggml"){need(role=="activation","activation role");dump(output,activate(floats(input/"gate.f32",size_t(n)*9216),floats(input/"up.f32",size_t(n)*9216),n,mode=="scalar"));}
 else if(mode=="codes"){need(role=="down","codes down scope");auto initialized=backend();auto x=codes(floats(input/"activation.f32",size_t(n)*9216),n);write(output,x.data(),x.size());}
 else{need(mode=="raw"||mode=="preferred"||mode=="prequantized","closed projection mode");const bool down=role=="down";need(down||role=="gate"||role=="up","projection role");const int width=down?9216:2560,rows=down?2560:9216;const auto type=down?GGML_TYPE_Q6_K:GGML_TYPE_Q4_K;dump(output,project(floats(input/(down?"activation.f32":"postnorm.f32"),size_t(n)*width),n,read(weights/(role+".quant"),ggml_row_size(type,width)*size_t(rows)),role,mode=="preferred",mode=="prequantized"));}
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
