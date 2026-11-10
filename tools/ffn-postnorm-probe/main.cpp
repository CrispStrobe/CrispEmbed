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
static ggml_type activation_type(){const auto*t=ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);need(t&&t->vec_dot_type==GGML_TYPE_Q8_K&&ggml_type_size(t->vec_dot_type)==292,"actual Q4 activation type and declared Q8_K layout");return t->vec_dot_type;}
static Bytes codes(const V&x,int n){
 need(x.size()==size_t(n)*2560,"closed Q4 gate/up input");const auto type=activation_type();const auto*t=ggml_get_type_traits_cpu(type);need(t&&t->from_float,"public actual activation converter");const size_t row=ggml_row_size(type,2560);Bytes b(row*size_t(n));for(int i=0;i<n;++i)t->from_float(x.data()+size_t(i)*2560,b.data()+size_t(i)*row,2560);return b;
}
static void validate_codes(const Bytes&b,int n){
 need(b.size()==size_t(n)*10*292,"complete declared Q8_K input blocks");for(size_t off=0;off<b.size();off+=292){float d;std::memcpy(&d,b.data()+off,4);need(std::isfinite(d),"finite packed scale");for(int group=0;group<16;++group){int sum=0;for(int i=0;i<16;++i){int8_t q;std::memcpy(&q,b.data()+off+4+group*16+i,1);sum+=q;}int16_t stored;std::memcpy(&stored,b.data()+off+260+group*2,2);need(sum==stored,"Q8_K sums agree with integer codes");}}
}
static V project(const V&x,int n,const Bytes&weights,const std::string&role,bool preferred,bool prequantized,const Bytes*override_codes=nullptr){
 need(role=="gate"||role=="up","closed actual Q4 FFN role");need(!preferred||!prequantized,"preferred only actual F32-input path");const int width=2560,rows=9216;const auto type=GGML_TYPE_Q4_K;
 need(x.size()==size_t(n)*width&&weights.size()==ggml_row_size(type,width)*size_t(rows),"exact actual matrix/input extent");auto c=context(8ull<<20);auto wc=context(65536,true);auto b=backend();auto*w=ggml_new_tensor_2d(wc.get(),type,width,rows);Buffer storage(nullptr,ggml_backend_buffer_free);
 if(preferred){auto*dev=ggml_backend_get_device(b.get());auto fn=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_backend_dev_get_extra_bufts");need(fn,"public preferred buffer extension");auto**types=fn(dev);ggml_backend_buffer_type_t selected=nullptr;for(size_t i=0;types&&types[i]&&i<32;++i)if(std::string(ggml_backend_buft_name(types[i]))=="CPU_REPACK"){need(!selected,"unique selected buffer");selected=types[i];}need(selected,"actual Q4 preferred buffer");storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(wc.get(),selected));need(storage&&w->data&&w->buffer&&w->extra,"initialized actual preferred weight");ggml_backend_buffer_set_usage(storage.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(w,weights.data(),0,weights.size());}
 else{storage.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),weights.size()));need(bool(storage),"aligned raw original quantized storage");w->data=ggml_backend_buffer_get_base(storage.get());need(w->data,"raw base");std::memcpy(w->data,weights.data(),weights.size());}
 auto*input=ggml_new_tensor_2d(c.get(),prequantized?activation_type():GGML_TYPE_F32,width,n);if(prequantized){auto packed=override_codes?*override_codes:codes(x,n);validate_codes(packed,n);need(packed.size()==ggml_nbytes(input),"actual packed input geometry");std::memcpy(input->data,packed.data(),packed.size());}else std::memcpy(input->data,x.data(),x.size()*4);
 auto*y=ggml_mul_mat(c.get(),w,input);ggml_mul_mat_set_prec(y,GGML_PREC_F32);std::fill_n(static_cast<float*>(y->data),size_t(n)*rows,std::numeric_limits<float>::quiet_NaN());auto*graph=ggml_new_graph(c.get());ggml_build_forward_expand(graph,y);need(ggml_backend_graph_compute(b.get(),graph)==GGML_STATUS_SUCCESS,"actual same-weight FFN projection");return V(static_cast<float*>(y->data),static_cast<float*>(y->data)+size_t(n)*rows);
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--controls"){
  auto initialized=backend();Bytes weights(ggml_row_size(GGML_TYPE_Q4_K,2560)*9216,0);V x(2560,1.0f/32);auto packed=codes(x,1);validate_codes(packed,1);
  for(bool preferred:{false,true}){auto y=project(x,1,weights,"gate",preferred,false);need(std::all_of(y.begin(),y.end(),[](float v){return v==0;}),"analytic zero actual Q4 backend");}
  for(bool explicit_codes:{false,true}){auto y=project(x,1,weights,"gate",false,true,explicit_codes?&packed:nullptr);need(std::all_of(y.begin(),y.end(),[](float v){return v==0;}),"actual Q8_K RHS support");}
  auto bad=packed;bad[260]^=1;bool rejected=false;try{validate_codes(bad,1);}catch(const std::exception&){rejected=true;}need(rejected,"invalid integer/sum coupling rejected");
  std::puts("FFN_POSTNORM_CONTROLS PASS model=0 packed_rhs=1 synthetic_weights=1");return 0;
 }
 need(argc==7,"MODE ROLE TOKENS INPUT_DIR WEIGHTS_DIR OUTPUT_FILE");const std::string mode=argv[1],role=argv[2];const int n=std::stoi(argv[3]);need((n==1||n==4||n==16)&&(role=="gate"||role=="up"),"closed actual extent/role");const std::filesystem::path input=argv[4],weights=argv[5],output=argv[6];
 if(mode=="codes"){auto initialized=backend();auto b=codes(floats(input/"postnorm.f32",size_t(n)*2560),n);validate_codes(b,n);write(output,b.data(),b.size());}
 else{need(mode=="raw"||mode=="preferred"||mode=="prequantized"||mode=="packed","closed actual projection/intervention");V x;Bytes override_codes;if(mode=="packed"){x.resize(size_t(n)*2560);override_codes=read(input/"q8_codes.bin",size_t(n)*10*292);validate_codes(override_codes,n);}else x=floats(input/"postnorm.f32",size_t(n)*2560);auto w=read(weights/(role+".quant"),ggml_row_size(GGML_TYPE_Q4_K,2560)*9216);dump(output,project(x,n,w,role,mode=="preferred",mode=="prequantized"||mode=="packed",mode=="packed"?&override_codes:nullptr));}
 return 0;
}catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
