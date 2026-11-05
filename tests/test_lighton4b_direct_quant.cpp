#include "qwen35_math.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
static void gate(const char * name,const char * value) {
#ifdef _WIN32
 _putenv_s(name,value);
#else
 setenv(name,value,1);
#endif
}
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 std::cout<<std::setprecision(17);
 gate("CRISPEMBED_QWEN35_OCR","1");gate("CRISPEMBED_QWEN35_QUANT","1");
 gate("CRISPEMBED_LIGHTON4B_QUANT","1");gate("CRISPEMBED_QWEN35_STREAM_PROJECTION","1");
 for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_Q4_K}) for(auto dims:{std::pair<int,int>{2560,8192},{9216,2560},{2560,257}}) {
  const int width=dims.first,rows=dims.second;
  std::unique_ptr<ggml_context,decltype(&ggml_free)>ctx(ggml_init({16384,nullptr,true}),ggml_free);
  auto *w=ggml_new_tensor_2d(ctx.get(),type,width,rows);
  std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)>backend(ggml_backend_cpu_init(),ggml_backend_free);
  std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)>buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);
  if(!buffer)return 3;
  std::vector<float>values(size_t(width)*rows);uint32_t state=17;
  for(auto &x:values){state=1664525u*state+1013904223u;x=(float(int(state>>8)%65536)-32768.0f)/(32768.0f*std::sqrt(float(width)));}
  if(ggml_quantize_chunk(type,values.data(),w->data,0,rows,width,nullptr)!=ggml_nbytes(w))return 4;
  values.clear();values.shrink_to_fit();
  for(int tokens:{1,16}) {
   std::vector<float>x(size_t(width)*tokens);for(auto &v:x){state=1664525u*state+1013904223u;v=float(int(state>>8)%65536-32768)/11000.0f;}
   std::vector<std::vector<float>>baseline,direct,off;
   gate("CRISPEMBED_LIGHTON4B_DIRECT_QUANT","0");auto a=std::chrono::steady_clock::now();
   if(!qwen35_ocr::project_cpu({w},x.data(),width,tokens,4,baseline))return 5;
   auto b=std::chrono::steady_clock::now();gate("CRISPEMBED_LIGHTON4B_DIRECT_QUANT","1");
   if(!qwen35_ocr::project_cpu({w},x.data(),width,tokens,4,direct))return 6;
   auto c=std::chrono::steady_clock::now();gate("CRISPEMBED_LIGHTON4B_QUANT","0");
   if(!qwen35_ocr::project_cpu({w},x.data(),width,tokens,4,off)||off!=baseline)return 7;
   gate("CRISPEMBED_LIGHTON4B_QUANT","1");gate("CRISPEMBED_LIGHTON4B_DIRECT_QUANT","0");
   if(!qwen35_ocr::project_cpu({w},x.data(),width,tokens,4,off)||off!=baseline)return 8;
   gate("CRISPEMBED_LIGHTON4B_DIRECT_QUANT","1");std::vector<float>stream;
   if(!qwen35_ocr::project_rows_cpu(w,x.data(),width,tokens,4,128,stream)||stream.size()!=direct[0].size())return 9;
   double error=0,squares=0,dot=0,aa=0,bb=0,mag=0,nmag=0,seam=0;
   for(size_t i=0;i<stream.size();++i) {
    double p=baseline[0][i],q=direct[0][i];if(!std::isfinite(q))return 10;
    error=std::max(error,std::abs(p-q));squares+=(p-q)*(p-q);dot+=p*q;aa+=p*p;bb+=q*q;
    mag=std::max(mag,std::abs(p));nmag=std::max(nmag,std::abs(q));
    // One-row tails intentionally retain old F32 arithmetic rather than invent a quant row.
    if(i%rows!=size_t(rows-1)||rows%128!=1)seam=std::max(seam,std::abs(double(stream[i])-q));
   }
   const std::string label=std::string(ggml_type_name(type))+"-"+std::to_string(width)+"-"+std::to_string(rows)+"-"+std::to_string(tokens);
   for(auto item:{std::pair<const char*,const std::vector<float>*>{"baseline",&baseline[0]},{"direct",&direct[0]},{"stream",&stream}}) {
    std::ofstream f(std::string(argv[1])+"/"+label+"-"+item.first+".f32",std::ios::binary);
    f.write((const char*)item.second->data(),item.second->size()*4);if(!f)return 11;
   }
   std::cout<<"{\"case\":\""<<label<<"\",\"max_abs\":"<<error<<",\"rms_error\":"<<std::sqrt(squares/stream.size())
    <<",\"cosine\":"<<std::fixed<<dot/std::sqrt(aa*bb)<<",\"baseline_max_abs\":"<<mag<<",\"direct_max_abs\":"<<nmag
    <<",\"stream_max_abs\":"<<seam<<",\"baseline_ms\":"<<std::chrono::duration<double,std::milli>(b-a).count()
    <<",\"direct_ms\":"<<std::chrono::duration<double,std::milli>(c-b).count()<<"}\n";
  }
 }
 std::cout<<"DIRECT_QUANT_CONTROLS passed=1 quality_qualified=0\n";
}
