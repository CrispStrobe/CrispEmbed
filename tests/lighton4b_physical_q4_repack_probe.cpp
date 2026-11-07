// Reference-only authenticated source-fed project_cpu adapter; never a product CLI.
#include "lighton4b_physical_native_ops.h"
#include "qwen35_linear.h"
#include "qwen35_math.h"
#include "qwen35_weight_cache.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace qwen35_ocr;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static std::vector<char> read(const char* file,size_t bytes){std::ifstream in(file,std::ios::binary);require(bool(in),"input open");std::vector<char> x(bytes);in.read(x.data(),bytes);require(size_t(in.gcount())==bytes&&in.peek()==EOF,"exact file bytes");return x;}
int main(int argc,char**argv){try{
 require(argc==9,"weights input type width rows tokens mode output required");
 int type=std::stoi(argv[3]),width=std::stoi(argv[4]),rows=std::stoi(argv[5]),tokens=std::stoi(argv[6]);std::string mode=argv[7];
 require((type==GGML_TYPE_Q4_K||type==GGML_TYPE_Q6_K)&&width==2560&&rows>0&&rows<=9216&&(tokens==1||tokens==4||tokens==16)&&(mode=="off"||mode=="on"),"closed source-fed shape/mode");
 for(auto*key:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","CRISPEMBED_LIGHTON4B_QUANT","CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL","CRISPEMBED_QWEN35_QUANT","CRISPEMBED_LIGHTON4B_Q6_K","CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS","CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION"})setenv(key,"1",1);
 for(auto*key:{"CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS","CRISPEMBED_LIGHTON4B_BF16_LINEAR","CRISPEMBED_LIGHTON4B_HEAD_BF16"})setenv(key,"0",1);
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS_ABLATION","all",1);setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK",mode=="on"?"1":"0",1);
 require(lighton4b_physical_f16_attention_available(),"supported real profile required");
 const size_t packed=ggml_row_size(ggml_type(type),width)*rows;require(packed<=64u*1024*1024,"matrix bound");auto raw=read(argv[1],packed),input=read(argv[2],size_t(width)*tokens*4);std::vector<float>x(size_t(width)*tokens);std::memcpy(x.data(),input.data(),input.size());
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({65536,nullptr,true}),ggml_free);auto*t=ggml_new_tensor_2d(ctx.get(),ggml_type(type),width,rows);
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),packed),ggml_backend_buffer_free);require(bool(buffer),"aligned raw allocation");t->data=ggml_backend_buffer_get_base(buffer.get());t->buffer=buffer.get();std::memcpy(t->data,raw.data(),packed);
 converted_weight_cache cache(0);if(mode=="on")require(cache.enable_physical_repack(true),"repack manager");std::vector<std::vector<float>> output;
 require(project_cpu({t},x.data(),width,tokens,4,output,mode=="on"?&cache:nullptr,true)&&output.size()==1&&output[0].size()==size_t(rows)*tokens,"actual runtime project");require(cache.bytes()==0&&cache.entries()==0,"no F32 weight cache");require(mode=="off"||cache.physical_repack_entries()==size_t(type==GGML_TYPE_Q4_K),"only Q4 preferred storage");
 require(std::memcmp(raw.data(),t->data,raw.size())==0&&!t->extra,"immutable original storage");
 std::ofstream out(argv[8],std::ios::binary);out.write(reinterpret_cast<char*>(output[0].data()),output[0].size()*4);require(bool(out),"output write");std::cout<<"RUNTIME_REPACK_SOURCEFED PASS "<<mode<<" "<<type<<" "<<tokens<<" charge="<<cache.physical_repack_bytes()<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
