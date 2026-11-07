#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static std::vector<char> read(const char * path,size_t bytes){std::ifstream in(path,std::ios::binary);require(bool(in),"input open");std::vector<char> out(bytes);in.read(out.data(),bytes);require(size_t(in.gcount())==bytes&&in.peek()==EOF,"exact input bytes");return out;}
int main(int argc,char **argv){try{
 require(argc==11,"weights input type width rows tokens mode output-prefix threads chunk required");
 const int type=std::stoi(argv[3]),width=std::stoi(argv[4]),rows=std::stoi(argv[5]),tokens=std::stoi(argv[6]);const std::string mode=argv[7];const int chunk=std::stoi(argv[10]);require(chunk>=1&&chunk<=144,"closed chunk");require(std::stoi(argv[9])==1||std::stoi(argv[9])==4,"closed threads");
 require((type==GGML_TYPE_Q6_K)&&width>0&&width<=16384&&width%256==0&&rows>0&&rows<=16384&&(tokens>=1&&tokens<=144),"closed tensor shape/type");
 require(mode=="borrowed"||mode=="mapped","closed buffer mode");require(mode!="repack"||type==GGML_TYPE_Q4_K,"only observed Q4 repack type");
 const size_t packed=ggml_row_size(ggml_type(type),width)*size_t(rows);require(packed<=64*1024*1024,"bounded quantized matrix");auto bytes=read(argv[1],packed),input_bytes=read(argv[2],size_t(width)*tokens*sizeof(float));std::vector<float> input(size_t(width)*tokens);std::memcpy(input.data(),input_bytes.data(),input_bytes.size());require(std::all_of(input.begin(),input.end(),[](float x){return std::isfinite(x);}),"finite actual input");
 std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(),ggml_backend_free);require(bool(backend),"CPU backend");ggml_backend_cpu_set_n_threads(backend.get(),std::stoi(argv[9]));auto * device=ggml_backend_get_device(backend.get());require(device&&ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU,"actual CPU device");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> weights(ggml_init({65536,nullptr,true}),ggml_free);require(bool(weights),"bounded weights context");auto * weight=ggml_new_tensor_2d(weights.get(),ggml_type(type),width,rows);ggml_set_name(weight,"authenticated.weight");
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> storage(nullptr,ggml_backend_buffer_free),mapped(nullptr,ggml_backend_buffer_free);
 if(mode=="repack"){
  auto * registry=ggml_backend_dev_backend_reg(device);auto function=(ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(registry,"ggml_backend_dev_get_extra_bufts");require(function,"public CPU extra-buffer extension");auto ** types=function(device);ggml_backend_buffer_type_t selected=nullptr;
  for(size_t i=0;types&&types[i]&&i<32;++i)if(std::string(ggml_backend_buft_name(types[i]))=="CPU_REPACK"){require(!selected,"unique CPU_REPACK buffer type");selected=types[i];}
  require(selected,"observed CPU_REPACK type available");storage.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(),selected));require(storage&&weight->buffer&&weight->data&&weight->extra,"supported initialized repack metadata before upload");ggml_backend_buffer_set_usage(storage.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(weight,bytes.data(),0,packed);
 }else{
  storage.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(),packed));require(bool(storage),"aligned raw storage");void * data=ggml_backend_buffer_get_base(storage.get());require(data,"raw storage base");std::memcpy(data,bytes.data(),packed);weight->data=data;
  if(mode=="mapped"){mapped.reset(ggml_backend_cpu_buffer_from_ptr(data,packed));require(bool(mapped),"mapped storage wrapper");weight->buffer=mapped.get();}
 }
 std::vector<float> result(size_t(rows)*tokens);auto started=std::chrono::steady_clock::now();
 for(int offset=0;offset<tokens;offset+=chunk){const int count=std::min(chunk,tokens-offset);
  std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({32*1024*1024,nullptr,false}),ggml_free);require(bool(ctx),"bounded graph context");
  auto * x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width,count);std::memcpy(x->data,input.data()+size_t(offset)*width,size_t(count)*width*sizeof(float));auto * y=ggml_mul_mat(ctx.get(),weight,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);require(ggml_backend_dev_supports_op(device,y),"actual CPU supports selected buffer operation");auto * graph=ggml_new_graph_custom(ctx.get(),32,false);ggml_build_forward_expand(graph,y);require(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"selected matmul compute");
  auto * values=static_cast<float*>(y->data);require(std::all_of(values,values+size_t(rows)*count,[](float v){return std::isfinite(v);}),"finite projection output");std::memcpy(result.data()+size_t(offset)*rows,values,size_t(rows)*count*sizeof(float));
 }
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();auto * values=result.data();
 std::ofstream output(std::string(argv[8])+".f32",std::ios::binary);output.write(reinterpret_cast<char*>(values),size_t(rows)*tokens*sizeof(float));require(bool(output),"projection output write");output.close();
 auto * buffer=weight->buffer;std::ofstream meta(std::string(argv[8])+".json");meta<<"{\"mode\":"<<std::quoted(mode)<<",\"type\":"<<type<<",\"width\":"<<width<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"chunk\":"<<chunk<<",\"threads\":"<<std::stoi(argv[9])<<",\"quantized_bytes\":"<<packed<<",\"buffer\":"<<std::quoted(buffer?ggml_backend_buffer_name(buffer):"<none>")<<",\"buffer_type\":"<<std::quoted(buffer?ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer)):"<none>")<<",\"extra\":"<<(weight->extra?"true":"false")<<",\"host\":"<<(buffer&&ggml_backend_buffer_is_host(buffer)?"true":"false")<<",\"compute_seconds\":"<<std::setprecision(17)<<seconds<<"}\n";require(bool(meta),"metadata write");std::cout<<"REPACK_OPERATOR PASS "<<mode<<" "<<tokens<<"\n";return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
