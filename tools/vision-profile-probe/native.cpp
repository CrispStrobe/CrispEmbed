// Reference-only adapter: authenticated vision suffix, never a product loader.
#include "lighton4b_image_compose.h"
#include "ggml-cpu.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
static void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
struct mapped_file {
 int fd=-1;void* data=MAP_FAILED;size_t size=0;
 explicit mapped_file(const char*path){fd=open(path,O_RDONLY);require(fd>=0,"suffix open");struct stat s{};require(fstat(fd,&s)==0&&s.st_size==1334056960,"exact vision-only suffix bytes");size=size_t(s.st_size);data=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);require(data!=MAP_FAILED,"readonly mapping");}
 ~mapped_file(){if(data!=MAP_FAILED)munmap(data,size);if(fd>=0)close(fd);}
};
static void floats(const std::string&p,const std::vector<float>&x){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),x.size()*4);require(bool(f),"output write");}
static void owner_gates() {
 for(auto*k:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL","CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE","CRISPEMBED_LIGHTON4B_VISION_OWNER"})setenv(k,"1",1);
}
static core_gguf::tensor_map schema_metadata(const char*schema_path,ggml_context*ctx,void*base,ggml_backend_buffer_t buffer) {
 core_gguf::tensor_map tensors;std::ifstream schema(schema_path);std::string name;size_t cursor=0;int count=0;
 while(schema>>name){int64_t ne[4];size_t offset,bytes;require(bool(schema>>ne[0]>>ne[1]>>ne[2]>>ne[3]>>offset>>bytes),"complete schema row");require(name.rfind("v.",0)==0&&ne[0]>0&&ne[1]>0&&ne[2]>0&&ne[3]>0&&offset==cursor&&bytes<=size_t(1334056960)-offset,"closed contiguous F32 schema");auto*t=ggml_new_tensor(ctx,GGML_TYPE_F32,4,ne);require(ggml_nbytes(t)==bytes,"schema shape bytes");ggml_set_name(t,name.c_str());t->data=base?static_cast<char*>(base)+offset:nullptr;t->buffer=buffer;require(tensors.emplace(name,t).second,"unique tensor");cursor+=bytes;++count;}
 require(schema.eof()&&count==297&&cursor==size_t(1334056960),"all297 protected vision tensors only");
 return tensors;
}
int main(int argc,char**argv){try{
 owner_gates();
 if(argc==4&&std::string(argv[1])=="--metadata-controls") {
  require(lighton4b_physical_f16_vision_available(),"supported metadata control profile");
  std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({4u<<20,nullptr,true}),ggml_free);require(bool(ctx),"metadata context");
  mapped_file sparse(argv[2]);
  std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_cpu_buffer_from_ptr(sparse.data,sparse.size),ggml_backend_buffer_free);require(bool(buffer),"metadata mapped CPU buffer");
  auto tensors=schema_metadata(argv[3],ctx.get(),sparse.data,buffer.get());vision_weights weights;vision_config config;
  require(map_lighton4b_vision(tensors,weights,config)&&weights.layers.size()==24,"actual positive metadata mapping");
  setenv("CRISPEMBED_QWEN35_OCR","0",1);require(!map_lighton4b_vision(tensors,weights,config)&&weights.layers.empty(),"OCR-off metadata mapping closed");
  owner_gates();require(map_lighton4b_vision(tensors,weights,config),"restored actual metadata mapping");
  std::cout<<"VISION_METADATA_CONTROLS PASS tensors=297 checkpoint_bytes=0\n";return 0;
 }

 require(argc==8,"suffix schema RGB height width legacy|f16 output required");
 const int h=std::stoi(argv[4]),w=std::stoi(argv[5]);const std::string mode=argv[6],out=argv[7];
 require(h>0&&w>0&&h<=4096&&w<=4096&&(mode=="legacy"||mode=="f16"),"bounded RGB and mode");
 require(lighton4b_physical_f16_vision_available(),"supported real CPU/private ABI required");
 mapped_file file(argv[1]);
 std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({4u<<20,nullptr,true}),ggml_free);require(bool(ctx),"metadata context");
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_cpu_buffer_from_ptr(file.data,file.size),ggml_backend_buffer_free);require(bool(buffer),"mapped CPU buffer");
 auto tensors=schema_metadata(argv[2],ctx.get(),file.data,buffer.get());
 vision_weights weights;vision_config config;require(map_lighton4b_vision(tensors,weights,config),"actual validated vision map");
 std::vector<uint8_t>rgb(size_t(h)*w*3);std::ifstream input(argv[3],std::ios::binary);input.read(reinterpret_cast<char*>(rgb.data()),rgb.size());require(size_t(input.gcount())==rgb.size()&&input.peek()==EOF,"exact RGB bytes");
 // The legacy lane is closed by controller-provided frozen BF16 gate tuple.
 setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE",mode=="f16"?"1":"0",1);setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION",mode=="f16"?"1":"0",1);
 image_inputs image;vision_outputs result;require(lighton4b_preprocess_rgb(rgb.data(),h,w,image,true),"actual physical preprocess");require(lighton4b_image_vision_cpu(weights,config,image,4,result,true),"actual physical vision");
 std::ofstream resized(out+"/resized.rgb",std::ios::binary);resized.write(reinterpret_cast<const char*>(image.resized_rgb.data()),image.resized_rgb.size());require(bool(resized),"RGB output");
 floats(out+"/processor.patches.f32",image.patches);floats(out+"/vision.patch.f32",result.patch);floats(out+"/vision.position.f32",result.position);floats(out+"/vision.embedded.f32",result.embedded);for(size_t i=0;i<result.hidden.size();++i)floats(out+"/vision.layer."+std::to_string(i)+".f32",result.hidden[i]);floats(out+"/vision.merger_norm.f32",result.merger_norm);floats(out+"/vision.merged.f32",result.merged);
 std::cout<<"VISION_PROFILE PASS "<<mode<<" layers="<<result.hidden.size()<<" patches="<<image.grid[1]*image.grid[2]<<" text_model=0\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
