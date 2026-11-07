#include "clip.h"
#include "clip-impl.h"
#include "mtmd-image.h"
#include "ggml-backend.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iomanip>
static void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
int main(int argc,char**argv){try{
 require(argc==6,"projector RGB height width output-prefix required");int h=std::stoi(argv[3]),w=std::stoi(argv[4]);require(h>0&&w>0&&h<=4096&&w<=4096,"observed image bound");
 std::ifstream f(argv[2],std::ios::binary);std::vector<uint8_t> bytes(size_t(h)*w*3);f.read(reinterpret_cast<char*>(bytes.data()),bytes.size());require(size_t(f.gcount())==bytes.size()&&f.peek()==EOF,"exact actual RGB");
 std::ofstream raw(std::string(argv[5])+".input.rgb",std::ios::binary);raw.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());require(bool(raw),"actual read RGB capture");raw.close();
 ggml_backend_load_all();clip_context_params params{};params.use_gpu=false;params.device=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);require(params.device,"CPU device");params.flash_attn_type=CLIP_FLASH_ATTN_TYPE_DISABLED;params.image_min_tokens=64;params.image_max_tokens=256;params.warmup=false;
 auto model=clip_init(argv[1],params);require(model.ctx_v&&!model.ctx_a&&!model.ctx_gen_a,"vision-only projector");
 clip_image_u8 image;image.set_size({w,h},false);image.cpy_buf(bytes);mtmd_image_preprocessor_qwen3vl processor(model.ctx_v);auto prepared=processor.preprocess(image);require(prepared.entries.size()==1&&!prepared.has_overview(),"single observed image geometry");auto &img=prepared.entries[0];int tokens=clip_n_output_tokens(model.ctx_v,&img);require(tokens==140&&clip_n_mmproj_embd(model.ctx_v)==2560,"frozen form140x2560");
 const auto &pixels=img.get_ro_buf();std::ofstream pf(std::string(argv[5])+".pixels.f32",std::ios::binary);pf.write(reinterpret_cast<const char*>(pixels.data()),pixels.size()*4);require(bool(pf),"normalized pixels output");pf.close();
 std::vector<float> out(size_t(tokens)*2560);require(clip_image_encode(model.ctx_v,4,&img,out),"source encoder");require(std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);}),"finite merged output");std::ofstream of(std::string(argv[5])+".merged.f32",std::ios::binary);of.write(reinterpret_cast<const char*>(out.data()),out.size()*4);require(bool(of),"merged output");
 std::ofstream meta(std::string(argv[5])+".json");meta<<std::setprecision(9);const auto*hp=clip_get_hparams(model.ctx_v);meta<<"{\"mean\":["<<hp->image_mean[0]<<","<<hp->image_mean[1]<<","<<hp->image_mean[2]<<"],\"std\":["<<hp->image_std[0]<<","<<hp->image_std[1]<<","<<hp->image_std[2]<<"],";meta<<"\"height\":"<<img.ny()<<",\"width\":"<<img.nx()<<",\"tokens\":"<<tokens<<",\"threads\":4,\"embedding\":2560,\"text_model_loaded\":false}\n";require(bool(meta),"metadata");clip_free(model.ctx_v);std::cout<<"PROJECTOR_SOURCE PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
