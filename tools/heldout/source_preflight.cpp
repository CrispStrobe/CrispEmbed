// Reference-only public upstream metadata/tokenization API; never encode/decode.
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>
#include <utility>
static void need(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
// Upstream decode_embd_batch::set_position_mrope_2d packs t,y,x,z;
// decoder_pos field declaration order is t,x,y,z and is not batch order.
static std::array<uint32_t,3> source_axes(const mtmd_decoder_pos&p){return {p.t,p.y,p.x};}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--axis-control"){
  need(source_axes(mtmd_decoder_pos{7,11,13,0})==std::array<uint32_t,3>{7,13,11},"actual source t,y,x batch axes");
  std::cout<<"SOURCE_HELDOUT_AXIS PASS t,y,x\n";return 0;
 }
 need(argc==8,"Q4 model projector RGB height width template output-prefix required");
 int h=std::stoi(argv[4]),w=std::stoi(argv[5]);need(h>0&&w>0&&h<=4096&&w<=4096,"bounded RGB");
 std::ifstream rf(argv[3],std::ios::binary);std::vector<unsigned char>rgb(size_t(h)*w*3);rf.read(reinterpret_cast<char*>(rgb.data()),rgb.size());need(size_t(rf.gcount())==rgb.size()&&rf.peek()==EOF,"exact RGB bytes");
 std::ifstream tf(argv[6]);std::string templ((std::istreambuf_iterator<char>(tf)),{});need(bool(tf)&&templ.size()<4096,"bounded frozen template");
 ggml_backend_load_all();llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.vocab_only=true;
 std::unique_ptr<llama_model,decltype(&llama_model_free)>model(llama_model_load_from_file(argv[1],mp),llama_model_free);need(bool(model),"vocabulary-only source model");
 auto cp=mtmd_context_params_default();cp.use_gpu=false;cp.device=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);cp.warmup=false;cp.n_threads=4;cp.image_min_tokens=64;cp.image_max_tokens=256;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 std::unique_ptr<mtmd_context,decltype(&mtmd_free)>ctx(mtmd_init_from_file(argv[2],model.get(),cp),mtmd_free);need(bool(ctx)&&mtmd_decode_use_mrope(ctx.get()),"source image/MRoPE metadata");
 const std::string marker="<__media__>";auto at=templ.find(marker);need(at!=std::string::npos&&templ.find(marker,at+marker.size())==std::string::npos,"one frozen media marker");templ.replace(at,marker.size(),mtmd_default_marker());
 std::unique_ptr<mtmd_bitmap,decltype(&mtmd_bitmap_free)>bitmap(mtmd_bitmap_init(w,h,rgb.data()),mtmd_bitmap_free);need(bool(bitmap),"source bitmap");
 std::unique_ptr<mtmd_input_chunks,decltype(&mtmd_input_chunks_free)>chunks(mtmd_input_chunks_init(),mtmd_input_chunks_free);need(bool(chunks),"source input chunks");
 mtmd_input_text text{templ.data(),templ.size(),false,true};const mtmd_bitmap*images[]={bitmap.get()};need(mtmd_tokenize(ctx.get(),chunks.get(),&text,images,1)==0,"source tokenize/preprocess only");
 const auto*vocab=llama_model_get_vocab(model.get());llama_token pad[4];need(llama_tokenize(vocab,"<|image_pad|>",13,pad,4,false,true)==1,"one image-pad token");
 std::vector<llama_token>ids;std::vector<int>modalities;std::vector<std::array<uint32_t,3>>positions;llama_pos past=0;size_t image_count=0,image_tokens=0;
 for(size_t i=0;i<mtmd_input_chunks_size(chunks.get());++i){const auto*chunk=mtmd_input_chunks_get(chunks.get(),i);const auto kind=mtmd_input_chunk_get_type(chunk);
  if(kind==MTMD_INPUT_CHUNK_TYPE_TEXT){size_t n=0;const auto*t=mtmd_input_chunk_get_tokens_text(chunk,&n);need(t&&n<512,"bounded text chunk");for(size_t j=0;j<n;++j){ids.push_back(t[j]);modalities.push_back(0);positions.push_back({uint32_t(past+j),uint32_t(past+j),uint32_t(past+j)});}past+=n;}
  else{need(kind==MTMD_INPUT_CHUNK_TYPE_IMAGE&&++image_count==1,"one still image only");const auto*im=mtmd_input_chunk_get_tokens_image(chunk);const size_t n=mtmd_input_chunk_get_n_tokens(chunk);need(im&&n>0&&n<=256,"bounded merged image tokens");std::vector<mtmd_decoder_pos>pos(n);mtmd_helper_image_get_decoder_pos(im,past,pos.data());for(const auto&p:pos){need(p.z==0,"reserved source image axis");ids.push_back(pad[0]);modalities.push_back(1);positions.push_back(source_axes(p));}image_tokens=n;past+=mtmd_input_chunk_get_n_pos(chunk);}
 }
 need(image_count==1&&ids.size()<=512&&ids.size()+1536<=2048,"frozen context budget before forward");const std::string out=argv[7];
 for(auto item:{std::pair<const char*,const std::vector<llama_token>*>{"prompt",&ids}}){std::ofstream f(out+"."+item.first+".ids");for(auto t:*item.second)f<<t<<'\n';need(bool(f),"prompt output");}
 std::ofstream mods(out+".modalities.ids");for(auto x:modalities)mods<<x<<'\n';need(bool(mods),"modality output");
 std::vector<float>axes(3*ids.size());for(size_t i=0;i<ids.size();++i)for(int a=0;a<3;++a)axes[size_t(a)*ids.size()+i]=positions[i][a];std::ofstream pf(out+".positions.f32",std::ios::binary);pf.write(reinterpret_cast<char*>(axes.data()),axes.size()*4);need(bool(pf),"source positions output");
 std::ofstream meta(out+".json");meta<<"{\"prompt_tokens\":"<<ids.size()<<",\"image_tokens\":"<<image_tokens<<",\"model_forward_calls\":0,\"vocab_only\":true,\"warmup\":false}\n";need(bool(meta),"preflight metadata");std::cout<<"SOURCE_HELDOUT_PREFLIGHT PASS forwards=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
