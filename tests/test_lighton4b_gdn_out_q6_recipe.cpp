#include "lighton4b_native_context.h"
#include "core/clean_exit.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
using namespace qwen35_ocr;
static void gate(const char * n,const char * v) {
#ifdef _WIN32
    _putenv_s(n,v?v:"");
#else
    if(v)setenv(n,v,1);else unsetenv(n);
#endif
}
static int run(int argc,char ** argv) {
    if(argc!=2)return 2;
    int checks=0;auto check=[&](bool ok){if(!ok){std::cerr<<"Q6out control "<<checks<<" failed\n";std::exit(1);}++checks;};
    for(auto * flag:{"CRISPEMBED_QWEN35_OCR","CRISPEMBED_QWEN35_QUANT","CRISPEMBED_LIGHTON4B_QUANT","CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT","CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER","CRISPEMBED_QWEN35_GDN_GROUPED_HEADS","CRISPEMBED_QWEN35_STREAM_PROJECTION"})gate(flag,"1");
    auto * headers=ggml_init({426*ggml_tensor_overhead()+4096,nullptr,true});core_gguf::tensor_map tensors;std::ifstream file(argv[1]);std::string name;int kind;int64_t ne[4];
    while(file>>name>>kind>>ne[0]>>ne[1]>>ne[2]>>ne[3]){auto * t=ggml_new_tensor(headers,ggml_type(kind),4,ne);check(t&&!t->data&&tensors.emplace(name,t).second);}check(file.eof()&&tensors.size()==426);
    gate("CRISPEMBED_LIGHTON4B_Q6_K","1");gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6","1");lighton4b_text_owner owner;check(map_lighton4b_text(tensors,owner));lighton4b_quant_recipe chosen;
    check(lighton4b_select_recipe(owner.decoder,lighton4b_gdn_out_q6_recipe,chosen)&&chosen==lighton4b_quant_recipe::gdn_out_q6);
    check(!lighton4b_quant_recipe_matches(owner.decoder,lighton4b_quant_recipe::q4km_q6_q8head));check(owner.decoder.head==owner.decoder.embedding&&owner.decoder.head->type==GGML_TYPE_Q8_0);
    for(auto & row:tensors)if(row.second->type!=GGML_TYPE_F32){auto old=row.second->type;row.second->type=old==GGML_TYPE_Q6_K?GGML_TYPE_Q4_K:GGML_TYPE_Q6_K;check(!lighton4b_quant_recipe_matches(owner.decoder,lighton4b_quant_recipe::gdn_out_q6));row.second->type=old;}
    for(auto * flag:{"CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4","CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8"}){gate(flag,"1");check(!lighton4b_select_recipe(owner.decoder,lighton4b_gdn_out_q6_recipe,chosen));gate(flag,"0");}
    for(auto * value:{static_cast<const char *>(nullptr),"","0","false"}){gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6",value);check(!lighton4b_select_recipe(owner.decoder,lighton4b_gdn_out_q6_recipe,chosen));}
    gate("CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q6","1");gate("CRISPEMBED_LIGHTON4B_Q6_K","0");check(!lighton4b_select_recipe(owner.decoder,lighton4b_gdn_out_q6_recipe,chosen));gate("CRISPEMBED_LIGHTON4B_Q6_K","1");
    auto * meta=gguf_init_empty();gguf_set_val_str(meta,"qwen35_ocr.mixed_quant_recipe",lighton4b_gdn_out_q6_recipe);gguf_set_val_u32(meta,"general.file_type",15);gguf_set_val_str(meta,"qwen35_ocr.mixed_quant_base_sha256","f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91");gguf_set_val_str(meta,"qwen35_ocr.q6_source_checkpoint_sha256","6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67");gguf_set_val_str(meta,"qwen35_ocr.q6_names_sha256","2b6d3e1109215160bbbd9760b1d11fc88bd6a34c70d9fceffdcbe93e68259845");gguf_set_val_u32(meta,"qwen35_ocr.mixed_quant_selected_count",24);gguf_set_val_u32(meta,"qwen35_ocr.mixed_quant_payload_delta",64880640);check(lighton4b_gdn_out_q6_metadata_matches(meta));
    for(auto item:{std::pair<const char *,uint32_t>{"general.file_type",14},{"qwen35_ocr.mixed_quant_selected_count",32},{"qwen35_ocr.mixed_quant_payload_delta",125829120}}){gguf_set_val_u32(meta,item.first,item.second);check(!lighton4b_gdn_out_q6_metadata_matches(meta));gguf_set_val_u32(meta,item.first,!std::strcmp(item.first,"general.file_type")?15:!std::strcmp(item.first,"qwen35_ocr.mixed_quant_selected_count")?24:64880640);}
    gguf_set_val_str(meta,"qwen35_ocr.q6_source_checkpoint_sha256","wrong");check(!lighton4b_gdn_out_q6_metadata_matches(meta));gguf_free(meta);ggml_free(headers);
    // Official16:32 grouping with small width8; Q6 out columns are256.
    auto * ctx=ggml_init({65536,nullptr,true});auto * backend=ggml_backend_cpu_init();linear_weights w;
    w.qkv=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,512);w.z=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,256);w.a=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,32);w.b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,32);w.out=ggml_new_tensor_2d(ctx,GGML_TYPE_Q6_K,256,256);auto * outref=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,256);w.conv=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4,1,512);w.a_log=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32);w.dt_bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32);w.norm=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,8);auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer!=nullptr);
    std::vector<float> original(65536),dequant(65536),hidden(512);for(size_t i=0;i<original.size();++i)original[i]=float(int(i%257)-128)/128;std::vector<uint8_t> raw(ggml_nbytes(w.out));check(ggml_quantize_chunk(GGML_TYPE_Q6_K,original.data(),raw.data(),0,256,256,nullptr)==raw.size());ggml_get_type_traits(GGML_TYPE_Q6_K)->to_float(raw.data(),dequant.data(),65536);ggml_backend_tensor_set(w.out,raw.data(),0,raw.size());ggml_backend_tensor_set(outref,dequant.data(),0,dequant.size()*4);
    for(auto * t:{w.qkv,w.z,w.a,w.b,w.conv,w.a_log,w.dt_bias,w.norm}){std::vector<float> v(size_t(ggml_nelements(t)));for(size_t i=0;i<v.size();++i)v[i]=t==w.norm?1:t==w.a_log?-1:float(int(i%7)-3)/1024;ggml_backend_tensor_set(t,v.data(),0,v.size()*4);}for(size_t i=0;i<hidden.size();++i)hidden[i]=float(int(i%37)-18)/65536;
    auto ref=w;ref.out=outref;projected_linear_outputs a,b;check(linear_grouped_cpu(w,16,hidden.data(),1,2,2,nullptr,nullptr,a)&&linear_grouped_cpu(ref,16,hidden.data(),1,2,2,nullptr,nullptr,b));check(a.qkv==b.qkv&&a.attention==b.attention&&a.core.state==b.core.state);
    const auto state=a.core.state,conv=a.core.conv_history;projected_linear_outputs d,e;check(linear_grouped_cpu(w,16,hidden.data()+256,1,1,2,conv.data(),state.data(),d)&&linear_grouped_cpu(ref,16,hidden.data()+256,1,1,2,b.core.conv_history.data(),b.core.state.data(),e));check(d.attention==e.attention&&d.core.state==e.core.state&&a.core.state==state&&a.core.conv_history==conv);std::vector<uint8_t> after(raw.size());ggml_backend_tensor_get(w.out,after.data(),0,after.size());check(after==raw);
    gate("CRISPEMBED_LIGHTON4B_Q6_K","0");check(!linear_grouped_cpu(w,16,hidden.data(),1,2,2,nullptr,nullptr,d)&&d.core.state.empty());ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);
    std::cout<<"GDN_OUT_Q6 passed=1 checks="<<checks<<" q6_matrices=24 q8_tied_head=1 grouped_out_prefill_decode=1 storage_unchanged=1 model_payload=0\n";return 0;
}
int main(int argc,char ** argv){core_util::clean_exit(run(argc,argv));}
