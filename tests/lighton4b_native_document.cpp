#include "lighton4b_native_context.h"
#include "lighton4b_image_compose.h"
#include "lighton4b_frontend.h"
#include "core/clean_exit.h"
#include "lighton4b_profile_scope.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <chrono>
using namespace qwen35_ocr;
static int run(int argc,char ** argv) {
    lighton4b_diagnostic_profile diagnostic_profile;
    if(argc!=10)return 2; // gguf RGB H W teacher.ids out maxgen threads mode
    const int h=std::stoi(argv[3]),w=std::stoi(argv[4]),cap=std::stoi(argv[7]),threads=std::stoi(argv[8]);
    const std::string mode=argv[9],out=argv[6];
    if(h<1||w<1||h>4096||w>4096||cap<1||cap>256||threads<1||threads>16||
       (mode!="smoke"&&mode!="full"))return 2;
    auto emit=[&](const std::string &name,const std::vector<float>&v) {
        std::ofstream f(out+"/"+name+".f32",std::ios::binary);f.write((const char*)v.data(),v.size()*4);
        if(!f)throw std::runtime_error("stage write failed");
    };
    auto ids_write=[&](const std::string &name,const std::vector<int>&v) {
        std::ofstream f(out+"/"+name+".ids");for(auto id:v)f<<id<<'\n';
    };
    auto *meta=core_gguf::open_metadata(argv[1]);lighton4b::frontend_tokenizer tokenizer;
    if(!meta||!tokenizer.load(meta))return 4;core_gguf::free_metadata(meta);
    std::ifstream file(argv[2],std::ios::binary);std::vector<uint8_t>rgb((std::istreambuf_iterator<char>(file)),{});
    if(rgb.size()!=size_t(h)*w*3)return 5;
    image_inputs image;if(!lighton4b_measured(profile_event::preprocess,profile_phase::other,[&]{return lighton4b_preprocess_rgb(rgb.data(),h,w,image);}))return 6;
    emit("processor.patches",image.patches);ids_write("processor.grid",image.grid);
    std::vector<int>ids,mods;std::string rendered;
    if(!tokenizer.image_prompt("",image.grid[1]*image.grid[2]/4,ids,mods,rendered))return 7;
    ids_write("prompt",ids);ids_write("modalities",mods);std::ofstream(out+"/rendered.txt")<<rendered;
    auto context=lighton4b_native_context::open(argv[1]);if(!context)return 3;
    const auto &model=context->model();vision_outputs vision;
    if(!lighton4b_measured(profile_event::vision,profile_phase::vision,[&]{return lighton4b_image_vision_cpu(model.vision,model.vision_parameters,image,threads,vision);}))return 8;
    emit("vision.patch",vision.patch);emit("vision.position",vision.position);emit("vision.embedded",vision.embedded);
    for(size_t i=0;i<vision.hidden.size();++i)emit("vision.layer."+std::to_string(i),vision.hidden[i]);
    emit("vision.merged",vision.merged);
    std::vector<float>embedded;if(!lighton4b_measured(profile_event::embedding,profile_phase::other,[&]{return embed_cpu(model.text.decoder.embedding,ids,embedded);}))return 9;
    emit("text.embedded",embedded);lighton4b_image_prefill prefill;
    if(!lighton4b_image_prefill_cpu(embedded,mods,image.grid,vision.merged,prefill))return 10;
    // Release large vision/probe and patch arrays before decoder execution.
    vision={};image.patches.clear();embedded.clear();
    emit("prefill.input",prefill.hidden);emit("prefill.positions",prefill.positions.axes);
    lighton4b_decoder_outputs initial;
    if(!lighton4b_measured(profile_event::prefill,profile_phase::prefill,[&]{return lighton4b_prefill_cpu(model.text.decoder,prefill.hidden.data(),prefill.positions.axes.data(),int(ids.size()),threads,nullptr,initial);}))return 11;
    for(size_t i=0;i<initial.hidden.size();++i)emit("prefill.layer."+std::to_string(i),initial.hidden[i]);
    emit("prefill.norm",initial.norm);
    std::vector<float>first;
    if(!lighton4b_measured(profile_event::selection,profile_phase::prefill,[&]{return project_rows_cpu(model.text.decoder.head,initial.norm.data()+(ids.size()-1)*2560,2560,1,threads,256,first);}))return 12;
    emit("prefill.logits",first);initial.hidden.clear();initial.norm.clear();
    auto step=[&](int id,lighton4b_decoder_cache &cache,std::vector<float>&logits,const std::string&label) {
        std::vector<float>x,pos;if(!embed_cpu(model.text.decoder.embedding,{id},x)||
          !cached_positions(cache.tokens,1,prefill.positions.rope_delta,pos))return false;
        lighton4b_decoder_outputs next;
        if(!lighton4b_measured(profile_event::cached_decode,profile_phase::decode,[&]{return lighton4b_decoder_hidden_cpu(model.text.decoder,x.data(),pos.data(),1,threads,&cache,next);}))return false;
        if(!label.empty()) {
            emit(label+".input",x);emit(label+".positions",pos);
            for(size_t i=0;i<next.hidden.size();++i)emit(label+".layer."+std::to_string(i),next.hidden[i]);
            emit(label+".norm",next.norm);
        }
        if(!lighton4b_measured(profile_event::selection,profile_phase::decode,[&]{return project_rows_cpu(model.text.decoder.head,next.norm.data(),2560,1,threads,256,logits);}))return false;
        cache=std::move(next.cache);return true;
    };
    std::vector<int>teacher;int id;std::ifstream gold(argv[5]);while(gold>>id)teacher.push_back(id);
    if(teacher.empty()||teacher.size()>256)return 13;
    auto teacher_cache=initial.cache;std::vector<float>logits=first;
    const size_t teacher_count=mode=="smoke"?1:teacher.size();
    for(size_t i=0;i<teacher_count;++i) {
        emit("teacher."+std::to_string(i)+".logits",logits);
        if(i+1<teacher_count&&!step(teacher[i],teacher_cache,logits,"teacher."+std::to_string(i+1)))return 14;
    }
    teacher_cache={};auto greedy_cache=std::move(initial.cache);logits=first;std::vector<int>generated;
    for(int i=0;i<cap;++i) {
        int best=int(std::max_element(logits.begin(),logits.end())-logits.begin());generated.push_back(best);
        emit("greedy."+std::to_string(i)+".logits",logits);
        if(tokenizer.is_stop(best))break;
        if(i+1<cap&&!step(best,greedy_cache,logits,""))return 15;
    }
    ids_write("generated",generated);std::string text;if(!tokenizer.decode(generated,text))return 16;
    std::ofstream(out+"/generated.txt")<<text;
    std::ofstream summary(out+"/native-summary.json");
    summary<<"{\"prompt_tokens\":"<<ids.size()<<",\"image_tokens\":"<<image.grid[1]*image.grid[2]/4
      <<",\"teacher_positions\":"<<teacher_count<<",\"generated_tokens\":"<<generated.size()
      <<",\"eos\":"<<(tokenizer.is_stop(generated.back())?"true":"false")<<",\"cap\":"<<cap<<"}\n";
    std::cout<<"LIGHTON4B_NATIVE_DOCUMENT completed=1 teacher_positions="<<teacher_count<<" generated="<<generated.size()<<"\n";
    return 0;
}
int main(int argc,char **argv) {try {core_util::clean_exit(run(argc,argv));}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
