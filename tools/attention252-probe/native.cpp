// Diagnostic only: actual native input -> duplicate unchanged block -> strict main-trace closure.
#include "core/clean_exit.h"
#include "lighton4b_attention_block.h"
#include "lighton4b_source_prefill_chunks.h"
#include "qwen35_weight_cache.h"
#include <map>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
namespace fs = std::filesystem;
static void require(bool ok, const std::string & s) { if (!ok) throw std::runtime_error(s); }
static void read(const fs::path & p, void * data, size_t n) {
    std::ifstream f(p, std::ios::binary);
    require(bool(f.read(static_cast<char *>(data), n)) && f.peek() == EOF, "read: " + p.string());
}
static void write(const fs::path & p, const std::vector<float> & x) {
    for (float v : x) require(std::isfinite(v), "nonfinite: " + p.string());
    std::ofstream f(p, std::ios::binary);
    require(bool(f.write(reinterpret_cast<const char *>(x.data()), x.size() * sizeof(float))), "write: " + p.string());
}
static int run(int argc, char ** argv) try {
    require(argc == 4, "usage: native WEIGHTS CAPTURE_CASE OUTPUT");
    const fs::path weights = argv[1], capture = fs::path(argv[2]), out = argv[3];
    fs::create_directories(out);
    // All borrowed tensors together are <65MiB. No model/head or whole-model F32 cache.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({80ull << 20, nullptr, false}), ggml_free);
    require(bool(ctx), "context");
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    std::vector<buffer_ptr> storage;
    auto tensor = [&](const char * name, ggml_type type, int width, int rows = 1) {
        auto * t = ggml_new_tensor_2d(ctx.get(), type, width, rows);
        read(weights / (std::string(name) + ".bin"), t->data, ggml_nbytes(t));
        // Real loader tensors are backend-backed; tensor_floats intentionally
        // uses backend_tensor_get. Preserve that contract for this borrowed
        // diagnostic tensor rather than weakening production access checks.
        auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), ggml_nbytes(t));
        require(buffer != nullptr, "weight backend buffer");
        storage.emplace_back(buffer, ggml_backend_buffer_free);
        void * base = ggml_backend_buffer_get_base(buffer);
        require(base != nullptr && ggml_backend_buffer_is_host(buffer), "host weight buffer");
        std::memcpy(base, t->data, ggml_nbytes(t)); t->data = base; t->buffer = buffer;
        return t;
    };
    decoder_layer_weights w{}; w.full_attention = true; w.upstream_physical = true; w.attention.upstream_physical = true;
    w.input_norm = tensor("blk.3.attn_norm.weight", GGML_TYPE_F32, 2560);
    w.post_norm = tensor("blk.3.post_attention_norm.weight", GGML_TYPE_F32, 2560);
    w.gate = tensor("blk.3.ffn_gate.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.up = tensor("blk.3.ffn_up.weight", GGML_TYPE_Q4_K, 2560, 9216);
    w.down = tensor("blk.3.ffn_down.weight", GGML_TYPE_Q6_K, 9216, 2560);
    w.attention.q = tensor("blk.3.attn_q.weight", GGML_TYPE_Q4_K, 2560, 8192);
    w.attention.k = tensor("blk.3.attn_k.weight", GGML_TYPE_Q4_K, 2560, 1024);
    w.attention.v = tensor("blk.3.attn_v.weight", GGML_TYPE_Q6_K, 2560, 1024);
    w.attention.out = tensor("blk.3.attn_output.weight", GGML_TYPE_Q4_K, 4096, 2560);
    w.attention.q_norm = tensor("blk.3.attn_q_norm.weight", GGML_TYPE_F32, 256);
    w.attention.k_norm = tensor("blk.3.attn_k_norm.weight", GGML_TYPE_F32, 256);

    converted_weight_cache cache_owner(0);
    require(cache_owner.enable_physical_repack(true), "bounded physical Q4 cache");
    w.physical_q4_repack = w.attention.physical_q4_repack = true;
    w.physical_weights_cache = &cache_owner;
    std::vector<float> input(266*2560), expected(input.size()), allpos(3*266);
    read(capture/"prefill.layer.2.f32",input.data(),input.size()*4);
    read(capture/"prefill.layer.3.f32",expected.data(),expected.size()*4);
    read(capture/"prefill.positions.f32",allpos.data(),allpos.size()*4);
    std::vector<int> modalities(266,0);std::fill(modalities.begin()+4,modalities.begin()+256,1);
    std::vector<lighton4b_prefill_chunk> plan;
    require(lighton4b_source_prefill_plan(modalities,plan),"actual bounded source plan");
    std::map<std::string,std::vector<float>> whole;
    attention_cache previous;
    for(const auto &part:plan){
        const int offset=part.offset,n=part.tokens;std::vector<float> positions;
        require(lighton4b_prefill_chunk_positions(allpos.data(),266,part,positions),"same positions");
        lighton4b_attention_block_outputs b;
        require(lighton4b_attention_block_cpu(w,input.data()+size_t(offset)*2560,positions.data(),n,4,
                    offset?&previous:nullptr,b),"actual native block");
        require(!std::memcmp(b.hidden.data(),expected.data()+size_t(offset)*2560,b.hidden.size()*4),"native frozen layer3 byte closure");
        auto append=[&](const char*name,const std::vector<float>&x){auto&v=whole[name];v.insert(v.end(),x.begin(),x.end());};
        append("input_norm",b.input_norm);append("q_full",b.q);append("k_raw",b.k);append("v_raw",b.v);
        append("q_norm",b.q_norm);append("k_norm",b.k_norm);append("pooled",b.pooled);
        append("attention",b.attention);append("attention_residual",b.attention_residual);append("post_norm",b.post_norm);
        append("ffn_gate",b.gate);append("ffn_up",b.up);append("activation",b.activation);append("down",b.down);append("hidden",b.hidden);
        auto rotate=[&](std::vector<float>x,int heads){
            for(int t=0;t<n;++t)for(int h=0;h<heads;++h)for(int i=0;i<32;++i){
                int axis=0;if(i%3==1&&i<33)axis=1;if(i%3==2&&i<30)axis=2;
                const float frequency=1.f/std::pow(10000000.f,float(2*i)/64);
                const float angle=frequency*positions[axis*n+t],c=std::cos(angle),s=std::sin(angle);
                float*p=x.data()+(t*heads+h)*256;const float a=p[i],d=p[i+32];p[i]=a*c-d*s;p[i+32]=d*c+a*s;
            }return x;
        };
        append("q_rope",rotate(b.q_norm,16));append("k_rope",rotate(b.k_norm,4));
        std::vector<float> gate(size_t(n)*4096);
        for(int t=0;t<n;++t)for(int h=0;h<16;++h)std::copy_n(b.q.data()+(size_t(t)*16+h)*512+256,256,gate.data()+(size_t(t)*16+h)*256);
        append("attn_gate",gate);
        previous=std::move(b.cache);
        if(offset==0||offset+n==256){
            const auto tag=offset==0?"cache4":"cache256";
            write(out/(std::string(tag)+".keys.f32"),previous.keys);write(out/(std::string(tag)+".values.f32"),previous.values);
        }
    }
    for(const auto&kv:whole)write(out/(kv.first+".f32"),kv.second);
    require(cache_owner.bytes()==0&&cache_owner.physical_repack_entries()==5,"quantized-only5entries");
    std::cout<<"ACTUAL_NATIVE_LAYER3_ALL266_BYTE_CLOSURE PASS quantized_only=1 tiles="<<plan.size()<<"\n";
    return 0;
} catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}

int main(int argc,char**argv){core_util::clean_exit(run(argc,argv));}
