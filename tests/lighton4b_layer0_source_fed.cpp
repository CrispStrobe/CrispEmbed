// Diagnostic only: replay checkpoint layer0 on an authenticated retained input.
#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iterator>
#include <map>
static int run(int argc, char ** argv) {
    if (argc != 4) return 2;
    // RNE tie cases distinguish BF16 from truncation and FP16 conversion.
    if (qwen35_ocr::lighton4b_bf16_round(1.00390625f)!=1.0f ||
        qwen35_ocr::lighton4b_bf16_round(1.01171875f)!=1.015625f ||
        qwen35_ocr::lighton4b_bf16_round(-1.00390625f)!=-1.0f ||
        qwen35_ocr::lighton4b_bf16_round(0.0f)!=0.0f) return 6;
    auto backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "layer0-source-fed", weights)) return 2;
    auto get = [&](const char * name) { return core_gguf::try_get(weights.tensors, name); };
    qwen35_ocr::decoder_layer_weights w;
    w.input_norm = get("input_layernorm.weight"); w.post_norm = get("post_attention_layernorm.weight");
    w.gate = get("mlp.gate_proj.weight"); w.up = get("mlp.up_proj.weight"); w.down = get("mlp.down_proj.weight");
    w.linear = {get("linear_attn.in_proj_qkv.weight"), get("linear_attn.in_proj_z.weight"),
        get("linear_attn.in_proj_a.weight"), get("linear_attn.in_proj_b.weight"), get("linear_attn.out_proj.weight"),
        get("linear_attn.conv1d.weight"), get("linear_attn.A_log"), get("linear_attn.dt_bias"), get("linear_attn.norm.weight")};
    std::ifstream stream(argv[2], std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != 154 * 2560 * 4) return 2;
    std::vector<float> input(154 * 2560); stream.seekg(0); stream.read(reinterpret_cast<char *>(input.data()), input.size()*4);
    if (!stream) return 2;
    qwen35_ocr::lighton4b_linear_cache cache;
    std::map<std::string, std::vector<float>> stages;
    for (int offset = 0; offset < 154; offset += 16) {
        qwen35_ocr::lighton4b_linear_block_outputs b;
        if (!qwen35_ocr::lighton4b_linear_block_cpu(w,input.data()+offset*2560,std::min(16,154-offset),2,
                offset ? &cache : nullptr,b)) return 3;
        auto append = [&](const char * name,const std::vector<float> & v) { auto & target=stages[name]; target.insert(target.end(),v.begin(),v.end()); };
        qwen35_ocr::projected_linear_outputs linear;
        const bool replay=qwen35_ocr::lighton4b_bf16_linear_enabled() ?
            qwen35_ocr::lighton4b_bf16_grouped_cpu(w.linear,b.input_norm.data(),std::min(16,154-offset),2,
                offset ? cache.conv.data() : nullptr,offset ? cache.recurrent.data() : nullptr,linear) :
            qwen35_ocr::linear_grouped_cpu(w.linear,16,b.input_norm.data(),1,std::min(16,154-offset),2,
                offset ? cache.conv.data() : nullptr,offset ? cache.recurrent.data() : nullptr,linear);
        if (!replay ||
                linear.attention != b.attention || linear.core.state != b.cache.recurrent ||
                linear.core.conv_history != b.cache.conv) return 5;
        append("input_norm",b.input_norm); append("qkv",linear.qkv); append("z",linear.z);
        append("a",linear.a); append("b",linear.b); append("gdn_norm",linear.core.values);
        append("attention",b.attention); append("attention_residual",b.attention_residual);
        append("post_norm",b.post_norm); append("gate",b.gate); append("up",b.up); append("activation",b.activation);
        append("down",b.down); append("hidden",b.hidden); cache=std::move(b.cache);
    }
    stages["conv"]=cache.conv; stages["recurrent"]=cache.recurrent;
    for (auto & item: stages) {
        std::ofstream output(std::string(argv[3])+"/"+item.first+".f32",std::ios::binary);
        output.write(reinterpret_cast<const char *>(item.second.data()),item.second.size()*4);
        if (!output) return 4;
    }
    core_gguf::free_weights(weights); ggml_backend_free(backend); return 0;
}
int main(int argc,char ** argv) { core_util::clean_exit(run(argc,argv)); }
