#include "lighton4b_native_context.h"
#include <iostream>
using namespace qwen35_ocr;
int main() {
    ggml_tensor q8{},q4{},f16{};q8.type=GGML_TYPE_Q8_0;q4.type=GGML_TYPE_Q4_K;f16.type=GGML_TYPE_F16;
    decoder_weights w;w.embedding=w.head=&q8;
    for(int i=0;i<32;++i) {
        decoder_layer_weights l;l.full_attention=i%4==3;l.gate=l.up=l.down=&q4;
        if(l.full_attention)l.attention.q=l.attention.k=l.attention.v=l.attention.out=&q8;
        else l.linear.qkv=l.linear.z=l.linear.a=l.linear.b=l.linear.out=&q8;
        w.layers.push_back(l);
    }
    int checks=0;auto check=[&](bool ok){++checks;if(!ok){std::cerr<<"mixed policy control "<<checks<<" failed\n";std::exit(1);}};
    check(lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::attention_q8_ffn_q4));
    check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::q8_0));
    check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::q4_k));
    auto uniform8=w;
    for(auto &l:uniform8.layers)l.gate=l.up=l.down=&q8;
    check(lighton4b_quant_recipe_matches(uniform8,lighton4b_quant_recipe::q8_0));
    check(!lighton4b_quant_recipe_matches(uniform8,lighton4b_quant_recipe::attention_q8_ffn_q4));
    auto uniform4=w;
    for(auto &l:uniform4.layers) {
        if(l.full_attention)l.attention.q=l.attention.k=l.attention.v=l.attention.out=&q4;
        else l.linear.qkv=l.linear.z=l.linear.a=l.linear.b=l.linear.out=&q4;
    }
    check(lighton4b_quant_recipe_matches(uniform4,lighton4b_quant_recipe::q4_k));
    check(!lighton4b_quant_recipe_matches(uniform4,lighton4b_quant_recipe::attention_q8_ffn_q4));
    check(!lighton4b_quant_recipe_matches(w,static_cast<lighton4b_quant_recipe>(99)));
    for(auto &l:w.layers) {
        auto *saved=l.gate;l.gate=&q8;check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::attention_q8_ffn_q4));l.gate=saved;
        auto * &attn=l.full_attention?l.attention.q:l.linear.qkv;saved=attn;attn=&q4;
        check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::attention_q8_ffn_q4));attn=saved;
    }
    w.embedding=w.head=&f16;check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::attention_q8_ffn_q4));
    w.embedding=w.head=&q8;w.layers.pop_back();check(!lighton4b_quant_recipe_matches(w,lighton4b_quant_recipe::attention_q8_ffn_q4));
    std::cout<<"MIXED_POLICY controls="<<checks<<" passed=1 payload_bytes=0 quality_qualified=0\n";
}
