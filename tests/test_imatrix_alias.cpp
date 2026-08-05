// test_imatrix_alias.cpp — hermetic guard for the F7 imatrix naming contract
// (src/core/imatrix_alias.h). No weights, no GGUF, no network.
//
// The contract has two halves that MUST stay in agreement:
//   * qkv_merged_name(i)   — what src/crispembed.cpp names the runtime's
//                            pre-merged QKV tensor, i.e. the key the imatrix
//                            collector files statistics under;
//   * qkv_merged_alias(s)  — how tools/quantize.cpp maps an attn q/k/v weight
//                            name back to that key.
// If either drifts, the quantizer silently falls back to "no importance" for
// every BERT-family attn.{q,k,v}.weight — the exact shipped defect T19-E3
// found (arctic: 36 of 73 tensors covered, quantizer prints no error).
#include "core/imatrix_alias.h"

#include <cstdio>
#include <string>

static int g_checks = 0, g_failures = 0;

static void expect(const std::string & got, const std::string & want, const char * what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        fprintf(stderr, "FAIL %s: got \"%s\" want \"%s\"\n", what, got.c_str(), want.c_str());
    }
}

int main() {
    using core_imatrix::qkv_merged_alias;
    using core_imatrix::qkv_merged_name;

    // The two halves of the contract agree, for every layer index shape and
    // both GGUF naming schemes.
    for (int i : { 0, 1, 9, 10, 11, 23, 123 }) {
        const std::string merged = qkv_merged_name(i);
        const std::string n = std::to_string(i);
        for (const char * mid : { ".attn.q.weight", ".attn.k.weight", ".attn.v.weight" })
            expect(qkv_merged_alias("enc." + n + mid), merged, ("enc roundtrip layer " + n).c_str());
        for (const char * mid : { ".attn_q.weight", ".attn_k.weight", ".attn_v.weight" })
            expect(qkv_merged_alias("blk." + n + mid), merged, ("blk roundtrip layer " + n).c_str());
    }

    // The canonical name itself (pin the literal: the collector writes this
    // string into imatrix files, so changing it silently orphans every
    // previously collected imatrix).
    expect(qkv_merged_name(0), "enc.0.attn.qkv_merged.weight", "canonical name layer 0");
    expect(qkv_merged_name(11), "enc.11.attn.qkv_merged.weight", "canonical name layer 11");

    // Non-q/k/v tensors never alias (a wrong-provenance importance vector is
    // worse than none).
    expect(qkv_merged_alias("enc.0.attn.o.weight"), "", "attention output proj");
    expect(qkv_merged_alias("enc.0.ffn.fc1.weight"), "", "ffn");
    expect(qkv_merged_alias("enc.0.attn.q.bias"), "", "bias not weight");
    expect(qkv_merged_alias("enc.0.attn.qq.weight"), "", "tail must match exactly");
    expect(qkv_merged_alias("token_embd.weight"), "", "embedding");
    expect(qkv_merged_alias(qkv_merged_name(0)), "", "merged name is not itself q/k/v");

    // Decoder tensors never alias: the decoder path does not pre-merge, so
    // the merged vector would be wrong-provenance there (T19-E3: f2llm's
    // dec.* coverage was already correct at 56/57 and must not change).
    expect(qkv_merged_alias("dec.3.attn.q.weight"), "", "decoder q");
    expect(qkv_merged_alias("dec.0.attn_k.weight"), "", "decoder k, community tail");

    // Malformed layer heads never alias.
    expect(qkv_merged_alias("enc.x.attn.q.weight"), "", "non-numeric layer");
    expect(qkv_merged_alias("enc..attn.q.weight"), "", "empty layer");
    expect(qkv_merged_alias("enc.1a.attn.q.weight"), "", "mixed layer");
    expect(qkv_merged_alias(".attn.q.weight"), "", "no head at all");
    expect(qkv_merged_alias("attn.q.weight"), "", "headless");
    expect(qkv_merged_alias("vision.enc.0.attn.q.weight"), "", "extra prefix segment");
    expect(qkv_merged_alias(""), "", "empty string");

    printf("imatrix-alias: %d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
