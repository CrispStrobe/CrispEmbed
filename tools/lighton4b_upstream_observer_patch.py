#!/usr/bin/env python3
"""Reference-only after-compute capture hooks in one pinned external upstream tree.

No eval callback or graph rewrite: observer OFF/ON must retain all prefix IDs and
probability records exactly before native comparison. Never linked in our product.
"""
import hashlib
from pathlib import Path
ENV='CRISPEMBED_TEST_UPSTREAM_OBSERVER_DIR'
SOURCE='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
# Filled from immutable primary-source files, not fixture outputs.
SHA={'tools/mtmd/mtmd-helper.cpp': 'fd2b9fbb742c1d1d102cfacb70ab0dedc903603a22dc5cf504c77b389ce413d9', 'common/sampling.cpp': 'e88c45dfdc58ef2e86b5519aac68c83ced333952ade7d4bd8fca5f1ce5bbea02'}
def patch(root):
 root=Path(root);result={}
 for name,expected in SHA.items():
  p=root/name;actual=hashlib.sha256(p.read_bytes()).hexdigest()
  if actual!=expected:raise ValueError('pinned upstream observer source differs: '+name)
  result[name]={'original_sha256':actual}
 helper=root/'tools/mtmd/mtmd-helper.cpp';s=helper.read_text();needle='    int n_mmproj_embd = llama_model_n_embd_inp(model);';assert s.count(needle)==1
 insert=r'''
        // Test-only observer at the shared server/CLI decode boundary.
        // Encoding has finished; consume and capture the same unchanged pointer.
        if (const char * dir = std::getenv("CRISPEMBED_TEST_UPSTREAM_OBSERVER_DIR")) {
            if (chunk_type != MTMD_INPUT_CHUNK_TYPE_IMAGE) throw std::runtime_error("image-only source probe");
            const float * embd = encoded_embd;
            const auto tokens = mtmd_input_chunk_get_n_tokens(chunk);
            const auto dim = llama_model_n_embd_inp(llama_get_model(lctx));
            if (tokens < 1 || tokens > 256 || dim != 2560 || !embd) throw std::runtime_error("bounded source feature geometry");
            static int image_index = 0;
            const auto stem = std::string(dir) + "/source.vision." + std::to_string(image_index++);
            std::ofstream f(stem + ".f32", std::ios::binary);
            f.write(reinterpret_cast<const char *>(embd), size_t(tokens) * dim * sizeof(float));
            const auto * image = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image || !f) throw std::runtime_error("source feature capture failed");
            std::vector<mtmd_decoder_pos> positions(tokens);
            mtmd_helper_image_get_decoder_pos(image, n_past, positions.data());
            static_assert(sizeof(mtmd_decoder_pos) == 4*sizeof(uint32_t), "position capture layout");
            std::ofstream pos(stem + ".positions.u32", std::ios::binary);
            pos.write(reinterpret_cast<const char *>(positions.data()), positions.size()*sizeof(mtmd_decoder_pos));
            std::ofstream meta(stem + ".json");
            meta << "{\"tokens\":" << tokens << ",\"embedding\":" << dim << ",\"past\":" << n_past
                 << ",\"position_count\":" << mtmd_input_chunk_get_n_pos(chunk) << "}\n";
            if (!pos || !meta) throw std::runtime_error("source position capture failed");
        }
'''
 s='#include <fstream>\n#include <cstdlib>\n#include <stdexcept>\n'+s.replace(needle,needle+insert,1);helper.write_text(s)
 sampling=root/'common/sampling.cpp';s=sampling.read_text();needle='llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {\n    llama_synchronize(ctx);';assert s.count(needle)==1
 insert=r'''
    // Test-only raw F32 logits. This adds no graph callback or sampler transform.
    if (const char * dir = std::getenv("CRISPEMBED_TEST_UPSTREAM_OBSERVER_DIR")) {
        const auto n = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
        const float * values = llama_get_logits_ith(ctx, idx);
        if (n != 248320 || !values) throw std::runtime_error("source logits unavailable");
        static int sample_index = 0;
        std::ofstream f(std::string(dir) + "/source.logits." + std::to_string(sample_index++) + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(values), size_t(n)*sizeof(float));
        if (!f) throw std::runtime_error("source logits capture failed");
    }
'''
 s='#include <fstream>\n#include <cstdlib>\n#include <stdexcept>\n'+s.replace(needle,needle+insert,1);sampling.write_text(s)
 for name in SHA:result[name]['patched_sha256']=hashlib.sha256((root/name).read_bytes()).hexdigest()
 return result
