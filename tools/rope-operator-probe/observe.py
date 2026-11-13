"""Generate an arithmetic-preserving, test-only native attention observer."""
from pathlib import Path
import hashlib
ENV='CRISPEMBED_TEST_NATIVE_ROPE_CAPTURE_DIR'
ANCHOR='    auto & cache = output.cache;'
HOOK=r'''
    if (const char *dir=std::getenv("CRISPEMBED_TEST_NATIVE_ROPE_CAPTURE_DIR")) {
        auto append=[&](const char*name,const std::vector<float>&values){
            std::ofstream file(std::string(dir)+"/"+name,std::ios::binary|std::ios::app);
            file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(float));
            if(!file)throw std::runtime_error("native RoPE observer write");
        };
        append("queries.actual.f32",queries);append("keys.actual.f32",keys);
    }
'''

def generate(text):
 if text.count(ANCHOR)!=1 or text.count('auto queries = output.q_norm, keys = output.k_norm;')!=1:
  raise ValueError('exact native post-rotation boundary')
 return '#include <fstream>\n#include <cstdlib>\n#include <stdexcept>\n#include <string>\n'+text.replace(ANCHOR,HOOK+'\n'+ANCHOR)
