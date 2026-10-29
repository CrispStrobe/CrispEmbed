"""Closed disposable observer overlay; guarded checkout/old source never change."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def replace(text,before,after):
 if text.count(before)!=1:raise ValueError('Exact overlay anchor differs')
 return text.replace(before,after,1)
def apply(source):
 source=Path(source);changes={}
 def write(name,text):
  p=source/name;before=p.read_bytes();p.write_text(text);changes[name]=dict(before=hashlib.sha256(before).hexdigest(),after=hashlib.sha256(p.read_bytes()).hexdigest())
 p=source/'src/qwen35_decoder.cpp';text=p.read_text()
 # Only the observed archive defines this macro. The ordinary preprocessor path retains old arithmetic.
 text=replace(text,'#include "qwen35_decoder.h"', '#include "qwen35_decoder.h"\n#if defined(CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)\n#include "../tests/josev-ffn-imatrix/capture.h"\n#endif')
 anchor='''        std::vector<std::vector<float>> projections;
        if (!project_cpu({ layer.gate, layer.up }, normalized.data(), width, batch * tokens, threads, projections,'''
 text=replace(text,anchor,'''#if defined(CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
        if (i == 23 && !josev_ffn_imatrix_test::capture(normalized.data(), width, batch * tokens,
                layer.gate, layer.up, layer.post_norm, c.eps, batch, past)) return false;
#endif
'''+anchor);write('src/qwen35_decoder.cpp',text)
 p=source/'tests/josev-packed/probe.cpp';text=p.read_text()
 declarations='''#include "core/env_gate.h"
#if defined(CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
extern "C" int josev_ffn_imatrix_begin(const char *);
extern "C" int josev_ffn_imatrix_finish();
#endif
'''
 text=declarations+text
 anchor='''            const bool ok = context->decide(row.get("request_json").text, detailed, error);'''
 text=replace(text,anchor,'''#if defined(CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
            const bool calibration = core_env::on("CRISPEMBED_JOSEV_FFN_IMATRIX_CAPTURE");
            if (calibration) check(josev_ffn_imatrix_begin(directory.string().c_str()) == 1, "capture begin failed");
#endif
'''+anchor+'''
#if defined(CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
            if (calibration) check(ok && josev_ffn_imatrix_finish() == 1, "exactly one successful preFFN23 capture required");
#endif''');write('tests/josev-packed/probe.cpp',text)
 p=source/'tests/josev-q4k/CMakeLists.txt';text=p.read_text()
 text+='''
# Disposable test-only extension, not root build/profile policy.
target_compile_definitions(josev-q4k-observed PRIVATE CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
target_compile_definitions(josev-q4k-observed-probe PRIVATE CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)
add_executable(josev-ffn-imatrix-controls "${CMAKE_CURRENT_LIST_DIR}/../josev-ffn-imatrix/controls.cpp")
target_include_directories(josev-ffn-imatrix-controls PRIVATE "${CRISPEMBED_ROOT}/src")
target_link_libraries(josev-ffn-imatrix-controls PRIVATE ggml ggml-cpu)
''';write('tests/josev-q4k/CMakeLists.txt',text)
 return changes
