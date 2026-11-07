"""Expose only pinned upstream img_tool resize for a reference executable."""
import hashlib
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('projector_patch',Path(__file__).resolve().parents[1]/'projector-probe/source_patch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SUFFIX='''
// Reference-only bridge; no graph, trait or production arithmetic edits.
extern "C" bool crisp_reference_pillow22(const uint8_t* rgb,int h,int w,int oh,int ow,uint8_t* out) {
 if(!rgb||!out||h<1||w<1||oh<1||ow<1||h>4096||w>4096||oh>4096||ow>4096)return false;
 clip_image_u8 src,dst;src.set_size({w,h},false);
 src.cpy_buf(std::vector<uint8_t>(rgb,rgb+size_t(h)*w*3));
 img_tool::resize(src,dst,{ow,oh},RESIZE_ALGO_BICUBIC,PAD_NONE);
 const auto& values=dst.get_ro_buf();if(values.size()!=size_t(oh)*ow*3)return false;
 std::copy(values.begin(),values.end(),out);return true;
}
'''
def patch(root):
    path=Path(root)/'tools/mtmd/mtmd-image.cpp';raw=path.read_bytes()
    expected=module.PINNED['tools/mtmd/mtmd-image.cpp']
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('immutable resize source drift')
    path.write_text(raw.decode()+SUFFIX)
    return dict(before_sha256=expected,after_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
