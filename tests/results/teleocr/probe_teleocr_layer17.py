from pathlib import Path
import json, sys, importlib.util
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from safetensors import safe_open
import gguf

root=Path('/workspace/scratch/960329398638')
out=root/('validation/'+(sys.argv[1] if len(sys.argv)>1 else 'layer17-diagnosis'));out.mkdir(exist_ok=True)
torch.set_num_threads(2); torch.set_num_interop_threads(1)
spec=importlib.util.spec_from_file_location('modeling_naviocr_probe',root/'teleocr-hf/modeling_naviocr.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cfg=m.Qwen2_5_VLVisionConfig(**json.loads((root/'teleocr-hf/config.json').read_text())['vision_config']);cfg._attn_implementation='sdpa'
block=m.Qwen2_5_VLVisionBlock(cfg).eval()
prefix='visual.blocks.17.'
with safe_open(root/'teleocr-hf/model.safetensors',framework='pt') as f:
    state={key[len(prefix):]:f.get_tensor(key).float() for key in f.keys() if key.startswith(prefix)}
block.load_state_dict(state)
# Invoke the licensed source's geometry methods without constructing a full encoder.
v=SimpleNamespace(spatial_merge_size=2, spatial_merge_unit=4, window_size=112,patch_size=14,rotary_pos_emb=m.Qwen2_5_VisionRotaryEmbedding(40))
v.get_window_index=MethodType(m.Qwen2_5_VisionTransformerPretrainedModel.get_window_index,v)
v.rot_pos_emb=MethodType(m.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb,v)
grid=torch.tensor([[1,14,68]],dtype=torch.int64)
order,cu=v.get_window_index(grid);cu=torch.unique_consecutive(torch.tensor(cu,dtype=torch.int32))
r=v.rot_pos_emb(grid);r=r.reshape(-1,4,40)[order].reshape(-1,40);emb=torch.cat([r,r],dim=-1);positions=(emb.cos(),emb.sin())
x=torch.from_numpy(np.load(root/'validation/reference/text-stages/vis_layer_16.npy'))
if len(sys.argv)>2:
    x=torch.from_numpy(np.fromfile(root/sys.argv[2],np.float32).reshape(x.shape))
expected=torch.from_numpy(np.load(root/'validation/reference/text-stages/vis_layer_17.npy'))

def put(name,value):
    np.save(out/(name+'.npy'),value.detach().float().numpy())

with torch.inference_mode():
    post=x+block.attn(block.norm1(x),cu_seqlens=cu,position_embeddings=positions)
    y=block(post,cu_seqlens=cu,position_embeddings=positions) if False else post+block.mlp(block.norm2(post))
    print('original block reconstructed vs frozen',float((y-expected).abs().max()))
    put('post_attention',post);put('original_output',y)
    norm=block.norm2(post);gate=block.mlp.gate_proj(norm);up=block.mlp.up_proj(norm);silu=block.mlp.act_fn(gate);product=silu*up;down=block.mlp.down_proj(product)
    arrays={'input':post,'norm_w':block.norm2.weight,'gate_w':block.mlp.gate_proj.weight,'gate_b':block.mlp.gate_proj.bias,'up_w':block.mlp.up_proj.weight,'up_b':block.mlp.up_proj.bias,'down_w':block.mlp.down_proj.weight,'down_b':block.mlp.down_proj.bias,'norm':norm,'gate':gate,'up':up,'silu':silu,'product':product,'down':down,'output':y}
    writer=gguf.GGUFWriter(out/'probe.gguf','teleocr_probe')
    for name,a in arrays.items():
        a=a.detach().numpy();writer.add_tensor(name,a);np.save(out/(name+'.npy'),a)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    z=post.double(); n=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6)*block.norm2.weight.double()
    g=torch.nn.functional.linear(n,block.mlp.gate_proj.weight.double(),block.mlp.gate_proj.bias.double())
    u=torch.nn.functional.linear(n,block.mlp.up_proj.weight.double(),block.mlp.up_proj.bias.double())
    p=torch.nn.functional.silu(g)*u
    d=torch.nn.functional.linear(p,block.mlp.down_proj.weight.double(),block.mlp.down_proj.bias.double())
    yf=z+d
    put('float64_output',yf)
    metrics={}
    for name,b,a in [('norm',n,norm),('gate',g,gate),('up',u,up),('product',p,product),('down',d,down),('output',yf,y)]:
        diff=(a.double()-b).abs();i=np.unravel_index(int(diff.argmax()),diff.shape);ref=float(a[i]);val=float(b[i]);metric={'max_abs':float(diff.max()),'index':[int(j) for j in i],'float32':ref,'float64':val,'max_magnitude':float(a.abs().max()),'float32_spacing':float(np.spacing(np.float32(abs(ref))))};metrics[name]=metric;print(name,metric)
    (out/'original-f32-vs-f64.json').write_text(json.dumps(metrics,indent=2)+'\n')
