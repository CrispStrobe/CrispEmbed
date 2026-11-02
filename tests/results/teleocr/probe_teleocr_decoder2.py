from pathlib import Path
import json,sys,importlib.util
import numpy as np
import torch
from safetensors import safe_open
import gguf
root=Path('/workspace/scratch/960329398638');out=root/'validation/decoder2-diagnosis';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
spec=importlib.util.spec_from_file_location('modeling_naviocr_probe_decoder',root/'teleocr-hf/modeling_naviocr.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cfg=m.Qwen2_5_VLTextConfig(**json.loads((root/'teleocr-hf/config.json').read_text())['text_config']);cfg._attn_implementation='sdpa'
block=m.Qwen2_5_VLDecoderLayer(cfg,2).eval();prefix='model.layers.2.'
with safe_open(root/'teleocr-hf/model.safetensors',framework='pt') as f:state={k[len(prefix):]:f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}
block.load_state_dict(state)
b=torch.from_numpy(np.load(root/'validation/reference/text-stages/llm_layer_1.npy')).unsqueeze(0)
n=torch.from_numpy(np.fromfile(root/'validation/native-f32/text-native-stages/llm_layer_1.f32',np.float32).reshape(b.shape))
ref=torch.from_numpy(np.load(root/'validation/reference/text-stages/llm_layer_2.npy')).unsqueeze(0)
native=torch.from_numpy(np.fromfile(root/'validation/native-f32/text-native-stages/llm_layer_2.f32',np.float32).reshape(ref.shape))
ids=torch.from_numpy(np.load(root/'validation/reference/text-stages/mrope_positions.npy')).long().T.unsqueeze(1)
with torch.inference_mode():
    positions=block.self_attn.rotary_emb(b,ids)
    post=b+block.self_attn(block.input_layernorm(b),position_embeddings=positions,position_ids=ids[0])[0]
    norm=block.post_attention_layernorm(post);gate=block.mlp.gate_proj(norm);up=block.mlp.up_proj(norm);silu=block.mlp.act_fn(gate);product=silu*up;down=block.mlp.down_proj(product);original=post+down
    source_on_native=block(n,position_embeddings=positions,position_ids=ids[0])[0]
    arrays={'input':post[0],'norm_w':block.post_attention_layernorm.weight,'gate_w':block.mlp.gate_proj.weight,'gate_b':torch.zeros(cfg.intermediate_size),'up_w':block.mlp.up_proj.weight,'up_b':torch.zeros(cfg.intermediate_size),'down_w':block.mlp.down_proj.weight,'down_b':torch.zeros(cfg.hidden_size),'norm':norm[0],'gate':gate[0],'up':up[0],'silu':silu[0],'product':product[0],'down':down[0],'output':original[0]}
    writer=gguf.GGUFWriter(out/'probe.gguf','teleocr_probe')
    for name,a in arrays.items():a=a.detach().numpy();writer.add_tensor(name,a);np.save(out/(name+'.npy'),a)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    p=post.double();z=p*torch.rsqrt(p.square().mean(-1,keepdim=True)+1e-6)*block.post_attention_layernorm.weight.double()
    d=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(z,block.mlp.gate_proj.weight.double()))*torch.nn.functional.linear(z,block.mlp.up_proj.weight.double()),block.mlp.down_proj.weight.double())+p
    np.save(out/'float64_output.npy',d[0].numpy())
    results={}
    for label,a,c in [('source original input vs frozen',original,ref),('source native input vs frozen',source_on_native,ref),('native vs source native input',native,source_on_native),('source F32 vs F64 MLP',original.double(),d)]:
        diff=(a-c).abs();at=np.unravel_index(int(diff.argmax()),diff.shape);r={'max_abs':float(diff[at]),'index':[int(i) for i in at],'actual':float(a[at]),'reference':float(c[at]),'count_over_0_002':int(torch.count_nonzero(diff>.002))};results[label]=r;print(label,r)
    (out/'propagation.json').write_text(json.dumps(results,indent=2)+'\n')
