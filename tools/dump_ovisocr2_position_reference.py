#!/usr/bin/env python3
"""Actual original position-method observations, separate from full-model gold."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import warnings

UTIL_SHA='bcecd5a92b3266b9926272a549d2b1a0f1fe7646c698c0fa19bd96f976085356'
GRIDS=((1,32,64),(1,4,4),(1,32,32),(1,64,32),(2,8,8))


def observe(visual,grid):
    import torch
    from transformers import vision_utils as utils
    captures={};axes=[];previous=sys.getprofile()
    def profile(frame,event,arg):
        if event=='return':
            if frame.f_code is utils.get_vision_interpolation_indices_and_weights.__code__:
                if captures:raise ValueError('Duplicate original interpolation observation')
                captures['indices']=arg[0].detach().clone();captures['weights']=arg[1].detach().clone()
            if frame.f_code is utils._interpolation_axis_taps_weights.__code__:
                axes.append(frame.f_locals['src'].detach().clone())
        if previous is not None:previous(frame,event,arg)
    with warnings.catch_warnings(),torch.inference_mode():
        warnings.simplefilter('ignore',FutureWarning)
        ordinary=visual.fast_pos_embed_interpolate(grid)
        try:
            sys.setprofile(profile)
            observed=visual.fast_pos_embed_interpolate(grid)
        finally:sys.setprofile(previous)
    if not torch.equal(ordinary.view(torch.int32),observed.view(torch.int32)) or len(axes)!=2 or len(captures)!=2:
        raise ValueError('Original position observer not neutral/complete')
    return {**captures,'row_src':axes[0],'col_src':axes[1],'position':observed.detach()}


def build(model_dir,original,legacy,pixel_values,patch_original,legacy_patch,config,output):
    import gguf
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import Qwen3_5VisionConfig,vision_utils
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel,Qwen3_5VisionPatchEmbed
    from dump_ovisocr2_ffn_reference import sha,require_environment,metrics
    environment=require_environment();utility=Path(inspect.getfile(vision_utils))
    if sha(utility)!=UTIL_SHA:raise ValueError('Actual original vision utility source differs')
    torch.set_num_threads(4)
    cfg=Qwen3_5VisionConfig.from_dict(config);cfg._attn_implementation='eager'
    with torch.device('meta'):visual=Qwen3_5VisionModel(cfg)
    patch=Qwen3_5VisionPatchEmbed(cfg).float().eval()
    with safe_open(str(Path(model_dir)/'model.safetensors'),framework='pt',device='cpu') as sf:
        table=sf.get_tensor('model.visual.pos_embed.weight').float().contiguous()
        patch_state={k:sf.get_tensor('model.visual.patch_embed.'+k).float().contiguous() for k in patch.state_dict()}
    visual.pos_embed=torch.nn.Embedding.from_pretrained(table,freeze=True)
    patch.load_state_dict(patch_state)
    prepared={'v.pos_embed.weight':table.numpy(),
              'v.patch_embed.proj.weight':patch_state['proj.weight'].numpy().reshape(768,1536),
              'v.patch_embed.proj.bias':patch_state['proj.bias'].numpy(),
              'input.pixel_values':pixel_values,'legacy.position':legacy,
              'frozen.patch':patch_original,'legacy.patch':legacy_patch}
    report={'scope':'original position method/local patch only; old arrays unchanged',
            'environment':environment,'vision_utils_sha256':UTIL_SHA,'observer_neutrality_bit_exact':True,
            'grids':[],'weights':{k:{'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'shape':list(v.shape)}
                                  for k,v in prepared.items() if k.startswith('v.')}}
    for index,grid in enumerate(GRIDS):
        values=observe(visual,torch.tensor([grid],dtype=torch.int64))
        prefix='case.'+str(index)
        for name,value in values.items():prepared[prefix+'.'+name]=value.float().cpu().numpy()
        prepared[prefix+'.grid']=np.asarray(grid,dtype=np.float32)
        report['grids'].append({'id':index,'grid':list(grid),'patches':int(np.prod(grid))})
        if index==0:report['frozen_position_anchor']=metrics(prepared[prefix+'.position'],original)
    with torch.inference_mode():local_patch=patch(torch.from_numpy(pixel_values.copy())).detach().numpy()
    prepared['local.patch']=local_patch
    report['frozen_patch_anchor']=metrics(local_patch,patch_original)
    writer=gguf.GGUFWriter(str(output),'qwen35_position_operator_reference')
    for name,value in prepared.items():writer.add_tensor(name,np.ascontiguousarray(value,dtype=np.float32))
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    report['reference_sha256']=sha(output)
    report['arrays']={k:{'sha256':hashlib.sha256(v.tobytes()).hexdigest(),'shape':list(v.shape)} for k,v in prepared.items()}
    Path(output).with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not report['frozen_position_anchor']['bit_exact']:raise ValueError('Original position method drifts from immutable frozen anchor')
    if not report['frozen_patch_anchor']['passed']:raise ValueError('Original local patch exceeds unchanged frozen budget')
    return report
