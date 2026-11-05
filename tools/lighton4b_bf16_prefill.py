#!/usr/bin/env python3
"""One official BF16 generate-prefill, streaming bounded stage snapshots.

The frozen BF16 driver is executed with one AST substitution: its generate
expression becomes this observer's single-token generate call. No cache arrays,
full F32 model, original long decode, native code or numerical budget changes.
"""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DRIVER=ROOT/'tools/lighton4b_bf16_freegen.py'
DRIVER_SHA='f19f63b832fdd91db61f4586fb5872333719411e1c30ee0f40d43433db7024fd'
SCORE_SHA='a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd'
INPUT_SHA={
'attention_mask':'bf86c53fad0078f2c0b8ae6ebcc0bc1fad1469287a5fd2ce76c45558c45defe2',
'image_grid_thw':'31db2f395670c65f1329ed36856025a4e28158dd158ffcae567a939151283278',
'input_ids':'6f1b0a033c6d381c877956b6fbd215d8be4515d9a4ec733ad302b47a1743203d',
'mm_token_type_ids':'f5036a7310cca98c52e7846682523e02b245fbfb823ecf78fb59fe20832ae723',
'pixel_values':'4475c40a09fc35f429c104c6899a1d7a29e124d1327f893411c7d1699c09512b'}
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class Observer:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.handles=[];self.records={};self.total=0;self.calls=0
        self.report={'schema':'lighton4b-bf16-prefill/v1','status':'incomplete','stages':self.records,
            'model_dtype':'bfloat16','gpu_requested':False,'native_forward_executed':False,
            'original_full_generation_repeated':False,'observer_first_score_bit_exact':False,
            'baseline_source':'07543dcc80ebcecad644815d9397be2570a51dd4','baseline_first_score_sha256':SCORE_SHA,
            'native_parity_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'driver_sha256':DRIVER_SHA,'observer_sha256':sha(__file__)}
    def save(self):
        self.report['snapshot_bytes']=self.total
        temporary=self.output/'prefill.json.partial';temporary.write_text(json.dumps(self.report,indent=2,sort_keys=True)+'\n');temporary.replace(self.output/'prefill.json')
    def snapshot(self,name,tensor,shape=None):
        import torch
        import numpy as np
        require(name not in self.records and re.fullmatch(r'[A-Za-z0-9_.]+',name),'duplicate/invalid stage')
        require(torch.is_tensor(tensor),'hook returned no tensor: '+name)
        require(tensor.device.type=='cpu' and tensor.numel()<=16*2**20,'stage CPU/element bound: '+name)
        require(shape is None or list(tensor.shape)==shape,'stage shape differs: '+name)
        source_dtype=str(tensor.dtype);original_shape=list(tensor.shape)
        # A single stage is materialized and released before returning the hook.
        value=tensor.detach().float().contiguous().numpy()
        require(value.ndim in(1,2,3,4) and np.isfinite(value).all(),'nonfinite/rank stage: '+name)
        require(self.total+value.nbytes<=512*2**20,'total512MiB snapshot bound')
        import shutil
        require(shutil.disk_usage(self.output).free>=value.nbytes+2**30,'snapshot disk headroom')
        path=self.output/(name+'.f32');temporary=path.with_suffix('.partial')
        with temporary.open('wb') as stream:stream.write(memoryview(value))
        temporary.replace(path)
        self.total+=value.nbytes
        self.records[name]={'shape':original_shape,'original_dtype':source_dtype,'stored_dtype':'little-endian-float32',
            'file':path.name,'bytes':value.nbytes,'sha256':sha(path),'abs_max':float(np.abs(value).max()),
            'l2':float(np.linalg.norm(value.astype(np.float64).reshape(-1))),'layout':'original-contiguous-row-major'}
        self.save()
    def hook(self,name,shape=None):
        def capture(module,args,result):
            import torch
            if isinstance(result,tuple):result=result[0]
            require(torch.is_tensor(result),'unexpected module output: '+name)
            self.snapshot(name,result,shape)
        return capture
    def capture_generate(self,model,inputs,processor):
        import numpy as np
        import torch
        require(self.calls==0,'exactly one generate-prefill admitted');self.calls+=1
        require(not torch.cuda.is_available() and all(p.device.type=='cpu' for p in model.parameters()),'CPU-only capture required')
        require(set(inputs)==set(INPUT_SHA),'original input field set differs')
        for key,value in inputs.items():
            stream=io.BytesIO();np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
            require(hashlib.sha256(stream.getbuffer()).hexdigest()==INPUT_SHA[key],'original prepared input bytes differ: '+key)
        self.report['input_sha256']=INPUT_SHA
        self.snapshot('processor.patches',inputs['pixel_values'],[560,1536])
        visual=model.model.visual;language=model.model.language_model
        require(len(visual.blocks)==24 and len(language.layers)==32,'original layer counts differ')
        self.handles.append(visual.patch_embed.register_forward_hook(self.hook('vision.patch',[560,1024])))
        for index,block in enumerate(visual.blocks):
            self.handles.append(block.register_forward_hook(self.hook(f'vision.layer.{index}',[560,1024])))
        for name,label,shape in [('norm','norm',[560,1024]),('linear_fc1','up',[140,4096]),('act_fn','activation',[140,4096])]:
            self.handles.append(getattr(visual.merger,name).register_forward_hook(self.hook('vision.merger.'+label,shape)))
        self.handles.append(visual.merger.register_forward_hook(self.hook('vision.merged',[140,2560])))
        def vision_input(module,args,kwargs):
            self.snapshot('vision.embedded',args[0],[560,1024])
            cosine,sine=kwargs['position_embeddings']
            self.snapshot('vision.rotary_cos',cosine);self.snapshot('vision.rotary_sin',sine)
        self.handles.append(visual.blocks[0].register_forward_pre_hook(vision_input,with_kwargs=True))
        original_position=visual.fast_pos_embed_interpolate
        def position(grid):
            value=original_position(grid);self.snapshot('vision.position',value,[560,1024]);return value
        visual.fast_pos_embed_interpolate=position
        # Capture raw text lookup before masked image replacement.
        self.handles.append(model.get_input_embeddings().register_forward_hook(self.hook('text.embedded',[1,154,2560])))
        def language_input(module,args,kwargs):
            self.snapshot('prefill.input',kwargs['inputs_embeds'],[1,154,2560])
            positions=kwargs['position_ids']
            require(list(positions.shape) in([3,1,154],[4,1,154]),'original position axis convention differs')
            self.snapshot('prefill.positions',positions)
            self.report['position_axes']=int(positions.shape[0]);self.save()
        self.handles.append(language.register_forward_pre_hook(language_input,with_kwargs=True))
        for index,block in enumerate(language.layers):
            self.handles.append(block.register_forward_hook(self.hook(f'prefill.layer.{index}',[1,154,2560])))
        self.handles.append(language.norm.register_forward_hook(self.hook('prefill.norm',[1,154,2560])))
        calls=[0]
        def count(module,args):calls[0]+=1
        self.handles.append(model.register_forward_pre_hook(count))
        try:
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,
                    return_dict_in_generate=True,output_scores=True)
            require(calls[0]==1 and len(generated.scores)==1,'single original prefill forward required')
            ids=generated.sequences[:,inputs['input_ids'].shape[1]:].reshape(-1).tolist()
            require(ids==[27],'original first generated ID changed under observer')
            score=generated.scores[0].detach().cpu().float().contiguous()
            require(list(score.shape)==[1,248320],'original full first vocabulary required')
            self.snapshot('prefill.logits',score,[1,248320])
            require(self.records['prefill.logits']['sha256']==SCORE_SHA,'original first score changed under observer')
            required={'processor.patches','vision.patch','vision.position','vision.embedded','vision.merged',
                'text.embedded','prefill.input','prefill.positions','prefill.norm','prefill.logits'}
            required|={f'vision.layer.{i}'for i in range(24)}|{f'prefill.layer.{i}'for i in range(32)}
            require(required<=set(self.records),'missing original required stage')
            self.report.update(status='captured',generated_ids=ids,model_forward_calls=calls[0],
                observer_first_score_bit_exact=True,first_selected_probability=0.821857347181417)
            self.save();return generated
        finally:
            for handle in self.handles:handle.remove()
            visual.fast_pos_embed_interpolate=original_position
            self.save()
class ReplaceGenerate(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='generate':
            require(isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='model','frozen generation owner differs')
            self.count+=1
            node.value=ast.parse('observer.capture_generate(model, inputs, processor)',mode='eval').body
        return self.generic_visit(node)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    a=p.parse_args(argv)
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()),'hosted worker required')
    require(sha(DRIVER)==DRIVER_SHA,'frozen BF16 driver source differs')
    observer=Observer(a.output/'stages')
    transform=ReplaceGenerate();tree=transform.visit(ast.parse(DRIVER.read_text()));ast.fix_missing_locations(tree)
    require(transform.count==1,'one frozen generation expression required')
    namespace={'__name__':'targeted_bf16_driver','__file__':str(DRIVER),'observer':observer}
    exec(compile(tree,str(DRIVER),'exec'),namespace)
    # Frozen driver requires fresh output; stages live in a sibling directory.
    driver_output=a.output/'official'
    result=namespace['main'](['--source-sha',a.source_sha,'--output',str(driver_output),'--scratch',str(a.scratch),'--threads','2'])
    if result!=0:observer.report['status']='failed';observer.save();return result
    require(observer.report['observer_first_score_bit_exact'],'captured observer control missing')
    return 0
if __name__=='__main__':raise SystemExit(main())
