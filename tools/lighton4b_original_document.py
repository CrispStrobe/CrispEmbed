#!/usr/bin/env python3
"""Source-bound original4B document observer; no model download or native path.

The optional capture command is hosted-only. This module imports no model
framework until an actual capture is explicitly invoked. Synthetic controls use
the observer directly and never call the original forward.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_document_parity as document
CAPTURE_TRANSPORT_ADMITTED=False


def require(value,message):
    if not value:raise ValueError(message)


def array(tensor):
    import numpy as np
    return np.asarray(tensor.detach().cpu().float().contiguous().numpy(),dtype='<f4')


class GeneratedObserver:
    """Stream every position's raw logits, final norm, positions and32 caches.

    No tensors or cache entries are modified. Snapshots are separate diagnostic
    F32 shards; they are not model weights or production inputs.
    """
    def __init__(self,output,protocol,max_bytes=24*2**30,max_legacy_bytes=8*2**30):
        self.output=Path(output);require(not self.output.exists(),'fresh observer output required')
        self.output.mkdir(parents=True);self.protocol=protocol;self.max_bytes=max_bytes
        self.records={};self.bytes=0;self.handles=[];self.phase=None;self.model=None;self.paused=False
        self.legacy={};self.max_legacy_bytes=max_legacy_bytes
        self.report={'schema':'lighton4b-original-generated-observer/v1','status':'incomplete',
            'original_forward_observed':False,'native_forward_observed':False,'quality_qualified':False,
            'records':self.records,'bytes':0}
    def save_report(self):
        self.report['bytes']=self.bytes
        temporary=self.output/'observer.json.partial';temporary.write_text(json.dumps(self.report,indent=2,sort_keys=True)+'\n')
        temporary.replace(self.output/'observer.json')
    def snapshot(self,name,tensor,shape=None):
        import numpy as np
        require(name not in self.records and re.fullmatch(r'generated\.\d+\.[A-Za-z0-9_.]+',name),'duplicate/invalid observer field')
        value=array(tensor)
        require(value.ndim in(1,2,3,4) and 0<value.size<=64*2**20,'snapshot shape/element bound')
        require(shape is None or list(value.shape)==shape,'original cache/logit geometry differs: '+name)
        require(np.isfinite(value).all(),'nonfinite original observer field: '+name)
        require(self.bytes+value.nbytes<=self.max_bytes,'observer shard byte budget exceeded')
        require(shutil.disk_usage(self.output).free>=value.nbytes+2**30,'observer disk headroom exhausted')
        destination=self.output/(name+'.f32');temporary=destination.with_suffix('.partial')
        with temporary.open('wb')as stream:stream.write(memoryview(value))
        temporary.replace(destination)
        self.records[name]={'shape':list(value.shape),'bytes':value.nbytes,
            'sha256':hashlib.sha256(memoryview(value)).hexdigest(),'file':destination.name}
        self.bytes+=value.nbytes
    def tensor(self,name,tensor):
        if self.paused or tensor is None:return
        # Account before the unchanged helper allocates its retained NumPy copy.
        # Mirror only its explicit generated-field retention filter.
        legacy=not name.startswith('generated.')or '.decoder.position_ids'in name or '.scores.'in name or name=='generated.ids'
        if legacy and hasattr(tensor,'shape'):
            size=math.prod(tensor.shape)*(4 if tensor.is_floating_point()else tensor.element_size())
            require(sum(self.legacy.values())-self.legacy.get(name,0)+size<=self.max_legacy_bytes,'legacy trace RAM budget exceeded')
            self.legacy[name]=size
        match=re.fullmatch(r'generated\.(\d+)\.decoder\.(norm|position_ids)',name)
        if not match:return
        require(int(match[1])<256,'generated position outside frozen cap')
        value_shape=list(tensor.shape)
        if match[2]=='norm':require(value_shape[0]==1 and value_shape[-1]==2560,'original final norm geometry differs')
        self.snapshot(name,tensor)
    def validate_inputs(self,processor,inputs,image,rendered):
        import numpy as np
        case=self.protocol['case']
        require(rendered==case['rendered_prompt'],'actual original template differs')
        require(list(image.size)==case['original_rgb_size'] and
            hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()==case['original_rgb_sha256'],'actual original RGB differs')
        require(inputs['image_grid_thw'].tolist()==[[1,20,28]],'actual processor grid differs from frozen expectation')
        require(list(inputs['pixel_values'].shape)==[560,1536],'actual processor patches differ')
        require(len(inputs['input_ids'].shape)==2 and inputs['input_ids'].shape[0]==1 and
            140<inputs['input_ids'].shape[1]<=256,'actual original token-length bound')
        require(list(inputs['mm_token_type_ids'].shape)==list(inputs['input_ids'].shape),'original modality labels differ')
        self.report['actual_input_geometry_verified']=True
        self.report['actual_input_shapes']={name:list(value.shape)for name,value in inputs.items()}
        self.save_report()
    def attach(self,model,phase):
        self.model=model;self.phase=phase
        def capture(module,args,result):
            if self.paused:return
            match=re.fullmatch(r'generated\.(\d+)',str(phase[0]))
            if not match:return
            index=int(match[1]);require(index<256,'generated position outside cap')
            label='generated.'+str(index)
            self.snapshot(label+'.raw_logits',result.logits[:,-1],[1,248320])
            layers=result.past_key_values.layers;require(len(layers)==32,'original cache must contain32 layers')
            for i,layer in enumerate(layers):
                if i%4==3:
                    fields=('keys','values')
                    for field in fields:
                        tensor=getattr(layer,field)
                        require(len(tensor.shape)==4 and list(tensor.shape[:2])==[1,4]and tensor.shape[-1]==256 and 1<=tensor.shape[2]<=512,
                                'original full-attention cache geometry differs')
                        self.snapshot(f'{label}.cache.{i}.{field}',tensor)
                else:
                    self.snapshot(f'{label}.cache.{i}.conv_states',layer.conv_states,[1,8192,4])
                    self.snapshot(f'{label}.cache.{i}.recurrent_states',layer.recurrent_states,[1,32,128,128])
            self.report['original_forward_observed']=True;self.save_report()
        self.handles.append(model.register_forward_hook(capture))
    def finish(self,ids,scores,replay_ids,replay_scores,decoded):
        import numpy as np
        ids=np.asarray(ids).reshape(-1).tolist();count=len(ids)
        require(1<=count<=256 and len(scores)==count and len(replay_scores)==count,'original generation coverage differs')
        self.report.update(generated_ids=ids,decoded=decoded,eos_completed=bool(ids and ids[-1]==248046),
            cap_hit=bool(count==256 and ids[-1]!=248046))
        self.save_report()
        require(np.array_equal(np.asarray(replay_ids).reshape(-1),ids),'observer changes original free-generation IDs')
        for i,(score,replay)in enumerate(zip(scores,replay_scores)):
            a,b=array(score),array(replay)
            require(a.shape==(1,248320) and np.array_equal(a,b),'observer changes original full-vocabulary scores')
            require(int(a.argmax(axis=-1)[0])==ids[i],'original ID is not greedy raw-logit selection')
            raw=self.records.get(f'generated.{i}.raw_logits')
            require(raw is not None and raw['sha256']==hashlib.sha256(memoryview(a)).hexdigest(),'raw original logits differ from generate scores')
            for field in('decoder.norm','decoder.position_ids'):
                require(f'generated.{i}.'+field in self.records,'missing original final norm/position snapshot')
            require(sum(name.startswith(f'generated.{i}.cache.')for name in self.records)==64,'missing generated cache fields')
        require(len(self.records)==67*count,'extra/duplicate generated observer fields')
        require(self.report['eos_completed'],'original document did not reach EOS; preserve capped source failure')
        self.report.update(status='passed',observer_off_ids_and_all_logits_exact=True,
            raw_logits_equal_generate_scores=True,positions=count)
        self.save_report()
    def verify_generation(self,generated,inputs,processor):
        self.paused=True
        width=inputs['input_ids'].shape[1];ids=generated.sequences[:,width:]
        # The unchanged helper has reset both phase and generation_step here.
        # Its observation hooks are inert; our hook is paused for this independent
        # same-backend replay. No reference values are supplied to model caches.
        replay=self.model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True,
            return_dict_in_generate=True,output_scores=True)
        self.finish(ids.detach().cpu().numpy(),generated.scores,
            replay.sequences[:,width:].detach().cpu().numpy(),replay.scores,
            processor.batch_decode(ids,skip_special_tokens=True)[0])
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.model=None
        self.save_report()


def adapted_capture(observer):
    """Extend only four observation sites in the exact immutable legacy helper.

    Whole-source SHA and structural counts are checked before compiling. The
    original file and upstream model implementation bytes remain unchanged.
    """
    p=document.protocol();path=ROOT/'tools/dump_qwen35_ocr_reference.py'
    require(document.sha(path)==p['source_sha256']['tools/dump_qwen35_ocr_reference.py'],'original helper changed')
    tree=ast.parse(path.read_text());counts={'save':0,'inputs':0,'attach':0,'verify':0}
    class Extend(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            self.generic_visit(node)
            if node.name=='save':
                node.body.insert(0,ast.parse('observer.tensor(name, tensor)').body[0]);counts['save']+=1
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name)and node.targets[0].id=='model':
                require('Qwen3_5ForConditionalGeneration.from_pretrained'in ast.unparse(node.value),'unexpected original model construction')
                counts['inputs']+=1
                return [ast.parse('observer.validate_inputs(processor, inputs, image, rendered)').body[0],node]
            if len(node.targets)==1 and ast.unparse(node.targets[0])=='generation_step[0]'and isinstance(node.value,ast.Constant)and node.value.value is None:
                counts['verify']+=1
                return [node,ast.parse('observer.verify_generation(generated, inputs, processor)').body[0]]
            return node
        def visit_With(self,node):
            self.generic_visit(node)
            if len(node.items)==1 and ast.unparse(node.items[0].context_expr)=='torch.inference_mode()':
                counts['attach']+=1;return [ast.parse('observer.attach(model, phase)').body[0],node]
            return node
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='dump_reference')
    adapted=Extend().visit(function)
    require(counts==dict(save=1,inputs=1,attach=1,verify=1),'original observer injection structure differs')
    module=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]+[adapted],type_ignores=[])
    ast.fix_missing_locations(module);namespace={'observer':observer,'__name__':'lighton4b_frozen_original_adapter'}
    exec(compile(module,'<lighton4b-source-bound-original-adapter>','exec'),namespace)
    return namespace['dump_reference'],{'source_sha256':document.sha(path),'injections':counts,
        'adapted_ast_sha256':hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()}


def capture(model_dir,output):
    require(os.environ.get('GITHUB_ACTIONS')=='true' or
        (Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()),'original capture requires hosted worker')
    p=document.protocol();output=Path(output);require(not output.exists(),'fresh capture output required')
    output.mkdir(parents=True)
    report={'schema':'lighton4b-original-document-adapter/v1','status':'failed','quality_qualified':False,
            'native_supported':False,'document_bundle_admitted':False}
    observer=None
    try:
        require(CAPTURE_TRANSPORT_ADMITTED,'original forward blocked: separately reviewed all-position cache transport is not admitted')
        available=int(next(line.split()[1]for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
        limit=Path('/sys/fs/cgroup/memory.max');used=Path('/sys/fs/cgroup/memory.current')
        if limit.is_file()and used.is_file()and limit.read_text().strip()!='max':
            available=min(available,int(limit.read_text())-int(used.read_text()))
        require(available>=40*2**30 and shutil.disk_usage(output).free>=48*2**30,
            'capture requires40GiB available RAM and48GiB output headroom; full F32 original and legacy traces are large')
        require(Path(model_dir).resolve()!=output.resolve(),'model snapshot and diagnostic output must differ')
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(ROOT/'models'))
        spec=importlib.util.spec_from_file_location('lighton4b_source_exporter',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        _,config=converter.load_lighton4b_export_source(model_dir,p['model']['repository'],p['model']['revision'])
        report['checkpoint']=converter.verify_lighton4b_checkpoint(Path(model_dir)/'model.safetensors',config)
        import torch,transformers,tokenizers,numpy,gguf,PIL
        from transformers import Qwen3_5ForConditionalGeneration,Qwen3_5Config
        versions={'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,
                  'numpy':numpy.__version__,'gguf':__import__('importlib.metadata',fromlist=['version']).version('gguf'),'pillow':PIL.__version__}
        require(versions==p['packages'],'original framework package closure differs')
        hashes={name:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name,cls in [('model',Qwen3_5ForConditionalGeneration),('config',Qwen3_5Config)]}
        report['upstream']=converter.validate_lighton4b_meta_sources(transformers.__version__,hashes)
        gdn_spec=importlib.util.spec_from_file_location('lighton4b_original_source_guard',ROOT/'tools/ci-heavy/lighton4b_gdn_reference.py')
        gdn=importlib.util.module_from_spec(gdn_spec);gdn_spec.loader.exec_module(gdn)
        *_,report['torch_fallback_source']=gdn.load_original()
        observer=GeneratedObserver(output/'generated',p)
        function,report['adaptation']=adapted_capture(observer)
        function(model_dir,ROOT/p['case']['file'],output/'original-reference.npz',prompt='',max_new_tokens=256,threads=2)
        report['status']='passed';report['original_observer']=observer.report
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]')
        raise
    finally:
        if observer:observer.close();report['original_observer']=observer.report
        (output/'adapter.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();capture(args.model_dir,args.output)
