#!/usr/bin/env python3
"""Original BF16 four-case panel, frozen inputs before one model load.

No native candidate, reference-vector injection, hidden/cache dictionary or F32
model. Pinned baseline helpers persist every generated full-vocabulary score.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'tests/regression/lighton4b-mixed-v1/frozen-inputs.json'
INPUT_SHA='a07af76028a0d53c8bfc0ccafedac348eaeb725c8c8c55e3136906f202b8f65d'
PROTOCOL=ROOT/'docs/lighton4b-mixed-calibration-v1.json'
PROTOCOL_SHA='3a1738c26cf5a34680fea1468bea07c06ee26659f7ddb7e249299677746ba36d'
BASE=ROOT/'tools/lighton4b_bf16_freegen.py'
BASE_SHA='f19f63b832fdd91db61f4586fb5872333719411e1c30ee0f40d43433db7024fd'

def require(value,message):
    if not value:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--prepare-only',action='store_true');a=p.parse_args(argv)
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()),'hosted CPU worker required')
    require(a.threads==2 and len(a.source_sha)==40,'frozen thread/source contract')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==a.source_sha,'immutable source mismatch')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT),'tracked source dirty')
    require(digest(INPUTS)==INPUT_SHA and digest(PROTOCOL)==PROTOCOL_SHA and digest(BASE)==BASE_SHA,'frozen protocol/input/helper changed')
    a.output=a.output.resolve();a.scratch=a.scratch.resolve()
    require(not a.output.is_relative_to(a.scratch)and not a.scratch.is_relative_to(a.output),'separate scratch/output required')
    a.output.mkdir(parents=True,exist_ok=True);require(not any(a.output.iterdir()),'fresh output required');a.scratch.mkdir(parents=True,exist_ok=True)
    core=load('frozen_bf16_helpers',BASE);monitor=core.Monitor();started=time.monotonic()
    report={'schema':'lighton4b-fourcase-bf16/v1','status':'running','source_sha':a.source_sha,
        'driver_sha256':digest(__file__),'frozen_helper_sha256':BASE_SHA,'protocol_sha256':PROTOCOL_SHA,
        'input_manifest_sha256':INPUT_SHA,'gpu_requested':False,'model_dtype':'bfloat16','model_qualified':False,
        'native_forward_executed':False,'original_forward_executed':False,'model_loads':0,'cases':{}}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        with monitor.lock:report['memory_latest']=dict(monitor.latest);report['measured_memory_peaks']=dict(monitor.peaks)
        core.atomic(a.output/'batch.json',report)
    def stage(name):report['stage']=name;monitor.sample();save();print('SOURCE_STAGE',name,flush=True)
    def heartbeat():core.atomic(a.output/'heartbeat.json',{'stage':report.get('stage'),'elapsed_seconds':time.monotonic()-started,'cases_completed':sum(x.get('status')=='captured'for x in list(report['cases'].values()))})
    monitor.start(heartbeat)
    try:
        protocol=json.loads(PROTOCOL.read_text());frozen=json.loads(INPUTS.read_text());baseline=json.loads((ROOT/'docs/lighton4b-bf16-freegen-v1.json').read_text())
        require(protocol['schema']=='lighton4b-mixed-calibration/v1'and frozen['schema']=='lighton4b-mixed-frozen-inputs/v1','panel schema differs')
        require(len(protocol['cases'])==len(frozen['cases'])==4,'four frozen cases required')
        for name,sha in baseline['source_sha256'].items():require(digest(ROOT/name)==sha,'original pinned source/helper changed')
        available=core.memory()['available_bytes'];report['initial_available_ram_bytes']=available
        require(a.prepare_only or available>=22*2**30,'minimum22GiB cgroup-aware available RAM')
        needed=(0 if a.prepare_only else 2*9078620536)+4*2**30
        shared=a.scratch.stat().st_dev==a.output.stat().st_dev
        require(shutil.disk_usage(a.scratch).free>=needed+(3*2**30 if shared else 0),'scratch disk guard')
        require(shutil.disk_usage(a.output).free>=3*2**30,'four-case score/output headroom')
        os.environ.update(HF_HOME=str(a.scratch/'hf'),HF_HUB_CACHE=str(a.scratch/'hf/hub'),HF_HUB_DISABLE_XET='1')
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars();sys.path.insert(0,str(ROOT/'models'))
        converter=load('fourcase_source_converter',ROOT/'models/convert-qwen35-ocr-to-gguf.py');policy=converter._lighton4b_profile()
        stage('acquire-metadata-tokenizer')
        model_dir=Path(snapshot_download(policy.MODEL,revision=policy.REVISION,allow_patterns=list(converter.lighton4b_metadata_files()),cache_dir=os.environ['HF_HUB_CACHE'],token=False))
        proof,config=converter.load_lighton4b_export_source(model_dir,policy.MODEL,policy.REVISION)
        import torch,transformers,numpy as np
        from PIL import Image
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,Qwen3_5Config
        packages={name:importlib.metadata.version(name)for name in baseline['packages']};require(packages==baseline['packages'],'frozen CPU package versions differ');report['packages']=packages
        hashes={name:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes().replace(b'\r\n',b'\n')).hexdigest()for name,cls in [('model',Qwen3_5ForConditionalGeneration),('config',Qwen3_5Config)]}
        report['upstream']=converter.validate_lighton4b_meta_sources(transformers.__version__,hashes)
        report['source']={'model':policy.MODEL,'revision':policy.REVISION,'tokenizer_sha256':policy.TOKENIZER_SHA256,'source_metadata_sha256':proof['source_metadata_verified']}
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        prepared={};case_rows={x['id']:x for x in protocol['cases']}
        stage('prepare-all-four-inputs-before-model')
        for row in frozen['cases']:
            ident=row['id'];case=case_rows[ident];destination=a.output/ident;destination.mkdir()
            require(case['prompt']==''and case['max_new_tokens']==256 and case['eos_ids']==[248046]and case['do_sample']is False,'frozen request changed')
            path=INPUTS.parent/row['png_file'];require(path.resolve().is_relative_to(INPUTS.parent.resolve())and digest(path)==row['png_sha256'],'frozen PNG differs')
            image=Image.open(path).convert('RGB');require(list(image.size)==[row['width'],row['height']]and hashlib.sha256(np.asarray(image,dtype=np.uint8).tobytes()).hexdigest()==row['rgb_sha256'],'frozen decoded RGB differs')
            rendered=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':image}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            require(rendered==baseline['case']['rendered_prompt'],'frozen image-only template differs')
            inputs=processor(text=[rendered],images=[image],return_tensors='pt',return_mm_token_type_ids=True)
            grid=case['expected_grid'];patches=case['expected_patches'];count=case['expected_input_tokens']
            require(grid==row['expected_grid']and count==row['expected_input_tokens'],'protocol/input geometry declarations differ')
            require(inputs['image_grid_thw'].tolist()==[grid]and list(inputs['pixel_values'].shape)==[patches,1536],'actual original processor geometry differs')
            require(list(inputs['input_ids'].shape)==[1,count]and inputs['mm_token_type_ids'].shape==inputs['input_ids'].shape,'actual original token/modality count differs')
            require(int((inputs['mm_token_type_ids']==1).sum())==case['expected_image_tokens'],'actual original image token count differs')
            records={}
            for key,value in inputs.items():
                target=destination/('input-'+key+'.npy');np.save(target,value.detach().cpu().numpy(),allow_pickle=False)
                records[key]={'file':target.name,'shape':list(value.shape),'dtype':str(value.dtype),'sha256':digest(target)}
            (destination/'rendered-prompt.txt').write_text(rendered)
            report['cases'][ident]={'status':'prepared','split':case['split'],'png_sha256':row['png_sha256'],'rgb_sha256':row['rgb_sha256'],'prepared_inputs':records,'expected_grid':grid,'expected_input_tokens':count}
            prepared[ident]=inputs;save()
        report['all_inputs_verified_before_model_load']=True;stage('all-four-inputs-authenticated')
        if a.prepare_only:report['status']='prepared';return 0
        stage('acquire-verify-checkpoint')
        model_dir=Path(snapshot_download(policy.MODEL,revision=policy.REVISION,allow_patterns=list(converter.lighton4b_metadata_files())+['model.safetensors'],cache_dir=os.environ['HF_HUB_CACHE'],token=False))
        report['checkpoint']=converter.verify_lighton4b_checkpoint(model_dir/'model.safetensors',config)
        with(model_dir/'model.safetensors').open('rb')as stream:os.fsync(stream.fileno())
        require(core.memory()['available_bytes']>=22*2**30,'BF16 memory admission after checkpoint download')
        stage('load-one-official-bf16-model')
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).eval()
        parameters=list(model.parameters());require(all(p.device.type=='cpu'for p in parameters)and not torch.cuda.is_available(),'CPU-only model required')
        require(sum(p.numel()for p in parameters)==4539265536 and model.get_input_embeddings().weight is model.get_output_embeddings().weight,'original parameter/tied-head identity differs')
        require(all(p.dtype in(torch.bfloat16,torch.float32)for p in parameters)and sum(p.numel()*p.element_size()for p in parameters)<=4539265536*2+512*2**20,'BF16 storage bound')
        require(model.generation_config.eos_token_id in(248046,[248046])and model.generation_config.pad_token_id==248044,'official stop policy differs')
        report.update(model_loads=1,parameter_dtype_counts=dict(Counter(str(p.dtype)for p in parameters)),actual_parameter_bytes=sum(p.numel()*p.element_size()for p in parameters),mkldnn_enabled=torch.backends.mkldnn.enabled)
        for ident,inputs in prepared.items():
            stage('official-generate-'+ident);destination=a.output/ident;case=case_rows[ident];calls=[0];begin=time.monotonic()
            def count(module,args):
                calls[0]+=1;report['original_forward_executed']=True;report['current_case_forward_calls']=calls[0]
            handle=model.register_forward_pre_hook(count)
            try:
                torch.manual_seed(0)
                with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
            finally:handle.remove()
            ids=generated.sequences[:,inputs['input_ids'].shape[1]:].reshape(-1).tolist()
            require(1<=len(ids)<=256 and calls[0]==len(ids),'complete original generated positions required')
            eos=ids[-1]==248046
            capture={'schema':'lighton4b-fourcase-bf16-capture/v1','status':'captured','source_sha':a.source_sha,'case_id':ident,'split':case['split'],
                'protocol_sha256':PROTOCOL_SHA,'input_manifest_sha256':INPUT_SHA,'packages':packages,'source':report['source'],'checkpoint':report['checkpoint'],
                'prepared_inputs':report['cases'][ident]['prepared_inputs'],'generated_ids':ids,'decoded':processor.batch_decode(generated.sequences[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0],
                'eos_completed':eos,'cap_hit':len(ids)==256 and not eos,'stop':'eos'if eos else'max_tokens','generation_seconds':time.monotonic()-begin,
                'model_forward_calls':calls[0],'original_forward_executed':True,'native_forward_executed':False,'quality_qualified':False,'scores':[]}
            capture['scores']=core.score_files(generated.scores,ids,destination)
            require(sum(x['bytes']for x in capture['scores'])<=256*248320*4,'complete-score output bound')
            core.atomic(destination/'capture.json',capture)
            report['original_forward_executed']=True;report['cases'][ident].update(status='captured',capture_sha256=digest(destination/'capture.json'),generated_tokens=len(ids),eos_completed=eos,cap_hit=capture['cap_hit'],model_forward_calls=calls[0]);save()
            del generated
        report['status']='captured';stage('complete');return 0
    except Exception as error:
        detail=str(error).replace(str(ROOT),'[repository]').replace(str(a.scratch),'[scratch]').replace(str(a.output),'[output]')
        report.update(status='failed',error_type=type(error).__name__,error=detail[:2000]);return 1
    finally:monitor.finish();save()
if __name__=='__main__':raise SystemExit(main())
