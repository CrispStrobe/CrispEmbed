#!/usr/bin/env python3
"""One official CPU/BF16 original generate call, separately named reference lane.

No native model, legacy trace dictionary, FP32 model or per-step cache copy.
Only IDs/text/EOS, full-vocabulary generation scores, prepared inputs and measured
timing/RSS/cgroup receipts are persisted. F32 protocols/defaults are untouched.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/lighton4b-bf16-freegen-v1.json'


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while data:=stream.read(8*2**20):h.update(data)
    return h.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def memory(proc=Path('/proc'),cgroup=Path('/sys/fs/cgroup')):
    host=int(next(line.split()[1]for line in(proc/'meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024
    result={'host_available_bytes':host,'available_bytes':host}
    for limit,current,stat,peak in(('memory.max','memory.current','memory.stat','memory.peak'),
        ('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes','memory/memory.stat','memory/memory.max_usage_in_bytes')):
        if not(cgroup/limit).is_file()or not(cgroup/current).is_file():continue
        raw=(cgroup/limit).read_text().strip()
        if raw=='max':continue
        maximum=int(raw);used=int((cgroup/current).read_text());inactive=0
        if(cgroup/stat).is_file():
            stats=dict(line.split()for line in(cgroup/stat).read_text().splitlines())
            inactive=max(0,int(stats.get('inactive_file',stats.get('total_inactive_file',0)))-int(stats.get('file_dirty',0))-int(stats.get('file_writeback',0)))
        result.update(cgroup_limit_bytes=maximum,cgroup_current_bytes=used,cgroup_reclaimable_inactive_file_bytes=inactive,
            cgroup_hard_margin_bytes=max(0,maximum-used),available_bytes=min(host,max(0,maximum-used+inactive)))
        if(cgroup/peak).is_file():result['worker_cgroup_peak_bytes']=int((cgroup/peak).read_text())
        break
    status=(proc/'self/status').read_text()
    for label,key in(('VmRSS:','process_rss_bytes'),('VmHWM:','process_hwm_bytes')):
        for line in status.splitlines():
            if line.startswith(label):result[key]=int(line.split()[1])*1024
    result['process_ru_maxrss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    return result


class Monitor:
    def __init__(self):self.stop=threading.Event();self.lock=threading.Lock();self.peaks={};self.latest={};self.thread=None
    def sample(self):
        values=memory()
        with self.lock:
            self.latest=values
            for key in('process_rss_bytes','process_hwm_bytes','process_ru_maxrss_bytes','cgroup_current_bytes','worker_cgroup_peak_bytes'):
                self.peaks[key]=max(self.peaks.get(key,0),values.get(key,0))
        return values
    def start(self,progress):
        def run():
            while not self.stop.wait(2):
                self.sample();progress()
        self.sample();self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
    def finish(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=5)
        self.sample()


def score_files(scores,ids,output,on_row=None):
    import numpy as np
    require(1<=len(ids)<=256 and len(scores)==len(ids),'all generated score positions required')
    rows=[]
    for index,(score,token)in enumerate(zip(scores,ids)):
        value=score.detach().cpu().float().contiguous().numpy()
        require(value.shape==(1,248320)and np.isfinite(value).all(),'complete finite248320-vocabulary scores required')
        require(type(token)is int and int(value.argmax(axis=-1)[0])==token,'greedy generated ID differs from official score')
        path=output/f'scores-{index:03d}.f32';temporary=path.with_suffix('.partial')
        with temporary.open('wb')as stream:stream.write(memoryview(value))
        temporary.replace(path)
        shifted=value.astype(np.float64)-float(value.max());confidence=float(1.0/np.exp(shifted).sum())
        rows.append(dict(position=index,file=path.name,shape=[1,248320],bytes=value.nbytes,sha256=digest(path),
            selected_id=token,selected_probability=confidence,score_dtype='float32'))
        if on_row:on_row(rows[-1])
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--model-dir',type=Path);p.add_argument('--threads',type=int,default=2);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args(argv)
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()),'hosted CPU worker required')
    require(a.threads==2,'frozen thread count is2')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(actual==a.source_sha and len(actual)==40,'exact immutable source differs')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT),'tracked source is dirty')
    a.output=a.output.resolve();a.scratch=a.scratch.resolve()
    require(not a.output.is_relative_to(a.scratch)and not a.scratch.is_relative_to(a.output),'output and scratch must be separate trees')
    a.output.mkdir(parents=True,exist_ok=True);require(not any(a.output.iterdir()),'fresh empty output directory required')
    a.scratch.mkdir(parents=True,exist_ok=True)
    report={'schema':'lighton4b-bf16-freegen-capture/v1','status':'running','stage':'initializing','source_sha':actual,
        'driver_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),'requested_device':'cpu','requested_dtype':'bfloat16',
        'original_forward_executed':False,'native_forward_executed':False,'native_qualified':False,'model_qualified':False,
        'f32_reference_unchanged':True,'observer_off_replay_observed':False,'scores':[],'model_forward_calls':0}
    lock=threading.RLock();started=time.monotonic();monitor=Monitor()
    def save():
        with lock:
            report['elapsed_seconds']=time.monotonic()-started
            with monitor.lock:report['memory_latest']=dict(monitor.latest);report['measured_memory_peaks']=dict(monitor.peaks)
            atomic(a.output/'capture.json',report)
    def stage(name):
        report['stage']=name;monitor.sample();save();print(json.dumps({'stage':name,'elapsed_seconds':time.monotonic()-started}),flush=True)
    def heartbeat():
        with monitor.lock:peaks=dict(monitor.peaks);latest=dict(monitor.latest)
        atomic(a.output/'heartbeat.json',dict(stage=report.get('stage'),elapsed_seconds=time.monotonic()-started,
            model_forward_calls=report.get('model_forward_calls',0),memory_latest=latest,measured_memory_peaks=peaks))
    monitor.start(heartbeat)
    try:
        protocol=json.loads(PROTOCOL.read_text());case=protocol['case']
        require(protocol['schema']=='lighton4b-bf16-freegen/v1'and protocol['precision']=='bfloat16','distinct BF16 protocol required')
        for name,sha in protocol['source_sha256'].items():require(digest(ROOT/name)==sha,'frozen source/helper changed')
        require(digest(ROOT/case['file'])==case['sha256']and case['max_new_tokens']==256 and case['prompt']=='','frozen source/request differs')
        checkpoint_bytes=9078620536;output_needed=protocol['guards']['max_score_bytes']+32*2**20+2**30
        scratch_needed=(0 if a.prepare_only else 2*checkpoint_bytes)+19989343+4*2**30
        disks={name:dict(free_bytes=shutil.disk_usage(path).free,device=path.stat().st_dev)for name,path in(('scratch',a.scratch),('output',a.output))}
        report['disk_preflight']=dict(paths=disks,required_scratch_bytes=scratch_needed,required_output_bytes=output_needed)
        if disks['scratch']['device']==disks['output']['device']:
            require(disks['scratch']['free_bytes']>=scratch_needed+output_needed,'shared filesystem free-space bound failed')
        else:require(disks['scratch']['free_bytes']>=scratch_needed and disks['output']['free_bytes']>=output_needed,'separate filesystem free-space bound failed')
        require(a.prepare_only or memory()['available_bytes']>=protocol['guards']['minimum_available_ram_bytes'],'BF16 minimum22GiB cgroup-aware available RAM not met')
        os.environ['HF_HOME']=str(a.scratch/'hf');os.environ['HF_HUB_CACHE']=str(a.scratch/'hf/hub');os.environ['HF_HUB_DISABLE_XET']='1'
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        sys.path.insert(0,str(ROOT/'models'))
        spec=importlib.util.spec_from_file_location('bf16_source_converter',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        policy=converter._lighton4b_profile()
        stage('acquire-metadata-tokenizer')
        model_dir=a.model_dir or Path(snapshot_download(policy.MODEL,revision=policy.REVISION,
            allow_patterns=list(converter.lighton4b_metadata_files()),cache_dir=os.environ['HF_HUB_CACHE'],token=False))
        proof,config=converter.load_lighton4b_export_source(model_dir,policy.MODEL,policy.REVISION)
        report['source']=dict(model=policy.MODEL,revision=policy.REVISION,tokenizer_sha256=policy.TOKENIZER_SHA256,
            tokenizer_payload_verified=proof['tokenizer']['payload_verified'])
        import torch,transformers,tokenizers,numpy,PIL
        from PIL import Image
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,Qwen3_5Config
        packages={name:importlib.metadata.version(name)for name in protocol['packages']}
        require(packages==protocol['packages'],'frozen CPU framework package versions differ')
        report['packages']=packages
        hashes={name:hashlib.sha256(Path(inspect.getfile(cls)).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for name,cls in(('model',Qwen3_5ForConditionalGeneration),('config',Qwen3_5Config))}
        report['upstream']=converter.validate_lighton4b_meta_sources(transformers.__version__,hashes)
        torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
        stage('official-processor')
        processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        image=Image.open(ROOT/case['file']).convert('RGB')
        require(list(image.size)==case['original_rgb_size']and hashlib.sha256(numpy.asarray(image,dtype=numpy.uint8).tobytes()).hexdigest()==case['original_rgb_sha256'],'source RGB differs')
        messages=[{'role':'user','content':[{'type':'image','image':image}]}]
        rendered=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        require(rendered==case['rendered_prompt'],'official rendered prompt differs')
        inputs=processor(text=[rendered],images=[image],return_tensors='pt',return_mm_token_type_ids=True)
        require(inputs['image_grid_thw'].tolist()==[[1,20,28]]and list(inputs['pixel_values'].shape)==[560,1536],'actual processor geometry differs')
        require(140<inputs['input_ids'].shape[1]<=256 and inputs['input_ids'].shape==inputs['mm_token_type_ids'].shape,'actual token/modality bounds differ')
        prepared={}
        for name,value in inputs.items():
            path=a.output/('input-'+name+'.npy');numpy.save(path,value.detach().cpu().numpy(),allow_pickle=False)
            prepared[name]=dict(shape=list(value.shape),dtype=str(value.dtype),file=path.name,sha256=digest(path))
        (a.output/'rendered-prompt.txt').write_text(rendered)
        report.update(prepared_inputs=prepared,processor_class=type(processor).__module__+'.'+type(processor).__name__,
            image_processor_class=type(processor.image_processor).__module__+'.'+type(processor.image_processor).__name__,
            image_processor_backend=getattr(processor.image_processor,'backend',None),actual_grid=[[1,20,28]],
            image_processor_source_sha256=digest(inspect.getfile(type(processor.image_processor))),
            source_metadata_sha256=proof['source_metadata_verified'])
        if a.prepare_only:report['status']='prepared';stage('prepared-without-checkpoint-or-forward');return 0
        stage('acquire-verify-checkpoint')
        if a.model_dir is None:model_dir=Path(snapshot_download(policy.MODEL,revision=policy.REVISION,
            allow_patterns=list(converter.lighton4b_metadata_files())+['model.safetensors'],cache_dir=os.environ['HF_HUB_CACHE'],token=False))
        report['checkpoint']=converter.verify_lighton4b_checkpoint(model_dir/'model.safetensors',config)
        if a.model_dir is None:
            # Flush only this task-owned downloaded blob, so freshly dirty file
            # cache is not confused with unreclaimable model RAM admission.
            with(model_dir/'model.safetensors').open('rb')as stream:os.fsync(stream.fileno())
        require(memory()['available_bytes']>=protocol['guards']['minimum_available_ram_bytes'],'BF16 memory admission failed after checkpoint acquisition')
        stage('load-official-bf16-model');load_started=time.monotonic()
        model=Qwen3_5ForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='eager',low_cpu_mem_usage=True).eval()
        parameters=list(model.parameters());require(all(p.device.type=='cpu'for p in parameters),'CPU-only model required')
        require(sum(p.numel()for p in parameters)==4539265536 and model.get_input_embeddings().weight is model.get_output_embeddings().weight,'original parameter/tied-head identity differs')
        require(all(p.dtype in(torch.bfloat16,torch.float32)for p in parameters)and sum(p.numel()*p.element_size()for p in parameters)<=4539265536*2+512*2**20,'BF16 model storage census differs')
        report.update(model_load_seconds=time.monotonic()-load_started,parameter_dtype_counts=dict(Counter(str(p.dtype)for p in parameters)),
            actual_parameter_bytes=sum(p.numel()*p.element_size()for p in parameters),generation_config=model.generation_config.to_dict(),
            mkldnn_enabled=torch.backends.mkldnn.enabled)
        require(model.generation_config.eos_token_id in(248046,[248046])and model.generation_config.pad_token_id==248044,'actual official generation stop policy differs')
        stage('official-generate');generation_started=time.monotonic()
        def observed_forward(module,args):
            report['model_forward_calls']+=1;report['original_forward_executed']=True
        handle=model.register_forward_pre_hook(observed_forward)
        try:
            with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=256,do_sample=False,use_cache=True,
                return_dict_in_generate=True,output_scores=True)
        finally:handle.remove()
        ids=generated.sequences[:,inputs['input_ids'].shape[1]:].reshape(-1).tolist()
        report.update(generation_seconds=time.monotonic()-generation_started,generated_ids=ids,
            decoded=processor.batch_decode(generated.sequences[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0],
            eos_completed=bool(ids and ids[-1]==248046),cap_hit=bool(len(ids)==256 and ids[-1]!=248046))
        stage('persist-full-vocabulary-scores')
        def persisted(row):report['scores'].append(row);save()
        score_files(generated.scores,ids,a.output,on_row=persisted)
        require(sum(row['bytes']for row in report['scores'])<=protocol['guards']['max_score_bytes'],'score storage bound')
        report.update(status='captured',stop='eos'if report['eos_completed']else'max_tokens')
        stage('complete');return 0
    except Exception as error:
        details=str(error).replace(str(ROOT),'[repository]').replace(str(a.scratch),'[scratch]').replace(str(a.output),'[output]')
        for key in('HF_TOKEN','HF_TOKEN_2'):
            if os.environ.get(key):details=details.replace(os.environ[key],'[redacted]')
        report.update(status='failed',error_type=type(error).__name__,error=details[:2000]);return 1
    finally:
        monitor.finish();save()


if __name__=='__main__':raise SystemExit(main())
