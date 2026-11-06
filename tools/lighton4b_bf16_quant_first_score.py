#!/usr/bin/env python3
"""One-position genuine original BF16 Q8/Q4 oracle; no F32 model or native inference."""
import argparse
import ast
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/lighton4b-bf16-quant-first-score-v1.json'
HEADER_SHA='a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081'
def require(value,message):
    if not value:raise ValueError(message)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
    return h.hexdigest()
def source_name(name):
    if name.startswith('model.language_model.'):return 'l.'+name[len('model.language_model.'):]
    if name.startswith('model.visual.'):return 'v.'+name[len('model.visual.'):]
    raise ValueError('unsupported source tensor prefix')
def tensor_plan(header):
    result={}
    for name,row in header.items():
        if name=='__metadata__':continue
        require(row['dtype']=='BF16'and row['shape']and all(type(n)is int and n>0 for n in row['shape']),'source BF16 shape required')
        target=source_name(name);shape=row['shape'];quant=name.startswith('model.language_model.')and len(shape)==2
        stored=([shape[0],math.prod(shape[1:])]if target=='v.patch_embed.proj.weight'else shape)
        require(target not in result,'source tensor alias collision')
        result[target]={'source':name,'source_shape':shape,'stored_shape':list(reversed(stored)),'quantized':quant}
    require(len(result)==723 and sum(r['quantized']for r in result.values())==249,'source723/249 tensor policy differs')
    require(sum(math.prod(r['source_shape'])for r in result.values())==4539265536,'source parameter census differs')
    return result
def validate_inventory(parsed,plan,recipe):
    require(recipe in ('q8_0','q4_k'),'explicit frozen recipe required')
    require(set(parsed['tensors'])==set(plan),'complete candidate tensor map differs')
    for name,row in plan.items():
        actual=parsed['tensors'][name]
        expected=8 if row['quantized']and(recipe=='q8_0'or name=='l.embed_tokens.weight')else 12 if row['quantized']else 0
        require(actual['stored_shape']==row['stored_shape']and actual['type']==expected,'candidate shape/type policy differs: '+name)

def mixed_selection(protocol,plan):
    """A predeclared architectural partition, never inferred from measured scores."""
    policy=protocol.get('mixed_policy')
    if policy is None:return None
    wanted={name:('q4_k'if '.mlp.'in name else 'q8_0')for name,row in plan.items()if row['quantized']}
    require(all(name=='l.embed_tokens.weight'or'.mlp.'in name or'.linear_attn.'in name or'.self_attn.'in name for name in wanted),'unclassified mixed matrix')
    actual={}
    for recipe in ('q8_0','q4_k'):
        names=policy[recipe]['names'];require(names==sorted(set(names)),'unique sorted mixed names required')
        require(hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()==policy[recipe]['names_sha256'],'mixed name digest differs')
        require(len(names)==(153 if recipe=='q8_0'else 96),'mixed153/96 partition differs')
        for name in names:require(name not in actual,'overlapping mixed matrix');actual[name]=recipe
    require(actual==wanted,'fixed attention/embeddingQ8 and FFNQ4 partition differs')
    require(protocol['candidate_order']==['q8_0','q4_k']and protocol['expected_forward_arms']==2,'mixed explicit ordering/two-prefill contract differs')
    require(protocol['max_chunk_elements']==65536 and protocol['exploratory_mechanism_only']and not protocol['held_out_calibration']and not protocol['architecture_quality_selection'],'mixed bounded exploratory contract differs')
    return actual

def discard_clean_file(fd,start=0,length=0):
    require(hasattr(os,'posix_fadvise'),'bounded source cache requires posix_fadvise')
    os.posix_fadvise(fd,start,length,os.POSIX_FADV_DONTNEED)

def digest_with_bounded_cache(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        start=0
        while block:=f.read(4*2**20):
            h.update(block)
            if f.tell()-start>=64*2**20:discard_clean_file(f.fileno(),start,f.tell()-start);start=f.tell()
        discard_clean_file(f.fileno())
    return h.hexdigest()

def bounded_memory(original_memory):
    result=original_memory();available=result['available_bytes'];observed=finite=0
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy,controllers,relative=line.split(':',2)
        if hierarchy=='0'and controllers=='':base=Path('/sys/fs/cgroup');limit_name='memory.max';used_name='memory.current'
        elif 'memory'in controllers.split(','):base=Path('/sys/fs/cgroup/memory');limit_name='memory.limit_in_bytes';used_name='memory.usage_in_bytes'
        else:continue
        target=base/relative.lstrip('/')
        for node in (target,*target.parents):
            if node!=base and base not in node.parents:break
            limit=node/limit_name;used=node/used_name
            if not(limit.is_file()and used.is_file()):continue
            observed+=1;raw=limit.read_text().strip()
            if raw=='max'or int(raw)>=2**60:continue
            finite+=1;maximum=int(raw);current=int(used.read_text().strip());reclaimable=0;stat=node/'memory.stat'
            if stat.is_file():
                fields=dict(row.split()for row in stat.read_text().splitlines())
                reclaimable=max(0,int(fields.get('inactive_file',fields.get('total_inactive_file',0)))-int(fields.get('file_dirty',fields.get('total_dirty',0)))-int(fields.get('file_writeback',fields.get('total_writeback',0))))
            available=min(available,max(0,maximum-current)+reclaimable)
    require(observed>0,'actual cgroup memory controls must be inspectable')
    result.update(available_bytes=available,observed_cgroup_interfaces=observed,finite_cgroup_bounds=finite)
    return result

def bits_hash(tensor,torch):
    require(tensor.device.type=='cpu'and tensor.is_contiguous(),'contiguous CPU tensor required')
    raw=tensor.detach().reshape(-1).view(torch.uint8);h=hashlib.sha256()
    for start in range(0,raw.numel(),2**20):h.update(raw[start:start+2**20].numpy().tobytes())
    return h.hexdigest()
class Oracle:
    def __init__(self,original_root,output,scratch,protocol,memory):
        self.original_root=original_root;self.out=output;self.scratch=scratch;self.p=protocol;self.memory=lambda:bounded_memory(memory);self.called=0
        self.report={'schema':protocol.get('schema','lighton4b-bf16-quant-first-score/v1'),'status':'initializing','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'protocol_sha256':digest(PROTOCOL),'original_source':protocol['original_source'],'original_driver_sha256':protocol['original_driver_sha256'],
            'model_dtype':'bfloat16','f32_model_instantiated':False,'native_forward_executed':False,'generation_capture_repeated':False,
            'max_new_tokens_per_arm':1,'quality_qualified':False,'model_qualified':False,'arms':{},'stage':'initializing'}
        self.out.mkdir(parents=True,exist_ok=False)
    def save(self):
        self.report['memory_latest']=self.memory();temporary=self.out/'oracle.json.partial';temporary.write_text(json.dumps(self.report,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(self.out/'oracle.json')
    def stage(self,name):self.report['stage']=name;self.save();print(json.dumps({'stage':name}),flush=True)
    def model_guard(self,model,params,header,torch,identities=None):
        expected=set(header)-{'__metadata__'}
        require(set(params)==expected|{'lm_head.weight'},'original named-parameter map differs')
        require(model.get_input_embeddings().weight is model.get_output_embeddings().weight and params['lm_head.weight']is params['model.language_model.embed_tokens.weight'],'original tied head identity differs')
        require(len({id(params[k])for k in expected})==723,'source tensor ownership differs')
        for name in expected:
            value=params[name];require(list(value.shape)==header[name]['shape']and value.dtype==torch.bfloat16 and value.device.type=='cpu'and value.is_contiguous(),'source parameter shape/dtype/device differs: '+name)
            if identities is not None:require((id(value),value.data_ptr())==identities[name],'parameter ownership/storage changed')
        require(not model.training and sum(params[k].numel()for k in expected)==self.p['parameter_count'],'original eval/parameter census differs')
    def forward(self,label,model,inputs,processor,torch,np,baseline=None):
        for name,value in inputs.items():
            stream=io.BytesIO();np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
            require(hashlib.sha256(stream.getbuffer()).hexdigest()==self.p['input_sha256'][name],'prepared input changed before arm: '+name)
        self.stage(label+'-one-position');counts={'root':0,'vision':0,'text':0}
        def hook(key):
            def count(module,args):counts[key]+=1
            return count
        handles=[model.register_forward_pre_hook(hook('root')),model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('text'))]
        started=time.monotonic()
        try:
            with torch.inference_mode():result=model.generate(**inputs,max_new_tokens=1,do_sample=False,use_cache=True,return_dict_in_generate=True,output_scores=True)
        finally:
            for handle in handles:handle.remove()
        require(counts=={'root':1,'vision':1,'text':1}and len(result.scores)==1,'one fresh original image/prefill forward required')
        ids=result.sequences[:,inputs['input_ids'].shape[1]:].reshape(-1).tolist();require(len(ids)==1,'exactly one generated position required')
        score=result.scores[0].detach().cpu().float().contiguous().numpy().copy();require(score.shape==(1,248320)and np.isfinite(score).all(),'complete finite original vocabulary required')
        raw=score.astype('<f4').tobytes();(self.out/(label+'.scores.f32')).write_bytes(raw)
        row=score.reshape(-1).astype(np.float64);maximum=float(row.max());prob=np.exp(row-maximum);prob/=prob.sum()
        record={'ids':ids,'decoded_prefix':processor.batch_decode(result.sequences[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0],
            'shape':[1,248320],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'file':label+'.scores.f32','forward_counts':counts,
            'seconds':time.monotonic()-started,'selected_probability':float(prob[ids[0]]),'probability_of_original_id':float(prob[self.p['original_first_id']]),
            'stop_scope':'one-position diagnostic; no EOS/document-completion claim','parameter_dtype':'bfloat16','all_scores_finite':True}
        if baseline is not None:
            a=baseline.reshape(-1).astype(np.float64);error=row-a;na=float(np.linalg.norm(a));nb=float(np.linalg.norm(row));cos=float(a@row/(na*nb))
            record['vs_original_bf16']={'max_abs':float(np.abs(error).max()),'rms':float(np.sqrt(np.mean(error*error))),'cosine':cos,
                'original_abs_max':float(np.abs(a).max()),'candidate_abs_max':float(np.abs(row).max()),'exact_ids':ids==[self.p['original_first_id']],
                'strict_pass':bool(np.abs(error).max()<=self.p['max_abs']and cos>=self.p['min_cos'])}
        self.report['arms'][label]=record;self.save()
        minimal=types.SimpleNamespace(sequences=result.sequences.detach().clone(),scores=(result.scores[0].detach().clone(),))
        del result;gc.collect();return minimal,score
    def run(self,model,inputs,processor):
        import torch,numpy as np,gguf
        require(self.called==0,'one oracle invocation only');self.called+=1
        require(not torch.cuda.is_available()and sys.byteorder=='little','CPU-only little-endian source required')
        require(set(inputs)==set(self.p['input_sha256']),'original prepared input field set differs')
        for name,value in inputs.items():
            stream=io.BytesIO();np.save(stream,value.detach().cpu().numpy(),allow_pickle=False)
            require(hashlib.sha256(stream.getbuffer()).hexdigest()==self.p['input_sha256'][name],'frozen prepared input differs: '+name)
        raw=(self.original_root/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_bytes();require(len(raw)==89456 and hashlib.sha256(raw).hexdigest()==HEADER_SHA,'frozen723-source header differs')
        header=json.loads(raw);plan=tensor_plan(header);selection=mixed_selection(self.p,plan);installed={};params=dict(model.named_parameters(remove_duplicate=False));self.model_guard(model,params,header,torch)
        if selection is not None:self.report.update(exploratory_mechanism_only=True,held_out_calibration=False,mixed_policy=self.p['mixed_policy'],candidate_order=self.p['candidate_order'])
        identities={name:(id(params[name]),params[name].data_ptr())for name in header if name!='__metadata__'}
        self.stage('source-bf16-tensor-census');source_hashes={name:bits_hash(params[name],torch)for name in identities}
        self.report['original_bf16_tensor_sha256']=source_hashes
        if selection is not None:
            sources=list((self.scratch/'source/hf/hub').glob('**/model.safetensors'))
            require(len(sources)==1,'one authenticated task-owned checkpoint cache required')
            with sources[0].open('rb')as f:discard_clean_file(f.fileno())
        baseline_object,baseline=self.forward('original',model,inputs,processor,torch,np)
        require(self.report['arms']['original']['sha256']==self.p['original_first_score_sha256']and self.report['arms']['original']['ids']==[self.p['original_first_id']],'frozen original one-position anchor differs')
        self.report['original_first_score_bit_exact']=True;buffers={name:bits_hash(value,torch)for name,value in model.named_buffers()}
        sys.path.insert(0,str(ROOT/'tools'));from gguf_file_type_canonicalize import parse,span
        if selection is not None:
            # This is the task-owned original driver's HF cache. Advise clean source
            # pages away; never unlink or alter the authenticated original checkpoint.
            with sources[0].open('rb')as f:discard_clean_file(f.fileno())
            self.report['clean_cache_advice_enabled']=True
        order=self.p.get('candidate_order',list(self.p['candidates']))
        for recipe in order:
            pin=self.p['candidates'][recipe]
            self.stage(recipe+'-download');path=self.scratch/pin['file'];require(not path.exists(),'fresh candidate path required')
            require(__import__('shutil').disk_usage(self.scratch).free>=pin['bytes']+2*2**30,'candidate disk headroom')
            with urllib.request.urlopen(f"https://huggingface.co/{self.p['candidate_repository']}/resolve/{self.p['candidate_revision']}/{pin['file']}",timeout=120)as response,path.open('xb')as f:
                count=0;advised=0
                while block:=response.read(4*2**20):
                    count+=len(block);require(count<=pin['bytes'],'candidate transfer size bound');f.write(block)
                    if selection is not None and count-advised>=64*2**20:
                        f.flush();os.fdatasync(f.fileno());discard_clean_file(f.fileno(),advised,count-advised);advised=count
                f.flush();os.fsync(f.fileno())
                if selection is not None:discard_clean_file(f.fileno())
            require(count==pin['bytes']and (digest_with_bounded_cache(path)if selection is not None else digest(path))==pin['sha256'],'immutable full candidate hash differs')
            parsed=parse(path);validate_inventory(parsed,plan,recipe)
            expected_meta={'general.architecture':'qwen35_ocr','general.source.huggingface.repository':self.p['checkpoint_repository'],
                'general.source.huggingface.revision':self.p['checkpoint_revision'],'qwen35_ocr.profile':'lightonocr3-4b','qwen35_ocr.format_version':5,
                'qwen35_ocr.checkpoint_sha256':self.p['checkpoint_sha256'],'qwen35_ocr.tokenizer_sha256':'87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4','qwen35_ocr.primary_eos_id':248046,'qwen35_ocr.source_text_eos_id':248044,'qwen35_ocr.source_top_level_eos_id':248046,
                'qwen35_ocr.tie_word_embeddings':1,'qwen35_ocr.zero_centered_text_norm':1}
            require(all(k in parsed['metadata']and parsed['metadata'][k]['value']==v for k,v in expected_meta.items()),'candidate source/profile/EOS/tie metadata differs')
            self.stage(recipe+'-stream-bf16-weights');reader=gguf.GGUFReader(str(path));tensors={t.name:t for t in reader.tensors}
            require(len(reader.tensors)==len(tensors)==723 and set(tensors)==set(plan),'independent reader723-name census differs')
            records=[];quant_count=protected_count=0
            self.report[recipe+'-weights']={'candidate_sha256':pin['sha256'],'status':'streaming','tensors':records}
            with path.open('rb')as raw_file,torch.no_grad():
                for index,(name,row)in enumerate(sorted(plan.items())):
                    tensor=tensors[name];param=params[row['source']];record={**row,'gguf':name,'type':int(tensor.tensor_type),'raw_bytes':parsed['tensors'][name]['bytes'],
                        'raw_sha256':span(raw_file,parsed['tensors'][name]['data_offset'],parsed['tensors'][name]['bytes'])}
                    require(tensor.shape.tolist()==row['stored_shape'],'independent reader shape differs')
                    if row['quantized']:
                        quant_count+=1
                        if selection is not None and selection[name]!=recipe:
                            record.update(selected_for_mixed=False);records.append(record);discard_clean_file(raw_file.fileno(),parsed['tensors'][name]['data_offset'],parsed['tensors'][name]['bytes']);continue
                        rows,columns=row['source_shape'];per=max(1,self.p['max_chunk_elements']//columns);expected_hash=hashlib.sha256();round_error=0.0
                        for start in range(0,rows,per):
                            stop=min(rows,start+per);chunk=np.ascontiguousarray(gguf.quants.dequantize(tensor.data[start:stop],tensor.tensor_type)).reshape(stop-start,columns)
                            require(chunk.dtype==np.float32 and chunk.size<=self.p['max_chunk_elements']and np.isfinite(chunk).all(),'bounded independent dequantization differs')
                            bf16=torch.from_numpy(chunk).to(dtype=torch.bfloat16);require(bool(torch.isfinite(bf16).all()),'BF16 parameter conversion nonfinite')
                            expected_hash.update(bf16.view(torch.uint8).numpy().tobytes())
                            round_error=max(round_error,float(np.abs(chunk-bf16.float().numpy()).max()))
                            param[start:stop].copy_(bf16)
                            require(param[start:stop].view(torch.uint8).numpy().tobytes()==bf16.view(torch.uint8).numpy().tobytes(),'streamed BF16 parameter chunk readback differs')
                        actual_hash=bits_hash(param,torch);require(actual_hash==expected_hash.hexdigest(),'full BF16 parameter readback differs')
                        record.update(bf16_sha256=actual_hash,readback_exact=True,dequant_to_bf16_max_abs=round_error,changed_from_original=actual_hash!=source_hashes[row['source']])
                        if selection is not None:record.update(selected_for_mixed=True);require(name not in installed,'mixed parameter installed twice');installed[name]=actual_hash
                    else:
                        protected_count+=1;data=tensor.data.reshape(-1);actual=param.reshape(-1)
                        require(int(tensor.tensor_type)==0 and data.dtype==np.float32,'protected F32 GGUF tensor required')
                        for start in range(0,actual.numel(),self.p['max_chunk_elements']):
                            original=actual[start:start+self.p['max_chunk_elements']].float().numpy()
                            require(original.tobytes()==data[start:start+original.size].tobytes(),'protected tensor differs from original BF16 source: '+name)
                        require(bits_hash(param,torch)==source_hashes[row['source']],'protected original tensor changed')
                        record.update(bf16_sha256=source_hashes[row['source']],protected_original_exact=True)
                    records.append(record)
                    if selection is not None:
                        discard_clean_file(raw_file.fileno(),parsed['tensors'][name]['data_offset'],parsed['tensors'][name]['bytes'])
                        require(self.memory()['available_bytes']>=4*2**30,'mixed streaming headroom below4GiB')
                    if index%32==0:self.report['stream_progress']={'recipe':recipe,'complete_tensors':index+1};self.save()
            require(quant_count==249 and protected_count==474,'quant/protected249/474 census differs')
            self.model_guard(model,dict(model.named_parameters(remove_duplicate=False)),header,torch,identities)
            require({name:bits_hash(value,torch)for name,value in model.named_buffers()}==buffers,'original nonparameter buffers changed before candidate')
            self.report[recipe+'-weights']={'candidate_sha256':pin['sha256'],'quantized_tensors':quant_count,'protected_tensors':protected_count,
                'changed_quantized_tensors':sum(r.get('changed_from_original',False)for r in records),'parameter_dtype':'bfloat16','tied_head_preserved':True,
                'max_dequant_chunk_elements':self.p['max_chunk_elements'],'mapping_and_readback_verified':True,'status':'verified','tensors':records}
            if selection is not None:self.report[recipe+'-weights']['selected_mixed_tensors']=sum(r.get('selected_for_mixed',False)for r in records)
            del reader,tensors,tensor,data,actual,original,chunk,bf16;gc.collect()
            with path.open('rb')as f:
                if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
            path.unlink()
            if selection is not None:
                with sources[0].open('rb')as f:discard_clean_file(f.fileno())
                self.report['cache_trim_memory_after_'+recipe]=self.memory()
            if selection is not None and recipe!=order[-1]:continue
            if selection is not None:
                require(set(installed)==set(selection),'complete249 mixed parameter coverage differs')
                require(all(bits_hash(params[plan[name]['source']],torch)==sha for name,sha in installed.items()),'final mixed whole-parameter hash differs')
                self.report['mixed_final_bf16_parameter_sha256']=installed
            require(self.memory()['available_bytes']>=(8 if selection is not None else 6)*2**30,'resident BF16 model forward headroom below frozen minimum')
            if selection is not None:
                margin=self.memory().get('cgroup_hard_margin_bytes');require(margin is not None and margin>=2*2**30,'mixed actual cgroup hard margin below2GiB')
            _,candidate=self.forward('mixed'if selection is not None else recipe,model,inputs,processor,torch,np,baseline)
            self.model_guard(model,dict(model.named_parameters(remove_duplicate=False)),header,torch,identities)
            for name,row in plan.items():
                if not row['quantized']:require(bits_hash(params[row['source']],torch)==source_hashes[row['source']],'protected source tensor changed during forward')
            require({name:bits_hash(value,torch)for name,value in model.named_buffers()}==buffers,'original buffers changed during candidate')
            del candidate,_;gc.collect();self.save()
        self.report.update(status='compared',comparison_completed=True,total_one_position_forwards=self.p['expected_forward_arms'],
            numerical_rejection=any(not self.report['arms'][r]['vs_original_bf16']['strict_pass']for r in (['mixed']if selection is not None else self.p['candidates'])))
        self.stage('complete');return baseline_object
class ReplaceGenerate(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and node.value.func.attr=='generate':
            require(isinstance(node.value.func.value,ast.Name)and node.value.func.value.id=='model','one frozen original owner required')
            self.count+=1;node.value=ast.parse('oracle.run(model, inputs, processor)',mode='eval').body
        return self.generic_visit(node)
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--original-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True);a=parser.parse_args(argv)
    require((Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir())or os.environ.get('GITHUB_ACTIONS')=='true','hosted original CPU worker only')
    protocol=json.loads(PROTOCOL.read_text());original=a.original_root.resolve();a.scratch.mkdir(parents=True,exist_ok=False)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()==protocol['original_source'],'frozen original source revision differs')
    driver=original/'tools/lighton4b_bf16_freegen.py';require(digest(driver)==protocol['original_driver_sha256']and digest(original/'docs/lighton4b-bf16-freegen-v1.json')==protocol['original_protocol_sha256'],'frozen original driver/protocol differs')
    tree=ast.parse(driver.read_text());transform=ReplaceGenerate();tree=transform.visit(tree);ast.fix_missing_locations(tree);require(transform.count==1,'one frozen generate expression required')
    namespace={'__name__':'quant_first_score_original_driver','__file__':str(driver)}
    # The original checkout and every original source guard remain untouched.
    exec(compile(tree,str(driver),'exec'),namespace)
    oracle=Oracle(original,a.output/'quant-oracle',a.scratch,protocol,namespace['memory']);namespace['oracle']=oracle
    try:
        code=namespace['main'](['--source-sha',protocol['original_source'],'--output',str(a.output/'original-adapter-audit'),'--scratch',str(a.scratch/'source')])
        audit_path=a.output/'original-adapter-audit/capture.json'
        require(audit_path.is_file(),'frozen original driver audit absent')
        audit=json.loads(audit_path.read_text())
        oracle.report['original_adapter_audit_sha256']=digest(audit_path)
        oracle.report['original_driver_returncode']=code
        oracle.report['original_driver_status']=audit.get('status')
        for source,target in (('checkpoint','verified_original_checkpoint'),('packages','runtime_packages'),('upstream','original_module_source'),
                              ('measured_memory_peaks','measured_memory_peaks'),('prepared_inputs','prepared_inputs')):
            if source in audit:oracle.report[target]=audit[source]
        oracle.save()
        require(code==0 and oracle.report.get('comparison_completed'),'one-position source/candidate capture did not complete')
        require(audit['checkpoint']['payload_verified']and audit['checkpoint']['sha256']==protocol['checkpoint_sha256']and audit['checkpoint']['bytes']==protocol['checkpoint_bytes'],'full original checkpoint authentication differs')
        expected_calls=oracle.report.get('total_model_forward_calls',oracle.report.get('total_one_position_forwards'))
        require(audit['model_forward_calls']==expected_calls and(expected_calls==protocol['expected_forward_arms']or protocol.get('full_free_generation')is True),'exact frozen original forward count differs')
        require(len(audit['scores'])==1 and audit['scores'][0]['sha256']==protocol['original_first_score_sha256'],'adapter audit must persist baseline first score only')
        if oracle.report['numerical_rejection']:return 1
        return 0
    except Exception as error:
        oracle.report.update(error_type=type(error).__name__,error=str(error)[:240]if isinstance(error,ValueError)else 'bounded BF16 quant original failed');oracle.save();raise RuntimeError(oracle.report['error'])from None
    finally:oracle.save()
if __name__=='__main__':raise SystemExit(main())
