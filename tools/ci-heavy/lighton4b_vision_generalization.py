#!/usr/bin/env python3
"""One hosted vision-only diagnostic on four frozen cases; no decoder or gold rewrite."""
import hashlib, importlib.util, json, os, shutil, struct, subprocess, sys, time, urllib.request
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools/ci-heavy'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'};os.environ.update(dispatch)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'vision-generalization';scratch.mkdir()
 report={'schema':'lighton4b-vision-generalization/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dispatch':dispatch,'decoder_executed':False,'generation_executed':False,'quality_qualified':False,'cases':{},'limits':['New four-case source stages are diagnostics, not replacement frozen gold.','Historical four-case CPU dispatch was not recorded; current source uses qualified AVX2 math.','Checkpoint vision range is verified internally; full 9GB checkpoint and GGUF payload are not fetched.']}
 try:
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())};report['preflight']={'available_bytes':mem['MemAvailable'],'swap_total':mem['SwapTotal'],'swap_free':mem['SwapFree'],'free_bytes':shutil.disk_usage(scratch).free,'load':os.getloadavg()}
  require(mem['MemAvailable']>=10*1024**3 and report['preflight']['free_bytes']>=16*1024**3,'capacity guard')
  import numpy as np, torch
  from PIL import Image
  require(version('kaggle')=='2.2.2' and version('six')=='1.17.0','retained transport dependency pins')
  from kaggle.api.kaggle_api_extended import KaggleApi
  import transformers.models.qwen3_5.modeling_qwen3_5 as source
  from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
  require(torch.__version__=='2.13.0+cpu' and torch.version.git_version=='cf30153c4c131c8164ee7798e5022d810682e2cb' and np.__version__=='2.3.5' and __import__('transformers').__version__=='5.5.4' and __import__('PIL').__version__=='11.3.0','producer package pins')
  require(sha(source.__file__)=='aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048','actual source class differs')
  torch.set_num_threads(4);torch.set_num_interop_threads(1)
  report['packages']={'torch':torch.__version__,'torch_git':torch.version.git_version,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'source_sha256':sha(source.__file__)}
  transport=module('lighton4b_calibration_native');worker=module('lighton4b_frozen_case');compare=module('lighton4b_vision_bf16_compare');gates=module('lighton4b_rotary_q8_smoke').VISION_GATES
  os.environ.update({g:'1' for g in (*gates,'CRISPEMBED_QWEN35_OCR','CRISPEMBED_LIGHTON4B_VISION_OWNER','CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE')});os.environ.update({'CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2':'0','CRISPEMBED_LIGHTON4B_BF16_LINEAR':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0'})
  report['gates']={k:v for k,v in os.environ.items() if k.startswith('CRISPEMBED_LIGHTON4B') or k=='CRISPEMBED_QWEN35_WEIGHT_CACHE'}
  # Public immutable checkpoint range; suppress signed redirect/error text.
  base='https://huggingface.co/lightonai/LightOnOCR-3-4B/resolve/b71010f095dc3735de7ee6b9969bff8f8f50b39a/model.safetensors'
  def ranged(start,end,path):
   try:
    req=urllib.request.Request(base+'?range_start='+str(start),headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=180) as r,path.open('xb') as f:
     require(r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/9078620536','source range identity')
     count=0
     while b:=r.read(min(2**20,end-start+2-count)):
      count+=len(b);require(count<=end-start+1,'source range bound');f.write(b)
    require(count==end-start+1,'source range short read')
   except ValueError:raise
   except Exception:raise ValueError('checkpoint range transfer failed') from None
  header=scratch/'header';ranged(0,89463,header);raw=header.read_bytes();require(struct.unpack('<Q',raw[:8])[0]==89456 and hashlib.sha256(raw[8:]).hexdigest()=='a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081','header pin')
  index=json.loads(raw[8:]);selected={k:v for k,v in index.items() if k.startswith('model.visual.')};require(len(selected)==297 and all(v['dtype']=='BF16' for v in selected.values()),'vision inventory')
  lo=min(v['data_offsets'][0] for v in selected.values());hi=max(v['data_offsets'][1] for v in selected.values());require((lo,hi)==(8411502592,9078531072),'vision range')
  payload=scratch/'vision.bf16';ranged(lo+89464,hi+89463,payload);report['checkpoint']={'revision':'b71010f095dc3735de7ee6b9969bff8f8f50b39a','vision_bytes':payload.stat().st_size,'vision_range_sha256':sha(payload),'header_json_sha256':hashlib.sha256(raw[8:]).hexdigest()}
  configpath=ROOT/'tests/fixtures/lightonocr3-4b-source/config.json';require(sha(configpath)=='99b4c363a998c2d8c5ed5b3f3a40ec5cb97a156034df5002d0d789c74324f5f0','model geometry config')
  cfg=Qwen3_5VisionConfig(**json.loads(configpath.read_text())['vision_config']);cfg._attn_implementation='eager'
  torch.set_default_dtype(torch.bfloat16)
  try:model=source.Qwen3_5VisionModel(cfg).eval()
  finally:torch.set_default_dtype(torch.float32)
  require(model.rotary_pos_emb.inv_freq.dtype==torch.float32,'rotary buffer must remain F32')
  raw=payload.read_bytes();state={}
  for k,v in selected.items():
   a,b=v['data_offsets'];state[k.removeprefix('model.visual.')]=torch.frombuffer(bytearray(raw[a-lo:b-lo]),dtype=torch.bfloat16).reshape(v['shape'])
  model.load_state_dict(state,strict=True)
  weightroot=scratch/'weights';weightroot.mkdir();params={};offset=0
  with (weightroot/'weights.f32').open('wb') as f,(weightroot/'weights.txt').open('w') as meta:
   for name,t in model.named_parameters():
    original=state[name];require(torch.equal(t.view(torch.int16),original.view(torch.int16)),'parameter transfer changed bytes')
    params[name]=hashlib.sha256(t.detach().view(torch.uint8).numpy().tobytes()).hexdigest();x=t.detach().float().numpy();b=x.tobytes();rows=x.shape[0] if x.ndim>1 else 1;cols=x.size//rows
    meta.write(f'v.{name} {cols} {rows} {offset} {len(b)}\n');f.write(b);offset+=len(b)
  require(len(params)==297 and sum(t.numel()*2 for t in model.parameters())==667028480,'loaded parameter inventory');(out/'parameter-hashes.json').write_text(json.dumps(params,indent=2));del state,raw
  # Compile ordinary production path plus observer-only copy; no product source changes.
  code=(ROOT/'src/lighton4b_vision_bf16.cpp').read_text();needle='const bool capture = layer_index == 5 || layer_index == 11;';require(code.count(needle)==1,'observer source predicate changed');(scratch/'lighton4b_vision_observed.inc').write_text(code.replace(needle,'const bool capture = true;'))
  build=scratch/'build'
  def run(label,args):
   print('phase:',label,flush=True)
   report['phase']=label
   with (out/(label+'.log')).open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DLIGHTON4B_VISION_GENERALIZATION_DIR='+str(scratch)])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-generalization','--parallel','2'])
  # Form control: standalone real source class must reproduce old source hashes.
  provider=KaggleApi();provider.authenticate();original=scratch/'form-original';slug=os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill'
  provider.kernels_output(slug,str(original),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|processor\.patches\.f32)$',quiet=True)
  original=original/'lighton4b-bf16-prefill/stages';require(sha(original/'prefill.json')==compare.ORIGINAL_REPORT_SHA,'form source control manifest');manifest=json.loads((original/'prefill.json').read_text());require(manifest['observer_first_score_bit_exact'] and manifest['source_sha']==compare.SOURCE,'form control provenance')
  current={};handles=[]
  def save(name,value):
   value=value.detach().float().cpu().contiguous();value.numpy().tofile(current['source']/(name+'.f32'))
  def hook(name):return lambda mod,args,value:save(name,value)
  handles.append(model.patch_embed.register_forward_hook(hook('vision.patch')))
  interpolate=model.fast_pos_embed_interpolate
  def position(grid):
   value=interpolate(grid);save('vision.position',value);return value
  model.fast_pos_embed_interpolate=position
  def pre(mod,args,kwargs):
   save('vision.embedded',args[0]);save('vision.rotary_cos',kwargs['position_embeddings'][0]);save('vision.rotary_sin',kwargs['position_embeddings'][1])
  handles.append(model.blocks[0].register_forward_pre_hook(pre,with_kwargs=True))
  for i,block in enumerate(model.blocks):
   handles.append(block.register_forward_hook(hook(f'vision.layer.{i}')))
   # Keep first-divergence source operators on scratch; publish only selected layer.
   for name,component in [('norm1',block.norm1),('qkv',block.attn.qkv),('attention',block.attn.proj),('norm2',block.norm2),('up',block.mlp.linear_fc1),('activation',block.mlp.act_fn),('down',block.mlp.linear_fc2)]:
    handles.append(component.register_forward_hook(hook(f'probe.{i}.{name}')))
   handles.append(block.attn.proj.register_forward_pre_hook(lambda mod,args,i=i:save(f'probe.{i}.pooled',args[0])))
  for name,component in [('norm',model.merger.norm),('up',model.merger.linear_fc1),('activation',model.merger.act_fn)]:handles.append(component.register_forward_hook(hook('vision.merger.'+name)))
  handles.append(model.merger.register_forward_hook(hook('vision.merged')))
  def reference(folder,pixels,grid):
   current['source']=folder;folder.mkdir(parents=True);np.asarray(pixels,dtype='<f4').tofile(folder/'processor.patches.f32')
   with torch.inference_mode():model(torch.from_numpy(pixels.copy()),grid_thw=torch.tensor([grid],dtype=torch.int64))
  control=scratch/'form-source';reference(control,np.fromfile(original/'processor.patches.f32',dtype='<f4').reshape(560,1536),[1,20,28]);controls={name:sha(control/row['file'])==row['sha256'] for name,row in manifest['stages'].items() if name.startswith('vision.') or name=='processor.patches'}
  report['form_source_control']=controls;require(len(controls)==34 and all(controls.values()),'standalone source failed retained full vision control')
  urls=transport.original_urls();prefix='lighton4b-fourcase-bf16/'
  def get(member,path,digest,bound=8*1024**2):transport.bounded_download(urls[prefix+member],path,digest,bound)
  batchpath=out/'original-batch.json';get('batch.json',batchpath,transport.ORIGINAL['batch_sha256']);batch=json.loads(batchpath.read_text());require(batch['source_sha']==worker.ORIGINAL_SOURCE and batch['protocol_sha256']==worker.PROTOCOL_SHA and batch['input_manifest_sha256']==worker.FROZEN_SHA and batch['model_dtype']=='bfloat16','fourcase batch source')
  cases=json.loads((ROOT/'tests/regression/lighton4b-mixed-v1/frozen-inputs.json').read_text())['cases']
  for record in cases:
   caseid=record['id'];_,contract,_=worker.load_contract(caseid);folder=out/caseid;folder.mkdir();capturepath=folder/'original-capture.json';get(caseid+'/capture.json',capturepath,batch['cases'][caseid]['capture_sha256']);capture=json.loads(capturepath.read_text());worker.validate_original(capture,contract)
   prepared=scratch/caseid;prepared.mkdir()
   for field,row in capture['prepared_inputs'].items():
    require(Path(row['file']).name==row['file'],'prepared path');get(caseid+'/'+row['file'],prepared/row['file'],row['sha256'])
   pixels=np.load(prepared/capture['prepared_inputs']['pixel_values']['file'],allow_pickle=False);grid=np.load(prepared/capture['prepared_inputs']['image_grid_thw']['file'],allow_pickle=False).reshape(-1).tolist();require(grid==record['expected_grid'],'frozen grid')
   png=ROOT/'tests/regression/lighton4b-mixed-v1'/record['png_file'];require(sha(png)==record['png_sha256'],'frozen PNG');rgb=Image.open(png).convert('RGB').tobytes();require(hashlib.sha256(rgb).hexdigest()==record['rgb_sha256'],'frozen RGB');(folder/'input.rgb').write_bytes(rgb)
   reference(folder/'source',pixels,grid);(folder/'native').mkdir();started=time.monotonic();run(caseid,[build/'lighton4b-vision-generalization',weightroot,folder,record['height'],record['width'],grid[1],grid[2],4,'native'])
   summary=json.loads((folder/'native/native-summary.json').read_text());result={'native':summary,'elapsed_seconds':time.monotonic()-started,'grid':grid,'source_capture_sha256':sha(capturepath),'original_eos_completed':capture['eos_completed'],'original_cap_hit':capture['cap_hit'],'stages':{}};report['cases'][caseid]=result
   def compare_files(a,b):
    row=compare.compare(np.fromfile(a,dtype='<f4'),np.fromfile(b,dtype='<f4'));row.update(source_sha256=sha(a),native_sha256=sha(b),byte_exact=sha(a)==sha(b));return row
   for p in sorted((folder/'native').glob('*.f32')):result['stages'][p.stem]=compare_files(folder/'source'/p.name,p)
   if summary['first_differing_stage']=='processor.patches':
    (folder/'source-input').mkdir();run(caseid+'-source-input',[build/'lighton4b-vision-generalization',weightroot,folder,record['height'],record['width'],grid[1],grid[2],4,'source-input']);result['source_input_diagnostic']=json.loads((folder/'source-input/native-summary.json').read_text());result['source_input_stages']={p.stem:compare_files(folder/'source'/p.name,p) for p in sorted((folder/'source-input').glob('*.f32'))}
   result['rotary_table_exact']=all(result['stages'][n]['byte_exact'] for n in ('vision.rotary_cos','vision.rotary_sin'))
   # No need to retain every source block operator: preserve the first differing layers only.
   layers={summary['first_differing_layer'],result.get('source_input_diagnostic',{}).get('first_differing_layer',-1)}
   for p in (folder/'source').glob('probe.*.f32'):
    if int(p.name.split('.')[1]) not in layers:p.unlink()
   (out/'vision-generalization.json').write_text(json.dumps(report,indent=2))
  report['status']='passed';report['all_four_rotary_exact']=all(c['rotary_table_exact'] for c in report['cases'].values());report['all_four_native_vision_exact']=all(all(x['byte_exact'] for x in c['stages'].values()) for c in report['cases'].values())
 except Exception as e:
  report['failure_type']=type(e).__name__;report['failure']='bounded source/native vision diagnostic failed; inspect phase logs and preserved control metrics';raise
 finally:
  (out/'vision-generalization.json').write_text(json.dumps(report,indent=2));(out/'summary.md').write_text('Vision-only four-case diagnostic: '+report['status']+'. No decoder, generation or quality qualification.\n')
if __name__=='__main__':main()
