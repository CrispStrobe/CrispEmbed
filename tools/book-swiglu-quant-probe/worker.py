#!/usr/bin/env python3
"""Reference-only same-input replay for the frozen second image tile."""
import hashlib, importlib.util, json, os, platform, shutil, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('swiglu_baseline',ROOT/'tools/swiglu-source-probe/worker.py')
audit=load('book_audit',ROOT/'tools/book-stage-compare/audit.py')
require,sha=audit.require,audit.sha

def main():
 import numpy as np
 c=json.loads((HERE/'protocol.json').read_text());out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'quant-replay';work.mkdir();start=time.monotonic();deadline=base.Deadline(1800);deadline.__enter__()
 r=dict(schema=c['schema'],status='failed',commands={},model_forward_executed=False,decoded_generation_executed=False,quality_qualified=False,original_forward_executed=False)
 def save():(out/'book-quant-replay.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,limit=600):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:base.command(args,out/(label+'.log'),min(limit,1800-(time.monotonic()-start)),r['commands'][label],env=env,cwd=ROOT)
  finally:save()
 def locate(name,digest):
  matches=[p for p in Path('/kaggle/input').rglob(name) if p.is_file() and p.stat().st_size<4<<20 and sha(p)==digest];require(len(matches)==1,'one authenticated mounted receipt '+name);p=matches[0];return p,json.loads(p.read_text())
 def compare(a,b):return base.stats(np,np.fromfile(a,dtype='<f4'),np.fromfile(b,dtype='<f4'))
 env=dict(os.environ);env.update({k:'0' for k in env if k.startswith('CRISPEMBED_')});env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 try:
  require(np.__version__=='2.3.5' and platform.system()=='Linux' and sys.byteorder=='little' and Path('/kaggle/input').is_dir() and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU worker')
  require(c['tokens']==16 and c['image_rows']==[16,32] and not c['model_forward_allowed'] and not c['decoded_generation_allowed'],'frozen operator scope')
  r['source_sha']=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'controller source')
  r['capacity']=dict(memory=base.base.memory(),scratch_free=shutil.disk_usage(work).free);require(r['capacity']['memory']['MemAvailable']>=4<<30 and r['capacity']['scratch_free']>=6<<30,'operator resources')
  for name,digest in c['adapter_hashes'].items():require(sha(HERE/name)==digest,'closed reference adapter')
  npth,native=locate('image-layer0-probe.json',c['native_receipt_sha256']);spth,source=locate('book-text-trace.json',c['source_receipt_sha256'])
  require(native['status']=='measured_cache_closed_image_layer0_trace' and native['capture']['offset']==20 and native['capture']['cache_before_tokens']==20 and native['capture']['cache_after_tokens']==36 and native['all_historical_prefill_vision_stage_hashes_exact']==82 and all(native['capture'][key] for key in ['all_layer0_tiles_hidden_exact','final_layer0_cache_exact','previous_caches_immutable','core_replay_attn_state_conv_exact']),'actual native cache/stage closure')
  require(source['status']=='measured_neutral_book_text_trace' and source['all654_scores_observer_neutral'],'actual source neutrality')
  packpath=audit.owned(npth.parent,'physical-pack.json',native['artifacts']);require(sha(packpath)==c['pack_receipt']['sha256'],'same unchanged physical pack');pack=json.loads(packpath.read_text())
  for row in c['weights']:
   tensors=[x for x in pack['tensors'] if x.get('physical_name')==row['physical_name']];require(len(tensors)==1 and all(tensors[0][k]==row[k] for k in ['source_offset','bytes','type','source_shape','source_packed_sha256','sha256','whole_rows_permuted','quant_blocks_unchanged']),'unchanged physical quantized coordinates')
  graphs,_=audit.read_graphs(spth.parent,source);inputs=out/'inputs';inputs.mkdir();actual={}
  for role,width,name in [('gate',9216,'ffn_gate-0'),('up',9216,'ffn_up-0'),('activation',9216,'ffn_swiglu-0'),('down',2560,'ffn_out-0')]:
   nativepath=audit.owned(npth.parent,'native/image.layer0.'+role+'.f32',native['artifacts']);x=np.fromfile(nativepath,dtype='<f4').reshape(16,width);y=audit.source_rows(graphs[1],name,width,252)[16:32];require(np.isfinite(x).all(),'finite actual native inputs');x.tofile(inputs/('native-'+role+'.f32'));y.astype('<f4').tofile(inputs/('source-'+role+'.f32'));actual[role]=audit.delta(x,y)
  require(actual['gate']['bit_exact'] and actual['up']['bit_exact'],'same gate/up before activation attribution');r['actual_comparisons']=actual
  helper=work/'helper';helper.mkdir()
  run('historical-fetch',['git','fetch','--depth','1','origin',c['historical_source']])
  for name,digest in c['helper_hashes'].items():
   data=subprocess.check_output(['git','-C',ROOT,'show',c['historical_source']+':src/'+name]);require(hashlib.sha256(data).hexdigest()==digest,'historical helper exact bytes');p=helper/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
  upstream=work/'upstream'
  for label,args in [('up-init',['git','init',upstream]),('up-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',upstream,'fetch','--depth','1','origin',c['upstream_commit']]),('up-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==c['upstream_commit'],'pinned upstream')
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==c['native_ggml_commit'],'pinned native GGML');bins={}
  for kind,ggml in [('native',ROOT/'ggml'),('upstream',upstream/'ggml')]:
   build=work/(kind+'-build');run(kind+'-configure',['cmake','-S',HERE,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(ggml),'-DHISTORICAL_HELPER_DIR='+str(helper),*[f'-D{k}={v}' for k,v in base.PROFILE.items()]]);cache=(build/'CMakeCache.txt').read_text();require(all(f'{k}:BOOL={v}' in cache for k,v in base.PROFILE.items()),'matched CPU profile');shutil.copyfile(build/'CMakeCache.txt',out/(kind+'-CMakeCache.txt'));run(kind+'-build',['cmake','--build',build,'--target','swiglu-source-probe','--parallel','2']);bins[kind]=build/'swiglu-source-probe';run(kind+'-controls',[bins[kind],'--controls'],60)
  results=out/'results';results.mkdir();r['activation']={}
  for variant,kind,mode in [('scalar','native','scalar'),('native-fused','native','ggml'),('source-fused','upstream','ggml')]:
   folder=results/variant;folder.mkdir();shutil.copyfile(inputs/'native-gate.f32',folder/'gate.f32');shutil.copyfile(inputs/'native-up.f32',folder/'up.f32');run(variant+'-activation',[bins[kind],mode,'activation',16,folder,work,folder/'activation.f32']);r['activation'][variant]=dict(vs_native=compare(folder/'activation.f32',inputs/'native-activation.f32'),vs_source=compare(folder/'activation.f32',inputs/'source-activation.f32'))
  require(r['activation']['scalar']['vs_native']['bit_exact'] and all(r['activation'][k]['vs_source']['bit_exact'] for k in ['native-fused','source-fused']),'actual scalar/fused whole activation closures')
  weights=work/'weights';weights.mkdir();row=c['weights'][0];offset=row['source_offset'];size=row['bytes'];request=urllib.request.Request(c['source_url'],headers={'Range':f'bytes={offset}-{offset+size-1}'});target=weights/'down.quant'
  with urllib.request.urlopen(request,timeout=120) as response,target.open('xb') as stream:
   require(response.status==206 and response.headers.get('Content-Range')==f'bytes {offset}-{offset+size-1}/{c["source_bytes"]}','exact whole tensor range');count=0
   while chunk:=response.read(1<<20):count+=len(chunk);require(count<=size,'bounded quantized parameter transfer');stream.write(chunk)
  require(target.stat().st_size==size and sha(target)==row['source_packed_sha256'],'unchanged full Q6 tensor');r['parameter_bytes']=size
  r['closures']={};r['outputs']={}
  for variant in ['scalar','native-fused','source-fused']:
   folder=results/variant;r['outputs'][variant]={}
   for kind in ['native','upstream']:
    for mode in ['codes','raw','prequantized','tiled','tiled-prequantized']:
     path=folder/(kind+'-'+mode+('.bin' if mode=='codes' else '.f32'));run(variant+'-'+kind+'-'+mode,[bins[kind],mode,'down',16,folder,weights,path])
    for a,b in [('tiled','tiled-prequantized'),('raw','prequantized')]:r['closures'][variant+'-'+kind+'-'+a+'-forced']=compare(folder/(kind+'-'+a+'.f32'),folder/(kind+'-'+b+'.f32'))
    for mode in ['raw','tiled']:r['outputs'][variant][kind+'-'+mode]=dict(vs_native=compare(folder/(kind+'-'+mode+'.f32'),inputs/'native-down.f32'),vs_source=compare(folder/(kind+'-'+mode+'.f32'),inputs/'source-down.f32'))
   save()
  r['required_actual_closures']=dict(native_scalar_tiled=r['outputs']['scalar']['native-tiled']['vs_native']['bit_exact'],upstream_fused_raw=r['outputs']['source-fused']['upstream-raw']['vs_source']['bit_exact'],native_fused_tiled=r['outputs']['source-fused']['native-tiled']['vs_source']['bit_exact'])
  require(all(r['required_actual_closures'].values()) and all(x['bit_exact'] for k,x in r['closures'].items() if 'tiled' in k or '-upstream-raw-' in k),'actual whole down and forced-code closure required before attribution')
  dtype=np.dtype([('d','<f4'),('qs','i1',(256,)),('bsums','<i2',(16,))]);r['code_differences']={};r['row_output_differences']={}
  for kind in ['native','upstream']:
   a=np.fromfile(results/'scalar'/(kind+'-codes.bin'),dtype=dtype).reshape(16,36);b=np.fromfile(results/'source-fused'/(kind+'-codes.bin'),dtype=dtype).reshape(16,36);require(np.isfinite(a['d']).all() and np.isfinite(b['d']).all(),'finite actual converter scales')
   r['code_differences'][kind]=dict(scope='Public Q8_K converter intervention; forced-output closure required; no captured internal buffer or vocabulary IDs.',different_scales=int(np.count_nonzero(a['d'].view('<u4')!=b['d'].view('<u4'))),different_qs=int(np.count_nonzero(a['qs']!=b['qs'])),different_bsums=int(np.count_nonzero(a['bsums']!=b['bsums'])),changed_scale_coordinates=np.argwhere(a['d'].view('<u4')!=b['d'].view('<u4')).tolist(),changed_qs_coordinates=np.argwhere(a['qs']!=b['qs']).tolist(),changed_bsums_coordinates=np.argwhere(a['bsums']!=b['bsums']).tolist())
   for mode in ['raw','tiled']:
    x=np.fromfile(results/'scalar'/(kind+'-'+mode+'.f32'),dtype='<f4').reshape(16,2560);y=np.fromfile(results/'source-fused'/(kind+'-'+mode+'.f32'),dtype='<f4').reshape(16,2560);r['row_output_differences'][kind+'-'+mode]=[dict(local_row=i,image_row=16+i,**audit.delta(x[i],y[i])) for i in range(16)]
  r['status']='measured_closed_second_image_swiglu_quant_replay';r['limits']=['Frozen second image tile only, no runtime change or decoded quality claim.','Codes are public converter outputs with forced-output equivalence, not directly captured internal buffers.','Historical native tiled helper and upstream raw graph must independently reproduce actual down outputs.']
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-start;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='book-quant-replay.json'};save()
if __name__=='__main__':main()
