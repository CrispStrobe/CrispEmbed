#!/usr/bin/env python3
"""Hosted retained-input RoPE operators; no model/reference forward."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
from observe import generate,ENV,HOOK,ANCHOR
from anchors import stage_names
spec=importlib.util.spec_from_file_location('stages',ROOT/'tools/attention252-probe/worker.py');stages=importlib.util.module_from_spec(spec);spec.loader.exec_module(stages)
require,sha,owned,delta,source_array=stages.require,stages.sha,stages.owned,stages.delta,stages.source_array
CFG=json.loads((HERE/'protocol.json').read_text())

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and np.__version__=='2.3.5','hosted pinned arrays only');require(not any(CFG[k]for k in['model_forward_allowed','source_capture_repeat_allowed','whole_f32_model_allowed']),'closed operator scope')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);scratch=Path(os.environ['HEAVY_SCRATCH'])/'rope-operators';scratch.mkdir();started=time.monotonic();deadline=Deadline(CFG['total_timeout_seconds']);deadline.__enter__();r=dict(schema=CFG['schema'],status='failed',commands={},model_forward_executed=False,source_capture_repeated=False,whole_f32_model_instantiated=False,qualified=False)
 def save():(out/'rope-operators.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,limit=900):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True);left=CFG['total_timeout_seconds']-(time.monotonic()-started);require(left>0,'whole operator deadline')
  try:command(args,out/(label+'.log'),min(left,limit),r['commands'][label],env=env,cwd=ROOT)
  finally:save()
 try:
  r['runtime_source']=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip();require(r['runtime_source']==CFG['candidate_source']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact runtime source');require(subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD:src'],text=True).strip()==CFG['candidate_src_tree'],'production tree unchanged');r['diagnostic_source_commit']=os.environ['LIGHTON4B_TOOLS_SOURCE']
  mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if cap!='max':mem=min(mem,int(cap)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
  r['capacity']=dict(available_bytes=mem,scratch_free=shutil.disk_usage(scratch).free,output_free=shutil.disk_usage(out).free);require(mem>=4<<30 and r['capacity']['scratch_free']>=5<<30 and r['capacity']['output_free']>=1<<30,'bounded operator resources')
  found=[p for p in Path('/kaggle/input').rglob('attention252.json')if p.is_file()and p.stat().st_size<1<<20 and sha(p)==CFG['capture_receipt_sha256']];require(len(found)==1,'one exact actual stage capture');capture=found[0];prior=json.loads(capture.read_text());require(prior['status']=='measured_actual_layer3_stages'and prior['actual_native_all266_hidden3_bits']and prior['actual_four_token_cache_bits']and prior['source_retained_hidden_stages_exact'],'actual prior source/native closures');r['capture_receipt_sha256']=sha(capture)
  retained=Path(os.environ['LIGHTON4B_RETAINED_NATIVE']);partial=retained/'rope-operators.json';require(sha(partial)==CFG['retained_native_receipt_sha256'],'exact retained native capture receipt');p=json.loads(partial.read_text())
  require(p['status']=='failed'and p['error_type']=='ValueError'and p['error']=='closed owned path'and p['actual_native_observer_pairs_exact']==22 and p['native_rope_direct_capture_matches_reconstruction']and p['runtime_source']==CFG['candidate_source']and p['capture_receipt_sha256']==CFG['capture_receipt_sha256'],'frozen successful native capture in retained failed producer')
  require(sha(ROOT/'src/qwen35_attention.cpp')==CFG['native_attention_sha256']and hashlib.sha256(generate((ROOT/'src/qwen35_attention.cpp').read_text()).encode()).hexdigest()==p['observer_patch']['after_sha256'],'exact arithmetic-preserving native observer source')
  lines=owned(retained,'native-capture-CMakeCache.txt',p['artifacts']).read_text().splitlines()
  flags={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_AVX512':'OFF','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_BLAS':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_BACKEND_DL':'OFF'}
  for key,value in flags.items():require(f'{key}:BOOL={value}'in lines,'retained actual native profile')
  direct=out/'direct';direct.mkdir();neutral=[]
  for name,row in p['artifacts'].items():
   if name.startswith('direct/')and name.endswith('.f32'):
    source=owned(retained,name,p['artifacts']);shutil.copy2(source,direct/Path(name).name)
    old='native/'+Path(name).name
    if old in prior['artifacts']:require(row==prior['artifacts'][old],'retained actual native observer neutrality');neutral.append(old)
  require(len(neutral)==22 and len(list(direct.glob('*.f32')))==24,'complete retained native direct stage capture')
  for label,oldname in [('queries','q_rope'),('keys','k_rope')]:require((direct/(label+'.actual.f32')).read_bytes()==(direct/(oldname+'.f32')).read_bytes(),'actual native rotation equals reconstruction')
  r.update(actual_native_observer_pairs_exact=22,native_rope_direct_capture_matches_reconstruction=True,native_capture_reused=True,native_capture_repeated=False,retained_native_receipt_sha256=sha(partial),retained_native_diagnostic_source=p['diagnostic_source_commit'])
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==CFG['native_ggml_commit'],'native backend')
  # Preserve actual source inputs/types/geometry. Every ggml replay is compared
  # directly to captured source post-RoPE, with no parameter fitting or recapture.
  inputs=scratch/'inputs';inputs.mkdir();anchors={}
  for g,n in zip(range(1,5),CFG['positions']):
   index=prior['source_graph_indexes'][g]
   for role in ['q','k']:
    stage,rotated,heads=stage_names(role)
    for name in [stage,rotated,'inp_pos']:owned(capture.parent,'source/on/graph-%d.%s.bin'%(g,name),prior['artifacts'])
    dest=inputs/str(g)/role;dest.mkdir(parents=True);source_array(capture.parent/'source/on',index,stage).reshape(n,heads*256).astype('<f4').tofile(dest/'normalized.f32');source_array(capture.parent/'source/on',index,'inp_pos').reshape(4,n).astype('<i4').tofile(dest/'positions.i32');anchors[g,role]=source_array(capture.parent/'source/on',index,rotated).reshape(n,heads*256)
  up=scratch/'upstream'
  for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',CFG['upstream_commit']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  results=[]
  for backend,src in [('upstream',up/'ggml'),('native',ROOT/'ggml')]:
   b=scratch/(backend+'-rope-build');f=dict(flags);f.pop('CRISPEMBED_NATIVE');f['GGML_OPENMP']='ON'if backend=='upstream'else'OFF'
   run(backend+'-rope-configure',['cmake','-S',HERE,'-B',b,'-DGGML_SOURCE='+str(src),'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in f.items()]])
   lines=(b/'CMakeCache.txt').read_text().splitlines()
   for k,v in f.items():require(f'{k}:BOOL={v}'in lines,'actual operator backend profile')
   shutil.copy2(b/'CMakeCache.txt',out/(backend+'-rope-CMakeCache.txt'));run(backend+'-rope-build',['cmake','--build',b,'--target','rope-operator','--parallel','2'])
   for g,n in zip(range(1,5),CFG['positions']):
    for role,heads in [('q',16),('k',4)]:
     dest=out/backend/str(g)/role;run('%s-%d-%s'%(backend,g,role),[b/'rope-operator',inputs/str(g)/role,dest,n,heads],limit=60);candidate=np.fromfile(dest/'rotated.f32',dtype='<f4').reshape(n,heads*256);row=dict(backend=backend,graph=g,role=role,**delta(candidate,anchors[g,role]));results.append(row);require(row['bit_exact'],'actual source RoPE byte closure')
  r['operator_comparisons']=results;r['rope_profile']=CFG['rope'];r['same_input_native_and_upstream_ggml_match_actual_source']=True;r['status']='measured_same_input_rope_operators';r['limits']=['Direct native capture removes the prior reconstruction uncertainty for these exact inputs; full observer/block neutrality is enforced.','The fixed ggml IMROPE preset reproduces actual source outputs; unused YARN metadata fields are not independently observed.','No full-model/source capture repeat or decoded evaluation occurs here.','A gated production candidate still needs component/cache/platform and actual fixed decoded panel gates before promotion.']
 except BaseException as error:r['error_type'],r['error']=type(error).__name__,str(error);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and p.name!='rope-operators.json'};save()

if __name__=='__main__':main()
