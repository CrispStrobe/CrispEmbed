#!/usr/bin/env python3
"""Source-bound hosted controls/component; exact runtime bundle retained for models."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
s=importlib.util.spec_from_file_location('stages',ROOT/'tools/attention252-probe/worker.py');stages=importlib.util.module_from_spec(s);s.loader.exec_module(stages)
require,sha,owned,delta,source_array=stages.require,stages.sha,stages.owned,stages.delta,stages.source_array
CFG=json.loads((HERE/'protocol.json').read_text());GATE='CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE'
FLAGS={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_AVX512':'OFF','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_BLAS':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_BACKEND_DL':'OFF','GGML_METAL':'OFF','CRISPEMBED_BUILD_SHARED':'OFF'}
TARGETS=['test-lighton4b-physical-rope','test-lighton4b-source-prefill-chunks','test-lighton4b-physical-swiglu','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-native-ops','test-lighton4b-physical-f16-attention','test-lighton4b-decoder-compose','physical-rope-unsupported','physical-rope-wasm-guard','physical-rope-source-component','lighton4b-native-document']

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and np.__version__=='2.3.5','hosted pinned arrays only');require(not any(CFG[k]for k in['model_forward_allowed','source_capture_repeat_allowed','full_f32_model_allowed']),'model-free component scope')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);scratch=Path(os.environ['HEAVY_SCRATCH'])/'physical-rope-controls';scratch.mkdir();started=time.monotonic();deadline=Deadline(CFG['total_timeout_seconds']);deadline.__enter__();r=dict(schema=CFG['schema'],status='failed',passed=False,commands={},unit_controls={},checkpoint_loaded=False,model_forward_executed=False,source_capture_repeated=False,full_f32_model_instantiated=False,qualified=False)
 def save():(out/'physical-rope-controls.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,limit=900,expected=0):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True);left=CFG['total_timeout_seconds']-(time.monotonic()-started);require(left>0,'whole control deadline')
  try:
   try:command(args,out/(label+'.log'),min(left,limit),r['commands'][label],env=env,cwd=ROOT)
   except subprocess.CalledProcessError as error:require(error.returncode==expected,'expected actual return code')
   require(r['commands'][label]['returncode']==expected,'actual return code')
  finally:save()
 def locate(name,digest):
  p=[x for x in Path('/kaggle/input').rglob(name)if x.is_file()and x.stat().st_size<1<<20 and sha(x)==digest];require(len(p)==1,'one exact component prerequisite');return p[0]
 try:
  actual_source=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip();require(actual_source==CFG['candidate_source']==os.environ['LIGHTON4B_CANDIDATE_SOURCE'],'exact final candidate');require(subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD:src'],text=True).strip()==CFG['candidate_src_tree'],'exact candidate source tree');r['candidate_source']=actual_source;r['diagnostic_source_commit']=os.environ['LIGHTON4B_TOOLS_SOURCE']
  mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if cap!='max':mem=min(mem,int(cap)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
  r['capacity']=dict(available_bytes=mem,scratch_free=shutil.disk_usage(scratch).free,output_free=shutil.disk_usage(out).free);require(mem>=8<<30 and r['capacity']['scratch_free']>=6<<30 and r['capacity']['output_free']>=2<<30,'bounded source-fed component resources')
  cp=locate('attention252.json',CFG['capture_receipt_sha256']);capture=json.loads(cp.read_text());pp=locate('rope-operators.json',CFG['operator_proof_sha256']);proof=json.loads(pp.read_text());require(capture['actual_same_input_all266_bits']and capture['actual_native_all266_hidden3_bits']and capture['actual_four_token_cache_bits']and proof['same_input_native_and_upstream_ggml_match_actual_source']and proof['native_rope_direct_capture_matches_reconstruction'],'actual source/operator prerequisites');r['capture_receipt_sha256']=sha(cp);r['operator_proof_sha256']=sha(pp)
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==CFG['native_ggml_commit'],'exact native backend')
  build=scratch/'build';run('configure',['cmake','-S',HERE,'-B',build,'-DCRISPEMBED_SOURCE='+str(ROOT),'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in FLAGS.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines()
  for key,value in FLAGS.items():require(f'{key}:BOOL={value}'in cache,'actual candidate profile')
  shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt');r['cmake_cache']=dict(bytes=(out/'CMakeCache.txt').stat().st_size,sha256=sha(out/'CMakeCache.txt'));run('cpu-baseline',[sys.executable,ROOT/'scripts/check-cpu-baseline.py',build]);run('build',['cmake','--build',build,'--target',*TARGETS,'--parallel','2'],limit=1800)
  def executable(name):
   files=[p for p in build.rglob(name)if p.is_file()];require(len(files)==1,'one actual built executable');return files[0]
  flags=[p for p in build.rglob('flags.make')if p.parent.name=='crispembed-qwen35-delta.dir'];require(len(flags)==1 and 'CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI=1'in flags[0].read_text(),'actual supported CPU profile')
  for target in TARGETS:
   if target in ['physical-rope-source-component','lighton4b-native-document','test-lighton4b-decoder-compose']:continue
   run(target,[executable(target)],limit=240);text=(out/(target+'.log')).read_text();require('pass'in text.lower()and 'skip'not in text.lower(),'actual supported/guard control not skip');r['unit_controls'][target]=text.strip()
  r['build_only_controls']=['test-lighton4b-decoder-compose']
  require('PHYSICAL_ROPE source-node/chunk/tail/position/profile controls PASS'in r['unit_controls']['test-lighton4b-physical-rope'],'actual source node/alias/failure/concurrency controls');require(all('PHYSICAL_ROPE_UNSUPPORTED_GUARD PASS'in r['unit_controls'][name]for name in ['physical-rope-unsupported','physical-rope-wasm-guard']),'actual unsupported and WASM guard builds')
  native=executable('lighton4b-native-document');env=dict(os.environ);env.update(CFG['gates'])
  for mode in ['0','1']:
   env[GATE]=mode
   for cap,flag in [(512,'--observed-composed-cap512'),(1536,'--provisional-heldout-cap1536')]:run('cli-%s-%d'%(mode,cap),[native,scratch/'missing.gguf',scratch/'missing.rgb',1,1,scratch/'missing.ids',out,cap,4,'greedy-full',flag],env,30,4)
  # Preserve exactly this tested executable/lib set for later model A/B; no rebuild.
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin/lighton4b-native-document');libs=[p for p in build.rglob('libggml*.so*')if p.is_file()];require(1<=len(libs)<=19,'bounded runtime library set')
  for p in libs:shutil.copyfile(p,bundle/'lib'/p.name)
  members=[p for p in bundle.rglob('*')if p.is_file()];require(sum(p.stat().st_size for p in members)<=256<<20,'bounded tested runtime bundle');r['runtime_bundle']={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in members}
  # Exact retained source layer2/positions and all reference layer3 stages.
  actual=scratch/'actual';actual.mkdir();inputs=[];positions=[];sourceq=[];sourcek=[];sourcehidden=[]
  for g,count in zip(range(1,5),[4,252,6,4]):
   index=capture['source_graph_indexes'][g]
   for stage in ['l_out-2','l_out-3','inp_pos','Qcur-3','Kcur-3']:owned(cp.parent,'source/on/graph-%d.%s.bin'%(g,stage),capture['artifacts'])
   folder=cp.parent/'source/on';inputs.append(source_array(folder,index,'l_out-2').reshape(count,2560));sourcehidden.append(source_array(folder,index,'l_out-3').reshape(count,2560));positions.append(source_array(folder,index,'inp_pos').reshape(4,count)[:3]);sourceq.append(source_array(folder,index,'Qcur-3').reshape(count,4096));sourcek.append(source_array(folder,index,'Kcur-3').reshape(count,1024))
  np.concatenate(inputs).astype('<f4').tofile(actual/'prefill.layer.2.f32');np.concatenate(positions,axis=1).astype('<f4').tofile(actual/'prefill.positions.f32');np.concatenate(sourceq).astype('<f4').tofile(actual/'source.q_rope.f32');np.concatenate(sourcek).astype('<f4').tofile(actual/'source.k_rope.f32');shutil.copy2(owned(cp.parent,'native/hidden.f32',capture['artifacts']),actual/'native.hidden3.f32')
  weights=scratch/'weights';weights.mkdir()
  for row in CFG['tensors']:
   start,size=row['source_offset'],row['bytes'];request=urllib.request.Request(CFG['source_url'],headers={'Range':'bytes=%d-%d'%(start,start+size-1)})
   with urllib.request.urlopen(request,timeout=120)as response:
    require(response.status==206 and response.headers.get('Content-Range')=='bytes %d-%d/%d'%(start,start+size-1,CFG['source_bytes']),'bounded original tensor range');raw=response.read(size+1)
   require(len(raw)==size and hashlib.sha256(raw).hexdigest()==row['source_packed_sha256'],'same11quantized tensors');(weights/(row['physical_name']+'.bin')).write_bytes(raw)
  for mode in ['off','on']:
   env[GATE]='0'if mode=='off'else'1';run('component-'+mode,[executable('physical-rope-source-component'),weights,actual,out/mode,mode],env,900);require('SOURCE_FED_ROPE_COMPONENT PASS'in(out/('component-'+mode+'.log')).read_text(),'real component/source-node/cache-toggle closure')
  rows=[];offset=0
  pairs=[('attn_norm','input_norm',2560),('Qcur_full','q_full',8192),('Kcur_raw','k_raw',1024),('Vcur_raw','v_raw',1024),('Qcur_normed','q_norm',4096),('Kcur_normed','k_norm',1024),('Qcur','q_rope',4096),('Kcur','k_rope',1024),('gate_reshaped','attn_gate',4096),('attn_gated','pooled',4096),('attn_output','attention',2560),('attn_residual','attention_residual',2560),('attn_post_norm','post_norm',2560),('ffn_gate','ffn_gate',9216),('ffn_up','ffn_up',9216),('ffn_swiglu','activation',9216),('ffn_out','down',2560),('l_out','hidden',2560)]
  for g,count in zip(range(1,5),[4,252,6,4]):
   for stage,suffix,width in pairs:
    reference=source_array(cp.parent/'source/on',capture['source_graph_indexes'][g],stage+'-3').reshape(count,width);candidate=np.fromfile(out/'on'/(suffix+'.f32'),dtype='<f4').reshape(266,width)[offset:offset+count];rows.append(dict(graph=g,stage=stage,**delta(candidate,reference)))
   offset+=count
  for g,tag,count in [(1,'cache4',4),(2,'cache256',256)]:
   index=capture['source_graph_indexes'][g]
   for key,stage in [('keys','operand-kq-3-0'),('values','operand-kqv-3-0')]:
    owned(cp.parent,'source/on/graph-%d.%s.bin'%(g,stage),capture['artifacts']);source=source_array(cp.parent/'source/on',index,stage).astype('<f4').reshape(4,256,256);source=source[:,:count]if key=='keys'else source[:,:,:count].transpose(0,2,1).copy();candidate=np.fromfile(out/'on'/(tag+'.'+key+'.f32'),dtype='<f4').reshape(4,count,256);rows.append(dict(scope=tag,stage=key,**delta(candidate,source)))
  r['source_comparisons']=rows;r['all_source_stages_exact']=all(row['bit_exact']for row in rows);r['first_source_difference']=next((row for row in rows if not row['bit_exact']),None);r['off_original_all266_hidden_exact']=True;r['actual_cache_toggle_recovery_pass']=True;r['passed']=r['all_source_stages_exact'];r['status']='measured_source_fed_rope_controls';r['limits']=['Retained previously observed book component inputs only; no new model/source or decoded forward.','ON full layer3 source equality is required before observed model A/B; component proof does not qualify final decoded quality.','Exact runtime bundle is retained for subsequent source-bound observed and fixed model panels.']
 except BaseException as error:r['error_type'],r['error']=type(error).__name__,str(error);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and p.name!='physical-rope-controls.json'};save()

if __name__=='__main__':main()
