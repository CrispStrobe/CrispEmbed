#!/usr/bin/env python3
"""Exact legacy component closure then aligned same-source receipt stage gates."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('receipt_legacy',ROOT/'tools/receipt-layer0-probe/worker.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
require,sha,owned,delta,source_array=legacy.require,legacy.sha,legacy.owned,legacy.delta,legacy.source_array
CFG=json.loads((HERE/'protocol.json').read_text());GATE=CFG['changed_gate']
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and np.__version__=='2.3.5'and not list(Path('/dev').glob('nvidia[0-9]*')),'hosted CPU/pinned arrays');require(not any(CFG[k]for k in ['model_forward_allowed','source_capture_allowed','full_f32_model_allowed']),'operator only')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'receipt-aligned';work.mkdir();started=time.monotonic();deadline=Deadline(CFG['worker_seconds']);deadline.__enter__();r=dict(schema=CFG['schema'],status='failed',commands={},candidate_source=CFG['candidate_source'],model_forward_executed=False,source_capture_executed=False,full_f32_model_instantiated=False,qualified=False)
 def save():(out/'receipt-aligned-controls.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,limit=900,expected=0):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:
   try:command(args,out/(label+'.log'),min(limit,CFG['worker_seconds']-(time.monotonic()-started)),r['commands'][label],cwd=ROOT,env=env)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'expected control exit')
   require(r['commands'][label]['returncode']==expected,'actual return code');r['commands'][label].update(expected_returncode=expected,accepted_expected_returncode=True)
  finally:save()
 def locate(name,digest):
  paths=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(paths)==1,'one exact '+name);return paths[0],json.loads(paths[0].read_text())
 try:
  require(legacy.source_identity(ROOT)==CFG['candidate_source'],'exact source-aligned runtime');r['tools_source']=os.environ['LIGHTON4B_TOOLS_SOURCE']
  mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if cap!='max':mem=min(mem,int(cap)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
  r['capacity']=dict(available_bytes=mem,scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free,load=os.getloadavg());require(mem>=6<<30 and r['capacity']['scratch_free']>=6<<30 and r['capacity']['output_free']>=2<<30,'bounded hosted resources')
  op,o=locate('receipt-layer0.json',CFG['operator_receipt_sha256']);sp,s=locate('receipt-prefill-source.json',CFG['source_report_sha256']);fp,f=locate('physical-rope-fixed-panel.json',CFG['fixed_sha256']);require(o['status']=='measured_receipt_layer0_same_input'and o['all256_actual_native_hidden0_byte_exact']and o['duplicate_production_gdn_cache_exact']and s['firstscore_observer_neutral'],'admitted exact previous component/source')
  r['operator_receipt_sha256']=sha(op);r['source_receipt_sha256']=sha(sp);r['fixed_receipt_sha256']=sha(fp)
  def source(g,name):owned(sp.parent,'on/graph-%d.%s.bin'%(g,name),s['artifacts']);return source_array(sp.parent/'on',s['graph_indexes'][g],name)
  actual=work/'actual';actual.mkdir()
  for src,dst in [('prefill.input.f32','input.f32'),('prefill.layer.0.f32','hidden.f32')]:shutil.copy2(owned(fp.parent,'historical-receipt/on/'+src,f['artifacts']),actual/dst)
  embeds=np.concatenate([source(g,'model.input_embed').reshape(n,2560)for g,n in zip(range(1,5),CFG['source_prompt_rows'])]);require(embeds.tobytes()==(actual/'input.f32').read_bytes(),'all256 real input embeddings exact')
  pack=json.loads(owned(fp.parent,'physical-pack.json',f['artifacts']).read_text());require(pack['candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'and not pack['requantization_executed']and not pack['full_f32_model_created'],'unchanged physical model recipe')
  rows=[row for row in pack['tensors']if row.get('physical_name')and row['physical_name'].startswith('blk.0.')];require(rows==o['parameter_manifest']and sum(row['bytes']for row in rows)<=CFG['maximum_parameter_bytes'],'exact prior14 parameter ranges');r['parameter_manifest']=rows
  run('submodule',['git','submodule','update','--init','ggml']);require(legacy.source_identity(ROOT/'ggml')==CFG['ggml_source'],'exact GGML')
  build=work/'build';run('configure',['cmake','-S',HERE,'-B',build,'-DCRISPEMBED_SOURCE='+str(ROOT),'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in legacy.FLAGS.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines();require(all(f'{k}:BOOL={v}'in cache for k,v in legacy.FLAGS.items()),'actual frozen build flags');shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt');run('baseline',[sys.executable,ROOT/'scripts/check-cpu-baseline.py',build])
  targets=['receipt-aligned-component','test-lighton4b-source-prefill-chunks','test-lighton4b-physical-rope','test-lighton4b-physical-swiglu','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-q6-tiled','lighton4b-native-document'];run('build',['cmake','--build',build,'--target',*targets,'--parallel','2'],limit=1800)
  def executable(name):
   paths=[p for p in build.rglob(name)if p.is_file()];require(len(paths)==1,'one actual built target');return paths[0]
  for target in targets:
   if not target.startswith('test-'):continue
   run(target,[executable(target)],limit=300);text=(out/(target+'.log')).read_text();require('pass'in text.lower()and 'skip'not in text.lower(),'actual supported control');r.setdefault('unit_controls',{})[target]=text.strip()
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir()
  for name in ['receipt-aligned-component','lighton4b-native-document']:shutil.copy2(executable(name),bundle/'bin'/name)
  for library in build.rglob('libggml*.so*'):
   if library.is_file():shutil.copy2(library,bundle/'lib'/library.name)
  r['runtime_bundle']={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in bundle.rglob('*')if p.is_file()};require(0<sum(row['bytes']for row in r['runtime_bundle'].values())<=256<<20,'bounded retained exact runtime bundle')
  weights=work/'weights';weights.mkdir()
  for row in rows:
   begin,size=row['source_offset'],row['bytes'];request=urllib.request.Request(CFG['source_model_url']+'?download=true&aligned_range=%d-%d'%(begin,begin+size-1),headers={'Range':'bytes=%d-%d'%(begin,begin+size-1),'Accept-Encoding':'identity'})
   with urllib.request.urlopen(request,timeout=120)as response:
    require(response.status==206 and response.headers.get('Content-Range')=='bytes %d-%d/%d'%(begin,begin+size-1,CFG['source_model_bytes']),'bounded original tensor range');raw=response.read(size+1)
   require(len(raw)==size and hashlib.sha256(raw).hexdigest()==row['source_packed_sha256'],'whole source tensor SHA');name=row['name'];order=None
   if name.endswith('.linear_attn.conv1d.weight'):order=legacy.permutation('qkv',8192)
   elif name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')):order=legacy.permutation('a',32)
   elif row['whole_rows_permuted']:
    role=next(role for key,role in [('in_proj_qkv','qkv'),('in_proj_z','z'),('in_proj_a','a'),('in_proj_b','b')]if key in name);order=legacy.permutation(role,row['source_shape'][1])
   transformed=np.frombuffer(raw,dtype='u1').reshape(len(order),-1)[order].copy().tobytes()if order else raw;require(hashlib.sha256(transformed).hexdigest()==row['sha256'],'exact whole-row native parameter bytes');(weights/(row['physical_name']+'.bin')).write_bytes(transformed)
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(f['gates']['on']);r['base_gates']=f['gates']['on']
  for mode in ['off','on']:
   env[GATE]='0'if mode=='off'else'1';run('component-'+mode,[executable('receipt-aligned-component'),weights,actual,out/mode,mode],env,600)
  old_names=[name for name in o['artifacts']if name.startswith('native/')];require(len(old_names)==22,'exact prior22 native stages/cache');r['off_stage_hashes']={}
  for name in old_names:
   prior=owned(op.parent,name,o['artifacts']);current=out/'off'/Path(name).name;require(current.stat().st_size==prior.stat().st_size and sha(current)==sha(prior),'OFF every original native stage/cache byte');r['off_stage_hashes'][Path(name).name]=sha(current)
  r['all22_legacy_stage_cache_hashes_exact']=True;comparisons=[];offset=0
  for g,n in zip(range(1,5),CFG['source_prompt_rows']):
   for stage,role,width in legacy.PAIRS:
    candidate=np.fromfile(out/'on'/(role+'.f32'),dtype='<f4').reshape(256,width)[offset:offset+n];reference=source(g,stage+'-0').reshape(n,width)[:,legacy.permutation(role,width)];comparisons.append(dict(graph=g,stage=stage,**legacy.compare(candidate,reference)))
   offset+=n
  r['source_stage_comparisons']=comparisons;r['all56_source_stage_comparisons_exact']=len(comparisons)==56 and all(row['bit_exact']for row in comparisons);r['first_source_stage_difference']=next((row for row in comparisons if not row['bit_exact']),None)
  state=source(3,'state_predelta-0').reshape(32,128,128)[legacy.permutation('a',32)].transpose(0,2,1).copy();fifo=source(2,'linear_attn_qkv_mixed-0').reshape(242,8192)[-4:,legacy.permutation('qkv',8192)].T.copy();r['source_image_end_state']=delta(np.fromfile(out/'on/state246.f32',dtype='<f4').reshape(32,128,128),state);r['source_image_end_fifo']=delta(np.fromfile(out/'on/fifo246.f32',dtype='<f4').reshape(8192,4),fifo);save()
  require(r['all56_source_stage_comparisons_exact']and r['source_image_end_state']['bit_exact']and r['source_image_end_fifo']['bit_exact'],'all actual receipt source operator/state boundaries exact')
  r['status']='measured_exact_receipt_aligned_controls';r['passed']=True;r['limits']=['Actual receipt layer0 only; source-captured embeddings and original quantized parameters.','All original OFF22 stage/cache hashes and actual ON56 source stage/state/FIFO gates; no complete model/reference/decoded execution.','Retained source-identical native-document/runtime bundle must be reused for separately frozen observed/fixed decoded A/B.','All earlier receipt/observed negative quality floors remain binding; default-off/no qualification.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close();r['seconds']=time.monotonic()-started;closed={name+'.log'for name in r['commands']}|{'CMakeCache.txt'};r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in closed)};save()
if __name__=='__main__':main()
