#!/usr/bin/env python3
"""Bounded runtime-cache verification against sealed operator outputs; no model."""
import importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
sys.path.insert(0,str(Path(__file__).parent))
from worker import admit,sha,require,row_map,download_range,metric
from importlib.machinery import SourceFileLoader
profile=SourceFileLoader('attention_profile',str(ROOT/'tools/physical-f16-attention-probe/worker.py')).load_module()
SEALED='4dc28c17717082a36e9f2cad3729a31a83a97c3234b07d3c00002232744a66f5'
ACTUAL='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'
TARGETS=['test-lighton4b-physical-q4-repack','test-lighton4b-physical-q4-repack-main-stub','test-lighton4b-physical-q4-repack-sourcefed','test-lighton4b-physical-f16-attention','test-lighton4b-physical-native-ops','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head','test-lighton4b-main-f16-link']
def receipt(name,digest):
 matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest]
 require(len(matches)==1,'one exact '+name);return matches[0]
def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
 require(np.__version__=='2.3.5'and sys.byteorder=='little','numpy/endian')
 cfg=json.loads((ROOT/'docs/lighton4b-source-repack-operator-v1.json').read_text());admit(cfg)
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'runtime-repack';work.mkdir();start=time.monotonic()
 r={'schema':'lighton4b-physical-q4-repack-controls/v1','status':'failed','commands':{},'sourcefed':{},'model_forward_executed':False,'quality_qualified':False}
 def save():(out/'runtime-repack-controls.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,limit=600):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,2400-(time.monotonic()-start)),r['commands'][label],cwd=ROOT)
  finally:save()
 deadline=Deadline(2400);deadline.__enter__()
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  from worker import selected_dispatch
  spec=importlib.util.spec_from_file_location('capacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
  r['preflight']={'memory':base.memory(),'disk_free':shutil.disk_usage(work).free,'load':os.getloadavg()};require(r['preflight']['memory']['MemAvailable']>=4<<30 and r['preflight']['disk_free']>=8<<30,'capacity')
  rp=receipt('repack-operator.json',SEALED);old=json.loads(rp.read_text());require(old['status']=='measured_same_input_operator_only'and old['source_sha']=='3b41475c88f570cf899bac12ae6c4f80e4eb3400','sealed operator provenance')
  actualp=receipt('physical-f16-table512.json',ACTUAL);actual=json.loads(actualp.read_text());require(actual['status']=='measured_extended_table'and actual['old_cap256_prefix_anchors_exact'],'actual input receipt');selected_dispatch(actual['arms']['upstream-on']['tensor_dispatch'],cfg['matrices'])
  r['operator_receipt_sha256']=sha(rp);r['actual_receipt_sha256']=sha(actualp)
  run('submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_source'],'ggml pin')
  sys.path.insert(0,str(ROOT/'tools'));import qwen35_source_pins as pins
  commits=sorted({row[-1]for row in pins.pins(ROOT)});require(all(len(x)==40 for x in commits),'historical receipt commits');run('fetch-pins',['git','fetch','--depth','1','origin',*commits])
  run('source-pins',['python3',ROOT/'tools/qwen35_source_pins.py','--check','--verify-git']);run('environment',['python3',ROOT/'tests/test_env_vars_documented.py'])
  build=work/'build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in profile.BUILD_FLAGS.items()],' -DGGML_CPU_REPACK=ON'.strip()]);profile.validate_profile((build/'CMakeCache.txt').read_text().splitlines());require('GGML_CPU_REPACK:BOOL=ON'in(build/'CMakeCache.txt').read_text().splitlines(),'repack compile profile');shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
  # Compile the source-fed adapter before the expensive independent main archive.
  for target in [TARGETS[0],TARGETS[2],*TARGETS[3:],TARGETS[1]]:
   run('build-'+target,['cmake','--build',build,'--target',target,'--parallel','2'],900)
   if target.endswith('sourcefed'):continue
   run('run-'+target,[build/target],600)
  require('PHYSICAL_Q4_REPACK PASS 'in(out/'run-test-lighton4b-physical-q4-repack.log').read_text(),'real supported success required');require('PHYSICAL_Q4_REPACK_UNSUPPORTED_CLOSED PASS 'in(out/'run-test-lighton4b-physical-q4-repack-main-stub.log').read_text(),'main stub stays closed')
  for row in cfg['matrices']:
   role=row['role'];width,rows=row['source_shape'];inputp=actualp.parent/row['input'];inputmeta=actual['artifacts'][row['input']];require(sha(inputp)==inputmeta['sha256']and inputp.stat().st_size==inputmeta['bytes'],'actual complete input');x=np.fromfile(inputp,dtype='<f4').reshape(16,width);require(np.isfinite(x).all(),'finite actual input')
   raw=work/(role+'.packed');download_range(cfg,row,raw);permutation=row_map(role,rows);packed=np.frombuffer(raw.read_bytes(),dtype='u1').reshape(rows,-1)[permutation].copy().tobytes();require(__import__('hashlib').sha256(packed).hexdigest()==row['sha256'],'whole-row native transport');nf=work/(role+'-native.packed');nf.write_bytes(packed)
   for n in cfg['tokens']:
    inp=work/(role+str(n)+'.f32');x[:n].astype('<f4').tofile(inp)
    for mode in ['off','on']:
     preferred='repack'if mode=='on'and row['type']==12 else'mapped';reference_name=f'{role}-{n}-native-native-{preferred}.f32';reference=rp.parent/reference_name;meta=old['artifacts'][reference_name];require(reference.stat().st_size==meta['bytes']and sha(reference)==meta['sha256'],'closed preferred/raw source output')
     label=f'{role}-{n}-{mode}';output=out/(label+'.f32');run(label,[build/TARGETS[2],nf,inp,row['type'],width,rows,n,mode,output]);comparison=metric(np.fromfile(output,dtype='<f4'),np.fromfile(reference,dtype='<f4'));r['sourcefed'][label]={'output_sha256':sha(output),'reference_sha256':sha(reference),'metric':comparison};require(comparison['bit_exact'],'runtime preferred/raw exact source-fed output');save()
  r['status']='passed_operator_cache_controls'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-start;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.iterdir()if p.is_file()and p.name!='runtime-repack-controls.json'};save()
if __name__=='__main__':main()
