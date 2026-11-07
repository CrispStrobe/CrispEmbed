#!/usr/bin/env python3
"""Model-free hosted build, profile and pinned Pillow22 byte controls."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
sys.path.insert(0,str(Path(__file__).parent))
import resize_patch
SPEC=importlib.util.spec_from_file_location('profile',ROOT/'tools/physical-f16-attention-probe/worker.py')
profile=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(profile)
UPSTREAM='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
NATIVE='3a27b6c665b191cb230be367db5c52d81dd3e2ac'
PROJECTOR='427b38fd03a9d4aec47c9292f15f8771e99e263b2258f415149f0f20dd612f0c'
CASES=((1,1,1,1),(3,7,11,13),(19,31,7,11),(1,23,9,7),(17,1,5,13),(11,17,11,7),(13,19,5,19),(37,41,64,32))
TARGETS=('test-lighton4b-pillow-resize','test-lighton4b-physical-f16-vision','lighton4b-vision-profile-probe','test-lighton4b-pillow-resize-main','test-lighton4b-physical-f16-vision-main-stub')
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def locate(name,digest):
 rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest]
 require(len(rows)==1,'one exact immutable '+name);return rows[0]
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU/endian/numpy')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'vision-controls';work.mkdir();start=time.monotonic()
 report=dict(schema='lighton4b-physical-vision-controls/v1',status='failed',commands={},model_acquisition=False,checkpoint_vision_forward_executed=False,synthetic_vision_component_executed=False,quality_qualified=False,resize_cases=[])
 def save():(out/'vision-controls.json').write_text(json.dumps(report,indent=2)+'\n')
 def run(label,args,limit=900):
  report['phase']=label;report['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,2400-(time.monotonic()-start)),report['commands'][label],cwd=ROOT)
  finally:save()
 deadline=Deadline(2400);deadline.__enter__()
 def canceled(signum,_frame):raise SystemExit('bounded worker canceled by signal '+str(signum))
 prior={sig:signal.signal(sig,canceled)for sig in (signal.SIGTERM,signal.SIGINT)}
 try:
  run('source',['git','rev-parse','HEAD']);report['source_sha']=(out/'source.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  capacity=importlib.util.spec_from_file_location('capacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');base=importlib.util.module_from_spec(capacity);capacity.loader.exec_module(base)
  report['capacity']=dict(memory=base.memory(),scratch_free=shutil.disk_usage(work).free,load=os.getloadavg());require(report['capacity']['memory']['MemAvailable']>=4<<30 and report['capacity']['scratch_free']>=6<<30,'bounded build capacity')
  rp=locate('projector-audit.json',PROJECTOR);sealed=json.loads(rp.read_text());require(sealed['status']=='measured_projector_only_audit'and sealed['live_arithmetic_closures_exact']and sealed['source_observer_byte_neutral'],'valid live projector only')
  for name in ['on/source.input.rgb','on/resized.rgb','source.patches.f32']:
   p=rp.parent/name;meta=sealed['artifacts'][name];require(p.stat().st_size==meta['bytes']and sha(p)==meta['sha256'],'sealed '+name)
  run('ggml-init',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==NATIVE,'native ggml pin')
  native=work/'native';run('native-configure',['cmake','-S',ROOT,'-B',native,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in profile.BUILD_FLAGS.items()],'-DGGML_CPU_REPACK=ON']);profile.validate_profile((native/'CMakeCache.txt').read_text().splitlines());shutil.copyfile(native/'CMakeCache.txt',out/'native-CMakeCache.txt')
  for target in TARGETS:
   run('build-'+target,['cmake','--build',native,'--target',target,'--parallel','2'])
   if target!='lighton4b-vision-profile-probe':run('run-'+target,[native/target])
  require('PHYSICAL_F16_VISION_SYNTHETIC PASS 'in(out/'run-test-lighton4b-physical-f16-vision.log').read_text(),'synthetic valid production graph required');report['synthetic_vision_component_executed']=True
  require('PHYSICAL_F16_VISION PASS 'in(out/'run-test-lighton4b-physical-f16-vision.log').read_text(),'supported real production implementation')
  require('PHYSICAL_F16_VISION_UNSUPPORTED PASS 'in(out/'run-test-lighton4b-physical-f16-vision-main-stub.log').read_text(),'main stub closed')
  upstream=work/'upstream'
  for label,args in [('up-init',['git','init',upstream]),('up-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('up-fetch',['git','-C',upstream,'fetch','--depth','1','origin',UPSTREAM]),('up-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==UPSTREAM,'exact resize reference');report['reference_bridge']=resize_patch.patch(upstream)
  shutil.copyfile(ROOT/'tools/vision-profile-probe/resize_source.cpp',upstream/'crisp-resize.cpp')
  with(upstream/'CMakeLists.txt').open('a')as f:f.write('\nadd_executable(crisp-resize crisp-resize.cpp)\ntarget_link_libraries(crisp-resize PRIVATE mtmd)\n')
  refbuild=work/'reference';flags={k:v for k,v in profile.BUILD_FLAGS.items()if not k.startswith('CRISPEMBED_')};flags.update(GGML_OPENMP='ON',LLAMA_BUILD_TOOLS='ON',LLAMA_BUILD_SERVER='OFF',LLAMA_BUILD_TESTS='OFF',LLAMA_BUILD_EXAMPLES='OFF',MTMD_VIDEO='OFF',BUILD_SHARED_LIBS='ON',GGML_CPU_REPACK='ON')
  run('up-configure',['cmake','-S',upstream,'-B',refbuild,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);shutil.copyfile(refbuild/'CMakeCache.txt',out/'source-CMakeCache.txt');cache=(refbuild/'CMakeCache.txt').read_text().splitlines();require(all(any(line in cache for line in [f'{key}:BOOL={value}',f'{key}:STRING={value}'])for key,value in flags.items()),'exact source build profile')
  run('up-build',['cmake','--build',refbuild,'--target','crisp-resize','--parallel','2'])
  for i,(h,w,oh,ow)in enumerate(CASES):
   pixels=(np.arange(h*w*3,dtype=np.uint64)*73+np.arange(h*w*3,dtype=np.uint64)**2*19+i*31)%256;rgb=work/f'synthetic-{i}.rgb';rgb.write_bytes(pixels.astype('u1').tobytes())
   prefixes=[]
   for role,exe in [('native',native/'test-lighton4b-pillow-resize'),('source',refbuild/'bin/crisp-resize')]:
    prefix=out/f'resize-{i}-{role}';run(f'resize-{i}-{role}',[exe,rgb,h,w,oh,ow,prefix]);prefixes.append(Path(str(prefix)+'.rgb'))
   require(prefixes[0].read_bytes()==prefixes[1].read_bytes(),'source22 synthetic exact');report['resize_cases'].append(dict(shape=[h,w,oh,ow],input_sha256=sha(rgb),output_sha256=sha(prefixes[0]),bit_exact=True));save()
  prefix=out/'actual-form';run('actual-preprocess',[native/'test-lighton4b-pillow-resize','--preprocess',rp.parent/'on/source.input.rgb',317,452,prefix]);require(Path(str(prefix)+'.rgb').read_bytes()==(rp.parent/'on/resized.rgb').read_bytes(),'actual source resized bytes');require(Path(str(prefix)+'.patches.f32').read_bytes()==(rp.parent/'source.patches.f32').read_bytes(),'actual normalized packed patches bytes')
  report['actual_form_preprocessing_exact']=True;report['status']='passed_model_free_vision_controls'
 except BaseException as e:report['error_type']=type(e).__name__;report['error']=str(e);raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  report['elapsed_seconds']=time.monotonic()-start
  owned={label+'.log'for label in report['commands']}|{'native-CMakeCache.txt','source-CMakeCache.txt'}
  report['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.iterdir()if p.is_file()and(p.name in owned or p.name.startswith('resize-')or p.name.startswith('actual-form.'))}
  report['artifact_scope']='Only completed task-owned command logs, caches and result arrays; outer bootstrap/active worker log excluded.';save()
if __name__=='__main__':main()
