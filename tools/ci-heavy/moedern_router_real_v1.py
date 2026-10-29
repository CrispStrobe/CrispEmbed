#!/usr/bin/env python3
"""Genuine hosted CPU retained-router owner; no full-model/reference forward."""
import argparse,importlib.metadata,json,os,platform,re,shutil,site,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/moedern-router-real-v1'),str(ROOT/'tools')]
import common as c
import prepare
from ci_process import run

def provider(source):
 c.require(platform.system()=='Linux'and platform.machine()=='x86_64'and sys.byteorder=='little'and hasattr(os,'wait4'),'Linux x64 littleendian CPU owner required')
 c.require(not os.environ.get('GITHUB_ACTIONS')and Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()
           and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')=='Batch'and os.environ.get('CRISPEMBED_CAPTURE_SOURCE')==source,'Genuine Kaggle CPU task required; no GHA spoofing')
 c.require(sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,12)and sys.flags.isolated==1 and not site.ENABLE_USER_SITE,'Parent must prepare isolated CPython3.12 and invoke with -I first')
 return dict(provider='Kaggle CPU',session_service_authenticated=False,external_exact_version_Get_Status_every_List_script_proof_required=True,mounted_capture_version='v4',credential_required=False)

def main(a):
 cfg=c.verify(a.reviewed_source);host=provider(a.reviewed_source)
 c.require(not any(k in sys.modules for k in ['torch','numpy','transformers']),'No numerical import before selected data admission')
 c.require(not a.out.exists()and not a.scratch.exists(),'Fresh task-owned output/scratch required')
 a.out.mkdir(parents=True);a.scratch.mkdir(parents=True);review=a.out/'review';raw=a.out/'raw';review.mkdir();raw.mkdir()
 report=dict(schema='crispembed-moedern-router-real-owner-v1',source=a.reviewed_source,completed=False,phase='source',provider=host,commands={},full_model_forwards=0,original_full_model_forwards=0,publication_qualified=False)
 save=lambda:c.write(review/'hosted.json',report);save();deadline=time.monotonic()+5400
 env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')and k not in ['PYTHONPATH','PYTHONHOME','LD_PRELOAD','GH_TOKEN','GITHUB_TOKEN']}
 env.update(CRISPEMBED_CAPTURE_SOURCE=a.reviewed_source,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PIP_CONFIG_FILE='/dev/null',PIP_USER='0',PIP_DISABLE_PIP_VERSION_CHECK='1')
 def command(name,argv,timeout=180,allowed=(0,)):
  report['phase']=name;save();left=deadline-time.monotonic();c.require(left>0,'Overall90minute component budget')
  status=run(argv,review/(name+'.log'),cwd=ROOT,env=env,timeout=min(left,timeout))
  report['commands'][name]=dict(status,log=name+'.log',**c.identity(review/(name+'.log')));save()
  c.require(not status['timed_out']and status['exit']in allowed,'Hosted command rejected: '+name);return status
 try:
  ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
  c.require(ram>=4*2**30 and shutil.disk_usage(a.scratch).free>=8*2**30 and os.cpu_count()>=2,'4GiB RAM/8GiB disk/two CPUs required')
  report['resources']=dict(available_ram_bytes=ram,free_disk_bytes=shutil.disk_usage(a.scratch).free,workers=2)
  c.require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','--show-toplevel'],text=True).strip()==str(ROOT/'ggml')
            and subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['ggml']
            and not subprocess.check_output(['git','-C',ROOT/'ggml','status','--porcelain','--untracked-files=no']),'Actual clean pinned backend required')
  report['selected_admission']=prepare.authenticate(a.capture_root,raw/'input',cfg);report['phase']='all-selected-bytes-admitted-before-packages-model-or-array-import';save()
  command('locked-packages',[sys.executable,'-m','pip','install','--no-deps','--require-hashes','--only-binary=:all:','--no-cache-dir','--force-reinstall','--report',review/'install.json','-r',ROOT/'tools/moedern-router-real-v1/requirements.txt'],900)
  command('pip-check',[sys.executable,'-m','pip','check'])
  pins=c.read(ROOT/'tools/moedern-router-real-v1/wheels.json')
  versions={k:importlib.metadata.version(k)for k in pins if k!='uv'}
  c.require(versions=={k:v['version']for k,v in pins.items()if k!='uv'},'Installed CPU lock differs')
  report['packages']=versions;save()
  fragment=c.extract(a.scratch/'generated',cfg)
  c.write(review/'native-fragment.json',dict(declaration=cfg['native_fragment'],actual=c.identity(fragment),source_arithmetic_changed=False))
  root=a.scratch/'root';build=a.scratch/'build'
  flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DBUILD_SHARED_LIBS=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_BACKEND_DL=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_LLAMAFILE=ON']
  command('root-configure',['cmake','-S',ROOT,'-B',root,*flags]);command('root-ggml-build',['cmake','--build',root,'--target','ggml','ggml-cpu','--parallel','2'],1800)
  command('probe-configure',['cmake','-S',ROOT/'tests/moedern-router-real-v1','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DROOT_BUILD='+str(root),'-DFRAGMENT_DIR='+str(fragment.parent)])
  command('probe-build',['cmake','--build',build,'--parallel','2'],1200)
  old=c.read(root/'compile_commands.json');actual=c.read(build/'compile_commands.json');commands=[r for r in actual if Path(r['file']).name=='probe.cpp']
  isa=lambda text:{f for f in text.split()if f.startswith(('-mavx','-mfma','-mf16c'))}
  expected={f for r in old if Path(r['file']).name=='moedern_text_engine.cpp'for f in isa(r['command'])}
  c.require(len(commands)==1 and isa(commands[0]['command'])==expected and '-ffp-contract=off'in commands[0]['command']and '-ffast-math'not in commands[0]['command'],'Actual old scalar projection ISA/contraction differs')
  c.require(c.sha(fragment)==cfg['native_fragment']['sha256'],'Compiled selector fragment changed')
  command('symbols',['nm','-C',build/'router-real-probe']);command('strings',['strings',build/'router-real-probe'])
  c.require('moedern_text_test::'not in (review/'symbols.log').read_text()and 'ffn.router_probabilities'not in (review/'strings.log').read_text(),'Production macro/observer exclusion differs')
  c.write(review/'build-profile.json',dict(root_cache=c.identity(root/'CMakeCache.txt'),root_commands=[r for r in old if Path(r['file']).name=='moedern_text_engine.cpp'],actual_commands=actual,fragment=c.identity(fragment),binaries={n:c.identity(build/n)for n in ['router-real-probe','router-real-ref']},root_production_library_built=False,full_model_forward=False))
  (raw/'binaries').mkdir()
  for name in ['router-real-probe','router-real-ref']:shutil.copyfile(build/name,raw/'binaries'/name)
  shutil.copyfile(fragment,raw/'native_router.inc')
  command('invalid-before-math-controls',[build/'router-real-probe','--contracts'])
  prepare.verify_prepared(raw/'input',cfg)
  model=a.scratch/'model.gguf';report['model']=prepare.model(model,cfg['model']);save()
  # Read-only actual loaded router bytes; zero component/model computation here.
  env['CRISPEMBED_MOEDERN_COLBERT']='1'
  command('loaded-router-weights',[build/'router-real-probe','--weights',model,raw/'weights'])
  weight_proof={}
  for layer in cfg['layers']:
   name='weight'+str(layer['index'])+'.f32';pin=c.identity(raw/'weights'/name)
   c.require(pin==dict(bytes=4096,sha256=layer['weight_sha256']),'Original loaded router and actual native GGUF bytes differ')
   weight_proof[name]=dict(**pin,original_loaded_tensor='layers.'+str(layer['index'])+'.ffn.router.weight',torch_shape=[8,128],ggml_shape=[128,8],byte_layout_changed=False)
  c.write(review/'weights.json',weight_proof)
  status=command('component-replays',[sys.executable,'-I',ROOT/'tools/moedern-router-real-v1/worker.py','--reviewed-source',a.reviewed_source,'--input',raw/'input','--model',model,'--weights',raw/'weights','--build',build,'--out',a.out/'components'],1800,allowed=(0,1))
  result=c.read(a.out/'components/review/router-real.json');report.update(completed=result['completed'],phase=result['phase'],numeric_exit=status['exit'],own_anchors_byte_exact=result.get('own_anchors_byte_exact'),component_result_sha256=c.sha(a.out/'components/review/router-real.json'),actual_component_counts=result.get('actual_component_counts'),limits='Exact selected mounted actual-v4 bytes; not full400-file or ZIP authentication. Zero full-model/original forward. No precision/tie/default change.')
  save();return status['exit']
 except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise
 finally:
  c.write(review/'file-manifest.json',{str(p.relative_to(a.out)):c.identity(p)for p in a.out.rglob('*')if p.is_file()and p.name!='file-manifest.json'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
 for name in ['capture-root','out','scratch']:p.add_argument('--'+name,type=Path,required=True)
 raise SystemExit(main(p.parse_args()))
