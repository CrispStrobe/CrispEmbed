#!/usr/bin/env python3
"""GHA owner: authenticate retained NEW20, then one development-selected model."""
import argparse,importlib.util,json,os,platform,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/josev-new20-owner')]
import josev2b_task_quality_native_v1 as c
import provider
from ci_process import run as run_child


def write(p,v):provider.write(p,v)
def worker():
 spec=importlib.util.spec_from_file_location('josev_new20_native_worker',ROOT/'tools/ci-heavy/josev2b_task_quality_native_v1.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def environment(identity):
 expected={'GITHUB_ACTIONS':'true','RUNNER_ENVIRONMENT':'github-hosted','RUNNER_OS':'Linux','RUNNER_ARCH':'X64','GITHUB_REPOSITORY':'CrispStrobe/CrispEmbed'}
 c.require(all(os.environ.get(k)==v for k,v in expected.items())and platform.system()=='Linux'and platform.machine()=='x86_64','Genuine Linux x64 hosted owner required')
 c.require(os.environ.get('GITHUB_SHA')==identity['source']and os.environ.get('GITHUB_EVENT_NAME')=='workflow_dispatch','Exact standalone source dispatch required')
 c.require(all(os.environ.get(k)for k in ['GH_TOKEN','KAGGLE_API_TOKEN','KAGGLE_ACCOUNT']),'Existing retained/artifact credentials required before tools')
 run=provider.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 c.require(run['id']==int(os.environ['GITHUB_RUN_ID'])and run['head_sha']==identity['source']and run['run_attempt']==int(os.environ['GITHUB_RUN_ATTEMPT'])and
  run['path']=='.github/workflows/josev-new20-native-v1.yml'and run['status']=='in_progress'and run['event']=='workflow_dispatch'and run['repository']['full_name']=='CrispStrobe/CrispEmbed','Actual hosted service/source/attempt differs')
 return {k:run[k]for k in ['id','head_sha','path','run_attempt','event']}

def admission(scratch):
 identity=c.source_guard();cfg=c.config();selected=c.selected_candidate(cfg);c.runtime_prerequisites(cfg);c.canonical_prerequisites(cfg) # Fail before APIs/packages/payloads when pending.
 service=environment(identity)
 scratch.mkdir(parents=True,exist_ok=True)
 ram=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024;disk=shutil.disk_usage(scratch).free
 c.require(ram>=cfg['minimum_available_ram_bytes']and disk>=cfg['initial_free_disk_bytes']and(os.cpu_count()or 0)>=2,'Need14GiB RAM/64GiB initialdisk/two CPUs before transfer')
 return dict(source=identity,service=service,available_ram_bytes=ram,free_disk_bytes=disk,threads=2,selected_recipe=selected['recipe'],cuda_execution=False,original_rerun=False)

def stage(args):
 proof=admission(args.scratch);cfg=c.config();args.inputs.mkdir(parents=True,exist_ok=False);review=args.out/'review';review.mkdir(parents=True,exist_ok=True)
 write(review/'runtime-source-proof.json',c.runtime_source_proof(cfg))
 w=worker();proof['original']=provider.original(cfg,args.inputs/'original',review,w.original_guard)
 proof['old_four_negative']=provider.old_four(cfg,args.inputs,review)
 model,report,proof['candidate']=provider.candidate(cfg,args.inputs/'candidate',review)
 # Only after every original NPY/header/source and selected development receipt
 # is authenticated may the public F32 coefficients be transferred.
 pin=cfg['f32_model'];url=f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{pin['filename']}"
 f32=args.inputs/'published-f32.gguf';proof['f32_model']=provider.stream(url,f32,pin,8*2**30)
 proof.update(completed=True,selected_model=str(model.relative_to(args.inputs)),selected_report=str(report.relative_to(args.inputs)),all1480_before_models=True,native_outputs=False)
 write(review/'staged.json',proof)


def run(args):
 identity=c.source_guard();cfg=c.config();c.native_prerequisites();environment(identity)
 staged=json.loads((args.out/'review/staged.json').read_text());c.require(staged['completed']and staged['all1480_before_models']and staged['native_outputs']is False,'Complete authenticated staging required')
 w=worker();w.original_guard(args.inputs/'original',c.native_prerequisites())
 runtime=args.scratch/'runtime';c.require(not runtime.exists(),'Fresh exact runtime checkout required')
 result=run_child(['git','worktree','add','--detach',runtime,c.native_prerequisites()['runtime_source']],args.out/'review/runtime-checkout.log',cwd=ROOT,timeout=180)
 c.require(result['exit']==0,'Explicit runtime checkout failed')
 result=run_child(['git','-C',runtime,'submodule','update','--init','--depth','1','ggml'],args.out/'review/runtime-submodule.log',cwd=ROOT,timeout=600)
 c.require(result['exit']==0,'Exact runtime backend fetch failed');w.runtime_guard(runtime,c.native_prerequisites())
 c.require(args.worker_python and args.worker_python.is_file(),'Hash-locked numerical worker Python required')
 # Staged originals have already passed stdlib verification; check the isolated
 # actual numerical owner before native import/forward, never install frameworks.
 check="""import importlib.metadata,json,pathlib,sys
import numpy as np
assert sys.prefix!=sys.base_prefix and np.__version__=='2.2.6' and 'torch'not in sys.modules
owner=importlib.metadata.distribution('numpy');assert pathlib.Path(owner._path).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
assert pathlib.Path(np.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())
import hashlib
print(json.dumps(dict(python=sys.version.split()[0],numpy=np.__version__,isolated=True,numpy_owner_metadata_sha256=hashlib.sha256((pathlib.Path(owner._path)/'METADATA').read_bytes()).hexdigest())))"""
 result=run_child([args.worker_python,'-I','-c',check],args.out/'review/numerical-environment.log',cwd=ROOT,timeout=30)
 c.require(result['exit']==0,'Isolated NumPy owner/version rejected')
 native=args.out/'native';model=args.inputs/staged['selected_model'];report=args.inputs/staged['selected_report']
 result=run_child([args.worker_python,'-I',ROOT/'tools/ci-heavy/josev2b_task_quality_native_v1.py',
  '--runtime',runtime,'--reference',args.inputs/'original','--candidate-report',report,'--f32-model',args.inputs/'published-f32.gguf',
  '--q4-model',model,'--build',args.scratch/'build','--out',native],args.out/'review/native-owner.log',cwd=ROOT,
  env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},timeout=18000)
 write(args.out/'review/native-process.json',result)
 summary=dict(schema='crispembed-josev-new20-native-owner-result-v1',source=c.source_guard(),completed=False,selected_recipe=cfg['selected_candidate']['recipe'],original_rerun=False,publication_authorized=False)
 try:
  c.require(not result['timed_out']and result['exit']==0,'Native process rejected; partial reports/logs retained')
  parent=json.loads((native/'hosted.json').read_text());child=json.loads((native/'native.json').read_text())
  c.require(parent['completed']and child['completed'],'Native source/input/structural execution incomplete')
  summary.update(completed=True,heldout_qualified=child['owned_panel_qualified'],numerical_report_sha256=c.sha256(native/'native.json'),parent_report_sha256=c.sha256(native/'hosted.json'),new20_fitting=False)
  if not child['owned_panel_qualified']:summary['classification']='heldout-quality-negative';return 1
  summary['classification']='owned-heldout-panel-passed-not-broad-publication';return 0
 except Exception as error:summary['error']=type(error).__name__+': '+str(error);raise
 finally:write(args.out/'review/owner.json',summary)


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['admit','stage','run']);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker-python',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 try:
  if a.phase=='admit':write(a.out/'review/admission.json',admission(a.scratch));return 0
  if a.phase=='stage':stage(a);return 0
  return run(a)
 except Exception as error:write(a.out/'review'/('rejection-'+a.phase+'.json'),dict(phase=a.phase,error=type(error).__name__+': '+str(error),publication_authorized=False));raise
if __name__=='__main__':raise SystemExit(main())
