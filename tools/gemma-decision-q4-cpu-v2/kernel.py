"""Genuine isolated CPU entry; complete numeric rejection is preserved separately."""
import argparse,json,os,resource,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import cpu_contract as K
sys.path.insert(0,str(K.ROOT/'tools'))
from ci_process import run

def result_scope(out,phase,code):
 # Structural completeness and the unchanged quality verdict are separate.
 if phase=='baseline':
  p=out/'review/f32-panel.json';barrier=out/'review/f32-barrier.json'
  if p.is_file()and barrier.is_file():
   r=K.read(p);b=K.read(barrier)
   if r['completed']and not b['passed']:return dict(completed=True,numerical_passed=False,phase='f32-rejected',candidates_executed=False,encoder_forward_maximum=88)
 if phase=='candidates'and(out/'review/result.json').is_file():
  r=K.read(out/'review/result.json')
  if r['completed']:
   K.require([x['recipe']for x in r['recipes']]==['broad-mixed','protected-input']and r['selected_recipe']is None and r['publication_qualified']is False,'Exact complete recipe outcome scope')
   K.require((code==0)==all(x['panel']['strict_f32_passed']for x in r['recipes']),'Exit/result numeric disagreement')
   return dict(completed=True,numerical_passed=code==0,phase='candidate-panels-complete',candidates_executed=True,encoder_forward_maximum=264)
 return None

def budget(root,cfg):
 cap=cfg['resources'];free=shutil.disk_usage(root).free;size=sum(p.lstat().st_size for p in root.rglob('*')if p.is_file())
 K.require(free>=cap['minimum_live_reserve_bytes']and size<=cap['output_cap_bytes'],'Declared output cap/reserve rejected; evidence retained')
 return dict(output_bytes=size,free_disk_bytes=free)

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--reviewed-source',required=True);p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 cfg=K.audit_tree(a.reviewed_source);K.provider(a.reviewed_source)
 K.require(sys.version_info[:2]==(3,12)and sys.prefix!=sys.base_prefix and sys.flags.isolated,'Isolated original Python3.12 required')
 K.require(not any(n in sys.modules for n in ['numpy','torch','transformers','gguf','safetensors']),'No optional imports before source/preflight')
 _,hard=resource.getrlimit(resource.RLIMIT_FSIZE);limit=cfg['resources']['single_file_cap_bytes'];K.require(hard==resource.RLIM_INFINITY or hard>=limit,'Existing file cap below declared bound');resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
 report=dict(schema='crispembed-gemma-decision-q4-cpu-kernel-v2',source=a.reviewed_source,completed=False,original_forwards=0,
  publication_qualified=False,public_capi_qualified=False,recipe_selected=False,external_exact_version_Get_Status_every_List_page_required=True,commands=[])
 report.update(protocol_sha256=K.sha(K.PROFILE),resource_policy=cfg['resources'])
 state=out.parent/(out.name+'-kernel.json');save=lambda:K.write(state,report);save();began=time.monotonic()
 env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_GEMMA_DECISION')}
 env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PIP_CONFIG_FILE='/dev/null',PIP_USER='0',PIP_REQUIRE_VIRTUALENV='1')
 def command(label,args,timeout=3600):
  K.require(time.monotonic()-began<5*3600,'Five-hour kernel deadline before next phase');log=out.parent/(out.name+'-'+label+'.log')
  status=run(args,log,cwd=K.ROOT,env=env,timeout=min(timeout,5*3600-(time.monotonic()-began)));report['commands'].append(dict(label=label,**status,log=log.name,**K.identity(log)));save()
  K.require(not status['timed_out'],'Phase timed out; partial evidence retained');return status['exit']
 try:
  for phase in ['preflight','numpy','numeric-controls','build','prepare','baseline','candidates']:
   report['phase']=phase;save()
   args=[sys.executable,'-I',K.ROOT/'tools/ci-heavy/gemma_decision_q4_cpu_v2.py',phase,'--reviewed-source',a.reviewed_source,'--output',out]
   if phase=='numpy':args=[sys.executable,'-I','-m','pip','--isolated','install','--no-deps','--require-hashes','-r',K.ROOT/'tools/gemma-decision-q4-v1/requirements.lock']
   if phase=='numeric-controls':args=[sys.executable,'-I',K.ROOT/'tests/test_gemma_decision_first_q4.py','--numeric','-v']
   code=command(phase,args,7200 if phase in ['baseline','candidates']else 3600)
   if out.exists():report['resource_final']=budget(out,cfg);save()
   if code:
    completed=result_scope(out,phase,code)
    if completed:report.update(completed);return 1
    raise ValueError('Structural/source/process rejection at '+phase)
  final=result_scope(out,'candidates',0);K.require(final is not None,'Complete fixed candidate results required');report.update(final);return 0
 except Exception as error:report['error']=type(error).__name__+': '+str(error);return 1
 finally:save()
if __name__=='__main__':raise SystemExit(main())
