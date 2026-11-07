#!/usr/bin/env python3
"""Kaggle CPU focused physical-format6 controls; no checkpoint or learned model."""
import hashlib,json,os,shutil,signal,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(x,s):
 if not x:raise ValueError(s)
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'Kaggle CPU only')
 protocol=ROOT/'docs/lighton4b-upstream-physical-controls-v1.json';cfg=json.loads(protocol.read_text());out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'physical-controls';scratch.mkdir();active=None
 spec=__import__('importlib.util',fromlist=['util']);s=spec.spec_from_file_location('audit',ROOT/'tools/ci-heavy/lighton4b_upstream_physical_audit.py');audit=spec.module_from_spec(s);s.loader.exec_module(audit)
 report={'schema':'lighton4b-upstream-physical-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(protocol),'model_payload_bytes':0,'learned_model_inference':False,'quality_qualified':False,'preflight':{'memory':audit.resources(),'scratch_free':shutil.disk_usage(scratch).free,'output_free':shutil.disk_usage(out).free,'load':os.getloadavg()},'checks':{}}
 def save(): (out/'physical-controls.json').write_text(json.dumps(report,indent=2)+'\n')
 def stop():
  nonlocal active
  if active is not None and active.poll()is None:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  active=None
 def interrupted(signum,frame):stop();save();raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(2700)
 def run(label,args,timeout=1200):
  nonlocal active
  report['phase']=label;save();print('phase',label,flush=True)
  with(out/(label+'.log')).open('w')as f:
   active=subprocess.Popen(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,start_new_session=True)
   try:code=active.wait(timeout=timeout);require(code==0,label+' failed');report['checks'][label]=True
   finally:stop()
 try:
  require(report['preflight']['memory']['MemAvailable']>=4*1024**3,'4GiB effective RAM');require(report['preflight']['scratch_free']>=4*1024**3 and report['preflight']['output_free']>=1024**3,'disk guards')
  run('python-physical-pack',[sys.executable,ROOT/'tests/test_lighton4b_upstream_physical_pack.py']);run('python-physical-map',[sys.executable,ROOT/'tests/test_lighton4b_upstream_physical_audit.py']);run('env-inventory',[sys.executable,ROOT/'tools/list_env_vars.py','--check']);run('submodules',['git','-C',ROOT,'submodule','update','--init','--depth','1','--recursive'])
  sys.path.insert(0,str(ROOT/'tools'));from lighton4b_q6_recipe import plan
  p=plan();physical=[];legacy=[]
  for row in p['all_tensors']:
   if not row['name'].startswith('l.'):continue
   dims=list(reversed(row['source_shape']));dims+= [1]*(4-len(dims));kind=row['type'];legacy.append(row['name']+'\t'+str(kind)+'\t'+'\t'.join(map(str,dims))+'\n');physical.append(row['name']+'\t'+str(14 if row['name']=='l.embed_tokens.weight'else kind)+'\t'+'\t'.join(map(str,dims))+'\n')
  schema=out/'physical-schema.tsv';schema.write_text(''.join(physical));oldschema=out/'legacy-schema.tsv';oldschema.write_text(''.join(legacy));build=scratch/'build'
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  targets=['test-lighton4b-upstream-physical','test-lighton4b-imatrix-q4km-recipe','test-lighton4b-q6-policy','test-lighton4b-q6-tied-head','test-lighton4b-release-fence','test-lighton4b-native-context','test-lighton4b-mixed-policy','test-lighton4b-bf16-linear-controls','test-lighton4b-bf16-dequant-weights','lighton4b-native-document','lighton4b-native-case-preflight']
  require(targets==cfg['targets']and cfg['model_payload_bytes']==0 and not cfg['checkpoint_download_allowed'],'fixed model-free control scope')
  run('build',['cmake','--build',build,'--target',*targets,'--parallel','4'],1800);shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt')
  run('physical-semantics',[build/'test-lighton4b-upstream-physical',schema]);run('hf-weighted-profile',[build/'test-lighton4b-imatrix-q4km-recipe',schema]);run('legacy-q6-profile',[build/'test-lighton4b-q6-policy',oldschema]);run('q6-tied-head',[build/'test-lighton4b-q6-tied-head']);run('public-release-fence',[build/'test-lighton4b-release-fence',scratch/'public-header-only.gguf']);run('legacy-native-context',[build/'test-lighton4b-native-context']);run('legacy-mixed-policy',[build/'test-lighton4b-mixed-policy']);run('legacy-bf16-linear',[build/'test-lighton4b-bf16-linear-controls']);run('legacy-bf16-weight',[build/'test-lighton4b-bf16-dequant-weights'])
  report['binary_sha256']={name:sha(build/name)for name in targets};owned=['src/lighton4b_upstream_physical_recipe.h','src/lighton4b_upstream_physical_semantics.h','src/lighton4b_native_context.h','src/lighton4b_gguf_inspection.h','src/lighton4b_tokenizer.cpp','src/lighton4b_linear_block.h','src/lighton4b_attention_block.h','src/lighton4b_decoder_compose.h','src/qwen35_linear.cpp','src/qwen35_linear.h','src/qwen35_delta.cpp','src/qwen35_delta.h','src/qwen35_attention.cpp','src/qwen35_attention.h','src/qwen35_decoder.cpp','src/qwen35_decoder.h','src/core/lighton4b_release_fence.h','src/crispembed.cpp','tools/lighton4b_upstream_physical_pack.py'];report['source_sha256']={f:sha(ROOT/f)for f in owned};report['status']='passed'
 finally:stop();save()
if __name__=='__main__':main()
