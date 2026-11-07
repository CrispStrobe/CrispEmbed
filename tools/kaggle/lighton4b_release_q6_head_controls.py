#!/usr/bin/env python3
"""CPU-only checkpoint-free Q6 tied head and current no-ship fence checks."""
import hashlib,json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 require(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir(),'Kaggle CPU only; no GitHub workflow dispatch')
 require(not list(Path('/dev').glob('nvidia[0-9]*')),'model-free CPU controls only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);build=scratch/'release-q6-head-build'
 r={'schema':'lighton4b-release-q6-head-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'execution_host':'kaggle-cpu','model_payload_bytes':0,'model_forward_executed':False,'quantization_executed':False,'quality_qualified':False,'current_q4_shipping_approved':False,'gate_or_alias_promoted':False}
 def run(label,args):
  with(out/(label+'.log')).open('w')as f:subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  memory={k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapTotal','SwapFree')};available=int(memory['MemAvailable'].split()[0])*1024
  limit=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current');maximum=limit.read_text().strip()if limit.exists()else'max';used=int(current.read_text())if current.exists() else 0
  if maximum!='max':available=min(available,max(0,int(maximum)-used))
  r['preflight']={'memory_swap':memory,'available_bytes':available,'cgroup_limit':maximum,'cgroup_current_bytes':used,'disk_free_bytes':shutil.disk_usage(scratch).free,'load_average':os.getloadavg()};require(available>2*2**30 and r['preflight']['disk_free_bytes']>4*2**30,'bounded build resources unavailable')
  # Explicit shipping policy, independent of low-level model execution tests.
  registry=(ROOT/'examples/cli/model_mgr.cpp').read_text();block=registry.split('static const ModelEntry k_registry[] = {',1)[1].split('\n};',1)[0];literals=re.findall(r'"((?:\\.|[^"\\])*)"',block);normalized=re.sub('[^a-z0-9]','',''.join(literals).lower())
  for forbidden in('lightonocr34b','lighton4b','f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'):require(forbidden not in normalized,'unqualified LightOn4B registry publication')
  public=(ROOT/'src/qwen35_ocr.cpp').read_text();admission=public.split('const bool allowed_format =',1)[1].split('const bool ready',1)[0];formats={int(x)for x in re.findall(r'format\s*==\s*(\d+)',admission)};require(formats=={1,2,3,4}and not re.search(r'format\s*(?:>=|>|!=)',admission),'public format5 no-ship fence changed')
  release=(ROOT/'.github/workflows/release.yml').read_text();binaries=re.findall(r'^\s+binaries:\s*(.+)$',release,re.M);require(binaries and all('lighton4b'not in row.lower()and not any(x in row for x in('*','?'))for row in binaries),'release binary allowlist admits experimental probes')
  for path in(ROOT/'src').glob('*'):
   if path.suffix in('.h','.cpp'):
    text=path.read_text();require(not re.search(r'(?:setenv|_putenv_s)\s*\(\s*"CRISPEMBED_LIGHTON4B_[^"]+"\s*,\s*"1"',text),'production source enables experiment internally')
  dispatcher=(ROOT/'src/crispembed.cpp').read_text();require('unqualified_4b = core_lighton4b_release_fence::blocked(g)'in dispatcher and dispatcher.index('if (unqualified_4b)')<dispatcher.index('switch (type)',dispatcher.index('auto type = detect_arch(path, unqualified_4b)')),'unqualified physical profile is not rejected before backend dispatch')
  r['release_fence']={'registry_no_4b_alias_or_payload':True,'public_allowed_formats':sorted(formats),'explicit_release_binary_lists':binaries,'source_default_gates_not_forced_on':True,'note':'Existing experimental remote GGUFs remain research artifacts; not release/default aliases. This is no shipping approval.'}
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);run('build-q6-head',['cmake','--build',build,'--target','test-lighton4b-q6-tied-head','--parallel','2'])
  run('q6-tied-head',[build/'test-lighton4b-q6-tied-head']);r['q6_controls']=(out/'q6-tied-head.log').read_text()
  run('build-release-fence',['cmake','--build',build,'--target','test-lighton4b-release-fence','--parallel','2']);run('public-release-fence',[build/'test-lighton4b-release-fence',scratch/'format5-header-only.gguf'])
  r['q6_controls']=(out/'q6-tied-head.log').read_text();r['public_fence_controls']=(out/'public-release-fence.log').read_text();require('Q6_TIED_HEAD passed=1'in r['q6_controls']and'LIGHTON4B_RELEASE_FENCE passed=1'in r['public_fence_controls'],'native controls missing')
  r['binary_sha256']={name:sha(build/name)for name in('test-lighton4b-q6-tied-head','test-lighton4b-release-fence')};r['source_sha256']={p:sha(ROOT/p)for p in('src/qwen35_decoder.cpp','src/qwen35_linear.cpp','src/qwen35_ocr.cpp','src/crispembed.cpp','src/core/lighton4b_release_fence.h','src/lighton4b_native_context.h','src/lighton4b_bf16_dequant_weights.h','tests/test_lighton4b_q6_tied_head.cpp','tests/test_lighton4b_release_fence.cpp','examples/cli/model_mgr.cpp','.github/workflows/release.yml')};r['status']='passed'
 except Exception as e:
  r['error_type']=type(e).__name__;r['error']=str(e)if isinstance(e,ValueError)else'bounded native controls failed; inspect logs';raise
 finally:(out/'release-q6-head-controls.json').write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
