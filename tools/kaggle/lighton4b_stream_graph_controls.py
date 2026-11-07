#!/usr/bin/env python3
"""One bounded CPU control; optional authenticated head-only A/B, no decode."""
import hashlib,json,os,platform,resource,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir(),'Kaggle only'
 assert not list(Path('/dev').glob('nvidia[0-9]*')),'CPU only'
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True)
 scratch=Path(os.environ['HEAVY_SCRATCH']);scratch.mkdir(parents=True,exist_ok=True)
 build=Path(os.environ.get('HEAVY_EXISTING_BUILD',str(scratch/'stream-graph-build')))
 source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert source==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source required'
 report={'schema':'lighton4b-stream-graph-controls/v1','source_sha':source,'status':'failed','quality_qualified':False,'model_decode_executed':False,'cpu':platform.processor(),'cpuinfo':Path('/proc/cpuinfo').read_text(),'timing_scope':'head-only, output writes excluded; no model throughput claim'}
 def run(label,args):
  with (out/(label+'.log')).open('w')as f:subprocess.run(list(map(str,args)),cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
 try:
  available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
  maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip();used=int(Path('/sys/fs/cgroup/memory.current').read_text())
  if maximum!='max':available=min(available,int(maximum)-used)
  report['preflight']={'available_bytes':available,'disk_free_bytes':shutil.disk_usage(scratch).free,'load':os.getloadavg(),'cgroup_limit':maximum,'cgroup_current':used}
  assert available>3*2**30 and report['preflight']['disk_free_bytes']>4*2**30,'insufficient resources'
  run('pins',['python3','tools/qwen35_source_pins.py','--check','--verify-git'])
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  targets=['test-qwen35-stream-projection','test-lighton4b-q6-tied-head','test-lighton4b-stream-head-benchmark']
  run('build',['cmake','--build',build,'--target',*targets,'--parallel','2'])
  for name in targets[:2]:run(name,[build/name])
  report['controls']={name:(out/(name+'.log')).read_text()for name in targets[:2]}
  if os.environ.get('LIGHTON4B_HEAD_MANIFEST'):
   manifest=Path(os.environ['LIGHTON4B_HEAD_MANIFEST']);assert sha(manifest)==os.environ['LIGHTON4B_HEAD_MANIFEST_SHA256'],'manifest mismatch'
   m=json.loads(manifest.read_text());assert m['rows']==248320 and 256<=m['width']<=4096 and m['width']%256==0
   assert m['type']in('q8_0','q4_k','q6_k')
   paths=[]
   for role in ('weights','hidden'):
    p=manifest.parent/m[role]['file'];assert sha(p)==m[role]['sha256'],role+' mismatch';paths.append(p)
   report['head_manifest']=m;report['head_manifest_sha256']=sha(manifest)
   run('head-ab',[build/targets[2],m['type'],m['width'],m['rows'],*paths,out/'scores'])
   files=list(out.glob('scores.*.f32'));assert len(files)==4 and len({sha(p)for p in files})==1,'head output drift'
   report['head_ab_log']=(out/'head-ab.log').read_text();report['head_scores_sha256']=sha(files[0])
  report['binary_sha256']={name:sha(build/name)for name in targets};report['status']='passed'
 finally:
  report['peak_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
  (out/'stream-graph-controls.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
