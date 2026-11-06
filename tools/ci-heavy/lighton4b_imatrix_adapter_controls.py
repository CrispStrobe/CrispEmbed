#!/usr/bin/env python3
"""One model-free real-imatrix adapter/actual-consumer check. Never quantize weights."""
import hashlib,importlib.util,json,os,shutil,struct,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_adapter as adapter
from lighton4b_q6_recipe import plan

def target_header(path,mutation=None):
 rows=plan()['all_tensors'];meta=[adapter.text('general.architecture')+struct.pack('<I',8)+adapter.text('qwen35_ocr'),adapter.text('qwen35_ocr.profile')+struct.pack('<I',8)+adapter.text('lightonocr3-4b')];descriptors=[];offset=0
 for row in rows:
  name=row['name'];shape=list(reversed(row['source_shape']))
  if mutation=='width'and name=='l.layers.0.mlp.gate_proj.weight':shape[0]+=1
  if mutation=='name'and name=='l.layers.0.mlp.gate_proj.weight':name='wrong.domain.weight'
  offset=(offset+31)//32*32;descriptors.append(adapter.text(name)+struct.pack('<I',len(shape))+b''.join(struct.pack('<Q',x)for x in shape)+struct.pack('<IQ',0,offset));size=4
  for x in shape:size*=x
  offset+=size
 raw=b'GGUF'+struct.pack('<IQQ',3,len(rows),len(meta))+b''.join(meta)+b''.join(descriptors);raw+=b'\0'*((-len(raw))%32);path.write_bytes(raw)

def main():
 kaggle=Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir();github=os.environ.get('GITHUB_ACTIONS')=='true';adapter.require((kaggle or github)and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','actual hosted portable environment required');adapter.require(not list(Path('/dev').glob('nvidia[0-9]*')),'CPU-only controls required');out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton-imatrix';scratch.mkdir();build=scratch/'build';r={'schema':'lighton4b-imatrix-adapter-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'quantization_executed':False,'model_forward_executed':False,'model_payload_bytes':0,'quality_qualified':False,'execution_host':'kaggle-cpu'if kaggle else'github-cpu','gpu_devices':[]}
 def run(name,args,expect=0):
  result=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True);(out/(name+'.log')).write_text(result.stdout+result.stderr);adapter.require(result.returncode==expect,'control failed: '+name);return result.stdout
 try:
  r['load_average']=os.getloadavg();r['disk_free_bytes']=shutil.disk_usage(scratch).free;r['memory_swap']={k:v.strip()for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in('MemAvailable','SwapFree','SwapTotal')};adapter.require(r['disk_free_bytes']>2*2**30,'bounded build disk unavailable')
  source=out/'source-imatrix.gguf';url='https://huggingface.co/'+adapter.SOURCE_REPO+'/resolve/'+adapter.SOURCE_REV+'/LightOnOCR-3-4B-imatrix.gguf'
  with urllib.request.urlopen(url,timeout=90)as f:raw=f.read(3626497)
  adapter.require(len(raw)==3626496 and hashlib.sha256(raw).hexdigest()==adapter.SOURCE_SHA,'approved real imatrix transfer differs');source.write_bytes(raw)
  run('python-controls',[sys.executable,ROOT/'tests/test_lighton4b_imatrix_adapter.py','--source',source]);adapted=out/'adapted-imatrix.gguf';audit=adapter.adapt(source,adapted,out/'adapter.json');r['adapter']=audit
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);run('build',['cmake','--build',build,'--target','crispembed-quantize','--parallel','2']);binary=build/'crispembed-quantize';r['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();header=scratch/'target-header-only.gguf';target_header(header)
  actual=json.loads(run('real-consumer',[binary,'--check-lighton4b-imatrix',adapted,header]));expected={row['target']:row['normalized_f32_sha256']for row in audit['tensors']};adapter.require(actual['quantization_executed']is False and actual['model_payload_bytes']==0 and actual['normalized_f32_sha256']==expected,'actual C++ consumer means differ');r['actual_consumer']=actual
  for mutation in('width','name'):
   wrong=scratch/(mutation+'.gguf');target_header(wrong,mutation);run('target-'+mutation+'-negative',[binary,'--check-lighton4b-imatrix',adapted,wrong],1)
  tampered=scratch/'tampered.gguf';raw=bytearray(adapted.read_bytes());raw[-1]^=1;tampered.write_bytes(raw);run('tamper-negative',[binary,'--check-lighton4b-imatrix',tampered,header],1)
  # Real upstream format must fail, rather than silently run unweighted; target has no model payload.
  forbidden=scratch/'must-not-publish.gguf';run('raw-upstream-no-fallback',[binary,header,forbidden,'q4_k','--imatrix',source],1);adapter.require(not forbidden.exists(),'failed imatrix request published an output');r['status']='passed';r['all248_actual_means_byte_exact']=True
 finally:(out/'imatrix-adapter-controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
