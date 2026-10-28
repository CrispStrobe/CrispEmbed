#!/usr/bin/env python3
"""Hosted-only natural-input/workspace cross-admission. Never a model forward."""
import argparse,ctypes as C,importlib.util,json,math,os,shutil,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-q8k-cross'),str(ROOT/'tools/ci-heavy')]
import protocol as P
import wasm_f32_operators_v1 as old
save,require,digest=old.save,P.require,old.digest

def hosted():
 cfg=P.source();old.hosted()
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and
  sys.platform=='linux'and os.environ.get('RUNNER_ARCH')=='X64','Genuine Linux x64 GitHub-hosted task required')
 return cfg

def verify(out,binaries=False):
 cfg=hosted();old.prepared(out)
 proof=json.loads((out/'review/cross-prepared.json').read_text())
 require(proof['protocol_sha256']==digest(P.PATH),'Cross protocol changed')
 for name,row in proof['files'].items():
  path=out/name;require(path.is_file()and not path.is_symlink()and path.stat().st_size==row['bytes']and digest(path)==row['sha256'],'Retained input/workspace changed: '+name)
 require(digest(out/'browser/q8k-packets.json')==proof['packets_sha256'],'Cross packets changed')
 if binaries:old.verified_build(out)
 return cfg,proof

def prepare(out):
 cfg=hosted()
 # Admission before any package/archive/model transfer. Only original frozen
 # coefficient payload is inspected; there is no tokenizer or model forward.
 mem={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')}
 require(mem['MemAvailable']>=4*2**30 and shutil.disk_usage(out.parent).free>=3*2**30 and (os.cpu_count()or 0)>=2,'Hosted4GiB RAM/3GiB disk/twoCPU admission')
 old.prepare(out)
 current=json.loads((out/'review/prepared.json').read_text())['service']
 require(current['path']=='.github/workflows/wasm-q8k-cross-v1.yml'and current['event']=='workflow_dispatch','Exact cross-admission workflow dispatch required')
 service=old.api('actions/runs/40771950208')
 require((service['head_sha'],service['path'],service['status'],service['conclusion'],service['run_attempt'])==
  ('2f864d4beb2f5213a323a21557415dc5e6fab9ad','.github/workflows/wasm-f32-operators-v1.yml','completed','success',1),'Reviewed operator service differs')
 old.download(40771950208,'wasm-f32-operators-v1-review-40771950208',out/'operator-proof.zip',cfg['operator_review'])
 proof_files=old.safe_zip(out/'operator-proof.zip',16*2**20)
 require(P.sha(proof_files['review/operators.json'])==cfg['operator_report_sha256'],'Reviewed operator report differs')
 operators=json.loads(proof_files['review/operators.json'])
 require(operators['completed']and operators['cpp']['stages']==48 and operators['cpp']['status']=='passed'and operators['own_anchors']['exact_own_input_output_anchors']==64 and operators['all_boundaries_within_frozen_budgets']and not operators['gate_failures'],'Reviewed64/48 dependency incomplete')
 raw=old.safe_zip(out/'old-raw.zip',96*2**20);review=old.safe_zip(out/'old-review.zip',16*2**20)
 trace=json.loads(review['review/layer3.json']);inputs={};workspaces={};files={};packets={}
 def retain(name,value):
  path=out/'browser'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
  row=dict(file=name,bytes=len(value),sha256=P.sha(value));files['browser/'+name]={k:row[k]for k in ['bytes','sha256']};return row
 require(trace['weights'][P.WEIGHT]['type']==23 and trace['weights'][P.WEIGHT]['shape']==[1536,384],'Actual stored coefficient type/shape differs')
 weight=raw['native/1/capture/'+P.WEIGHT+'.weight.bin'];require(len(weight)==313344 and P.sha(weight)==trace['weights'][P.WEIGHT]['sha256'],'Exact IQ4_XS coefficient payload differs')
 weight_row=retain('q8k-inputs/weight.bin',weight)
 for axis,prefix,label in [('native','native/1/capture/','native1'),('candidate','browser/arrays/wasm1/','wasm1')]:
  kernel=trace['kernels'][label][P.WEIGHT]
  meta=kernel['metadata'];require(meta['weight_type']==23 and meta['vec_dot_type']==15 and meta['input_shape']==[1536,12]and meta['output_shape']==[384,12]and meta['converted_bytes']==21024,'Actual conversion type/layout differs')
  inputs[axis]=raw[prefix+P.WEIGHT+'.input.f32'];workspaces[axis]=P.valid_workspace(raw[prefix+P.WEIGHT+'.converted.bin'])
  require(P.sha(inputs[axis])==kernel['input_sha256']and len(inputs[axis])==73728 and P.sha(workspaces[axis])==kernel['converted_sha256'],'Natural input/workspace SHA differs')
  retain('q8k-inputs/'+axis+'.f32',inputs[axis]);retain('q8k-inputs/'+axis+'.q8k',workspaces[axis])
  packets['f32_'+axis]=dict(mode=0,input_axis=axis,workspace=None,expected_workspace_sha256=kernel['converted_sha256'])
 for arm in P.ARMS:
  if arm in ('native_scale_candidate_integers','candidate_scale_native_integers'):
   scale,ints=('native','candidate')if arm.startswith('native')else('candidate','native')
   value=P.hybrid(workspaces[scale],workspaces[ints]);input_axis='native'
  else:
   input_axis,which=arm.split('_');value=workspaces[which]
  row=retain('q8k-inputs/'+arm+'.q8k',value)
  packets[arm]=dict(mode=1,input_axis=input_axis,workspace=row)
 for key,packet in packets.items():
  packet['input']=dict(file='q8k-inputs/'+packet['input_axis']+'.f32',bytes=73728,sha256=P.sha(inputs[packet['input_axis']]))
  packet['weight']=weight_row;packet['expected_outputs']={}
  axis=packet['input_axis']if packet['mode']==0 else key.split('_')[-1]if key in P.ARMS[:4]else None
  if axis:
   for backend,prefix in [('native','replay/native/'),('wasm','browser/arrays/replay/')]:
    member=prefix+axis+'/'+P.WEIGHT+'.output.f32';value=raw[member]
    packet['expected_outputs'][backend]=retain('q8k-targets/'+backend+'-'+axis+'.f32',value)
 save(out/'browser/q8k-packets.json',packets);save(out/'review/q8k-packets.json',packets)
 save(out/'review/cross-prepared.json',dict(natural_workspace_blocks={k:P.describe(v)for k,v in workspaces.items()},protocol_sha256=digest(P.PATH),files=files,packets_sha256=digest(out/'browser/q8k-packets.json'),operator_service={k:service[k]for k in ['id','head_sha','path','conclusion','run_attempt']},old_trace_report_sha256=old.TRACE_REPORT,weight=trace['weights'][P.WEIGHT],qualification=False))


def build(out):
 verify(out);source=out/'source'
 old.command(['git','worktree','add','--detach',source,old.BASE],out,'source-worktree')
 old.command(['git','-C',source,'submodule','update','--init','--depth','1','ggml'],out,'source-submodule')
 require(old.git(source,'rev-parse','HEAD')==old.BASE and old.git(source/'ggml','rev-parse','HEAD')==old.GGML,'Frozen blueprint/backend differs')
 file=source/'ggml/src/ggml-cpu/ggml-cpu.c';text=file.read_text();before=digest(file)
 require(before=='b8c04a65c4ee295428501c0951c52b334920fef9101cb100061731b1b9fbee94','Exact a06 CPU source required')
 anchor='    ggml_barrier(params->threadpool);\n\n#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {'
 require(text.count(anchor)==1,'Unique actual conversion barrier required')
 capture=(ROOT/'tests/wasm-layer3/kernel_capture.inc').read_text()
 text=text.replace(anchor,'    ggml_barrier(params->threadpool);\n\n'+capture+'\n#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {')
 file.write_text(text);save(out/'review/overlay.json',dict(file='ggml/src/ggml-cpu/ggml-cpu.c',before_sha256=before,after_sha256=digest(file),capture_sha256=P.sha(capture.encode()),test_only=True,numerical_operations_changed=False))
 for label in ['native','wasm']:
  args=old.configuration(out,source,label)
  args=[str(x).replace(str(ROOT/'tests/wasm-f32-operators'),str(ROOT/'tests/wasm-q8k-cross'))for x in args]
  old.command(args,out,label+'-configure');shutil.copy2(out/(label+'-build/compile_commands.json'),out/'review'/('compile-'+label+'.json'))
  old.command(['cmake','--build',out/(label+'-build'),'--parallel','2'],out,label+'-build',1800)
 save(out/'review/compile-profile.json',old.compile_profile(out))
 destination=out/'browser/module';destination.mkdir()
 for path in (out/'wasm-build').glob('wasm-q8k-cross.*'):
  if path.suffix in ['.js','.wasm']:shutil.copy2(path,destination)
 require(all((destination/('wasm-q8k-cross'+ext)).is_file()for ext in ['.js','.wasm']),'Actual module pair required')
 save(out/'review/build-artifacts.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))for folder in [out/'native-build',destination]for p in folder.rglob('*')if p.is_file()and('.so'in p.name or p.suffix in ['.js','.wasm'])})
 old.ref_controls(out)


def native(out,interventions=False):
 verify(out,True)
 if interventions:anchors(out)
 lib=C.CDLL(str(out/'native-build/libwasm-q8k-cross.so'))
 binaries=json.loads((out/'review/build-artifacts.json').read_text())
 mappings={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if '/'in line and 'libggml'in line.split()[-1]}
 require(mappings,'Actual GGML provider mappings required')
 for name in mappings:
  path=Path(name).resolve();require(path.is_relative_to((out/'native-build').resolve()),'Unexpected GGML provider')
  row=binaries.get(str(path.relative_to(out.resolve())));require(row and digest(path)==row['sha256'],'Actual GGML provider changed')
 save(out/'review/native-mappings.json',dict(all_exact_frozen_build=True,providers=sorted(mappings)))
 f=lib.crispembed_q8k_cross;f.argtypes=[C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int];f.restype=C.c_int
 setter=lib.crispembed_q8k_setenv;setter.argtypes=[C.c_char_p];setter.restype=C.c_int
 packets=json.loads((out/'browser/q8k-packets.json').read_text());report=json.loads((out/'review/native-cross.json').read_text())if interventions else {}
 for threads in [1,2]:
  rows=report.setdefault('native'+str(threads),{})
  for key,p in packets.items():
   if (key in P.ARMS[-2:])!=interventions:continue
   inputs={k:(out/'browser'/p[k]['file']).read_bytes()for k in ['weight','input']}
   if p['workspace']:inputs['workspace']=(out/'browser'/p['workspace']['file']).read_bytes()
   arrays={k:C.create_string_buffer(v)for k,v in inputs.items()};output=(C.c_float*4608)()
   folder=out/'raw'/('native'+str(threads))/key;folder.mkdir(parents=True)
   require(setter(str(folder).encode()if p['mode']==0 else None)==0,'Capture setting failed')
   def call():return f(p['mode'],arrays['weight'],313344,arrays['input'],18432,arrays.get('workspace'),21024 if p['mode']else 0,output,4608,threads)
   require(call()==4608,'Actual native call failed: '+key);value=C.string_at(output,18432)
   require(call()==4608 and C.string_at(output,18432)==value,'Native repeat differs')
   require(all(C.string_at(arrays[k],len(v))==v for k,v in inputs.items()),'Native changed read-only inputs')
   if p['mode']==0:require(digest(folder/(P.WEIGHT+'.converted.bin'))==p['expected_workspace_sha256'],'Actual native conversion workspace differs')
   (folder/'output.f32').write_bytes(value)
   rows[key]=dict(file=str((folder/'output.f32').relative_to(out)),bytes=len(value),sha256=P.sha(value),repeat_bit_exact=True,workspace_sha256=p['expected_workspace_sha256']if p['mode']==0 else p['workspace']['sha256'],input_sha256=p['input']['sha256'])
   if 'native'in p['expected_outputs']:require(P.sha(value)==p['expected_outputs']['native']['sha256'],'Native normal/direct route anchor failed: '+key)
   save(out/'review/native-cross.json',report)
  # Actual failed-call output-preservation and recovery, using retained weights.
  setter(None);weight=C.create_string_buffer(inputs['weight']);zeros=(C.c_float*18432)();output=(C.c_float*4608)(*([17.0]*4608))
  require(f(0,weight,313343,zeros,18432,None,0,output,4608,threads)<0 and all(x==17 for x in output),'Wrong extent/recovery contract')
  zeros[0]=float('nan');require(f(0,weight,313344,zeros,18432,None,0,output,4608,threads)<0 and all(x==17 for x in output),'Nonfinite input rejection')
  zeros[0]=0;q=C.create_string_buffer(21024);struct.pack_into('<f',q,0,float('nan'))
  require(f(1,weight,313344,zeros,18432,q,21024,output,4608,threads)<0 and all(x==17 for x in output),'Nonfinite scale rejection')
  struct.pack_into('<f',q,0,0);q[260]=b'\1'
  require(f(1,weight,313344,zeros,18432,q,21024,output,4608,threads)<0 and all(x==17 for x in output),'Invalid sum rejection')
  q[260]=b'\0';require(f(1,weight,313344,zeros,18432,q,21024,output,4608,threads)==4608 and all(x==0 for x in output),'Native positive/recovery')
 save(out/'review/native-controls.json',dict(positive=True,wrong_extent_rejected=True,nonfinite_rejected=True,invalid_workspace_rejected=True,output_preserved_on_failure=True,recovery=True,threads=[1,2]))


def anchors(out):
 verify(out,True)
 packets=json.loads((out/'browser/q8k-packets.json').read_text());native=json.loads((out/'review/native-cross.json').read_text());browser=json.loads((out/'browser/q8k-report.json').read_text())
 require(browser['baseline_completed']and browser['isolated']and browser['pthreads']and not browser['page_errors']and P.valid_controls(browser['controls']),'Actual browser baselines incomplete')
 P.valid_controls(json.loads((out/'review/native-controls.json').read_text()),True)
 rows_by_backend=dict(native,wasm=browser['outputs']);checked={}
 for backend,rows in rows_by_backend.items():
  for key,p in packets.items():
   if key in P.ARMS[-2:]:continue
   row=rows[key];path=out/row['file'];expected=p['expected_outputs']['wasm'if backend=='wasm'else'native']
   require(path.stat().st_size==18432 and digest(path)==row['sha256']==expected['sha256']and row['repeat_bit_exact']and row['input_sha256']==p['input']['sha256'],'Original normal/direct baseline is not byteexact')
   workspace=p['expected_workspace_sha256']if p['mode']==0 else p['workspace']['sha256']
   require(row['workspace_sha256']==workspace,'Saved workspace identity changed')
   if p['mode']==0:
    capture=path.parent/('converted.bin'if backend=='wasm'else P.WEIGHT+'.converted.bin')
    require(capture.stat().st_size==21024 and digest(capture)==workspace,'Actual regenerated workspace changed')
   checked[backend+'/'+key]=row['sha256']
 require(len(checked)==18,'Exact18 baseline anchors before interventions')
 save(out/'review/cross-anchors.json',dict(passed=True,source=old.source_identity()['source'],anchors=checked,qualification=False))


def compare(out):
 verify(out,True);anchors(out)
 native=json.loads((out/'review/native-cross.json').read_text());browser=json.loads((out/'browser/q8k-report.json').read_text());packets=json.loads((out/'browser/q8k-packets.json').read_text())
 require(browser['completed']and browser['isolated']and browser['pthreads']and not browser['page_errors']and P.valid_controls(browser['controls']),'Actual isolated browser/controls incomplete')
 outputs=dict(native,wasm=browser['outputs']);pairs=[];measurements={};anchor_keys=[]
 for backend,rows in outputs.items():
  require(set(rows)==set(packets),'Exact eight-arm coverage required')
  for key,p in packets.items():
   row=rows[key];path=out/row['file'];require(path.stat().st_size==18432 and digest(path)==row['sha256']and row['repeat_bit_exact']and row['input_sha256']==p['input']['sha256'],'Actual input/output changed')
   expected_workspace=p['expected_workspace_sha256']if p['mode']==0 else p['workspace']['sha256']
   require(row['workspace_sha256']==expected_workspace,'Workspace identity differs')
   if p['mode']==0:
    capture=path.parent/('converted.bin'if backend=='wasm'else P.WEIGHT+'.converted.bin')
    require(capture.stat().st_size==21024 and digest(capture)==expected_workspace,'Actual conversion capture changed')
   expected=p['expected_outputs'].get('wasm'if backend=='wasm'else'native')
   if expected:require(row['sha256']==expected['sha256'],'Normal/direct route neutrality failed: '+backend+'/'+key);anchor_keys.append(backend+'/'+key)
 require(all(native['native1'][k]['sha256']==native['native2'][k]['sha256']for k in packets),'Native thread-count output differs')
 # Six anchored arms x three backend/thread contexts before interventions.
 require(len(anchor_keys)==18,'Exact18 original normal/direct anchors required')
 import numpy as np
 def pair(name,a,b):
  a=out/a['file'];b=out/b['file'];shape=[12,384]
  pairs.append(dict(name=name,original=dict(path=str(a),shape=shape,sha256=digest(a)),native=dict(path=str(b),shape=shape,sha256=digest(b))))
  measurements[name]=old.metrics(np.memmap(a,dtype='<f4',mode='r',shape=tuple(shape)),np.memmap(b,dtype='<f4',mode='r',shape=tuple(shape)))
 for key in packets:
  pair('backend.'+key,native['native1'][key],browser['outputs'][key])
 for backend,rows in outputs.items():
  for key in P.ARMS:
   for original in ['f32_native','f32_candidate']:pair(backend+'.'+original+'.'+key,rows[original],rows[key])
 adapters=old.write_pair(out/'control.gguf',out/'replay.gguf',pairs)
 result=subprocess.run([out/'native-build/wasm-f32-diff',out/'control.gguf',out/'replay.gguf',out/'review/ref.json'],capture_output=True)
 (out/'ref.log').write_bytes(result.stdout+result.stderr);cpp=old.collect(result.returncode,out/'review/ref.json',measurements)
 require(len(pairs)==44,'Exact44 predeclared comparisons required')
 selected={backend:{key:[struct.unpack_from('<f',(out/row['file']).read_bytes(),token*384*4)[0]for token in range(12)]for key,row in rows.items()}for backend,rows in outputs.items()}
 save(out/'review/cross.json',dict(selected_graph_output_channel0=selected,schema='crispembed-wasm-q8k-cross-v1',completed=True,source=old.source_identity()['source'],protocol_sha256=digest(P.PATH),anchors=anchor_keys,comparisons=len(pairs),cpp=cpp,independent=measurements,adapters=adapters,qualification=False,whole_model_negative_preserved=True,original_rerun=False,production_change=False,limits='Two saved natural GELU inputs, two savedQ8_K workspaces and two consistent scale/integer hybrids, one frozen layer3 IQ4_XS projection. Workspace mode retains original F32 input only as attribution metadata; the Q8_K tensor is its arithmetic operand. Numeric intervention differences are measured at unchanged gates and never qualify a model.'))
 save(out/'review/raw-manifest.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))for folder in [out/'raw',out/'browser/q8k-inputs',out/'browser/q8k-targets',out/'browser/module']for p in folder.rglob('*')if p.is_file()})


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','build','native','anchors','native-interventions','verify','compare']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:
  if a.phase=='verify':verify(a.output,True)
  elif a.phase=='native-interventions':native(a.output,True)
  else:globals()[a.phase](a.output)
 except Exception as error:save(a.output/'review'/('cross-rejection-'+a.phase+'.json'),dict(phase=a.phase,error=str(error),qualification=False));raise
if __name__=='__main__':main()
