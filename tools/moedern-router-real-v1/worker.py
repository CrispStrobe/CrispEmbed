"""Hosted retained-input component replays; no original/full-model forward."""
import argparse,hashlib,importlib.util,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common as c
import prepare
sys.path.insert(0,str(c.ROOT/'tools'))
from ci_process import run
STAGES=dict(input=('logits','probs','weights'),logits=('probs','weights'),probs=('weights',))
SHAPES=dict(logits=[307,8],probs=[307,8],weights=[307,2],indices=[307,2],values=[307,2],denominator=[307,1])

def main(a):
 cfg=c.verify(a.reviewed_source);prepared=prepare.verify_prepared(a.input,cfg)
 c.require(sys.version_info[:2]==(3,12)and sys.prefix!=sys.base_prefix,'Isolated CPython3.12 required')
 c.require(c.identity(a.model)=={k:cfg['model'][k]for k in ['bytes','sha256']},'Exact F32 model bytes required')
 for layer in cfg['layers']:c.require(c.identity(a.weights/('weight'+str(layer['index'])+'.f32'))==dict(bytes=4096,sha256=layer['weight_sha256']),'Actual native-loaded/original router weight bytes differ')
 import numpy as np
 import torch
 c.require(np.__version__=='2.2.6'and torch.__version__=='2.8.0+cpu','Frozen original CPU arithmetic packages required')
 torch.set_num_interop_threads(1)
 c.require(torch.get_num_interop_threads()==1,'Original interop thread count required')
 sys.path.insert(0,str(c.ROOT/'tools/moedern-native-v2'))
 import replay as ref
 import diff_adapter
 review=a.out/'review';raw=a.out/'raw';review.mkdir(exist_ok=True);raw.mkdir(exist_ok=True)
 report=dict(schema='crispembed-moedern-router-real-result-v1',source=a.reviewed_source,completed=False,phase='actual-ref-controls',
             full_model_forwards=0,original_full_model_forwards=0,original_full_corpus_recaptured=False,new_gold=False,publication_qualified=False,
             selected_admission=prepared,torch=dict(version=torch.__version__,numpy=np.__version__,interop_threads=1,config=torch.__config__.show()),
             component_runs=[],own_controls=[],repeat_controls=[],integer=[],comparisons={},source_rejections=[],skipped_comparisons=[])
 save=lambda:c.write(review/'router-real.json',report)
 records={};pairs={k:[]for k in ['own','same_input','source_effect']};measured={k:{}for k in pairs}
 def tensor(pin):
  data=(a.input/pin['path']).read_bytes();t=torch.frombuffer(bytearray(data),dtype=torch.float32).clone()
  # Native's singleton capture describes the existing flat C++ input vector.
  # The original's directly captured [307,128] matrix is never reshaped to fit native.
  shape=[307,128]if pin['shape']in ([307,128],[1,307,128])else pin['shape']
  return t.view(shape)
 def files(directory):
  return {name:dict(path=directory/(name+('.i64'if name=='indices'else'.f32')),shape=SHAPES[name],**c.identity(directory/(name+('.i64'if name=='indices'else'.f32'))))for name in SHAPES if(directory/(name+('.i64'if name=='indices'else'.f32'))).is_file()}
 def emit(directory,name,t):
  expected=torch.int64 if name=='indices'else torch.float32
  c.require(t.dtype==expected and list(t.shape)==SHAPES[name]and bool(torch.isfinite(t).all()),'Actual Torch component layout/finite check failed')
  (directory/(name+('.i64'if name=='indices'else'.f32'))).write_bytes(t.detach().contiguous().numpy().tobytes())
 def torch_run(layer,source,entry,threads,directory):
  directory.mkdir();torch.set_num_threads(threads)
  result=dict(outcome='incomplete',linear_batches=0,softmax_rows=0,selection_rows_started=0,selection_rows_completed=0,scalar_divisions_completed=0,full_model_forwards=0,denominator_sum_add_batches=0)
  save_result=lambda:c.write(directory/'result.json',result)
  save_result()
  try:
   x=tensor(layer[source][entry]);weight=torch.frombuffer(bytearray((a.weights/('weight'+str(layer['index'])+'.f32')).read_bytes()),dtype=torch.float32).clone().view(8,128)
   with torch.inference_mode():
    p=x
    if entry=='input':
     result['linear_batches']=1;save_result();p=torch.nn.functional.linear(x,weight,None);emit(directory,'logits',p)
    if entry!='probs':
     result['softmax_rows']=307;save_result();p=torch.ops.aten.softmax.int(p,-1);emit(directory,'probs',p)
    result['selection_rows_started']=307;save_result();values,ids=torch.ops.aten.topk.default(p,2)
    result['selection_rows_completed']=307;save_result();denominator=values.sum(dim=-1,keepdim=True)+1e-9
    result['denominator_sum_add_batches']=1;save_result();weights=torch.ops.aten.div.Tensor(values,denominator)
    result['scalar_divisions_completed']=614;save_result()
    for name,t in [('values',values),('indices',ids),('denominator',denominator),('weights',weights)]:emit(directory,name,t)
   result.update(outcome='completed',actual_torch_ops=['linear'if entry=='input'else None,'aten.softmax.int'if entry!='probs'else None,'aten.topk.default','sum+epsilon','aten.div.Tensor'])
   return result
  except Exception as error:result.update(outcome='structural_rejection',error=type(error).__name__+': '+str(error));raise
  finally:save_result()
 def native_run(layer,source,entry,threads,directory):
  log=review/(directory.name+'.log');result=run([a.build/'router-real-probe',a.model,str(layer['index']),entry,a.input/layer[source][entry]['path'],directory,str(threads)],log,cwd=c.ROOT,env=dict(os.environ,CRISPEMBED_MOEDERN_COLBERT='1'),timeout=120)
  c.require(not result['timed_out']and result['exit']in (0,2),'Native component process failure')
  row=c.read(directory/'result.json');row['process']=dict(result,log=log.name,**c.identity(log))
  c.require(row['outcome']in ['completed','native_exact_tie_rejection'],'Native structural rejection: '+row['error'])
  c.require(c.identity(directory/'weight.f32')==dict(bytes=4096,sha256=layer['weight_sha256']),'Actual component router weight differs')
  return row
 def record(backend,layer,source,entry,threads,repeat,phase):
  name=f'{phase}_{backend}_l{layer["index"]}_{source}_{entry}_t{threads}_r{repeat}';directory=raw/name
  row=(torch_run if backend=='torch'else native_run)(layer,source,entry,threads,directory)
  result=dict(name=name,backend=backend,layer=layer['index'],source_input=source,entry=entry,threads=threads,repeat=repeat,phase=phase,result=row,files=files(directory))
  records[backend,layer['index'],source,entry,threads,repeat]=result
  report['component_runs'].append({**result,'files':{k:{**v,'path':str(v['path'].relative_to(a.out))}for k,v in result['files'].items()}})
  if row['outcome']!='completed':report['source_rejections'].append(dict(name=name,**row))
  if repeat==1:
   first=records[backend,layer['index'],source,entry,threads,0]
   report['repeat_controls'].append(dict(name=name,passed=row['outcome']==first['result']['outcome']and set(first['files'])==set(result['files'])and all(first['files'][k]['sha256']==v['sha256']for k,v in result['files'].items())))
  save();return result
 def pinfile(layer,side,stage):
  pin=layer[side][stage];return dict(path=a.input/pin['path'],shape=SHAPES[stage],sha256=c.sha(a.input/pin['path']))
 def values(spec):return np.fromfile(spec['path'],dtype='<i8'if str(spec['path']).endswith('.i64')else'<f4').reshape(spec['shape'])
 def compare(family,label,left,right,stages):
  for stage in stages:
   name=label+'_'+stage
   if stage not in left or stage not in right:
    report['skipped_comparisons'].append(dict(family=family,name=name,reason='actual native source rejected exact tie; no synthesized output'));continue
   aa,bb=left[stage],right[stage];pairs[family].append(dict(name=name,original={k:aa[k]for k in ['path','shape','sha256']},native={k:bb[k]for k in ['path','shape','sha256']}))
   x,y=values(aa),values(bb);r=ref.compare(x,y);r['max_absolute_index']=[int(v)for v in np.unravel_index(abs(y.astype('float64')-x.astype('float64')).argmax(),x.shape)]
   r['bit_exact']=Path(aa['path']).read_bytes()==Path(bb['path']).read_bytes();measured[family][name]=r
  if 'indices'in left and 'indices'in right:
   aa,bb=values(left['indices']),values(right['indices']);changed=np.flatnonzero(np.any(aa!=bb,axis=1));c.require(((aa>=0)&(aa<8)).all()and((bb>=0)&(bb<8)).all(),'Actual expert range differs')
   report['integer'].append(dict(family=family,name=label,passed=not len(changed),changed_rows=len(changed),first_rows=changed[:16].tolist(),row58_left=aa[58].tolist(),row58_right=bb[58].tolist()))
  else:report['skipped_comparisons'].append(dict(family=family,name=label+'_indices',reason='no output after native exact-tie rejection'))
 def refs(family):
  if not pairs[family]:return
  folder=raw/('ref_'+family);folder.mkdir()
  left,right=folder/'left.gguf',folder/'right.gguf';adapters=ref.write_pair(left,right,pairs[family])
  actual=ref.actual_ref(a.build/'router-real-ref',left,right,review/(family+'-ref.json'),review/(family+'-ref.log'),measured[family])
  c.require(actual['all_metrics_agree'],'Actual Ref/independent metric disagreement')
  report['comparisons'][family]=dict(actual_ref=actual,independent=measured[family],adapters=adapters,stages=len(pairs[family]))
  save()
 try:
  # Actual Ref controls are tiny authored F32 values, not new model operations.
  x=np.arange(16,dtype='<f4').reshape(2,8)/16;y=x.copy();y[0,0]+=.01
  controls=[];paths={}
  for name,actual in [('positive',x),('numeric',y)]:
   folder=raw/('control_'+name);folder.mkdir();specs=[]
   for side,data in [('left',x),('right',actual)]:
    path=folder/(side+'.f32');path.write_bytes(data.tobytes());specs.append(dict(path=path,shape=[2,8],sha256=c.sha(path)))
   paths[name]=ref.write_pair(folder/'left.gguf',folder/'right.gguf',[dict(name='control',original=specs[0],native=specs[1])]);ind={'control':ref.compare(x,actual)}
   result=ref.actual_ref(a.build/'router-real-ref',folder/'left.gguf',folder/'right.gguf',review/(name+'-ref.json'),review/(name+'-ref.log'),ind)
   controls.append(result['all_metrics_agree']and ind['control']['passed']==(name=='positive'))
  wrong=raw/'wrong.f32';wrong.write_bytes(x.tobytes());diff_adapter.write_gguf(raw/'wrong.gguf',[dict(name='control',path=wrong,shape=[4,4],sha256=c.sha(wrong))])
  status=run([a.build/'router-real-ref',raw/'control_positive/left.gguf',raw/'wrong.gguf',review/'shape-ref.json'],review/'shape-ref.log',cwd=c.ROOT,env=os.environ.copy(),timeout=60)
  c.require(all(controls)and status['exit']==2 and not status['timed_out'],'Actual Ref positive/numeric/shape controls failed');report['actual_ref_controls']=dict(calls=3,passed=True)
  for phase in ['own','cross']:
   report['phase']=phase;save()
   for layer in cfg['layers']:
    for backend in ['torch','native']:
     own='original'if backend=='torch'else'native';source=own if phase=='own'else ('native'if own=='original'else'original')
     for entry in ['input','logits','probs']:
      for threads in [1,2]:
       for repeat in [0,1]:
        result=record(backend,layer,source,entry,threads,repeat,phase)
        if phase=='own':
         checks=[result['result']['outcome']=='completed']
         for stage in [*STAGES[entry],'indices']:
          checks.append(stage in result['files']and Path(result['files'][stage]['path']).read_bytes()==(a.input/layer[own][stage]['path']).read_bytes())
         if backend=='torch':
          for stage in ['values','denominator']:checks.append(Path(result['files'][stage]['path']).read_bytes()==(a.input/layer['original'][stage]['path']).read_bytes())
         report['own_controls'].append(dict(name=result['name'],passed=all(checks)))
         if repeat==0:compare('own',result['name'],{k:pinfile(layer,own,k)for k in [*STAGES[entry],'indices']},result['files'],STAGES[entry])
        elif repeat==0:compare('source_effect',result['name'],{k:pinfile(layer,own,k)for k in [*STAGES[entry],'indices']},result['files'],STAGES[entry])
   prepare.verify_prepared(a.input,cfg)
   if phase=='own':
    refs('own');report['own_anchors_byte_exact']=all(r['passed']for r in report['own_controls'])and all(r['passed']for r in report['repeat_controls']);save()
    if not report['own_anchors_byte_exact']:
     report.update(phase='own-anchor-rejected',completed=False,passed=False);save();return 1
  for layer in cfg['layers']:
   for source in ['original','native']:
    for entry in ['input','logits','probs']:
     for threads in [1,2]:
      left=records['torch',layer['index'],source,entry,threads,0]['files'];right=records['native',layer['index'],source,entry,threads,0]['files']
      compare('same_input',f'l{layer["index"]}_{source}_{entry}_t{threads}',left,right,STAGES[entry])
  refs('same_input');refs('source_effect')
  expected_runs=96;c.require(len(report['component_runs'])==expected_runs,'All96 own/cross component runs required')
  report['full_comparison_coverage']=sum(len(p)for p in pairs.values())==144 and len(report['integer'])==72 and not report['skipped_comparisons']
  report.update(completed=True,phase='completed',passed=report['full_comparison_coverage']and all(r['passed']for r in report['repeat_controls'])and all(r['passed']for rows in measured.values()for r in rows.values())and all(r['passed']for r in report['integer']),
                actual_ref_rows=sum(len(p)for p in pairs.values()),integer_comparisons=len(report['integer']),
                limits='Two layers, full307 retained rows, component-only. Native norm singleton dimensions are declared separately; original input shape is unchanged. Selected mounted authentication only. Native raw selected values are diagnostic gathers; no new model/reference/precision/tie rule.')
  save();return 0 if report['passed']else 1
 except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise
 finally:
  receipts=[c.read(path)for path in raw.glob('*/result.json')]
  report['actual_component_counts']=dict(scope='durable component receipts including incomplete/error runs; pre-call counters are attempts',
   component_run_receipts=len(receipts),linear=sum(r['linear_batches']for r in receipts),softmax_batches_started=sum(r['softmax_rows']>0 for r in receipts),
   selector_batches_attempted=sum(r['selection_rows_started']>0 for r in receipts),selector_batches_completed=sum(r['selection_rows_completed']==307 for r in receipts),
   renormalization_batches_completed=sum(r['scalar_divisions_completed']==614 for r in receipts),selector_rows_completed=sum(r['selection_rows_completed']for r in receipts),
   renormalization_scalar_values_completed=sum(r['scalar_divisions_completed']for r in receipts),torch_denominator_sum_add_batches=sum(r.get('denominator_sum_add_batches',0)for r in receipts))
  report['model_unchanged']=c.identity(a.model)=={k:cfg['model'][k]for k in ['bytes','sha256']};save()
  c.write(review/'file-manifest.json',{str(p.relative_to(a.out)):c.identity(p)for p in a.out.rglob('*')if p.is_file()and p.name!='file-manifest.json'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
 for key in ['input','model','weights','build','out']:p.add_argument('--'+key,type=Path,required=True)
 raise SystemExit(main(p.parse_args()))
