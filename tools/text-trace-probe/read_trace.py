#!/usr/bin/env python3
"""Artifact-only correction; no model staging, build, encoding or decoding."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import analysis
ROOT=Path(__file__).resolve().parents[2]
PRODUCER='8df1604435314b09ea8cbb60f47833a04307a492349191901313ed9b39540186'
TABLE='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def locate(name,digest):
 rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one complete immutable '+name);return rows[0]
def verify(root,name,row):
 path=Path(name);require(not path.is_absolute()and'..'not in path.parts and (root/path).is_file(),'closed retained artifact name');p=root/path;require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'retained artifact bytes/hash '+name)
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'bounded pinned CPU reader')
 out=Path(os.environ['HEAVY_OUT']);started=time.monotonic();report={'schema':'lighton4b-upstream-text-trace-reader/v1','status':'failed','inference_executed':False,'model_downloaded':False,'quality_qualified':False,'source_sha':os.environ['LIGHTON4B_EXPECTED_SOURCE'],'producer_receipt_sha256':PRODUCER,'table_receipt_sha256':TABLE}
 def stop(sig,frame):raise TimeoutError('bounded300s artifact reader')
 old=signal.signal(signal.SIGALRM,stop);signal.alarm(300)
 try:
  pp=locate('text-trace.json',PRODUCER);p=json.loads(pp.read_text());require(p['source_sha']=='195b181432251beb5a42b31689aed2b68648b2cc'and p['status']=='failed'and p['error']=='actual table prompt token census'and p['all282_scores_observer_neutral'],'preserved complete source capture with reader-only negative')
  tp=locate('physical-f16-table512.json',TABLE);t=json.loads(tp.read_text());require(t['source_observer_full_neutral']and t['status']=='measured_extended_table','completed frozen table')
  require(0<len(p['artifacts'])<=1500 and sum(r['bytes']for r in p['artifacts'].values())<2<<30,'bounded closed producer files')
  for name,row in p['artifacts'].items():verify(pp.parent,name,row)
  report['all_producer_artifacts_rehashed']=len(p['artifacts']);report['producer_artifact_bytes']=sum(r['bytes']for r in p['artifacts'].values())
  verify(tp.parent,'upstream-off/response.json',t['artifacts']['upstream-off/response.json']);frozen=json.loads((tp.parent/'upstream-off/response.json').read_text());require(len(frozen['tokens'])==282 and frozen['tokens'][-1]==248046,'frozen complete independentEOS')
  for arm in ['off','on']:
   a=p['arms'][arm];require(a['status']=='source-frozen-exact'and a['generated_ids']==frozen['tokens']and a['decoded']==frozen['content']and a['eos']and a['all282_ids_text_probability_records_exact']and a['all282_raw_scores_exact']and a['source_features_exact'],'complete source capture/neutrality')
   for i in range(282):require(p['artifacts'][f'{arm}/source.logits.{i}.f32']==t['artifacts'][f'upstream-on/source.logits.{i}.f32'],'all282 actual source score hashes unchanged')
  names=[n for n in t['artifacts']if n.startswith('native/prefill.')or n=='native/greedy.126.logits.f32']
  for n in names:verify(tp.parent,n,t['artifacts'][n])
  native_ids=t['arms']['native']['generated_ids'];require(native_ids[:126]==frozen['tokens'][:126]and native_ids[126]!=frozen['tokens'][126],'shared actual causal126 prefix')
  report['analysis']=analysis.analyze(pp.parent/'on',tp.parent/'native',pp.parent/'on/source.logits.0.f32',pp.parent/'on/source.logits.126.f32',frozen['tokens']);report['producer_seconds']=p['seconds'];report['source_arms']=p['arms'];report['status']='measured_neutral_aligned_text_trace';report['limits']=['One observed calibration table; no heldout or release qualification.','Original capture error retained; this corrects only causal admission of the pinned two-zero-token compatibility probe.','Actual source batching/cropping retained; unequal intermediate inputs/batching do not establish a pure operator defect.','Position126 native hidden states do not exist in this capture; only actual logits are compared there.']
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);report['seconds']=time.monotonic()-started;(out/'text-trace-analysis.json').write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
