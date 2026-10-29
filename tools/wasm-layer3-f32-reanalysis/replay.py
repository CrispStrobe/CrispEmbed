#!/usr/bin/env python3
"""HOSTED compare-only consolidation of saved actual F32 graphs and Ref output."""
import argparse,importlib.util,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/ci-heavy')]
import protocol as P
import wasm_f32_operators_v1 as old
save,require,digest=old.save,P.require,old.digest


def run(out):
 cfg=P.source()
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and os.environ.get('RUNNER_ARCH')=='X64'and sys.platform=='linux','Genuine Linux x64 hosted reanalysis required')
 require(sys.prefix!=sys.base_prefix and sys.flags.isolated,'Isolated compare-only Python required')
 require(os.environ.get('GITHUB_SHA')==old.git(ROOT,'rev-parse','HEAD'),'Exact current source required')
 ram=int(next(v.split()[1]for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:')))*1024
 require(ram>=2*2**30 and shutil.disk_usage(out.parent).free>=512*2**20,'2GiB available RAM/512MiB disk before transfers')
 out.mkdir(parents=True,exist_ok=False);review_dir=out/'review';review_dir.mkdir()
 report=dict(schema='crispembed-wasm-layer3-f32-reanalysis-v1',completed=False,phase='authenticate',source=old.git(ROOT,'rev-parse','HEAD'),producer_source=P.SOURCE,protocol_sha256=digest(P.PATH),model_forward=False,operator_forward=False,build=False,qualification=False,production_change=False,optional_f64_dot=dict(computed=False,scope='Omitted; no F64 value is cast/retyped as an F32 reference'))
 def record():save(review_dir/'reanalysis.json',report)
 record()
 try:
  current=old.api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
  require(current['head_sha']==report['source']and current['path']=='.github/workflows/wasm-layer3-f32-reanalysis-v1.yml'and current['run_attempt']==int(os.environ['GITHUB_RUN_ATTEMPT'])and current['status']=='in_progress'and current['event']=='workflow_dispatch','Current reanalysis service differs')
  pin=cfg['producer'];service=old.api('actions/runs/'+str(pin['run']))
  require((service['head_sha'],service['path'],service['status'],service['conclusion'],service['run_attempt'])==(P.SOURCE,'.github/workflows/wasm-layer3-f32-v1.yml','completed','failure',1),'Exact failed producer service differs')
  report['producer_service']={k:service[k]for k in ['id','head_sha','path','conclusion','run_attempt']}
  for kind in ['review','raw']:old.download(pin['run'],'wasm-layer3-f32-v1-'+kind+'-'+str(pin['run']),out/(kind+'.zip'),pin[kind])
  review=P.archive(out/'review.zip',8*2**20);raw=P.archive(out/'raw.zip',64*2**20)
  initial=json.loads(P.INITIAL.read_text())
  require(set(review)==set(initial['review_members']),'Exact small review inventory differs')
  for name,row in initial['review_members'].items():require(len(review[name])==row['bytes']and P.sha(review[name])==row['sha256'],'Producer review member differs: '+name)
  browser=P.controls(review);native=json.loads(review['review/native-cast.json']);outputs=dict(native,wasm=browser['outputs'])
  prepared=json.loads(review['review/cast-prepared.json']);anchors=json.loads(review['review/cast-anchors.json'])
  require(prepared['source']==P.SOURCE and prepared['protocol_sha256']==initial['producer_protocol_sha256']and anchors['passed']is True and len(anchors['anchors'])==6,'Frozen producer preparation/anchors differ')
  expected=dict(prepared['files']);expected.update(json.loads(review['review/build-artifacts.json']))
  coefficient=prepared['coefficient_sha256'];keys={'quant_native','quant_candidate','cast_native','cast_candidate'}
  for backend,rows in outputs.items():
   require(set(rows)==keys,'Exact four-arm producer outputs required')
   for name,row in rows.items():
    require(row['bytes']==18432 and all(row[k]is True for k in ['ordinary_observed_bit_exact','repeat_bit_exact','inputs_unchanged']),'Actual producer output/neutrality incomplete')
    expected[row['file']]={k:row[k]for k in ['bytes','sha256']}
    axis=name.split('_',1)[1];require(row['input_sha256']==prepared['files']['browser/cast-inputs/'+axis+'.f32']['sha256'],'Natural input identity differs')
    if name.startswith('quant_'):
     target=prepared['files']['browser/cast-targets/'+backend+'-'+axis+'.f32'];require(row['sha256']==anchors['anchors'][backend+'/'+axis]==target['sha256'],'Old native/browser quantized byte anchor differs')
    else:require(row['coefficient_sha256']==coefficient,'Actual cast coefficient identity differs')
   expected['raw/'+backend+'/coefficients.f32']=dict(bytes=2359296,sha256=coefficient)
  require(set(raw)==set(expected)|{'control.gguf','replay.gguf'},'Exact raw payload/binary/adapter inventory required')
  for name,row in expected.items():require(len(raw[name])==row['bytes']and P.sha(raw[name])==row['sha256'],'Producer raw member differs: '+name)
  # Recheck literal coefficient identity; no matrix multiplication or graph runs.
  spec=importlib.util.spec_from_file_location('_immutable_e3_format',ROOT/'tools/wasm-layer3-f32/protocol.py');fmt=importlib.util.module_from_spec(spec);spec.loader.exec_module(fmt)
  require(P.sha(fmt.decode(raw['browser/cast-inputs/weight.bin']))==coefficient,'Independent IQ4_XS coefficient identity differs')
  pairs=P.pairs(outputs);materialized=out/'authenticated';materialized.mkdir()
  for name,value in raw.items():
   # Never materialize/import/load a retained .so/.js/.wasm binary.
   if name.startswith(('raw/','browser/cast-inputs/','browser/cast-targets/')):
    path=materialized/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
  descriptors=[]
  for name,a,b in pairs:descriptors.append(dict(name=name,original=dict(path=materialized/a,shape=[12,384],sha256=P.sha(raw[a])),native=dict(path=materialized/b,shape=[12,384],sha256=P.sha(raw[b]))))
  adapters=old.write_pair(out/'control.gguf',out/'replay.gguf',descriptors)
  require((out/'control.gguf').read_bytes()==raw['control.gguf']and(out/'replay.gguf').read_bytes()==raw['replay.gguf'],'Lossless regenerated Ref operands differ from actual producer operands')
  report.update(phase='compare-retained-f32',six_original_anchors=anchors,all_raw_members_authenticated={n:dict(bytes=len(v),sha256=P.sha(v))for n,v in raw.items()},all_actual_coefficient_sha256=coefficient,adapters=adapters);record()
  # F32 bytes remain F32. Optional F64 matrix-dot from the failed producer is
  # deliberately absent; it is not needed to verify these immutable25 rows.
  import numpy as np
  require(np.__version__=='2.2.6'and Path(np.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())and 'torch'not in sys.modules,'Exact isolated NumPy and no framework required')
  independent={}
  for name,a,b in pairs:independent[name]=old.metrics(np.frombuffer(raw[a],dtype='<f4').reshape(12,384),np.frombuffer(raw[b],dtype='<f4').reshape(12,384))
  ref=review_dir/'producer-ref.json';ref.write_bytes(review['review/ref.json'])
  actual=old.collect(1,ref,independent)
  count=sum(row['passed']for row in independent.values())
  require(actual['status']=='numeric_rejection'and actual['stages']==25 and actual['all_metrics_agree']and count==7,'Immutable actual Ref negative changed')
  # Saved compiler records, already byte authenticated, must retain exact old
  # CPU/WASM profiles. Reapply pure source helpers without compiling anything.
  evidence=json.loads(review['review/compile-profile-inputs.json']);profiles={}
  for label in ['native','wasm']:
   rows=json.loads(review['review/compile-'+label+'.json']);observed=[]
   for item in evidence[label]['original']:
    name=item['file'].replace('\\','/').split('/ggml-cpu/',1)[1];matches=[r for r in rows if r['file'].replace('\\','/').endswith('/ggml-cpu/'+name)];require(len(matches)==1 and old.arithmetic_flags(matches[0])==old.arithmetic_flags(item),'Saved original/producer compile flags differ')
    observed.append(dict(file=name,arithmetic_flags=old.arithmetic_flags(matches[0])))
   profiles[label]=observed
  old.wasm_sources(json.loads(review['review/compile-wasm.json']))
  report.update(completed=True,phase='retained-negative-reanalysis-complete',producer_actual_ref=actual,independent_f32=independent,pass_count=count,fail_count=25-count,compile_profiles=profiles,operator_supported=False,whole_model_negative_preserved=True,f64_reference_coercion=False,original_ref_report_sha256=P.sha(review['review/ref.json']),initial_failure_preserved=json.loads(review['review/cast-rejection-compare.json']),limits='Retained arrays/Ref output only. No new numerical experiment, original/operator/model forward or build. Old native anchors and .002/.99999 stay unchanged. Job completion verifies the negative, not candidate qualification.');record()
 except Exception as error:report['error']=type(error).__name__+': '+str(error);record();raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
