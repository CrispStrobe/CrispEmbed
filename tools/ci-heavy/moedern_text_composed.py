#!/usr/bin/env python3
"""Hosted raw-text composition; reviewed tokenizer barrier before every payload."""
import argparse
import json
import math
import os
from pathlib import Path,PurePosixPath
import shutil
import struct
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/moedern-text'))
import contract as c
# Only immutable low-level comparison/schema helpers, never historical guards.
sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
import contract_v2 as prior
import replay as reference
from diff_adapter import write_pair


def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))
def verify_run(run,source):
 row=api('actions/runs/'+str(run))
 c.require((row['head_sha'],row['status'],row['conclusion'],row['run_attempt'])==(source,'completed','success',1),'immutable producer run differs')
 return {k:row[k]for k in ('id','head_sha','status','conclusion','run_attempt')}

def download(run,source,pin,path,cap):
 row=api('actions/artifacts/'+str(pin['id']))
 c.require(row['workflow_run']['id']==run and row['workflow_run']['head_sha']==source and not row['expired'] and
  row['name']==pin['name']and row['size_in_bytes']==pin['bytes']and row['digest']=='sha256:'+pin['sha256']and 0<pin['bytes']<=cap,'immutable artifact descriptor/budget differs')
 with path.open('xb')as stream:subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{pin["id"]}/zip'],stdout=stream,check=True,timeout=300)
 c.require(path.stat().st_size==pin['bytes']and c.sha(path.read_bytes())==pin['sha256'],'ZIP bytes/SHA differ')
 return {k:row[k]for k in ('id','name','size_in_bytes','digest','expires_at')}

def members(archive,names):
 with zipfile.ZipFile(archive)as z:
  infos=z.infolist();c.require(len(infos)<=2048 and len({i.filename for i in infos})==len(infos)and sum(i.file_size for i in infos)<=256*1024**2,'ZIP extent/duplicate differs')
  for i in infos:
   p=PurePosixPath(i.filename);c.require(not p.is_absolute()and '..'not in p.parts and '\\'not in i.filename and ((i.external_attr>>16)&0o170000)!=0o120000,'unsafe ZIP entry')
  c.require(set(names)<=set(z.namelist()),'explicit reviewed artifact member absent')
  return {name:z.read(name)for name in names}

def verify_tokenizer_reports(cfg,native,original):
 pin=cfg['unigram']
 c.require(native['schema']=='crispembed-moedern-unigram-native-v1'and native['completed']is True and native['tokenizer_parity']is True and
  native['source']==pin['source']and native['frozen_ten_exact']is True and not native['failures']and native['cases']==34 and
  native['normalization_sequences']==1040 and native['negative_cases']==4 and native['unigram_unit_cases']==10 and native['metadata_negatives']==['revision','flags','matcher-id','matcher-content','score-type','score-nonfinite'] and
  native['ordinary_test_neutrality']is True and native['ordinary_excludes_diagnostics']is True and native['model_weights']is False and native['public_api']is False,'reviewed complete tokenizer evidence differs')
 c.require(original['schema']=='crispembed-moedern-unigram-original-v1'and original['completed']is True and original['frozen_ten_exact']is True and
  original['source']==pin['source']and original['primary_sha256']==cfg['primary']['receipt_sha256']and original['tokenizers']=='0.22.2'and original['model_weights']is False and original['native_executed']is False,'original tokenizer lineage differs')
 c.require(original['files']['tokenizer.gguf']=={'bytes':pin['component_bytes'],'sha256':pin['component_sha256']},'reviewed tokenizer component differs')
 # New source closure never waives the old whole-source guard. Only exact
 # measured frontend implementation/header equality admits its component.
 producer=json.loads(subprocess.check_output(['git','show',pin['source']+':docs/moedern-unigram-frontend.json'],cwd=ROOT))
 for name in ('src/moedern_text_frontend.cpp','src/moedern_text_frontend.h'):
  c.require(c.sha((ROOT/name).read_bytes())==producer['source_sha256'][name],'measured frontend changed before composition')


def prepare(out,scratch,cfg):
 c.require(not scratch.exists(),'fresh scratch required');scratch.mkdir()
 available=int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))*1024
 c.require(available>=2*2**30 and shutil.disk_usage(scratch).free>=4*2**30,'hosted2GiB RAM/4GiB disk required')
 pin=cfg['unigram'];proof={'unigram_run':verify_run(pin['run'],pin['source'])}
 proof['unigram_review']=download(pin['run'],pin['source'],pin['review'],scratch/'tokenizer-review.zip',4*1024**2)
 files=members(scratch/'tokenizer-review.zip',['native.json','original.json'])
 c.require(c.sha(files['native.json'])==pin['report_sha256']and c.sha(files['original.json'])==pin['original_report_sha256'],'reviewed tokenizer report SHA differs')
 native,original=json.loads(files['native.json']),json.loads(files['original.json']);verify_tokenizer_reports(cfg,native,original)
 c.write(out/'review/tokenizer-native.json',native);c.write(out/'review/tokenizer-original.json',original)
 proof['unigram_metadata']=download(pin['run'],pin['source'],pin['metadata'],scratch/'tokenizer-metadata.zip',64*1024**2)
 files=members(scratch/'tokenizer-metadata.zip',[pin['component_member'],pin['gold_member']]);component=files[pin['component_member']];gold=files[pin['gold_member']]
 c.require(len(component)==pin['component_bytes']and c.sha(component)==pin['component_sha256'],'component bytes differ before model')
 c.require({'bytes':len(gold),'sha256':c.sha(gold)}==original['files']['original-gold.json'],'tokenizer original-gold bytes differ')
 (out/'input').mkdir();(out/'input/tokenizer.gguf').write_bytes(component);(out/'input/tokenizer-gold.json').write_bytes(gold)
 # Only AFTER the reviewed tokenizer gate may immutable old model artifacts load.
 primary=cfg['primary'];proof['primary_run']=verify_run(primary['run'],primary['source'])
 proof['primary_review']=download(primary['run'],primary['source'],primary['review_artifact'],scratch/'primary-review.zip',1024**2)
 report_data=members(scratch/'primary-review.zip',['moedern-original/review/original.json'])['moedern-original/review/original.json']
 c.require(c.sha(report_data)==primary['receipt_sha256'],'old primary receipt differs');report=json.loads(report_data)
 c.require(report['completed']is True and report['github_sha']==primary['source']and len(report['cases'])==10 and len(report['scores'])==3,'old primary scope differs')
 c.write(out/'review/primary.json',report)
 tok=json.loads(gold);c.require(tok['frozen_ten_exact']is True and tok['primary_sha256']==primary['receipt_sha256']and len(tok['cases'])==34,'tokenizer original scope differs')
 for old,new in zip(report['cases'],tok['cases'][:10]):
  expected={'ids':old['token_ids'],'supplied_mask':old['supplied_attention_mask'],'effective_mask':old['effective_attention_mask'],'scoring_mask':list(map(int,old['scoring_mask'])),'expansion_positions':[int(not x)for x in old['supplied_attention_mask']]if old['role']=='query'else []}
  c.require(all(new[k]==old[k]for k in ('id','role','text'))and new['encoded']==expected,'tokenizer/model old input lineage differs')
 prior_pin=cfg['native_v2'];proof['native_v2_run']=verify_run(prior_pin['run'],prior_pin['source'])
 proof['native_v2_review']=download(prior_pin['run'],prior_pin['source'],prior_pin['review'],scratch/'native-v2-review.zip',1024**2)
 native_data=members(scratch/'native-v2-review.zip',['native-v2.json'])['native-v2.json'];c.require(c.sha(native_data)==prior_pin['report_sha256'],'native-v2 report differs')
 prior_report=json.loads(native_data);c.require(prior_report['native_parity']is True and prior_report['completed']is True and not prior_report['failures']and prior_report['primary_replaced']is False,'native-v2 not accepted bounded evidence')
 proof['primary_raw']=download(primary['run'],primary['source'],primary['raw_artifact'],scratch/'primary-raw.zip',32*1024**2)
 selected=[r for r in report['stage_arrays']if r['stage']in ('ordinary.features.token_embeddings','ordinary.selected')]
 c.require(len(selected)==20,'exact twenty old final vectors required')
 prefix='moedern-original/';names=[prefix+'model-F32.gguf',prefix+'model-F32.conversion.json']+[prefix+r['path']for r in selected]
 data=members(scratch/'primary-raw.zip',names);reference_root=out/'reference';reference_root.mkdir()
 for row in selected:
  relative=PurePosixPath(row['path']);c.require(not relative.is_absolute()and '..'not in relative.parts and '\\'not in row['path']and str(relative)==row['path']and len(row['shape'])==2 and row['shape'][1]==128 and type(row['shape'][0])is int and 1<=row['shape'][0]<=512 and row['bytes']==4*math.prod(row['shape']),'old vector extent/path differs')
  payload=data[prefix+row['path']];c.require(row['dtype']=='f32'and len(payload)==row['bytes']and c.sha(payload)==row['sha256'],'old vector bytes differ')
  target=reference_root/row['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
 model=data[prefix+'model-F32.gguf'];conversion=data[prefix+'model-F32.conversion.json']
 c.require(c.sha(model)==primary['gguf_sha256']and c.sha(conversion)==primary['conversion_receipt_sha256'],'old model/conversion bytes differ')
 conversion_report=json.loads(conversion);c.require(conversion_report['completed']is True and conversion_report['original_report_sha256']==primary['receipt_sha256']and conversion_report['gguf_sha256']==primary['gguf_sha256']and conversion_report['tensor_count']==32,'old lossless F32 conversion differs')
 (out/'input/model.gguf').write_bytes(model);(out/'input/model.conversion.json').write_bytes(conversion)
 c.write(out/'review/prepared.json',{'schema':'crispembed-moedern-text-prepared-v1','source':os.environ['GITHUB_SHA'],'protocol_sha256':c.sha(c.PATH.read_bytes()),'service':proof,'primary_replaced':False,'model_recapture':False,
  'files':{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':c.sha(p.read_bytes())}for root in [out/'input',out/'reference']for p in root.rglob('*')if p.is_file()}})


def command(args,log,env,allowed=(0,)):
 with log.open('w')as stream:r=subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1200)
 c.require(r.returncode in allowed,'structural/process failure: '+log.name);return r.returncode

def execute(out,scratch,cfg):
 prepared=json.loads((out/'review/prepared.json').read_text());c.require(prepared['source']==os.environ['GITHUB_SHA']and prepared['protocol_sha256']==c.sha(c.PATH.read_bytes()),'prepared source/protocol differs')
 for name,identity in prepared['files'].items():c.require(identity=={'bytes':(out/name).stat().st_size,'sha256':c.sha((out/name).read_bytes())},'preserved input bytes changed')
 c.require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['ggml'],'actual ggml checkout differs')
 if any(x in sys.modules for x in ('torch','transformers','sentence_transformers','tokenizers')):raise ValueError('no framework/tokenizer reexecution permitted')
 import importlib.metadata
 for name,version in [('numpy','2.2.6'),('gguf','0.17.1')]:c.require(importlib.metadata.version(name)==version,'exact array tools required')
 import numpy as np
 env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')};env.update(CRISPEMBED_MOEDERN_TEXT_COMPOSED='1',CRISPEMBED_MOEDERN_TEXT_FRONTEND='1',CRISPEMBED_MOEDERN_COLBERT='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 os.environ.clear();os.environ.update(env)
 build=scratch/'build';command(['cmake','-S',ROOT/'tests/moedern-text','-B',build,'-DCMAKE_BUILD_TYPE=Release'],out/'review/configure.log',env);command(['cmake','--build',build,'--parallel','2'],out/'review/build.log',env)
 shutil.copyfile(build/'compile_commands.json',out/'review/compile_commands.json')
 plain,observed,controls,diff=[build/name for name in ('moedern-text-plain-probe','moedern-text-observed-probe','moedern-text-controls','moedern-text-diff')]
 command([controls],out/'review/controls.log',env);fixtures=scratch/'fixtures';fixtures.mkdir();schema=prior.schema_module(scratch/'schema-source');reference.typed_fixtures(controls,fixtures,out/'review',schema.tensor_schema)
 raw=out/'raw';raw.mkdir();ref_controls=reference.comparator_controls(diff,raw/'ref-controls')
 for path in (plain,observed,controls,diff,build/'libmoedern-text-plain.a',build/'libmoedern-text-observed.a'):shutil.copyfile(path,raw/path.name)
 symbols=subprocess.check_output(['nm','-C',build/'libmoedern-text-plain.a'],text=True);strings=subprocess.check_output(['strings',build/'libmoedern-text-plain.a'],text=True);positive=subprocess.check_output(['nm','-C',build/'libmoedern-text-observed.a'],text=True)
 (out/'review/ordinary-symbols.txt').write_text(symbols);(out/'review/ordinary-strings.txt').write_text(strings)
 c.require('moedern_text_test::'not in symbols and 'std::function<'not in symbols and all(label not in strings for label in ('embedding.lookup','embedding.projection','attn.probabilities','ffn.router_probabilities','Observer rejected:')),'reference callback/stage data leaked into ordinary archive')
 c.require('moedern_text_test::encode_observed'in positive,'observer compiled positive control absent')
 (out/'review/ordinary-symbols.txt').write_text(symbols)
 report=json.loads((out/'review/primary.json').read_text());cases=report['cases'];rows={x['id']:x for x in cases};text=scratch/'raw-text.tsv'
 text.write_text(''.join(x['id']+'\t'+x['role']+'\t'+x['text'].encode().hex()+'\n'for x in cases))
 Path(str(text)+'.pairs').write_text(''.join(s['query']+'--'+s['document']+'\t'+rows[s['query']]['text'].encode().hex()+'\t'+rows[s['document']]['text'].encode().hex()+'\n'for s in report['scores']))
 exits={mode:command([binary,out/'input/model.gguf',out/'input/tokenizer.gguf',text,raw/mode],out/'review'/(mode+'.log'),env,allowed=(0,1))for mode,binary in [('plain',plain),('observed',observed)]}
 result={'schema':'crispembed-moedern-raw-text-composed-v1','completed':False,'text_model_parity':False,'source':os.environ['GITHUB_SHA'],'cases':[],'scores':[],'failures':[],'reference_controls':ref_controls,'producer_exit_codes':exits,'raw_text_only_inputs':True,'original_ids_or_hidden_injected':False,'primary_replaced':False,'default_enabled':False,'public_api':False,'retrieval_quality':False,'release_qualified':False}
 def save():c.write(out/'review/composed.json',result)
 save();original_index={(r['case'],r['stage']):r for r in report['stage_arrays']}
 for case in cases:
  cid=case['id'];record={'id':cid,'checks':{},'status':'started'};result['cases'].append(record)
  try:
   folders={m:raw/m/cid for m in ('plain','observed')}
   for mode,folder in folders.items():
    c.require(not (folder/'rejection.json').exists(),'actual '+mode+' case rejection: '+((folder/'rejection.json').read_text()if (folder/'rejection.json').exists()else ''))
   meta=json.loads((folders['plain']/'tokens.json').read_text());observed_meta=json.loads((folders['observed']/'tokens.json').read_text())
   expected={'ids':case['token_ids'],'supplied_mask':case['supplied_attention_mask'],'effective_mask':case['effective_attention_mask'],'scoring_mask':list(map(int,case['scoring_mask'])),'expansion_positions':[int(not x)for x in case['supplied_attention_mask']]if case['role']=='query'else [],'all_shape':[len(case['token_ids']),128],'selected_shape':case['shape']}
   c.require(meta==observed_meta==expected,'raw text token/three-mask/shape lineage differs')
   for mode,folder in folders.items():
    flags=json.loads((folder/'controls.json').read_text());c.require(all(flags[k]is True for k in ('repeat','gate_recovery','role_bounds_recovery','context_ownership')),'actual ordinary/recovery controls incomplete')
    if mode=='observed':c.require(flags['observer_neutrality']is True and flags['observer_recovery']is True and flags['stages']==55,'actual observer controls incomplete')
   stages,stage_files=reference.stages(folders['observed']);c.require(set(stages)==prior.native_names(),'all55 test-only stages required')
   for name,key in [('input_ids','ids'),('supplied_mask','supplied_mask'),('effective_mask','effective_mask')]:c.require(np.array_equal(stages[name],np.asarray(meta[key],dtype='<i8')[None,:]),'captured actual engine inputs differ')
   for kind in ('all','selected'):c.require((folders['plain']/(kind+'.f32')).read_bytes()==(folders['observed']/(kind+'.f32')).read_bytes(),'ordinary/test callback variant output bits differ')
   c.require(c.sha((folders['plain']/'all.f32').read_bytes())==stage_files['normalized']['sha256'],'normalized captured/ordinary storage differs')
   pairs=[]
   for kind,stage in [('all','ordinary.features.token_embeddings'),('selected','ordinary.selected')]:
    target=original_index[cid,stage];shape=target['shape'];left=out/'reference'/target['path'];right=folders['plain']/(kind+'.f32');a=np.fromfile(left,dtype='<f4').reshape(shape);b=np.fromfile(right,dtype='<f4').reshape(shape)
    metrics=reference.compare(a,b);record['checks'][kind]=metrics
    pairs.append({'name':kind,'original':{'path':left,'shape':shape,'sha256':target['sha256']},'native':{'path':right,'shape':shape,'sha256':c.sha(right.read_bytes())}})
   adapter=raw/'comparisons'/cid;adapter.mkdir(parents=True);identity=write_pair(adapter/'original.gguf',adapter/'native.gguf',pairs)
   measured=reference.actual_ref(diff,adapter/'original.gguf',adapter/'native.gguf',adapter/'ref.json',adapter/'ref.log',record['checks'])
   c.require(measured['all_metrics_agree'],'independent/actual Ref metrics disagree');record.update(actual_ref=measured,adapters=identity,status='passed'if all(v['passed']for v in record['checks'].values())else 'numeric_rejection')
   for name,value in record['checks'].items():
    if not value['passed']:result['failures'].append({'case':cid,'stage':name,**value})
  except Exception as error:record.update(status='structural_rejection',error=type(error).__name__+': '+str(error));result['failures'].append({'case':cid,'error':record['error']})
  save()
 plain_scores=[json.loads(x)for x in (raw/'plain/scores.jsonl').read_text().splitlines()];observed_scores=[json.loads(x)for x in (raw/'observed/scores.jsonl').read_text().splitlines()]
 c.require(len(plain_scores)==len(observed_scores)==3,'all three actual raw-text scores required')
 for target,left,right in zip(report['scores'],plain_scores,observed_scores):
  pair=target['query']+'--'+target['document'];row={'query':target['query'],'document':target['document'],'original':target['original_maxsim'],'passed':False}
  try:
   c.require(left['id']==right['id']==pair and 'score'in left and 'score'in right,'raw-text score producer rejected/differs')
   binary=(raw/'plain'/(pair+'.f32')).read_bytes();c.require(binary==(raw/'observed'/(pair+'.f32')).read_bytes()and len(binary)==4,'score variants not bit exact')
   value=struct.unpack('<f',binary)[0];c.require(value==left['score']==right['score']and math.isfinite(value),'actual score JSON/raw differ')
   q=np.fromfile(raw/'plain'/target['query']/'selected.f32',dtype='<f4').reshape(-1,128);d=np.fromfile(raw/'plain'/target['document']/'selected.f32',dtype='<f4').reshape(-1,128);independent=float((q@d.T).max(axis=1).sum())
   row.update(native=value,independent=independent,absolute_original_difference=abs(value-target['original_maxsim']),absolute_independent_difference=abs(value-independent),passed=max(abs(value-target['original_maxsim']),abs(value-independent))<=.002)
  except Exception as error:row['error']=type(error).__name__+': '+str(error)
  result['scores'].append(row)
  if not row['passed']:result['failures'].append({'stage':'raw-text-maxsim',**row})
  save()
 result.update(completed=True,text_model_parity=not result['failures'],ordinary_reference_exclusion=True,coverage={'old_final_vectors':sum(len(x['checks'])for x in result['cases']),'old_maxsim':len(result['scores']),'cases':len(result['cases'])},products={p.name:{'sha256':c.sha(p.read_bytes()),'bytes':p.stat().st_size}for p in (plain,observed,diff,build/'libmoedern-text-plain.a',build/'libmoedern-text-observed.a')});save()
 c.require(not result['failures']and result['coverage']=={'old_final_vectors':20,'old_maxsim':3,'cases':10},'bounded raw-text composition rejected; complete outcomes retained')


def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['preflight','prepare','run'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args();out=a.output.resolve();scratch=a.scratch.resolve();(out/'review').mkdir(parents=True,exist_ok=True)
 try:
  c.require(os.environ.get('GITHUB_ACTIONS')=='true','all payload/build/inference access hosted only');cfg=c.source();c.require(c.git('rev-parse','HEAD')==os.environ.get('GITHUB_SHA'),'service/source differs')
  c.write(out/'review/preflight.json',{'source':os.environ['GITHUB_SHA'],'base':c.BASE,'source_sha256':cfg['source_sha256'],'unigram_reviewed':True,'primary_replaced':False})
  if a.phase=='prepare':prepare(out,scratch,cfg)
  elif a.phase=='run':execute(out,scratch,cfg)
 except Exception as error:c.write(out/'review'/('rejection-'+a.phase+'.json'),{'phase':a.phase,'error_type':type(error).__name__,'error':str(error),'qualified':False,'primary_replaced':False});raise
 finally:c.write(out/'review'/('files-'+a.phase+'.json'),{str(f.relative_to(out)):{'sha256':c.sha(f.read_bytes()),'bytes':f.stat().st_size}for f in out.rglob('*')if f.is_file()and f.name!='files-'+a.phase+'.json'})
if __name__=='__main__':main()
