#!/usr/bin/env python3
"""Hosted tokenizer-only original-first blueprint. No model weights or forwards."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path,PurePosixPath
import random
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/moedern-unigram'))
import contract as c
import original as oracle


def api(path):return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=60))
def dependency(pin,target):
 run=api('actions/runs/'+str(pin['run']));require=c.require
 require((run['head_sha'],run['status'],run['conclusion'])==(pin['source'],'completed','success'),'frozen service differs')
 row=api('actions/artifacts/'+str(pin['artifact_id']))
 require(row['workflow_run']['id']==pin['run'] and row['workflow_run']['head_sha']==pin['source'] and not row['expired'] and row['size_in_bytes']==pin['bytes'] and row['digest']=='sha256:'+pin['sha256'] and row['size_in_bytes']<1024**2,'small artifact identity/budget differs')
 with target.open('xb') as f:subprocess.run(['gh','api',f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{pin["artifact_id"]}/zip'],stdout=f,check=True,timeout=90)
 require(target.stat().st_size==pin['bytes'] and c.sha(target.read_bytes())==pin['sha256'],'small ZIP identity differs')
 with zipfile.ZipFile(target) as z:
  infos=z.infolist();require(len(infos)<100 and sum(i.file_size for i in infos)<8*1024**2 and len({i.filename for i in infos})==len(infos),'small ZIP budget/duplicate differs')
  require(all(not PurePosixPath(i.filename).is_absolute() and '..' not in PurePosixPath(i.filename).parts and '\\' not in i.filename for i in infos),'unsafe ZIP member')
  data=z.read(pin['member'])
 require(c.sha(data)==pin['member_sha256'],'review member differs')
 return json.loads(data),{k:row[k] for k in ('id','name','size_in_bytes','digest')}


def fetch(url,path,identity):
 c.require(0<identity['bytes']<=2*1024**2,'source/tokenizer budget differs')
 with urllib.request.urlopen(url,timeout=45) as response:data=response.read(identity['bytes']+1)
 c.require(len(data)==identity['bytes'] and c.sha(data)==identity['sha256'],'authenticated source bytes differ: '+path.name)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def prepare(out,cfg):
 primary,pp=dependency(cfg['primary'],out/'primary-review.zip');native,np=dependency(cfg['native_v2'],out/'native-v2-review.zip')
 c.require(primary['completed']is True and primary['github_sha']==cfg['primary']['source'] and len(primary['cases'])==10 and
  native['completed']is True and native['native_parity']is True and native['primary_replaced']is False and not native['failures'],'reviewed outcomes incomplete/different')
 c.require(primary['cases']==cfg['primary_cases'] and primary['tokenizer_descriptor']==cfg['raw_descriptor'],'frozen10cases/raw descriptor differs')
 c.write(out/'review/primary.json',primary);c.write(out/'review/dependencies.json',{'primary':pp,'native_v2':np})
 for name,identity in cfg['source_payloads'].items():
  fetch(f'https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-1.5M/resolve/{c.REV}/{name}',out/'source'/name,identity)
 fetch(f'https://raw.githubusercontent.com/huggingface/sentence-transformers/{c.ST_REV}/sentence_transformers/multi_vector_encoder/model.py',out/'source/st_model.py',cfg['st_registration_source'])
 c.write(out/'review/prepared.json',{'source_head':os.environ['GITHUB_SHA'],'primary_sha256':c.REPORT,'model_weights':False,
  'files':{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':c.sha(p.read_bytes())} for p in (out/'source').glob('*')}})


def original(out,cfg):
 c.require(importlib.metadata.version('tokenizers')=='0.22.2','Rust backend version differs')
 prefix=os.environ.get('CRISPEMBED_MOEDERN_TOKENIZER_VENV');c.require(prefix and Path(sys.prefix).resolve()==Path(prefix).resolve() and sys.prefix!=sys.base_prefix,'isolated Rust-only venv required')
 import tokenizers
 c.require(Path(tokenizers.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'Rust import escaped venv')
 proof=json.loads((out/'review/prepared.json').read_text());c.require(proof['source_head']==os.environ['GITHUB_SHA'],'prepared source differs')
 for name,identity in proof['files'].items():c.require(identity=={'bytes':(out/name).stat().st_size,'sha256':c.sha((out/name).read_bytes())},'prepared source bytes differ')
 raw=json.loads((out/'source/tokenizer.json').read_text());backend=tokenizers.Tokenizer.from_file(str(out/'source/tokenizer.json'))
 role_cfg=json.loads((out/'source/config_sentence_transformers.json').read_text());bert=json.loads((out/'source/sentence_bert_config.json').read_text())
 c.require(bert=={'max_seq_length':511,'do_lower_case':False} and role_cfg['query_prefix']=='[Q] ' and role_cfg['document_prefix']=='[D] ' and role_cfg['query_length']==32 and role_cfg['document_length']==512 and role_cfg['do_query_expansion']is True and role_cfg['attend_to_expansion_tokens']is False,'role migration configuration differs')
 before={p:backend.encode(p,add_special_tokens=False).tokens for p in ['[Q] ','[D] ']}
 actions=oracle.register(backend,out/'source/st_model.py',role_cfg)
 effective=json.loads(backend.to_str());skip=sorted({backend.token_to_id(x) for x in role_cfg['skiplist_words'] if backend.token_to_id(x) not in (None,1)})
 c.write(out/'review/effective-tokenizer.json',{'raw_descriptor':cfg['raw_descriptor'],'effective_added_tokens':effective['added_tokens'],'prefix_before':before,'prefix_after':{p:backend.encode(p,add_special_tokens=False).tokens for p in before},'exact_registration_actions':actions,'normalized_matcher_history':oracle.normalized_matchers(raw,effective),'skip_ids':skip})
 rows=[]
 for case in cfg['primary_cases']:
  value,detail=oracle.tokens(backend,case['text'],case['role'],skip)
  row={'id':case['id'],'role':case['role'],'text':case['text'],'encoded':value,'rust':detail};rows.append(row)
  c.write(out/'review/frozen-ten.json',rows)
  expected={'ids':case['token_ids'],'supplied_mask':case['supplied_attention_mask'],'effective_mask':case['effective_attention_mask'],'scoring_mask':list(map(int,case['scoring_mask'])),'expansion_positions':[int(not m) for m in case['supplied_attention_mask']] if case['role']=='query' else []}
  c.require(value==expected,'original tokenizer-only reproduction differs: '+case['id'])
 fixture=json.loads((ROOT/'tests/fixtures/moedern-unigram-v1.json').read_text())
 for case in fixture['cases']:
  value,detail=oracle.tokens(backend,case['text'],case['role'],skip);rows.append(dict(case,encoded=value,rust=detail))
 c.write(out/'original-gold.json',{'source':os.environ['GITHUB_SHA'],'primary_sha256':c.REPORT,'frozen_ten_exact':True,'cases':rows})
 c.write(out/'review/original-token-streams.json',rows)
 # Production normalization data, sampled from the exact Rust normalizer on ALL scalars.
 # These mappings are tokenizer component data, never model-weight probes/gold.
 c.require(backend.pre_tokenizer.pre_tokenize_str('é')==[('▁é',(0,1))],'Rust pretokenizer offset units differ')
 start=time.monotonic();normalizer=backend.normalizer;mapping={}
 for cp in range(0x110000):
  if 0xd800<=cp<=0xdfff:continue
  char=chr(cp);value=normalizer.normalize_str(char)
  if value!=char:mapping[cp]=value
 sequences=list(fixture['normalization_sequences']);rng=random.Random(fixture['composition_random_seed'])
 pool=list(mapping)+[0x301,0x300,0x315,0x327,0x345,0xf73,0xf75,0x20,0x60,0x27,0x2581]
 for _ in range(fixture['composition_random_cases']):sequences.append(''.join(chr(rng.choice(pool)) for _ in range(rng.randrange(1,9))))
 diagnostics=[]
 for i,text in enumerate(sequences):
  normalized=normalizer.normalize_str(text);derived=oracle.mapped(text,mapping)
  parts=backend.pre_tokenizer.pre_tokenize_str(normalized)
  row={'id':'normal-'+str(i),'text':text,'normalized':normalized,'mapped':derived,'segments':[p for p,_ in parts],
       'rust_original_character_offsets':[list(offset) for _,offset in parts],'offset_unit':'original Unicode characters within normalized input; NOT native source offsets',
       'segment_utf8_boundaries':[[sum(len(c.encode()) for c in p[:k]) for k in range(len(p)+1)] for p,_ in parts]}
  diagnostics.append(row)
  if normalized!=derived:
   c.write(out/'review/normalizer-composition-rejection.json',row);raise ValueError('normalizer scalar-map composition differs; no CPP/model fallback')
 c.write(out/'normalization-gold.json',diagnostics)
 c.write(out/'review/normalizer-summary.json',{'all_unicode_scalars':0x110000-0x800,'changed_scalars':len(mapping),'sequences':len(diagnostics),'composition_exact':True,'generation_seconds':time.monotonic()-start,'original_character_offsets_vs_native_normalized_utf8_only':True})
 metadata=oracle.component(raw,effective,role_cfg,mapping,skip);(out/'tokenizer.gguf').write_bytes(c.gguf(metadata))
 score_bytes=b''.join(struct.pack('<d',x[1]) for x in raw['model']['vocab']);effective_scores=b''.join(struct.pack('<d',x[1]) for x in effective['model']['vocab'])
 # Rust's loaded/serialized F64 scores are authoritative, not a lossy F32 cast or a different JSON parser.
 c.require(c.read_gguf((out/'tokenizer.gguf').read_bytes())['moedern.tokenizer.scores_f64'][1]==[x[1]for x in effective['model']['vocab']],'actual Rust scores differ from component')
 parsed=c.read_gguf((out/'tokenizer.gguf').read_bytes());c.require(b''.join(struct.pack('<d',x)for x in parsed['moedern.tokenizer.scores_f64'][1])==effective_scores,'serialized scoreF64 bytes differ')
 units=[]
 for name,a_score,aa_score in [('ties',0.0,0.0),('precision',0.1,0.19999999999999998)]:
  toy=deepcopy(metadata);vocab=['<pad>','<unk>','[CLS]','[SEP]','[MASK]','a','aa']+['__owned_unused_'+str(i)+'__'for i in range(7,30000)]+['[Q] ','[D] ']
  scores=[-1000.0]*30000;scores[5]=a_score;scores[6]=aa_score
  model=tokenizers.models.Unigram(list(zip(vocab[:30000],scores)),unk_id=1,byte_fallback=False)
  toy['tokenizer.ggml.tokens']=vocab;toy['moedern.tokenizer.scores_f64']=('f64',scores);toy['moedern.tokenizer.owned_toy']=name
  target='toy-'+name+'.gguf';(out/target).write_bytes(c.gguf(toy))
  units.append({'id':name,'component':target,'cases':[{'id':name+'-'+str(i),'text':text,'ids':[t.id for t in model.tokenize(text)]}for i,text in enumerate(['aa','aaa','aaaa','xyz','axyaa'])]})
 c.write(out/'unigram-unit-gold.json',units)

 c.write(out/'review/original.json',{'schema':'crispembed-moedern-unigram-original-v1','completed':True,'source':os.environ['GITHUB_SHA'],'primary_sha256':c.REPORT,
  'frozen_ten_exact':True,'cases':len(rows),'normalization_sequences':len(diagnostics),'normalizer_map_entries':len(mapping),'scores_count':30000,'score_f64_sha256':c.sha(effective_scores),'source_json_python_parsed_score_f64_sha256':c.sha(score_bytes),'source_json_and_rust_loaded_scores_equal':score_bytes==effective_scores,'tokenizers':tokenizers.__version__,
  'files':{name:{'bytes':(out/name).stat().st_size,'sha256':c.sha((out/name).read_bytes())} for name in ['original-gold.json','normalization-gold.json','tokenizer.gguf','unigram-unit-gold.json','toy-ties.gguf','toy-precision.gguf']},'model_weights':False,'native_executed':False})


def inputs(path,rows):path.write_text(''.join(r['id']+'\t'+r.get('role','document')+'\t'+r['text'].encode().hex()+'\n' for r in rows))
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def command(args,log,expected=0):
 with log.open('w') as f:r=subprocess.run(list(map(str,args)),cwd=ROOT,env=dict(os.environ,CRISPEMBED_MOEDERN_TEXT_FRONTEND='1'),stdout=f,stderr=subprocess.STDOUT,timeout=120)
 c.require(r.returncode==expected,'command exit differs: '+log.name)


def native(out,build,cfg):
 proof=json.loads((out/'review/original.json').read_text());c.require(proof['completed']is True and proof['frozen_ten_exact']is True and proof['source']==os.environ['GITHUB_SHA'] and proof['primary_sha256']==c.REPORT and proof['native_executed']is False,'original-first barrier differs')
 c.require(set(proof['files'])=={'original-gold.json','normalization-gold.json','tokenizer.gguf','unigram-unit-gold.json','toy-ties.gguf','toy-precision.gguf'},'original file scope differs')
 for name,identity in proof['files'].items():c.require(identity=={'bytes':(out/name).stat().st_size,'sha256':c.sha((out/name).read_bytes())},'gold/component bytes differ')
 c.require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['ggml'],'actual ggml checkout differs')
 gold=json.loads((out/'original-gold.json').read_text());diagnostics=json.loads((out/'normalization-gold.json').read_text());component=out/'tokenizer.gguf';ordinary=build/'moedern-text-plain-probe';tested=build/'moedern-text-tested-probe'
 inputs(out/'inputs.tsv',gold['cases']);command([tested,'--gates',component,out/'inputs.tsv'],out/'review/gates.log')
 for name,binary in [('ordinary',ordinary),('tested',tested)]:command([binary,component,out/'inputs.tsv',out/(name+'.jsonl')],out/'review'/(name+'.log'))
 c.require((out/'ordinary.jsonl').read_bytes()==(out/'tested.jsonl').read_bytes(),'ordinary/test method neutrality failed')
 actual=read(out/'ordinary.jsonl');failures=[];c.require(len(actual)==len(gold['cases']),'native case count differs')
 for row,expected in zip(actual,gold['cases']):
  if row['id']!=expected['id'] or row['accepted']is not True or row.get('encoded')!=expected['encoded']:failures.append({'id':expected['id'],'expected':expected,'actual':row})
 c.write(out/'review/native-cases.json',actual);c.write(out/'review/parity-failures.json',failures)
 inputs(out/'normalize.tsv',diagnostics);command([tested,'--normalize',component,out/'normalize.tsv',out/'normalize.jsonl'],out/'review/normalization.log')
 observed=read(out/'normalize.jsonl');c.require(len(observed)==len(diagnostics),'normalization count differs')
 for row,expected in zip(observed,diagnostics):
  if row['id']!=expected['id'] or row['normalized']!=expected['normalized'] or row['segments']!=expected['segments'] or row['segment_utf8_boundaries']!=expected['segment_utf8_boundaries']:failures.append({'kind':'normalizer/pretokenizer','expected':expected,'actual':row})
 c.write(out/'review/parity-failures.json',failures)
 units=json.loads((out/'unigram-unit-gold.json').read_text());unit_results=[]
 for unit in units:
  inputs(out/('unit-'+unit['id']+'.tsv'),unit['cases']);result=out/('unit-'+unit['id']+'.jsonl')
  command([tested,'--segments',out/unit['component'],out/('unit-'+unit['id']+'.tsv'),result],out/'review'/('unit-'+unit['id']+'.log'))
  actual_unit=read(result);unit_results.append({'id':unit['id'],'expected':unit['cases'],'actual':actual_unit,'owned_synthetic_tokenizer':True});c.write(out/'review/unigram-unit-results.json',unit_results)
  c.require(len(actual_unit)==len(unit['cases']),'unit coverage differs')
  for actual_row,expected in zip(actual_unit,unit['cases']):c.require(actual_row['id']==expected['id'] and actual_row['segment_ids']==expected['ids'],'actual Unigram unknown-fusion/tie/F64 unit failed')
 c.write(out/'review/unigram-unit-results.json',unit_results)
 metadata=c.read_gguf(component.read_bytes());bad_metadata=[]
 mutations={'revision':('moedern.tokenizer.revision','wrong'),'flags':('moedern.tokenizer.added_flags',[0]+metadata['moedern.tokenizer.added_flags'][1:]),'matcher-id':('moedern.tokenizer.normalized_matcher_ids',[30001,30000]),'matcher-content':('moedern.tokenizer.normalized_matcher_content',['[q] ','[d] ']),'score-type':('moedern.tokenizer.scores_f64',['wrong']*30000),'score-nonfinite':('moedern.tokenizer.scores_f64',('f64',[float('nan')]+metadata['moedern.tokenizer.scores_f64'][1][1:]))}
 for name,(key,value)in mutations.items():
  bad=out/('rejected-'+name+'.gguf');bad.write_bytes(c.gguf(dict(metadata,**{key:value})))
  log=out/'review'/('metadata-'+name+'.log');command([tested,bad,out/'inputs.tsv',out/('rejected-'+name+'.jsonl')],log,expected=2)
  expected='source/profile differs'if name=='revision'else 'unsupported effective added token flags'if name=='flags'else 'normalized matcher history differs'if name.startswith('matcher-')else 'array type/budget'if name=='score-type'else 'nonfinite/empty Unigram item'
  c.require(expected in log.read_text(),'metadata rejected at unrelated guard: '+name);bad_metadata.append(name)
 fixture=json.loads((ROOT/'tests/fixtures/moedern-unigram-v1.json').read_text());negative=[]
 for entry in fixture['negative_inputs']:
  raw=bytes.fromhex(entry['hex']) if 'hex' in entry else b'a'*entry['repeat_a'];negative.append((entry,raw))
 with (out/'negative.tsv').open('w') as f:
  for entry,raw in negative:
   first=gold['cases'][0];f.write(entry['id']+'-before\t'+first['role']+'\t'+first['text'].encode().hex()+'\n');f.write(entry['id']+'\t'+entry['role']+'\t'+raw.hex()+'\n');f.write(entry['id']+'-after\t'+first['role']+'\t'+first['text'].encode().hex()+'\n')
 command([tested,component,out/'negative.tsv',out/'negative.jsonl'],out/'review/negative.log');rejected=read(out/'negative.jsonl')
 c.require(len(rejected)==3*len(negative),'negative/recovery count differs')
 for i,(entry,_) in enumerate(negative):
  before,bad,after=rejected[3*i:3*i+3];c.require(before['accepted']and after['accepted']and before['encoded']==after['encoded']==gold['cases'][0]['encoded']and bad['accepted']is False and bad.get('output_empty')is True,'negative/recovery failed: '+entry['id'])
 c.write(out/'review/negative-recovery.json',rejected)
 symbols=subprocess.check_output(['nm','-C',str(build/'libmoedern-text-plain.a')],text=True);testsymbols=subprocess.check_output(['nm','-C',str(build/'libmoedern-text-tested.a')],text=True)
 c.require('text_frontend::test_'not in symbols and 'text_frontend::test_normalize'in testsymbols,'diagnostic symbol isolation failed')
 c.write(out/'review/native.json',{'schema':'crispembed-moedern-unigram-native-v1','completed':True,'tokenizer_parity':not failures,'failures':failures,'cases':len(actual),'frozen_ten_exact':not any(f.get('id')in {x['id']for x in gold['cases'][:10]}for f in failures),'normalization_sequences':len(observed),'negative_cases':len(negative),'metadata_negatives':bad_metadata,'unigram_unit_cases':sum(len(x['cases'])for x in units),'ordinary_test_neutrality':True,'ordinary_excludes_diagnostics':True,'source':os.environ['GITHUB_SHA'],'default_enabled':False,'model_weights':False,'native_text_model_composed':False,'public_api':False,'retrieval_quality':False,'release_qualified':False,
  'files':{p.name:{'sha256':c.sha(p.read_bytes()),'bytes':p.stat().st_size}for p in [component,out/'ordinary.jsonl',out/'tested.jsonl',out/'normalize.jsonl',ordinary,tested]}})
 c.require(not failures,'tokenizer-only parity rejected; all outcomes retained')


def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','original','native'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--build',type=Path);a=p.parse_args();out=a.output.resolve();(out/'review').mkdir(parents=True,exist_ok=True)
 try:
  c.require(os.environ.get('GITHUB_ACTIONS')=='true','hosted tokenizer-only execution required');c.require(c.git('rev-parse','HEAD')==os.environ.get('GITHUB_SHA'),'service/source differs');cfg=c.source()
  c.require(not {'torch','transformers','sentence_transformers'}&set(sys.modules),'no model framework import allowed')
  if a.phase=='prepare':prepare(out,cfg)
  elif a.phase=='original':original(out,cfg)
  else:c.require(a.build is not None,'build path required');native(out,a.build.resolve(),cfg)
 except Exception as error:c.write(out/'review'/('rejection-'+a.phase+'.json'),{'phase':a.phase,'error_type':type(error).__name__,'error':str(error),'model_weights':False,'qualified':False});raise
if __name__=='__main__':main()
