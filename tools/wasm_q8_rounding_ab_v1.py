#!/usr/bin/env python3
"""Versioned source/protocol controls for the conditional rounding A/B lane."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from successor_wasm_trace_contract import (PRODUCER,MODEL_SHA,STAGES,digest,f32_bytes,
 floats,metrics,tokens,stage_rows,replace_one)

BASE='973b200edd5d1d7429edecff1bd3ff42f0d80ca7'
CANDIDATE_GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
OLD_GGML='890278a8342c620197c90e702e1188bcab94f510'
ENCODER_SHA='99bb8a2308021ce997e1570927322d749155c89155dd08af11fdcac6b99e7be1'
TRACE_RUN=40379523009
TRACE_SOURCE='f8bac049a7c143773b330cfc4c217b1f0a47ef56'
NEUTRALITY=('packaged-1-ordinary','packaged-1-dump','packaged-2-ordinary','packaged-2-dump',
 'wasm-packaged','wasm-diagnostic-ordinary','wasm-diagnostic-dump','wasm-pool-final')
GATES=('unset','empty','0','1')
PANEL_SHA='d01c0fcc3046bf1c3a1024f890e1c721852052b2cfc84f37e7b7c4e4f2b03406'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def panel(spec):
 cases=[]
 for item in spec['panel']:
  if 'literal' in item:text=item['literal']
  else:
   recipe=item['repeat']
   if type(recipe['count']) is not int or not 1<=recipe['count']<=1000:raise ValueError('Panel repeat budget')
   text=recipe['separator'].join([recipe['literal']]*recipe['count'])
  if not re.fullmatch('[a-z0-9-]+',item['id']) or '\0' in text:raise ValueError('Panel ID/text invalid')
  cases.append({'id':item['id'],'text':text,'utf8_sha256':digest(text.encode()),'expected_token_count':item.get('expected_token_count')})
 if len(cases)!=10 or len({c['id'] for c in cases})!=10 or cases[0]['id']!='original':raise ValueError('Panel coverage differs')
 if digest(canonical(cases))!=PANEL_SHA or spec.get('panel_sha256')!=PANEL_SHA:raise ValueError('Frozen panel content differs')
 return cases

def reviewed_spec(spec):
 trace=spec['trace']
 if spec['schema']!='crispembed-wasm-q8-rounding-ab-v1' or spec['gates']!={'max_abs':.002,'cosine':.99999}:
  raise ValueError('Protocol/gates differ')
 if trace['run']!=TRACE_RUN or trace['source']!=TRACE_SOURCE or trace.get('reviewed') is not True:
  raise ValueError('First trace has not been independently reviewed and pinned')
 for key in ('review_artifact','raw_artifact'):
  item=trace[key]
  if type(item.get('id')) is not int or item['id']<=0 or type(item.get('bytes')) is not int or not 0<item['bytes']<=32*1024**2 or not re.fullmatch('[0-9a-f]{64}',item.get('sha256') or ''):
   raise ValueError('Missing/invalid reviewed trace artifact identity')
 if not re.fullmatch('[0-9a-f]{64}',trace['review_artifact'].get('report_sha256') or ''):raise ValueError('Missing reviewed trace report SHA')
 panel(spec)
 return spec

def trusted_report(report):
 if report.get('schema')!='crispembed-successor-wasm-source-trace-v1' or report.get('state')!='attribution-complete' or report.get('source')!=TRACE_SOURCE or report.get('producer_source')!=PRODUCER or report.get('model_sha256')!=MODEL_SHA:
  raise ValueError('Completed source-bound first trace required')
 controls=report.get('neutrality',{})
 if set(controls)!=set(NEUTRALITY) or any(c.get('bit_exact') is not True or c.get('sha256')!=c.get('expected_sha256') or not re.fullmatch('[0-9a-f]{64}',c.get('sha256') or '') for c in controls.values()):
  raise ValueError('First trace neutrality incomplete/negative')
 for name in ('packaged-1-dump','packaged-2-dump'):
  rows=report.get('stages',{}).get(name,[])
  if [r['stage'] for r in rows]!=list(STAGES):raise ValueError('First trace stage coverage differs')
  for row in rows:
   if row.get('shape')!=[384,len(report['token_ids'])]:raise ValueError('First trace stage shape differs')
   if not re.fullmatch('[0-9a-f]{64}',row['native_sha256']) or not re.fullmatch('[0-9a-f]{64}',row['wasm_sha256']):raise ValueError('First trace stage hashes absent')
 if not report.get('token_ids') or report['token_ids'][0]!=101 or report['token_ids'][-1]!=102:raise ValueError('First trace token blueprint differs')
 return report

def operator_receipt(source, result, current_source, declaration):
 if (source.get('schema')!='crispembed-wasm-rounding-operator-source-v1'
     or source.get('source')!=current_source or source.get('ggml')!=CANDIDATE_GGML
     or source.get('declaration')!=declaration or source.get('model_access') is not False
     or source.get('qualification') is not False):
  raise ValueError('Operator source receipt differs from the current candidate')
 if (result.get('schema')!='crispembed-wasm-rounding-operator-result-v1'
     or result.get('passed') is not True or result.get('qualification') is not False):
  raise ValueError('Actual operator controls have not passed')
 rows=result.get('controls',[])
 if len(rows)!=5 or rows[0].get('control')!='converter' or any(r.get('passed') is not True for r in rows):
  raise ValueError('Incomplete operator control coverage')
 if ({(r.get('type'),r.get('indirect')) for r in rows[1:]}!={(8,False),(8,True),(20,False),(20,True)}
     or any(r.get('control')!='backend' or r.get('legacy')!=158 or r.get('nearest')!=189 for r in rows[1:])):
  raise ValueError('Actual operator values or backend coverage differ')
 return {'source':source,'result':result}

def git(root,*args):return subprocess.check_output(['git',*args],cwd=root,text=True,encoding='utf-8').strip()

def source_guard(root,runtime_paths):
 root=Path(root)
 declaration_path=root/'docs/wasm-q8-rounding-candidate.json'
 declaration=json.loads(declaration_path.read_text())
 spec=json.loads((root/'docs/wasm-q8-rounding-ab-v1.json').read_text())
 if digest(declaration_path.read_bytes())!=spec['candidate_declaration_sha256']:raise ValueError('Candidate declaration bytes differ')
 if digest((root/'src/crispembed.cpp').read_bytes())!=ENCODER_SHA or declaration['crispembed_encoder_candidate_sha256']!=ENCODER_SHA or git(root,'rev-parse','HEAD:ggml')!=CANDIDATE_GGML:
  raise ValueError('Candidate encoder/gitlink differs')
 ggml=root/'ggml'
 if not (ggml/'.git').exists() or Path(git(ggml,'rev-parse','--show-toplevel')).resolve()!=ggml.resolve() or git(ggml,'rev-parse','HEAD')!=CANDIDATE_GGML:
  raise ValueError('Candidate submodule not initialized at exact commit')
 # Every production object except the two declared changes stays bound to both
 # the immediate pre-candidate base and the immutable40 producer.
 unchanged=[p for p in runtime_paths if p not in ('src','ggml')]
 for base in (BASE,PRODUCER):
  if git(root,'ls-tree',base,'--',*unchanged)!=git(root,'ls-tree','HEAD','--',*unchanged):raise ValueError('Unrelated production/build/binding objects differ')
  def src_rows(revision):
   return [line for line in git(root,'ls-tree','-r',revision,'--','src').splitlines() if not line.endswith('\tsrc/crispembed.cpp')]
  if src_rows(base)!=src_rows('HEAD'):raise ValueError('Unrelated source objects differ')
 if git(root,'status','--porcelain','--',*runtime_paths):raise ValueError('Dirty production source rejected')
 expected=declaration['files']
 if declaration['ggml_base']!=OLD_GGML or declaration['ggml_candidate']!=CANDIDATE_GGML:raise ValueError('Candidate dependency declaration differs')
 changed=git(ggml,'diff','--name-only',OLD_GGML,CANDIDATE_GGML).splitlines()
 if set(changed)!=set(expected):raise ValueError('Unexpected ggml source delta')
 for name,identity in expected.items():
  if digest((ggml/name).read_bytes())!=identity['candidate_sha256']:raise ValueError('Candidate dependency bytes differ: '+name)
 if git(ggml,'status','--porcelain'):raise ValueError('Dirty dependency source rejected')
 return {'base':BASE,'producer':PRODUCER,'encoder_sha256':ENCODER_SHA,'ggml':CANDIDATE_GGML,'dependency_files':expected,'unchanged_paths':unchanged}

def overlay_ab(source):
 """Minimal test-only environment/pooling overlay; no trait-converter probes."""
 source=Path(source);changes={}
 p=source/'wasm/embed_wrapper.c';before=p.read_bytes()
 p.write_bytes(before+b'\n/* A/B test overlay only; never shipped. */\nWASM_EXPORT\nint wasm_trace_setenv(const char * key, const char * value) { return value ? setenv(key, value, 1) : unsetenv(key); }\n')
 changes['wasm/embed_wrapper.c']={'before':digest(before),'after':digest(p.read_bytes())}
 p=source/'src/crispembed.cpp';before=p.read_bytes();text=before.decode()
 statement='''    // Test-only A/B capture of actual pooling buffer.
    if (const char * path = std::getenv("CRISPEMBED_TRACE_POOL")) {
        const std::string file = std::string(path) + "%s";
        FILE * f = fopen(file.c_str(), "wb");
        if (!f || fwrite(pooled.data(), sizeof(float), pooled.size(), f) != pooled.size()) std::abort();
        if (fclose(f) != 0) std::abort();
    }
'''
 anchor='    // L2 normalize\n    auto t_pool = std::chrono::steady_clock::now();'
 text=replace_one(text,anchor,(statement % '.mean.f32')+anchor)
 anchor='\n    return pooled;\n}\n\n// C3:'
 text=replace_one(text,anchor,'\n'+(statement % '.final.f32')+anchor)
 p.write_text(text);changes['src/crispembed.cpp']={'before':digest(before),'after':digest(p.read_bytes())}
 p=source/'build-embed-wasm.sh';before=p.read_bytes();text=before.decode()
 text=replace_one(text,"'_wasm_embed_version',","'_wasm_trace_setenv',\\\n'_wasm_embed_version',")
 text=replace_one(text,'-j$(nproc 2>/dev/null || echo 4)','-j2')
 p.write_text(text);changes['build-embed-wasm.sh']={'before':digest(before),'after':digest(p.read_bytes())}
 return changes
