#!/usr/bin/env python3
"""macOS arm64 desktop drift attribution v1 (hosted only, audit-only).

Runs the exact successor-v4 packaged CLI (producer run 40724938611) on the
fixture text with its default backend selection and with CRISPEMBED_FORCE_CPU=1,
and compares both against the stored v4 consumer vectors (run 41587063385) of
linux-x86_64, linux-arm64 and macos-arm64. No gate, default or release changes.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/ci-heavy'));sys.path.insert(0,str(ROOT/'tools'))
from successor_assets import MODEL_ALIAS, MODEL_SHA, TEXT, artifact, sha
import math
V4_RUN=41587063385
REFS=('linux-x86_64','linux-arm64','macos-arm64')

def metrics(a,b):
 if len(a)!=384 or len(b)!=384:raise ValueError('shape')
 e=max(abs(x-y) for x,y in zip(a,b));aa=sum(x*x for x in a);bb=sum(y*y for y in b)
 return {'max_abs':e,'cosine':sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb),
  'strict_pass':e<=.002 and sum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb)>=.99999}

def references(out):
 refs={}
 for name in REFS:
  z=out/('ref-'+name+'.zip');ident=artifact(V4_RUN,'successor-consumer-'+name+'-'+str(V4_RUN),z,50_000_000)
  with zipfile.ZipFile(z) as f:data=json.loads(f.read('consumer.json'))
  if data.get('model_sha256')!=MODEL_SHA or data.get('text')!=TEXT:raise ValueError('reference fixture differs')
  refs[name]={'artifact':ident,'vector':data['vector']}
 return refs

def run_cli(binary,cache,arm,repeat,out):
 env=dict(os.environ,CRISPEMBED_CACHE_DIR=str(cache),CRISPEMBED_THREADS='2')
 env.pop('CRISPEMBED_FORCE_CPU',None)
 if arm=='force_cpu':env['CRISPEMBED_FORCE_CPU']='1'
 r=subprocess.run([str(binary),'-m',MODEL_ALIAS,'-t','2','--json',TEXT],env=env,text=True,encoding='utf-8',capture_output=True,timeout=900)
 (out/(arm+'-'+str(repeat)+'.log')).write_text(r.stderr,encoding='utf-8')
 if r.returncode:raise ValueError(arm+' CLI failed '+str(r.returncode))
 rows=json.loads(r.stdout)
 if len(rows)!=1 or rows[0]['text']!=TEXT:raise ValueError('CLI response invalid')
 backend=[l.strip() for l in r.stderr.splitlines() if l.startswith('crispembed: using ') or l.startswith('crispembed: forcing')]
 return rows[0]['embedding'],backend

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted only')
 out=a.output;receipt=json.loads((out/'consumer.json').read_text(encoding='utf-8'))
 binary=list(out.glob('crispembed-*/crispembed'))
 if len(binary)!=1:raise ValueError('Expected one package')
 binary=binary[0].resolve();binary.chmod(0o700)
 for f in binary.parent.glob('*'):
  if f.is_file():f.chmod(0o700)
 cache=(out/'model-cache').resolve()
 refs=references(out);arms={}
 for arm in ('default','force_cpu'):
  vecs=[];backend=None
  for k in range(2):
   v,backend=run_cli(binary,cache,arm,k,out);vecs.append(v)
  models=list(cache.rglob('all-MiniLM-L6-v2-iq4_xs.gguf'))
  if len(models)!=1 or sha(models[0])!=MODEL_SHA:raise ValueError('model hash differs')
  arms[arm]={'backend_lines':backend,'repeat_identical':vecs[0]==vecs[1],
   'vs_stored_v4':{n:metrics(vecs[0],refs[n]['vector']) for n in REFS},'vector':vecs[0]}
 arms['force_cpu']['vs_default']=metrics(arms['force_cpu']['vector'],arms['default']['vector'])
 result={'schema':'crispembed.macos-desktop-backend-ab.v1','label':a.label,'runner':{'system':platform.system(),
  'machine':platform.machine(),'release':platform.release(),'image':os.environ.get('ImageOS'),'image_version':os.environ.get('ImageVersion')},
  'package':receipt.get('archives'),'producer_run':receipt.get('producer_run'),'stored_v4_run':V4_RUN,
  'references':{n:refs[n]['artifact'] for n in REFS},'model_sha256':MODEL_SHA,'text':TEXT,'arms':arms,
  'gates_changed':False,'defaults_changed':False}
 (out/'ab.json').write_text(json.dumps(result,indent=1)+'\n',encoding='utf-8')
 for arm,d in arms.items():print(arm,d['backend_lines'],'repeat_identical',d['repeat_identical'],
  {n:(m['max_abs'],m['cosine']) for n,m in d['vs_stored_v4'].items()})

if __name__=='__main__':main()
