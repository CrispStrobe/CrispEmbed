#!/usr/bin/env python3
"""Retained-artifact audit; no checkpoint, inference, or protected text output."""
import argparse,json,hashlib,zipfile,time,signal
from pathlib import Path
import numpy as np
if not __debug__:raise RuntimeError("run without Python optimization")
signal.alarm(600)
parser=argparse.ArgumentParser(description='Artifact-only native OFF/ON common-prefix audit. Run on hosted resources; no model inference.');parser.add_argument('--off',type=Path,required=True);parser.add_argument('--on',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
start=time.monotonic();out=args.output;r={'status':'failed','model_forward_executed':False,'qualified':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def locate(path,h):
 assert path.is_file() and path.stat().st_size<4<<20 and sha(path.read_bytes())==h,'exact frozen receipt';return path
def vec(p,rec):
 b=p.read_bytes();assert len(b)==rec['bytes']and sha(b)==rec['sha256'];return np.frombuffer(b,dtype='<f4')
def metrics(a,b):
 assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all();a=a.astype('float64');b=b.astype('float64');d=a-b
 return {'rms_error':float(np.sqrt(np.mean(d*d))),'max_error':float(np.max(abs(d))),'off_rms':float(np.sqrt(np.mean(a*a))),'on_rms':float(np.sqrt(np.mean(b*b))),'off_max':float(np.max(abs(a))),'on_max':float(np.max(abs(b)))}
try:
 new=locate(args.on,'d2ea232376c2fa4697613d615d33b8b10a4dc0d5ab987fdac7d1ed3d99561760');old=locate(args.off,'f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6');n=json.loads(new.read_text());o=json.loads(old.read_text());a=o['arms']['baseline-old'];b=n['arm'];prefix=next(i for i,(x,y)in enumerate(zip(a['ids'],b['ids']))if x!=y);r['common_prefix_tokens']=prefix;r['rows']=[]
 z0=zipfile.ZipFile(old.parent/'baseline-old'/a['raw_rows']['archive']['file']);z1=zipfile.ZipFile(new.parent/'planner-on'/b['raw_rows']['archive']['file'])
 for i in range(prefix+1):
  xs=[]
  for z,arm in[(z0,a),(z1,b)]:
   raw=z.read(f'greedy.{i}.logits.f32');rec=arm['raw_rows']['rows'][i];assert sha(raw)==rec['sha256']and len(raw)==993280;x=np.frombuffer(raw,dtype='<f4');assert np.isfinite(x).all()and int(x.argmax())==arm['ids'][i];xs.append(x)
  m=metrics(*xs);m['position']=i;m['argmax_equal']=a['ids'][i]==b['ids'][i]
  for label,x in zip(['off','on'],xs):
   top=np.partition(x,-2)[-2:];m[label+'_margin']=float(max(top)-min(top));m[label+'_off_winner_score']=float(x[a['ids'][i]]);m[label+'_on_winner_score']=float(x[b['ids'][i]])
  r['rows'].append(m)
 r['prefill']={}
 names=[k.split('/',1)[1]for k in o['artifacts']if k.startswith('baseline-old/prefill.')and k.endswith('.f32')]
 for name in names:
  oldname='baseline-old/'+name;newname='planner-on/'+name
  if newname not in n['artifacts']:continue
  x=vec(old.parent/oldname,o['artifacts'][oldname]);y=vec(new.parent/newname,n['artifacts'][newname]);m=metrics(x[:min(x.size,y.size)],y[:min(x.size,y.size)]);m.update(off_values=int(x.size),on_values=int(y.size),compared_values=int(min(x.size,y.size)));r['prefill'][name]=m
 r['status']='complete';r['receipt_hashes']=[sha(old.read_bytes()),sha(new.read_bytes())]
except Exception as e:r['error']=repr(e)
finally:r['elapsed_seconds']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2))
