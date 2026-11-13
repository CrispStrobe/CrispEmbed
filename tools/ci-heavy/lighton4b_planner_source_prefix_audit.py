#!/usr/bin/env python3
"""Hosted artifact-only pairwise audit; no protected text or token output."""
import argparse,json,hashlib,zipfile,time,signal
from pathlib import Path
import numpy as np
if not __debug__:raise RuntimeError("run without Python optimization")
signal.alarm(600)
parser=argparse.ArgumentParser(description='Hosted retained source/native prefix audit; no inference.');parser.add_argument('--source',type=Path,required=True);parser.add_argument('--off',type=Path,required=True);parser.add_argument('--on',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
start=time.monotonic();out=args.output;r={'status':'failed','model_forward_executed':False,'qualified':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def locate(path,h):
 assert path.is_file()and path.stat().st_size<4<<20 and sha(path.read_bytes())==h,'exact frozen receipt';return path
def vec(p,rec):
 b=p.read_bytes();assert len(b)==rec['bytes']and sha(b)==rec['sha256'];return np.frombuffer(b,dtype='<f4')
def metrics(a,b):
 assert a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all();a=a.astype('float64');b=b.astype('float64');d=a-b
 return {'rms_error':float(np.sqrt(np.mean(d*d))),'max_error':float(np.max(abs(d))),'off_rms':float(np.sqrt(np.mean(a*a))),'on_rms':float(np.sqrt(np.mean(b*b))),'off_max':float(np.max(abs(a))),'on_max':float(np.max(abs(b)))}
try:
 source=locate(args.source,'9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722');new=locate(args.on,'d2ea232376c2fa4697613d615d33b8b10a4dc0d5ab987fdac7d1ed3d99561760');old=locate(args.off,'f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6');sr=json.loads(source.read_text());nr=json.loads(new.read_text());orr=json.loads(old.read_text());sarm=sr['arms']['upstream-on'];srows=sr['source_raw_rows'];r['pairs']={}
 for label,path,arm,folder in [('off',old,orr['arms']['baseline-old'],'baseline-old'),('on',new,nr['arm'],'planner-on')]:
  limit=next(i for i,(a,b)in enumerate(zip(sarm['ids'],arm['ids']))if a!=b);pair={'common_prefix_tokens':limit,'rows':[]};r['pairs'][label]=pair
  with zipfile.ZipFile(source.parent/'upstream-on'/srows['archive']['file'])as zs,zipfile.ZipFile(path.parent/folder/arm['raw_rows']['archive']['file'])as zn:
   for i in range(limit+1):
    vectors=[]
    for z,rec,token,name in[(zs,srows['rows'][i],sarm['ids'][i],f'source.logits.{i}.f32'),(zn,arm['raw_rows']['rows'][i],arm['ids'][i],f'greedy.{i}.logits.f32')]:
     raw=z.read(name);assert sha(raw)==rec['sha256']and len(raw)==993280;v=np.frombuffer(raw,dtype='<f4');assert np.isfinite(v).all()and int(v.argmax())==token;vectors.append(v)
    m=metrics(*vectors);m.update(position=i,argmax_equal=sarm['ids'][i]==arm['ids'][i]);m['source_rms']=m.pop('off_rms');m['native_rms']=m.pop('on_rms');m['source_max']=m.pop('off_max');m['native_max']=m.pop('on_max')
    for name,v in zip(['source','native'],vectors):
     top=np.partition(v,-2)[-2:];m[name+'_margin']=float(max(top)-min(top));m[name+'_source_winner_score']=float(v[sarm['ids'][i]]);m[name+'_native_winner_score']=float(v[arm['ids'][i]])
    pair['rows'].append(m)
 r['status']='complete';r['receipt_hashes']=[sha(source.read_bytes()),sha(old.read_bytes()),sha(new.read_bytes())]
except Exception as e:r['error']=repr(e)
finally:r['elapsed_seconds']=time.monotonic()-start;out.write_text(json.dumps(r,indent=2))
