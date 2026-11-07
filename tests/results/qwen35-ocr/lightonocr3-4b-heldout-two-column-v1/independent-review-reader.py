"""Private report-only independent review; prints no recognized text or IDs."""
from pathlib import Path
import sys,json,hashlib,math,importlib.util
import numpy as np
repo=Path(sys.argv[3]);p=Path(sys.argv[1]);dest=Path(sys.argv[2]);r=json.loads((p/'heldout-result.json').read_text());assert r['qualified']is False and r['status']=='measured_provisional_heldout'and r['source_sha']=='d378c6cc055e3a31b0d4cc9dad54a5b41f10d792';case=r['case'];cfg=json.loads((repo/'docs/lighton4b-heldout-v1.json').read_text());selected=next(c for c in cfg['cases']if c['id']==case);gold=json.loads((repo/selected['gold']).read_text())
s=importlib.util.spec_from_file_location('ex',repo/'tools/heldout/execution.py');ex=importlib.util.module_from_spec(s);s.loader.exec_module(ex)
def dist(a,b):
 if not a:return len(b)
 masks={}
 for i,c in enumerate(a):masks[c]=masks.get(c,0)|(1<<i)
 full=(1<<len(a))-1;vp=full;vn=0;top=1<<(len(a)-1);score=len(a)
 for c in b:
  eq=masks.get(c,0);xv=eq|vn;xh=(((eq&vp)+vp)^vp)|eq;ph=vn|~(xh|vp);mh=vp&xh
  if ph&top:score+=1
  elif mh&top:score-=1
  ph=(ph<<1)|1;mh<<=1;vp=(mh|~(xv|ph))&full;vn=(ph&xv)&full
 return score
def naive(a,b):
 d=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  q=[i]
  for j,y in enumerate(b,1):q.append(min(q[-1]+1,d[j]+1,d[j-1]+(x!=y)))
  d=q
 return d[-1]
import random
rng=random.Random(8117)
for _ in range(200):
 a=[rng.randrange(4)for i in range(rng.randrange(20))];b=[rng.randrange(4)for i in range(rng.randrange(20))];assert dist(a,b)==naive(a,b)
verified={}
for f in p.rglob('*'):
 if f.is_file()and str(f.relative_to(p))in r['artifacts']:
  n=str(f.relative_to(p));meta=r['artifacts'][n];assert f.stat().st_size==meta['bytes']and ex.sha(f)==meta['sha256'];verified[n]=meta
metrics={};ids_by_arm={}
for arm,a in r['arms'].items():
 text=(p/arm/'generated.txt').read_bytes().decode();ids=list(map(int,(p/arm/'generated.ids').read_text().split()));assert ids==a['ids']and text==a['decoded']and 0<len(ids)<=1536 and a['complete']==(ids[-1]in[248044,248046]);assert ex.score(gold,text,a['complete'])==a['score'];score=a['score'];content=score.get('parsed_text');want=' '.join(gold['text'].split());metrics[arm]=dict(tokens=len(ids),eos=a['complete'],raw_char_edits=dist(gold['text'],text),raw_char_total=len(gold['text']),raw_word_edits=dist(gold['text'].split(),text.split()),raw_word_total=len(gold['text'].split()),parse_ok=score['parse_ok'],content_char_edits=dist(want,content)if content is not None else None,content_char_total=len(want),content_word_edits=dist(want.split(),content.split())if content is not None else None,content_word_total=len(want.split()),table=score['table'],table_shape_valid=score.get('table_shape_valid'));ids_by_arm[arm]=ids
for key in ['tokens','content','completion_probabilities','stop_type','tokens_evaluated','truncated']:
 off=json.loads((p/'upstream-off/response.json').read_text());on=json.loads((p/'upstream-on/response.json').read_text());assert off[key]==on[key]
source=np.fromfile(p/'upstream-on/first-score.f32',dtype='<f4');firsts={};merged={}
for arm in ['native-composed','native-source-features']:
 native=np.fromfile(p/arm/'prefill.logits.f32',dtype='<f4');assert source.size==native.size==248320 and np.isfinite(source).all()and np.isfinite(native).all();d=native.astype('f8')-source;firsts[arm]={'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max())};old=r['arms'][arm]['first_score_vs_source'];assert firsts[arm]['max_abs']==old['max_abs']and abs(firsts[arm]['rms']-old['rms'])<=2*math.ulp(old['rms']);x=(p/arm/'vision.merged.f32').read_bytes();y=(p/'upstream-on/source.vision.0.f32').read_bytes();assert len(x)==len(y);merged[arm]={'bytes':len(x),'bit_exact':x==y};assert merged[arm]['bit_exact']==r['arms'][arm]['merged_vs_source']['bit_exact']
x=ids_by_arm['upstream-off'];y=ids_by_arm['native-composed'];firstdiff=next((i for i,(a,b)in enumerate(zip(x,y))if a!=b),min(len(x),len(y)));same_rows=r['arms']['native-composed']['raw_rows']['rows']==r['arms']['native-source-features']['raw_rows']['rows']
result=dict(schema='lighton4b-provisional-independent-case-review/v1',case=case,source_sha=r['source_sha'],receipt_sha256=ex.sha(p/'heldout-result.json'),selected_owned_files=verified,metrics=metrics,source_observer_full_neutral_independently_verified=True,firstscore_recomputed=firsts,merged_vectors=merged,native_raw_row_hashes_equal_secondary=same_rows,native_ids_equal_secondary=ids_by_arm['native-composed']==ids_by_arm['native-source-features'],first_native_source_id_difference=firstdiff,primary_floor=ex.floor(r['arms']['native-composed']['score'],r['arms']['upstream-off']['score'],case=='inventory-table'),qualified=False,raw_output_publication_allowed=case!='two-column-page',reference_limits='Provisional prior transcription; no expert human adjudication. Owned inventory literal is authored, not representative OCR qualification.',independent_distance_implementation='Bit-parallel Levenshtein; crosschecked 200 short random list cases against independent DP.',large_raw_score_archives_remain_remote=True)
dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in['selected_owned_files']},indent=2))
