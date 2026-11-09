"""Hash and score only selected completed-case captures; never publishes text/IDs."""
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
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

def main():
 p=argparse.ArgumentParser();p.add_argument('--new',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--sha',required=True);a=p.parse_args()
 sys.path.insert(0,str(a.repo/'tools'));import ocr_quality as q
 sys.path.insert(0,str(a.repo/'tools/planner-heldout'));import contracts as c
 cfg=json.loads((a.repo/'docs/lighton4b-planner-heldout-v1.json').read_text());c.admit(cfg)
 r=json.loads((a.new/'planner-heldout.json').read_text());old=json.loads((a.old/'heldout-result.json').read_text());need(sha(a.new/'planner-heldout.json')==a.sha and sha(a.old/'heldout-result.json')==cfg['old_case_receipts'][r['case']],'exact completed receipt identities')
 need(r['source_sha']=='a6164205620c71427a39c97ab386591406c4ca73'and r['status']=='measured_provisional_planner_heldout'and r['qualified']is False and r['previous_corrected_receipt_sha256']==cfg['previous_case_receipts'][r['case']],'fixed planner producer and prior corrected baseline');selection=json.loads((a.repo/'docs/lighton4b-heldout-v1.json').read_text());case=next(x for x in selection['cases']if x['id']==r['case']);gold=json.loads((a.repo/case['gold']).read_text());want=q.words(gold['text'])
 result=dict(status='independent-selected-data-pass',case=r['case'],receipt_sha256=a.sha,old_receipt_sha256=sha(a.old/'heldout-result.json'),qualified=False,references_expert_adjudicated=False,selected_files={},arms={},references={},local_numpy=np.__version__,method='Separate bit-parallel Levenshtein, frozen content parser; raw F32 byte hashes exact. RMS scalar reductions may differ by at most two double ULPs.')
 def counts(text):
  parsed,tables=q.parse_document(text);out={'content':{'char_edits':dist(want,parsed),'char_count':len(want),'word_edits':dist(want.split(),parsed.split()),'word_count':len(want.split())},'raw':{'char_edits':dist(gold['text'],text),'char_count':len(gold['text']),'word_edits':dist(gold['text'].split(),text.split()),'word_count':len(gold['text'].split())}}
  if gold['tables']is not None:out['literal_tables_exact']=tables==gold['tables'];out['expected_cells']=sum(len(row)for table in gold['tables']for row in table)
  return out
 for arm,v in r['arms'].items():
  for name in ['generated.ids','generated.txt','native-summary.json','prefill.logits.f32','vision.merged.f32']:
   f=a.new/arm/name;row=r['artifacts'][arm+'/'+name];need(f.stat().st_size==row['bytes']and sha(f)==row['sha256'],'actual selected output bytes');result['selected_files'][arm+'/'+name]=row
  ids=list(map(int,(a.new/arm/'generated.ids').read_text().split()));text=(a.new/arm/'generated.txt').read_bytes().decode();summary=json.loads((a.new/arm/'native-summary.json').read_text());need(ids==v['ids']and text==v['decoded']and summary==v['summary'],'actual output identity');c.native_complete(summary,ids,arm=='corrected-source-features');need(v['complete']==summary['eos'],'actual completion flag')
  m=counts(text)
  for view in ['content','raw']:
   need(m[view]['char_edits']/m[view]['char_count']==v['score'][view]['cer']and m[view]['word_edits']/m[view]['word_count']==v['score'][view]['wer'],'independent actual edit distance')
  result['arms'][arm]=dict(tokens=len(ids),eos=summary['eos'],text_sha256=sha(a.new/arm/'generated.txt'),ids_sha256=sha(a.new/arm/'generated.ids'),**m)
 for arm in ['upstream-off','upstream-on']:result['references'][arm]=dict(tokens=len(old['arms'][arm]['ids']),eos=old['arms'][arm]['complete'],**counts(old['arms'][arm]['decoded']))
 source=a.old/'upstream-on/first-score.f32';need(sha(source)==old['artifacts']['upstream-on/first-score.f32']['sha256'],'historical source first row hash');srow=np.fromfile(source,dtype='<f4').astype('f8');need(srow.size==248320 and np.isfinite(srow).all(),'source vocabulary shape/finite')
 feature=a.old/'upstream-on/source.vision.0.f32';need(sha(feature)==old['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'historical source feature hash')
 for arm,v in r['arms'].items():
  x=np.fromfile(a.new/arm/'prefill.logits.f32',dtype='<f4').astype('f8');need(x.size==248320 and np.isfinite(x).all(),'native vocabulary shape/finite');d=x-srow;rms=float(np.sqrt(np.mean(d*d)));maximum=float(abs(d).max());need(maximum==v['first_score_vs_source']['max_abs'],'actual max error');ulp=abs(rms-v['first_score_vs_source']['rms'])/math.ulp(v['first_score_vs_source']['rms']);need(ulp<=2,'RMS scalar reduction agreement');result['arms'][arm]['firstscore']=dict(rms=rms,max_abs=maximum,rms_double_ulp_difference=ulp,candidate_max_abs=float(abs(x).max()),source_max_abs=float(abs(srow).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),source_rms_magnitude=float(np.sqrt(np.mean(srow*srow))));need((a.new/arm/'vision.merged.f32').read_bytes()==feature.read_bytes(),'actual merged upstream byte identity')
 result['all_merged_upstream_bit_exact']=True;result['corrected_ids_equal']=r['arms']['corrected-composed']['ids']==r['arms']['corrected-source-features']['ids'];result['corrected_text_equal']=r['arms']['corrected-composed']['decoded']==r['arms']['corrected-source-features']['decoded'];result['old_vs_corrected_content_char_edits']=dist(q.parse_document(r['arms']['baseline-old']['decoded'])[0],q.parse_document(r['arms']['corrected-composed']['decoded'])[0])
 primary=result['arms']['corrected-composed'];reference=result['references']['upstream-off'];floor=primary['eos']and reference['eos']and all(primary['content'][key]<=reference['content'][key]for key in ['char_edits','word_edits'])
 if gold['tables']is not None:floor=floor and primary['literal_tables_exact']
 need(floor==r['primary_meets_provisional_source_floor'],'independent floor decision');result['source_floor_met']=bool(floor);(a.new/'independent-selected-review.json').write_text(json.dumps(result,indent=2)+'\n');print('Selected15files/independent edits/3vectors/merged/EOS PASS',r['case'],primary['content'],'source',reference['content'],'floor',floor)
if __name__=='__main__':main()
