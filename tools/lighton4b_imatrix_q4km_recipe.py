#!/usr/bin/env python3
"""Native HF-layout weighted Q4_K_M-style schedule; not publisher-byte equivalence."""
import argparse,hashlib,json
from pathlib import Path
from lighton4b_q6_recipe import plan as old_plan
from lighton4b_imatrix_adapter import inventory
RECIPE='lighton4b-weighted-q4km-style-q6head-v1'
SOURCE_F16_SHA='757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606'
IMATRIX_SHA='10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7'
def plan():
 p=old_plan();rows=[dict(r)for r in p['all_tensors']]
 for r in rows:
  if r['name']=='l.embed_tokens.weight':r['type']=14
 names=sorted(r['name']for r in rows if r['type']==14);counts={k:sum(r['type']==k for r in rows)for k in(0,12,14)}
 if len(names)!=33 or counts!={0:474,12:216,14:33}:raise ValueError('exact474/216/33 schedule required')
 mapping=inventory();up={r['target']:name for name,r in mapping.items()};up['l.embed_tokens.weight']='token_embd.weight'
 return {'schema':'lighton4b-imatrix-q4km-style-recipe/v1','recipe':RECIPE,'general_file_type':15,'all_tensors':rows,'q6_names':names,'q6_names_sha256':hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest(),'upstream_q6_names':sorted(up[n]for n in names),'types':{str(k):v for k,v in counts.items()},'expected_imatrix_applications':248,'q6_body_imatrix_applications':32,'q4_body_imatrix_applications':216,'q6_tied_head_imatrix':'absent in source; intentionally unweighted Q6, never silent missing-body fallback','source_f16_sha256':SOURCE_F16_SHA,'imatrix_source_sha256':IMATRIX_SHA,'extra_payload_bytes_vs_plain_q4_q8head':10946560,'coordinate_profile':'HF grouped V heads;24 inverse input-stat vectors; no publisher quant-block repack','publisher_byte_equivalence':False,'quality_qualified':False,'heldout_selection':False}
def validate_publisher_header(header):
 p=plan();actual=sorted(n for n,t in header['tensors'].items()if t['type']==14)
 if actual!=p['upstream_q6_names']:raise ValueError('publisher33-Q6 named schedule differs')
 if len(header['tensors'])!=426 or sum(t['type']==12 for t in header['tensors'].values())!=216:raise ValueError('publisher full decoder schedule differs')
 return True
def quantizer_arguments(source,output,adapted):
 return [str(source),str(output),'q4_k','--lighton4b-imatrix-q4km-style','--imatrix',str(adapted)]
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True,type=Path);a.add_argument('--publisher-header',type=Path);v=a.parse_args()
 if v.publisher_header:validate_publisher_header(json.loads(v.publisher_header.read_text()))
 v.output.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
