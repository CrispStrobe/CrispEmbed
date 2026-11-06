#!/usr/bin/env python3
"""Fixed calibration-only24-GDN-output Q6 plan, directly from original BF16."""
import copy
import hashlib
from lighton4b_q6_recipe import plan as original_plan
RECIPE='lighton4b-q4k-gdn-out-q6-v1'
def plan():
 p=copy.deepcopy(original_plan());p['recipe']=RECIPE;p['schema']='lighton4b-gdn-output-q6-recipe/v1';p['upstream']={'policy':'24 recurrent GDN output projections; separate from preserved upstream-style32-Q6 recipe','selection':'layer%4!=3, fixed before new calibration measurements','exception':'Q8_0 tied embedding/output preserved'}
 names=[];counts={0:0,8:0,12:0,14:0}
 for row in p['all_tensors']:
  if row['type']in(12,14):row['type']=14 if row['name'].endswith('.linear_attn.out_proj.weight')else 12
  if row['type']==14:names.append(row['name'])
  counts[row['type']]+=1
 if counts!={0:474,8:1,12:224,14:24}:raise ValueError('fixed24Q6/224Q4/Q8head/protected inventory differs')
 p.update(q6_names=sorted(names),q6_names_sha256=hashlib.sha256(('\n'.join(sorted(names))+'\n').encode()).hexdigest(),types={str(k):v for k,v in counts.items()},extra_payload_bytes_vs_plain_q4=64880640,metadata_strings={'qwen35_ocr.mixed_quant_base_sha256':'f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91'},metadata_uint32={'qwen35_ocr.mixed_quant_selected_count':24,'qwen35_ocr.mixed_quant_payload_delta':64880640})
 return p
