#!/usr/bin/env python3
"""Artifact-only remote window metrics; no payload values, IDs or model calls."""
import argparse,importlib.util,json
from pathlib import Path
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('receipt_window_arrays',ROOT/'tools/attention252-probe/worker.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
SOURCE_SHA='9761e11b911ebc8a51ab3e4e49d286d43a697794d62131beecc64277591db25c';FIXED_SHA='8001a29fab5abc3481e601168e10ca4383d10060ad7e4356a9110eb345c0b56a'
WINDOWS=[(224,232),(232,234),(234,242)]
def summarize(source_report,fixed_report):
 import numpy as np
 a.require(a.sha(source_report)==SOURCE_SHA and a.sha(fixed_report)==FIXED_SHA,'exact completed source/fixed reports');s=json.loads(source_report.read_text());f=json.loads(fixed_report.read_text());a.require(s['status']=='measured_neutral_receipt-prefill_source'and s['firstscore_observer_neutral'],'completed source neutrality')
 def source(g,name):
  n='on/graph-%d.%s.bin'%(g,name);a.owned(source_report.parent,n,s['artifacts']);return a.source_array(source_report.parent/'on',s['graph_indexes'][g],name)
 def stats(x):
  x=x.astype('f8');a.require(x.size and np.isfinite(x).all(),'finite window');return dict(elements=int(x.size),rms_magnitude=float(np.sqrt(np.mean(x*x))),max_abs=float(np.abs(x).max()))
 r=dict(schema='lighton4b-receipt-artifact-window/v1',source_report_sha256=SOURCE_SHA,fixed_report_sha256=FIXED_SHA,model_forward_executed=False,source_stages={},hidden0=[])
 for stage in ['attn_norm','linear_attn_qkv_mixed','z','alpha','beta','conv_output_raw','conv_output_silu','q_conv_predelta','k_conv_predelta','v_conv_predelta','final_output','linear_attn_out','attn_residual','attn_post_norm','ffn_gate','ffn_up','ffn_swiglu','ffn_out']:
  x=source(2,stage+'-0');a.require(x.size%242==0,'actual image stage rows');x=x.reshape(242,-1);r['source_stages'][stage]=[dict(image_rows=[begin,end],**stats(x[begin:end]))for begin,end in WINDOWS]
 state=source(2,'state_predelta-0');r['source_incoming_recurrent_state']=dict(shape=list(state.shape),**stats(state));name='historical-receipt/on/prefill.layer.0.f32';p=a.owned(fixed_report.parent,name,f['artifacts']);native=np.fromfile(p,dtype='<f4').reshape(256,2560)[4:246];ref=source(2,'l_out-0').reshape(242,2560)
 for begin,end in [(0,224),*WINDOWS]:r['hidden0'].append(dict(image_rows=[begin,end],**a.delta(native[begin:end],ref[begin:end])))
 r['limits']=['Source-only inner-stage magnitudes are context, not source/native operator error.','Native hidden0 ranges are retained actual outputs; same-input projection/component probes remain necessary.'];return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.write_text(json.dumps(summarize(args.source,args.fixed),indent=2)+'\n')
