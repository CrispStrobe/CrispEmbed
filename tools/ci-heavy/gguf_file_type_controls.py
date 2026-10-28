#!/usr/bin/env python3
"""Hosted, model-free actual quantizer/header controls; no framework imports."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_canonicalize import require,parse,sha,span
from gguf_file_type_contract import verify
from ci_process import run
# CLI, tensor encoding, internal legacy header, standardized GGUF header.
METHODS=[('f16',1,1,1),('q4_0',2,2,2),('q4_1',3,3,3),('q5_0',6,8,8),('q5_1',7,9,9),
 ('q8_0',8,7,7),('q2_k',10,10,10),('q3_k',11,11,11),('q4_k',12,12,14),
 ('q5_k',13,13,16),('q6_k',14,14,18),('iq4_nl',20,19,25),('iq4_xs',23,22,30)]
CONTROL_TYPES={0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),
 10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),20:(32,18),23:(256,136)}


def main(args):
    cfg=verify(args.reviewed_source);require(os.environ.get('GITHUB_ACTIONS')=='true','hosted compile controls only')
    args.out.mkdir(parents=True,exist_ok=False);args.build.mkdir(parents=True,exist_ok=False)
    result=dict(schema='crispembed-gguf-file-type-controls-v1',source=args.reviewed_source,completed=False,models_accessed=False,
                commands={},methods=[],publication_qualified=False,source_profile_sha256=sha(ROOT/'docs/gguf-standard-file-type-v1.json'))
    def save():(args.out/'controls.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def command(label,cmd):
        log=args.out/(label+'.log');p=run(cmd,log,cwd=ROOT,timeout=1200)
        result['commands'][label]=dict(p,log=log.name,log_sha256=sha(log));save()
        require(not p['timed_out']and p['exit']==0,'actual compiler/quantizer control rejected: '+label)
    save()
    try:
        command('configure',['cmake','-S',ROOT/'tests/gguf-file-type','-B',args.build,'-DCMAKE_BUILD_TYPE=Release'])
        command('build',['cmake','--build',args.build,'--parallel','2'])
        fixture=args.build/'gguf-file-type-fixture';quantizer=args.build/'gguf-file-type-quantize';source=args.out/'authored-f32.gguf'
        command('write',[fixture,'write',source])
        for name,tensor,legacy,standard in METHODS:
            a=args.out/(name+'-legacy.gguf');b=args.out/(name+'-standard.gguf')
            command(name+'-legacy',[quantizer,source,a,name,'--legacy-file-type-header'])
            command(name+'-standard',[quantizer,source,b,name])
            command(name+'-legacy-check',[fixture,'check',a,str(tensor),str(legacy)])
            command(name+'-standard-check',[fixture,'check',b,str(tensor),str(standard)])
            x,y=parse(a,CONTROL_TYPES),parse(b,CONTROL_TYPES)
            require(x['tensors']==y['tensors']and x['bytes']==y['bytes'],'header policy changed tensor/layout')
            offset=x['metadata']['general.file_type']['offset'];require(y['metadata']['general.file_type']['offset']==offset,'field moved')
            with a.open('rb')as left,b.open('rb')as right:
                require(span(left,0,offset)==span(right,0,offset)and span(left,offset+4,x['bytes']-offset-4)==span(right,offset+4,y['bytes']-offset-4),'non-header bytes changed')
            result['methods'].append(dict(method=name,tensor_type=tensor,legacy_header=legacy,standard_header=standard,
                legacy_sha256=sha(a),standard_sha256=sha(b),all_non_header_bytes_exact=True));save()
        result.update(completed=True,actual_cli_methods=13,all_payloads_exact=True);save()
    except Exception as error:result['error']=type(error).__name__+': '+str(error);save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for n in ('build','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
