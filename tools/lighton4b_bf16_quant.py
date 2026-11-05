#!/usr/bin/env python3
"""Compare authenticated retrieved BF16 scores with real native quantized captures.

No download, model load or native forward is performed by this consumer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from lighton4b_document_v2 import LocalReader, tiny_archive

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/lighton4b-bf16-quant-v1.json'
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def protocol():
    p=json.loads(PROTOCOL.read_text())
    registry=json.loads((ROOT/'docs/lighton4b-bf16-quant-admissions-v1.json').read_text())
    require(registry['schema']=='lighton4b-bf16-quant-admissions/v1' and registry['protocol_sha256']==sha(PROTOCOL),'admission registry protocol differs')
    require(isinstance(registry['native_producers'],list),'admission registry tuple inventory differs')
    p['native_producers']=registry['native_producers']
    return p
def original(root,p):
    root=Path(root);require(sha(root/'capture.json')==p['original_capture_sha256'],'original receipt hash differs')
    a=json.loads((root/'capture.json').read_text())
    require(a['status']=='captured' and a['requested_dtype']=='bfloat16' and a['source_sha']==p['original_source_sha'],'actual BF16 original required')
    require(a['generated_ids']==p['original_generated_ids'] and a['decoded']==p['original_decoded'] and a['eos_completed'] and not a['cap_hit'],'frozen original decoded output differs')
    require(a['scores']==p['score_records'],'original score inventory differs')
    return a

def candidate(b,a,p,synthetic=False,receipt_sha256=None):
    require(b['schema']=='lighton4b-bf16-quant-native/v1','native schema differs')
    require(b['precision'] in p['precisions'],'undeclared quantization')
    require(b['original_capture_sha256']==p['original_capture_sha256'] and b['protocol_sha256']==sha(PROTOCOL),'reference/protocol identity differs')
    require(b['teacher_forced_ids']==a['generated_ids'],'all original prefixes required')
    require(b['free_generation_independent'] is True,'independent free generation required')
    ids=b['generated_ids'];require(isinstance(ids,list) and 1<=len(ids)<=256 and all(type(i)is int and 0<=i<248320 for i in ids),'invalid native IDs')
    require(not any(i==248046 for i in ids[:-1]),'tokens after EOS')
    eos=ids[-1]==248046
    require(b['stop']==('eos' if eos else 'max_tokens') and b['cap_hit']==(not eos and len(ids)==256) and (eos or len(ids)==256),'native stop/cap differs')
    require(isinstance(b['decoded'],str),'decoded output required')
    require(len(b['scores'])==p['positions'],'missing teacher-forced score positions')
    for i,row in enumerate(b['scores']):
        require(row['position']==i and row['shape']==[1,248320] and row['bytes']==993280,'complete full-vocabulary rows required')
    if synthetic:require(b.get('synthetic_control') is True,'explicit synthetic fixture required')
    else:
        require(b.get('synthetic_control') is not True and b.get('native_forward_executed') is True,'actual native execution required')
        identity={'capture_sha256':receipt_sha256,'source_sha':b['producer_source_sha'],'run_url':b['producer_run_url'],'weights_sha256':b['weights_sha256'],'precision':b['precision']}
        require(identity in p['native_producers'],'native producer/weights not admitted')

def compare(original_root,native_root,binary,output,synthetic=False):
    require(os.environ.get('GITHUB_ACTIONS')=='true' or Path('/kaggle/working').is_dir(),'hosted comparison required')
    out=Path(output);require(not out.exists(),'fresh output required');out.mkdir(parents=True)
    r={'schema':'lighton4b-bf16-quant-comparison/v1','status':'blocked','native_execution_admitted':False,'quality_qualified':False,'synthetic_control':synthetic,'original_forward_executed_by_comparer':False,'native_forward_executed_by_comparer':False,'earliest_observed_score_difference':None,'earliest_score_budget_failure':None,'internal_stage_attribution_available':False}
    try:
        p=protocol();a=original(original_root,p);b=json.loads((Path(native_root)/'native.json').read_text());candidate(b,a,p,synthetic,sha(Path(native_root)/'native.json'))
        r.update(original_decoded=a['decoded'],native_decoded=b['decoded'],original_generated_ids=a['generated_ids'],native_generated_ids=b['generated_ids'],native_precision=b['precision'],protocol_sha256=sha(PROTOCOL),binary_sha256=sha(binary),native_execution_admitted=not synthetic)
        readers=[LocalReader(original_root),LocalReader(native_root)];numerical=True;rows=[]
        with tempfile.TemporaryDirectory(prefix='bf16-quant-row-') as folder:
            folder=Path(folder)
            for i,(ar,br) in enumerate(zip(a['scores'],b['scores'])):
                name=f'generated.scores.{i}'
                for label,row,reader in zip(('original','native'),(ar,br),readers):
                    require(row['position']==i and row['shape']==[1,248320] and row['bytes']==993280,'score geometry differs')
                    data=reader.read(row['file'],0,row['bytes']);require(hashlib.sha256(data).hexdigest()==row['sha256'],'score payload hash differs')
                    require((reader.root/row['file']).stat().st_size==row['bytes'],'trailing score bytes')
                    tiny_archive(folder/(label+'.gguf'),name,row['shape'],data)
                order=folder/'order.tsv';order.write_text(f'{name} float 248320 1 1 1\n')
                run=subprocess.run([str(Path(binary).resolve()),str(folder/'original.gguf'),str(folder/'native.gguf'),str(order)],capture_output=True,text=True)
                (out/f'position-{i:03d}.log').write_text(run.stdout+run.stderr)
                stage=[x for x in run.stdout.splitlines() if x.startswith('STAGE ')]
                require(len(stage)==1 and stage[0].split()[1]==name and run.returncode in (0,1),'actual Ref comparison incomplete')
                rows.append(stage[0]); numerical &= run.returncode==0
                metrics=dict(item.split('=',1) for item in stage[0].split()[2:] if '=' in item)
                if float(metrics['max_abs'])>0 and r['earliest_observed_score_difference'] is None:r['earliest_observed_score_difference']=i
                if run.returncode and r['earliest_score_budget_failure'] is None:r['earliest_score_budget_failure']=i
        r.update(compared_positions=len(rows),metrics=rows,numerical_passed=numerical,decoded_and_ids_exact=a['generated_ids']==b['generated_ids'] and a['decoded']==b['decoded'],native_eos_completed=b['stop']=='eos' and not b['cap_hit'])
        r['status']='passed' if numerical and r['decoded_and_ids_exact'] and r['native_eos_completed'] else 'failed'
    except Exception as e:
        message=str(e)
        for path,label in ((ROOT,'[repository]'),(original_root,'[original]'),(native_root,'[native]'),(output,'[report]'),(binary,'[binary]')):message=message.replace(str(path),label)
        r.update(error_type=type(e).__name__,error=message)
    (out/'comparison.json').write_text(json.dumps(r,indent=2)+'\n')
    return 0 if r['status']=='passed' else 2 if r['status']=='blocked' else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','native','binary','output'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();return compare(a.original,a.native,a.binary,a.output)
if __name__=='__main__':raise SystemExit(main())
