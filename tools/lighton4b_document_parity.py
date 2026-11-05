#!/usr/bin/env python3
"""Validate a frozen 4B document protocol or compare already captured archives.

There is intentionally no capture/forward/download command. Existing dumpers
need a separately reviewed 4B adapter before original and native bundles exist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/lighton4b-document-parity-v1.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(8*2**20):h.update(chunk)
    return h.hexdigest()


def require(condition,why):
    if not condition:raise ValueError(why)


def protocol():
    p=json.loads(PROTOCOL.read_text())
    require(p['schema']=='lighton4b-document-parity/v1' and p['model']['revision']=='b71010f095dc3735de7ee6b9969bff8f8f50b39a', 'pinned protocol identity differs')
    for name,digest in p['source_sha256'].items():require(sha(ROOT/name)==digest,'frozen capture/helper source differs: '+name)
    case=p['case']
    for name,key in (('file','sha256'),('source_manifest','source_manifest_sha256'),('annotation','annotation_sha256')):
        require(sha(ROOT/case[name])==case[key],'frozen case/license/annotation differs')
    require(case['prompt']=='' and case['max_new_tokens']==256 and case['eos_ids']==[248046], 'request/cap/stop differs')
    require(hashlib.sha256(case['rendered_prompt'].encode()).hexdigest()==case['rendered_prompt_sha256'],'rendered prompt differs')
    require(p['comparison']['max_abs']==0.002 and p['comparison']['min_row_cosine']==0.99999,'threshold drift')
    return p


def stages(tokens):
    require(type(tokens)is int and 1<=tokens<=256,'generated-position count outside bound')
    names=[('input.rgb','exact'),('input.image_grid_thw','exact'),('input.input_ids','exact'),
           ('input.mm_token_type_ids','exact'),('input.pixel_values','float')]
    names += [('prefill.vision.'+name,'float')for name in ('patch_embed','position','embedded','rotary_cos','rotary_sin')]
    names += [(f'prefill.vision.layer.{i}','float')for i in range(24)]
    names += [('prefill.vision.'+name,'float')for name in ('merger.norm','merger.up','merger.activation','merger')]
    for phase in ('prefill','cached','full_prefix'):
        names += [(phase+'.decoder.position_ids','exact'),(phase+'.decoder.inputs_embeds','float')]
        for i in range(32):
            prefix=f'{phase}.'+('attention'if i%4==3 else'linear')+f'.{i}.'
            fields=('input','q','k','v','q_norm','k_norm','pooled','output')if i%4==3 else('input','qkv','z','a','b','norm','output')
            names += [(prefix+field,'float')for field in fields]
            names.append((f'{phase}.decoder.layer.{i}','float'))
            if phase!='full_prefix':
                names += [(f'{phase}.cache.{i}.'+field,'float')for field in
                          (('keys','values')if i%4==3 else('conv_states','recurrent_states'))]
        names += [(phase+'.decoder.norm','float'),(phase+'.logits','float')]
    for i in range(tokens):
        names += [(f'generated.{i}.decoder.position_ids','exact'),(f'generated.scores.{i}','float')]
    return names


def validate_bundle(folder, role, p):
    folder=Path(folder);receipt=json.loads((folder/'bundle.json').read_text())
    require(receipt['schema']=='lighton4b-document-bundle/v1' and receipt['role']==role,'bundle role/schema differs')
    require(receipt['protocol_sha256']==sha(PROTOCOL),'bundle protocol differs')
    require(receipt['model']==p['model'] and receipt['case']==p['case'],'model/request/geometry contract differs')
    require(receipt['packages']==p['packages'],'original framework/package closure differs')
    require(re.fullmatch('[0-9a-f]{40}',receipt['producer_source_sha'])is not None,'immutable producer source required')
    require(receipt['producer_source_sha']in p['execution_admission'][role+'_producer_sources'],
            'producer source is not admitted by this prepared-only protocol')
    require(receipt['producer_run_url'].startswith('https://github.com/')or receipt['producer_run_url'].startswith('https://www.kaggle.com/'),'hosted producer identity required')
    ids=receipt['generated_ids'];case=p['case']
    require(isinstance(ids,list) and 1<=len(ids)<=256 and all(type(x)is int and 0<=x<248320 for x in ids),'invalid/full vocabulary generated IDs')
    complete=ids[-1]in case['eos_ids']
    require(not any(x in case['eos_ids']for x in ids[:-1]) and
            receipt['stop']==('eos'if complete else'max_tokens') and
            receipt['cap_hit']==(not complete and len(ids)==256),'inconsistent independent stop/cap receipt')
    if role=='original':require(complete,'original source document did not reach EOS; preserve source failure')
    require(isinstance(receipt['decoded'],str),'decoded output required')
    require(receipt['free_generation_independent']is True,'independent free-generation run required')
    forced=receipt['teacher_forced_ids']
    require(isinstance(forced,list)and 1<=len(forced)<=256 and all(type(x)is int and 0<=x<248320 for x in forced), 'original-position teacher-forced IDs required')
    if role=='original':require(forced==ids,'original score positions differ from original generation')
    expected=stages(len(forced));require(receipt['ordered_stages']==[list(x)for x in expected],'missing/reordered capture stages')
    require(set(receipt['stage_shapes'])=={n for n,_ in expected},'stage shape inventory differs')
    for name,shape in receipt['stage_shapes'].items():
        require(isinstance(shape,list)and 1<=len(shape)<=4 and all(type(x)is int and x>0 for x in shape),'invalid tensor shape')
        count=1
        for axis in shape:count*=axis
        require(count<=p['comparison']['max_stage_elements'],'stage memory bound')
        if name.startswith('generated.scores.')or name.endswith('.logits'):
            require(shape==[1,248320],'all248320 logits required at every generated position')
    artifact=folder/'stages.gguf'
    require(artifact.stat().st_size<=p['comparison']['max_bundle_bytes']and sha(artifact)==receipt['archive_sha256'],'frozen diagnostic archive differs/oversized')
    require(receipt['observer_neutral_ids_and_logits_exact']is True,'observer-neutral original control required')
    if role=='original':require(receipt['precision']=='f32','original must use CPU/F32 eager source')
    else:require(receipt['precision']in p['comparison']['precision_order'],'undeclared native precision')
    return receipt,expected


def _compare_existing(original,candidate,binary,output):
    require(os.environ.get('GITHUB_ACTIONS')=='true' or Path('/kaggle/working').is_dir(),'hosted comparison required')
    p=protocol();a,order=validate_bundle(original,'original',p);b,_=validate_bundle(candidate,'native',p)
    require(b['original_bundle_sha256']==sha(Path(original)/'bundle.json'),'original must be frozen before native capture')
    require(b['teacher_forced_ids']==a['generated_ids'],'native full-vocabulary scores must use every original-generated prefix')
    decoded_same=a['generated_ids']==b['generated_ids'] and a['decoded']==b['decoded']
    require(a['stage_shapes']==b['stage_shapes'],'original/native shapes differ')
    require(b['precision']in('f32','f16'),'quantized candidates require a separately closed F32/F16 and held-out protocol')
    output=Path(output)
    manifest=output/'stage-order.tsv';manifest.write_text(''.join(n+' '+kind+'\n'for n,kind in order))
    argv=[str(Path(binary).resolve()),str(Path(original)/'stages.gguf'),str(Path(candidate)/'stages.gguf'),str(manifest)]
    env=os.environ.copy();env.pop('CRISPEMBED_DIFF_COS_THRESHOLD',None);env.pop('CRISPEMBED_DIFF_DUMP_DIR',None)
    process=subprocess.run(argv,capture_output=True,text=True,env=env)
    (output/'crispembed-diff.stdout').write_text(process.stdout);(output/'crispembed-diff.stderr').write_text(process.stderr)
    rows=[line for line in process.stdout.splitlines()if line.startswith('STAGE ')]
    require(len(rows)==len(order),'comparer did not cover every frozen stage')
    first=next((line.split()[1]for line in rows if 'passed=0'in line),None)
    completed=b['stop']=='eos'and not b['cap_hit']
    report={'schema':'lighton4b-document-comparison/v1','status':'passed'if process.returncode==0 and decoded_same and completed else'failed',
        'protocol_sha256':sha(PROTOCOL),'binary_sha256':sha(binary),
        'command':[Path(binary).name,'$ORIGINAL/stages.gguf','$NATIVE/stages.gguf','$REPORT/stage-order.tsv'],
        'earliest_differing_stage':first,
        'original_bundle_sha256':sha(Path(original)/'bundle.json'),'native_bundle_sha256':sha(Path(candidate)/'bundle.json'),
        'original_decoded':a['decoded'],'native_decoded':b['decoded'],
        'original_generated_ids':a['generated_ids'],'native_generated_ids':b['generated_ids'],
        'decoded_and_ids_exact':decoded_same,'eos_completed':completed,'compared_stages':len(rows),
        'original_forward_executed_by_comparer':False,'native_forward_executed_by_comparer':False,
        'quality_qualified':False,'public_alias_qualified':False,'returncode':process.returncode}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    return process.returncode if process.returncode else(0 if decoded_same and completed else 1)


def compare(original,candidate,binary,output):
    output=Path(output);require(not output.exists(),'fresh report directory required');output.mkdir(parents=True)
    try:return _compare_existing(original,candidate,binary,output)
    except Exception as error:
        report={'schema':'lighton4b-document-comparison/v1','status':'blocked','error_type':type(error).__name__,
                'error':str(error).replace(str(ROOT),'[repository]'),'original_forward_executed_by_comparer':False,
                'native_forward_executed_by_comparer':False,'quality_qualified':False}
        for role,folder in [('original',original),('native',candidate)]:
            path=Path(folder)/'bundle.json'
            if path.is_file() and path.stat().st_size<4*2**20:
                try:
                    receipt=json.loads(path.read_text())
                    report[role]={k:receipt.get(k)for k in ('generated_ids','decoded','stop','cap_hit','producer_source_sha')}
                except Exception as diagnostic_error:report[role]={'receipt_read_error':type(diagnostic_error).__name__}
        (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        return 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan')
    c=sub.add_parser('compare')
    for name in('original','candidate','binary','output'):c.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='plan':print(json.dumps(protocol(),indent=2));return 0
    return compare(args.original,args.candidate,args.binary,args.output)


if __name__=='__main__':raise SystemExit(main())
