#!/usr/bin/env python3
"""Hosted-only composed Josev audit; rejects unreviewed dependencies before API/payload access."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_composed_contract import verify
from josev2b_composed_panel import compare,ref_controls,require,write


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()


def api(path):
    return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path],timeout=90))


def policy():
    cfg=verify()
    require(os.environ.get('GITHUB_ACTIONS')=='true','hosted-only composed audit')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(head==os.environ.get('GITHUB_SHA'),'checkout/service source differs')
    deps=json.loads((ROOT/'docs/josev2b-composed-dependencies.json').read_text())
    require(deps['schema']=='crispembed-josev-composed-dependencies-v1' and deps['reviewed'] is True,
            'explicit frontend/serializer dependency review pending')
    for kind in ('frontend','answer'):
        pin=deps[kind]
        require(type(pin['run']) is int and len(pin['source'])==40 and
                all(type(a['id']) is int and type(a['bytes']) is int and len(a['sha256'])==64 for a in pin['artifacts'].values()),
                'missing explicit dependency source/artifact pins')
    return cfg,deps,head


def command(args,log,env,timeout):
    start=time.monotonic()
    with log.open('w') as stream:
        p=subprocess.run([str(a) for a in args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    return dict(exit=p.returncode,elapsed_seconds=time.monotonic()-start,log_sha256=sha(log))


def fetch(pin,scratch,include=None):
    run=api('actions/runs/'+str(pin['run']))
    require(run['head_sha']==pin['source'] and run['event']=='workflow_dispatch' and run['run_attempt']==1 and
            run['status']=='completed' and run['conclusion']=='success' and run['path']=='.github/workflows/ovisocr2-metadata.yml',
            'immutable dependency workflow/source/attempt differs')
    artifact=api('actions/artifacts/'+str(pin['id']))
    require(artifact['workflow_run']['id']==pin['run'] and artifact['workflow_run']['head_sha']==pin['source'] and
            artifact['name']==pin['name'] and artifact['size_in_bytes']==pin['bytes'] and not artifact['expired'] and
            artifact['digest']=='sha256:'+pin['sha256'],'immutable artifact service identity differs')
    directory=scratch/str(pin['id']);directory.mkdir()
    target=scratch/(str(pin['id'])+'.zip')
    with target.open('xb') as stream:
        subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip'],
                       stdout=stream,check=True,timeout=1800)
    require(target.stat().st_size==pin['bytes'] and sha(target)==pin['sha256'],'artifact ZIP bytes differ')
    with zipfile.ZipFile(target) as archive:
        entries=archive.infolist();names=[e.filename for e in entries]
        require(len(entries)<=8192 and len(set(names))==len(names) and sum(e.file_size for e in entries)<=20*2**30,
                'artifact ZIP extent/duplicate member differs')
        require(include is None or set(include)<=set(names),'declared artifact member absent; no basename fallback')
        for row in entries:
            name=row.filename;p=PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and
                    p.as_posix()==name.rstrip('/') and (row.external_attr>>16)&0o170000!=0o120000,'unsafe artifact member')
            if not row.is_dir() and (include is None or name in include):
                dest=directory/name;dest.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(row) as source,dest.open('xb') as out:shutil.copyfileobj(source,out,1024*1024)
    target.unlink() # Task-owned ZIP only; upstream immutable archive stays remote.
    return directory,{k:artifact[k] for k in ('id','name','size_in_bytes','digest','expires_at')}


def barrier(deps,scratch):
    proofs={}
    for kind,flag,schema in (('frontend','tokenizer_schema_parity','crispembed-josev-tokenizer-native-v1'),
                             ('answer','answer_serializer_parity','crispembed-josev-answer-native-v1')):
        source=deps[kind];pin=dict(source['artifacts']['review'],run=source['run'],source=source['source'])
        require(pin['bytes']<=5*2**20,'small dependency review size exceeded')
        directory,proof=fetch(pin,scratch)
        report_path=directory/'native.json';require(sha(report_path)==source['native_report_sha256'],'reviewed dependency report bytes differ')
        report=json.loads(report_path.read_text())
        require(report['schema']==schema and report['completed'] is True and report[flag] is True and
                report['failures']==[] and report['source_head']==source['source'] and report['model_weights'] is False,
                'dependency parity incomplete/rejected')
        if kind=='frontend':require(report['original_four_exact'] is True and report['cases']==18,'frontend original/authored scope differs')
        else:require(report['ordinary_test_neutrality'] is True and report['ordinary_excludes_test_symbols'] is True,
                     'serializer production/test neutrality differs')
        proofs[kind]=proof
    return proofs


def run(out,scratch,build):
    cfg,deps,head=policy() # Must precede even dependency API access.
    require(importlib.metadata.version('numpy')=='2.2.6','pinned independent metric package differs')
    out.mkdir(parents=True);scratch.mkdir(parents=True)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available>=10*2**30 and shutil.disk_usage(scratch).free>=25*2**30,'hosted10GiBRAM/25GiBdisk required')
    require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==cfg['protected_roots']['ggml'],
            'actual backend checkout differs')
    env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    report=dict(schema='crispembed-josev-composed-hosted-v1',source=head,completed=False,full_model_qualified=False,commands={},dependencies={})
    write(out/'hosted.json',report)
    for name,args in (('configure',['cmake','-S',ROOT/'tests/josev-composed','-B',build,'-DCMAKE_BUILD_TYPE=Release']),
                      ('build',['cmake','--build',build,'--parallel','2']),
                      ('gates-plain',[build/'josev-composed-plain-probe','--gates']),
                      ('gates-observed',[build/'josev-composed-observed-probe','--gates'])):
        report['commands'][name]=command(args,out/(name+'.log'),env,1200)
        write(out/'hosted.json',report);require(report['commands'][name]['exit']==0,'build/control rejected: '+name)
    commands=json.loads((build/'compile_commands.json').read_text())
    names={'josev2b_composed.cpp','josev2b_backbone.cpp','josev2b_head.cpp',
           'josev2b_text_frontend.cpp','josev2b_answer_serializer.cpp','qwen35_decoder.cpp','shortest_double.c'}
    arithmetic=[row for row in commands if Path(row['file']).name in names]
    require(len(arithmetic)==2*len(names) and all('-ffp-contract=off' in row['command'] and
            not any(flag in row['command'] for flag in ('-ffast-math','-Ofast','-ffp-contract=fast')) for row in arithmetic),
            'actual ordinary/observed arithmetic compiler flags differ')
    write(out/'compile-commands.json',arithmetic)
    plain=build/'libjosev-composed-plain.a';observed=build/'libjosev-composed-observed.a'
    symbols=subprocess.check_output(['nm','-C',str(plain)],text=True)
    require(not any(marker in symbols for marker in ('composed_test::','josev2b_head::test::','run_observed')),
            'observer symbols leaked into ordinary archive')
    strings=subprocess.check_output(['strings',str(plain)],text=True)
    require('head/option_context_projection' not in strings,'reference-stage label leaked into ordinary archive')
    tested=subprocess.check_output(['nm','-C',str(observed)],text=True)
    require('josev2b::composed_test::observe' in tested and 'josev2b_head::test::run_observed' in tested,
            'instrumented archive positive control missing')
    report['archives']={'plain_sha256':sha(plain),'observed_sha256':sha(observed),
                        'ordinary_excludes_observers':True,'instrumented_symbols_present':True}
    write(out/'hosted.json',report)
    ref_controls(build/'josev-composed-diff',out/'ref-controls')
    report['dependencies']=barrier(deps,scratch);write(out/'hosted.json',report)
    front=deps['frontend']; member=front['component_member']
    require(PurePosixPath(member).as_posix()==member and not PurePosixPath(member).is_absolute() and
            '..' not in PurePosixPath(member).parts,'unsafe explicit tokenizer component member')
    token,tp=fetch(dict(front['artifacts']['raw'],run=front['run'],source=front['source']),scratch,{member})
    tokenizer=token/member
    require(tokenizer.stat().st_size==front['component_bytes'] and sha(tokenizer)==front['component_sha256'],
            'reviewed tokenizer component bytes differ')
    pins=deps['fixed']
    backbone,bp=fetch(pins['backbone'],scratch,{'raw/backbone-f32.gguf','raw/backbone-f32.conversion.json'})
    model=backbone/'raw/backbone-f32.gguf'
    require(model.stat().st_size==7527324576 and sha(model)=='8dc144a89e4b2bbf253fe395eeb9bc4d1eb7b3a27474442847535e55c79b361c',
            'reviewed exact320F32 backbone bytes differ')
    original,hp=fetch(pins['head'],scratch)
    receipt=json.loads((original/'review/artifact.json').read_text())
    require(receipt['completed'] is True and receipt['arrays']==251 and receipt['reference_sha256']==
            '51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090','original receipt scope differs')
    files={str(p.relative_to(original)) for p in original.rglob('*') if p.is_file()}
    require(files==set(receipt['files'])|{'review/artifact.json'},'original complete member inventory differs')
    for name,pin in receipt['files'].items():
        path=original/name
        require(path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'original preserved member bytes differ')
    reference=original/'original'
    require(sha(reference/'input-freeze.json')=='9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7',
            'frozen raw request source differs')
    require((reference/'reference.json').read_bytes()==(ROOT/'tests/results/josev2b/original-reference.json').read_bytes(),
            'frozen reference/public source differs')
    frozen=json.loads((reference/'input-freeze.json').read_text());gold=json.loads((reference/'reference.json').read_text())
    report['payloads']={'backbone':bp,'head':hp,'tokenizer':tp};write(out/'hosted.json',report)
    requests=out/'requests.json'
    write(requests,[dict(id=r['id'],request_json=json.dumps(r['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')))
                    for r in frozen['records']])
    env.update({k:'1' for k in ('CRISPEMBED_JOSEV_COMPOSED','CRISPEMBED_JOSEV_BACKBONE','CRISPEMBED_QWEN35_OCR',
                               'CRISPEMBED_JOSEV_HEAD','CRISPEMBED_JOSEV_TEXT_FRONTEND','CRISPEMBED_JOSEV_ANSWER_SERIALIZER')})
    outputs={}
    for variant,binary in (('plain','plain'),('observed','observed'),('repeat','observed')):
        directory=out/variant;outputs[variant]=directory
        report['commands'][variant]=command([build/('josev-composed-'+binary+'-probe'),model,
            original/'head/joint-head-f32.gguf',tokenizer,requests,directory],out/(variant+'.log'),env,1800)
        write(out/'hosted.json',report)
    require(all(report['commands'][v]['exit']==0 for v in outputs),'native composition structural rejection; all logs retained')
    result=compare(reference,frozen,gold,outputs,build/'josev-composed-diff',out/'comparison')
    report.update(completed=True,numeric_and_complete_typed_parity=result['numeric_and_complete_typed_parity'],
                  compared_stages=result['compared_stages'],failures=result['failures'])
    write(out/'hosted.json',report)
    return int(not result['numeric_and_complete_typed_parity'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','scratch','build'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    return run(args.output,args.scratch,args.build)

if __name__=='__main__':sys.exit(main())
