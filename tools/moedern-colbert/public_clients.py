"""Actual C/Python/CLI/HTTP/Rust/Dart consumers for every MoEdern size.

Called on a hosted worker after its original freeze and root source closure.
Uses the existing public wrapper probes and unchanged vector/MaxSim judge.
Q4 numeric drift is retained separately from cross-client API conformance.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(Path(__file__).parent)]
from quantize_q4k_padded import PROFILES,sha_file

def require(ok,why):
    if not ok:raise ValueError(why)

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

old=module('old_public_clients',ROOT/'tools/ci-heavy/moedern_public_consumers.py')
DART_URL='https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-linux-x64-release.zip'
DART_SHA='ea864bc64df30a6b8bdf30b2e32550f7717d9a890de8f40293aeabb924fe232b'

def hosted():
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')),'hosted public clients only')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def command(args,log,env=None,cwd=ROOT,timeout=1800):
    with log.open('w')as stderr:
        r=subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=stderr,timeout=timeout,text=True)
    return r.returncode,r.stdout

def parse_outputs(payload,cases,pairs):
    def invalid(value):raise ValueError('nonfinite JSON constant: '+value)
    rows=[json.loads(line,parse_constant=invalid,parse_int=lambda text:-0.0 if text=='-0'else int(text))for line in payload.splitlines()if line.strip()]
    expected=[('case',i)for i in cases]+[('score',i)for i in pairs]
    require([(r.get('kind'),r.get('id'))for r in rows]==expected,'complete ordered distinct client cases/scores required')
    for r in rows:
        if r['kind']=='case':
            for name in ('all_vectors','selected_vectors'):
                require(isinstance(r.get(name),list)and all(isinstance(x,(float,int))and not isinstance(x,bool)and math.isfinite(x)for x in r[name]),'finite actual client vectors required')
        else:require(isinstance(r.get('score'),(float,int))and math.isfinite(r['score']),'finite actual client score required')
    return rows

def canonical_bits(outputs,*,include_all=True):
    """Compare F32 payload bits, including signed zero, across JSON printers."""
    keys=('kind','id','role','ids','supplied_mask','effective_mask','scoring_mask','expansion_positions')
    result=[]
    for row in outputs:
        item={k:row[k]for k in keys if k in row}
        for name in ('all_vectors','selected_vectors'):
            if name in row and (name!='all_vectors'or include_all):item[name]=struct.pack('<'+'f'*len(row[name]),*row[name])
        if 'score'in row:item['score']=struct.pack('<f',row['score'])
        result.append(item)
    return result

def dart_sdk(scratch):
    # Pin both version and official archive checksum; never fetch it on the VPS.
    hosted();scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=False);archive=scratch/'sdk.zip'
    with urllib.request.urlopen(DART_URL,timeout=120)as response,archive.open('xb')as target:
        count=0
        while chunk:=response.read(1024**2):
            count+=len(chunk);require(count<=300*1024**2,'Dart archive exceeds bound');target.write(chunk)
    require(sha_file(archive)==DART_SHA,'official Dart SDK archive SHA differs')
    with zipfile.ZipFile(archive)as z:
        require(all(not Path(i.filename).is_absolute()and '..'not in Path(i.filename).parts for i in z.infolist()),'unsafe SDK archive')
        z.extractall(scratch)
        for i in z.infolist():
            if i.external_attr>>16&0o111:(scratch/i.filename).chmod((i.external_attr>>16)&0o777)
    dart=scratch/'dart-sdk/bin/dart';require(dart.is_file(),'Dart binary absent');return dart

def execute(variant,model,original_dir,build,out,*,quantized=False,dart=None):
    hosted();require(variant in PROFILES,'explicit model variant required');build,out,model=Path(build),Path(out),Path(model);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'docs/moedern-q4k-padded.json').read_text());driver=module('q4_native_inputs',ROOT/'tools/ci-heavy/moedern_q4k_padded.py')
    original,arrays=driver.original_inputs(Path(original_dir).resolve(),variant,cfg)
    vectors={(c,s):(row['shape'],row['path'].read_bytes())for(c,s),row in arrays.items()}
    encoded={}
    for c in original['cases']:
        supplied=list(map(int,c['supplied_attention_mask']))
        encoded[c['id']]=dict(ids=c['token_ids'],supplied_mask=supplied,effective_mask=list(map(int,c['effective_attention_mask'])),scoring_mask=list(map(int,c['scoring_mask'])),expansion_positions=[1-x for x in supplied]if c['role']=='query'else[])
    lib=build/'libcrispembed.so';server_binary=build/'crispembed-server';cli=build/'crispembed'
    require(all(p.is_file()for p in (lib,server_binary,cli,model)),'actual root library/server/CLI/model required')
    report=dict(schema='crispembed-moedern-all-size-clients-v1',variant=variant,quantized=quantized,completed=False,
                model_sha256=sha_file(model),library_sha256=sha_file(lib),original_report_sha256=cfg['originals'][variant]['report_sha256'],consumers={},default_enabled=False,model_qualified=False)
    save=lambda:write(out/'review.json',report);save()
    text={c['id']:c['text']for c in original['cases']};hx=lambda s:s.encode().hex()or'-';pairs_ids=[p['query']+'|'+p['document']for p in original['scores']]
    cases,pairs=out/'cases.tsv',out/'pairs.tsv'
    cases.write_text(''.join(c['id']+'\t'+c['role']+'\t'+hx(c['text'])+'\n'for c in original['cases']))
    pairs.write_text(''.join(pid+'\t'+hx(text[p['query']])+'\t'+hx(text[p['document']])+'\n'for pid,p in zip(pairs_ids,original['scores'])))
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
    env.update(old.GATES,CARGO_BUILD_JOBS='2',CRISPEMBED_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CRISPEMBED_SYS_LIB_DIR=str(build),LD_LIBRARY_PATH=str(build))
    if variant!='1.5M'or quantized:env.update(CRISPEMBED_MOEDERN_FAMILY='1',CRISPEMBED_MOEDERN_FAST='1')
    if quantized:env.update(CRISPEMBED_MOEDERN_Q4_K_PADDED='1',CRISPEMBED_MOEDERN_QUANT='1')
    try:
        consumer=out/'c-consumer';code,_=command(['cc','-O2','-I',ROOT/'src',ROOT/'tests/moedern-public/consumer.c','-L',build,'-lcrispembed','-Wl,-rpath,'+str(build),'-o',consumer],out/'c-build.log',env)
        require(code==0,'actual C consumer compile failed')
        # Every model gate unset; value-parsed zero retains rejection for newQ4.
        off={k:v for k,v in env.items()if not k.startswith('CRISPEMBED_MOEDERN_')}
        code,_=command([consumer,model,cases,pairs],out/'gate-off.log',off);require(code!=0,'gate-off model accepted');report['gate_off_rejected']=True
        rust=out/'rust';(rust/'src').mkdir(parents=True)
        (rust/'Cargo.toml').write_text('[package]\nname="moedern-consumer"\nversion="0.0.0"\nedition="2021"\n[dependencies]\ncrispembed={path="'+str(ROOT/'crispembed')+'"}\n[workspace]\n')
        shutil.copyfile(ROOT/'tests/moedern-public/rust_consumer.rs',rust/'src/main.rs')
        dart_root=out/'dart';(dart_root/'lib').mkdir(parents=True);(dart_root/'bin').mkdir()
        shutil.copyfile(ROOT/'flutter/crispembed/lib/src/moedern_text.dart',dart_root/'lib/moedern_text.dart');shutil.copyfile(ROOT/'tests/moedern-public/dart_consumer.dart',dart_root/'bin/main.dart')
        (dart_root/'pubspec.yaml').write_text('name: moedern_audit\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n')
        dart=Path(dart)if dart else dart_sdk(out/'dart-tools');code,_=command([dart,'pub','get'],out/'dart-pub.log',env,dart_root);require(code==0,'Dart dependency setup failed')
        raw={}
        def record(name,outputs):
            rows=parse_outputs(outputs,list(encoded),pairs_ids)
            for c,row in zip(original['cases'],[r for r in rows if r['kind']=='case']):require(row.get('role')==c['role'],'client role label differs')
            result=old.judge(rows,original,vectors,encoded)
            report['consumers'][name]=dict(reference_parity=result,typed_exact=all(all(c['typed'].values())for c in result['cases'].values()),source='actual client process')
            if name=='cli':
                for case in result['cases'].values():case['all_vectors']['scope']='borrowed C output; not observed from CLI'
            raw[name]=rows;save()
        clients={
            'c':[consumer,model,cases,pairs],
            'c-repeat':[consumer,model,cases,pairs],
            'python':[sys.executable,ROOT/'tests/moedern-public/python_consumer.py',model,lib,cases,pairs,ROOT/'python'],
            'rust':['cargo','run','--release','--quiet','--manifest-path',rust/'Cargo.toml','--',model,cases,pairs],
            'dart':[dart,'run','bin/main.dart',model,lib,cases,pairs]}
        for name,cmd in clients.items():
            code,stdout=command(cmd,out/(name+'.log'),env,dart_root if name=='dart'else ROOT,3600);(out/(name+'.jsonl')).write_text(stdout);require(code==0,name+' process failed');record(name,stdout)
        rows=[]
        for c in original['cases']:
            code,stdout=command([cli,'-m',model,'--moedern',c['role'],c['text']],out/'cli.log',env);require(code==0,'CLI failed');row=dict(json.loads(stdout),kind='case',id=c['id']);row['all_vectors']=next(r for r in raw['c']if r.get('kind')=='case'and r['id']==c['id'])['all_vectors'];rows.append(row)
        for pid,p in zip(pairs_ids,original['scores']):
            code,stdout=command([cli,'-m',model,'--moedern-score',text[p['query']],text[p['document']]],out/'cli.log',env);require(code==0,'CLI score failed');rows.append(dict(kind='score',id=pid,score=json.loads(stdout)['score']))
        record('cli','\n'.join(json.dumps(r,allow_nan=False)for r in rows));report['cli_all_vectors_scope']='Shape-only borrowed C vectors; CLI selected vectors, IDs and masks are actual outputs.'
        with (out/'http.log').open('w')as log:
            server=subprocess.Popen([server_binary,'--moedern-model',model,'--port','18431'],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+90
                while True:
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18431/health/ready',timeout=2)as response:
                            if json.load(response)['status']=='ready':break
                    except OSError:pass
                    require(server.poll()is None and time.monotonic()<deadline,'HTTP server failed startup');time.sleep(.25)
                def post(path,body):
                    request=urllib.request.Request('http://127.0.0.1:18431'+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
                    with urllib.request.urlopen(request,timeout=120)as response:return json.load(response)
                try:post('/moedern/encode',dict(text='x',role='other'));require(False,'invalid HTTP role accepted')
                except urllib.error.HTTPError as error:require(error.code==400,'wrong HTTP role error')
                rows=[dict(post('/moedern/encode',dict(text=c['text'],role=c['role'])),kind='case',id=c['id'])for c in original['cases']]
                rows += [dict(kind='score',id=pid,score=post('/moedern/score',dict(query=text[p['query']],document=text[p['document']]))['score'])for pid,p in zip(pairs_ids,original['scores'])]
                record('http','\n'.join(json.dumps(r,allow_nan=False)for r in rows))
            finally:
                server.terminate()
                try:server.wait(timeout=15)
                except subprocess.TimeoutExpired:server.kill();server.wait()
        report['cross_client_float32_exact']={name:canonical_bits(values,include_all=name!='cli')==canonical_bits(raw['c'],include_all=name!='cli')for name,values in raw.items()}
        report['cross_client_vector_scope']={name:('actual selected vectors and score only'if name=='cli'else'actual all/selected vectors and score')for name in raw}
        report['completed']=True;report['api_conformance']=all(report['cross_client_float32_exact'].values())and all(c['typed_exact']for c in report['consumers'].values())
        report['f32_reference_parity']=all(c['reference_parity']['passed']for c in report['consumers'].values())
        require(report['api_conformance'],'cross-client/typed conformance failed')
        if not quantized:require(report['f32_reference_parity'],'F32 original vector/score gates failed')
        save();return report
    except Exception as error:report['error']=str(error);save();raise
