#!/usr/bin/env python3
"""Bounded CPU/F32 public text qualification; original full-stage audit stays separate."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultrabert_contract import QUERY, sha256
from ultrabert_native import prepare_frozen_artifacts


def parse_pooled(log):
    result, rows, total = {}, {}, None
    for line in log.splitlines():
        fields = line.split('\t')
        if fields[0] == 'POOLED':
            if len(fields) != 9 or fields[1] not in ('cls_raw', 'cls_normalized') or fields[1] in result:
                raise ValueError('Unexpected/duplicate pooled stage')
            values = [float(x) for x in fields[2:8]]
            passed = all(math.isfinite(x) for x in values) and values[0] >= .99999 and values[1] <= .002
            if fields[8] != str(int(passed)):
                raise ValueError('Pooled verdict mismatch')
            result[fields[1]] = dict(zip(('cos_min','max_abs','mean_abs','rms','native_norm','reference_norm'),values))
            result[fields[1]]['pass'] = passed
        elif fields[0] == 'IDS':
            row = int(fields[1])
            if row in rows:raise ValueError('Duplicate actual token row')
            rows[row] = [int(x) for x in fields[2:]]
        elif fields[0] == 'TOTAL':
            if total is not None:raise ValueError('Duplicate total')
            total = [int(x) for x in fields[1:]]
    if set(result) != {'cls_raw','cls_normalized'} or total != [len(rows),2,sum(not x['pass'] for x in result.values())]:
        raise ValueError('Incomplete public comparisons')
    if len(rows) not in (1,2) or set(rows) != set(range(len(rows))):raise ValueError('Wrong token rows')
    for ids in rows.values():
        if not ids or len(ids)>512 or ids[0]!=0 or any(x<0 or x>=49152 for x in ids):
            raise ValueError('Invalid actual token IDs')
    return dict(stages=result, ids=[rows[i] for i in range(len(rows))], failures=total[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frozen', type=Path)
    args = parser.parse_args()
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH']) / 'ultrabert-api'
    out.mkdir(parents=True, exist_ok=True);scratch.mkdir(parents=True, exist_ok=True)
    report = dict(source_commit=os.environ['GITHUB_SHA'],driver_sha256=sha256(__file__),
                  scope='CPU/F32 public single-text and sequential batch pooled outputs; no full-model or task-quality promotion',
                  context_limit=512,unsupported=['pair public API','reserved token surfaces','F16/quantized','GPU','Matryoshka'],
                  environment=dict(platform=platform.platform(),python=sys.version),cases=[],status='started')
    def save():
        (out/'ultrabert-api.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def run(command, name, env=None, expected=0):
        result=subprocess.run([str(x) for x in command],cwd=ROOT,env=env,capture_output=True,text=True,timeout=3600)
        log=out/(name+'.log');log.write_text(result.stdout+'\n'+result.stderr)
        if result.returncode != expected:raise ValueError(name+' exit '+str(result.returncode))
        return result.stdout
    save()
    try:
        subprocess.run([sys.executable,str(ROOT/'tests/test_ultrabert_api_log.py'),'-v'],check=True)
        baseline=json.loads((ROOT/'tests/results/ultrabert/reference-audit.json').read_text())
        original, refs=prepare_frozen_artifacts(baseline,args.frozen,scratch)
        report['frozen_report_sha256']=sha256(ROOT/'tests/results/ultrabert/reference-audit.json')
        report['original_source']=original['source_commit']
        report['f32_model_sha256']=sha256(scratch/'f32.gguf')
        report['native_source_sha256']={n:sha256(ROOT/n) for n in ('src/ultrabert_embed.cpp','src/ultrabert_profile.h','src/crispembed.cpp','tests/test_ultrabert_api.cpp')}
        build=scratch/'build'
        run(['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCRISPEMBED_BUILD_SHARED=ON',
             '-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF'], 'configure')
        run(['cmake','--build',build,'--target','crispembed-cli','crispembed-server','crispembed-shared',
             'test-ultrabert-profile','test-ultrabert-api','--parallel','2'], 'build')
        def binary(name):
            return next(p for p in (build/name,build/'bin'/name) if p.is_file())
        api=binary('test-ultrabert-api');cli=binary('crispembed-cli');server=binary('crispembed-server')
        library=next(build.rglob('libcrispembed.so'))
        report['library_sha256']=sha256(library)
        run([binary('test-ultrabert-profile')], 'model-free-profile')
        env={**os.environ,'CRISPEMBED_ULTRABERT':'1','CRISPEMBED_FORCE_CPU':'1','OMP_NUM_THREADS':'2'}
        for gate in ('unset','0'):
            gate_env=env.copy();gate_env.pop('CRISPEMBED_ULTRABERT',None)
            if gate=='0':gate_env['CRISPEMBED_ULTRABERT']='0'
            run([api,scratch/'f32.gguf','--reject'],'gate-'+gate,gate_env)
        run([api,scratch/'f16.gguf','--reject'],'reject-f16',env)
        for record in original['reference']['cases']:
            case=record['case']
            if 'text_pair' in case:
                report['pair_case']='Explicitly deferred: no public pair-embedding API; original prepared-ID audit retained'
                continue
            texts=case['text'] if isinstance(case['text'],list) else [case['text']]
            if case.get('prompt')=='query':texts=[QUERY+t for t in texts]
            files=[]
            for i,text in enumerate(texts):
                path=scratch/(case['id']+'-'+str(i)+'.txt');path.write_text(text);files.append(path)
            log=run([api,scratch/'f32.gguf',refs/record['archive'],*files],case['id'],env)
            result=parse_pooled(log)
            expected=[[token for token,keep in zip(ids,mask) if keep] for ids,mask in zip(record['ids'],record['attention_mask'])]
            if result['ids']!=expected or result['failures']:raise ValueError('Public token/vector mismatch')
            report['cases'].append(dict(case=case['id'],reference_archive_sha256=record['archive_sha256'],
                                       log_sha256=sha256(out/(case['id']+'.log')),**result));save()
        if len(report['cases'])!=6 or sum(len(c['ids']) for c in report['cases'])!=7:
            raise ValueError('Missing bounded case coverage')
        # Existing Python binding, CLI and actual HTTP server use normal model loading.
        os.environ.update({k:env[k] for k in ('CRISPEMBED_ULTRABERT','CRISPEMBED_FORCE_CPU','OMP_NUM_THREADS')})
        sys.path.insert(0,str(ROOT/'python'))
        import numpy as np
        from crispembed import CrispEmbed
        model=CrispEmbed(str(scratch/'f32.gguf'),n_threads=2,lib_path=str(library),offline=True)
        texts=['Berlin is the capital of Germany.','Grüße — café\n東京 العربية 👋','']
        values=model.encode(texts)
        ids,hidden=model.encode_tokens(texts[0],normalize=False)
        if values.shape!=(3,768) or hidden.shape!=(len(ids),768) or ids[0]!=0:raise ValueError('Python binding shape/IDs')
        if not np.allclose(values[0],hidden[0]/np.linalg.norm(hidden[0]),atol=.002,rtol=0):raise ValueError('Python raw/normalized CLS')
        normalized_ids,normalized_tokens=model.encode_tokens(texts[0],normalize=True)
        if not np.array_equal(ids,normalized_ids) or not np.allclose(normalized_tokens[0],values[0],atol=.002,rtol=0):
            raise ValueError('Python normalized token API')
        cli_output=run([cli,'-m',scratch/'f32.gguf','--prefix','','--json',texts[0]],'cli-normal',env)
        if not np.allclose(json.loads(cli_output)[0]['embedding'],values[0],atol=.002,rtol=0):raise ValueError('CLI output differs')
        bad=run([cli,'-m',scratch/'f32.gguf','--prefix','','--json',texts[0],'<|mask|>'],'cli-mixed-rejection',env,expected=1)
        if len(json.loads(bad))!=1:raise ValueError('CLI failure JSON corrupted')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with (out/'server.log').open('w') as log:
            process=subprocess.Popen([str(server),'-m',str(scratch/'f32.gguf'),'--host','127.0.0.1','--port',str(port)],
                                     cwd=ROOT,env=env,stdout=log,stderr=log)
            try:
                url='http://127.0.0.1:'+str(port)+'/embed'
                for _ in range(60):
                    if process.poll() is not None:raise ValueError('Server exited during init')
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url,data=json.dumps({'texts':texts}).encode(),headers={'Content-Type':'application/json'}),timeout=30) as response:
                            response=json.load(response)
                        break
                    except urllib.error.URLError:time.sleep(.5)
                else:raise ValueError('Server unavailable')
                if not np.allclose(response['embeddings'],values,atol=.002,rtol=0):raise ValueError('HTTP output differs')
                try:
                    urllib.request.urlopen(urllib.request.Request(url,data=b'{"texts":["<|mask|>"]}',headers={'Content-Type':'application/json'}),timeout=30)
                except urllib.error.HTTPError as error:
                    if error.code!=500:raise ValueError('Unexpected server rejection')
                else:raise ValueError('Server accepted unsupported text')
            finally:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        report['surfaces']={'c_api':True,'python':True,'cli':True,'http':True,'rust_dart':'Existing generic ABI routes; model-backed binding tests pending'}
        report['status']='completed bounded public F32 pooled-output checks; no independent task-quality qualification'
        save()
    except BaseException as error:
        report['status']='failed';report['error']=str(error);save();raise


if __name__=='__main__':main()
