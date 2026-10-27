#!/usr/bin/env python3
"""Actual packed SystemOne C/Python/CLI/HTTP consumers; hosted-only, frozen inputs.

The caller must first authenticate the packed model and pass its 227-stage gate.
This program never generates original gold and never accepts injected vectors.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
ROOT = Path(__file__).resolve().parents[2]
PROFILE = 'official-pytorch-text-batch1-512-no-extra-temperature'
FREEZE_SHA = '9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7'
GATES = ['JOSEV_PACKED','JOSEV_COMPOSED','JOSEV_BACKBONE','QWEN35_OCR',
         'JOSEV_HEAD','JOSEV_TEXT_FRONTEND','JOSEV_ANSWER_SERIALIZER']

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1024*1024): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    for name in ('model','library','cli','server','freeze','reference','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=['c','python','cli','http'], required=True)
    args = p.parse_args()
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted model consumer only')
    require(sha(args.freeze) == FREEZE_SHA, 'immutable raw requests differ')
    require(args.reference.read_bytes() == (ROOT/'tests/results/josev2b/original-reference.json').read_bytes(),
            'immutable original receipt differs')
    frozen, gold = json.loads(args.freeze.read_text()), json.loads(args.reference.read_text())
    records, cases = frozen['records'], gold['cases']
    require(len(records) == len(cases) == 4 and [r['id'] for r in records] == [r['id'] for r in cases], 'four original cases required')
    for gate in GATES: os.environ['CRISPEMBED_'+gate] = '1'
    os.environ['CRISPEMBED_OFFLINE'] = '1'
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    result = {'schema':'crispembed-systemone-public-consumer-v1','mode':args.mode,
              'source':os.environ['GITHUB_SHA'], 'completed':False, 'full_typed_parity':False,
              'cases':[], 'controls':{}, 'model_quality':False}
    args.output.mkdir(parents=True, exist_ok=False)
    def save(): (args.output/'consumer.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    save()
    raw = [json.dumps(r['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')) for r in records]
    run, close = None, lambda: None
    try:
        if args.mode == 'c':
            lib = ctypes.CDLL(str(args.library.resolve()))
            lib.crispembed_set_offline.argtypes=[ctypes.c_int]; lib.crispembed_set_offline(1)
            lib.crispembed_systemone_init.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.crispembed_systemone_init.restype=ctypes.c_void_p
            lib.crispembed_systemone_free.argtypes=[ctypes.c_void_p];lib.crispembed_systemone_free.restype=None
            for name in ('error','profile'):
                fn=getattr(lib,'crispembed_systemone_'+name);fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_char_p
            lib.crispembed_systemone_run_json.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.crispembed_systemone_run_json.restype=ctypes.c_char_p
            require(not lib.crispembed_systemone_init(b'no-such-systemone.gguf',2), 'missing model accepted')
            require(bool(lib.crispembed_systemone_error(None)), 'missing-model error absent')
            ctx=lib.crispembed_systemone_init(str(args.model.resolve()).encode(),2)
            require(bool(ctx), lib.crispembed_systemone_error(None).decode())
            close=lambda:lib.crispembed_systemone_free(ctx)
            require(lib.crispembed_systemone_profile(ctx).decode()==PROFILE,'profile differs')
            def c_run(text):
                value=lib.crispembed_systemone_run_json(ctx,text.encode())
                if value is None: raise ValueError(lib.crispembed_systemone_error(ctx).decode())
                return value.decode()
            run=c_run
            require(not lib.crispembed_systemone_run_json(ctx,None),'NULL request accepted')
            result['controls']['null_request_error']=bool(lib.crispembed_systemone_error(ctx))
        elif args.mode == 'python':
            sys.path.insert(0,str(ROOT/'python'))
            from crispembed import CrispSystemOne
            ctx=CrispSystemOne(str(args.model.resolve()),lib_path=str(args.library.resolve()),auto_download=False,offline=True,n_threads=2)
            close=ctx.close;run=ctx.run_json
            require(ctx.profile==PROFILE,'profile differs')
            for invalid in ('x\0y','Ä'*32769):
                try: run(invalid)
                except ValueError: pass
                else: raise ValueError('invalid byte/NUL request accepted')
            result['controls']['wrapper_nul_byte_limit']=True
        elif args.mode == 'cli':
            def cli_run(text):
                q=subprocess.run([str(args.cli.resolve()),'-m',str(args.model.resolve()),'-t','2','--systemone',text],capture_output=True,text=True,timeout=240)
                if q.returncode: raise ValueError(q.stderr)
                return q.stdout.strip()
            run=cli_run
        else:
            log=(args.output/'server.log').open('w')
            proc=subprocess.Popen([str(args.server.resolve()),'--systemone-model',str(args.model.resolve()),'--host','127.0.0.1','--port','18083','-t','2'],stdout=log,stderr=subprocess.STDOUT)
            base='http://127.0.0.1:18083'
            def stop():
                proc.terminate()
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=15)
                log.close()
            close=stop
            def get(path):
                with urllib.request.urlopen(base+path,timeout=2) as reply: return json.load(reply)
            deadline=time.monotonic()+60
            while True:
                require(proc.poll() is None,'SystemOne-only server exited')
                try:
                    if get('/health/ready')['status']=='ready': break
                except (OSError,urllib.error.URLError): pass
                require(time.monotonic()<deadline,'SystemOne-only readiness timeout')
                time.sleep(.2)
            require(get('/health')['systemone'] is True and get('/capabilities')['routes']['systemone'] is True,'SystemOne discovery absent')
            result['controls']['standalone_ready_discovery']=True
            def http_run(text):
                request=urllib.request.Request(base+'/systemone',text.encode(),{'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=240) as reply: return reply.read().decode()
                except urllib.error.HTTPError as error:
                    body=error.read().decode()
                    require(error.code==400,'unexpected HTTP status')
                    raise ValueError(body) from error
            run=http_run
            for text in ('x\0y','Ä'*32769):
                try: run(text)
                except ValueError: pass
                else: raise ValueError('HTTP byte/NUL bounds accepted')
            result['controls']['http_nul_byte_limit']=True
        # Malformed/schema/media requests must reject, then the same real context
        # must recover on all four frozen original requests.
        media = dict(records[0]['request'], images=['unsupported.png'])
        for bad in ('{','{}',json.dumps(media, ensure_ascii=False, allow_nan=False)):
            try: run(bad)
            except ValueError: pass
            else: raise ValueError('unsupported request accepted')
        result['controls']['three_input_rejections']=True; save()
        first=None
        for record, case, text in zip(records,cases,raw):
            started=time.monotonic(); response=run(text); decoded=json.loads(response)
            same=decoded==case['official_response']
            row={'id':record['id'],'response':decoded,'complete_typed_exact':same,'elapsed_seconds':time.monotonic()-started,
                 'response_utf8_sha256':hashlib.sha256(response.encode()).hexdigest()}
            result['cases'].append(row);save();require(same,'complete original response differs: '+record['id'])
            if first is None:first=response
        # A failure after success must not expose stale success; subsequent valid
        # request must reproduce the actual first response bytes.
        try: run('{')
        except ValueError: pass
        else: raise ValueError('post-success malformed request accepted')
        require(run(raw[0])==first,'post-error recovery/repeat differs')
        result['controls']['post_success_recovery_repeat']=True
        result.update(completed=True,full_typed_parity=True);save()
    except Exception as e:
        result['failure']=str(e);save();raise
    finally: close()

if __name__=='__main__': main()
