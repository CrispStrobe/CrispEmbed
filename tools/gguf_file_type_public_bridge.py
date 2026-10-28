"""Actual SystemOne C ABI old/canonical byte continuity, four frozen requests.

Called only after source/library/model/original identities and stage controls
have passed in the hosted bridge. No original model forward or hidden injection.
"""
import ctypes
import json
import os
import subprocess
from pathlib import Path
from gguf_file_type_canonicalize import require,sha


def compare(library,proof,source,legacy,canonical,freeze):
    require(proof['source']==source and proof['sha256']==sha(library),'exact freshly built public C ABI library proof required')
    require(len(freeze['records'])==4,'old four frozen requests required')
    symbols=subprocess.check_output(['nm','-C',str(library)],text=True,timeout=120)
    strings=subprocess.check_output(['strings',str(library)],text=True,timeout=120)
    require('josev2b::composed_test::'not in symbols and 'josev2b_head::test::'not in symbols and
            'head/option_context_projection'not in strings,'public library contains Josev observers')
    lib=ctypes.CDLL(str(Path(library).resolve()))
    lib.crispembed_systemone_init.argtypes=[ctypes.c_char_p,ctypes.c_int];lib.crispembed_systemone_init.restype=ctypes.c_void_p
    lib.crispembed_systemone_free.argtypes=[ctypes.c_void_p];lib.crispembed_systemone_free.restype=None
    lib.crispembed_systemone_run_json.argtypes=[ctypes.c_void_p,ctypes.c_char_p];lib.crispembed_systemone_run_json.restype=ctypes.c_char_p
    lib.crispembed_systemone_error.argtypes=[ctypes.c_void_p];lib.crispembed_systemone_error.restype=ctypes.c_char_p
    gates=('CRISPEMBED_JOSEV_QUANT','CRISPEMBED_QWEN35_QUANT');saved={g:os.environ.get(g)for g in gates};arms={}
    def flag(key,value):
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value
    try:
        for label,path in (('legacy',legacy),('canonical',canonical)):
            for g in gates:flag(g,'1')
            ctx=lib.crispembed_systemone_init(str(Path(path).resolve()).encode(),2)
            require(bool(ctx),'public model init rejected')
            try:
                responses=[]
                for row in freeze['records']:
                    request=json.dumps(row['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                    answer=lib.crispembed_systemone_run_json(ctx,request);require(answer is not None,'public decision rejected')
                    answer=bytes(answer);json.loads(answer.decode(),parse_constant=lambda _:require(False,'nonfinite public response'))
                    responses.append(answer)
                request=json.dumps(freeze['records'][0]['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                require(lib.crispembed_systemone_run_json(ctx,None)is None and bool(lib.crispembed_systemone_error(ctx)),'public NULL failure absent')
                checks=[]
                for g in gates:
                    for value in (None,'','0'):
                        flag(g,value)
                        rejected=lib.crispembed_systemone_init(str(Path(path).resolve()).encode(),2)
                        if rejected:lib.crispembed_systemone_free(rejected)
                        require(not rejected,'disabled public quant init accepted')
                        require(lib.crispembed_systemone_run_json(ctx,request)is None and bool(lib.crispembed_systemone_error(ctx)),'disabled public run accepted')
                        checks.append(dict(gate=g,value=value,init_rejected=True,run_rejected=True));flag(g,'1')
                recovery=lib.crispembed_systemone_run_json(ctx,request)
                require(recovery is not None and bytes(recovery)==responses[0],'public recovery bytes changed')
                arms[label]=dict(response_texts=[v.decode()for v in responses],response_sha256=[__import__('hashlib').sha256(v).hexdigest()for v in responses],
                                 null_rejected=True,gate_controls=checks,recovery_exact=True)
            finally:lib.crispembed_systemone_free(ctx)
    finally:
        for g,value in saved.items():flag(g,value)
    require(arms['legacy']['response_texts']==arms['canonical']['response_texts'],'public typed response bytes changed after canonicalization')
    return dict(completed=True,source=source,library_sha256=proof['sha256'],cases=4,arms=arms,
                all_public_capi_response_bytes_exact=True,production_observers_excluded=True,publication_qualified=False)
