#!/usr/bin/env python3
"""Explicit packed-runtime source closure; source only, no payloads."""
from pathlib import Path
import hashlib
import json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def verify():
    cfg=json.loads((ROOT/'docs/josev2b-packed-runtime.json').read_text())
    if cfg['schema']!='crispembed-josev-packed-runtime-v1' or cfg['default_enabled']:
        raise ValueError('packed runtime policy differs')
    before=set(git('ls-tree','-r','--name-only',cfg['base'],'src').decode().splitlines())
    after=set(git('ls-tree','-r','--name-only','HEAD','src').decode().splitlines())
    if set(cfg['protected'])!=before-set(cfg['changed_runtime']) or after!=before|set(cfg['new_runtime']):
        raise ValueError('packed native source closure omitted')
    for name,expected in cfg['protected'].items():
        if git('rev-parse','HEAD:'+name).decode().strip()!=expected or git('rev-parse',cfg['base']+':'+name).decode().strip()!=expected:
            raise ValueError('old native source differs: '+name)
    for name,expected in cfg['protected_roots'].items():
        if git('rev-parse','HEAD:'+name).decode().strip()!=expected:raise ValueError('old root differs: '+name)
    for name,expected in cfg['owned_sha256'].items():
        data=(ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected or git('show','HEAD:'+name)!=data:
            raise ValueError('packed source differs: '+name)
    paths=list(cfg['protected'])+list(cfg['owned_sha256'])+['docs/josev2b-packed-runtime.json']
    if git('status','--porcelain','--',*paths):raise ValueError('dirty packed source')
    return cfg
if __name__=='__main__':
    c=verify();print(json.dumps({'completed':True,'protected_runtime_files':len(c['protected']),
                               'owned_files':len(c['owned_sha256']),'model_payloads':False}))
