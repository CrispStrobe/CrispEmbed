#!/usr/bin/env python3
"""Exact all-size Q8 source closure; declarations never waive model gates."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'docs/moedern-q8-f32-critical-v1.json'


def verify(reviewed_source):
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    def require(ok,why):
        if not ok:raise ValueError(why)
    p=json.loads(PROFILE.read_text())
    require(re.fullmatch('[0-9a-f]{40}',reviewed_source or '')is not None and git('rev-parse','HEAD')==reviewed_source,'exact reviewed source required')
    paths=[*p['protected_git_objects'],*p['owned_sha256'],str(PROFILE.relative_to(ROOT))]
    require(not git('status','--porcelain','--',*paths),'dirty Q8 protected source')
    for path,obj in p['protected_git_objects'].items():
        require(git('rev-parse',p['base']+':'+path)==obj and git('rev-parse','HEAD:'+path)==obj,'protected Git object differs: '+path)
        if (ROOT/path).is_file():require(git('hash-object','--',path)==obj,'protected worktree differs: '+path)
    for path,digest in p['owned_sha256'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,'owned Q8 source differs: '+path)
    for path,obj in p['changed_from_base_git_objects'].items():require(git('rev-parse',p['base']+':'+path)==obj,'initial MoE source object differs')
    return dict(schema=p['schema'],source=reviewed_source,base=p['base'],profile_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
                protected_objects=len(p['protected_git_objects']),owned_files=len(p['owned_sha256']),model_qualified=False,dispatch_authorized=False)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--reviewed-source',required=True)
    print(json.dumps(verify(a.parse_args().reviewed_source),indent=2))
