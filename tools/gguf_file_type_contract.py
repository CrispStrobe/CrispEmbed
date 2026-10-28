"""Closed metadata-only successor; preserved9c→633 MoE additions are explicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/gguf-standard-file-type-v1.json'
EXISTING={'src/josev2b_packed_profile.h','src/ultrabert_profile.h','src/moedern_q4k_profile.h','tools/quantize.cpp'}
NEW_HEADER='src/core/gguf_file_type.h'
MOE={'src/moedern_family_engine.cpp','src/moedern_family_engine.h','src/moedern_q4k_profile.h',
     'src/moedern_text.cpp','src/moedern_text_frontend.cpp','src/moedern_text_frontend.h'}
from josev2b_q4k_contract import REFERENCE_SHA,FREEZE_SHA,ORIGINAL_SOURCE,reference_extents


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while block:=f.read(4*2**20):h.update(block)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def validate(cfg):
    require(cfg['schema']=='crispembed-gguf-standard-file-type-source-v1'and cfg['publication_qualified']is False and
            cfg['runtime_math_changed']is False and cfg['original_rerun']is False,'metadata-only scope differs')
    require(cfg['existing_delta'].keys()==EXISTING and cfg['preserved_moedern_delta'].keys()==MOE,'exact metadata/baseline deltas required')
    require(git('rev-parse',cfg['base']+':src').decode().strip()==cfg['baseline_633_src_tree']and
            cfg['baseline_633_src_tree']=='0c4341808c7136b84f0085da85e8c22b951c27a1','exact633 baseline required')
    require(set(git('diff','--name-only',cfg['baseline_9c'],cfg['base'],'--','src').decode().splitlines())==MOE,'9c→633 baseline delta differs')
    for name,pin in cfg['preserved_moedern_delta'].items():
        require(git('rev-parse',cfg['base']+':'+name).decode().strip()==pin['base_blob'],'preserved633 object differs')
        if pin['old_blob']is not None:require(git('rev-parse',cfg['baseline_9c']+':'+name).decode().strip()==pin['old_blob'],'old9c object differs')
        else:require(name not in git('ls-tree','-r','--name-only',cfg['baseline_9c'],'src').decode().splitlines(),'old absent object differs')
    require(not git('diff','--name-only',cfg['baseline_9c'],cfg['base'],'--','CMakeLists.txt','bindings'),'baseline CMake/bindings changed')
    required=set(git('ls-tree','-r','--name-only',cfg['base'],'src','tests/josev-q4k','tests/josev-packed','bindings').decode().splitlines())-EXISTING
    legacy=json.loads((ROOT/'docs/josev2b-q4k-v1.json').read_text())
    required|=(set(legacy['owned_sha256'])|set(legacy['protected_sha256']))-EXISTING
    required|={'docs/josev2b-q4k-v1.json','tools/quantize_josev2b_mixed.py','tools/ci-heavy/josev2b_mixed_recipes.py',
               'tools/josev2b_mixed_contract.py','docs/josev2b-mixed-recipes-v1.json'}
    require(required<=set(cfg['protected_sha256']),'complete source/build/scorer/original/legacy closure required')
    require({p for p in cfg['owned_sha256']if p.startswith('src/')}=={NEW_HEADER},'only declared standard metadata header may be new runtime source')
    require(cfg['ggml']=='a06fc0bb46b73cc10011fc90ae4054aee8438aeb','backend must remain exacta06')
    return cfg


def verify(reviewed_source):
    cfg=validate(json.loads(PATH.read_text()));head=git('rev-parse','HEAD').decode().strip()
    require(head==reviewed_source,'explicit reviewed source differs')
    if os.environ.get('GITHUB_ACTIONS')=='true':require(os.environ.get('GITHUB_SHA')==head,'service source differs')
    rel=str(PATH.relative_to(ROOT));require(git('show','HEAD:'+rel)==PATH.read_bytes(),'dirty source protocol')
    allowed=set(cfg['owned_sha256'])|EXISTING|{rel}
    require(set(git('diff','--name-only',cfg['base'],'HEAD').decode().splitlines())<=allowed,'undeclared source delta')
    before=set(git('ls-tree','-r','--name-only',cfg['base'],'src').decode().splitlines())
    require(set(git('ls-tree','-r','--name-only','HEAD','src').decode().splitlines())==before|{NEW_HEADER},'runtime file inventory changed')
    for group in ('protected_sha256','owned_sha256'):
        for name,pin in cfg[group].items():
            p=ROOT/name
            require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'source differs: '+name)
            if group=='protected_sha256':require(git('show',cfg['base']+':'+name)==p.read_bytes(),'protected baseline changed: '+name)
    for name,pin in cfg['existing_delta'].items():
        p=ROOT/name
        require(sha(p)==pin['sha256']and git('show','HEAD:'+name)==p.read_bytes()and
                hashlib.sha256(git('show',cfg['base']+':'+name)).hexdigest()==pin['base_sha256'],'metadata delta differs: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'GGML gitlink differs')
    if os.environ.get('GITHUB_ACTIONS')=='true':
        require(not(ROOT/'ggml').is_symlink()and
                subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==cfg['ggml']and
                not subprocess.check_output(['git','-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no']),
                'actual backend checkout differs or is dirty')
    return cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True);a=p.parse_args()
    print(json.dumps(verify(a.reviewed_source),indent=2))
