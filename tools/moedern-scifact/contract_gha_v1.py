"""Separately versioned provider adapter; all corrected3c math stays frozen."""
import json
from pathlib import Path
import re
import subprocess
import contract_native_v1 as old
ROOT=old.ROOT
PATH=ROOT/'docs/moedern-scifact-gha-adapter-v1.json'
BASE='3c3e4c8e7f5e645855aef754250e8d6fda1321ff'
require=old.require
sha=old.sha
git=old.git
OWNED={'PLAN.md','docs/moedern-scifact-gha-adapter-v1.md','docs/moedern-scifact-gha-adapter-v1.json',
       'tools/moedern-scifact/contract_gha_v1.py','tools/moedern-scifact/worker_gha_v1.py',
       'tools/ci-heavy/moedern_scifact_gha_v1.py','tests/test_moedern_scifact_gha_adapter.py',
       '.github/workflows/moedern-scifact-gha-adapter-v1.yml'}


def declaration():
    cfg=json.loads(PATH.read_text())
    require(cfg['schema']=='crispembed-moedern-scifact-gha-adapter-v1'and cfg['base']==BASE and
            cfg['enabled_variants']==['1.5M'] and cfg['minimum_available_ram_bytes']==14*2**30 and
            cfg['minimum_free_disk_bytes']==24*2**30 and cfg['inference_changes']is False and
            cfg['publication_qualified']is False,'Provider-only declared scope/resource limits differ')
    return cfg


def source(variant=None):
    cfg=declaration()
    require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Dirty GHA adapter declaration')
    require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared GHA adapter source delta')
    require(set(cfg['owned_sha256'])==OWNED-{'PLAN.md',str(PATH.relative_to(ROOT))},'Exact GHA adapter ownership required')
    for group in ['protected_sha256','owned_sha256']:
        for name,expected in cfg[group].items():
            path=ROOT/name
            require(path.is_file()and not path.is_symlink()and sha(path)==expected and
                    git('show','HEAD:'+name)==path.read_bytes(),'Provider/dependency source differs: '+name)
            if group=='protected_sha256':require(git('show',BASE+':'+name)==path.read_bytes(),'Corrected3c source changed: '+name)
    require(git('rev-parse','HEAD:src')==git('rev-parse',BASE+':src')and
            git('rev-parse','HEAD:ggml')==git('rev-parse',BASE+':ggml'),'Runtime/backend changed in provider adapter')
    require(not git('status','--porcelain','--untracked-files=no','--','src','ggml','CMakeLists.txt'),'Dirty production/backend')
    baseline=old.validate(json.loads(old.PATH.read_text()))
    required=(set(baseline['protected_sha256'])|set(baseline['owned_sha256'])|{str(old.PATH.relative_to(ROOT))})-{'PLAN.md'}
    require(required<=set(cfg['protected_sha256']),'Frozen worker/helper/math closure incomplete')
    if variant is not None:
        require(variant in cfg['enabled_variants'],'Only first1.5M GHA activation declared; other variants pending review')
        old.admit(baseline,variant)
    baseline['provider_adapter']=dict(schema=cfg['schema'],base=BASE,contract_sha256=sha(PATH),
                                     inference_changes=False,source=git('rev-parse','HEAD').decode().strip())
    return baseline
