"""Explicit new A/B/C source closure; neither old9c nor holdout guards change."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/josev2b-mixed-recipes-v1.json'
from josev2b_q4k_contract import require,reference_extents,REFERENCE_SHA,FREEZE_SHA,ORIGINAL_SOURCE
from quantize_josev2b_mixed import COUNTS,EXPECTED_BYTES,SOURCE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while block:=f.read(4*2**20):h.update(block)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def validate(cfg):
    require(cfg['schema']=='crispembed-josev-mixed-recipes-v1'and cfg['default_enabled']is False and
            cfg['publication_qualified']is False and cfg['holdout_access']is False and cfg['recipe_selected']is False,'development scope differs')
    require(cfg['runtime_source']=='9c475d0dac70264adc123a63b6c1827b4b7280d5'and
            git('rev-parse',cfg['base']+':src')==git('rev-parse',cfg['runtime_source']+':src'),'runtime source changed')
    require(cfg['source_model']==SOURCE and cfg['recipes']=={n:dict(counts={str(t):c for t,c in COUNTS[n].items()},
            expected_bytes=EXPECTED_BYTES[n])for n in ('A','B','C')},'immutable role recipes differ')
    require(cfg['gates']==dict(max_absolute=.002,min_row_cosine=.99999,logits_absolute=.002,
            probabilities_absolute=.0001,ordinal_expectation_absolute=.001),'historical budgets changed')
    require(cfg['original_reference_sha256']==REFERENCE_SHA and cfg['original_freeze_sha256']==FREEZE_SHA,'original identities differ')
    old=json.loads((ROOT/'docs/josev2b-q4k-v1.json').read_text())
    protected=set(old['protected_sha256'])|set(old['owned_sha256'])|{'docs/josev2b-q4k-v1.json',
        'tools/ci-heavy/josev2b_composed.py','tools/josev2b_composed_contract.py','docs/josev2b-composed-dependencies.json'}
    protected|=set(git('ls-tree','-r','--name-only',cfg['base'],'src','tests/josev-q4k','tests/josev-packed').decode().splitlines())
    require(protected<=set(cfg['protected_sha256']),'complete runtime/build/reference/scorer/legacy closure required')
    require(not any(p.startswith('src/')for p in cfg['owned_sha256']),'no runtime changes in this caller')
    require(cfg['original_artifact']==json.loads((ROOT/'docs/josev2b-composed-dependencies.json').read_text())['fixed']['head'],
            'reviewed original artifact differs')
    return cfg


def verify():
    cfg=validate(json.loads(PATH.read_text()));rel=str(PATH.relative_to(ROOT))
    require(git('show','HEAD:'+rel)==PATH.read_bytes(),'dirty new contract')
    require(set(git('diff','--name-only',cfg['base'],'HEAD').decode().splitlines())<=set(cfg['owned_sha256'])|{rel},'undeclared source delta')
    require(git('ls-tree','-r','HEAD','src')==git('ls-tree','-r',cfg['base'],'src'),'runtime inventory differs')
    for group in ('protected_sha256','owned_sha256'):
        for name,pin in cfg[group].items():
            p=ROOT/name
            require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'source differs: '+name)
            if group=='protected_sha256':require(git('show',cfg['base']+':'+name)==p.read_bytes(),'protected source changed: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'backend gitlink differs')
    return cfg


if __name__=='__main__':print(json.dumps(verify(),indent=2))
