"""New protected Q4_K experiment closure; historical guards remain immutable."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'docs/josev2b-q4k-v1.json'
REFERENCE_SHA='51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090'
FREEZE_SHA='9c85fae56186517b5721be547bc74e8f92bd8e7cdb45d38afb4340e029161fe7'
ORIGINAL_SOURCE='b2edf8c83c9aca1f8a9210198d47798ac13828b8'
REQUIRED={'tools/quantize.cpp','tools/josev2b_composed_panel.py','tools/josev2b_composed_diff.py',
          'tools/josev2b_head_schema.py','tests/josev-packed/probe.cpp','tests/josev-packed/json.h',
          'tests/josev-packed/diff.cpp','tests/josev-packed/metadata.cpp','tests/josev-packed/head-shapes.h',
          'tests/josev-quant/controls.cpp','tests/josev-quant/backbone-shapes.h',
          'tests/josev2b_composed_observer.h','tests/josev2b_head_observer.h',
          'docs/josev2b-environment-lock.json','tools/josev2b_environment.py',
          'tests/results/josev2b/backbone-f32.json','tests/results/josev2b/original-reference.json',
          'tests/results/josev2b/original-input-freeze.json','CMakeLists.txt','.gitmodules'}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(4*2**20):h.update(block)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def validate(cfg):
    require(cfg['schema']=='crispembed-josev-q4k-source-v1'and cfg['default_enabled']is False and
            cfg['publication_qualified']is False and cfg['independent_task_quality']['reviewed']is False,'Q4_K policy differs')
    runtime=set(git('ls-tree','-r','--name-only',cfg['base'],'src').decode().splitlines())
    require(runtime|REQUIRED<=set(cfg['protected_sha256']),'complete production/helper source closure required')
    require(not any(name.startswith('src/')for name in cfg['owned_sha256']),'this packer/harness lane cannot own inference source')
    require(cfg['original']==dict(version_label='v5',source=ORIGINAL_SOURCE,reference_sha256=REFERENCE_SHA,
                                  freeze_sha256=FREEZE_SHA,arrays=251),'frozen original identity differs')
    require(cfg['f32_gates']==dict(max_absolute=.002,min_row_cosine=.99999,logits_absolute=.002,
                                 probabilities_absolute=.0001,ordinal_expectation_absolute=.001),'F32 gates changed')
    require(cfg['quant_task_decisions']==['complete_typed_json_exact','ordered_discrete_option_argmax_exact',
                                        'ordinal_expectation_within_original_0.001'],'quant decision checks changed')
    return cfg


def verify():
    cfg=validate(json.loads(PATH.read_text()))
    require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'dirty Q4_K contract')
    declared=set(cfg['owned_sha256'])|{str(PATH.relative_to(ROOT))}
    require(set(git('diff','--name-only',cfg['base'],'HEAD').decode().splitlines())<=declared,'undeclared source delta')
    require(git('ls-tree','-r','--name-only','HEAD','src')==git('ls-tree','-r','--name-only',cfg['base'],'src'),
            'new or removed inference source is outside this lane')
    for group in ('owned_sha256','protected_sha256'):
        for name,pin in cfg[group].items():
            path=ROOT/name
            require(path.is_file()and not path.is_symlink()and sha(path)==pin and git('show','HEAD:'+name)==path.read_bytes(),'source differs: '+name)
            if group=='protected_sha256':require(git('show',cfg['base']+':'+name)==path.read_bytes(),'protected base changed: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'GGML source differs')
    return cfg


if __name__=='__main__':print(json.dumps(verify(),indent=2))
