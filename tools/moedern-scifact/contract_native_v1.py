"""Explicit new full-retrieval replay closure; no historical guard exceptions."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/moedern-scifact-native-q4k-v1.json'
REQUIRED={'CMakeLists.txt','.gitmodules','tools/quantize.cpp','tools/ci_process.py',
 'tools/moedern-colbert/public_clients.py','tools/moedern-colbert/quantize_q4k_padded.py',
 'tools/ci-heavy/moedern_q4k_padded.py','tools/ci-heavy/moedern_public_consumers.py',
 'tools/moedern-native-v2/replay.py','tools/moedern-native-v2/diff_adapter.py',
 'tools/moedern-native-v2/contract_v2.py','tools/moedern_scifact.py',
 'tools/ci-heavy/moedern_scifact.py','tools/moedern-colbert/scifact_original.py',
 'docs/moedern-scifact-parity-v1.json','docs/moedern-colbert-source-contract.json',
 'docs/moedern-q4k-padded.json','docs/moedern-colbert-native-contract-v2.json',
 'tests/moedern-q4k/family_control.cpp','tests/moedern-text/diff.cpp',
 'tools/moedern-colbert/quantize_gguf.py','docs/moedern-text-composed.json',
 'tests/moedern-q4k/controls.cpp','tests/moedern-q4k/recovery.cpp','tests/moedern-public/consumer.c',
 'tests/moedern-public/python_consumer.py','tests/moedern-public/rust_consumer.rs','tests/moedern-public/dart_consumer.dart',
 'tests/fixtures/moedern-colbert/panel-v1.json','tests/test_moedern_all_size_clients.py'}
OWNED={'PLAN.md','tests/test_ci_process.py','docs/moedern-scifact-native-q4k-v1.md','tools/moedern-scifact/contract_native_v1.py',
 'tools/moedern-scifact/original_inputs_v1.py','tools/moedern-scifact/providers_v1.py',
 'tools/moedern-scifact/capture_v1.py','tools/moedern-scifact/compare_v1.py',
 'tools/moedern-scifact/authored_v1.py','tools/moedern-scifact/worker_v1.py','tools/moedern-scifact/diff_adapter_v1.py',
 'tests/moedern-scifact/CMakeLists.txt','tests/moedern-scifact/tokenize.cpp','tests/moedern-scifact/diff.cpp',
 'tools/kaggle/moedern-scifact-native-q4k/kernel.py','tools/kaggle/moedern-scifact-native-q4k/kernel-metadata.json',
 'tools/kaggle/moedern-scifact-native-q4k/requirements.lock','tests/test_moedern_scifact_native_q4k.py'}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(4*2**20):h.update(block)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def validate(cfg):
    require(cfg['schema']=='crispembed-moedern-scifact-native-q4k-v1'and cfg['default_enabled']is False and
            cfg['publication_qualified']is False and cfg['shards']==8,'native retrieval policy differs')
    require(cfg['gates']==dict(max_absolute=.002,min_row_cosine=.99999,max_metric_drop=.001,min_top1_agreement=.99),'unchanged full gates required')
    paths=set(git('ls-tree','-r','--name-only',cfg['base'],'src','examples','python','crispembed','crispembed-sys','flutter/crispembed/lib','cmake').decode().splitlines())
    require(paths|REQUIRED<=set(cfg['protected_sha256']),'complete runtime/consumer/helper closure required')
    require(set(cfg['owned_sha256'])==OWNED and not set(cfg['owned_sha256'])&set(cfg['protected_sha256']),'exact replay ownership required')
    require(set(cfg['originals'])=={'1.5M','27M','115M'},'all three original declarations required')
    for variant,pin in cfg['originals'].items():
        name=pin['public_report'];require(name in cfg['protected_sha256']and cfg['protected_sha256'][name]==pin['report_sha256'],
                                         'original public report identity omitted: '+variant)
    return cfg


def source(variant=None):
    cfg=validate(json.loads(PATH.read_text()))
    require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'dirty native retrieval contract')
    declared=set(cfg['owned_sha256'])|{str(PATH.relative_to(ROOT))}
    require(set(git('diff','--name-only',cfg['base'],'HEAD').decode().splitlines())<=declared,'undeclared replay source delta')
    for group in ('owned_sha256','protected_sha256'):
        for name,pin in cfg[group].items():
            path=ROOT/name;require(path.is_file()and not path.is_symlink()and sha(path)==pin and git('show','HEAD:'+name)==path.read_bytes(),'source differs: '+name)
            if group=='protected_sha256':require(git('show',cfg['base']+':'+name)==path.read_bytes(),'protected source differs from base: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'backend source differs')
    require(not git('status','--porcelain','--untracked-files=no','--','src','ggml','CMakeLists.txt','tools/quantize.cpp'),
            'dirty production/backend source')
    if variant is not None:admit(cfg,variant)
    return cfg


def admit(cfg,variant):
    require(variant in ('1.5M','27M','115M'),'explicit variant required')
    pin=cfg['originals'][variant]
    require(pin['reviewed']is True and pin['version_label'].startswith('v')and
            re.fullmatch('[0-9a-f]{64}',pin.get('report_sha256')or'')and
            type(pin.get('report_bytes'))is int and 0<pin['report_bytes']<2*2**20,
            'reviewed exact original freeze pending; no payload/model access')
    return pin


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=('1.5M','27M','115M'));args=p.parse_args()
    print(json.dumps(source(args.variant),indent=2))
