"""Frozen input-only source/protocol admission. Standard library only."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/ocr-rgb-input-v1.json'
PROFILE=('-O3','-DNDEBUG','-fPIC','-mavx2','-mfma','-mf16c','-std=gnu++17')
GGML='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
MANIFEST=dict(path='docs/ocr-real-scans-v1.json',bytes=75543,sha256='0857bffc6f4a3616292db255d7f3f34c0b58a52a7286a2791cb00b77499c7700')
REQUIRED={'tools/ocr-rgb-input/contract.py','tools/ocr-rgb-input/audit.py','tools/ocr-rgb-input/metrics.py',
 'tools/ocr-rgb-input/requirements.lock','tests/ocr-rgb-input/CMakeLists.txt','tests/ocr-rgb-input/probe.cpp',
 'tests/test_ocr_rgb_input.py','.github/workflows/ocr-rgb-input-audit-v1.yml','docs/ocr-rgb-input-v1.md',
 'CMakeLists.txt','.gitmodules','src/image_preprocess.cpp','src/image_preprocess.h','src/scan_cleanup.h',
 'src/core/clean_exit.h','src/core/json.h','examples/cli/main.cpp','src/minicpm_ocr.cpp','src/unlimited_ocr.cpp',
 'docs/ocr-real-scans-v1.json','docs/unlimited-ocr2-source.json','tools/kaggle/ocr-real-scans/ref_worker.py',
 'tools/kaggle/ocr-real-scans/kernel.py','tools/ci_process.py'}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while data:=stream.read(2**20):h.update(data)
    return h.hexdigest()


def prefix(source):
    marker=b'#include "../ggml/examples/stb_image.h"\n'
    require(source.count(marker)==1,'exact production STB include required')
    value=source[:source.index(marker)+len(marker)]
    require(value.count(b'#define STB_IMAGE_IMPLEMENTATION')==1 and b'STB_IMAGE_STATIC'not in value and
            b'#define STBI_'not in value,'production decoder macros differ')
    return value


def admit(data,pin):
    require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],
            'encoded image identity differs; decoder forbidden')


def admitted_decode(data,pin,decoder):
    admit(data,pin)
    return decoder(data)


def validate(cfg):
    require(cfg['schema']=='crispembed-ocr-rgb-input-v1'and cfg['pages']==20 and
            cfg['normalization_changed']is False and cfg['model_execution']is False and
            cfg['model_qualified']is False and cfg['cause_established']is False and
            tuple(cfg['compile_profile'])==PROFILE and cfg['manifest']==MANIFEST and cfg['ggml']==GGML and
            REQUIRED<=set(cfg['source_sha256']),'input audit policy differs')
    return cfg


def source(header=False):
    cfg=validate(json.loads(PATH.read_text()))
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT)
    require(git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'dirty input audit protocol')
    for name,pin in cfg['source_sha256'].items():
        p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'input audit source differs: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'production STB source gitlink differs')
    require(not git('status','--porcelain','--','ggml'),'dirty production decoder submodule')
    manifest=ROOT/'docs/ocr-real-scans-v1.json'
    require(manifest.stat().st_size==cfg['manifest']['bytes']and sha(manifest)==cfg['manifest']['sha256'],'frozen manifest differs')
    rows=json.loads(manifest.read_text())['pages'];require(len(rows)==20 and len({r['id']for r in rows})==20,'full unique20-page scope required')
    require(hashlib.sha256(prefix((ROOT/'src/image_preprocess.cpp').read_bytes())).hexdigest()==cfg['production_prefix_sha256'],'production prefix differs')
    if header:
        p=ROOT/'ggml/examples/stb_image.h'
        require(p.is_file()and p.stat().st_size==cfg['stb_header']['bytes']and sha(p)==cfg['stb_header']['sha256'],'exact production STB header differs')
    return cfg


if __name__=='__main__':print(json.dumps(source(),indent=2))
