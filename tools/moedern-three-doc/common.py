"""New three-document attribution closure; no framework/model import on load."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/moedern-three-doc-attribution-v1.json'
KEYS=['d:24726600','d:24928817','d:5849439']


class ContractError(ValueError):pass

def require(ok,why):
    if not ok:raise ContractError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while b:=f.read(4*2**20):h.update(b)
    return h.hexdigest()


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('w')as f:json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def declaration():
    cfg=json.loads(PATH.read_text())
    require(cfg['schema']=='crispembed-moedern-three-doc-source-v1'and cfg['keys']==KEYS and
            cfg['gates']==dict(max_absolute=.002,min_row_cosine=.99999)and
            cfg['publication_qualified']is False and cfg['full_original_recapture']is False,'exact bounded diagnostic protocol required')
    require(cfg['runtime_source']=='93d27614caba00a1b4d7aece6c78968f5cd9bb7f'and cfg['ggml']=='a06fc0bb46b73cc10011fc90ae4054aee8438aeb','frozen runtime identity differs')
    original=json.loads((ROOT/cfg['original']['public_report']).read_text())
    native=json.loads((ROOT/cfg['native']['public_report']).read_text())
    reviewed=json.loads((ROOT/'tests/results/moedern-colbert/scifact-native-1p5m-v3-review.json').read_text())
    require(native['f32']['capture']['completed']is True and native['f32']['capture']['rows']==5483 and
            native['f32']['capture']['failure_count']==0 and native['f32']['tokens']['passed']is True,'retained native must be structurally complete')
    require(cfg['original']['members']=={n:original['arrays'][n]for n in('data.json','rows.json','all.f32','selected.f32')}and
            cfg['native']['members']==native['f32']['capture']['files'],'frozen file inventory differs')
    require(cfg['historical_failures']==reviewed['vectors']['negative_rows']and cfg['failing_shapes']==
            {r['key']:r['shape']for r in reviewed['vectors']['negative_rows']if r['stage']=='normalized_all'},'old numerical negatives changed')
    require(cfg['original']['service']['source']==original['source']and cfg['original']['service']['version_label']=='v1'and
            cfg['original']['service']['status']=='COMPLETE'and cfg['native']['service']['source']==native['source']and
            cfg['native']['service']['version_label']=='v3'and cfg['native']['service']['status']=='ERROR','frozen producer versions/states differ')
    required=set(git('ls-tree','-r','--name-only',cfg['base'],'src','tests/moedern-text','tools/moedern-colbert',
                          'tools/moedern-native-v2','tools/moedern-scifact').decode().splitlines())
    required|={'CMakeLists.txt','.gitmodules','VERSION','tools/ci_process.py','docs/moedern-colbert-source-contract.json',
        'docs/moedern-colbert-native-contract-v2.json','docs/moedern-scifact-parity-v1.json',
        'tests/results/moedern-colbert/scifact-original-1p5m.json','tests/results/moedern-colbert/scifact-native-1p5m-v3-native.json',
        'tests/results/moedern-colbert/scifact-native-1p5m-v3-review.json','tests/results/moedern-colbert/publication-1p5m.json'}
    require(required<=set(cfg['protected_sha256']),'complete source/original/observer/comparator closure required')
    require(not any(p.startswith('src/')for p in cfg['owned_sha256']),'diagnostic may not own runtime code')
    return cfg


def verify(reviewed):
    cfg=declaration();require(git('rev-parse','HEAD').decode().strip()==reviewed,'exact source required')
    require(git('ls-tree','-r','HEAD','src')==git('ls-tree','-r',cfg['runtime_source'],'src')and
            git('show','HEAD:CMakeLists.txt')==git('show',cfg['runtime_source']+':CMakeLists.txt'),'runtime/root build differs from93')
    rel=str(PATH.relative_to(ROOT));require(git('show','HEAD:'+rel)==PATH.read_bytes(),'dirty protocol')
    require(set(git('diff','--name-only',cfg['base'],'HEAD').decode().splitlines())<=set(cfg['owned_sha256'])|{rel},'undeclared diagnostic delta')
    for group in ('owned_sha256','protected_sha256'):
        for name,pin in cfg[group].items():
            p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'source differs: '+name)
            if group=='protected_sha256':require(git('show',cfg['base']+':'+name)==p.read_bytes(),'protected baseline changed: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==cfg['ggml'],'backend gitlink differs')
    if os.environ.get('GITHUB_ACTIONS')=='true':
        require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==cfg['ggml']and
                not subprocess.check_output(['git','-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no']),
                'actual backend checkout differs or is dirty')
    return cfg


def span(path,offset,count):
    require(type(offset)is int and type(count)is int and offset>=0 and count>0 and offset+count<=Path(path).stat().st_size,'bounded frozen span required')
    with Path(path).open('rb')as f:f.seek(offset);data=f.read(count)
    require(len(data)==count,'span truncated');return data
