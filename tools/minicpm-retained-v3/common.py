"""Frozen-v3 source/catalog authentication. No array/model package imports."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
BASE='6ec547197510ea912815ed5f88030de10c5f8ee3'
OLD_SOURCE='83eac856cc684808e417ebfed7db7c2f677dc09f'
CATALOG=ROOT/'docs/minicpm-graph-retained-v3-catalog.json'
ANALYSIS=ROOT/'docs/minicpm-graph-retained-analysis-v1.json'
CONTINUE=ROOT/'docs/minicpm-graph-seven-arm-v1.json'
OUT_CAP=12*2**30
RESERVE=4*2**30


class ContractError(ValueError):pass

def require(ok,why):
    if not ok:raise ContractError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        while block:=stream.read(4*2**20):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    """An exhausted write cannot truncate the previous valid receipt."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8')as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


def declaration(mode):
    require(mode in ['analysis','continue'],'Explicit analysis/continue mode required')
    path=ANALYSIS if mode=='analysis'else CONTINUE
    cfg=json.loads(path.read_text())
    require(cfg['base']==BASE and cfg['old_source']==OLD_SOURCE and cfg['working_output_cap_bytes']==OUT_CAP and
            cfg['working_reserve_bytes']==RESERVE and cfg['original_rerun']is False and cfg['runtime_math_changes']is False,
            'Frozen source/storage/measurement scope differs')
    require(cfg['budgets']==dict(max_absolute=.002,min_row_cosine=.99999,paired_confidence_absolute=.002),
            'Historical numerical budgets may not change')
    require(sha(CATALOG)==cfg['catalog_sha256'],'Frozen23 catalog differs')
    if mode=='analysis':require(cfg['native_forwards']==0 and cfg['model_access']is False,'Analysis must not access weights or run forwards')
    else:require(cfg['remaining_arms']==[['tall','q4_k',1]]+[['tiny',d,g]for d,g in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]],
                 'Only seven absent arms may run; no23 replay')
    return cfg


def source_guard(mode,reviewed):
    cfg=declaration(mode)
    require(re.fullmatch('[0-9a-f]{40}',reviewed)and git('rev-parse','HEAD').decode().strip()==reviewed,'Explicit reviewed caller SHA differs')
    require(not git('status','--porcelain','--untracked-files=no'),'Dirty tracked caller source')
    path=ANALYSIS if mode=='analysis'else CONTINUE
    require(git('show','HEAD:'+str(path.relative_to(ROOT)))==path.read_bytes(),'Dirty versioned source contract')
    for group in ['protected_sha256','owned_sha256']:
        for name,expected in cfg[group].items():
            actual=ROOT/name
            require(actual.is_file()and not actual.is_symlink()and sha(actual)==expected and git('show','HEAD:'+name)==actual.read_bytes(),
                    'Frozen runtime/provider/scorer source differs: '+name)
            if group=='protected_sha256':require(git('show',BASE+':'+name)==actual.read_bytes(),'Runtime/helper drift from6ec: '+name)
    require(git('rev-parse','HEAD:src')==git('rev-parse',OLD_SOURCE+':src')and
            git('rev-parse','HEAD:ggml').decode().strip()=='a06fc0bb46b73cc10011fc90ae4054aee8438aeb','Runtime/backend differs from actualv3')
    return dict(source=reviewed,base=BASE,old_source=OLD_SOURCE,contract_sha256=sha(path),
                catalog_sha256=sha(CATALOG),protected_files=len(cfg['protected_sha256']),owned_files=len(cfg['owned_sha256']))


def safe(root,name):
    relative=PurePosixPath(name)
    require(not relative.is_absolute()and '..'not in relative.parts and '\\'not in name and relative.parts,
            'Unsafe retained artifact member')
    path=Path(root)/relative
    require(path.resolve().is_relative_to(Path(root).resolve())and not path.is_symlink(),'Unsafe/symlink retained artifact member')
    return path


def checked(root,name,identity):
    path=safe(root,name)
    require(path.is_file()and path.stat().st_size==identity['bytes']and sha(path)==identity['sha256'],
            'Retained canonical/report bytes differ: '+name)
    return path


def canonical(name):
    return (name.startswith('dump/')and name.endswith(('.npy','.json')))or bool(re.fullmatch(r'projected/projection_\d+\.(control|native)\.f32',name))


def duplicate(name):
    return bool(re.fullmatch(r'(original|native)_\d+\.f32',name)or re.fullmatch(r'projected/(layer_\d+|norm|logits)\.f32',name)or
                name in ['probe-input.gguf','merged.f32','ids-as-exact-f32.f32','prefill/original.gguf','prefill/native.gguf',
                         'projected/comparison/original.gguf','projected/comparison/native.gguf'])


def remaining(catalog):
    found={(r['id'],r['dtype'],r['graph'])for r in catalog['cases']}
    expected={(cid,d,g)for cid in ['line','para-en','table','tall','tiny']for d,g in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]}
    absent=expected-found
    result=[('tall','q4_k',1)]+[('tiny',d,g)for d,g in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]]
    require(len(found)==23 and absent==set(result),'Immutable23/remaining7 partition differs')
    return result


def room(current,next_bound,free):
    require(type(next_bound)is int and 0<=next_bound<=2*2**30 and current+next_bound<=OUT_CAP and free>=next_bound+RESERVE,
            'Working-output quota/reserve guard rejected before write/forward')


def used(directory):return sum(p.stat().st_size for p in Path(directory).rglob('*')if p.is_file()and not p.is_symlink())


def resources(out,scratch,next_bound=0):
    require(os.environ.get('GITHUB_ACTIONS')=='true'or(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in ['Batch','Interactive']and
            Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()),'Hosted-only task required')
    if shutil.which('nvidia-smi'):
        gpu=subprocess.run(['nvidia-smi','-L'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=10)
        require(not(gpu.returncode==0 and gpu.stdout.strip()),'CPU analysis/continuation must not reserve a GPU session')
    import platform
    require(platform.system()=='Linux'and platform.machine()=='x86_64','Frozen Linux x64 replay profile required')
    for path in [out,scratch]:Path(path).mkdir(parents=True,exist_ok=True)
    ram=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    require(ram>=10*2**30 and shutil.disk_usage(scratch).free>=32*2**30,'Unchanged10GiB RAM/32GiB scratch admission required')
    free=shutil.disk_usage(out).free;size=used(out);room(size,next_bound,free)
    return dict(available_ram_bytes=ram,scratch_free_bytes=shutil.disk_usage(scratch).free,working_free_bytes=free,
                working_retained_bytes=size,working_cap_bytes=OUT_CAP,working_reserve_bytes=RESERVE,cpu_count=os.cpu_count(),native_threads=4)


def numerical_environment():
    import importlib.metadata
    import sys
    require(not any(name in sys.modules for name in ['torch','transformers']),'Original framework import forbidden in retained replay')
    require(sys.version_info[:2]==(3,12)and sys.prefix!=sys.base_prefix,'Isolated original-v3 CPython3.12 tooling required')
    result={}
    for name,version in [('numpy','2.2.6'),('gguf','0.17.1'),('kaggle','2.2.2')]:
        distribution=importlib.metadata.distribution(name);metadata=Path(distribution._path)/'METADATA'
        require(distribution.version==version and metadata.resolve().is_relative_to(Path(sys.prefix).resolve()),'Tooling package owner/version differs: '+name)
        result[name]=dict(version=version,metadata_sha256=sha(metadata),isolated_owner=True)
    return dict(python_minor='3.12',packages=result,original_framework_imported=False)


def authenticate_old(directory):
    cfg=json.loads(CATALOG.read_text());remaining(cfg)
    report=checked(directory,'panel.json',cfg['panel']);panel=json.loads(report.read_text())
    require(panel['completed']is False and panel['qualified']is False and panel['error']=='[Errno 28] No space left on device'and
            panel['source']['source']==OLD_SOURCE and len(panel['cases'])==23,'Retained incomplete/numerical history differs')
    for name,identity in cfg['initial_files'].items():checked(Path(directory)/'initial',name,identity)
    rows=[]
    for pin,summary in zip(cfg['cases'],panel['cases']):
        require([pin[k]for k in ['id','dtype','graph']]==[summary[k]for k in ['id','dtype','graph']],'Retained case order differs')
        path=checked(directory,pin['case_report'],pin['report']);case=json.loads(path.read_text());folder=path.parent
        checked(folder,'files.json',pin['files_manifest']);files=json.loads((folder/'files.json').read_text())
        require(case['passed']==pin['strict_passed']and case['exit']==0 and case['ids_exact']and case['text_exact']and
                case['input_ids_exact']and case['eos']and not case['cap_hit'],'Old23 decoded/strict judgments changed')
        for name,identity in files.items():
            if canonical(name):checked(folder,name,identity)
        rows.append(dict(pin=pin,summary=summary,case=case,folder=folder,files=files))
    return cfg,panel,rows


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['analysis','continue'])
    parser.add_argument('--reviewed-source',required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    result=dict(completed=False,model_accessed=False)
    try:
        require(os.environ.get('KAGGLE_API_TOKEN')and os.environ.get('KAGGLE_ACCOUNT'),'Existing Kaggle secrets absent; stop before tools/fetch')
        result.update(source=source_guard(args.mode,args.reviewed_source),resources=resources(args.out,args.scratch),completed=True)
    except Exception as error:result['error']=str(error);raise
    finally:atomic(args.out/'preflight.json',result)
