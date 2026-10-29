"""New continuation closure. Frozen transfer, panel and runtime remain untouched."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/unlimited-reused-q4-first-panel-v1.json'
REVIEW=ROOT/'tests/results/unlimited-ocr2/retained-v3-structural-review.json'
sys.path[:0]=[str(ROOT/'tools/unlimited-ocr2-q4k'),str(ROOT/'tools')]
import contract as frozen
require,digest=frozen.require,frozen.digest

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('w')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def stage_order():
    # Observed hn/IDs/gate/up precede their layer output in the original graph.
    result=['layer_0_chunk_0']
    for layer in range(1,12):
        prefix='layer_'+str(layer)+'_chunk_0'
        result.extend(prefix+'_'+part for part in ('hn','ids','gate','up'));result.append(prefix)
    return result+['final_norm','logits']

def declaration():
    c=json.loads(PATH.read_text());r=json.loads(REVIEW.read_text())
    require(c['schema']=='crispembed-unlimited-reused-q4-first-panel-v1'and c['runtime_source']==frozen.BASE and
        c['ggml']=='a06fc0bb46b73cc10011fc90ae4054aee8438aeb'and c['recipe']=='last1','exact existing first-candidate/runtime required')
    require(c['gates']==dict(max_absolute=.002,min_row_cosine=.99999,aligned_box_max_absolute=3)and c['cases']==frozen.CASES,'frozen gates/cases differ')
    require(c['counts']==dict(original_forwards=0,quantizer_runs=0,f16_forwards=16,q4_forwards=12,projection_executions=16,
        panel_files=540,additional_png_files=4,all_authenticated_files=544),'declared call/input counts differ')
    require(c['control']==dict(bytes=frozen.F16_BYTES,sha256=frozen.F16_SHA,url=frozen.F16_URL)and c['candidate']==r['model']and
        c['candidate']['sha256']=='8b9d8f28e225a229107da1c2dd043452d5c30a3f568337eff1741181a5da4f37','exact stored control/candidate required')
    require(digest(REVIEW)==c['structural_review_sha256']and r['completed']is True and r['authenticated_files']==540 and
        r['authenticated_bytes']==7901314457 and r['native_forwards']==r['quantizer_runs']==0,'reviewed structural evidence differs')
    require(c['producer_panel_sha256']==r['producer_panel_sha256']and c['structural_run']==r['run']and c['structural_source']==r['source']and
        c['structural_small_artifact']==r['small_artifact']and c['structural_small_members']==r['small_members'],'exact reviewed structural service/member pins required')
    _,images=frozen.references()
    require(c['images']=={case:dict(member='unlimited-q4k/pages/'+case+'.png',sha256=images[case],maximum_bytes=2*2**20)for case in frozen.CASES},'only four original frozen PNG identities admitted')
    require(c['publication_qualified']is False and c['original_intermediate_gold']is False and c['original_rerun']is False and c['default_enabled']is False,'diagnostic scope differs')
    require(c['threads']==4 and c['max_tokens']==1024 and c['chunk']==2048 and c['workflow_minutes']==360 and c['panel_seconds']==18000,'predeclared profile/bounds differ')
    require(c['f16_admission']==dict(cross_old_float_stages=47,additional_assembled_vision_stages=1,cross_old_integer_stages=11,
        cross_old_floating_sha_admission=False,cross_old_vision_sha_admission=False,current_stage_repeat_byte_exact=True,
        current_vision_repeat_byte_exact=True,cross_old_decoded_bytes_exact=True),'cross-host/current-host baseline scope differs')
    required=set(git('ls-tree','-r','--name-only',c['base'],*frozen.PREFIXES,'tools/unlimited-ocr2-q4k','tests/unlimited-q4k',
        'tools/kaggle/unlimited-ocr2-q4k','tools/unlimited-ocr2','tests/unlimited-real-decode').decode().splitlines())
    required.discard('ggml')
    required|={'docs/unlimited-ocr2-q4k-v1.json','docs/unlimited-ocr2-q4k-v1.md','docs/unlimited-retained-gha-fetch-v1.json',
        'docs/unlimited-retained-gha-fetch-v1.md','docs/unlimited-retained-v3-structural-result.md',str(REVIEW.relative_to(ROOT)),
        'tests/results/unlimited-ocr2/mixed-q4-v3-panel.json','tests/results/unlimited-ocr2/publication.json',
        'tools/unlimited-ocr2/gen_text_fixture.py','tests/test_unlimited_ocr2_q4k.py',
        'tests/test_unlimited_retained_fetch.py','tests/test_unlimited_real_decode.py','tools/kaggle/ocr-real-scans/receipt.py','docs/ocr-real-scans-v1.json'}
    require(set(c['protected_sha256'])==required,'complete inherited source/evidence closure required')
    owned={'tools/unlimited-reused-q4-v1/'+n+'.py'for n in ('lane','fetch','replay','run','process')}
    owned|={'.github/workflows/unlimited-reused-q4-first-panel-v1.yml','tests/test_unlimited_reused_q4_v1.py','docs/unlimited-reused-q4-first-panel-v1.md'}
    require(set(c['owned_sha256'])==owned,'exact eight-file continuation closure required')
    require((ROOT/'tools/unlimited-reused-q4-v1/process.py').read_bytes()==git('show',c['process_helper']['source']+':tools/ci_process.py'),'reviewed process helper must remain byte exact')
    return c

def guard(source,hosted=False):
    c=declaration();require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD').decode().strip()==source,'exact reviewed source required')
    rel=str(PATH.relative_to(ROOT));require(git('show','HEAD:'+rel)==PATH.read_bytes(),'dirty protocol')
    require(set(git('diff','--name-only',c['base'],'HEAD').decode().splitlines())<=set(c['owned_sha256'])|{rel},'undeclared continuation delta')
    require(git('ls-tree','-r','HEAD','--',*frozen.PREFIXES)==git('ls-tree','-r',frozen.BASE,'--',*frozen.PREFIXES),'frozen runtime/blueprint/build changed')
    for group in ('protected_sha256','owned_sha256'):
        for name,pin in c[group].items():
            p=ROOT/name;require(p.is_file()and not p.is_symlink()and digest(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'source identity differs: '+name)
            if group=='protected_sha256':require(git('show',c['base']+':'+name)==p.read_bytes(),'inherited source/evidence changed: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==c['ggml'],'backend gitlink differs')
    if hosted:
        require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and os.environ.get('GITHUB_SHA')==source,'genuine exact-source GHA required')
        require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==c['ggml']and
            not subprocess.check_output(['git','-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no']),'actual backend differs/dirty')
    return c

def resources(parent,initial=False):
    import shutil
    ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    free=shutil.disk_usage(parent).free
    require(ram>=14*2**30 and free>=(48 if initial else 8)*2**30 and os.cpu_count()>=2,'14GiB RAM/two CPUs and predeclared disk reserve required')
    return dict(available_ram_bytes=ram,free_disk_bytes=free,cpu_count=os.cpu_count(),initial=initial)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();c=guard(a.source)
    print(json.dumps(dict(protected=len(c['protected_sha256']),owned=len(c['owned_sha256']),protocol_sha256=digest(PATH))))
