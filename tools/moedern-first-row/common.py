"""Closed one-document diagnostic; importing this module requires only stdlib."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/moedern-first-row-stages-v1.json'
KEY='d:24726600'
CID='doc-24726600'

def require(ok,why):
    if not ok:raise ValueError(why)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        while block:=f.read(2**20):h.update(block)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('w')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def declaration():
    c=json.loads(PATH.read_text())
    require(c['schema']=='crispembed-moedern-first-row-stages-v1'and c['key']==KEY and c['gates']==dict(max_absolute=.002,min_row_cosine=.99999),'bounded document/gates differ')
    require(c['runtime_source']=='93d27614caba00a1b4d7aece6c78968f5cd9bb7f'and c['ggml']=='a06fc0bb46b73cc10011fc90ae4054aee8438aeb','runtime/backend differs')
    require(c['limits']==dict(original_targeted_forward_max=5,native_full_computations=14,native_plain_full=6,native_observed_full=8,
        native_encode_attempts=21,native_pre_math_observer_abort=1,native_frontend_rejections=6,native_init_rejections=6),'forward budgets differ')
    require(c['original_full_corpus_recaptured']is False and c['new_gold']is False and c['publication_qualified']is False,'diagnostic scope differs')
    publisher=c['publisher_router_source']
    original=json.loads((ROOT/'docs/moedern-colbert-source-contract.json').read_text())['variants']['1.5M']['original']
    pin=original['files'][publisher['path']]
    require(publisher['revision']==original['revision']and publisher['repository']==original['repository']and
        publisher['sha256']==pin['sha256']and publisher['bytes']==pin['bytes']and publisher['out_of_place']is True and
        publisher['normalization_source']=='tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-9)'and
        publisher['expected_dispatch']=='aten.div.Tensor'and publisher['dense_work']==628736 and publisher['dense_work']<publisher['dense_threshold']and
        publisher['input_tokens']<publisher['chunk_tokens'],'pinned publisher router source facts differ')
    require(c['sdpa_dispatch_metadata']['maximum_entries']==256 and c['sdpa_dispatch_metadata']['repeat_exact_required']is True and
        c['sdpa_dispatch_metadata']['ordinary_backend_identity_claimed']is False and c['sdpa_dispatch_metadata']['historical_binary_backend_identity_claimed']is False and
        c['sdpa_dispatch_metadata']['backend_forced']is False,'SDPA dispatch metadata scope differs')
    protected=set(git('ls-tree','-r','--name-only',c['base'],'src','tools/moedern-colbert','tools/moedern-native-v2','tools/moedern-three-doc',
                     'tools/moedern-scifact','tests/moedern-text','tests/moedern-three-doc').decode().splitlines())
    protected|={'CMakeLists.txt','.gitmodules','VERSION','tools/ci_process.py','docs/moedern-colbert-source-contract.json',
        'docs/moedern-colbert-native-contract-v2.json','docs/moedern-three-doc-attribution-v1.json','docs/moedern-scifact-parity-v1.json',
        'tools/moedern_scifact.py','tools/ci-heavy/moedern_scifact.py','tests/results/moedern-colbert/scifact-original-1p5m.json',
        'tests/results/moedern-colbert/scifact-native-1p5m-v3-native.json','tests/results/moedern-colbert/scifact-native-1p5m-v3-review.json',
        'tests/results/moedern-colbert/publication-1p5m.json'}
    require(protected==set(c['protected_sha256']),'complete exact protected closure required')
    owned={'tools/moedern-first-row/'+n+'.py'for n in ('common','prepare','original','compare','kernel')}
    owned|={'tools/ci-heavy/moedern_first_row_v1.py','tests/moedern-first-row/CMakeLists.txt','tests/moedern-first-row/probe.cpp',
        'tests/test_moedern_first_row_v1.py','.github/workflows/moedern-first-row-stages-v1.yml','docs/moedern-first-row-stages-v1.md'}
    require(set(c['owned_sha256'])==owned,'exact eleven-file diagnostic closure required')
    return c

def verify(source):
    c=declaration();require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD').decode().strip()==source,'exact reviewed source required')
    require(git('ls-tree','-r','HEAD','src')==git('ls-tree','-r',c['runtime_source'],'src')and
            git('show','HEAD:CMakeLists.txt')==git('show',c['runtime_source']+':CMakeLists.txt'),'production runtime/build changed')
    rel=str(PATH.relative_to(ROOT));require(git('show','HEAD:'+rel)==PATH.read_bytes(),'dirty protocol')
    allowed=set(c['owned_sha256'])|{rel,'PLAN.md'}
    require(set(git('diff','--name-only',c['base'],'HEAD').decode().splitlines())<=allowed,'undeclared source delta')
    for group in ('owned_sha256','protected_sha256'):
        for name,pin in c[group].items():
            p=ROOT/name;require(p.is_file()and not p.is_symlink()and sha(p)==pin and git('show','HEAD:'+name)==p.read_bytes(),'source identity differs: '+name)
            if group=='protected_sha256':require(git('show',c['base']+':'+name)==p.read_bytes(),'protected source changed: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip()==c['ggml'],'backend gitlink differs')
    return c

def hosted(source):
    require(git('rev-parse','HEAD').decode().strip()==source,'submitted source differs')
    if os.environ.get('GITHUB_ACTIONS')=='true':
        require(os.environ.get('GITHUB_SHA')==source and os.environ.get('GITHUB_RUN_ID','').isdigit(),'GHA source/run identity absent')
        return 'GitHub Actions'
    require(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and
        os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')and
        os.environ.get('CRISPEMBED_CAPTURE_SOURCE')==source,'genuine hosted CPU environment required')
    return 'Kaggle CPU; external exact-version submitted-script/status/output proof required'

def authenticated_input(root,c):
    p=root/'input-packet.json';prepared=json.loads((root/'prepare.json').read_text());packet=json.loads(p.read_text())
    require(prepared['completed']is True and prepared['input_packet_sha256']==sha(p)and prepared['protocol_sha256']==sha(PATH),'authenticated input preparation required')
    require(packet['schema']=='crispembed-moedern-first-row-input-v1'and packet['key']==KEY and packet['case']['key']==KEY,'only first document admitted')
    require(prepared['source']==git('rev-parse','HEAD').decode().strip(),'prepared source identity differs')
    # Re-authenticate whole retained ZIPs on each phase, not merely an editable prepare.json.
    for name in ('small','raw'):
        q=root/(name+'.zip');pin=c['retained'][name]
        require(q.is_file()and not q.is_symlink()and q.stat().st_size==pin['bytes']and sha(q)==pin['sha256'],'retained ZIP changed after preparation')
    with zipfile.ZipFile(root/'raw.zip')as z:
        retained=json.loads(z.read('raw/retained/three-doc-input/input-packet.json'))
    exact=next(r for r in retained['cases']if r['key']==KEY)
    require(packet['case']==exact and retained['keys']==['d:24726600','d:24928817','d:5849439'],'target input differs from immutable retained packet')
    case=packet['case'];require(case['id']==CID and len(case['ids'])==307 and case['ids'][58]==29084 and sum(case['scoring_mask'])==278,'frozen row/token identities differ')
    for side in ('original','native'):
        for kind in ('all','selected'):
            pin=case['anchors'][side][kind];q=root/pin['path']
            require(q.is_file()and not q.is_symlink()and q.stat().st_size==pin['bytes']and sha(q)==pin['sha256'],'frozen anchor changed')
    require(hashlib.sha256(case['text'].encode()).hexdigest()==case['text_sha256'],'raw document changed')
    return case

def actual_mapping(ref):
    mapping=ref.stage_map()
    for i in range(2):
        p=f'layer.{i}.ffn.';del mapping[p+'selection_margin']
        for key,op in [('router_probabilities','softmax'),('topk_indices','topk.indices'),('topk_weights','div')]:
            mapping[p+key]=f'actual.layers.{i}.ffn.{op}'
    require(len(mapping)==46,'44 F32/two integer mapped stage declaration differs')
    return mapping
