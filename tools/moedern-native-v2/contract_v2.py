"""Versioned dual-reference contract; historical completion guards stay untouched."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path,PurePosixPath
import subprocess
import struct

ROOT=Path(__file__).resolve().parents[2]
PATH=ROOT/'docs/moedern-colbert-native-contract-v2.json'
LANE=('src/moedern_colbert.cpp','src/moedern_colbert.h','tests/moedern-native-v2/CMakeLists.txt',
 'tests/moedern-native-v2/probe.cpp','tests/moedern-native-v2/controls.cpp','tests/moedern-native-v2/diff.cpp',
 'tools/moedern-native-v2/contract_v2.py','tools/moedern-native-v2/replay.py','tools/moedern-native-v2/diff_adapter.py',
 'tools/ci-heavy/moedern_native_v2.py','tests/test_moedern_native_v2.py','docs/moedern-colbert-native-contract-v2.json',
 'docs/moedern-colbert-native-v2.md','.github/workflows/moedern-native-v2.yml','.github/workflows/ovisocr2-metadata.yml')
SHARED=('src','ggml','cmake','CMakeLists.txt','VERSION','README.md','LICENSE','Cargo.lock','python','flutter',
 'crispembed','crispembed-sys','wasm','examples','tools/quantize.cpp','build-wasm.sh','build-embed-wasm.sh',
 'scripts/check-cpu-baseline.py','scripts/check-bundled-deps.py','.github/workflows/release.yml',
 '.github/workflows/release-wasm.yml','tools/successor_asset_manifest.py','.gitattributes')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()


def protocol():return json.loads(PATH.read_text())


def git_blob(source,path):return subprocess.check_output(['git','show',source+':'+path],cwd=ROOT)


def historical(path):
    cfg=protocol();data=git_blob(cfg['primary']['source'],path)
    if hashlib.sha256(data).hexdigest()!=cfg['historical_sources'][path]:raise ValueError('Historical schema/input source differs')
    return data


def read_contract():return json.loads(historical('docs/moedern-colbert-source-contract.json'))


def panel():return json.loads(historical('tests/fixtures/moedern-colbert/panel-v1.json'))


def schema_module(directory):
    """Materialize exact old stdlib schema, not workers/guards or original inference."""
    for path in protocol()['historical_sources']:
        target=directory/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(historical(path))
    path=directory/'tools/moedern-colbert/contract.py'
    spec=importlib.util.spec_from_file_location('moedern_authenticated_original_schema',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source():
    cfg=protocol()
    changed=subprocess.check_output(['git','diff','--name-only',cfg['base_source'],'HEAD','--',*SHARED],cwd=ROOT,text=True).splitlines()
    if set(changed)!={'src/moedern_colbert.cpp','src/moedern_colbert.h'}:raise ValueError('Shared runtime/source changed outside new1.5M files')
    if subprocess.check_output(['git','status','--porcelain','--',*SHARED,*LANE],cwd=ROOT):raise ValueError('Dirty/untracked measured source')
    if subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=ROOT,text=True).strip()!=cfg['ggml']:raise ValueError('Explicit v2 ggmla06 differs')
    for path,pin in cfg['copied_native_sources'].items():
        target=path.replace('tests/moedern-native/','tests/moedern-native-v2/')
        if digest(ROOT/target)!=pin or hashlib.sha256(git_blob(cfg['native_blueprint'],path)).hexdigest()!=pin:
            raise ValueError('Native79 math/probe/control bytes changed: '+path)
    return dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),base=cfg['base_source'],ggml=cfg['ggml'],
        linked_src_tree=subprocess.check_output(['git','rev-parse','HEAD:src'],cwd=ROOT,text=True).strip(),files={p:digest(ROOT/p) for p in LANE},
        protected_git_objects={p:subprocess.check_output(['git','rev-parse','HEAD:'+p],cwd=ROOT,text=True).strip() for p in SHARED})


def indexed(report):
    rows={(r['case'],r['stage']):r for r in report['stage_arrays']}
    if len(rows)!=len(report['stage_arrays']):raise ValueError('Duplicate reference stage')
    return rows


def check_reports(primary,supplemental):
    cfg=protocol();owned=panel();profile=read_contract()
    if (primary.get('completed')is not True or supplemental.get('completed')is not False or
            primary.get('github_sha')!=cfg['primary']['source'] or supplemental.get('github_sha')!=cfg['supplemental']['source'] or
            cfg['supplemental']['accepted_original_continuity']is not False or cfg['primary_replaced']is not False or
            cfg['supplemental']['reviewed_neutral_measurements']is not True or cfg['diagnosis']['reviewed']is not True):
        raise ValueError('Primary success and reviewed supplemental rejection must remain distinct')
    for key in ('schema','variant','revision','source_contract_sha256','panel_sha256','versions','pipeline',
                'tokenizer_descriptor','checkpoint_files','inventory','loaded_tensor_sha256'):
        if primary[key]!=supplemental[key]:raise ValueError('Original source/model/runtime differs: '+key)
    if (primary['source_contract_sha256']!=cfg['historical_sources']['docs/moedern-colbert-source-contract.json'] or
            primary['panel_sha256']!=cfg['historical_sources']['tests/fixtures/moedern-colbert/panel-v1.json'] or
            primary['variant']!='1.5M' or primary['revision']!=profile['variants']['1.5M']['original']['revision']):
        raise ValueError('Frozen original profile differs')
    for kind,report,pin in (('primary',primary,cfg['primary']),('supplemental',supplemental,cfg['supplemental'])):
        if report['driver_sources']!=cfg['original_driver_sources'][kind]:raise ValueError('Complete original driver-source coverage differs')
        for path,value in report['driver_sources'].items():
            if hashlib.sha256(git_blob(pin['source'],path)).hexdigest()!=value:raise ValueError('Original driver-source identity differs')
        if len(report['cases'])!=10:raise ValueError('All10 original cases required')
        versions={'torch':'2.8.0','transformers':'5.3.0','sentence-transformers':'6.1.0','numpy':'2.2.6'}
        if any(report['versions'].get(k,'').split('+')[0]!=v for k,v in versions.items()):raise ValueError('Original package profile differs')
        for expected,actual in zip(owned['cases'],report['cases']):
            if any(actual[k]!=expected[k] for k in ('id','role','text')) or not actual['ordinary_observer_bit_exact'] or not actual['public_slice_bit_exact']:
                raise ValueError('Original role/input/neutrality differs')
            ids=actual['token_ids']
            if (not 3<=len(ids)<=512 or ids[:2]!=[2,30000 if actual['role']=='query' else 30001] or
                    any(type(x)is not int or not 0<=x<30002 for x in ids) or
                    any(len(actual[k])!=len(ids) for k in ('scoring_mask','supplied_attention_mask','effective_attention_mask')) or
                    any(x not in (0,1,False,True) for k in ('scoring_mask','supplied_attention_mask','effective_attention_mask') for x in actual[k])):
                raise ValueError('Original ID/three-mask contract differs')
        if [(s['query'],s['document']) for s in report['scores']]!=[tuple(p) for p in owned['score_pairs']]:raise ValueError('All3scorepairs required')
    def cases(r):return [{k:v for k,v in x.items() if k!='seconds_including_observer'} for x in r['cases']]
    if cases(primary)!=cases(supplemental) or cases(primary)!=cfg['case_contracts']:raise ValueError('Original IDs/masks/ordinary shape differ')
    old,new=indexed(primary),indexed(supplemental)
    extra=set(new)-set(old)
    if len(old)!=250 or len(new)!=271 or not set(old)<=set(new) or len(extra)!=21:
        raise ValueError('Exact250primary+21supplemental coverage required')
    expected=cfg['supplemental_extra_descriptors']
    if [r for r in supplemental['stage_arrays'] if (r['case'],r['stage']) in extra]!=expected:
        raise ValueError('Exact reviewed21supplemental descriptors differ')
    for row in expected:
        if row['dtype']=='f32':continue
        case=next(x for x in primary['cases'] if x['id']==row['case'])
        if row['stage'].endswith('.input_ids'):values=case['token_ids']
        elif row['stage'].endswith('.query_expansion_positions'):values=[int(not x) for x in case['supplied_attention_mask']]
        elif row['stage'].startswith('pipeline.1.'):values=case['supplied_attention_mask']
        else:values=case['scoring_mask']
        data=struct.pack('<'+'q'*len(values),*values) if row['dtype']=='i64' else bytes(values)
        if row['shape']!=[1,len(values)] or hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Supplemental IDs/masks/expansion semantics differ')
    for key,row in old.items():
        other=new[key]
        if any(row[k]!=other[k] for k in ('case','stage','dtype','shape','bytes','path')):
            raise ValueError('Shared original array layout/type differs')
    if all(old[k]['sha256']==new[k]['sha256'] for k in old) or primary['scores']==supplemental['scores']:
        raise ValueError('Retained rejected continuity evidence was lost')
    return dict(primary_arrays=250,supplemental_only_arrays=21,shared_changed_sha=sum(old[k]['sha256']!=new[k]['sha256'] for k in old),
                accepted_original_continuity=False,primary_replaced=False)


def verify_arrays(directory,report):
    indexed(report)
    for row in report['stage_arrays']:
        p=PurePosixPath(row['path'])
        if p.is_absolute() or '..' in p.parts or '\\' in str(p):raise ValueError('Unsafe original array path')
        width={'f32':4,'i64':8,'u8':1}[row['dtype']]
        if (not row['shape'] or any(type(n)is not int or not 1<=n<=512 for n in row['shape']) or
                row['bytes']!=width*math.prod(row['shape'])):raise ValueError('Original array shape/extent differs')
        path=directory/p
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Original array bytes differ')


def primary_map():
    from replay import stage_map
    return {k:v for k,v in stage_map().items() if k not in ('dense','normalized')}


def native_names():
    result={'input_ids','supplied_mask','effective_mask','embedding.lookup','embedding.projection','embedding.norm','final_norm','dense','normalized'}
    suffix=('attn.norm','attn.q','attn.k','attn.v','attn.q_rope','attn.k_rope_repeat','attn.v_repeat',
        'attn.probabilities','attn.sdpa','attn.output_projection','attn.residual','ffn.norm','ffn.router',
        'ffn.router_probabilities','ffn.topk_indices','ffn.topk_weights','ffn.selection_margin','ffn.routed',
        'ffn.shared_up','ffn.shared_gate','ffn.shared_down','ffn.output','output')
    return result|{f'layer.{i}.'+s for i in range(2) for s in suffix}


def coverage(primary):
    full=panel()['full_capture'];mapping=primary_map();mapped=set()
    for row in primary['stage_arrays']:
        if row['stage'] in ('ordinary.features.token_embeddings','ordinary.selected') or (row['case'] in full and row['stage'] in mapping.values()):
            mapped.add((row['case'],row['stage']))
    retained=[]
    for row in primary['stage_arrays']:
        if (row['case'],row['stage']) not in mapped:
            retained.append(dict(case=row['case'],stage=row['stage'],dtype=row['dtype'],shape=row['shape'],
                reason='Historical module input/activation/alias or feature bookkeeping not emitted by unchanged79 native probe; retained and compared original-to-original, never claimed native parity.'))
    return dict(retained_primary_arrays=250,mapped_primary_arrays=len(mapped),retained_not_mapped=retained,
        primary_native_map=dict(sorted(mapping.items())),required_native_stages=sorted(native_names()),
        native_without_reference={f'layer.{i}.'+s:'No independently captured original attention probabilities or routed-only sum; finite native observation retained without parity claim.' for i in range(2) for s in ('attn.probabilities','ffn.routed')},
        supplemental_native_map={'pipeline.1.output#0.token_embeddings':'dense','pipeline.2.output#0.token_embeddings':'dense','pipeline.3.output#0.token_embeddings':'normalized'},
        supplemental_integer_policy='Pipeline1 supplied mask; pipeline2/3 scoring mask; exact IDs; query expansion from logical-not supplied mask. All15 metadata leaves checked separately; no float casts.',
        non_full_cases='Only original ordinary all/selected vector targets and exact inputs/masks exist; unreferenced native intermediates retained but never claimed parity.')
