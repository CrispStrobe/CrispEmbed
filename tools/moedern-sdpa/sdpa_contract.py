"""Standard-library guards and independent metrics; no framework import."""
import hashlib
from contextlib import contextmanager
import json
import math
from pathlib import Path, PurePosixPath
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'docs/moedern-sdpa-reproducibility-v1.json'
LANE = ('docs/moedern-sdpa-reproducibility-v1.json', 'docs/moedern-sdpa-reproducibility.md',
        'tools/moedern-sdpa/sdpa_contract.py', 'tools/moedern-sdpa/diff_adapter.py',
        'tools/ci-heavy/moedern_sdpa_reproducibility.py', 'tests/test_moedern_sdpa_reproducibility.py',
        'tests/moedern-sdpa/CMakeLists.txt', 'tests/moedern-sdpa/diff.cpp',
        '.github/workflows/moedern-sdpa-reproducibility.yml', '.github/workflows/ovisocr2-metadata.yml')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def protocol():
    return json.loads(PROTOCOL.read_text())


def source_guard():
    cfg = protocol()
    for name, expected in cfg['protected'].items():
        old = subprocess.check_output(['git', 'show', cfg['base_source']+':'+name], cwd=ROOT)
        if sha(old) != expected or sha((ROOT/name).read_bytes()) != expected:
            raise ValueError('Historical source/scorer/strict guard changed: '+name)
    changed = subprocess.check_output(['git', 'diff', '--name-only', cfg['base_source'], 'HEAD', '--',
        'src', 'ggml', 'CMakeLists.txt', 'cmake', 'VERSION', 'python', 'flutter', 'crispembed',
        'crispembed-sys', 'wasm'], cwd=ROOT)
    if changed:
        raise ValueError('Production source changed outside operator diagnosis')
    if subprocess.check_output(['git', 'status', '--porcelain', '--', *LANE, *cfg['protected']], cwd=ROOT):
        raise ValueError('Dirty/untracked measured diagnostic source')
    return {'head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'files': {n: sha((ROOT/n).read_bytes()) for n in (*LANE, *cfg['protected'])}}


def indexed(report):
    result = {(r['case'], r['stage']): r for r in report['stage_arrays']}
    if len(result) != len(report['stage_arrays']):
        raise ValueError('Duplicate frozen stage')
    return result


def checked_arrays(root, report):
    """Authenticate all retained rows before importing Torch/NumPy."""
    indexed(report)
    for row in report['stage_arrays']:
        p = PurePosixPath(row['path'])
        if p.is_absolute() or '..' in p.parts or '\\' in row['path']:
            raise ValueError('Unsafe historical array path')
        if (not isinstance(row['shape'], list) or not row['shape'] or
                any(type(n) is not int or n < 1 for n in row['shape'])):
            raise ValueError('Frozen array dimensions differ')
        width = {'f32':4, 'i64':8, 'u8':1}[row['dtype']]
        path = Path(root)/p
        if path.is_symlink() or path.stat().st_size != row['bytes'] or row['bytes'] != width*math.prod(row['shape']):
            raise ValueError('Historical array extent/layout differs')
        if sha(path.read_bytes()) != row['sha256']:
            raise ValueError('Historical array byte hash differs')


def historical_contract(old, observed, cfg=None):
    cfg = protocol() if cfg is None else cfg
    if old['completed'] is not True or observed['completed'] is not False:
        raise ValueError('Original success and observer rejection must remain distinct')
    if old['github_sha'] != cfg['old_source'] or observed['github_sha'] != cfg['observer_source']:
        raise ValueError('Historical producer source differs')
    for key in ('source_contract_sha256', 'panel_sha256', 'versions', 'checkpoint_files',
                'inventory', 'loaded_tensor_sha256', 'pipeline'):
        if old[key] != observed[key]:
            raise ValueError('Historical inputs/loaded model/runtime differ: '+key)
    a, b = indexed(old), indexed(observed)
    for case, pin in cfg['cases'].items():
        for name, expected in pin['common'].items():
            if a[case,name] != expected or b[case,name] != expected:
                raise ValueError('Frozen shared SDPA input differs: '+case+'/'+name)
        for name, expected in pin['old_outputs'].items():
            if a[case,name] != expected:
                raise ValueError('Old SDPA output/gold changed')
        for name, expected in pin['observer_outputs'].items():
            if b[case,name] != expected:
                raise ValueError('Rejected observer output changed')
        if a[case,'actual.sdpa.0.output']['sha256'] == b[case,'actual.sdpa.0.output']['sha256']:
            raise ValueError('Expected historical SDPA difference missing')
    return {'common_inputs_exact': True, 'old_outputs_unchanged': True,
            'observer_continuity_remains_rejected': True,
            'old_array_count': len(a), 'observer_array_count': len(b),
            'old_arrays_with_changed_descriptors': sum(b.get(k) != v for k,v in a.items())}


def tensor_layout(tensor):
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
            'storage_offset': tensor.storage_offset(), 'contiguous': tensor.is_contiguous(),
            'dtype': str(tensor.dtype), 'device': str(tensor.device),
            'pointer_alignment_mod64': tensor.data_ptr()%64,
            'provenance': 'current original-source replay; historical strides were not captured'}


def kernel_evidence(backend, names):
    actual = sorted(set(n for n in names if 'scaled_dot_product' in n))
    if not actual:
        raise ValueError('Actual SDPA profiler kernel evidence absent')
    if backend == 'math' and (not any('attention_math' in n for n in actual) or any('flash_attention' in n for n in actual)):
        raise ValueError('Forced math backend silently selected another kernel')
    if backend == 'cpu-flash' and (not any('flash_attention_for_cpu' in n for n in actual) or any('attention_math' in n for n in actual)):
        raise ValueError('Forced CPU flash unavailable or fallback selected')
    return actual


def exact_control(plain, *others):
    if any(value != plain for value in others):
        raise ValueError('Ordinary/repeat/observer/profiled output bits differ')


@contextmanager
def replace_call(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


def metrics(left, right, dims):
    if len(left) != len(right) or len(left) != 4*math.prod(dims) or not dims:
        raise ValueError('Exact F32 metric dimensions/extent required')
    a = [v[0] for v in struct.iter_unpack('<f',left)]
    b = [v[0] for v in struct.iter_unpack('<f',right)]
    if not a or not all(math.isfinite(x) for x in a+b):
        raise ValueError('Nonfinite metric input')
    delta = [y-x for x,y in zip(a,b)]
    cosine = 1.
    for start in range(0,len(a),dims[-1]):
        x,y = a[start:start+dims[-1]],b[start:start+dims[-1]]
        aa,bb = math.fsum(v*v for v in x),math.fsum(v*v for v in y)
        value = 1. if aa == 0 and bb == 0 else 0. if aa == 0 or bb == 0 else math.fsum(u*v for u,v in zip(x,y))/math.sqrt(aa*bb)
        cosine = min(cosine,max(-1.,min(1.,value)))
    result = {'shape':list(dims),'elements':len(a),'max_absolute':max(map(abs,delta)),
        'mean_absolute':math.fsum(map(abs,delta))/len(a),'mean_error':math.fsum(delta)/len(a),
        'rms':math.sqrt(math.fsum(x*x for x in delta)/len(a)),'min_row_cosine':cosine,
        'original_l2':math.sqrt(math.fsum(x*x for x in a)),'native_l2':math.sqrt(math.fsum(x*x for x in b))}
    for name,data in [('original',a),('native',b)]:
        result.update({name+'_min':min(data),name+'_max':max(data),
                       name+'_abs_min':min(map(abs,data)),name+'_abs_max':max(map(abs,data))})
    result['passed'] = result['max_absolute'] <= .002 and cosine >= .99999
    return result
