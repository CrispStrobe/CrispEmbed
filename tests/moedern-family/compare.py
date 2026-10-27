#!/usr/bin/env python3
"""Run the family probe on a frozen original panel and compare under the unchanged
0.002 / 0.99999 vector and 0.002 MaxSim gates. Hosted only (Kaggle or Actions)."""
import glob, json, math, os, struct, subprocess, sys
from pathlib import Path
GATES = dict(max_absolute=0.002, min_row_cosine=0.99999, maxsim_max_absolute=0.002)
probe, mount, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
root = Path(glob.glob(str(mount/'**/review/original.json'), recursive=True)[0]).parent.parent
report = json.loads((root/'review/original.json').read_text())
out.mkdir(parents=True, exist_ok=True); (out/'native').mkdir(exist_ok=True)
def flag(v): return [int(x) for x in (json.loads(v.replace('True', 'true').replace('False', 'false')) if isinstance(v, str) else v)]
(out/'cases.tsv').write_text(''.join(f"{c['id']}\t{c['role']}\t{','.join(map(str, c['token_ids']))}\t{','.join(map(str, c['supplied_attention_mask']))}\n" for c in report['cases']))
env = dict(os.environ, CRISPEMBED_MOEDERN_FAMILY='1', CRISPEMBED_MOEDERN_COLBERT='1')
r = subprocess.run([str(probe), str(root/'model-F32.gguf'), str(out/'cases.tsv'), str(out/'native')], env=env, capture_output=True, text=True)
print(r.stdout, r.stderr)
def rows(path, w=128):
    b = Path(path).read_bytes(); n = len(b)//4; v = struct.unpack(f'<{n}f', b); return [v[i:i+w] for i in range(0, n, w)]
def metrics(a, b):
    worst, cos = 0.0, 1.0
    for x, y in zip(a, b):
        worst = max(worst, max(abs(p-q) for p, q in zip(x, y)))
        nx, ny = math.sqrt(sum(p*p for p in x)), math.sqrt(sum(q*q for q in y))
        cos = min(cos, sum(p*q for p, q in zip(x, y))/(nx*ny))
    return dict(max_absolute=worst, min_row_cosine=cos, rows=len(a), shape_equal=len(a) == len(b),
                passed=len(a) == len(b) and worst <= GATES['max_absolute'] and cos >= GATES['min_row_cosine'])
arr = {(s['case'], s['stage']): root/s['path'] for s in report['stage_arrays']}
result = dict(schema='crispembed-moedern-family-parity-v1', variant=report['variant'], revision=report['revision'],
              probe_exit=r.returncode, gates=GATES, cases={}, scores=[])
selected = {}
for c in report['cases']:
    f = out/'native'/(c['id']+'.f32')
    if not f.exists(): result['cases'][c['id']] = dict(passed=False, error='no native output'); continue
    native = rows(f); mask = flag(c['scoring_mask'])
    sel = [v for v, m in zip(native, mask) if m]; selected[c['id']] = sel
    result['cases'][c['id']] = dict(all=metrics(native, rows(arr[(c['id'], 'ordinary.features.token_embeddings')])),
                                    selected=metrics(sel, rows(arr[(c['id'], 'ordinary.selected')])))
    result['cases'][c['id']]['passed'] = result['cases'][c['id']]['all']['passed'] and result['cases'][c['id']]['selected']['passed']
for s in report['scores']:
    q, d = selected.get(s['query']), selected.get(s['document'])
    native = sum(max(sum(a*b for a, b in zip(x, y)) for y in d) for x in q) if q and d else float('nan')
    result['scores'].append(dict(pair=s['query']+'|'+s['document'], native=native, original=s['original_maxsim'],
                                 passed=abs(native-s['original_maxsim']) <= GATES['maxsim_max_absolute']))
result['passed'] = r.returncode == 0 and all(c['passed'] for c in result['cases'].values()) and all(s['passed'] for s in result['scores'])
(out/'parity.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('variant', 'passed', 'scores')}, indent=1))
sys.exit(0 if result['passed'] else 1)
