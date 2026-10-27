"""Actual Python consumer through CrispMoEdern; same hex TSV protocol as the C consumer."""
import json
import sys
sys.path.insert(0, sys.argv[5])
from crispembed import CrispMoEdern
model, lib, cases, pairs = sys.argv[1:5]
def text(h): return '' if h == '-' else bytes.fromhex(h).decode('utf-8')
with CrispMoEdern(model, lib_path=lib, auto_download=False, offline=True) as m:
    for bad in (lambda: m.encode('x', 'other'), lambda: m.encode('x\0y', 'query')):
        try: bad()
        except ValueError: pass
        else: raise SystemExit('invalid input accepted')
    for line in open(cases):
        cid, role, *h = line.rstrip('\n').split('\t')
        r = m.encode(text(h[0] if h else '-'), role)
        print(json.dumps(dict(kind='case', id=cid, **{k: r[k] for k in ('role', 'ids', 'supplied_mask', 'effective_mask', 'scoring_mask', 'expansion_positions', 'all_vectors', 'selected_vectors')})))
    for line in open(pairs):
        pid, q, d = line.rstrip('\n').split('\t')
        print(json.dumps(dict(kind='score', id=pid, score=m.score(text(q), text(d)))))
