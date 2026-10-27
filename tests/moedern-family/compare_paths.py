#!/usr/bin/env python3
"""Run compare.py for the default and the CRISPEMBED_MOEDERN_FAST path on the same
frozen panel and write one per-path error/timing report. Hosted only (Kaggle).

  compare_paths.py PROBE MOUNT OUT

Arms: default@2 threads (the reference path), fast@2 and fast@4. Every arm uses the
unchanged 0.002 / 0.99999 vector and 0.002 MaxSim gates vs the frozen originals.
The fast path differs from the default only by float summation order. Token IDs and
masks are the frozen ones, identical for every arm by construction. Exit 0 only if
every arm passes."""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = (('default', False, 2), ('fast', True, 2), ('fast', True, 4))


def worst(cases, part, key, pick):
    vals = [c[part][key] for c in cases.values() if part in c]
    return pick(vals) if vals else None


def main(probe, mount, out):
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(schema='crispembed-moedern-family-paths-v1', arms={})
    ok = True
    for name, fast, threads in ARMS:
        arm = '%s-t%d' % (name, threads)
        env = {k: v for k, v in os.environ.items() if k != 'CRISPEMBED_MOEDERN_FAST'}
        env['MOEDERN_PROBE_THREADS'] = str(threads)
        if fast: env['CRISPEMBED_MOEDERN_FAST'] = '1'
        r = subprocess.run([sys.executable, str(HERE/'compare.py'), str(probe), str(mount), str(out/arm)], env=env)
        path = out/arm/'parity.json'
        if not path.exists():
            summary['arms'][arm] = dict(passed=False, error='no parity.json (exit %d)' % r.returncode); ok = False; continue
        res = json.loads(path.read_text())
        summary['arms'][arm] = dict(
            path=res['path'], threads=res['threads'], passed=res['passed'], probe_exit=res['probe_exit'],
            total_encode_seconds=res['encode_seconds'].get('total'),
            all_max_absolute=worst(res['cases'], 'all', 'max_absolute', max),
            all_min_row_cosine=worst(res['cases'], 'all', 'min_row_cosine', min),
            selected_max_absolute=worst(res['cases'], 'selected', 'max_absolute', max),
            selected_min_row_cosine=worst(res['cases'], 'selected', 'min_row_cosine', min),
            maxsim_max_abs_diff=max((abs(s['native'] - s['original']) for s in res['scores']), default=None))
        ok = ok and res['passed']
    base = summary['arms'].get('default-t2', {}).get('total_encode_seconds')
    for a in summary['arms'].values():
        if base and a.get('total_encode_seconds'): a['speedup_vs_default_t2'] = base / a['total_encode_seconds']
    summary['passed'] = ok
    (out/'paths.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
