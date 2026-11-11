#!/usr/bin/env python3
"""Fresh replay of the WASM Q8 rounding A/B (run 40391544739) on a later source.

The frozen guards in tools/wasm_q8_rounding_ab_v1.py pin the exact candidate
encoder bytes and ggml gitlink, so the unchanged driver refuses any later
source. This wrapper changes only those identity assertions: it records the
current encoder/ggml identity instead of requiring the frozen one, then runs
the unchanged driver phases (same panel, same gates, same comparisons, same
immutable producer packages, trace and fixture). The final `replay-compare`
phase requires every candidate vector, stage array and token stream to be
bit-identical to the frozen producer report; anything else is reported as a
diff, never re-pinned. Guarded files are not modified.

    frozen_guard_replay_wasm_q8.py <driver-phase> --output DIR
    frozen_guard_replay_wasm_q8.py replay-compare --output DIR --frozen FROZEN_AB_JSON
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
import wasm_q8_rounding_ab_v1 as contract  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('ab_driver_v1', ROOT / 'tools/ci-heavy/wasm_q8_rounding_ab_v1.py')
driver = importlib.util.module_from_spec(_spec)
sys.modules['ab_driver_v1'] = driver
_spec.loader.exec_module(driver)

REPLAY_WORKFLOW = '.github/workflows/frozen-guard-replay-wasm-q8.yml'
FROZEN_RUN = 40391544739


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def current_ggml():
    return git('rev-parse', 'HEAD:ggml')


def replay_source_guard(root, runtime_paths):
    """Record (not require) the encoder/ggml identity; keep the cleanliness checks."""
    root = Path(root)
    ggml = root / 'ggml'
    head = current_ggml()
    if not (ggml / '.git').exists() or git('rev-parse', 'HEAD', cwd=ggml) != head:
        raise ValueError('ggml submodule not initialized at the recorded gitlink')
    if git('status', '--porcelain', '--', *runtime_paths, cwd=root) or git('status', '--porcelain', cwd=ggml):
        raise ValueError('Dirty production source rejected')
    declaration = json.loads((root / 'docs/wasm-q8-rounding-candidate.json').read_text())
    spec = json.loads((root / 'docs/wasm-q8-rounding-ab-v1.json').read_text())
    if contract.digest((root / 'docs/wasm-q8-rounding-candidate.json').read_bytes()) != spec['candidate_declaration_sha256']:
        raise ValueError('Candidate declaration bytes differ')
    files = {name: {'frozen_candidate_sha256': row['candidate_sha256'],
                    'current_sha256': contract.digest((ggml / name).read_bytes()) if (ggml / name).exists() else None}
             for name, row in declaration['files'].items()}
    return {'replay': True, 'base': contract.BASE, 'producer': contract.PRODUCER,
            'frozen_encoder_sha256': contract.ENCODER_SHA,
            'encoder_sha256': contract.digest((root / 'src/crispembed.cpp').read_bytes()),
            'frozen_ggml': contract.CANDIDATE_GGML, 'ggml': head,
            'ggml_files_changed_since_frozen_candidate': git('diff', '--name-only', contract.CANDIDATE_GGML, head, cwd=ggml).splitlines(),
            'candidate_dependency_files': files,
            'production_files_changed_since_base': git('diff', '--name-only', contract.BASE, 'HEAD', '--', 'src', 'wasm', cwd=root).splitlines()}


def replay_operator_barrier(out):
    """Same checks as driver.operator_barrier, but for this replay workflow."""
    run = os.environ['GITHUB_RUN_ID']; current = os.environ['GITHUB_SHA']
    meta = driver.api('actions/runs/' + run)
    if meta['head_sha'] != current or meta['path'] != REPLAY_WORKFLOW:
        raise ValueError('Current operator parent identity differs')
    jobs = driver.api('actions/runs/' + run + '/jobs?per_page=100')
    selected = [j for j in jobs['jobs'] if j['name'] == 'operator-controls / operators' or j['name'] == 'operators']
    if jobs['total_count'] > 100 or len(selected) != 1 or selected[0]['status'] != 'completed' or selected[0]['conclusion'] != 'success':
        raise ValueError('Successful same-source operator job required before model access')
    identity = driver.artifact(run, 'wasm-rounding-operator-controls-' + run, out / 'operator-controls.zip', 4 * 1024**2)
    files = driver.zip_files(out / 'operator-controls.zip', 16 * 1024**2)

    def one(basename):
        matches = [data for name, data in files.items() if Path(name).name == basename]
        if len(matches) != 1:
            raise ValueError('Missing/ambiguous operator receipt: ' + basename)
        return json.loads(matches[0])
    declaration = json.loads((ROOT / 'docs/wasm-q8-rounding-candidate.json').read_text())
    proof = contract.operator_receipt(one('source.json'), one('controls.json'), current, declaration)
    proof.update(run=int(run), job=selected[0]['id'], artifact=identity,
                 members={name: {'bytes': len(data), 'sha256': contract.digest(data)} for name, data in files.items()})
    driver.save(out / 'operator-proof.json', proof)
    return proof


def local_operator_barrier(directory):
    """Kaggle fallback: same operator_receipt checks on receipts produced earlier in the same session.

    Deviation from the hosted lane (recorded in the proof): there is no GitHub run, so the
    same-run job/artifact service identity cannot be checked; the receipts come from the
    operator controls executed in this session on the same checkout before model access.
    """
    def barrier(out):
        directory_path = Path(directory)
        files = {p.name: p.read_bytes() for p in directory_path.iterdir() if p.is_file()}
        declaration = json.loads((ROOT / 'docs/wasm-q8-rounding-candidate.json').read_text())
        proof = contract.operator_receipt(json.loads(files['source.json']), json.loads(files['controls.json']),
                                          os.environ['GITHUB_SHA'], declaration)
        proof.update(barrier='local-session-receipts (no GitHub run; same-run artifact service identity not checked)',
                     members={n: {'bytes': len(b), 'sha256': contract.digest(b)} for n, b in files.items()})
        driver.save(out / 'operator-proof.json', proof)
        return proof
    return barrier


def install():
    head = current_ggml()
    contract.CANDIDATE_GGML = head        # operator_receipt compares source.ggml to this
    driver.CANDIDATE_GGML = head          # recorded in candidate-overlay.json
    driver.source_guard = replay_source_guard
    driver.operator_barrier = replay_operator_barrier


def vector_rows(report):
    rows = {}
    for case, gates in report['cases'].items():
        for gate, row in gates.items():
            rows['final/%s/%s' % (case, gate)] = row['sha256']
    for gate, threads in report.get('first_sentence_stages', {}).items():
        for t, stages in threads.items():
            for s in stages:
                rows['stage/%s/%s/%s' % (gate, t, s['stage'])] = s['candidate_sha256']
    for i, row in enumerate(report.get('interleaving', [])):
        rows['interleaving/%d/%s' % (i, row['label'])] = contract.digest(contract.f32_bytes(row['vector']))
    return rows


def replay_compare(out, frozen_path):
    current = json.loads((out / 'ab.json').read_text())
    frozen = json.loads(Path(frozen_path).read_text())
    old_freeze = json.loads((out / 'old-panel-freeze.json').read_text())
    a, b = vector_rows(frozen), vector_rows(current)
    diffs = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    token_diffs = []
    frozen_freeze = Path(frozen_path).with_name('old-panel-freeze.json')
    if frozen_freeze.exists():
        ff = json.loads(frozen_freeze.read_text())
        for case, row in ff['cases'].items():
            if row['token_ids'] != old_freeze['cases'][case]['token_ids']:
                token_diffs.append(case)
    metric_diffs = {}
    for case, gates in frozen['cases'].items():
        for gate, row in gates.items():
            now = current['cases'].get(case, {}).get(gate)
            if now is None or now['native_comparison'] != row['native_comparison'] or now['final_gate_pass'] != row['final_gate_pass']:
                metric_diffs['%s/%s' % (case, gate)] = {'frozen': row['native_comparison'], 'current': now and now['native_comparison']}
    result = {'schema': 'crispembed-frozen-guard-replay-wasm-q8-v1', 'frozen_run': FROZEN_RUN,
              'frozen_source': frozen['source'], 'replay_source': os.environ.get('GITHUB_SHA'),
              'replay_run': int(os.environ.get('GITHUB_RUN_ID', '0') or 0),
              'source_guard': current['source_guard'], 'compared_hashes': len(a),
              'hash_diffs': diffs, 'token_diffs': token_diffs, 'metric_diffs': metric_diffs,
              'frozen_failures': frozen.get('failures'), 'current_failures': current.get('failures'),
              'frozen_state': frozen.get('state'), 'current_state': current.get('state'),
              'bit_identical': not diffs and not token_diffs and not metric_diffs and frozen.get('failures') == current.get('failures')}
    driver.save(out / 'replay-compare.json', result)
    print(json.dumps({k: result[k] for k in ('compared_hashes', 'bit_identical')}), 'diffs:', diffs[:20], token_diffs, list(metric_diffs)[:20])
    if not result['bit_identical']:
        raise SystemExit('Replay differs from frozen run %d' % FROZEN_RUN)


if __name__ == '__main__':
    install()
    if '--operator-receipts' in sys.argv:
        i = sys.argv.index('--operator-receipts')
        driver.operator_barrier = local_operator_barrier(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    if sys.argv[1:2] == ['replay-compare']:
        import argparse
        p = argparse.ArgumentParser(); p.add_argument('phase'); p.add_argument('--output', type=Path, required=True)
        p.add_argument('--frozen', type=Path, required=True); a = p.parse_args()
        replay_compare(a.output, a.frozen)
    else:
        # Run the unchanged driver CLI with the identity assertions replaced.
        sys.argv[0] = str(ROOT / 'tools/ci-heavy/wasm_q8_rounding_ab_v1.py')
        # Execute only the driver's __main__ block against the patched module namespace.
        src = (ROOT / 'tools/ci-heavy/wasm_q8_rounding_ab_v1.py').read_text()
        main = src[src.index("if __name__=='__main__':"):]
        exec(compile(main, sys.argv[0], 'exec'), driver.__dict__ | {'__name__': '__main__'})
