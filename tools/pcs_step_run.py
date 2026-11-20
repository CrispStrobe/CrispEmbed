#!/usr/bin/env python3
"""One hosted PCS step panel; retain all failures and observer neutrality."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pcs_step_capture import pack
from pcs_steps import inventory

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'pcs-step-evidence'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    original = ROOT / '.pcs-original'
    reference = out / 'reference'
    model = ROOT / '.pcs-models/pcs-xlmr-base-q4_k-imatrix.gguf'
    report = {'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'scope': 'one frozen case, original/stored weights, composed and independent-input operator replay',
              'absolute_budget': 1e-4, 'cosine_policy': 'unchanged crispembed_diff::Report::is_pass default',
              'commands': [], 'arms': {}, 'comparisons': {}}
    def save():
        (out / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    def run(label, argv, env=None):
        with (out / (label + '.stdout')).open('xb') as stdout, (out / (label + '.stderr')).open('xb') as stderr:
            result = subprocess.run(list(map(str, argv)), stdout=stdout, stderr=stderr, env=env, timeout=600)
        report['commands'].append({'label': label, 'argv': list(map(str, argv)), 'returncode': result.returncode})
        save()
        return result.returncode
    if run('reference', [sys.executable, ROOT / 'tools/dump_pcs_reference.py', '--steps',
                         '--model-dir', original, '--stored-model', model, '--output', reference]):
        raise RuntimeError('Original dumper failed; retain exact error')
    input_text = reference / 'input.txt'
    base = dict(os.environ, LD_LIBRARY_PATH=str(ROOT / 'build'), PCS_GATE_CAPTURE='0',
                PCS_STEP_CAPTURE='0', PCS_STEP_REPLAY='0', PCS_GATE_F32_MATMUL='0', PCS_FORCE_CPU='1')
    # Do not accept inherited relaxed comparison thresholds or experimental paths.
    for key in ['CRISPEMBED_DIFF_COS_THRESHOLD', 'CRISPEMBED_DIFF_DUMP_DIR', 'PCS_FLASH_ATTN',
                'PCS_DUMP_LAYER', 'PCS_DUMP_HIDDEN', 'PCS_STEP_REPLAY_REF', 'PCS_STEP_CAPTURE_PATH']:
        base.pop(key, None)
    original_directory = out / 'original-driver'
    original_directory.mkdir()
    original_environment = dict(base)
    for head in ['LOGITS', 'PRE', 'SEG', 'CAP']:
        original_environment['PCS_DUMP_' + head] = str(original_directory / (head.lower() + '.txt'))
    if run('original-driver', [ROOT / 'build/punct-ab', model, input_text], original_environment):
        raise RuntimeError('Original driver failed')
    original_heads = {head: (original_directory / (head.lower() + '.txt')).read_bytes()
                      for head in ['LOGITS', 'PRE', 'SEG', 'CAP']}
    composed_heads = {}
    def native(label, binary, materialized=False, captured=False, replay=None):
        directory = out / label
        directory.mkdir()
        env = dict(base, PCS_GATE_F32_MATMUL='1' if materialized else '0',
                   PCS_GATE_CAPTURE='1' if captured else '0', PCS_GATE_CAPTURE_PATH=str(directory / 'conditioning.tsv'),
                   PCS_STEP_CAPTURE='1' if captured else '0', PCS_STEP_CAPTURE_PATH=str(directory),
                   PCS_STEP_REPLAY='1' if replay else '0')
        if replay:
            env['PCS_STEP_REPLAY_REF'] = str(replay)
        for head in ['LOGITS', 'PRE', 'SEG', 'CAP']:
            env['PCS_DUMP_' + head] = str(directory / (head.lower() + '.txt'))
        if run(label, [ROOT / 'build' / binary, model, input_text], env):
            raise RuntimeError(label + ' failed')
        heads = {head: (directory / (head.lower() + '.txt')).read_bytes() for head in ['LOGITS', 'PRE', 'SEG', 'CAP']}
        text = (out / (label + '.stdout')).read_bytes()
        report['arms'][label] = {'decoded': text.decode(), 'head_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in heads.items()},
                                 'mode': 'independent-input replay, not composed quality' if replay else 'composed'}
        if not replay:
            control = 'materialized' if materialized else 'legacy'
            if captured:
                expected_heads, expected_text = composed_heads[control]
                if heads != expected_heads or text != expected_text:
                    raise RuntimeError(label + ' observer changes head dumps or decoded output')
                report['arms'][label]['observer_neutrality'] = 'all four head bytes and decoded bytes exact'
            else:
                composed_heads[control] = (heads, text)
                if not materialized and text != (out / 'original-driver.stdout').read_bytes():
                    raise RuntimeError('Disabled observer changes original decoded output')
                if not materialized and heads != original_heads:
                    raise RuntimeError('Disabled observer changes original head bytes')
        if captured:
            pack(directory, directory / 'capture.gguf')
        save()
    native('legacy-off', 'punct-ab-step')
    native('legacy-on', 'punct-ab-step', captured=True)
    native('materialized-off', 'punct-ab-step-matmul', materialized=True)
    native('materialized-on', 'punct-ab-step-matmul', materialized=True, captured=True)
    # Each native observed operation receives upstream independent same-weight
    # reference results. Both matrix paths remain separate and unchanged.
    native('legacy-replay', 'punct-ab-step', captured=True, replay=reference / 'stored.gguf')
    native('materialized-replay', 'punct-ab-step-matmul', materialized=True, captured=True, replay=reference / 'stored.gguf')
    names = out / 'inventory.txt'
    names.write_text('input.ids\n' + '\n'.join(inventory()) + '\n')
    pairs = [('stored-vs-original', reference / 'original.gguf', reference / 'stored.gguf')]
    for arm in ['legacy-on', 'materialized-on', 'legacy-replay', 'materialized-replay']:
        pairs.append((arm + '-vs-stored', reference / 'stored.gguf', out / arm / 'capture.gguf'))
    failed = False
    for label, ref, actual in pairs:
        code = run(label, [ROOT / 'build/crispembed-diff-pcs', ref, actual, names, '0.0001'], base)
        report['comparisons'][label] = {'exit_code': code, 'stdout': label + '.stdout',
                                         'classification': 'observed comparison; not a production gate replacement'}
        failed |= code != 0
    report['comparison_failures_retained'] = failed
    report['production_fixed'] = False
    save()
    # Negative numerical comparisons remain a failed job even after collecting
    # every arm. An execution-success label must not hide the known failure.
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
