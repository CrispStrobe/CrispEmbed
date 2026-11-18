#!/usr/bin/env python3
"""Hosted frozen original/diagnostic PCS neutrality and unchanged gate runner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pcs_gate_capture import analyze, provenance, write_new

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
OUT = ROOT / 'pcs-evidence'
MODEL = ROOT / '.pcs-models/pcs-xlmr-base-q4_k-imatrix.gguf'
REF = ROOT / 'tests/regression/pcs/blueprint_ref.txt'
sys.path.insert(0, str(ROOT / 'tests'))
from pcs_parity import parse_ref


def run(name, argv, env=None):
    result = subprocess.run(list(map(str, argv)), env=env, capture_output=True)
    (OUT / (name + '.stdout')).write_bytes(result.stdout)
    (OUT / (name + '.stderr')).write_bytes(result.stderr)
    write_new(OUT / (name + '.status.json'), {'argv': list(map(str, argv)), 'returncode': result.returncode})
    return result


def main():
    OUT.mkdir(exist_ok=True)
    records, _ = parse_ref(REF)
    corpus = OUT / 'corpus.txt'
    corpus.write_text('\n'.join(r['text'] for r in records) + '\n')
    base = dict(os.environ, LD_LIBRARY_PATH=str(BUILD))
    base.pop('PCS_GATE_CAPTURE', None)
    base.pop('PCS_GATE_CAPTURE_PATH', None)
    inventories = {}
    for label, binary, enabled in [('original', 'punct-ab', False), ('diagnostic-off', 'punct-ab-capture', False),
                                    ('diagnostic-on', 'punct-ab-capture', True)]:
        directory = OUT / label
        directory.mkdir()
        env = dict(base, PCS_GATE_CAPTURE='1' if enabled else '0', PCS_GATE_CAPTURE_PATH=str(OUT / 'trace.tsv'))
        for key in ('LOGITS', 'PRE', 'SEG', 'CAP'):
            env['PCS_DUMP_' + key] = str(directory / (key.lower() + '.txt'))
        result = run(label, [BUILD / binary, MODEL, corpus], env)
        if result.returncode:
            raise RuntimeError(label + ' driver failed')
        inventories[label] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
        if len(inventories[label]) != 4:
            raise RuntimeError('Missing head dump')
    neutral = all(inventories[x] == inventories['original'] and
                  (OUT / (x + '.stdout')).read_bytes() == (OUT / 'original.stdout').read_bytes()
                  for x in ('diagnostic-off', 'diagnostic-on'))
    write_new(OUT / 'neutrality.json', {'all_four_head_dumps_and_decoded_bytes_equal': neutral, 'sha256': inventories})
    if not neutral:
        raise RuntimeError('Observer neutrality failed; trace is not admissible')
    # Separate same-request encoder readback for original head attribution.
    one = OUT / 'row37-corpus.txt'
    one.write_text(records[3]['text'] + '\n')
    probe_env = dict(base, PCS_DUMP_HIDDEN=str(OUT / 'row37-hidden.f32'))
    probe = run('row37-hidden', [BUILD / 'punct-ab', MODEL, one], probe_env)
    if probe.returncode or probe.stdout.splitlines() != [(OUT / 'original.stdout').read_bytes().splitlines()[3]]:
        raise RuntimeError('Separate hidden readback changes decoded output')
    # Run the unchanged comparator against the real original driver.
    result = run('original-gate', [sys.executable, ROOT / 'tests/pcs_parity.py', BUILD, MODEL, REF], base)
    print(result.stdout.decode(), flush=True)
    provenance(MODEL, MODEL.with_suffix('.gguf.origin.json'), BUILD / 'punct-ab-capture', BUILD, OUT / 'provenance.json')
    report = analyze(OUT / 'trace.tsv', REF)
    write_new(OUT / 'analysis.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    # A failing gate is evidence, and must remain a failing job.
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
