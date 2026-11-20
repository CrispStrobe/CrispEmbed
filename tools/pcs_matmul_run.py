#!/usr/bin/env python3
"""Hosted-only PCS same-weight intervention and per-layer captures."""
import json
import os
from pathlib import Path
import subprocess
import sys

from pcs_capture_run import BUILD, OUT, MODEL, ROOT, REF, run
from pcs_gate_capture import analyze, write_new


def main():
    env = dict(os.environ, LD_LIBRARY_PATH=str(BUILD), PCS_GATE_CAPTURE='0', PCS_GATE_F32_MATMUL='0')
    binary = BUILD / 'punct-ab-matmul-control'
    original = run('matmul-off', [binary, MODEL, OUT / 'corpus.txt'], env)
    if original.returncode or original.stdout != (OUT / 'original.stdout').read_bytes():
        raise RuntimeError('Matmul intervention OFF changes original decoded bytes')
    # The unmodified comparator sees the candidate in its expected driver slot.
    gate_dir = OUT / 'matmul-on-gate'
    gate_dir.mkdir()
    (gate_dir / 'punct-ab').symlink_to(binary)
    on = dict(env, PCS_GATE_F32_MATMUL='1', PCS_GATE_CAPTURE='1',
              PCS_GATE_CAPTURE_PATH=str(OUT / 'matmul-on-trace.tsv'))
    result = run('matmul-on-gate', [sys.executable, ROOT / 'tests/pcs_parity.py', gate_dir, MODEL, REF], on)
    print(result.stdout.decode(), flush=True)
    write_new(OUT / 'matmul-on-analysis.json', analyze(OUT / 'matmul-on-trace.tsv', REF))
    # Separate one-request outputs retain both legacy and changed computations.
    for label, switch in [('legacy', '0'), ('materialized', '1')]:
        for layer in range(-1, 12):
            stage_env = dict(env, PCS_GATE_F32_MATMUL=switch, PCS_DUMP_LAYER=str(layer),
                             PCS_DUMP_HIDDEN=str(OUT / (label + '-layer-' + str(layer) + '.f32')))
            stage = run(label + '-layer-' + str(layer), [binary, MODEL, OUT / 'row37-corpus.txt'], stage_env)
            if stage.returncode:
                raise RuntimeError('Layer readback failed')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
