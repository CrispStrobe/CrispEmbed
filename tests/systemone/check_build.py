#!/usr/bin/env python3
"""Actual compiler/export/CLI controls, no model or generated response gold."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
assert os.environ.get('GITHUB_ACTIONS') == 'true'
build, out = map(Path, sys.argv[1:])
rows = json.loads((build/'compile_commands.json').read_text())
units = ['josev2b_backbone.cpp','josev2b_head.cpp','josev2b_composed.cpp','josev2b_text_frontend.cpp',
         'josev2b_answer_serializer.cpp','qwen35_decoder.cpp','qwen35_attention.cpp','qwen35_linear.cpp',
         'qwen35_delta.cpp','qwen35_norm.cpp','qwen35_positions.cpp','qwen35_profile.cpp']
selected = [r for r in rows if Path(r['file']).name in units]
assert len(selected) == 24, 'both real static/shared copies of all twelve units required'
for row in selected:
    flags = shlex.split(row['command'])
    assert '-ffp-contract=off' in flags and '-ffast-math' not in flags and '-Ofast' not in flags
    assert not any('JOSEV_' in x and 'TESTING' in x for x in flags)
symbols = (out/'symbols.txt').read_text()
for name in ('init','free','run_json','error','profile'):
    assert ' crispembed_systemone_'+name+'\n' in symbols
assert 'composed_test' not in symbols and 'josev2b_head::test' not in symbols
cli = build/'crispembed'
for args, expected in [(['--systemone','{}','--dim'], 'embedding options are invalid'),
                       (['--systemone','{}','--colbert'], 'choose only one encoding mode')]:
    run = subprocess.run([str(cli), '-m', 'absent.gguf', *args], text=True, capture_output=True, timeout=20)
    assert run.returncode != 0 and expected in run.stderr, run.stderr
(out/'build-controls.json').write_text(json.dumps({'completed':True, 'compiler_rows':len(selected),
    'exports':5, 'cli_invalid_options':2, 'model_loaded':False}, indent=2)+'\n')
