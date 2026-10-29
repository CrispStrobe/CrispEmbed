"""Render @SOURCE@ once after review; actual model-free numerical CPU audit.
No fake GitHub environment, model/corpus access or old guard changes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

SOURCE = '@SOURCE@'
OUT = Path('/kaggle/working/moedern-router-topk-v1')
SCRATCH = Path('/kaggle/temp/moedern-router-topk-v1')
OUT.mkdir(parents=True, exist_ok=False)
report = dict(source=SOURCE, completed=False, model_access=False, provider='kaggle-cpu', commands=[])

def save():
    (OUT / 'setup.json').write_text(json.dumps(report, indent=2) + '\n')

def command(argv, name, timeout):
    log = OUT / (name + '.log')
    with log.open('wb') as f:
        p = subprocess.run(list(map(str, argv)), stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    report['commands'].append(dict(name=name, exit=p.returncode, log=log.name, sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
    save()
    return p.returncode

try:
    assert re.fullmatch('[a-f0-9]{40}', SOURCE), 'Render reviewed source before submission'
    assert os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Batch' and not os.environ.get('GITHUB_ACTIONS')
    SCRATCH.mkdir(parents=True, exist_ok=False)
    checkout = SCRATCH / 'source'
    assert command(['git', 'clone', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', checkout], 'clone', 180) == 0
    assert command(['git', '-C', checkout, 'checkout', SOURCE], 'checkout', 120) == 0
    assert command(['git', '-C', checkout, 'submodule', 'update', '--init', 'ggml'], 'ggml', 180) == 0
    assert subprocess.check_output(['git', '-C', checkout, 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
    code = command(['python3', '-I', checkout / 'tools/ci-heavy/moedern_router_topk_v1.py', '--provider', 'kaggle-cpu',
                    '--reviewed-source', SOURCE, '--out', OUT / 'audit', '--scratch', SCRATCH / 'audit'], 'audit', 5700)
    result = json.loads((OUT / 'audit/review/hosted.json').read_text())
    report.update(completed=result['completed'], exit=code,
                  external_exact_session_get_status_list_script_required=True)
    save()
    raise SystemExit(code)
except Exception as e:
    report['error'] = type(e).__name__ + ': ' + str(e)
    save()
    raise
