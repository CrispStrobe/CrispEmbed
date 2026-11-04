"""Run the existing real-scan kernel on the hosted CPU worker, with a hard budget."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
out = Path(os.environ['HEAVY_OUT']) / 'mineru-q4-scans'
scratch = Path(os.environ['HEAVY_SCRATCH']) / 'mineru-q4-scans'
out.mkdir(parents=True, exist_ok=True)
scratch.mkdir(parents=True, exist_ok=True)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
source = (root / 'tools/kaggle/ocr-real-scans/kernel.py').read_text()
replacements = {
    "COMMIT = '@COMMIT@'": f'COMMIT = {commit!r}',
    "MODE = '@MODE@'": "MODE = 'native-mineru-q4'",
    'BUDGET = 10.9 * 3600': 'BUDGET = 4 * 3600',
    "OUT = '/kaggle/working/ocr-real-scans'": f'OUT = {str(out)!r}',
    "TMP = '/kaggle/temp'": f'TMP = {str(scratch)!r}',
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f'kernel source contract changed: {old!r}')
    source = source.replace(old, new, 1)
rendered = scratch / 'kernel.py'
rendered.write_text(source)
result = subprocess.run([sys.executable, '-u', str(rendered)], cwd=root)
if result.returncode:
    raise SystemExit(result.returncode)
report = json.loads((out / 'result.json').read_text())
rows = report['runs']
expected = set(report['scope']['selected_page_ids'])
if (report.get('fatal') or report['failed_pages'] or len(rows) != len(expected)
        or {r['page'] for r in rows} != expected
        or any(r.get('exit') != 0 or r.get('skipped') for r in rows)):
    raise SystemExit('incomplete or failed scan coverage; retain the partial receipt')
print(f'Completed {len(rows)} authenticated recognition requests; EOS remains unobserved.')
