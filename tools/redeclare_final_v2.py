#!/usr/bin/env python3
"""Re-declare the Ovis final-integrated v2 and successor-assets v2 protocols at a new base.

Any commit that changes a protected object (src, CMakeLists.txt, examples) or an individually
declared source file invalidates both declarations. This tool rewrites exactly the fields that
pin them and nothing else:

  - BASE in tools/ovisocr2_final_integrated_v2.py, runtime_base / final_base in the v2 JSONs,
    and the base mentions in docs/successor-assets-v2.md and tests/test_successor_assets_v2.py;
  - the git tree/blob ids of the protected objects in docs/ovisocr2-final-integrated-v2.json;
  - DECLARED sha256 values that changed (only in the v2 script and v2 JSON);
  - the auditor sha256 of tools/ovisocr2_final_integrated_v2.py.

Frozen records (v1 protocols, receipts under tests/results, other protocols) are never touched,
even if they mention the same hashes. Usage:

  tools/redeclare_final_v2.py [--base COMMIT] [--check] [--force-base]

--base defaults to HEAD, which must be committed (the protected objects are read from it).
--check only reports what would change and exits 1 if anything is stale.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'tools/ovisocr2_final_integrated_v2.py'
FINAL_JSON = 'docs/ovisocr2-final-integrated-v2.json'
BASE_FILES = (SCRIPT, FINAL_JSON, 'docs/successor-assets-v2.json', 'docs/successor-assets-v2.md',
              'tests/test_successor_assets_v2.py')
PROTECTED = ('src', 'CMakeLists.txt', 'examples')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def plan(base, force=False):
    edits = {}  # path -> list of (old, new)
    script = (ROOT / SCRIPT).read_text()
    old_base = re.search(r"^BASE = '([0-9a-f]{40})'", script, re.M).group(1)
    final = (ROOT / FINAL_JSON).read_text()
    for obj in PROTECTED:
        m = re.search(r'"%s": "([0-9a-f]{40})"' % re.escape(obj), final)
        if m and m.group(1) != git('rev-parse', f'{base}:{obj}'):
            edits.setdefault(FINAL_JSON, []).append((m.group(0), f'"{obj}": "{git("rev-parse", f"{base}:{obj}")}"'))
    declared = re.search(r'DECLARED = \{(.*?)\}', script, re.S).group(1)
    for path, digest in re.findall(r'"([^"]+)":\s*"([0-9a-f]{64})"', declared):
        now = sha256(path)
        if now != digest:
            for p in (SCRIPT, FINAL_JSON):
                edits.setdefault(p, []).append((digest, now))
    # Only move the base when something it pins actually changed (or when forced).
    if (edits or force) and old_base != base:
        for p in BASE_FILES:
            if old_base in (ROOT / p).read_text():
                edits.setdefault(p, []).insert(0, (old_base, base))
    return old_base, edits


def apply(edits):
    for path, pairs in edits.items():
        text = (ROOT / path).read_text()
        for old, new in pairs:
            text = text.replace(old, new)
        (ROOT / path).write_text(text)
    # The auditor hash covers the script itself, so it is recomputed after the script changed.
    final = (ROOT / FINAL_JSON).read_text()
    auditor = re.compile(r'("%s": ")([0-9a-f]{64})(")' % re.escape(SCRIPT))
    (ROOT / FINAL_JSON).write_text(auditor.sub(lambda m: m.group(1) + sha256(SCRIPT) + m.group(3), final))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', default='HEAD')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--force-base', action='store_true', help='move the base even if no pinned object changed')
    a = ap.parse_args(argv)
    base = git('rev-parse', a.base)
    old_base, edits = plan(base, a.force_base)
    stale = {p: len(v) for p, v in edits.items()}
    final = json.loads((ROOT / FINAL_JSON).read_text())
    auditor_stale = json.dumps(final).find(sha256(SCRIPT)) < 0
    if a.check:
        print(f'base {old_base[:8]} -> {base[:8]}; stale edits: {stale or "none"}; auditor stale: {auditor_stale}')
        return 1 if (stale or auditor_stale) else 0
    apply(edits)
    print(f're-declared {old_base[:8]} -> {base[:8]}: ' + ', '.join(f'{p} ({n})' for p, n in stale.items()))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
