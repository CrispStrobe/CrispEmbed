"""Kaggle CPU kernel: run tools/ci-heavy/ocr_experimental_alias_consumer.py for the GOT-OCR2 experimental
aliases (got-q8, got-mix) at @COMMIT@: alias download + SHA-256, offline reuse, alias/path equality and
licence-gate rejection. Output: /kaggle/working/got-alias-consumer/<key>/report.json."""
import os, subprocess, sys
COMMIT = '@COMMIT@'
TMP = '/kaggle/temp'; OUT = '/kaggle/working/got-alias-consumer'
def sh(*a, **k): print('+', *a, flush=True); return subprocess.run(list(a), **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', TMP + '/ce', check=True)
os.chdir(TMP + '/ce'); sh('git', 'checkout', '-q', COMMIT, check=True)
sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml', check=True)
fail = 0
KEYS = '@KEYS@'
for key in (('got-q8', 'got-mix') if KEYS.startswith('@') else KEYS.split(',')):
    env = dict(os.environ, HEAVY_OUT=f'{OUT}/{key}', HEAVY_SCRATCH=f'{TMP}/s-{key}')
    r = sh(sys.executable, 'tools/ci-heavy/ocr_experimental_alias_consumer.py', key, env=env)
    print(key, 'exit', r.returncode, flush=True); fail |= r.returncode != 0
sys.exit(1 if fail else 0)
