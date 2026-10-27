"""Kaggle CPU kernel: MoEdern 27M/115M, CrispEmbed runtime vs the author's (VAGOsolutions) runtime.

Diagnostic speed/output comparison only. Mounts the frozen original capture
(kernel_sources crispembed-moedern-<slug>-original) and the reviewed Unigram
tokenizer component dataset. Clones an exact commit (script kernels upload only
this file; the worker lives in the repo) and runs the worker in a clean
Python 3.12 venv built with uv. @COMMIT@ and @VARIANT@ are substituted before
`kaggle kernels push`.
"""
import os, subprocess, sys
COMMIT, VARIANT, MOUNT = '@COMMIT@', '@VARIANT@', '/kaggle/input'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
# Clean venv: Kaggle's preinstalled stack is unpinned and its python lacks ensurepip.
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'tokenizers==0.22.1')
sh(PY, 'tests/test_moedern_vago_compare.py')
env = dict(os.environ, MOEDERN_COMMIT=COMMIT, MOEDERN_SCRATCH='/kaggle/temp/moedern-vago')
r = subprocess.run([PY, 'tools/kaggle/moedern-vago-compare/worker.py', VARIANT, MOUNT, '/kaggle/working/compare'], env=env)
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n'); sys.exit(r.returncode)
