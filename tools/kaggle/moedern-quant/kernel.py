"""Kaggle CPU kernel: MoEdern 27M/115M quantized weights (F32-fast vs Q8_0 vs Q4_0).

Mounts both frozen original captures read-only (kernel_sources
crispembed-moedern-{27m,115m}-original), clones an exact commit (@COMMIT@ is
substituted before push), builds only the small standalone family probe (ggml CPU),
and runs tools/kaggle/moedern-quant/run.py in a clean uv Python 3.12 venv:
quantizes each model-F32.gguf to Q8_0/Q4_0, then reports cosine/max-abs vs the
frozen original vectors and docs/s at 2 and 4 threads per precision to
/kaggle/working/quant/quant.json. Diagnostic speed arm; no default changes.
"""
import os, subprocess, sys
COMMIT, MOUNT = '@COMMIT@', '/kaggle/input'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh('cmake', '-S', 'tests/moedern-family', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release')
sh('cmake', '--build', '/kaggle/temp/b', '--parallel', '4')
# Clean venv: Kaggle's preinstalled stack is unpinned and its python lacks ensurepip.
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.19.0')
sh(PY, 'tests/test_moedern_quantize.py')
env = dict(os.environ, MOEDERN_COMMIT=COMMIT)
r = subprocess.run([PY, 'tools/kaggle/moedern-quant/run.py', '/kaggle/temp/b/moedern-family-probe', MOUNT, '/kaggle/working/quant'], env=env)
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n'); sys.exit(r.returncode)
