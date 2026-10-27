"""Kaggle CPU kernel: MoEdern 27M/115M through the public raw-text API (root library).

Mounts (read-only, /kaggle/input): the frozen original capture kernel output
(kernel_sources crispembed-moedern-27m-original or -115m-original) and a dataset
holding the reviewed Unigram tokenizer component (sha256-pinned in
tools/pack_moedern_text_f32.py). Clones an exact commit, then runs worker.py.
@COMMIT@ and @VARIANT@ are substituted before `kaggle kernels push`.
"""
import os, subprocess, sys
COMMIT, VARIANT, MOUNT = '@COMMIT@', '@VARIANT@', '/kaggle/input'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
env = dict(os.environ, MOEDERN_COMMIT=COMMIT, MOEDERN_SCRATCH='/kaggle/temp/moedern-family')
r = subprocess.run([sys.executable, 'tools/kaggle/moedern-family-public/worker.py', VARIANT, MOUNT, '/kaggle/working/public'], env=env)
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n'); sys.exit(r.returncode)
