"""Kaggle CPU kernel: native MoEdern family engine vs a frozen original panel.

The original capture kernel's output is mounted read-only via kernel_sources.
Builds only the small standalone probe (ggml CPU) from an exact commit.
"""
import os, subprocess, sys
COMMIT, MOUNT = '@COMMIT@', '/kaggle/input'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh('cmake', '-S', 'tests/moedern-family', '-B', '/kaggle/temp/b', '-DCMAKE_BUILD_TYPE=Release')
sh('cmake', '--build', '/kaggle/temp/b', '--parallel', '4')
r = subprocess.run([sys.executable, 'tests/moedern-family/compare.py', '/kaggle/temp/b/moedern-family-probe', MOUNT, '/kaggle/working/parity'])
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n'); sys.exit(r.returncode)
