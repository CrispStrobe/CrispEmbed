"""Kaggle CPU kernel: fresh frozen MoEdern original capture (27M) from an exact commit.

Clones the pinned branch commit, installs the same pinned runtime as the GitHub
workflow and writes outputs to /kaggle/working. No CrispEmbed native inference.
"""
import subprocess, sys, os
COMMIT = os.environ.get('MOEDERN_COMMIT', '@COMMIT@')
VARIANT = '@VARIANT@'
def sh(*a): print('+', *a, flush=True); subprocess.run(list(a), check=True)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT)
# Clean venv: Kaggle's preinstalled stack conflicts with the pinned runtime.
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')  # Kaggle python lacks ensurepip
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, 'tests/test_moedern_colbert_contract.py')
sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.8.0', '--index-url', 'https://download.pytorch.org/whl/cpu')
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'transformers==5.3.0',
   'sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git@b0bf39968ba461c34b29372144794eff1be06f6d', 'gguf==0.17.1')
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
sh(PY, 'tools/moedern-colbert/fresh_reference.py', '--variant', VARIANT,
   '--output', '/kaggle/working/moedern-original', '--convert-after-original')
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n')
