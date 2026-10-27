"""Kaggle CPU kernel: Unlimited-OCR-2 precision sweep (F32/F16/Q8_0/Q4_K/mixed) vs the frozen
original reference (mounted read-only from crispembed-unlimited-ocr2-reference via kernel_sources).

Script kernels upload only this file, so it clones the repo at @COMMIT@ (substituted before
push), builds a uv Python 3.12 venv and runs tools/kaggle/unlimited-ocr2-precision/precision_worker.py.
Output: /kaggle/working/precision/precision.json + summary.md (no model files are kept).
"""
import os, subprocess, sys
COMMIT = '@COMMIT@'
def sh(*a, **k): print('+', *a, flush=True); subprocess.run(list(a), check=True, **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT); sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'tokenizers==0.21.4', 'huggingface-hub==0.36.0', 'safetensors==0.6.2')
sh(PY, '-u', 'tools/kaggle/unlimited-ocr2-precision/precision_worker.py', env=dict(os.environ, UOCR2_PREC_COMMIT=COMMIT))
