"""Kaggle CPU kernel: original-first CPU F32 reference for openbmb/MiniCPM-OCR-1.0.

Clones an exact CrispEmbed commit and runs tools/minicpm-ocr/reference.py with
the checkpoint's pinned Transformers (5.21.0, native `minicpm_ocr`). The 1.3B
F32 model needs about 6 GB RAM. Outputs land in /kaggle/working/minicpm-ocr.
Replace @COMMIT@ with the pushed commit before pushing the kernel.
"""
import os, subprocess, sys
COMMIT = '@COMMIT@'
def sh(*a): print('+', *a, flush=True); subprocess.run(list(a), check=True)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')  # Kaggle python lacks ensurepip
sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv'); PY = '/kaggle/temp/venv/bin/python'
sh(PY, '-m', 'pip', 'install', '-q', 'torch==2.13.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
sh(PY, '-m', 'pip', 'install', '-q', 'transformers==5.21.0', 'pillow==11.3.0', 'numpy==2.2.6')
sh(PY, '-m', 'pip', 'freeze')
os.environ.update(OMP_NUM_THREADS=str(os.cpu_count()), HF_HUB_DISABLE_TELEMETRY='1')
sh(PY, 'tools/minicpm-ocr/reference.py', '--output', '/kaggle/working/minicpm-ocr', '--max-new-tokens', '1024')
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n')
