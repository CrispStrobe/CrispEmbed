"""Kaggle CPU kernel: original-first CPU F32 reference for baidu/Unlimited-OCR-2.

Clones an exact CrispEmbed commit and runs tools/unlimited-ocr2/reference.py with
the model card's pinned runtime. Outputs land in /kaggle/working/uocr2.
"""
import os, subprocess, sys
COMMIT = '@COMMIT@'
def sh(*a): print('+', *a, flush=True); subprocess.run(list(a), check=True)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', '/kaggle/temp/ce')
os.chdir('/kaggle/temp/ce'); sh('git', 'checkout', '-q', COMMIT)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'torch==2.13.0', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
sh(sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.52.1', 'huggingface-hub==0.36.0', 'tokenizers==0.21.4',
   'addict==2.4.0', 'einops==0.8.2', 'easydict==1.13', 'pillow==11.3.0', 'numpy==2.2.6')
os.environ.update(OMP_NUM_THREADS=str(os.cpu_count()), HF_HUB_DISABLE_TELEMETRY='1')
sh(sys.executable, 'tools/unlimited-ocr2/reference.py', '--output', '/kaggle/working/uocr2', '--max-new-tokens', '384')
open('/kaggle/working/commit.txt', 'w').write(COMMIT+'\n')
