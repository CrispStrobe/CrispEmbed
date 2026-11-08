"""Kaggle CPU kernel: run tools/ci-heavy/unlimited_ocr2_q8sh16_publish.sh (census-gated experimental upload) at
@COMMIT@ with the HF token from the attached token dataset. Resolves the token BEFORE any build/compute.
Output: /kaggle/working/out/."""
import os, subprocess, sys
COMMIT = '@COMMIT@'; MODE = '@MODE@'
TMP = '/kaggle/temp'; OUT = '/kaggle/working/out'
def sh(*a, **k): print('+', *a, flush=True); return subprocess.run(list(a), **k)
sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', TMP + '/ce', check=True)
os.chdir(TMP + '/ce'); sh('git', 'checkout', '-q', COMMIT, check=True)
sys.path.insert(0, os.getcwd()); import kaggle_harness as kh
tok = kh.resolve_hf_token(require=(MODE == 'publish'))
sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml', check=True)
sh(sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.17.1', 'huggingface_hub', check=True)
os.makedirs(OUT, exist_ok=True); os.makedirs(TMP + '/s', exist_ok=True)
env = dict(os.environ, HEAVY_OUT=OUT, HEAVY_SCRATCH=TMP + '/s', HF_TOKEN_2=tok or '')
r = sh('bash', 'tools/ci-heavy/unlimited_ocr2_q8sh16_publish.sh', MODE, env=env)
sh('rm', '-rf', TMP + '/s')
sys.exit(r.returncode)
