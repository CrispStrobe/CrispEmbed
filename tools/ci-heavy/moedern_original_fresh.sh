#!/usr/bin/env bash
# heavy-cpu entry: fresh frozen original capture for MoEdern 27M or 115M (arg 1).
# Same pinned runtime as moedern-colbert-original.yml; outputs land in $HEAVY_OUT.
set -euo pipefail
VARIANT="${1:?27M or 115M}"
case "$VARIANT" in 27M|115M) ;; *) echo "variant must be 27M or 115M"; exit 1 ;; esac
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1
python tests/test_moedern_colbert_contract.py -v
python -m pip install -q torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -q numpy==2.2.6 transformers==5.3.0 'sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git@b0bf39968ba461c34b29372144794eff1be06f6d' gguf==0.17.1
python -m pip freeze > "$HEAVY_OUT/requirements.txt"
python tools/moedern-colbert/fresh_reference.py --variant "$VARIANT" --output "$HEAVY_OUT/moedern-original" --convert-after-original
python - "$HEAVY_OUT/moedern-original/review/original.json" > "$HEAVY_OUT/summary.md" <<'P'
import json,sys
r=json.load(open(sys.argv[1]))
print(f"### MoEdern {r['variant']} fresh original: completed={r['completed']} cases={len(r['cases'])} scores={len(r['scores'])}")
P
