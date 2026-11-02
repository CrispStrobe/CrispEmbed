#!/usr/bin/env bash
# OXR D-FINE F16 investigation on the original PyTorch service model (CPU):
# per-tensor F16 rounding audit and per-group F16 weight-rounding sensitivity
# of the final selected regions (tools/oxr_dfine_f16_audit.py, _sensitivity.py).
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/oxr_dfine_f16_probe.sh -f isolated_lane=oxr-dfine-f16
set -uo pipefail
TOP=${HEAVY_OUT:?}; S=${HEAVY_SCRATCH:?}
REV=c708407f9d2b4df658f2543d9132b62bb7221df4; BP=c1a1dcb312ee9f691404f127785510453c0d9350
python -m pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q numpy==1.26.4 opencv-python-headless==4.11.0.86 gguf pillow huggingface_hub pydantic httpx || exit 3
git clone -q https://github.com/SoMarkAI/OXR.git "$S/oxr" && git -C "$S/oxr" checkout -q $BP || exit 3
CKPT=$(python -c "from huggingface_hub import hf_hub_download as d; d('SoMarkAI/OXR-1.0','layout_analysis/metadata.json',revision='$REV'); print(d('SoMarkAI/OXR-1.0','layout_analysis/last.pth',revision='$REV'))") || exit 5
echo "source=$(git rev-parse HEAD)" > "$TOP/provenance.txt"
python tools/oxr_dfine_f16_audit.py --model SoMarkAI/OXR-1.0 --output "$TOP/f16-audit.json" > "$TOP/f16-audit.txt" 2>&1
echo "audit=$?" >> "$TOP/results.txt"
IMGS=()
for e in receipt:cc0/receipt_example.png german:cc0/german_official_print.jpg scan:scan_page_pd.png table:cc0/simple_table.jpg japanese:japanese_print.png; do
    id=${e%%:*}; src=${e#*:}
    python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" "tests/regression/images/$src" "$S/$id.png"
    IMGS+=(--image "$S/$id.png")
done
python tools/oxr_dfine_f16_sensitivity.py --blueprint "$S/oxr" --checkpoint "$CKPT" "${IMGS[@]}" --output "$TOP/f16-sensitivity.json" > "$TOP/f16-sensitivity.log" 2>&1
echo "sensitivity=$?" >> "$TOP/results.txt"
cat "$TOP/results.txt"
