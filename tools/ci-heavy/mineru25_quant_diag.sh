#!/usr/bin/env bash
# Is MinerU's Q8_0 vision drift a runtime error or real weight/activation precision?
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/mineru25_quant_diag.sh \
#       -f args="[CASE_IDS]" -f isolated_lane=mineru25-quant-diag
#
# Per case it freezes two original-model references: A (original F32 weights) and
# B (the vision matrices replaced by the exact dequantized weights stored in our
# Q8_0 GGUF, tools/mineru25_validate.py --vision-weights-gguf). Then:
#   f32-vs-A           control, native F32 (expected ~1.0 everywhere)
#   q8-vs-A            native Q8_0 as shipped (the reported spikes)
#   q8f32act-vs-B      native Q8_0 weights with float32 activations against the SAME
#                      weights: any vision drift here is a native runtime error
#   q8-vs-B            8-bit activation rounding alone
# Diagnostic only: references B are never qualification references. Exit 0 always.
set -uo pipefail
CASE_IDS=${1:-formula,arabic}
THREADS=4
MODEL=opendatalab/MinerU2.5-Pro-2604-1.2B
REV=d3f5e08d073c21466bbabe21c71bb1e9c2e595da
BLUEPRINT_REV=c9797cde8f54cc48306196fc6aebe73ea31dd4a3
TOP=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_MINERU25_PRO=1 QWEN2VL_OCR_FORCE_CPU=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
echo "source=$(git rev-parse HEAD)" > "$TOP/provenance.txt"

python -m pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q -r tools/mineru25-reference-requirements.txt || exit 3
git init -q "$S/blueprint" && git -C "$S/blueprint" fetch -q --depth 1 \
    https://github.com/opendatalab/mineru-vl-utils.git "$BLUEPRINT_REV" &&
    git -C "$S/blueprint" checkout -q --detach FETCH_HEAD || exit 3
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF >/dev/null &&
    cmake --build "$S/build" -j4 --target test-mineru25-diff crispembed-quantize > "$S/build.log" 2>&1 ||
    { tail -40 "$S/build.log"; exit 4; }
DIFF=$(find "$S/build" -type f -name test-mineru25-diff -perm -u+x | head -1)
QUANT=$(find "$S/build" -type f -name crispembed-quantize -perm -u+x | head -1)
python - "$MODEL" "$REV" "$S/checkpoint" <<'PY' || exit 5
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
mkdir -p "$S/gguf"
python models/convert-mineru25-pro-to-gguf.py --model "$S/checkpoint" --dtype f32 --output "$S/gguf/f32.gguf" || exit 5
"$QUANT" "$S/gguf/f32.gguf" "$S/gguf/q8_0.gguf" q8_0 > "$TOP/quantize-q8_0.log" 2>&1 || exit 5

CASES=$(python - "$CASE_IDS" <<'PY'
import json, sys
inputs = json.load(open("docs/mineru25-pro-publication-v1.json"))["case_inputs"]
for i in sys.argv[1].split(","):
    print(i, *inputs[i])
PY
)
while read -r id image task; do
    OUT=$TOP/$id; mkdir -p "$OUT"
    IMG=tests/regression/images/$image
    case "${IMG,,}" in *.jpg | *.jpeg)
        python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" \
            "$IMG" "$S/$id.png" && IMG=$S/$id.png ;; esac
    for kind in A B; do
        extra=(); [ $kind = B ] && extra=(--vision-weights-gguf "$S/gguf/q8_0.gguf")
        python tools/mineru25_validate.py reference --model "$S/checkpoint" --blueprint "$S/blueprint" \
            --image "$IMG" --task "$task" --dtype f32 --device cpu --threads $THREADS --repeat 2 --max-tokens 64 \
            --output "$OUT/python-$kind.json" --dump-reference "$S/$id-$kind.gguf" "${extra[@]}" \
            > "$OUT/reference-$kind.log" 2>&1 || echo "reference $kind failed" >> "$OUT/errors.txt"
    done
    run() { # name model ref [env...]
        local name=$1 model=$2 ref=$3; shift 3
        env "$@" "$DIFF" "$S/gguf/$model.gguf" "$S/$id-$ref.gguf" $THREADS 0.002 0.99999 > "$OUT/$name.txt" 2>&1
    }
    run f32-vs-A f32 A
    run q8-vs-A q8_0 A
    run q8f32act-vs-B q8_0 B CRISPEMBED_MINERU25_PRO_VISION_F32_ACT=1
    run q8-vs-B q8_0 B
    run q8f32act-vs-A q8_0 A CRISPEMBED_MINERU25_PRO_VISION_F32_ACT=1
done <<< "$CASES"

python - "$TOP" <<'PY'
import json, sys
from pathlib import Path
top = Path(sys.argv[1]); lines = []
for case in sorted(p for p in top.iterdir() if p.is_dir()):
    lines.append(f"### {case.name}")
    lines.append("| stage | " + " | ".join(n for n in ("f32-vs-A", "q8-vs-A", "q8f32act-vs-B", "q8-vs-B", "q8f32act-vs-A")) + " |")
    lines.append("|---" * 6 + "|")
    data = {}
    for n in ("f32-vs-A", "q8-vs-A", "q8f32act-vs-B", "q8-vs-B", "q8f32act-vs-A"):
        f = case / f"{n}.txt"
        if f.exists() and "{" in f.read_text():
            t = f.read_text(); data[n] = {s["name"]: s for s in json.loads(t[t.index("{"):t.rindex("}") + 1])["stages"]}
    for stage in [f"vis_layer_{i}" for i in range(32)] + ["vis_merger_output", "llm_layer_0", "llm_final_norm", "llm_logits_last"]:
        cells = []
        for n in ("f32-vs-A", "q8-vs-A", "q8f32act-vs-B", "q8-vs-B", "q8f32act-vs-A"):
            s = data.get(n, {}).get(stage)
            r = (s or {}).get("rows") or {}
            cells.append("-" if not s else f"min {s['cos_min']:.4f} p01 {r.get('p01_cos', float('nan')):.4f} "
                         f"row {r.get('worst_row')} n {r.get('worst_ref_norm', 0):.3g}/{r.get('median_ref_norm', 0):.3g}")
        lines.append(f"| {stage} | " + " | ".join(cells) + " |")
(top / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
exit 0
