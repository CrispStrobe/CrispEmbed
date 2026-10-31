#!/usr/bin/env bash
# MinerU2.5-Pro-2604 full-model CPU qualification for one request case.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/mineru25_qualify.sh \
#       -f args="CASE TASK [VARIANTS] [MAX_TOKENS] [REPEAT]" -f isolated_lane=mineru25-<case>
#
# CASE is a path under tests/regression/images (gold, if any, at <stem>.gt.txt).
# VARIANTS defaults to f32,f16,q8_0,q4_k,q4_k-dq8 (decoder q8_0),q4_k-df16 (decoder f16). The F32 arm is the gate: preprocessing,
# all 64 stages (max abs <= 0.002 and cosine >= 0.99999), and exact IDs/text/EOS
# plus warm speed versus the pinned original Python, for both the ordinary and the
# reuse decode graph. Other precisions are measured against the same frozen
# reference and reported, never gated here. Every arm runs even after a failure;
# the exit code is nonzero when any F32 gate failed.
set -uo pipefail

CASE=${1:?case image path under tests/regression/images}
TASK=${2:?task: layout|text|table|formula|image}
VARIANTS=${3:-f32,f16,q8_0,q4_k,q4_k-dq8,q4_k-df16}
MAX_TOKENS=${4:-2048}
REPEAT=${5:-3}
THREADS=4
MODEL=opendatalab/MinerU2.5-Pro-2604-1.2B
REV=d3f5e08d073c21466bbabe21c71bb1e9c2e595da
BLUEPRINT_REV=c9797cde8f54cc48306196fc6aebe73ea31dd4a3

ROOT=$(pwd)
IMAGE=$ROOT/tests/regression/images/$CASE
GOLD=${IMAGE%.*}.gt.txt
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
[ -f "$IMAGE" ] || { echo "missing $IMAGE"; exit 2; }
export CRISPEMBED_MINERU25_PRO=1 QWEN2VL_OCR_FORCE_CPU=1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

declare -A RESULT
gate_fail=0
note() { RESULT[$1]=$2; echo "== $1: $2"; }
run_gate() { # name, command...
    local name=$1; shift
    if "$@"; then note "$name" pass; else note "$name" "fail($?)"; gate_fail=1; fi
}
run_report() {
    local name=$1; shift
    if "$@"; then note "$name" pass; else note "$name" "fail($?)"; fi
}

{
    echo "source=$(git rev-parse HEAD)"; git submodule status
    echo "case=$CASE task=$TASK variants=$VARIANTS max_tokens=$MAX_TOKENS repeat=$REPEAT threads=$THREADS"
    lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Flags' | cut -c1-300
} > "$OUT/provenance.txt"

echo "::group::reference environment"
python -m pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu &&
    python -m pip install -q -r tools/mineru25-reference-requirements.txt || exit 3
python -m pip freeze > "$OUT/packages.txt"
git init -q "$S/blueprint" && git -C "$S/blueprint" fetch -q --depth 1 \
    https://github.com/opendatalab/mineru-vl-utils.git "$BLUEPRINT_REV" &&
    git -C "$S/blueprint" checkout -q --detach FETCH_HEAD || exit 3
echo "::endgroup::"

echo "::group::build"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF >/dev/null &&
    cmake --build "$S/build" -j4 --target test-mineru25-run test-mineru25-diff test-mineru25-preprocess crispembed-quantize \
        > "$S/build.log" 2>&1 || { tail -60 "$S/build.log"; exit 4; }
B=$S/build
QUANT=$(find "$B" -type f -name crispembed-quantize -perm -u+x | head -1)
echo "::endgroup::"

echo "::group::checkpoint and conversion"
python - "$MODEL" "$REV" "$S/checkpoint" <<'PY' || exit 5
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
PY
mkdir -p "$S/gguf"
python models/convert-mineru25-pro-to-gguf.py --model "$S/checkpoint" --dtype f32 --output "$S/gguf/mineru25-f32.gguf" || exit 5
cp "$S/gguf/mineru25-f32.gguf.json" "$OUT/conversion-f32.json" 2>/dev/null
IFS=, read -ra VARS <<< "$VARIANTS"
for v in "${VARS[@]}"; do
    case "$v" in
        f32) ;;
        f16) python models/convert-mineru25-pro-to-gguf.py --model "$S/checkpoint" --dtype f16 \
                 --output "$S/gguf/mineru25-f16.gguf" || note "convert-f16" fail ;;
        *)
            # Mixed recipes: the 896-wide Qwen2 decoder (l.*) is the precision-sensitive part.
            case "$v" in
                q4_k-dq8) qargs=(q4_k --tensor-type 'l.*=q8_0') ;;
                q4_k-df16) qargs=(q4_k --decoder-f16) ;;
                *) qargs=("$v") ;;
            esac
            "$QUANT" "$S/gguf/mineru25-f32.gguf" "$S/gguf/mineru25-$v.gguf" "${qargs[@]}" > "$OUT/quantize-$v.log" 2>&1 ||
                note "quantize-$v" fail ;;
    esac
done
for f in "$S"/gguf/*.gguf; do printf '%s %s %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" "$(basename "$f")"; done |
    tee "$OUT/gguf-sha256.txt"
echo "::endgroup::"

echo "::group::original reference"
run_gate reference python tools/mineru25_validate.py reference --model "$S/checkpoint" --blueprint "$S/blueprint" \
    --image "$IMAGE" --task "$TASK" --dtype f32 --device cpu --threads $THREADS --repeat "$REPEAT" \
    --max-tokens "$MAX_TOKENS" --output "$OUT/python.json" --dump-reference "$S/python.gguf"
echo "::endgroup::"
[ -f "$OUT/python.json" ] || { note summary "no reference"; exit 1; }
# Large captures stay in scratch; keep them only when small enough to retain.
[ "$(stat -c %s "$S/python.gguf")" -lt $((400 * 1024 * 1024)) ] && cp "$S/python.gguf" "$OUT/"

echo "::group::preprocessing and 64-stage F32 numerical comparison"
run_gate preprocess bash -c "'$B/test-mineru25-preprocess' --image '$IMAGE' --reference '$S/python.gguf' --task '$TASK' > '$OUT/preprocessing.txt' 2>&1"
tail -5 "$OUT/preprocessing.txt"
run_gate stages-f32 bash -c "'$B/test-mineru25-diff' '$S/gguf/mineru25-f32.gguf' '$S/python.gguf' $THREADS 0.002 0.99999 > '$OUT/stages-f32.txt' 2>&1"
tail -15 "$OUT/stages-f32.txt"
for v in "${VARS[@]}"; do # diagnostic only: same budgets, not a gate for reduced precision
    [ "$v" = f32 ] || [ ! -f "$S/gguf/mineru25-$v.gguf" ] && continue
    run_report "stages-$v" bash -c "'$B/test-mineru25-diff' '$S/gguf/mineru25-$v.gguf' '$S/python.gguf' $THREADS 0.002 0.99999 > '$OUT/stages-$v.txt' 2>&1"
done
# A/B: CRISPEMBED_MINERU25_PRO_EXACT_GELU=1 evaluates the vision quick_gelu in F32
# instead of ggml's CPU fp16 table. Same budgets; reported, not yet a gate.
for v in f32 f16; do
    [ -f "$S/gguf/mineru25-$v.gguf" ] || continue
    run_report "stages-$v-xg" env CRISPEMBED_MINERU25_PRO_EXACT_GELU=1 bash -c "'$B/test-mineru25-diff' '$S/gguf/mineru25-$v.gguf' '$S/python.gguf' $THREADS 0.002 0.99999 > '$OUT/stages-$v-xg.txt' 2>&1"
done
echo "::endgroup::"

GOLD_ARGS=()
[ -f "$GOLD" ] && GOLD_ARGS=(--gold "$GOLD")
compare() { # variant reuse [exact_gelu]
    local tag=$1
    [ "${3:-0}" = 1 ] && tag=$1-xg
    CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH=$2 CRISPEMBED_MINERU25_PRO_EXACT_GELU=${3:-0} \
        python tools/mineru25_validate.py compare \
        --reference "$OUT/python.json" --native "$B/test-mineru25-run" --gguf "$S/gguf/mineru25-$1.gguf" \
        --image "$IMAGE" "${GOLD_ARGS[@]}" --max-slowdown 1.0 --output "$OUT/native-$tag-reuse$2.json"
}
echo "::group::decoded output and same-host speed"
run_gate decode-f32 compare f32 0
run_gate decode-f32-reuse compare f32 1
for v in "${VARS[@]}"; do
    [ "$v" = f32 ] || [ ! -f "$S/gguf/mineru25-$v.gguf" ] && continue
    run_report "decode-$v" compare "$v" 0
done
run_report decode-f32-xg compare f32 0 1
run_report decode-f32-xg-reuse compare f32 1 1
for v in "${VARS[@]}"; do
    [ "$v" = f32 ] || [ ! -f "$S/gguf/mineru25-$v.gguf" ] && continue
    run_report "decode-$v-xg" compare "$v" 0 1
done
echo "::endgroup::"

python - "$OUT" "$CASE" "$TASK" "$gate_fail" <<'PY'
import json, sys
from pathlib import Path
out, case, task, gate_fail = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "1"
ref = json.loads((out / "python.json").read_text())
warm = lambda r: sorted(r["timing"]["runs_ms"][1:])[len(r["timing"]["runs_ms"][1:]) // 2] if len(r["timing"]["runs_ms"]) > 1 else None
rows, arms = [], {}
for f in sorted(out.glob("native-*-reuse*.json")):
    r = json.loads(f.read_text())
    n = r.get("native", {})
    arm = {k: r.get(k) for k in ("status", "request_parity_pass", "speed_comparable", "speed_pass", "native_over_python")}
    arm.update(tokens=len(n.get("completion_ids", n.get("token_ids", [])) or []), eos=n.get("eos"),
               text_equal=n.get("text") == ref["text"], ground_truth=r.get("ground_truth"))
    arms[f.stem] = arm
    rows.append(f"| {f.stem} | {arm['request_parity_pass']} | {arm['text_equal']} | {arm['tokens']} | {arm['eos']} | {arm['native_over_python']} |")
summary = {"schema": "crispembed.mineru25.hosted-case.v1", "case": case, "task": task,
           "python": {"tokens": len(ref.get("token_ids", [])), "eos": ref["eos"], "capped": ref["capped"],
                      "warm_median_ms": warm(ref), "text": ref["text"]},
           "arms": arms, "f32_gates_pass": not gate_fail}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
(out / "summary.md").write_text("\n".join([
    f"### MinerU2.5-Pro {case} ({task}) — F32 gates {'PASS' if not gate_fail else 'FAIL'}",
    f"Python: {summary['python']['tokens']} tokens, EOS {ref['eos']}, warm median {summary['python']['warm_median_ms']} ms", "",
    "| arm | request parity | text equal | tokens | EOS | native/python warm |", "|---|---|---|---|---|---|", *rows, ""]))
PY
for k in "${!RESULT[@]}"; do echo "$k=${RESULT[$k]}"; done | sort | tee "$OUT/results.txt"
exit $gate_fail
