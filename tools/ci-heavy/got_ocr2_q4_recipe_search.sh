#!/usr/bin/env bash
# GOT-OCR2 mixed Q4 recipe search (research only; never publishes).
# Plan and rationale: docs/got-ocr2-q4-research.md.
#
# Every recipe is quantized on the runner from the token-exact F16-r2 (Hub c0f1aa7b) with the
# existing crispembed-quantize flags (--tensor-type / --decoder-f16 / --imatrix), then measured
# against the ORIGINAL stepfun-ai/GOT-OCR2_0 chat() pipeline on the frozen 3-case panel
# (fox / receipt_example / simple_form):
#   - generated token IDs: exact?, first divergence, character ratio, CER vs the original text
#   - teacher-forced final-logit cosine vs F16-r2 (CRISPEMBED_GOT_OCR_FORCE_IDS on F16-r2's own
#     IDs, CRISPEMBED_GOT_OCR_LOGITS_OUT): min / mean over all steps, top-1 agreement
#   - stage diff (19 stages) vs got-ocr2-ref-full.gguf (Python F32 reference)
#   - file size
# imatrix recipes calibrate on CC0 images that are NOT in the panel.
#
#   gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/got_ocr2_q4_recipe_search.sh \
#       -f isolated_lane=got-q4-search -f timeout_minutes=300 \
#       -f pip="torch transformers>=4.49 accelerate pillow gguf numpy safetensors huggingface_hub"
#   args: [recipe,recipe,...|all] [max_new_tokens, default 256]
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1
WANT=${1:-all}
MAXNEW=${2:-256}
REPO=cstr/got-ocr2-crispembed-GGUF
F16REV=c0f1aa7b5b52c15002e911d5d630e6d0dcb3ba8f
OLDREV=5a0b0a1adc
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
I=tests/regression/images
B=$S/build-cpu
note() { echo "$*" | tee -a "$OUT/results.txt"; }
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-cli crispembed-quantize test-got-ocr-diff > "$OUT/build.log" 2>&1 \
    || { tail -40 "$OUT/build.log"; exit 1; }
BIN="$B/crispembed"
Q=$(find "$B" -name crispembed-quantize -type f | head -1)
DIFF=$(find "$B" -name test-got-ocr-diff -type f | head -1)
M=$S/models; mkdir -p "$M"
curl -sfL --retry 3 -o "$M/f16.gguf" "https://huggingface.co/$REPO/resolve/$F16REV/got-ocr2-f16-r2.gguf" || exit 1
curl -sfL --retry 3 -o "$M/published_q4k.gguf" "https://huggingface.co/$REPO/resolve/$OLDREV/got-ocr2-q4_k.gguf" || exit 1
curl -sfL --retry 3 -o "$M/ref.gguf" "https://huggingface.co/$REPO/resolve/$OLDREV/got-ocr2-ref-full.gguf" || exit 1
sha256sum "$M"/*.gguf | sed "s|$M/||" > "$OUT/model-sha256.txt"

# llama.cpp Q4_K_M-style "more bits" layers for n_layer=24: first/last eighth + every third.
KM_LAYERS="0 1 2 5 8 11 14 17 20 21 22 23"
km_rules() { for i in $KM_LAYERS; do echo "--tensor-type l.blk.$i.attn_v.weight=q6_k --tensor-type l.blk.$i.ffn_down.weight=q6_k"; done; }
VIS="--tensor-type v.*=f16"
declare -A R
R[q4k_repro]=""                                         # current default policy (vision Q8_0, embed Q8_0)
R[dec_f16]="--decoder-f16"                              # attribution: vision Q8_0 only
R[vis_f16]="$VIS"                                       # vision + projector F16, decoder Q4_K
R[vis_f16_km]="$VIS $(km_rules)"                        # + Q4_K_M-style attn_v/ffn_down Q6_K
R[vis_f16_attn_q8]="$VIS --tensor-type l.blk.*.attn_?.weight=q8_0"
R[vis_f16_emb_f16]="$VIS --tensor-type l.embed_tokens.weight=f16"
R[vis_f16_imat]="$VIS --imatrix $M/imatrix.gguf"
R[vis_f16_km_imat]="$VIS $(km_rules) --imatrix $M/imatrix.gguf"
ORDER="q4k_repro dec_f16 vis_f16 vis_f16_km vis_f16_attn_q8 vis_f16_emb_f16 vis_f16_imat vis_f16_km_imat"
[ "$WANT" = all ] || ORDER=${WANT//,/ }

# imatrix calibration on F16-r2, CC0 images outside the panel.
if [[ " $ORDER " == *imat* ]]; then
    for img in commons_example_receipt.png commons_test_ocr_document.jpg german_official_print.jpg \
               simple_table.jpg receipt_historical.png handwritten_letter.jpg; do
        CRISPEMBED_IMATRIX_OUT="$M/imatrix.gguf" "$BIN" -t 4 -m "$M/f16.gguf" --ocr "$I/cc0/$img" \
            --ocr-max-tokens "$MAXNEW" > /dev/null 2>> "$OUT/calibration.log"
    done
    [ -s "$M/imatrix.gguf" ] && note "imatrix ok $(sha256sum "$M/imatrix.gguf" | cut -c1-16)" \
        || { note "imatrix MISSING (calibration did not flush)"; ORDER=$(echo "$ORDER" | tr ' ' '\n' | grep -v imat | xargs); }
fi

for r in $ORDER; do
    set -f  # the GLOB=TYPE rules must reach the quantizer unexpanded
    # shellcheck disable=SC2086
    "$Q" "$M/f16.gguf" "$M/$r.gguf" q4_k ${R[$r]} > "$OUT/quant.$r.log" 2>&1 || { set +f; note "quantize $r FAILED"; continue; }
    set +f
    note "recipe $r bytes=$(stat -c %s "$M/$r.gguf") flags=${R[$r]//$M\//}"
done

CASES=(fox=$I/fox.png receipt_example=$I/cc0/receipt_example.png simple_form=$I/cc0/simple_form.png)
python -m venv "$S/venv-orig"
VP="$S/venv-orig/bin/python"
"$VP" -m pip install -q --upgrade pip
"$VP" -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
"$VP" -m pip install -q "transformers==4.37.2" "tokenizers<0.16" tiktoken verovio requests accelerate pillow "numpy<2" safetensors \
    > "$OUT/venv-orig.log" 2>&1 || { note "venv-orig install FAILED"; exit 1; }
"$VP" -I tools/ci-heavy/got_ocr2_orig_probe.py "$OUT" "$MAXNEW" 20 "${CASES[@]}" > "$OUT/orig.log" 2>&1 \
    || { note "original probe FAILED"; tail -40 "$OUT/orig.log"; exit 1; }
rm -f "$OUT"/orig.*.pixels.npy

ARMS="f16 published_q4k $ORDER"
for t in $ARMS; do
    [ -s "$M/$t.gguf" ] || continue
    "$DIFF" "$M/$t.gguf" "$M/ref.gguf" > /dev/null 2> "$OUT/diff.$t.err"
    for c in "${CASES[@]}"; do
        id=${c%%=*}; img=${c#*=}; d="$OUT/native/$t.$id"; mkdir -p "$d"
        t0=$EPOCHREALTIME
        CRISPEMBED_GOT_OCR_DUMP_DIR="$d" "$BIN" -t 4 -m "$M/$t.gguf" --ocr "$img" --ocr-max-tokens "$MAXNEW" \
            > "$OUT/ocr.$t.$id.txt" 2> "$OUT/ocr.$t.$id.err"
        echo "$id $t secs=$(python -c "print(round($EPOCHREALTIME-$t0,2))")" >> "$OUT/timing.txt"
        rm -f "$d/pixels.f32"
        # Teacher-forced logits on F16-r2's own token stream (F16-r2 is token-exact vs the original).
        CRISPEMBED_GOT_OCR_FORCE_IDS="$OUT/native/f16.$id/gen_ids.txt" CRISPEMBED_GOT_OCR_LOGITS_OUT="$S/logits.$t.$id.f32" \
            "$BIN" -t 4 -m "$M/$t.gguf" --ocr "$img" --ocr-max-tokens "$MAXNEW" > /dev/null 2>&1
        # Reduce to per-step stats now; a full logit stream is ~0.6 MB per step.
        python - "$S/logits.f16.$id.f32" "$S/logits.$t.$id.f32" > "$OUT/logits.$t.$id.json" <<'PYEOF'
import json, os, sys
import numpy as np
V = 151860
def load(p):
    if not os.path.exists(p): return None
    a = np.fromfile(p, np.float32)
    return a.reshape(-1, V) if a.size and a.size % V == 0 else None
b, x = load(sys.argv[1]), load(sys.argv[2])
if b is None or x is None or b.shape != x.shape:
    print(json.dumps({"error": "missing or shape mismatch"})); sys.exit()
c = (b * x).sum(1) / np.maximum(np.linalg.norm(b, axis=1) * np.linalg.norm(x, axis=1), 1e-30)
print(json.dumps({"steps": int(b.shape[0]), "logit_cos_min": round(float(c.min()), 6),
                  "logit_cos_mean": round(float(c.mean()), 6), "argmin_step": int(c.argmin()),
                  "top1_agree": round(float((b.argmax(1) == x.argmax(1)).mean()), 4)}))
PYEOF
        [ "$t" = f16 ] || rm -f "$S/logits.$t.$id.f32"
    done
done

python - "$OUT" "$S" "$ARMS" > "$OUT/compare.json" <<'EOF'
import difflib, json, os, sys, re
out, scratch, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
V = 151860
STOPS = (151643, 151645)
orig = json.load(open(os.path.join(out, "orig.json")))
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
def logit_stats(t, cid):
    p = os.path.join(out, f"logits.{t}.{cid}.json")
    try:
        return json.load(open(p))
    except Exception:
        return {}
rec = {"reference": "stepfun-ai/GOT-OCR2_0 chat() (original remote code, CPU F32, no_repeat_ngram 20)",
       "logit_reference": "F16-r2 teacher-forced on its own IDs", "cases": [], "arms": {}}
for t in arms:
    p = os.path.join(out, f"diff.{t}.err")
    if os.path.exists(p):
        cos = [float(m) for m in re.findall(r"cos_min=([0-9.]+)", open(p).read())]
        rec["arms"][t] = {"stages": len(cos), "min_stage_cos": min(cos) if cos else None}
for cid in ("fox", "receipt_example", "simple_form"):
    ref = orig["cases"][cid]["runs"]["20"]
    ref_ids = [i for i in ref["token_ids"] if i not in STOPS]
    rtxt = ref["text"].strip()
    case = {"id": cid, "reference_ids": len(ref_ids), "arms": {}}
    for t in arms:
        tp = os.path.join(out, f"ocr.{t}.{cid}.txt")
        if not os.path.exists(tp): continue
        txt = open(tp, errors="replace").read().strip()
        gp = os.path.join(out, "native", f"{t}.{cid}", "gen_ids.txt")
        ids = [int(x) for x in open(gp).read().split() if int(x) not in STOPS] if os.path.exists(gp) else None
        div = None
        if ids is not None and ids != ref_ids:
            div = next((k for k, (x, y) in enumerate(zip(ids, ref_ids)) if x != y), min(len(ids), len(ref_ids)))
        a = {"ids_exact": ids == ref_ids, "first_divergence": div, "n_ids": None if ids is None else len(ids),
             "char_ratio": round(difflib.SequenceMatcher(None, txt, rtxt, autojunk=False).ratio(), 4),
             "cer": round(lev(txt, rtxt) / max(1, len(rtxt)), 4)}
        a.update({k: v for k, v in logit_stats(t, cid).items() if k != "error"})
        case["arms"][t] = a
    rec["cases"].append(case)
print(json.dumps(rec, indent=1))
EOF
rm -f "$S"/logits.*.f32
python - "$OUT/compare.json" <<'EOF' | tee -a "$OUT/results.txt"
import json, sys
r = json.load(open(sys.argv[1]))
for t, v in r["arms"].items():
    row = [c["arms"].get(t, {}) for c in r["cases"]]
    print(f"{t:18s} exact={sum(bool(x.get('ids_exact')) for x in row)}/3 "
          f"div={[x.get('first_divergence') for x in row]} cer={[x.get('cer') for x in row]} "
          f"logit_cos_min={[x.get('logit_cos_min') for x in row]} top1={[x.get('top1_agree') for x in row]} "
          f"min_stage_cos={v['min_stage_cos']}")
EOF
