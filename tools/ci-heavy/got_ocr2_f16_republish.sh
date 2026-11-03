#!/usr/bin/env bash
# GOT-OCR2 F16 reconversion + qualification (+ optional publication).
#
# Root cause (got_ocr2_f16_probe.sh / got_ocr2_tensor_audit.py): the published
# got-ocr2-f16.gguf (Hub 5a0b0a1adc, sha256 aeaf1b44...) carries an 8 MiB block
# of zero bytes inside l.embed_tokens.weight (token rows 43..2091), so most
# English/ASCII token embeddings are zero and the decoder emits "colorcolor...".
# Every other tensor matches the original checkpoint; the converter is correct.
#
#   1. Reconvert stepfun-ai/GOT-OCR2_0 -> F16 with --verify (byte readback).
#   2. Tensor audit of the new file against the original checkpoint.
#   3. Byte map of new vs old file (expected: only the zeroed span differs).
#   4. Stage diff vs got-ocr2-ref-full.gguf; native CPU decode of the frozen
#      cases (F16-r2, published Q4_K, old F16) vs the ORIGINAL stepfun-ai/GOT-OCR2_0
#      chat() pipeline (got_ocr2_orig_probe.py in a transformers==4.37.2 venv, CPU
#      F32, no_repeat_ngram 20), on generated token IDs (CRISPEMBED_GOT_OCR_DUMP_DIR)
#      and text. Until 3 November 2026 this step compared against the HF port
#      (stepfun-ai/GOT-OCR-2.0-hf via got_ocr2_hf_probe.py), which is not a faithful
#      reference: its processor adds a leading-space " OCR" token and drops the
#      system string's indentation and the ngram rule. The page ratios in run
#      41391040980 used the HF port.
#   5. With "publish": upload as NEW file got-ocr2-f16-r2.gguf (the old file is
#      kept), then verify the Hub LFS sha256 and an empty-cache readback hash.
#
#   gh workflow run heavy-cpu.yml --ref fix/got-ocr2-f16 \
#       -f script=tools/ci-heavy/got_ocr2_f16_republish.sh -f isolated_lane=got-f16 [-f args=publish] \
#       -f pip="torch transformers>=4.49 accelerate pillow gguf numpy safetensors huggingface_hub"
#   args: [qualify|publish] [max_new_tokens, default 256]
set -uo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
export CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1
MODE=${1:-qualify}
MAXNEW=${2:-256}
NEW=got-ocr2-f16-r2.gguf
REPO=cstr/got-ocr2-crispembed-GGUF
OLDREV=5a0b0a1adc
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
python -c "import torch, transformers, gguf; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gguf', gguf.__version__)" >> "$OUT/provenance.txt"
I=tests/regression/images
B=$S/build-cpu
fail=0
note() { echo "$*" | tee -a "$OUT/results.txt"; }
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-cli test-got-ocr-diff > "$OUT/build.log" 2>&1 \
    || { tail -40 "$OUT/build.log"; exit 1; }
BIN="$B/crispembed"
DIFF=$(find "$B" -name test-got-ocr-diff -type f | head -1)
M=$S/models; mkdir -p "$M"
get() { [ -s "$M/$1" ] || curl -sfL --retry 3 -o "$M/$1" "https://huggingface.co/$REPO/resolve/$OLDREV/$1"; }
get got-ocr2-f16.gguf; get got-ocr2-q4_k.gguf; get got-ocr2-ref-full.gguf

# 1. reconvert
python models/convert-got-ocr-to-gguf.py --model stepfun-ai/GOT-OCR2_0 --output "$M/$NEW" --dtype f16 --verify \
    > "$OUT/convert.log" 2>&1 || { note "convert FAILED"; tail -20 "$OUT/convert.log"; exit 1; }
note "convert ok: $(grep -E '^Verified' "$OUT/convert.log")"
sha256sum "$M"/*.gguf | sed "s|$M/||" > "$OUT/model-sha256.txt"
cat "$OUT/model-sha256.txt" | tee -a "$OUT/results.txt"

# 2. tensor audit
AUDIT_NAME=tensor-audit-r2.json python tools/ci-heavy/got_ocr2_tensor_audit.py "$M/$NEW" "$M/got-ocr2-f16.gguf" \
    > "$OUT/tensor-audit.log" 2>&1 || { note "tensor audit FAILED"; fail=1; }
grep -E "tensors at cos|== " "$OUT/tensor-audit.log" | tee -a "$OUT/results.txt"

# 3. byte map new vs old
python - "$M/$NEW" "$M/got-ocr2-f16.gguf" <<'EOF' | tee -a "$OUT/results.txt"
import sys, numpy as np
a = np.memmap(sys.argv[1], np.uint8, 'r'); b = np.memmap(sys.argv[2], np.uint8, 'r')
print(f"bytemap sizes new={a.size} old={b.size}")
if a.size == b.size:
    d = np.nonzero(a != b)[0]
    if d.size:
        print(f"bytemap differing bytes={d.size} first={d[0]} last={d[-1]} span={d[-1]-d[0]+1} "
              f"old_span_all_zero={bool((b[d[0]:d[-1]+1] == 0).all())}")
    else:
        print("bytemap identical")
EOF

# 4. stage diff + decodes
for tag in r2:"$M/$NEW" q4_k:"$M/got-ocr2-q4_k.gguf"; do
    t=${tag%%:*}; f=${tag#*:}
    "$DIFF" "$f" "$M/got-ocr2-ref-full.gguf" > "$OUT/diff.$t.out" 2> "$OUT/diff.$t.err"
    grep -E "cos_min" "$OUT/diff.$t.err" | sed "s/^/diff.$t /" >> "$OUT/results.txt"
done
CASES=(fox=$I/fox.png receipt_example=$I/cc0/receipt_example.png simple_form=$I/cc0/simple_form.png)
# The reference is the original pipeline. Same venv recipe as got_ocr2_pipeline_parity.sh: the
# remote code pins transformers 4.37.2.
python -m venv "$S/venv-orig"
VP="$S/venv-orig/bin/python"
"$VP" -m pip install -q --upgrade pip
"$VP" -m pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
"$VP" -m pip install -q "transformers==4.37.2" "tokenizers<0.16" tiktoken verovio requests accelerate pillow "numpy<2" safetensors \
    > "$OUT/venv-orig.log" 2>&1 || { note "venv-orig install FAILED"; tail -20 "$OUT/venv-orig.log"; exit 1; }
"$VP" -m pip freeze | grep -i -E "^(torch|torchvision|transformers|tiktoken|pillow|numpy)=" >> "$OUT/provenance.txt"
"$VP" -I tools/ci-heavy/got_ocr2_orig_probe.py "$OUT" "$MAXNEW" 20 "${CASES[@]}" > "$OUT/orig.log" 2>&1 \
    || { note "original probe FAILED"; tail -40 "$OUT/orig.log"; exit 1; }
rm -f "$OUT"/orig.*.pixels.npy  # 12 MiB each; the probe keeps their sha256 in orig.json
for c in "${CASES[@]}"; do
    id=${c%%=*}; img=${c#*=}
    for tag in r2:"$M/$NEW" q4_k:"$M/got-ocr2-q4_k.gguf" old_f16:"$M/got-ocr2-f16.gguf"; do
        t=${tag%%:*}; f=${tag#*:}
        d="$OUT/native/$t.$id"; mkdir -p "$d"
        t0=$EPOCHREALTIME
        CRISPEMBED_GOT_OCR_DUMP_DIR="$d" "$BIN" -t 4 -m "$f" --ocr "$img" --ocr-max-tokens "$MAXNEW" \
            > "$OUT/ocr.$t.$id.txt" 2> "$OUT/ocr.$t.$id.err"
        echo "$id $t rc=$? secs=$(python -c "print(round($EPOCHREALTIME-$t0,2))")" >> "$OUT/timing.txt"
        rm -f "$d/pixels.f32"
    done
done
python - "$OUT" > "$OUT/compare.json" <<'EOF'
import difflib, json, os, sys
out = sys.argv[1]
STOPS = (151643, 151645)
orig = json.load(open(os.path.join(out, "orig.json")))
timing = {}
for line in open(os.path.join(out, "timing.txt")):
    p = line.split(); timing[(p[0], p[1])] = {"rc": p[2], "secs": float(p[3].split("=")[1])}
def ids_of(path):
    return [int(x) for x in open(path).read().split()] if os.path.exists(path) else None
rec = {"reference": "stepfun-ai/GOT-OCR2_0 chat() (original remote code, CPU F32, no_repeat_ngram 20)",
       "revision": orig["revision"], "versions": orig.get("versions"), "cases": []}
for cid in ("fox", "receipt_example", "simple_form"):
    oc = orig["cases"].get(cid)
    if oc is None:
        continue
    ref = oc["runs"]["20"]
    ref_ids = [i for i in ref["token_ids"] if i not in STOPS]
    case = {"id": cid, "reference_text": ref["text"], "reference_ids": ref_ids, "reference_secs": ref["seconds"],
            "reference_hit_cap": ref["hit_cap"], "arms": {}}
    for arm in ("r2", "q4_k", "old_f16"):
        txt = open(os.path.join(out, f"ocr.{arm}.{cid}.txt"), errors="replace").read().strip()
        ids = ids_of(os.path.join(out, "native", f"{arm}.{cid}", "gen_ids.txt"))
        ids = [i for i in ids if i not in STOPS] if ids is not None else None
        div = None
        if ids is not None and ids != ref_ids:
            div = next((k for k, (x, y) in enumerate(zip(ids, ref_ids)) if x != y), min(len(ids), len(ref_ids)))
        case["arms"][arm] = {"text": txt, "text_exact": txt == ref["text"].strip(), "ids_exact": ids == ref_ids,
                             "first_divergence": div,
                             "char_ratio": round(difflib.SequenceMatcher(None, txt, ref["text"].strip(),
                                                                         autojunk=False).ratio(), 4),
                             "n_ids": None if ids is None else len(ids), **timing.get((cid, arm), {})}
    rec["cases"].append(case)
print(json.dumps(rec, indent=1))
EOF
python - "$OUT/compare.json" <<'EOF' | tee -a "$OUT/results.txt"
import json, sys
for c in json.load(open(sys.argv[1]))["cases"]:
    print(f"case {c['id']}: original {c['reference_text'][:70]!r}")
    for a, v in c["arms"].items():
        print(f"   {a:8s} exact={v['text_exact']} ids_exact={v['ids_exact']} div={v['first_divergence']} "
              f"ratio={v['char_ratio']} secs={v.get('secs')} {v['text'][:70]!r}")
EOF
# Publication gate = the owner's bar (docs/precision-acceptance.md, F16 > 0.98): every stage of
# the stage diff vs the original-model reference above the F16 bar, and fox token-exact vs the
# original. Coordinator decision 2026-11-02 after run 41391040980 (kept on record) was rejected
# by the earlier page-similarity gate (r2 char ratio >= published q4_k on all three cases); that
# gate is still computed and recorded here, as a diagnostic only. The receipt/form gaps appear
# with the shipped q4_k too. Since 3 November 2026 both are measured against the original pipeline.
owner_ok=$(python - "$OUT/diff.r2.err" "$OUT/compare.json" <<'PYEOF'
import json, re, sys
cos = [float(m) for m in re.findall(r"cos_min=([0-9.]+)", open(sys.argv[1]).read())]
c = json.load(open(sys.argv[2]))["cases"]
fox = next((x for x in c if x["id"] == "fox"), None)
ok = len(cos) == 19 and min(cos) > 0.98 and fox is not None and fox["arms"]["r2"]["ids_exact"]
print(f"{int(ok)} stages={len(cos)} min_stage_cos={min(cos) if cos else None} fox_ids_exact={fox and fox['arms']['r2']['ids_exact']}")
PYEOF
)
sim_ok=$(python -c "import json;c=json.load(open('$OUT/compare.json'))['cases'];print(int(len(c)==3 and all(x['arms']['r2']['char_ratio']>=x['arms']['q4_k']['char_ratio'] for x in c)))")
note "owner_bar_gate=${owner_ok}"
note "page_similarity_diagnostic(r2>=q4_k on all cases)=${sim_ok} (recorded, not a publication gate)"
[ "${owner_ok%% *}" = 1 ] || fail=1

# 5. publish (new file; the old file stays untouched)
if [ "$MODE" = publish ] && [ $fail = 0 ]; then
    HF_TOKEN=${HF_TOKEN_2:?} python - "$M/$NEW" "$NEW" "$REPO" "$(git rev-parse HEAD)" <<'EOF' | tee -a "$OUT/results.txt"
import hashlib, os, sys, tempfile
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
path, name, repo, src = sys.argv[1:5]
api = HfApi(token=os.environ["HF_TOKEN"])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
local = sha(path)
readme = hf_hub_download(repo, "README.md", token=api.token)
text = open(readme).read()
text += (f"\n\n## {name} (2026-11-02)\n\nThe original `got-ocr2-f16.gguf` contains an 8 MiB block of zero bytes "
         "inside `l.embed_tokens.weight` (token rows 43-2091), so it decodes repeated `colorcolor...` on every "
         "backend. It is kept unchanged for provenance. `" + name + "` is a fresh F16 conversion with a "
         "byte-readback check; every tensor matches the original checkpoint. Use `" + name + "` for F16.\n")
rp = os.path.join(tempfile.mkdtemp(), "README.md"); open(rp, "w").write(text)
ci = api.create_commit(repo, operations=[CommitOperationAdd(name, path), CommitOperationAdd("README.md", rp)],
                       commit_message=f"Add {name}: reconverted F16 (old got-ocr2-f16.gguf has a zeroed 8 MiB "
                                      f"block in embed_tokens; kept) [CrispEmbed {src[:12]}]")
rev = ci.oid
info = api.get_paths_info(repo, [name], revision=rev)[0]
d = tempfile.mkdtemp()
rb = hf_hub_download(repo, name, revision=rev, cache_dir=d, token=api.token)
print(f"published revision={rev} file={name} local_sha256={local} lfs_sha256={info.lfs.sha256} "
      f"readback_sha256={sha(rb)} size={info.size}")
assert local == info.lfs.sha256 == sha(rb)
print("publish_verified=true")
EOF
    [ ${PIPESTATUS[0]} = 0 ] || fail=1
fi
{ echo "### GOT-OCR2 F16 r2 ($(git rev-parse --short HEAD), mode=$MODE)"; echo '```'; cat "$OUT/results.txt"; echo '```'; } > "$OUT/summary.md"
exit $fail
