#!/usr/bin/env bash
# Publish the two gold-corpus-evaluated GOT-OCR2 quantizations as EXPERIMENTAL files (owner decision
# 7 November 2026; evidence docs/got-ocr2-q4-research.md sections 6-8). Existing Hub files are untouched.
#   got-ocr2-q8_0-r2.gguf   Q8_0 decoder + Q8_0 vision, from F16-r2
#   got-ocr2-q8-q4k-mix.gguf Q8_0 base, decoder ffn_gate/ffn_up Q4_K (recipe q8_ffn_gu_q4k)
# Each file is quantized from the pinned F16-r2 and must reproduce the sha256 measured on the Kaggle gold
# run (tests/results/got-ocr2-q4/gold-kaggle-20261107/summary.json) before upload; then Hub LFS sha256 and
# an empty-cache readback must match. args: [check|publish]
set -uo pipefail
OUT=${HEAVY_OUT:?}; S=${HEAVY_SCRATCH:?}; MODE=${1:-check}
export CRISPEMBED_ACCEPT_TONGYI_QIANWEN_LICENSE=1
REPO=cstr/got-ocr2-crispembed-GGUF
F16REV=c0f1aa7b5b52c15002e911d5d630e6d0dcb3ba8f
F16SHA=98b431abc3e6b5715363e7296f7fc665a2eca829127ccb8693511926c4784a7f
note() { echo "$*" | tee -a "$OUT/results.txt"; }
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
B=$S/b
cmake -S . -B "$B" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
    > "$OUT/configure.log" 2>&1 || exit 1
cmake --build "$B" --parallel 4 --target crispembed-quantize > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 1; }
Q=$(find "$B" -name crispembed-quantize -type f | head -1)
M=$S/m; mkdir -p "$M"
curl -sfL --retry 3 -o "$M/f16.gguf" "https://huggingface.co/$REPO/resolve/$F16REV/got-ocr2-f16-r2.gguf" || exit 1
[ "$(sha256sum "$M/f16.gguf" | cut -d' ' -f1)" = $F16SHA ] || { note "F16 sha mismatch"; exit 1; }
set -f
"$Q" "$M/f16.gguf" "$M/got-ocr2-q8_0-r2.gguf" q8_0 > "$OUT/quant.q8.log" 2>&1 || exit 1
"$Q" "$M/f16.gguf" "$M/got-ocr2-q8-q4k-mix.gguf" q8_0 --tensor-type 'l.blk.*.ffn_gate.weight=q4_k' \
    --tensor-type 'l.blk.*.ffn_up.weight=q4_k' > "$OUT/quant.mix.log" 2>&1 || exit 1
set +f
declare -A WANT=([got-ocr2-q8_0-r2.gguf]=15920499f65b71bcfbf4d44d0e92ffbaf7f7c4523ad13f7b7bef160dc5edf098
                 [got-ocr2-q8-q4k-mix.gguf]=5ccd3d291e52cb9a0f923ccd1b7f70f7c25469382bbe7492e7373ac1bd2b73b5)
fail=0
for f in "${!WANT[@]}"; do
    got=$(sha256sum "$M/$f" | cut -d' ' -f1)
    note "$f sha256=$got evaluated=${WANT[$f]} match=$([ "$got" = "${WANT[$f]}" ] && echo yes || echo NO)"
    [ "$got" = "${WANT[$f]}" ] || fail=1
done
if [ "$MODE" = publish ] && [ $fail = 0 ]; then
    HF_TOKEN=${HF_TOKEN_2:?} python - "$M" "$REPO" "$(git rev-parse HEAD)" <<'PY' | tee -a "$OUT/results.txt"
import hashlib, os, sys, tempfile
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
m, repo, src = sys.argv[1:4]
api = HfApi(token=os.environ["HF_TOKEN"])
names = ["got-ocr2-q8_0-r2.gguf", "got-ocr2-q8-q4k-mix.gguf"]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()
local = {n: sha(os.path.join(m, n)) for n in names}
text = open(hf_hub_download(repo, "README.md", token=api.token)).read()
text += ("\n\n## Experimental quantizations from F16-r2 (2026-11-07)\n\n"
         "Research/evaluation use only under the Tongyi Qianwen RESEARCH LICENSE (see LICENSE and NOTICE).\n\n"
         "| file | recipe | size | gold corpus mean CER (55 pages) | token-exact vs F16-r2 |\n|---|---|---|---|---|\n"
         "| got-ocr2-f16-r2.gguf (recommended) | F16 | 1.44 GB | 0.363 | - |\n"
         "| got-ocr2-q8_0-r2.gguf | Q8_0 | 600 MB | 0.357 | 17/55 (3/3 on the short panel) |\n"
         "| got-ocr2-q8-q4k-mix.gguf | Q8_0, decoder ffn_gate/ffn_up Q4_K | 530 MB | 0.345 | 11/55 |\n\n"
         "Both are EXPERIMENTAL: on par with F16 on aggregate gold CER but not token-exact; individual pages can "
         "regress. They supersede got-ocr2-q4_k.gguf (kept for provenance). Method and receipts: "
         "https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/got-ocr2-q4-research.md\n")
rp = os.path.join(tempfile.mkdtemp(), "README.md"); open(rp, "w").write(text)
ops = [CommitOperationAdd(n, os.path.join(m, n)) for n in names] + [CommitOperationAdd("README.md", rp)]
rev = api.create_commit(repo, operations=ops, commit_message=f"Add experimental Q8_0-r2 and Q8/Q4_K mix [CrispEmbed {src[:12]}]").oid
d = tempfile.mkdtemp()
for n in names:
    info = api.get_paths_info(repo, [n], revision=rev)[0]
    rb = sha(hf_hub_download(repo, n, revision=rev, cache_dir=d, token=api.token))
    print(f"published revision={rev} file={n} local={local[n]} lfs={info.lfs.sha256} readback={rb} size={info.size}")
    assert local[n] == info.lfs.sha256 == rb
for n in ["LICENSE", "NOTICE"]:
    assert api.file_exists(repo, n, revision=rev), n
print("publish_verified=true")
PY
    [ ${PIPESTATUS[0]} = 0 ] || fail=1
fi
exit $fail
