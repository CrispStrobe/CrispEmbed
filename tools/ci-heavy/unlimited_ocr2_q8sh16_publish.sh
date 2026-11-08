#!/usr/bin/env bash
# Publish Unlimited-OCR-2 recipe q8sh16 (Q8_0, shared experts F16; tools/unlimited_ocr2_quant_recipes.json) as an
# EXPERIMENTAL file (owner decision 8 November 2026; evidence docs/unlimited-ocr2-q4k-recipes.md round 3:
# 0.9728 CPU / 0.9762 CUDA). Quantized from the same pinned F16 the sweep used; the file size and the full
# per-tensor type census must equal the evaluated Kaggle census (tests/results/unlimited-ocr2/q4k-sweep/
# kaggle-cpu-v3/q8sh16.census.json) before upload; then Hub LFS sha256 and an empty-cache readback must match.
# Existing Hub files are untouched. args: [check|publish]
set -uo pipefail
OUT=${HEAVY_OUT:?}; S=${HEAVY_SCRATCH:?}; MODE=${1:-check}
H=tools/unlimited_ocr2_q4k.py
NAME=unlimited-ocr-2-q8_0-sh16.gguf
REPO=cstr/Unlimited-OCR-2-GGUF
EVAL=tests/results/unlimited-ocr2/q4k-sweep/kaggle-cpu-v3/q8sh16.census.json
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
cmake -S . -B "$S/b" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF \
    -DGGML_METAL=OFF -DGGML_VULKAN=OFF > "$OUT/cmake.log" 2>&1 &&
    cmake --build "$S/b" --target crispembed-quantize -j"$(nproc)" > "$OUT/build.log" 2>&1 || { tail -40 "$OUT/build.log"; exit 2; }
Q=$(find "$S/b" -name crispembed-quantize -type f -perm -u+x | head -1)
F16=$S/f16.gguf
curl -sSL --retry 5 -o "$F16" \
    "https://huggingface.co/cstr/Unlimited-OCR-2-GGUF/resolve/e3af0c0c4bfa241c161db64ca36f3c63d3ca8378/unlimited-ocr-2-f16.gguf" || exit 2
echo "487b887eddaf73e070cd98c3e10d112d21749036ca06b5e7fc103c3fb4def517  $F16" | sha256sum -c - || exit 2
mapfile -t qargs < <(python $H args q8sh16) || exit 2
"$Q" "$F16" "$S/$NAME" "${qargs[@]}" > "$OUT/quantize.log" 2>&1 || { tail -20 "$OUT/quantize.log"; exit 2; }
python $H census "$S/$NAME" "$OUT/census.json" "$F16" || exit 3
python - "$OUT/census.json" "$EVAL" <<'PY' || exit 3
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ok = a['file_bytes'] == b['file_bytes'] and a['types'] == b['types']
print(f"census_equal_to_evaluated={ok} bytes={a['file_bytes']} tensors={a['tensors']}")
sys.exit(0 if ok else 1)
PY
sha256sum "$S/$NAME" | tee "$OUT/sha256.txt"
[ "$MODE" = publish ] || exit 0
HF_TOKEN=${HF_TOKEN_2:?} python - "$S/$NAME" "$NAME" "$REPO" "$(git rev-parse HEAD)" <<'PY' | tee "$OUT/publish.txt"
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
text = open(hf_hub_download(repo, "README.md", token=api.token)).read()
text += (f"\n\n## {name} (EXPERIMENTAL, 2026-11-08)\n\nQ8_0 with the shared experts kept F16 (recipe `q8sh16`), "
         "quantized from unlimited-ocr-2-f16.gguf. Teacher-forced final-output cosine vs F16 on the four frozen cases: "
         "min 0.9728 on CPU, 0.9762 on CUDA (T4); box-stripped text equals F16 on 4/4; box coordinates can differ. "
         "Method: https://github.com/CrispStrobe/CrispEmbed/blob/main/docs/unlimited-ocr2-q4k-recipes.md\n")
rp = os.path.join(tempfile.mkdtemp(), "README.md"); open(rp, "w").write(text)
rev = api.create_commit(repo, operations=[CommitOperationAdd(name, path), CommitOperationAdd("README.md", rp)],
                        commit_message=f"Add experimental {name} [CrispEmbed {src[:12]}]").oid
info = api.get_paths_info(repo, [name], revision=rev)[0]
rb = sha(hf_hub_download(repo, name, revision=rev, cache_dir=tempfile.mkdtemp(), token=api.token))
print(f"published revision={rev} file={name} local={local} lfs={info.lfs.sha256} readback={rb} size={info.size}")
assert local == info.lfs.sha256 == rb
assert api.file_exists(repo, "LICENSE", revision=rev)
print("publish_verified=true")
PY
exit ${PIPESTATUS[0]}
