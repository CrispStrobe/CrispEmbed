#!/usr/bin/env bash
# INTERNAL RESEARCH ONLY: preserve the exact hosted-tested Big-LaMa GGUFs in the owner's
# PRIVATE Hugging Face repo (owner decision 2026-11-01; checkpoint rights unresolved).
#
#   gh workflow run heavy-cpu.yml --ref feat/issue71-five-engines \
#       -f script=tools/ci-heavy/lama_quant_publish_private.sh -f isolated_lane=lama-quant-publish
#
# Rebuilds every file exactly as lane 41309379208 did (official archive -> converter ->
# crispembed-quantize lama-v2 recipes on x86-64; Q4_K bytes are platform-dependent, so they
# are not produced elsewhere), requires each SHA-256 to equal the tested one, and uploads
# only to an EXISTING repo that the API reports private, re-checking privacy afterwards.
# It never creates a repo, never changes visibility, and never uploads workflow artifacts.
set -euo pipefail
OUT=${HEAVY_OUT:?}
S=${HEAVY_SCRATCH:?}
REPO=cstr/big-lama-research-GGUF
ZIP_URL=https://huggingface.co/smartywu/big-lama/resolve/05cb2be7f8dbe6ca7c6e78f4fc827a4b2baaa4a9/big-lama.zip
ZIP_SHA=f1b358ca24093b93a106183b98a3dea6e8ed09f3b43ea7251eb2c81e7b4575f6
LAMA_SRC=786f5936b27fb3dacd2b1ad799e4de968ea697e7
# path in repo | local file | requested | overrides | tested sha256 (lane 41309379208)
FILES=(
  "big-lama-f32.gguf|big-lama-f32.gguf|||24736f267c68e013f0000ccb69560c4353b6a6d7983fc3cc2537d9f2e03f304c"
  "big-lama-f16.gguf|big-lama-f16.gguf|f16||31d5a0c4dae9a0d638a5b75bb8921ea7e588e35c8c00060e4ce16fc0bc0d3f3b"
  "big-lama-q8_0.gguf|big-lama-q8_0.gguf|q8_0||c01175e15445494220480fe2a5ceb4f6d34cffa293675892b5217bfff34c1f42"
  "big-lama-q4_k-s8e8.gguf|big-lama-q4_k-s8e8.gguf|q4_k|generator.*.convg2g.*=q8_0 generator.model.2.*=q8_0 generator.model.3.*=q8_0 generator.model.4.*=q8_0|ce64f5895fb8da51761e3a79aefc7aac1e2860f310fe01185928c0b299b44803"
  "failed-owner-bar/big-lama-q4_k.gguf|big-lama-q4_k.gguf|q4_k||881b99ff035411d18d0d2b0790394fbbd64a79ea2249f18c3b15808a49a4c039"
)
[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN secret unavailable" >&2; exit 1; }
echo "source=$(git rev-parse HEAD)" > "$OUT/provenance.txt"
python -m pip install -q huggingface_hub==0.35.3 PyYAML==6.0.2 numpy==2.1.3
python - "$REPO" <<'PY'
import os, sys
from huggingface_hub import HfApi
info = HfApi(token=os.environ["HF_TOKEN"]).model_info(sys.argv[1])
if info.private is not True:
    sys.exit("refusing: target repo is not private")
print("pre-upload: repo exists and is private")
PY
M=$S/models; mkdir -p "$M"
curl -sfL -o "$S/big-lama.zip" "$ZIP_URL"
echo "$ZIP_SHA  $S/big-lama.zip" | sha256sum -c -
git init -q "$S/src" && git -C "$S/src" remote add origin https://github.com/advimman/lama.git
git -C "$S/src" fetch -q --depth 1 origin $LAMA_SRC && git -C "$S/src" checkout -q --detach FETCH_HEAD
python models/convert-lama-to-gguf.py --archive "$S/big-lama.zip" --source-dir "$S/src" --output "$M/big-lama-f32.gguf"
cmake -S . -B "$S/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_BLAS=OFF \
    -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF \
    -DCRISPEMBED_BUILD_SHARED=OFF > "$OUT/configure.log" 2>&1
cmake --build "$S/build" --target crispembed-quantize -j4 > "$OUT/build.log" 2>&1
Q=$(find "$S/build" -name crispembed-quantize -type f | head -1)
set -f
for row in "${FILES[@]}"; do
    IFS='|' read -r remote local type overrides sha <<< "$row"
    if [ -n "$type" ]; then
        qa=(); for o in $overrides; do qa+=(--tensor-type "$o"); done
        "$Q" "$M/big-lama-f32.gguf" "$M/$local" "$type" "${qa[@]}" > "$OUT/quantize-$local.log" 2>&1
        python tools/lama_quant_census.py --source "$M/big-lama-f32.gguf" --variant "$M/$local" --requested "$type" \
            "${qa[@]}" --output "$OUT/census-$local.json" > /dev/null
    fi
    actual=$(sha256sum "$M/$local" | awk '{print $1}')
    [ "$actual" = "$sha" ] || { echo "$local sha $actual differs from tested $sha; refusing upload" >&2; exit 1; }
    echo "$remote $sha $(stat -c %s "$M/$local")" >> "$OUT/upload-plan.txt"
done
set +f
python - "$REPO" "$M" "$OUT/upload-plan.txt" "$OUT/upload-receipt.json" <<'PY'
import json, os, sys
from huggingface_hub import HfApi
repo, models, plan, receipt = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
local = {"big-lama-f32.gguf": "big-lama-f32.gguf", "big-lama-f16.gguf": "big-lama-f16.gguf",
         "big-lama-q8_0.gguf": "big-lama-q8_0.gguf", "big-lama-q4_k-s8e8.gguf": "big-lama-q4_k-s8e8.gguf",
         "failed-owner-bar/big-lama-q4_k.gguf": "big-lama-q4_k.gguf"}
rows = [line.split() for line in open(plan)]
for remote, sha, size in rows:
    if api.model_info(repo).private is not True:
        sys.exit("refusing: repo is no longer private")
    api.upload_file(path_or_fileobj=os.path.join(models, local[remote]), path_in_repo=remote, repo_id=repo,
                    repo_type="model", commit_message=f"Internal research: {remote} (sha256 {sha})")
    print("uploaded", remote, flush=True)
info = api.model_info(repo, files_metadata=True)
files = {s.rfilename: {"size": s.size, "lfs_sha256": (s.lfs or {}).get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)}
         for s in info.siblings}
checks = [{"path": r, "expected_sha256": h, "expected_bytes": int(n), "hub": files.get(r),
           "match": bool(files.get(r)) and files[r]["lfs_sha256"] == h and files[r]["size"] == int(n)} for r, h, n in rows]
json.dump({"repo": repo, "private_after_upload": info.private, "revision": info.sha, "files": checks},
          open(receipt, "w"), indent=2)
print(json.dumps({"private": info.private, "all_match": all(c["match"] for c in checks)}))
if info.private is not True or not all(c["match"] for c in checks):
    sys.exit(1)
PY
