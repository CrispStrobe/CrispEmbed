#!/usr/bin/env bash
# Portable quantizer artifact only; no model access and no service writes.
set -euo pipefail
[[ "${GITHUB_ACTIONS:-}" == true ]]
[[ "${CRISPEMBED_BUILD_MODE:-}" == portable ]]
build="$HEAVY_SCRATCH/lighton4b-quantizer-build"
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF 2>&1 | tee "$HEAVY_OUT/configure.log"
cmake --build "$build" --target crispembed-quantize --parallel 2 2>&1 | tee "$HEAVY_OUT/build.log"
cp "$build/crispembed-quantize" "$HEAVY_OUT/crispembed-quantize"
ldd "$HEAVY_OUT/crispembed-quantize" > "$HEAVY_OUT/ldd.txt"
! grep -E 'libggml|not found' "$HEAVY_OUT/ldd.txt"
"$HEAVY_OUT/crispembed-quantize" > "$HEAVY_OUT/usage.txt" 2>&1 && exit 1 || test "$?" -eq 1
python - <<'PY'
import hashlib,json,os,pathlib,subprocess
out=pathlib.Path(os.environ['HEAVY_OUT'])
files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}
receipt={'schema':'lighton4b-portable-quantizer-build/v1','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'submodules':subprocess.check_output(['git','submodule','status','--recursive'],text=True).splitlines(),
 'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0],
 'run_id':os.environ['GITHUB_RUN_ID'],'files':files,'model_accessed':False}
(out/'quantizer-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
