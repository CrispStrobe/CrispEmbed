#!/usr/bin/env bash
# Processor-only original/native comparison; never download model weights.
set -euo pipefail
[[ "${GITHUB_ACTIONS:-}" == true ]] || { echo 'Run through heavy-cpu.yml on a hosted runner.' >&2; exit 2; }
[[ "${CRISPEMBED_BUILD_MODE:-portable}" == portable ]] || { echo 'Portable mode is required.' >&2; exit 2; }
OUT=${HEAVY_OUT:?}
SCRATCH=${HEAVY_SCRATCH:?}/nemotron-parse-frontend
mkdir -p "$OUT" "$SCRATCH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - "$SCRATCH" "$OUT" <<'PY'
import json,pathlib,shutil,subprocess,sys
scratch,out=map(pathlib.Path,sys.argv[1:])
free=shutil.disk_usage(scratch).free
assert free >= 8*1024**3, 'At least 8 GiB free scratch is required for isolated CPU dependencies and reference arrays'
(out/'preflight.json').write_text(json.dumps({'runtime_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'build_mode':'portable','free_scratch_bytes':free,'model_weights_downloaded':False,'model_execution':False},indent=2)+'\n')
PY
python -m venv "$SCRATCH/venv"
PYTHON=$SCRATCH/venv/bin/python
"$PYTHON" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu > "$OUT/install-torch.log" 2>&1
"$PYTHON" -m pip install -r tools/nemotron_parse_frontend-requirements.txt > "$OUT/install-reference.log" 2>&1
"$PYTHON" -m pip freeze > "$OUT/packages.txt"
"$PYTHON" - "$SCRATCH/source" <<'PY'
import hashlib,importlib.util,pathlib,sys,urllib.request
spec=importlib.util.spec_from_file_location('frontend','tools/nemotron_parse_frontend.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
out=pathlib.Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
for name,expected in module.SOURCES.items():
    url=f'https://huggingface.co/{module.REPOSITORY}/resolve/{module.REVISION}/{name}'
    with urllib.request.urlopen(url,timeout=90) as response:
        data=response.read(2*1024*1024+1)
    assert len(data)<=2*1024*1024 and hashlib.sha256(data).hexdigest()==expected, 'Small source hash differs: '+name
    (out/name).write_bytes(data)
PY
cmake -S . -B "$SCRATCH/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF > "$OUT/cmake.log" 2>&1
cmake --build "$SCRATCH/build" -j2 --target test-nemotron-parse-geometry > "$OUT/build.log" 2>&1
cp "$SCRATCH/build/CMakeCache.txt" "$OUT/CMakeCache.txt"
"$SCRATCH/build/test-nemotron-parse-geometry" self > "$OUT/native-controls.json"
"$PYTHON" tools/nemotron_parse_frontend.py capture --source-directory "$SCRATCH/source" --out "$OUT/original-frontend" > "$OUT/original.log" 2>&1
"$PYTHON" tools/nemotron_parse_frontend.py compare --native "$SCRATCH/build/test-nemotron-parse-geometry" --reference-directory "$OUT/original-frontend" --out "$OUT/native-frontend" > "$OUT/comparison.log" 2>&1
"$PYTHON" - "$OUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1]);report=json.loads((out/'native-frontend/comparison.json').read_text())
(out/'summary.md').write_text('# Nemotron processor frontend\n\n'
    +f"Frozen cases: {len(report['cases'])}; geometry/packing pass: {report['frontend_parity_pass']}.\n\n"
    +'No neural model, native resize, OCR or publication qualification. Original and native arrays are retained in this hosted artifact.\n')
PY
