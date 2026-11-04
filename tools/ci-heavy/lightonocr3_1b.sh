#!/usr/bin/env bash
# Existing heavy-cpu.yml entrypoint: dependencies -> portable build -> controls
# -> bounded original/native lane. Large models/builds/cache stay in scratch;
# original/native arrays and failure receipts stay in the uploaded evidence tree.
set -euo pipefail
if [[ ${GITHUB_ACTIONS:-} != true || ${CRISPEMBED_BUILD_MODE:-} != portable || $(uname -s) != Linux ]]; then
    echo 'LightOnOCR-3-1B qualification requires hosted Linux and portable build mode.' >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --output|--output=*|--build|--build=*|--scratch|--scratch=*)
            echo 'The hosted wrapper owns output/build/scratch paths.' >&2; exit 2 ;;
    esac
done
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
LANE_SCRATCH=${HEAVY_SCRATCH:?}/lightonocr3-1b
BOOT_OUT=${HEAVY_OUT:?}/lightonocr3-1b-bootstrap
LANE_OUT=${HEAVY_OUT:?}/lightonocr3-1b
if [[ -e "$BOOT_OUT" || -e "$LANE_OUT" || -e "$LANE_SCRATCH/build" || -e "$LANE_SCRATCH/venv" ]]; then
    echo 'Use a fresh hosted lane; existing build/evidence is never overwritten.' >&2
    exit 2
fi
mkdir -p "$BOOT_OUT" "$LANE_SCRATCH/tmp"
export HF_HOME="$LANE_SCRATCH/hf" HF_HUB_CACHE="$LANE_SCRATCH/hf/hub"
export PIP_CACHE_DIR="$LANE_SCRATCH/pip-cache" TMPDIR="$LANE_SCRATCH/tmp"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
BOOT_STAGE=preflight
finish() {
    local code=$?
    printf '{"stage":"%s","exit_code":%d,"published":false}\n' "$BOOT_STAGE" "$code" > "$BOOT_OUT/status.json"
    exit "$code"
}
trap finish EXIT
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo 'The hosted source tree must match its immutable commit.' >&2
    exit 2
fi
git rev-parse HEAD > "$BOOT_OUT/source-commit.txt"
git submodule status > "$BOOT_OUT/submodules.txt"
python - "$LANE_SCRATCH" "$HEAVY_OUT" > "$BOOT_OUT/preflight.json" <<'PY'
import json, os, pathlib, shutil, sys
scratch, output = map(pathlib.Path, sys.argv[1:])
memory = int(next(row.split()[1] for row in pathlib.Path('/proc/meminfo').read_text().splitlines()
                  if row.startswith('MemAvailable:'))) * 1024
record = dict(memory_available_bytes=memory, scratch_free_bytes=shutil.disk_usage(scratch).free,
              output_free_bytes=shutil.disk_usage(output).free, load_average=os.getloadavg())
print(json.dumps(record, indent=2))
if memory < 12 * 2**30:
    raise SystemExit('Need at least 12 GiB available RAM before the bounded build/reference lane')
if (record['scratch_free_bytes'] < 16 * 2**30 or record['output_free_bytes'] < 8 * 2**30
        or scratch.stat().st_dev == output.stat().st_dev and record['output_free_bytes'] < 24 * 2**30):
    raise SystemExit('Need model/build scratch plus separate capacity for retained raw evidence')
PY
stage() {
    BOOT_STAGE=$1
    shift
    if "$@" > "$BOOT_OUT/$BOOT_STAGE.log" 2>&1; then
        return 0
    else
        local code=$?
        tail -80 "$BOOT_OUT/$BOOT_STAGE.log"
        return "$code"
    fi
}
stage venv python -m venv "$LANE_SCRATCH/venv"
LANE_PYTHON=$LANE_SCRATCH/venv/bin/python
stage requirements "$LANE_PYTHON" -m pip install --report "$BOOT_OUT/dependency-resolution.json" \
    -r models/requirements-lightonocr3-reference.txt
stage environment-freeze "$LANE_PYTHON" -m pip freeze --all
stage dependency-check "$LANE_PYTHON" -m pip check
stage source-audit env PYTHONPATH=tools "$LANE_PYTHON" -c 'from dump_lightonocr3_reference import verify_environment; print(verify_environment())'
stage configure cmake -S . -B "$LANE_SCRATCH/build" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_METAL=OFF \
    -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DCRISPEMBED_BUILD_SHARED=OFF
stage build cmake --build "$LANE_SCRATCH/build" --parallel 4 \
    --target test-lightonocr3-contract lightonocr3-trace
stage native-contract "$LANE_SCRATCH/build/test-lightonocr3-contract"
stage python-controls "$LANE_PYTHON" tests/test_lightonocr3_profile.py
"$LANE_PYTHON" - "$LANE_SCRATCH/build" "$BOOT_OUT" <<'PY'
import hashlib, json, pathlib, shutil, subprocess, sys
build, out = map(pathlib.Path, sys.argv[1:])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
receipt = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               binary_sha256=sha(build/'lightonocr3-trace'),
               cmake_cache_sha256=sha(build/'CMakeCache.txt'), build_mode='portable',
               native_contract_pass=True, python_controls_pass=True)
(build/'lightonocr3-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(build/'lightonocr3-build.json',out/'build.json')
shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
PY
stage qualification "$LANE_PYTHON" tools/ci-heavy/lightonocr3_1b.py \
    --build "$LANE_SCRATCH/build" --scratch "$LANE_SCRATCH" --output "$LANE_OUT" "$@"
BOOT_STAGE=complete
