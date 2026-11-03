#!/usr/bin/env bash
# Separate current-source replay, one unchanged quality document per dispatch.
set -euo pipefail
[[ $# == 1 ]] || { echo "Exactly one frozen case ID is required." >&2; exit 2; }
CASE_ID=${1:?historical-receipt, two-column-page, or inventory-table}
[[ "${GITHUB_ACTIONS:-}" == true ]] || { echo 'Use the hosted heavy-cpu workflow.' >&2; exit 2; }
[[ "${CRISPEMBED_BUILD_MODE:-portable}" == portable ]] || { echo 'Portable build required.' >&2; exit 2; }
OUT=${HEAVY_OUT:?}
SCRATCH=${HEAVY_SCRATCH:?}/lighton08-portable-environment
mkdir -p "$OUT" "$SCRATCH"
python tools/ci-heavy/lighton08_portable.py "$CASE_ID" --preflight-only > "$OUT/lighton08-preflight.json"
python - "$SCRATCH" "$OUT" <<'PY'
import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location('portable','tools/ci-heavy/lighton08_portable.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.storage_capacity(*sys.argv[1:])
info=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
assert int(info['MemAvailable'].split()[0])*1024 >= 10*1024**3, 'At least 10 GiB available memory required'
PY
python -m venv "$SCRATCH/venv"
PYTHON=$SCRATCH/venv/bin/python
"$PYTHON" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu > "$OUT/lighton08-install-torch.log" 2>&1
python - "$OUT/lighton08-requirements.txt" <<'PY'
import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location('frozen','tools/ci-heavy/ovisocr2_lighton_fixture.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pathlib.Path(sys.argv[1]).write_text(''.join(f'{name}=={version}\n' for name,version in module.VERSIONS.items() if name not in ('torch','torchvision')))
PY
"$PYTHON" -m pip install -r "$OUT/lighton08-requirements.txt" > "$OUT/lighton08-install-reference.log" 2>&1
"$PYTHON" -m pip freeze > "$OUT/lighton08-packages.txt"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/usr/bin/time -v -o "$OUT/lighton08-process-resources.txt" "$PYTHON" -u tools/ci-heavy/lighton08_portable.py "$CASE_ID"
