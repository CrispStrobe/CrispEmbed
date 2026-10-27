# M1 build-only handoff: smallest MoEdern prototype

This is an independent compilation and synthetic-control task for the available
M1 Mac. It does not authorize model inference or bypass the native experiment's
pending original-reference gate. Keep all build products and machine details
outside Git. No model, tokenizer, corpus, framework or release download is needed.

Use an isolated worktree at the immutable source
`79e350a096db831543dca711fd6393e61d63fa79` from
[the native branch](https://github.com/CrispStrobe/CrispEmbed/tree/feat/moedern-native-1p5m-20261026).
Its exact ggml gitlink is `890278a8342c620197c90e702e1188bcab94f510`.
Inspect load, available RAM, swap and disk before starting; use at most two build
workers and stop if memory pressure grows. Preserve unrelated work and credentials.
The original-reference and native-parity fields remain unqualified.

```sh
git fetch origin feat/moedern-native-1p5m-20261026
git worktree add --detach "$MOEDERN_SOURCE" 79e350a096db831543dca711fd6393e61d63fa79
git -C "$MOEDERN_SOURCE" submodule update --init ggml
git -C "$MOEDERN_SOURCE" rev-parse HEAD:ggml
git -C "$MOEDERN_SOURCE/ggml" rev-parse HEAD
cmake -S "$MOEDERN_SOURCE/tests/moedern-native" -B "$MOEDERN_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$MOEDERN_BUILD" --parallel 2
"$MOEDERN_BUILD/moedern-native-controls"
```

Choose task-owned fresh source/build paths for the variables. The standalone
CMake file disables Metal, CUDA, BLAS, native-ISA tuning and llamafile. It builds
the prototype, probe, weight-free controls and actual `crispembed-diff::Ref`
comparator. Do not run `native.py --phase run`, initialize a real checkpoint or
produce model embeddings in this task.

Use only the standard library for the comparator's synthetic fixtures. The
following creates three-row F32 adapters without NumPy, Torch or GGUF packages:

```sh
export MOEDERN_SOURCE MOEDERN_BUILD MOEDERN_CONTROL_OUTPUT
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
sys.path.insert(0, str(Path(os.environ['MOEDERN_SOURCE'])/'tools/moedern-colbert'))
from native_diff_adapter import write_pair
out=Path(os.environ['MOEDERN_CONTROL_OUTPUT'])
out.mkdir(parents=True,exist_ok=False)
for name,values,expected in [('same',[1.,0.,0.,1.,1.,0.],0),
                              ('changed-row',[1.,0.,0.,1.,0.,1.],1)]:
    directory=out/name;directory.mkdir()
    specs=[]
    for label,data in [('original',[1.,0.,0.,1.,1.,0.]),('native',values)]:
        path=directory/(label+'.f32');raw=struct.pack('<6f',*data);path.write_bytes(raw)
        specs.append(dict(path=path,shape=[3,2],sha256=hashlib.sha256(raw).hexdigest()))
    write_pair(directory/'original.gguf',directory/'native.gguf',
               [dict(name='control',original=specs[0],native=specs[1])])
    process=subprocess.run([str(Path(os.environ['MOEDERN_BUILD'])/'moedern-native-diff'),
        str(directory/'original.gguf'),str(directory/'native.gguf'),str(directory/'diff.json')],
        capture_output=True,text=True,timeout=30)
    (directory/'diff.log').write_text(process.stdout+process.stderr)
    assert process.returncode==expected,(name,process.returncode,process.stderr)
    report=json.loads((directory/'diff.json').read_text())
    row=report['stages'][0]
    assert row['elements']==6 and row['ggml_shape']==[2,3] and row['row_axis']==0
    assert row['passed'] is (expected==0)
    assert row['max_absolute']==(0. if expected==0 else 1.)
    assert row['min_row_cosine']==(1. if expected==0 else 0.)
print('PASS synthetic comparator positive and one-bad-row rejection')
PY
```

Acceptance is successful compilation of all targets, the actual weight-free
control pass, and comparator exit0 for equal rows/exit1 for the changed row.
Preserve complete failure logs if anything fails. Return source and dependency
SHAs, compiler/CMake versions, commands with private paths replaced by placeholders,
binary SHA256s, control outputs, both comparator JSON/logs and measured peak RSS.
Machine paths, account/access information and raw local-environment notes stay
outside public Git. Compilation/synthetic success grants no model parity,
retrieval quality, performance, tokenizer or release qualification.

The next model step remains the corrected original capture
[40423259251](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40423259251),
then an explicitly versioned native source/reference contract and hosted
same-input comparison of all intermediates, vectors, masks and MaxSim scores.
