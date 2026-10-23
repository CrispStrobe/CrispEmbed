# Optional M1 follow-up: actual native F32 public text API

This is complementary Apple CPU coverage while hosted Linux CI queues. The
completed original-Python four-arm M1 audit must not be repeated. This task builds
and exercises the new native public backend against the same unchanged original
arrays. It creates no release, alias, weights publication or app pin update.

## Pinned source and resource boundary

- Implementation: `0e615ea470b8af25925393abc566bbd42606d0df`,
  [draft PR74](https://github.com/CrispStrobe/CrispEmbed/pull/74).
- Hosted Linux run39939481805 is already queued; do not dispatch another copy.
- Original arrays/source0573f1847 are preserved by39913800444; use the existing
  verified local copy where available. Never recompute or replace references.
- Inspect load, `memory_pressure`, `vm_stat`, RAM and free disk first. One build/
  experiment at a time, at most two build/compute threads, with sustained memory/
  swap monitoring and an immediate stop on pressure. Retain adequate headroom
  for normal use; no Rosetta, GPU/MPS or parallel model batches.
- Use an isolated detached worktree, separate environment and task-owned results.
  Read AGENTS/CLAUDE/contributing before edits; no edits are needed for this run.

Resolve/verify the full commit against the remote branch, then create the worktree.
Initialize its pinned ggml submodule. Reuse a task environment only if its pins
match; otherwise install `gguf==0.19.0 safetensors==0.8.0 numpy==2.2.6
huggingface-hub==1.33.0`. No Torch/Transformers or original-Python forward is needed.
Set `GITHUB_SHA` to the exact worktree commit, `CRISPEMBED_ULTRABERT=1`,
`CRISPEMBED_FORCE_CPU=1`, and all OMP/OpenBLAS/MKL/VecLib thread limits to2.

## Prepare and build

Use `prepare_frozen_artifacts` from `tools/ci-heavy/ultrabert_native.py` with the
reviewed `tests/results/ultrabert/reference-audit.json`, verified preserved
artifact directory and a new task scratch directory. This checks original
manifest/all payloads and regenerates byte-identical F32/F16 conversions from
the pinned checkpoint. Both conversions are needed to test F16 rejection.
Its imports require `tools` and `tools/ci-heavy` on `PYTHONPATH`.

Configure an isolated build with:

```sh
cmake -S . -B ../ultrabert-f32-native-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_METAL=OFF
cmake --build ../ultrabert-f32-native-build \
  --target test-ultrabert-profile test-ultrabert-api --parallel 2
python3 tests/test_ultrabert_api_log.py -v
../ultrabert-f32-native-build/test-ultrabert-profile
```

Do not run the full Linux `ultrabert_api.py` driver unmodified: its shared-library
consumer step selects a Linux `.so`. The native test above is portable. No local
source patch is necessary for the bounded Apple checks below.

## Native checks and return evidence

1. Run `test-ultrabert-api MODEL-F32 --reject` in separate unset and literal0
   gate processes; both must return0 because rejection is expected. Run with
   gate1 and `MODEL-F16 --reject`; it must also reject before weight allocation.
2. For each frozen case except `pair`, write the exact original text(s) to UTF-8
   files. For the query case, prepend the exact `QUERY` constant from
   `tools/ultrabert_contract.py`. Preserve Unicode/newlines and the empty file.
   Use every row of the original padded-batch case in its existing order.
3. Call `test-ultrabert-api MODEL-F32 ORIGINAL-CASE.gguf TEXTFILE...` once per
   case with gate1. Six cases/seven rows must complete. The test compares actual
   text-derived IDs with original active IDs, raw/normalized pooled outputs with
   `crispembed-diff`, and batch/repeat/context/prefix/overflow/failure contracts.
   Every numeric pooled check retains cosine0.99999/maxabs0.002.
4. Preserve stdout/stderr and exit status for each command. Parse pooled logs
   with `parse_pooled` in `tools/ci-heavy/ultrabert_api.py`; require twelve pooled
   comparisons and seven ID rows, with every failed comparison exiting nonzero.
   Include exact source/tool/native-file hashes, original archive/model hashes,
   compiler/CMake/ggml/CPU/thread fingerprints, wall time, peak RSS and swap change.
5. Return only small JSON/log summaries with hashes. Keep checkpoint/GGUF/build/
   archive payloads local or remote, outside Git. Public reports must remove
   machine/access/operator paths, retaining unmodified private raw copies/hashes.

This is F32 pooled-output qualification only. The public pair API, any input containing the literal `<|` prefix, F16/quantized/GPU/Matryoshka, independently scored retrieval
and official SentenceTransformer behavior remain open. The original full-stage
183/1294 native failures and original strict replay reports stay unchanged.
Python/CLI/HTTP checks belong to the hosted Linux run; model-backed Rust/Dart
checks are separately pending. Report genuine failures and preserve initial
logs before considering one bounded correction; never fit a new budget.
