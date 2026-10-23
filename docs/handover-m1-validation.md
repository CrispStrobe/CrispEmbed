# M1 16GB: completed UltraBERT original-projection audit

This task is authorized for the available M1 machine. It supplements the Linux
reference-reproducibility investigation; it does not substitute Apple outputs
for the frozen original arrays or qualify production UltraBERT support.
Read [AGENTS](../AGENTS.md), [the executable handoff](next-agent-handoff.md),
[the architecture contract](ultrabert.md) and [contributing](contributing.md).
Machine/access details belong in local ignored notes, never in this document.

## Completed result — do not repeat these four arms

[Returned summary](../tests/results/ultrabert/m1-validation.json),
[full original-module report](../tests/results/ultrabert/reproducibility-m1.json)
and [independent structural/hash review](../tests/results/ultrabert/m1-independent-review.json)
are integrated. Four arms/seven cases take63.45s,1.90GiB peak RSS, zero new system
swapouts. Sources/selected weight bits/same-input hashes match Linux. All arms
are identical; MKLDNN is unavailable. Nonempty cases first differ at RoPE
cosine5.96e-8; worst final-hidden0.005791664 exceeds the original0.002 audit cap.
Raw CLS max6.52e-5 and normalized6.41e-7 remain close. This is Python-versus-frozen
Python reproducibility, not new C++ parity or independent task accuracy.

Next preserve the unchanged audit, and scope F32 pooled-output/API/retrieval
qualification separately. Do not spend another Mac run repeating threads/MKLDNN
or regenerate the original anchors. The instructions below document the completed
experiment and its immutable evidence. Further local work needs a new bounded
handover and fresh resource inspection.

## Scope and immutable sources

- Original producer38997562520, source `0573f1847db68914052a98341a22fceeb1358518`.
- Verified copy39913800444: artifact
  `ultrabert-original-38997562520-preserved-39913800444`, expiry21 January2027.
  [Inventory](../tests/results/ultrabert/reference-preservation.json).
- Portable diagnostic source `f0dbe58d441db7f52269ff03d49af4749d57ef8b`.
  Verify this commit against remote before execution; never silently substitute
  arbitrary main. The pinned HF checkpoint and dependency versions are checked
  by the script itself.
- Prior Linux diagnostic39915714794 at8031ae1c retains all four backend/thread
  alternatives. All seven cases first differ at layer0 Q, with287 differing
  captured stages per case. MKLDNN changes nothing at fixed thread count;
  threads1/2 change document outputs but recover none of the frozen references.

## Prepare and run one job

First inspect `uptime`, `memory_pressure`, `vm_stat`, `sysctl hw.memsize` and
`df -h`. Run one task at a time with at most two compute threads. Keep adequate
memory/disk headroom for the operator; stop rather than inducing heavy swapping.
Do not remove another project's files or alter its Python environment.

From a local CrispEmbed checkout, use a detached isolated worktree and separate
Python3.12 environment. Resolve the full source commit from the public source
above before creating the worktree. Example portable commands:

```sh
git fetch origin main
git worktree add --detach ../crispembed-m1-audit f0dbe58d441db7f52269ff03d49af4749d57ef8b
cd ../crispembed-m1-audit
python3.12 -m venv ../crispembed-m1-python
. ../crispembed-m1-python/bin/activate
python -m pip install gguf==0.19.0 numpy==2.2.6 safetensors==0.8.0 huggingface-hub==1.33.0
python -m pip install torch==2.14.0 transformers==5.16.1
mkdir -p ../crispembed-m1-results/frozen
gh run download 39913800444 --repo CrispStrobe/CrispEmbed --name ultrabert-original-38997562520-preserved-39913800444 --dir ../crispembed-m1-results/frozen
python tools/ci-heavy/preserve_ultrabert_reference.py ../crispembed-m1-results/frozen ../crispembed-m1-results/preservation.json
export GITHUB_SHA="$(git rev-parse HEAD)"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/bin/time -l python tools/ci-heavy/ultrabert_reproducibility.py ../crispembed-m1-results/frozen ../crispembed-m1-results/reproducibility-m1.json
```

The dependency pins are mandatory. If the native arm64 Python/Torch wheel is
unavailable, record the exact installer failure and stop; do not use Rosetta,
upgrade Torch or change precision/reference inputs to get a result. Do not move
the model to MPS: this experiment is original CPU projection reproducibility.
The diagnostic itself performs no native build and requires no production-model
registry integration. Large reference/model files remain local task outputs or
remote artifacts, outside Git.

## Acceptance and return evidence

Require the preserved manifest hash, all seven archive hashes/2730 tensor payloads,
original conversion records and loaded selected-parameter bits to verify first.
Require four complete arms × seven cases ×386 forward-stage records, explicit raw/normalized CLS comparisons and all
same-input Q/gate/up/down records; original inputs and weights must remain fixed.
Record whether MKLDNN is actually available on this platform. An unavailable
backend is a duplicated control, not distinct backend coverage.

Compare first differing stages, absolute errors/RMS/norms, raw/normalized CLS
boundaries and payload hashes with the Linux report. Distinguish bit-exact frozen
replay, original same-input projection, F64 diagnostic and F16-weight diagnostic.
The prior Linux report has final-hidden metrics but no separate CLS metrics;
the updated portable script explicitly records both final CLS comparisons.
A successful diagnostic exit means measurement completed; it does not mean the
287 differences or existing strict native failures passed. Keep every failure
and original0.99999/0.002 budget. No new references or threshold fitting.

Return only the small diagnostic/preservation JSON and command/source/dependency/
architecture/memory/timing evidence, with hashes. Strip local paths and operator
identities from public provenance; preserve an unmodified private raw copy and
its hash when sanitization is necessary. Review before committing public reports.
Update HISTORY/PLAN and the shared handoff with exact scope and remaining gaps.
Do not publish weights, create release tags, alter the app's0.17.12 pin or merge
PR60 as part of this task.

PosFormer PR72's macOS50-case public API/gate validation already passed at0b73f5f9;
inspect the latest Linux reference/format/platform conclusions before deciding
whether another M1 build adds evidence. Avoid duplicate full runs.
