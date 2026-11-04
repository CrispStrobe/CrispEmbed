# M1 fallback: LightOn3 epsilon build and controls

**Preflight reported blocked:** no contract/library/style/inference ran. Swapouts
rose8180 pages under active workloads, with about16GB swap in use. The operator
reports4 owned hashes/11 protected objects match0473dd5e8. Keep this result as a
source-preflight receipt, not Apple build evidence; do not continue on that host
without sufficient headroom. Current cache-experiment source3e82a1205 also needs
its own Apple compile/components when capacity becomes available.

This is a build-only fallback for GitHub's macOS arm64 capacity failure, not a
request to repeat model inference. Build41885714691's Apple job126770570677 never
acquired a runner; its annotation says "The job was not acquired by Runner of
type hosted even after multiple attempts". The aggregator correctly failed.
Check current PR141 checks before starting; do not duplicate a running Apple job.

Use an isolated worktree of source
`0473dd5e813fdf48d2bafbd16811271cf6c42599` from `fix/ocr-qualification-nov4`.
Fetch submodules. Read public AGENTS/CLAUDE first. Record exact source and require
all four owned-file hashes and all protected Git objects in
`docs/lighton3-pixtral-epsilon-ab-v1.json` unchanged. The candidate is0.17.15
unreleased, with the new correction gate default-off; no app pin or release.

Inspect load, `vm_stat`, available RAM, swapout counter and free disk before
building. On the16GB machine use at most two build jobs. Stop if another workload
needs the headroom or memory pressure/swapouts increase. Use task-owned scratch
outside Git. Do not install frameworks, download weights or execute OCR.

Configure a Release portable CPU build with `GGML_NATIVE=OFF`,
`CRISPEMBED_NATIVE=OFF`, CUDA/Metal/Vulkan/BLAS/Accelerate OFF and
`CRISPEMBED_BUILD_SHARED=OFF`. Build `test-lightonocr3-contract` with parallelism2
and execute it. This checks actual GGML RMSNorm against independent scalar1e-5,
unset/empty/literal0 gate behavior, active gate behavior and legacy profile
isolation. Retain complete stdout/stderr, elapsed time, peak RSS, swapout delta,
compiler identity, CMake cache and executable SHA256. A successful authored norm
control does not establish model/document quality.

Then reproduce the macos-arm64 production-library configuration in
`.github/workflows/build.yml` at that exact source (Metal embedded, Apple BLAS,
macOS12 deployment, arm64), still two build jobs. Use a separate build directory;
record all flags and hash the resulting library. Verify reported version and
architecture/install names using the workflow's commands. Run the repository's
clang-format18.1.8 style check on macOS; the Linux wheel can wrap differently,
which is why the hosted style job also uses macOS. Do not change source just to
fit an unrelated formatter version.

Return a small JSON/log report and commit it on a separate audit branch. State
which checks actually ran and any failures; keep raw build trees/caches outside
Git. This is independent Apple evidence and does not turn a failed GitHub job
green. The coordinator still needs final-head ordinary hosted controls or an
explicitly recorded independent-platform integration decision. Do not upload
models, change defaults, tag a release or update downstream pins.
