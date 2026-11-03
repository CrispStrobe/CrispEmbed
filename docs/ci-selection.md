# Change-aware ordinary CI

The ordinary Build, OCR WASM, embedding WASM, Rust crates and inpainting CLI
workflows share [`tools/ci_selection.py`](../tools/ci_selection.py). Every PR to
`main` and every push to `main` starts a small selection job. Only selected
builds start runners or fetch their native dependencies. A final result job
checks that every selected job succeeded and that every skipped job was
explicitly unselected.

This routing does not change the source-pinned model qualification, evidence,
publication, release or preservation workflows. Their existing triggers and
acceptance rules remain in those workflows.

## What runs

| Changes | Selected expensive work |
| --- | --- |
| Markdown only | None; cheap guards, selection and result checks complete |
| Reviewed inpaint Python helpers, controls or JSON contracts | Python contracts on Linux, macOS and Windows; no submodules, CMake or native build |
| `src/inpaint*`, `src/migan.*`, `src/latent_inpaint*`, inpaint CLI source | Three desktop native builds plus three-platform inpaint native/CLI contracts; no mobile, sibling, WASM or Rust package builds |
| Inpaint native/CLI tests or the synthetic video harness | Three-platform inpaint native/CLI contracts |
| OCR WASM wrapper, build script or browser tests | OCR WASM build, Node tests and real browser smoke |
| Embedding WASM wrapper or build script | Embedding WASM build |
| Rust crates, workspace manifests or vendoring script | ARM64 C ABI type check, Rust source build and independently verified packaged crate |
| Root CMake, `cmake/`, `src/core/`, `ggml` gitlink or its contents, `.gitmodules`, version or license | All ordinary lanes |
| Selector/cache implementation or selector tests | All ordinary lanes |
| An ordinary workflow itself | All of that workflow's jobs |
| An unknown path or incomplete change detection | All ordinary lanes |
| Manual dispatch | All lanes of the dispatched workflow |

The rules form a union: a change to an inpaint source and an OCR WASM wrapper
runs both relevant groups. The allow-list is deliberately small. An unfamiliar
source, fixture, workflow or machine-readable document gets full validation
until its dependencies are reviewed. In particular, `docs/**` is **not** an
ignore rule. `docs/inpaint-publication-v1.json`, the latent and MI-GAN512
contracts, and `tests/results/migan/original-reference.json` are executable
inputs to the Python contracts and select those contracts. Other unclassified
receipts also fall back to full CI.

The CLI-only exception follows the current CMake graph: these inpaint libraries
are linked into the CLI and their tests, not `crispembed-shared` or either WASM
library. Routing tests check that relationship so a later library integration
requires updating the map. Android now builds `crispembed-shared`, the library
it uploads, rather than host CLI and test targets. Desktop runs for isolated
CLI edits build the installed CLI/tools, shared library and backend smoke target; shared or
unknown changes retain the complete native build and model-free test suite.

The all-platform Flutter bundle is produced only when desktop, Android and iOS
libraries were all selected. A desktop-only run still uploads its desktop
bundles; it does not assemble a partial mobile bundle.

## Complete comparisons and conservative failure

PR selection uses the merge-base-to-head diff (`base...head`), including all
commits in the PR. A docs-only final commit cannot hide an earlier native edit.
Unrelated changes that landed on the base branch are excluded. Push selection
uses the event's complete `before..after` comparison, including multi-commit
pushes, rather than `HEAD^`.

The detector uses Git directly, so GitHub path-filter file limits do not discard
paths. A NUL-delimited name/status parser handles whitespace and newlines in
filenames, deleted paths and both names of renamed/copied files. The exact
commit IDs come from the event and are validated before any fetch. A bounded
fetch fills ordinary shallow-checkout gaps. Missing/unavailable objects,
unresolved merge bases, invalid events, decoding errors or failed Git commands
select every lane and print a warning; they never produce an empty selection.

Every selection appears in the run's step summary. Preview a case locally:

```sh
python3 tools/ci_selection.py classify src/inpaint_temporal.cpp docs/inpainting-cli.md
python3 tools/ci_selection.py classify tools/prepare_migan_inpaint.py
python3 tools/ci_selection.py classify CMakeLists.txt
python3 tests/test_ci_selection.py -v
```

Root `CMakeLists.txt` intentionally still selects everything. A PR adding a new
target there, including the original stream PR, receives full proof. Moving
unrelated CMake sections into smaller modules is a possible separate refactor;
the selector does not infer safety from individual changed CMake lines.

## Stable result checks and cancellation

These result names are suitable for required status checks:

- `Build result`
- `OCR WASM result`
- `Embedding WASM result`
- `Rust crates result`
- `Inpainting result`

Their workflows have no top-level path exclusions. Each result job uses
`if: always()` and verifies the selector plus all its conditional dependencies.
A failed/cancelled selector, missing output, selected job that was skipped,
failed/cancelled job, or unchecked dependency fails the result. This avoids
treating a skipped required job as proof that its work succeeded. Repository
branch-protection settings are a separate configuration; this change does not
modify them. GitHub documents
[job-level skips](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-jobs-with-conditions)
and [required-check behavior](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/troubleshooting-required-status-checks).

Superseded PR runs cancel because each newer PR run rechecks the cumulative
diff. Main pushes and manual dispatches use a unique run ID in their concurrency
group. This is essential: a docs-only main push must not cancel the previous
code-bearing push and then skip its still-unvalidated build. Ordinary inpaint
and lint feature pushes no longer duplicate the validation already triggered
by their PR; main pushes still receive independent merged-source checks.

## Compiler cache

The shared [`compiler-cache` action](../.github/actions/compiler-cache/action.yml)
uses pinned sccache binaries and GitHub's persistent compiler-object cache for
native and Rust source builds. Cache namespaces separate operating system,
architecture, target/backend configuration and cache revision. Sccache also
keys compiler identity, inputs and compiler flags; a changed toolchain or
configuration cannot reuse an incompatible object. Its
[official CMake/MSVC guidance](https://github.com/mozilla/sccache#usage) and
[GitHub cache backend](https://github.com/mozilla/sccache/blob/main/docs/GHA.md)
describe the mechanism.

The Linux Rust and sibling builds retain their existing host-native defaults.
Their cache namespace additionally includes stable CPU vendor/family/model,
stepping and instruction flags, because the literal `-march=native` option can
mean different instructions on different runners. Frequency, processor index
and flags ordering do not change the signature. Missing or heterogeneous CPU
identities disable compiler caching for that job; they never share an
"unknown CPU" cache.

Every configure, link, test, install and package verification still runs. Build
directories and Cargo `target/` directories are not restored. In particular,
`cargo package --allow-dirty` still configures and links its unpacked vendored
sources independently; a cached in-tree library cannot replace that proof.
Rust's native build gets an explicit four-job parallelism limit.

Windows uses MSVC with Ninja Multi-Config because CMake's Visual Studio
generator ignores compiler launchers. This retains the existing `Release/`
paths. The portable-CPU guard scans all generated `.ninja` files, including
`CMakeFiles/impl-Release.ninja`, so the generator change does not bypass the
existing ISA-baseline check. A focused test plants a prohibited `/arch:AVX512`
flag in that included file and checks that the guard rejects it.

Cold-cache runtime and cache hit rates need hosted measurements. This change
does not claim a measured speedup or use cache hits to skip validation.

## Video regressions and bounded real-model evidence

The inpainting workflow runs cheap Python controls before fetching submodules.
For native changes it builds the small temporal and fake-driver targets first,
runs the temporal/CLI regressions and deterministic synthetic video harness,
and uploads the synthetic reports before starting the full production CLI
build. The remaining actual-CLI and image contracts still run on all three
platforms.

[`inpaint-video-quality.yml`](../.github/workflows/inpaint-video-quality.yml)
is a separate bounded evidence lane. Harness, harness-test, fixture or workflow
changes trigger it; ordinary runtime edits can request it by manual dispatch.
It reproduces the already reviewed MI-GAN256 F32 model, checks its SHA-256 and
compares independent and temporal processing against clean authored frames.
There are five six-frame clips at 96×64: 30 distinct inputs and 60 processed
frames across both arms, with two CPU threads, per-invocation deadlines and a
30-minute job limit. Reports, raw inputs/outputs, timings, hashes, model
replication records and failed-case logs remain downloadable as run artifacts.
Source/helper identities are recorded before setup, and model preparation
stdout/stderr are retained even if preparation fails before writing its receipt.
Numerical regressions are reported explicitly; successful synthetic transport
and contract checks do not by themselves qualify general video quality.
