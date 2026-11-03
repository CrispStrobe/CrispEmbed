# PR 118 stream hardening

## Source and initial findings

The review covers [PR118](https://github.com/CrispStrobe/CrispEmbed/pull/118) at
`60f0b6cf4657cb118049b98770e63736719049a8`, based on main
`a35607729de9fc267190c6db6a5fb6f1f8cfbe66`. The maintainer continuation preserves
the original contribution and all experimental engine and stream gates.

The initial lightweight command `bash tools/check_test_clean_exit.sh` exited 1,
naming only `tests/test_inpaint_temporal.cpp`. Its new main lacked the established
`core_util::clean_exit` wrapper. This is an observed source-guard failure; no GPU
teardown crash was claimed for the standalone temporal target.

The initial temporal counterexamples were calculations of the exact source
predicates, not native output or model benchmarks:

- An unchanged mask over a textured background brightened by one level per frame
  has adjacent ring mean/p90 of 1, below the static threshold 3. Recursive reuse
  can retain the initial fill even after the background has drifted by 50 levels.
- For the existing 192 by 64 test geometry and mask at x=[20,76), y=[40,52),
  one-pixel stripes alternating 50 and 200, translated by one pixel, give 72
  usable cells with mean/p90 both zero. Pixelwise ring absolute error is 150.
- At 3840 by 2160 and requested window 120, RGB-only sizing selects window 42.
  The 85 RGB inputs occupy 2017.09 MiB; a 3200 by 200 cropped mask per frame
  increases inputs alone to 2068.97 MiB. Full-frame masks raise that to
  2689.45 MiB, before outputs, scratch or the model.
- At EOF, `!finish() || !drain()` short-circuiting can discard earlier completed
  tail frames when a later engine invocation fails. Two fully masked 9 by 7
  records, window 2 and the existing driver's `INPAINT_TEST_FAIL_AT=2` isolate it.
- The workflow's `src/inpaint.*` pattern does not match either temporal source
  file. Current PR changes happen to select the workflow through other paths.

The initial raw-stream order test used three identical records and binarized its
input masks before invoking the CLI; it could not discriminate masked-frame
reordering or test the raw stream's 127/128 threshold.

## Acceptance scope

Runtime changes must retain exact known pixels, preserve independent-engine mode
when temporal mode is disabled, retain successfully completed output on failures,
and bound reuse against its originating source. Buffer documentation must identify
the accounted payload and the additional scratch, CLI and model allocations.
Resource/backpressure errors must have explicit lifecycle behavior.

Model-free tests cover the demonstrated defects and the stream's ordering, donor,
mask, border, lifecycle and failure boundaries. The owned video panel separately
records paired independent/temporal image and temporal errors, engine counts,
elapsed time and available process memory data with source/model/fixture hashes.
Fake-engine execution is harness evidence only. A small synthetic CPU panel does
not qualify representative film, GPU, every engine, or real-time performance.

## CI baseline

The original PR revision had nine workflows queued when inspected. No duration
claim was inferred for those queued jobs. Recent completed runs on other commits
locate the current cost:

| Source run | Work | Measured execution |
| --- | --- | --- |
| [41590661810](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41590661810), `0bbd3ff9` | inpainting macOS/Linux/Windows native compilation | 230 / 400 / 532 seconds |
| Same run | Python contracts | 7 / 10 / 12 seconds |
| Same run | native and CLI checks | 1 / less than 1 at reported resolution / 2 seconds |
| [41588592364](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41588592364) | Rust workspace / extracted package builds | 416 / 415 seconds |

The inpainting jobs began about 72–75 minutes after that run was created. These
are observed samples from existing CI, not a measured speedup of the changes here.
Routing must avoid unnecessary builds without skipping shared dependencies,
incomplete change lists, package verification or required final-result failures.

## Execution record

The maintainer continuation implements anchor-relative reuse, per-pixel residual
checks, RGB-aware flatness, a strict harmonic iteration ceiling, conservative
frame-payload sizing, recoverable output backpressure, and CLI failure-prefix
draining. There are 18 temporal scenario functions plus expanded stream CLI
controls. The corrected source guard reports all 239 one-shot mains compliant;
the initial failure above remains part of the record.

Focused local controls passed: 32 selector/git-diff/aggregate/cache/invariant tests
and 13 quality-oracle/process/evidence tests. Actionlint 1.7.12 passed all seven
changed/new workflows; formatting passes clang-format 18.1.8 on the four touched
C++ files. The existing workflow-pin guard cannot run in this local environment
because its Bash process substitution requires `/dev/fd`; it remains a hosted
guard, without a claimed local pass.
Native builds, native test execution, the real-model panel, and final-source
hosted checks remain pending. Source preparation and fake outputs are not
native/model qualification. PLAN tracks integration, and this record will name
the exact hosted source and runs once available.
