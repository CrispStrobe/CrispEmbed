# PixelHacker CPU execution record — 1 November 2026

PixelHacker completed one cold and three warm calls through the ordinary native
image API on the pinned 512×512 `formula_photo_bar` fixture. The candidate used
F32 weights and the stock diffusion schedule. Its four output hashes were
identical, and every known input RGB channel was preserved. This is a completed
CPU execution and repeatability check, **not original PixelHacker parity or a
perceptual-quality qualification**. The original CUDA/FLA implementation was not
executed on this CPU-only host.

The user then requested immediate delivery. The eager comparison was safely
cancelled after one cold and one warm call; the next call was interrupted. The
portable graph comparison was not started. Consequently, this panel supplies
**no completed cross-implementation RGB comparison and no speedup claim**.
The [public evidence record](../tests/results/inpainting/pixelhacker-cpu-2026-11-01.json)
retains the complete candidate evidence, partial eager stdout, cancellation
receipt, original reporting failure, independent revalidation, and all negative
qualification flags.

## Completed candidate measurements

The host was an AMD EPYC 9V74 with an eight-CPU quota, nine CPUs in its affinity
mask, and an 8 GiB memory limit. The process requested eight threads and had a
7.5 GiB address-space cap. No CUDA or Metal execution was available. All numbers
below are from the preserved native summary and process receipt; loading is
separate from image timing.

| Measurement | Recorded result |
|---|---:|
| Native model initialization and digest validation | 14.269750 s |
| Cold image call | 140.994445 s |
| Warm call 1 | 140.079573 s |
| Warm call 2 | 137.870631 s |
| Warm call 3 | 139.267917 s |
| Warm median | 139.267917 s |
| Warm mean | 139.072707 s |
| Warm population standard deviation | 0.912300 s |
| Peak process RSS | 6,284,516 KiB, approximately 5.99 GiB |
| Repeated output bytes | Identical across all four calls |
| Known input RGB errors | 0 |

These are three warm observations for one photograph, on one shared CPU host,
in a fixed run order. They do not establish confidence intervals or performance
on other hardware. Process RSS is the driver's `RUSAGE_CHILDREN` high-water mark;
it also includes its much smaller pre-run process-snapshot child. The receipt
contains the before/after resource snapshots and preserves that measurement
scope. There was no observer attached to the timed image calls.

The candidate requested `CRISPEMBED_INPAINT_GRAPH=1`,
`CRISPEMBED_INPAINT_GEMM_LAYOUT=1`, and
`CRISPEMBED_INPAINT_CPU_GLA=1`. Mixed precision remained disabled; all other
optimization switches in the frozen controller were explicitly zero. The
native summary confirmed those switches, and the actual execution profiles
reported `backend=CPU` and `precision=f32`. The profiler identifies the graph
and its executions, rather than separately naming each specialized kernel.

The settings were seed 0, guidance 4.5, noise offset 0.0357, and 20 requested
DDIM steps. The original schedule's strength handling results in 19 executed
timesteps per image. The retained profiles showed one encoder graph with four
runs, one combined UNet/CFG/DDIM graph with 76 runs, and one decoder graph with
four runs. They confirm reuse across the four complete image calls. This was
not a reduction in steps for the benchmark.

## Partial eager evidence and explicit stop

The eager implementation flushed a cold time of 311.464 seconds and one warm
time of 312.904 seconds. Those progress lines use lower printed precision than
the completed candidate summary. They report the same eager RGB hash on both
calls, but that hash differs from the candidate's. The eager process was
cancelled before it wrote the finalized RGB file and summary, so no independent
pixel comparison was possible. A difference in hashes alone supplies no
pixel-error verdict.

Only the verified owned controller was signalled. It terminated its native
process group, retained both completed-call records and its logs, and wrote a
final cancellation receipt with native return code `-15`. Its recorded partial
process lifetime was 842.872024 seconds and partial peak RSS was 4,483,672 KiB.
These are cancellation observations, not a completed eager benchmark. The
portable graph has an explicit `not_executed` entry. No median ratio is computed.

## Preserved timing-report correction

The candidate native executable returned success. Its first controller receipt
nevertheless failed while recomputing the warm standard deviation: the native
summary prints ten significant digits, so recomputing from the rounded samples
differs from the pre-print statistic by 0.0000227 milliseconds. The initial
checker did not account correctly for input serialization error.

The corrected checker derives its bounds from the decimal rounding of the
individual samples and aggregate, plus floating-point arithmetic slack. The
original failed receipt remains unchanged. A separate revalidation binds its
exact SHA-256, the original stdout/stderr/RGB files, the original and corrected
controllers, and the preserved executable. It repeats the image, identity,
switch, repeat-output, actual-backend, and 4/76/4 execution-count checks. That
independent revalidation passed. No model was rerun for this reporting
correction, and no image-error budget or runtime implementation was changed.

## Model identity and limits

The converted PixelHacker denoiser is 3,449,075,872 bytes; the shared F32 SDXL VAE
is 334,636,224 bytes. Their combined file size is 3,783,712,096 bytes, about
3.78 decimal GB. File size is distinct from runtime memory: the completed
candidate used approximately 5.99 GiB of peak process RSS.

The [conversion guide](latent-inpaint-conversion.md) documents the pinned
original checkpoints, strict tensor census, and reversible schema-based
`sha256-56-v1` name mapping. The mapping preserves the numerical tensor
payloads. The [source contract](latent-inpaint-source-contract.json) and public
evidence record retain checkpoint and converted-file hashes, source pins,
license evidence, and byte counts. No model weights are redistributed in this
record.

The preserved benchmark executable has SHA-256
`147be9f40d43f165f52b3a6bfa6ca942ece0c1e1354a82d344a46ed2e7844030`.
Its bytes and the observed source files were saved before execution. The source
worktree was dirty, so these observations are not presented as a clean-commit
build attestation. Component operator checks are documented separately in
[the CPU optimization guide](inpainting-cpu-optimization.md); neither those
checks nor this single-photo execution establishes original PixelHacker
full-model parity, population quality, or GPU qualification.
