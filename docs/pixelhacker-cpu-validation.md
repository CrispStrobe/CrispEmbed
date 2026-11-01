# PixelHacker CPU execution and native comparison — 1 November 2026

The current eager and optimized CPU implementations each completed one cold
and three warm calls through the ordinary image API on the same pinned
512×512 `formula_photo_bar` fixture. Their final RGB comparison passed the
unchanged native agreement limits: maximum filled-channel error **1**, mean
absolute error **0.0000949435764**, PSNR **88.3561 dB**, and **zero errors in
known input channels**. Each implementation reproduced its own output bytes
across all four calls. The two implementations are not byte-identical.

This is a completed native CPU execution, repeatability and final-image
agreement check. **Original PixelHacker parity remains unqualified:** the
original CUDA/FLA implementation was not executed on this CPU-only host.
Perceptual quality, representative subtitle removal, temporal consistency and
CUDA/Metal execution also remain outside this record. The
[current public evidence](../tests/results/inpainting/pixelhacker-api-cpu-2026-11-01.json)
retains both complete producers, every timing sample, exact output identities,
independent RGB checks, source snapshots and actual execution profiles.

## Current ordinary API pair

Both arms used the same executable, F32 model files, input, mask, eight-thread
setting and stock diffusion schedule. The eager arm ran first. The candidate
then enabled the persistent graph, CPU GEMM layout, CPU gated linear attention
and SHA-NI model verification. Each arm loaded one context and reused it for
its four image calls. Stage observers were disabled.

| Measurement | Eager | Optimized CPU + SHA-NI |
|---|---:|---:|
| Model initialization and digest validation | 16.535945 s | 3.195765 s |
| Cold image call | 314.108361 s | 149.552855 s |
| Warm call 1 | 310.958028 s | 144.243473 s |
| Warm call 2 | 311.826501 s | 141.174032 s |
| Warm call 3 | 306.760212 s | 145.701230 s |
| Warm median | 310.958028 s | 144.243473 s |
| Warm mean | 309.848247 s | 143.706245 s |
| Warm population standard deviation | 2.212168 s | 1.886856 s |
| Peak process RSS | 4,478,836 KiB | 6,275,732 KiB |
| Own repeated output bytes | Identical across four calls | Identical across four calls |
| Known input RGB errors | 0 | 0 |

The observed eager/candidate warm-median ratio is **2.1558**, and the observed
loading/validation ratio is **5.1743**. These are diagnostic ratios from one
photograph, three warm samples per arm and one fixed run order on a shared CPU
host. **All timing qualification flags remain false.** The samples do not
establish a general speedup, confidence interval or cross-machine prediction.
They also do not isolate each optimization's contribution.

Loading includes digest verification and native initialization and is reported
separately from image inference. The controller independently hashes the model
files before launching each native arm, so file data may already be cached.
These observations are not cold-storage measurements. The independently
measured hashing component and its narrower qualification are documented in
[the model-hashing guide](inpaint-model-hashing.md).

The host was an AMD EPYC 9V74 with an eight-CPU quota, nine CPUs in its affinity
mask and an 8 GiB memory limit. Each process had a 7.5 GiB address-space cap.
Process RSS is the controller's `RUSAGE_CHILDREN` high-water mark and includes
its much smaller pre-run process-snapshot child. It is distinct from graph
allocator bytes, model-file bytes and total cgroup memory or file cache.

## What was verified

The final comparison covers 73,728 filled RGB channels. The acceptance limits
remain maximum absolute error ≤2, mean absolute error ≤0.1, PSNR ≥50 dB and
exact preservation of every known input channel. The exporter recomputed the
comparison from both retained raw images and independently checked the known
pixels. It also verified each final RGB file against the retained cold output
and all four printed call hashes.

The candidate explicitly selected:

```sh
CRISPEMBED_PIXELHACKER=1
CRISPEMBED_INPAINT_GRAPH=1
CRISPEMBED_INPAINT_GEMM_LAYOUT=1
CRISPEMBED_INPAINT_CPU_GLA=1
CRISPEMBED_INPAINT_SHA256_ACCEL=1
CRISPEMBED_INPAINT_FAST=0
CRISPEMBED_INPAINT_BACKEND=cpu
```

The frozen controller explicitly disabled every other optimization switch,
including the Moebius depthwise, local and relative contraction candidates.
The native summary confirmed the selected gates. The candidate's actual
execution profiles identified the CPU backend and F32 precision; requesting a flag alone was not
used as execution proof.

Both arms used seed 0, guidance 4.5, noise offset 0.0357 and 20 requested DDIM
steps. These defaults are bound to the preserved source; the native summary
reports the requested step count but does not log those other defaults
individually. The original strength slicing retains 19 timesteps per image.
The candidate's actual profiles show one retained encoder graph with **4**
runs, one combined UNet/CFG/DDIM graph with **76** runs and one decoder graph
with **4** runs. They prove graph reuse and the 19-step count for all four
candidate calls. The eager implementation has no equivalent graph counters;
its source, settings, completed calls and repeated RGB are bound separately.
No model weights, resolution or diffusion-step setting were reduced.

The eager loader reported scalar SHA-256 for both complete model files. The
candidate reported actual `sha-ni` execution for **53,891,810** complete
64-byte PixelHacker input blocks and **5,228,691** VAE input blocks. Its
accelerated-block counts equal the complete-input-block counts, and both
digests match the pinned complete-file identities. These counts exclude SHA
padding blocks. The hardware flag did not bypass a digest check or weaken the
trust requirements.

## Identity and memory

The PixelHacker GGUF is 3,449,075,872 bytes and the shared F32 SDXL VAE is
334,636,224 bytes. Their combined file size is 3,783,712,096 bytes, approximately
3.78 decimal GB. The current candidate's 6,275,732 KiB process RSS is about
5.99 GiB; the eager arm used about 4.27 GiB. A model pair smaller than 4 GB on
disk therefore does not imply a process below 4 GB.

The [conversion guide](latent-inpaint-conversion.md) documents the pinned
checkpoints, strict tensor census and reversible `sha256-56-v1` physical tensor
names. The numerical tensor payloads remain unchanged. The
[source contract](latent-inpaint-source-contract.json) and current evidence
retain checkpoint and converted-file hashes, source identities, license
evidence and byte counts. No model weights are redistributed in this record.

Both current arms used executable SHA-256
`8e860a3401f8010ba7b1f023cba3ce9862fee5d0b6d6d4b87826290fe190a8bd`
and private controller SHA-256
`c056e9790db54d2394ba0afde78a07f02af0e319e6f1b33135b2a6963380fb93`.
The export uses the exact snapshotted validation source bytes and verifies
their identities against producer admission. The observed worktree was dirty;
these records do not infer a clean-commit build attestation. This API record
does not itself qualify the production CLI, which has separate live evidence.

## Preserved earlier delivery

The [earlier v2 evidence](../tests/results/inpainting/pixelhacker-cpu-2026-11-01.json)
is unchanged. It records a completed candidate, an eager comparison stopped at
the user's immediate-delivery request and a portable-graph arm that was never
started. Its measurements use an earlier executable and are not pooled into
the current pair.

That earlier candidate recorded a 14.269750-second load, a 140.994445-second
cold call and warm calls of 140.079573, 137.870631 and 139.267917 seconds. Its
warm median was 139.267917 seconds and peak RSS was 6,284,516 KiB. The stopped
eager process retained a 311.464-second cold call and one 312.904-second warm
call before its next call was interrupted. It had no finalized RGB file or
summary, so the earlier panel did not claim a completed native comparison.
The owned process group was terminated safely; its final cancellation receipt
retains return code `-15`, 842.872024 seconds of partial lifetime and
4,483,672 KiB of partial peak RSS.

The earlier candidate's native executable returned success, but the first
controller receipt failed on timing serialization: recomputing standard
deviation from ten-significant-digit printed samples differed from the
pre-print statistic by 0.0000227 milliseconds. The original failed receipt is
preserved. A separate revalidation used a bound derived from decimal rounding
and floating-point slack and repeated the source, binary, image, gate,
backend, repeatability and 4/76/4 counter checks. It passed without rerunning
the model or changing an image-error budget.

The current eager output reproduces the earlier eager progress hash, and the
current candidate reproduces the earlier candidate output hash. A separate
early cold-output comparison is retained as a cross-version diagnostic; the
current verdict nevertheless depends on both completed current producers.
The current plan did not require another portable-graph arm after the fixed
native RGB comparison passed. Original CUDA/FLA parity and GPU performance
remain unqualified.
