# Optional acceleration of inpainting model hashing

**The optional SHA-NI path preserves the exact SHA-256 model checks and improved
the measured memory-hashing throughput by 4.68× on the tested CPU.** It is off by
default. It changes the time spent verifying model files; it does not change
weights, inference arithmetic, resolution, diffusion steps or quality limits.

The [complete CPU evidence](../tests/results/inpainting/sha256-cpu-2026-11-01.json)
contains every sample, both full-file identities, actual block counts, source
and executable hashes, the normal/fallback controls and resource observations.
These are hashing measurements. Full-image results remain in the separate
[inpainting results guide](inpainting-optimization-results.md).

## Scope and activation

Set `CRISPEMBED_INPAINT_SHA256_ACCEL=1` alongside the existing engine gates and
continue supplying the same required GGUF SHA-256 values. The flag applies to
the model-validation paths for MI-GAN, Big-LaMa, Moebius, PixelHacker and the
shared SDXL VAE. It does not enable an engine, change its input contract, or
make an unverified model acceptable.

The implementation is in
[`core/sha256_accelerated.h`](../src/core/sha256_accelerated.h), with three
small call-site changes in the image, LaMa and latent loaders. It reuses
CrispEmbed's existing SHA state, scalar compressor, final padding, length
encoding and lowercase digest formatting. The existing
[`core/sha256.h`](../src/core/sha256.h) and model-manager defaults are unchanged.
With acceleration and profiling both off, the file call delegates directly to
the original function. Profiling an ordinary scalar load uses the same
`fread`/scalar-update/finalization sequence, with counters added.

GNU-compatible GCC/Clang builds on x86-64 can compile the isolated SHA-NI
function. Runtime checks require SHA, SSSE3 and SSE4.1 support. The stream
selects its compressor once, using cached feature availability. Other builds
or unsupported processors use scalar hashing, even when acceleration was
requested. The implementation follows the
[NIST SHA-256 specification](https://doi.org/10.6028/NIST.FIPS.180-4) and
[Intel instruction semantics](https://www.intel.com/content/dam/develop/external/us/en/documents/319433-024-697869.pdf);
it does not copy another project's implementation or add a crypto dependency.

Every byte still contributes to the digest. The accelerated routine handles
complete 64-byte input blocks. The existing scalar finalization handles a
remaining partial block and the one or two SHA padding blocks. An exact digest
mismatch or a file open/read/close failure remains an initialization failure.

## Observed execution evidence

`CRISPEMBED_INPAINT_PROFILE=1` enables a record after the complete computed
digest matches the required digest:

```text
inpaint_hash: label=MODEL backend=sha-ni|scalar requested=0|1 hardware_available=0|1 bytes=N complete_blocks=N accelerated_blocks=N sha256=HEX
```

The stable labels are `migan`, `lama`, `moebius`, `pixelhacker` and `sdxl_vae`.
The record contains no filesystem path. `complete_blocks` means full input
blocks, excluding SHA padding; `accelerated_blocks` counts those actually sent
to SHA-NI. On a supported requested-on load these counts equal `bytes / 64`.
Scalar execution reports zero accelerated blocks. Availability means that both
the compiled implementation and required CPU features are available.

The public live/replay test drivers accept `--sha256-accel`. Their additional
`--require-sha256-accel` option requires actual hardware execution, rather than
accepting a valid scalar fallback. They check the observed label, digest, byte
count, selected backend and block counts against the independently pinned
model. This hashing record alone does not establish image or numerical parity.
The evidence linked above contains direct full-file helper measurements;
ordinary API/CLI load records are reported separately with their image runs.

## Exactness checks that ran

Both the initial standalone builds and the subsequent integrated native build
passed the same exact controls:

| Build | Assertions passed | SHA-NI compiled | Hardware path available |
| --- | ---: | --- | --- |
| Normal native build | 6,985 | Yes | Yes |
| Forced fallback build | 6,988 | No | No |

These assertion counts cover a fixed matrix, not thousands of model cases.
The controls include four known SHA-256 vectors, including the million-byte
vector; binary inputs around the 55/56/63/64-byte padding boundaries; offsets
of zero, one and three bytes; and nine streaming chunk sizes. They also cover
missing CPU features, requested-on scalar fallback, null inputs, repeated
finalization, rejected updates after completion, unchanged input bytes, empty
and missing files, a directory/file-read failure, and actual byte/block
accounting. The fallback build explicitly proves zero accelerated blocks while
retaining the same digest. The integrated build's complete model-free native
suite passed all 20 command groups.

The full PixelHacker file exceeds the 32-bit bit-count range. Its exact match
therefore also exercises the existing 64-bit SHA length accumulation with a
real large file. The file probes checked size and modification state around
both reads, and the controller rechecked source, executable, build flags and
model metadata between every stage.

## Measured memory hashing

The host was an AMD EPYC 9V74 running Linux, with an eight-CPU quota, nine CPUs
in the affinity set and an 8 GiB memory limit. Hashing used one thread. Both
arms were compiled into the same executable with GCC 13.3, `-O3`, `-DNDEBUG`,
C++17 and `-march=native`, matching the CPU policy of the loader libraries.
The executable uses PIE and the loader libraries use PIC; their CPU feature
and optimization settings match. No GPU was exercised.

The benchmark reads the same immutable, unaligned 32 MiB memory buffer. Three
samples per arm alternate scalar/accelerated, accelerated/scalar, then
scalar/accelerated. All digests match an independent Python `hashlib` check of
the exact input sequence. The requested arm processed exactly 100,663,296 input
bytes and 1,572,864 accelerated blocks across its three samples.

| Hashing path | Raw samples, ms | Median, ms | Throughput from median |
| --- | --- | ---: | ---: |
| Existing scalar SHA-256 | 121.374591, 115.658736, 119.284250 | 119.284250 | 268.27 MiB/s |
| Requested and observed SHA-NI | 25.542431, 24.140143, 25.468902 | 25.468902 | 1,256.43 MiB/s |

The ratio of median throughput is **4.6835× for this memory-hashing component
test**. Input generation, correctness controls and file loading are outside
those timed sections. This result does not measure model initialization or
image inference.

## Full-file identity and diagnostic times

Both implementations reproduced the required GGUF digest exactly for both
files. Their complete lowercase digests appear in the public evidence and the
existing conversion receipts.

| Pinned file | Input bytes | Actual SHA-NI blocks | Scalar time, ms | SHA-NI time, ms |
| --- | ---: | ---: | ---: | ---: |
| PixelHacker denoiser | 3,449,075,872 | 53,891,810 | 13,278.181642 | 2,930.345938 |
| Shared SDXL VAE | 334,636,224 | 5,228,691 | 1,202.978839 | 279.883713 |

**These are single ordered identity probes: scalar first, SHA-NI second.**
They can include cache and order effects and are not a repeated model-load
benchmark. The latent API controller also independently hashes pinned files
before launching the native process. Its later load timings therefore include
validation and initialization on potentially cached file data; they do not
describe cold-storage performance. Neither kind of load timing should be
added to an inference-only speed claim.

The recorded cumulative child-process peak RSS for this bounded hashing run
was 34,776 KiB. It is a cumulative helper-process observation, not a separate
peak for each file or the memory needed to run either model. File verification
uses a 64 KiB read buffer. The much larger cgroup memory observations include
shared processes and charged file cache. The cgroup's throttled-period count
did not increase during any of the three measured stages.

## Reproduce the checks

From a configured Release build, compile the two model-free targets:

```sh
cmake --build build --target test-inpaint-sha256-accel test-inpaint-sha256-fallback -j 2
./build/test-inpaint-sha256-accel --require-accel
./build/test-inpaint-sha256-fallback --require-fallback
./build/test-inpaint-sha256-accel --require-accel --benchmark-mib 32 --iterations 3
```

On a machine without a usable SHA-NI path, omit `--require-accel` for the normal
correctness check; it then validates scalar fallback. The forced-fallback
target disables the intrinsic implementation at compile time. Performance
claims require observed hardware execution and matching CPU compile settings.

An optional read-only file identity probe accepts a model path and its exact
converter-receipt digest:

```sh
./build/test-inpaint-sha256-accel --require-accel \
  --file models/pixelhacker.names-v1.gguf \
  --expected-sha256 5b486aa5d80647dbf298b0296d2cea7cb410dfb978b8bdfea806617f73ea588d
```

This command performs two complete file reads and no model inference. The
benchmark bounds are 1–5 samples per arm and at most 256 MiB of generated
memory input. The retained experiment used the smaller fixed 32 MiB/three-
sample plan above. Weights are not included in this hashing evidence bundle.
