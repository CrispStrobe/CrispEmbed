# Inpainting CPU optimization: measured component results

The CPU graph operators have now been built and executed. The opt-in CPU FFT
and rectangular gated-linear-attention paths passed the existing component
budgets and reduced warm operator time for the measured geometries. These are
component results: a faster Fourier transform or attention operator does not
by itself establish a faster or numerically accepted complete inpainter.

The accompanying [public evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json)
contains all 87 benchmark cases, their raw timing samples, numerical errors,
graph allocations, process exit statuses, executable digests and observed
source-file digests. It also preserves the initial spectral failure and the
diagnostic evidence for its correction. Absolute host paths are removed.

MI-GAN has separate original-model evidence: both the immutable baseline and
optimized candidate passed 20 cases across the 256- and 512-pixel models, with
1,070 stage comparisons per implementation. Their corresponding native stage
hashes were identical. At each model size, two synthetic cases have full
stage capture, six further synthetic cases have three anchor stages, and two
photographs have three anchor stages. These finite fixtures do not establish
population-wide image quality. See the [MI-GAN CPU record](../tests/results/inpainting/migan-cpu-2026-11-01.json)
for that qualification and its exact configuration. This spectral component
record does **not** qualify complete LaMa, Moebius or PixelHacker inference.
LaMa's subsequent full-model run is now characterized separately: seven of
eight synthetic fixtures pass every stage in all three CPU variants, while
the full-mask fixture retains strict intermediate failures despite passing
final RGB checks. See [the LaMa validation report](lama-cpu-validation.md).
Latent-model qualification remains separate. CUDA and Metal were not available
for execution on this machine.

## Machine and measurement method

| Property | Recorded configuration |
| --- | --- |
| CPU | AMD EPYC 9V74, x86-64; AVX2, FMA and AVX-512 available |
| CPU allowance | 8 CPU equivalents (`800000 / 100000` quota); 9 CPUs in affinity |
| Memory limit | 8,589,934,592 bytes, or 8 GiB |
| OS | Linux 6.18.57, glibc 2.39 |
| GPU | No exposed NVIDIA or DRI device; no CUDA toolchain; no Metal on this Linux host |
| Build | Release, native CPU instructions, llamafile enabled; BLAS, OpenMP, CUDA, Metal and Vulkan disabled; static build |
| Numeric profiles measured | Portable F32; CPU recurrent F32; CPU FFT with F32 storage and F64 FFT arithmetic |

These values are copied from the hardware and build observations in the
[evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json).
The source base was `5dc87e1641d91f4c92517620af977e738d8c8aa5`, with local
corrections and candidates applied. The ggml revision was
`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. Each measurement series records its
own executable and source hashes. The working tree was not clean, and these
observations are **not** a clean build receipt proving that a particular
commit produced a particular executable.

The test invokes the production graph helpers with deterministic tensors.
Warm times measure `Builder::compute()` after graph allocation and input
upload. Construction, uploads, first execution, output downloads and independent
reference time are recorded separately. Warm samples perform no activation
upload or download and reuse the same graph and allocation. The reference
timer includes its wrapper allocation and copies, so it is not used as a
speedup baseline. See [the benchmark implementation](../tests/test_inpaint_graph_spectral.cpp).

The paired FFT experiment uses the same executable, two untimed warmups and
five timed samples per variant. The earlier portable GLA sweep has one warmup
and two timed samples; the CPU recurrent candidate uses a later executable,
two warmups and five samples. The raw samples expose variability that a single
median hides. These short runs support selecting candidates for further
testing; they do not provide confidence intervals, cross-machine estimates or
full-model throughput.

## The corrected baseline

The original `5dc87e1` spectral test did not pass. Initial transforms completed,
then graph replay reached an out-of-range `GET_ROWS` assertion. A diagnostic
read actual bit-reversal and Hermitian index tensors before and after execution:
the first execution had overwritten their integers with activation data.
Both the failed output and the retained-input replay are included under
`baseline_history` in the [public evidence](../tests/results/inpainting/cpu-components-2026-11-01.json).

In this pinned ggml allocator, marking an allocated tensor `INPUT` does not
retain its storage after its last use in a graph. Persistent parameters,
integer constants and prepared transpose-convolution weights now also carry
`OUTPUT`, which prevents that reuse. The tests snapshot the allocated F32 and
I32 leaf bytes and check them after execution, then replay the same graph.
The correction is in [the shared graph builder](../src/inpaint_graph.cpp);
the byte-preservation checks are in [the spectral tests](../tests/test_inpaint_graph_spectral.cpp).

Every timing baseline below already includes this correction. The original
failing graph supplies diagnostic evidence, not a valid speed comparison.
This distinction also matters for memory: retaining immutable inputs can
increase graph allocation, and that required storage is included in the
corrected measurements.

## LaMa FFT candidate

The default portable implementation expresses separable radix-2 FFTs as ggml
operations and uses Bluestein convolution for other axis lengths. The new
CPU-only candidate uses one ggml `CUSTOM` CPU node per transform, adapting the
existing [LaMa FFT mathematics](../src/lama_fft.cpp). It reads and writes
already allocated CPU backend buffers directly. There is no intermediate
backend download/upload, external FFT library or GPU-to-CPU fallback.
CUDA and Metal continue to use the portable graph. See
[the spectral implementation](../src/inpaint_graph_spectral.cpp).

Weights and graph tensors remain F32. Inside the CPU callback, complex FFT
arithmetic uses `double`, with conversion to F32 at the output. Its recorded
arithmetic name is `f32-storage-f64-fft`; describing this path as entirely F32
arithmetic would be incorrect. Cached bit reversal, twiddle and Bluestein
geometry avoid repeated setup during warm replay. Only the retained
half-spectrum receives the forward vertical transform. The inverse preserves
the real projection for arbitrary learned DC/Nyquist values, including odd
widths, and keeps batches independent.

### Paired FFT timings

All cases below use 192 channels, batch one and eight threads. Times are warm
median milliseconds from five samples. Ratios divide portable time by CPU
candidate time; they apply to the individual transform.

| Transform | Portable graph ms | CPU FFT ms | Component ratio | Portable nodes | CPU nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward 32 × 32 | 17.940267 | 0.323416 | 55.5× | 222 | 1 |
| Inverse 32 × 32 | 19.484453 | 0.390976 | 49.8× | 228 | 1 |
| Forward 64 × 64 | 71.126168 | 1.413473 | 50.3× | 258 | 1 |
| Inverse 64 × 64 | 53.098744 | 1.540601 | 34.5× | 264 | 1 |
| Forward 33 × 32 | 97.627085 | 2.307902 | 42.3× | 417 | 1 |
| Inverse 33 × 32 | 108.101379 | 2.291678 | 47.2× | 428 | 1 |

Source: `spectral-native-fft`, runs `model-shapes-native-fft-t8` and
`model-shapes-portable-fft-t8` in the
[evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json).
This comparison holds the executable and graph helper interface constant;
it intentionally compares different internal arithmetic, F32 graph operations
against the F64 CPU FFT implementation.

### CPU FFT thread sweep

| Transform | 2 threads ms | 4 threads ms | 8 threads ms |
| --- | ---: | ---: | ---: |
| Forward 32 × 32 | 0.728935 | 0.631732 | 0.323416 |
| Inverse 32 × 32 | 0.889421 | 0.795332 | 0.390976 |
| Forward 64 × 64 | 2.928207 | 2.704677 | 1.413473 |
| Inverse 64 × 64 | 3.963943 | 3.032871 | 1.540601 |
| Forward 33 × 32 | 5.404708 | 4.635474 | 2.307902 |
| Inverse 33 × 32 | 5.266153 | 3.535311 | 2.291678 |

All 18 native FFT case/thread measurements passed the unchanged component
gate. The largest measured absolute difference against the independent
existing F64 FFT reference was `4.99600361e-16`. That very small difference
reflects the shared mathematical precision and F32 output; it is not an
original-PyTorch or complete-LaMa parity result. Separate controls use direct
DFT calculations, arbitrary learned half-spectra, odd/even geometry, two
batches and replay. The source data are in the same
[FFT series](../tests/results/inpainting/cpu-components-2026-11-01.json).

## PixelHacker rectangular GLA candidate

The portable path uses stable log-gate prefixes, causal within-chunk
contractions and a rectangular carry state. The production default remains
64 tokens per chunk; the component benchmark also accepts 8, 16 and 32.
This all-device decomposition remains the CUDA and Metal path.

For CPU, an opt-in path reuses ggml's existing F32 `GATED_LINEAR_ATTN` kernel.
That kernel requires square heads. The adapter splits each value head into
`ceil(dv / dk)` independent groups of width `dk`, repeats Q/K/per-key decay
for those groups, and zero-pads only the final value group. It retains the
original `1 / sqrt(dk)` query scale. The batch dimension becomes independent
heads, every invocation starts from immutable zero state, and appended final
kernel state is discarded. For PixelHacker's `dv = 2 * dk`, the split does
not increase the total rectangular recurrent-state element count. The
implementation and geometry checks are in
[inpaint_graph_spectral.cpp](../src/inpaint_graph_spectral.cpp).

### GLA timings

All cases use eight heads, batch two and eight threads. The “best measured
portable chunk” column is the lowest observed median among 8/16/32/64 for
that geometry, not a changed production default.

| Tokens; key/value width per head | Portable chunk 64 ms | Best measured portable chunk; ms | CPU recurrent ms |
| --- | ---: | ---: | ---: |
| 4,096; 20 / 40 | 654.673376 | 8; 228.583655 | 36.665044 |
| 1,024; 40 / 80 | 229.515124 | 16; 110.700660 | 21.034123 |
| 256; 80 / 160 | 72.780964 | 16; 44.672198 | 9.164142 |

Source: the `spectral-portable-baseline` and `spectral-native-gla` series in
the [evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json).
These are separate builds with different warmup/sample counts as described
above. Showing smaller portable chunks avoids attributing the entire gain
to the native recurrent kernel when the default chunk size itself has a
large CPU cost.

| Tokens; key/value width | CPU recurrent, 2 threads ms | 4 threads ms | 8 threads ms |
| --- | ---: | ---: | ---: |
| 4,096; 20 / 40 | 47.006272 | 27.165702 | 36.665044 |
| 1,024; 40 / 80 | 23.671755 | 13.245333 | 21.034123 |
| 256; 80 / 160 | 12.426136 | 6.877322 | 9.164142 |

Four threads were faster than eight in these recorded GLA runs. That is a
measurement for this operator on this CPU quota, not a universal thread
recommendation. All nine native GLA case/thread measurements passed the
unchanged gates; maximum absolute error was at most `1.1920929e-7`, relative
L2 at most `5.62875622e-8`, and each recorded cosine rounded to one. Cold and
warm results were bit-identical within each measured graph. The reference
is an independent recurrent scalar implementation, not a PixelHacker
original-model capture.

## Fixed correctness gates and executed controls

The component budgets were declared before execution and remain unchanged:

| Control | Acceptance |
| --- | --- |
| FFT output | Finite; maximum absolute error ≤ `1e-4` |
| GLA output | Finite; maximum absolute error ≤ `1e-4`; relative L2 ≤ `0.002`; cosine ≥ `0.99999` |
| Warm graph execution | Same graph/allocation; no additional activation uploads or downloads |
| Persistent inputs/constants | Allocated leaf bytes preserved across execution and replay |

The latest spectral executable produced these results:

| CPU settings | Checks | Failures |
| --- | ---: | ---: |
| Portable FFT and portable GLA | 382 | 0 |
| CPU FFT and portable GLA | 382 | 0 |
| CPU FFT and CPU recurrent GLA | 172 | 0 |

The CPU recurrent branch has fewer checks because it does not repeat the same
kernel over four portable chunk widths. Its tests still exercise unequal
key/value widths, non-multiple tails, batch two, extreme finite gates,
future-token causality, independent state and replay. The raw suite summaries
and diagnostic output are included under the three `units-*` runs in
[the evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json).
Complete-model comparators retain their separately declared stage and final
image budgets; no threshold was relaxed to admit these candidates.

## Memory and tensor-name corrections

Fewer graph nodes do not imply less total memory. At the largest measured GLA
shape, CPU recurrence uses a graph arena of **83,988,480 bytes**, versus
**69,812,224 bytes** for portable chunk 64 and **52,633,856 bytes** for chunk
8. Repeating query/key/gate inputs for square-head adaptation has a storage
cost. At the other two shapes the native arenas are 42,352,640 and 22,609,920
bytes. These are component graph allocations, not model-wide peaks.

The CPU FFT graph arena is 1,622,016 bytes for 32 × 32, 6,389,760 for 64 × 64
and 1,646,592 for 33 × 32, including the graph input/output allocation. It
does **not** include callback stack or cached plans. Each worker's scratch
arrays occupy 268,288 bytes on this build. Axis lengths are bounded to 1–128;
all possible cached geometry vectors together have an 814,840-byte payload
bound per worker on a 64-bit build, plus plan objects, pointer/vector metadata
and allocator overhead. Plans are created lazily and contain no activations.
See `cpu_fft_memory` and each case's allocation fields in
[the evidence record](../tests/results/inpainting/cpu-components-2026-11-01.json).

The recorder's `max_rss_kib` is the cumulative maximum child-process RSS seen
by that recorder, not an isolated peak for each tensor case. A full-model
memory report must also include weights, skip activations, cached component
graphs, backend workspace, CPU stacks and caches. The model file-size
criterion “under 4 GB” does not guarantee a process fits within 4 GB RAM.

Actual latent weight conversion exposed a separate load blocker: many original
state names exceed the pinned ggml 64-byte name field. The converter now marks
`crispembed.tensor_name_encoding = sha256-56-v1`. Source names shorter than 64
UTF-8 bytes remain unchanged; longer physical names become `t.` followed by
the first 56 hexadecimal SHA-256 digits. The complete source census is
collision-checked, receipts retain `original_name` as well as the physical
name, and native loaders restore the original schema names. The ggml ABI,
source tensor identities and weight values are unchanged. See
[the converter contract](../tools/latent_inpaint_contract.py),
[native mapping](../src/inpaint_weight_names.h) and
[alias controls](../tests/test_latent_inpaint_contract.py).

The three real checkpoints were reconverted and every exported F32 payload
hash compared with the earlier conversion. All 2,080 exported tensor payloads
were identical; 540 Moebius names and 384 PixelHacker names required aliases.
The VAE has no long tensor names. Full readback and source-provenance identity
are recorded under `name_migration` in
[the public evidence](../tests/results/inpainting/cpu-components-2026-11-01.json).

| Converted artifact | File bytes | Aliased names / exported tensors |
| --- | ---: | ---: |
| Moebius denoiser F32 | 904,877,728 | 540 / 1,079 |
| PixelHacker denoiser F32 | 3,449,075,872 | 384 / 753 |
| Shared SDXL VAE F32 | 334,636,224 | 0 / 248 |

Moebius plus VAE therefore occupies about **1.24 GB**, and PixelHacker plus VAE
about **3.78 GB**, in decimal file bytes. Those sums describe converted file
storage only. Conversion/readback and graph allocation admission do not prove
complete inference, image quality or runtime peak-memory requirements.

## Selecting and reproducing the candidates

Both spectral candidates remain **default off**. They apply only on CPU;
selecting CUDA or Metal still uses the portable graph helpers. Full-model use
also requires `CRISPEMBED_INPAINT_GRAPH=1` and the existing model gate. These
CPU flags do not enable the F16 GPU mixed profile.

| Flag | CPU graph behavior |
| --- | --- |
| `CRISPEMBED_INPAINT_CPU_FFT=1` | Use the bounded native CPU FFT callback for LaMa transforms |
| `CRISPEMBED_INPAINT_CPU_GLA=1` | Use the native recurrent rectangular-head adapter for PixelHacker GLA |
| `CRISPEMBED_INPAINT_FAST=0` | Keep the strict weight/activation profile; CPU FFT still explicitly uses F64 internally |

Build the component target using the [graph build guide](inpainting-ggml-graphs.md),
including `-DGGML_OPENMP=OFF` to match this measurement configuration:

```sh
cmake --build build-inpaint --target test-inpaint-graph-spectral --parallel 2

# Corrected portable control, then the two CPU candidates together.
CRISPEMBED_INPAINT_CPU_FFT=0 CRISPEMBED_INPAINT_CPU_GLA=0 \
  CRISPEMBED_INPAINT_FAST=0 \
  ./build-inpaint/test-inpaint-graph-spectral --backend cpu --threads 2
CRISPEMBED_INPAINT_CPU_FFT=1 CRISPEMBED_INPAINT_CPU_GLA=1 \
  CRISPEMBED_INPAINT_FAST=0 \
  ./build-inpaint/test-inpaint-graph-spectral --backend cpu --threads 2

# Paired FFT benchmark: repeat with CPU_FFT=0 using the same executable.
CRISPEMBED_INPAINT_CPU_FFT=1 CRISPEMBED_INPAINT_CPU_GLA=0 \
  CRISPEMBED_INPAINT_FAST=0 \
  ./build-inpaint/test-inpaint-graph-spectral --backend cpu --threads 8 \
    --benchmark model-shapes --benchmark-op fft --warmup 2 --iterations 5

# Native GLA candidate; repeat at two and eight threads.
CRISPEMBED_INPAINT_CPU_FFT=0 CRISPEMBED_INPAINT_CPU_GLA=1 \
  CRISPEMBED_INPAINT_FAST=0 \
  ./build-inpaint/test-inpaint-graph-spectral --backend cpu --threads 4 \
    --benchmark model-shapes --benchmark-op gla --warmup 2 --iterations 5

# Portable GLA sweep; chunk 64 remains the production default.
CRISPEMBED_INPAINT_CPU_FFT=0 CRISPEMBED_INPAINT_CPU_GLA=0 \
  CRISPEMBED_INPAINT_FAST=0 \
  ./build-inpaint/test-inpaint-graph-spectral --backend cpu --threads 4 \
    --benchmark model-shapes --benchmark-op gla --gla-chunk all \
    --warmup 2 --iterations 5
```

The last command intentionally gathers five samples for a new paired trial;
the preserved initial portable GLA experiment used two. Use the original-model
comparators described in the [graph guide](inpainting-ggml-graphs.md) before
promoting any candidate to a default. Full-model timings must come from the
ordinary image API with observers disabled, after the same model/reference
configuration has passed its unchanged numerical and image gates.
