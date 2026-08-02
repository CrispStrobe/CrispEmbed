# CrispEmbed Performance

## PP-OCRv6 scalar detector — where the convolution time goes (Apple M1, 2026-08-02)

Per-convolution profile of the CPU scalar detector, the dominant cost of the
PP-OCRv6 lane and the shared shape family for the other classical lanes.
Enable with `CRISPEMBED_PPOCRV6_DET_PROFILE=1`; fixture
`tests/regression/images/cc0/german_official_print.jpg` (1920x2518, detector
input 960x736).

| convolution class | share of detector conv time | rate |
|---|--:|--:|
| 1x1 pointwise | 51.6% | ~1.2 GF/s |
| depthwise (groups == channels) | 20.4% | 0.02-0.19 GF/s |
| deconv (2x2 stride 2) | 6.4% | ~0.6 GF/s |
| all other convolutions | 21.6% | ~1.0 GF/s |

**Read the shares, not the absolute totals.** The same profile measured
17,506 ms and 12,089 ms of total convolution time on two runs minutes apart,
because this box routinely sits at load 30-110 with several agent builds
running; the class proportions moved by under 3 points across the same pair.
Any absolute figure here is contended wall clock and is not a benchmark.

Heaviest single layer: a 7x7 **depthwise** convolution, 96 channels at 240x184,
2394 ms and 13.7% of all convolution time at 0.17 GF/s. Depthwise is the
generic `conv2d_cpu` path's worst case by construction — with one input and one
output channel per group there is nothing to amortise the patch gather against,
so it gathers a kh*kw window and consumes it in a single `dot_product`, once per
output pixel.

### 1x1 convolution kernel A/B

Same binary, gate off vs on, CPU-seconds (`user+sys`) median-of-3, bracketed by
the external `tesseract` load control. Wall clock is unusable on this box; CPU
time held across the window.

| arm | CPU-s |
|---|--:|
| control (`tesseract`, before) | 0.40 |
| `CRISPEMBED_CONV1X1_FAST` off (default) | 8.59 |
| `CRISPEMBED_CONV1X1_FAST=1` | **7.81** |
| control (`tesseract`, after) | 0.42 |

9.1% CPU, against 6% for the previous axpy form. The difference is traversal,
not arithmetic: the old form streamed the whole output plane once per (oc, ic)
pair, the current one blocks the pixel axis into 8192-element tiles so a tile's
input slab stays L2-resident and computes four output channels at a time.
Decoded-text equivalence over the 26-fixture corpus was still running when this
was written — treat the number as measured but not yet accepted, and see
`PLAN.md` H1 for the acceptance gate. The gate stays opt-in regardless:
`conv2d_cpu` is shared by 15 engines and only the PP-OCRv6 detector has been
measured.

`conv2d_depthwise_cpu` (`CRISPEMBED_CONVDW_FAST=1`) is implemented and
equivalence-guarded but **has no speed measurement yet**; given the 20.4% share
and the 0.02-0.19 GF/s rate it is the next thing to A/B.

## EasyOCR GGML parity benchmarks — Apple M1, 2026-08-01

All measurements below use the same `scan_strip.png` input and Miniconda
PyTorch reference where stated. Native timings are warm graph timings unless
noted; they are acceptance evidence, not claims that the current implementation
meets the speed target.

### Recognizers

| Path | Native Metal | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---|
| Latin Gen2 formula, width 200 | 16.523 ms | 12.460 ms | 1.33x | `x=0442` both; all stages pass |
| Latin Gen2 scan, width 128 | 10.885 ms | 7.137 ms | 1.53x | `82` both; all stages pass |
| Latin Gen1 ResNet, width 128 | 154.082 ms | 78.648 ms | 1.96x | `==#` both; all stages pass |
| English Gen2 scan, width 200 | 16.536 ms | 10.035 ms | 1.65x | `032` both; all stages pass |
| English Gen2 scan, width 128 | 10.697 ms | 7.287 ms | 1.47x | `@32` both; strict timestep-11 row cosine remains open |

Native is slower in every recognizer measurement. These are cross-device
directional comparisons; graph/kernel and dynamic-width optimization remain
open. Repeated native outputs are stable after fixing persistent LSTM state
storage aliasing.

### CRAFT detector

| Backend/model | Native graph | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---|
| Metal, runtime-BN F16 | 850.018 ms | 396.027 ms | 2.15x | 106 boxes both; taps pass |

The runtime-BN F32 graph matches captured Python tensors to floating-point
noise. Runtime-BN F16 also decodes 106 boxes. The older folded-F16 artifact
decoded 107 because accumulated CNN/BN error crossed a threshold; it is stale.
CPU-forced and Metal CRAFT outputs are byte-identical on this fixture.

### DBNet detector

| Backend/model | Graph | Postprocess | Total | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---:|---:|---|
| CPU, F16, 1 thread | 4178.6 ms | 8.3 ms | 4186.9 ms | 1213.450 ms | 3.45x | all taps pass; 96 regions |
| CPU, F16, 4 threads, persistent graph | 5661.1 ms warm | ~10 ms | ~5661 ms | 1213.450 ms | 4.67x | 98 rapid regions; `Brighton` present |
| CPU, F16, 8 threads, persistent graph | 2907.2 ms warm | ~10 ms | ~2907 ms | 1213.450 ms | 2.40x | 98 rapid regions; `Brighton` present |
| Metal, F16, persistent graph | 3499.4 ms warm | ~10 ms | ~3499 ms | 577.342 ms MPS | 6.06x | 98 rapid regions; `Brighton` present |

The Python reference reports `torch.get_num_threads()=4` and
`torch.get_num_interop_threads()=8`. Thus the 8-thread native result is the
best available throughput measurement but is not a same-thread comparison;
native remains slower even with twice the reference compute threads. On the
same M1 Metal device, the Python MPS blueprint averages `577.342 ms`, making
native Metal `6.06x` slower; this isolates the remaining gap to CrispEmbed's
Metal convolution/deconvolution kernels. F16
matches the fresh official MMOCR reference at backbone, neck, head, and
probability-map boundaries. The detector now uses a shape-keyed persistent
GGML graph; diagnostic tap retention is opt-in via
`OCR_DETECT_CAPTURE_TAPS=1`. Native quality is on par on this fixture, but all
native backend timings miss the reference speed target. Increasing CPU threads
and graph persistence help operationally but do not close the compute gap. Q4_K decodes the same 96 regions but diverges
at `backbone_stage_0` (global cosine `0.9960006`, RMS `0.07697`) and ends at
final-map cosine `0.9311001`; Q4_K is a quantization-quality TODO, not an
accepted parity variant.

An opt-in `OCR_DETECT_DIRECT_CONV=1` experiment was not promoted. GGML's CPU
direct-convolution kernel requires F32 weights, and the F32 direct graph did
not complete a diff run within roughly two minutes on the shared M1; it is not
parity or performance evidence. The default persistent im2col path is
unchanged. A later optimized/vectorized direct kernel remains a performance
TODO. One subsequent baseline run was resource-contended (44.1 s cold /
66.7 s warm with 8 threads), so it is excluded from the stable ratios above.
An attempted cumulative per-tap profiler was also rejected: prefix graphs
shared the persistent tensor arena and changed the restored run to zero boxes.
It produced no valid stage timings; isolated-allocator profiling remains open.

Benchmark results on Intel Xeon Skylake (4 threads), CPU-only, no GPU.

## Server Mode Latency (model loaded once)

Single-text encoding latency via HTTP server (`/embed` endpoint).

| Model | Quant | Params | Dim | Avg (ms) | Texts/s |
|-------|-------|--------|-----|----------|---------|
| all-MiniLM-L6-v2 | F32 | 22M | 384 | 15.5 | 64 |
| arctic-embed-xs | F32 | 22M | 384 | 15.5 | 64 |
| gte-small | F32 | 33M | 384 | 30 | 33 |
| octen-0.6b | Q8_0 | 600M | 1024 | 308 | 3.2 |
| octen-0.6b | Q4_K | 600M | 1024 | 294 | 3.4 |

## macOS Metal (Apple M1)

Benchmarked with Metal backend + embedded shaders, `./benchmark.sh --multi -n 20`.

### all-MiniLM-L6-v2 (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 3.9 ms / 258 t/s | 19 ms / 533 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 4.0 ms / 248 t/s | 62 ms / 161 t/s |
| HuggingFace sentence-transformers | 11.4 ms / 88 t/s | 23 ms / 431 t/s |
| CrispEmbed Server (Metal + HTTP) | 21.9 ms / 45 t/s | 31 ms / 318 t/s |
| FastEmbed Python (ONNX) | 33.5 ms / 30 t/s | -- |

### gte-small (33M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 4.1 ms / 243 t/s | 21 ms / 479 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 6.4 ms / 155 t/s | 70 ms / 142 t/s |
| HuggingFace sentence-transformers | 22.6 ms / 44 t/s | 226 ms / 44 t/s |
| CrispEmbed Server (Metal + HTTP) | 24.9 ms / 40 t/s | 52 ms / 190 t/s |

### arctic-embed-xs (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| **CrispEmbed Python** (Metal, ctypes) | 3.7 ms / 267 t/s | 46 ms / 220 t/s |
| fastembed-rs (Rust, ONNX) | 4.0 ms / 251 t/s | 29 ms / 342 t/s |
| FastEmbed Python (ONNX) | 4.1 ms / 244 t/s | -- |
| CrispEmbed Server (Metal + HTTP) | 22.2 ms / 44 t/s | 35 ms / 287 t/s |

CrispEmbed Python wrapper (ctypes, Metal) matches or beats fastembed-rs for
single-text latency. Batch throughput gap is due to per-text Python loop --
a C-level batch API would close it.

### VLM OCR decode — GOT-OCR2 (Qwen2-0.5B decoder), per token

| Decoder weights | Decode / token | Size |
|-----------------|----------------|------|
| **Q4_K** | **~20 ms** | 445 MB |
| F16  | ~38 ms | 1.44 GB |
| Q8_0 | ~42 ms | 599 MB |

Q4_K is fastest **and** smallest — prefer it over Q8_0 for autoregressive
decode on M1. Q8_0 being slower than F16 here is a Metal `mul_mv` (single-token
mat-vec) kernel issue, not a bandwidth effect; see
[`docs/metal-q8_0-mul_mv-slow-m1.md`](docs/metal-q8_0-mul_mv-slow-m1.md). Correctness
is unaffected (all three quants: cos ≥ 0.99996 vs f32, identical OCR — see
[`docs/got-ocr2.md`](docs/got-ocr2.md)).

### DBNet text detection — scanline box scoring (2026-07-13, M1 CPU)

`extract_boxes`' polygon scoring was O(bbox_area × contour_len) (ray-cast every
bbox pixel against the full traced contour) — pathological when a degenerate
component yields a very long contour. Rewritten as a scanline polygon fill
(even-odd-identical → **byte-identical boxes**, `OCR_DETECT_SCALAR_SCORE=1` for
the old path). dbnet-ic15-q4_k, forced CPU, a 10-line page:

| Stage | Before | After |
|-------|--------|-------|
| DBNet postprocess | 43 326 ms | **1 540 ms** (~28×) |
| Detection total (graph 3 s + postproc) | 46.4 s | **4.9 s** |
| Full DBNet+TrOCR pipeline (14 regions) | ~46 s | **7.2 s** (detect 4.4 · batch-enc 2.5 · decode 0.3) |

Note the decode is not the pipeline bottleneck here — the detection conv graph
and the ViT crop encoder are (both inherent compute). See LEARNINGS / HISTORY.

### DBNet degenerate-component fallback (2026-07-31, Apple M1 Metal)

The existing scanline scorer exposed a second postprocessing failure: valid
4-connected DBNet components could produce a one-point contour, making polygon
score zero and rejecting every box at the default `box_threshold=0.5`. The
postprocessor now falls back to the component bounding box and mean probability
for contours with fewer than three points.

On `tests/regression/images/fox.png` with
`dbnet-ic15-q4_k.gguf`, this changes detection from **0 to 10 boxes**. The
full DBNet+TrOCR pipeline recognizes 10 regions in about **5.0 s warm** on the
M1 Metal path (detection ~3.0 s, batched crop encoding ~1.8 s, decoding ~0.2 s).

### External document-parser comparison (2026-07-31)

The local CrispEmbed live check used the repeatable `fox.png` fixture and
GGUF models from `$CRISPEMBED_GGUF_DIR`:

| Engine / environment | Detection | Recognition | Timing | Quality check |
|---|---:|---:|---:|---|
| CrispEmbed DBNet + TrOCR, Apple M1 Metal | 10 regions | 10 regions | ~5.0–5.3 s/image warm | 8/10 words exact, CER 6.1% |

Expected text was `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 12345`; the two
word errors were `TAX` for `FOX` and `IAZY` for `LAZY`.

The comparison implementation could not be executed on this host: its CPU probe requires the
OpenCV development package, its live production path requires the documented
CUDA/TensorRT stack, and no usable Docker daemon or NVIDIA device is available.
Its repository reports **520–559 images/s** for forms/receipts and **200+
images/s** for dense documents on one RTX 5090, plus **92% FUNSD / 93% CORD
word-F1** and **0.90 OmniDocBench-125 overall at 20 pages/s**. Those are
The external NVIDIA benchmark claims are not measurements from this machine, and
are not directly comparable to the M1 single-image fixture above.

The actionable conclusion is to keep CrispEmbed's portable GGUF/Metal path,
but prioritize OCR quality and detector/recognizer batching before claiming
production parity. A fair head-to-head requires the same document corpus,
warmup policy, output metric, and an NVIDIA CUDA/TensorRT host.

#### TrOCR quantization A/B

This was not a TrOCR-vs-Python failure. The same detector, crops, decoder, and
ggml runtime were run with the recommended Q8 recognizer versus the locally
available Q4 model:

| Recognizer | Model size | Output on fox fixture | Warm total |
|---|---:|---|---:|
| TrOCR-small-printed Q4_K | 43 MB | `TAX`, `IAZY` errors | 4.75 s |
| TrOCR-small-printed Q8_0 | 64 MB | exact 10/10 words | 5.13 s |

The model card explicitly warns that Q4_K degrades this narrow 256-dimensional
decoder and recommends Q8_0. Q8_0 is therefore the immediate quality fix;
the ~8% end-to-end cost increase is small because detection and the encoder
dominate this fixture.

## Ollama Integration (Q8_0, Apple M1)

All CrispEmbed models verified in Ollama fork with Ollama-compatible GGUF export.

### Encoder Models (Q8_0 and Q4_K vs HuggingFace F32)

| Model | Dim | Q8_0 cos | Q4_K cos | Q8_0 Size | Q4_K Size |
|-------|-----|----------|----------|-----------|-----------|
| all-MiniLM-L6-v2 | 384 | 0.9998 | 0.970 | 24 MB | 18 MB |
| gte-small | 384 | 0.9999 | 0.991 | 34 MB | 24 MB |
| arctic-embed-xs | 384 | 0.9999 | 0.995 | 24 MB | 18 MB |
| multilingual-e5-small | 384 | 0.9999 | 0.990 | 126 MB | 115 MB |
| pixie-rune-v1 | 1024 | cross-lingual OK | cross-lingual OK | 581 MB | 437 MB |
| arctic-embed-l-v2 | 1024 | L2-norm=1.0 | L2-norm=1.0 | 581 MB | 437 MB |

### Decoder Models (Q8_0 and Q4_K in Ollama)

| Model | Arch | Dim | Q8_0 Size | Q4_K Size | L2-Norm | Diversity |
|-------|------|-----|-----------|-----------|---------|-----------|
| qwen3-embed-0.6b | Qwen3 | 1024 | 610 MB | 300 MB | 1.000 | 0.599 |
| octen-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.649 |
| f2llm-v2-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.711 |
| harrier-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.504 |
| harrier-270m | Gemma3 | 640 | 287 MB | 239 MB | 1.000 | 0.922 |
| jina-v5-nano | Qwen3 | 768 | 222 MB | 168 MB | 1.000 | 0.237 |
| jina-v5-small | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.746 |

All 13 Ollama-verified Q4_K models: L2-normalized, semantically correct embeddings.
Diversity = 1 - avg cosine similarity between 4 different test texts (higher = better discrimination).

## GPU Inference (CUDA)

Tested on NVIDIA RTX A1000 Laptop GPU (4GB VRAM), via HTTP server.

| Model | Quant | Avg (ms) | Texts/s | Batch (10) |
|-------|-------|----------|---------|------------|
| all-MiniLM-L6-v2 | F32 | 7.4 | 135 | 211/s |

GPU inference **matches HuggingFace PyTorch** (10.6ms vs 10.8ms) and
**beats fastembed ONNX** (10.6ms vs 13.4ms). Both HF and CrispEmbed use
CUDA on this hardware. The ggml_backend_sched dispatcher offloads
matmul, flash attention, and norm ops to CUDA.

True batched encoding: single graph with 4D flash attention for B texts.
Batch mode (10 texts): 190-211 texts/s on CUDA. HF gets 347/s via
PyTorch's native batch parallelism (more mature GPU batching).

## CPU Batch Mode

| Model | Batch Latency | Per-text | Texts/s |
|-------|--------------|----------|---------|
| all-MiniLM-L6-v2 | 114ms | 11.4ms | 88 |

Optimizations: graph caching, flash attention (fused QKV), buffer reuse,
sorted batch processing (group by token count for graph cache hits).

**True single-graph batching for bidirectional encoders (2026-07, opt-in).** The
default path encodes each text in its own graph. Two fused batch paths are available
for absolute-position encoders (BERT/XLM-R/MiniLM/BGE/E5), both bit-parity with
per-sequence encoding (cos ≥ 0.9999):

| Path | Env | Attention | Notes |
|------|-----|-----------|-------|
| Packed block-diagonal | `CRISPEMBED_ENCODER_PACKED=1` | O(T_total²) | one graph, block-diagonal mask; token-budget grouped (`CRISPEMBED_ENCODER_PACK_MAXTOK`, def 384). Size/backend dependent |
| Rectangular 4D per-item | `CRISPEMBED_ENCODER_4D=1` | O(B·T²) | separate 4D items + per-item pad mask; length-sorted chunks (`CRISPEMBED_ENCODER_4D_GROUP`, def 32) |

The 4D path is **consistently faster than both sequential and packed** (≈1.2×–1.5× at
batch 8/32/128 of short texts, on M1 CPU) and is the recommended path; it stays opt-in
pending a real-Metal A/B (measurements above are CPU-only). See PLAN.md § C3.

## Comparison with HuggingFace and fastembed (ONNX)

Single-text latency, same hardware (CPU, 4 threads).

| Model | CrispEmbed | HF PyTorch | fastembed ONNX | vs HF | vs ONNX |
|-------|-----------|------------|----------------|-------|---------|
| MiniLM-L6-v2 | **15.5ms** | 54ms | 29.5ms | **3.5x faster** | **1.9x faster** |
| gte-small | **30ms** | 79ms | -- | **2.6x faster** | -- |
| arctic-embed-xs | **15.5ms** | -- | 4.9ms | -- | 0.32x |

Optimizations: graph caching, flash attention, pre-merged QKV weights, buffer reuse.

CrispEmbed is **1.9-3.5x faster than HF PyTorch** and **1.9x faster than fastembed ONNX**
for MiniLM on pure CPU. Fastembed ONNX is 3x faster for arctic-embed-xs due to ORT's
Level3 graph JIT compilation (operator fusion, fused LayerNorm, layout optimization).
We apply QKV weight fusion and flash attention but cannot match ORT's runtime compilation.

Key advantages:
- No Python runtime overhead (direct C++ inference)
- No ONNX runtime dependency
- Graph + work buffer reuse across calls
- ~20MB binary vs ~500MB Python + ONNX environment

## Model Sizes

| Model | F32 | Q8_0 | Q4_K | Q8_0 ratio |
|-------|-----|------|------|------------|
| all-MiniLM-L6-v2 | 87 MB | 24 MB | 19 MB | 3.6x |
| gte-small | 128 MB | 35 MB | 25 MB | 3.7x |
| arctic-embed-xs | 87 MB | 24 MB | 19 MB | 3.6x |
| multilingual-e5-small | 453 MB | 123 MB | 113 MB | 3.7x |
| pixie-rune-v1 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| arctic-embed-l-v2 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| octen-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| f2llm-v2-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| jina-v5-nano | 585 MB | 219 MB | 164 MB | 2.7x |
| jina-v5-small | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-270m | 741 MB | 279 MB | 231 MB | 2.7x |
| qwen3-embed-0.6b | 1.6 GB | 607 MB | 291 MB | 2.7x |

## Quantization Quality

Cosine similarity between F32 and quantized models (1.0 = identical).

| Model | Q8_0 | Q4_K |
|-------|------|------|
| all-MiniLM-L6-v2 | 0.9995 | 0.97 |
| gte-small | 0.9998 | 0.99 |
| arctic-embed-xs | 0.9999 | 0.99 |
| multilingual-e5-small | 0.9999 | 0.99 |
| pixie-rune-v1 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | 0.9989 | 0.95 |
| octen-0.6b | 0.9995 | 0.97 |
| harrier-0.6b | 0.9999 | 0.99 |
| harrier-270m | 0.9998 | 0.99 |
| qwen3-embed-0.6b | 0.9996 | 0.97 |

| all-mpnet-base-v2 | 0.9998 | 0.99 |
| nomic-embed-text-v1.5 | 0.9994 | -- |
| gte-modernbert-base | 0.9999 | -- |
| bge-small-en-v1.5 | 0.9999 | 0.99 |
| bge-base-en-v1.5 | 0.9999 | 0.99 |
| bge-large-en-v1.5 | 0.9999 | 0.99 |
| all-MiniLM-L12-v2 | 0.9999 | 0.99 |
| mxbai-embed-large-v1 | 1.0000 | 0.99 |
| snowflake-arctic-embed-m | 0.9999 | 0.99 |
| snowflake-arctic-embed-l | 0.9999 | 0.99 |

Q8_0: all > 0.995. Q4_K: most > 0.95.

## BLAS Acceleration

OpenBLAS 0.3.26, Intel Xeon Skylake, 4 threads.

| Model | Quant | no-BLAS | BLAS | Speedup |
|-------|-------|---------|------|---------|
| gte-small | F32 | 114ms | 123ms | 0.9x |
| gte-small | Q8_0 | 116ms | 116ms | 1.0x |
| octen-0.6b | Q8_0 | 422ms | 410ms | 1.0x |

BLAS provides minimal benefit because quantized kernels use ggml's SIMD paths.
Use Q8_0 for CPU speed, GPU (CUDA/Vulkan) for maximum throughput.

## RAG Retrieval Quality

Retrieval quality on synthetic IR dataset (50 documents, 15 queries, graded relevance).
Model: all-MiniLM-L6-v2. Hardware: Intel Xeon Skylake, 4 threads, CPU-only.

| Engine | Model | MRR@10 | NDCG@10 | Recall@10 | Recall@100 | Time |
|--------|-------|--------|---------|-----------|------------|------|
| CrispEmbed F32 | all-MiniLM-L6-v2 | 1.0000 | 0.7846 | 0.7556 | 1.0000 | 0.63s |
| CrispEmbed F32 | bge-small-en-v1.5 | 1.0000 | 0.7470 | 0.6889 | 1.0000 | 3.19s |
| CrispEmbed Q8_0 | bge-small-en-v1.5 | 1.0000 | 0.7470 | 0.6889 | 1.0000 | 3.00s |

MRR@10 = 1.0: the most relevant document is always ranked first.
Recall@100 = 1.0: all relevant documents found within top-100.

**Key finding**: GGUF F32 embeddings produce identical retrieval quality to
HuggingFace (both are bit-identical, cos >= 0.999). Q8_0 quantization
(cos >= 0.995) should produce negligible retrieval quality degradation.

## Bi-Encoder Reranking

Bi-encoder reranking uses cosine similarity of L2-normalized embeddings.
CrispEmbed's `rerank_biencoder()` encodes query + all documents in a single
batch call, then computes dot products.

Example (all-MiniLM-L6-v2, query: "What is machine learning?"):

| Document | Score |
|----------|-------|
| Machine learning is a subset of artificial intelligence. | 0.7124 |
| Neural networks learn patterns from training data. | 0.5897 |
| The weather in Paris is mild in spring. | 0.0153 |

Correct ranking with clear separation between relevant and irrelevant docs.

## Feature Parity with fastembed-rs

| Feature | CrispEmbed | fastembed-rs | Winner |
|---------|-----------|-------------|--------|
| Single-text latency (MiniLM, M1 Metal) | 3.6 ms | 3.8 ms | CrispEmbed |
| Batch throughput (10 texts, M1 Metal) | 787 t/s | 528 t/s | CrispEmbed |
| Binary size | ~20 MB | ~500 MB (ONNX) | CrispEmbed |
| Quantization quality (Q8_0) | cos > 0.995 | INT8 varies | CrispEmbed |
| Model count (embedding) | 37 | 49 | fastembed-rs |
| Model count (reranker) | 7 | 20 | fastembed-rs |
| Sparse retrieval | BGE-M3 + SPLADE | SPLADE + BGE-M3 | Tie |
| ColBERT multi-vector | Yes | No | CrispEmbed |
| Image embedding | SigLIP + BidirLM-Omni | 5 models | Tie |
| Prompt prefix | Yes | Yes | Tie |
| Bi-encoder reranking | Yes | Yes | Tie |
| GPU backends | CUDA/Metal/Vulkan | ONNX EP | Tie |

## Notes

- CrispEmbed uses ggml inference with SIMD-optimized quantized matmul
- Graph and work buffers are reused across calls (3.2x throughput improvement)
- When built with CUDA/Vulkan/Metal, `ggml_backend_sched` auto-dispatches to GPU
- Decoder models (Qwen3/Gemma3) are 10-15x slower than encoders (28 layers vs 6)
- Server mode eliminates model loading overhead (~100-300ms per cold start)
- Prompt prefix adds negligible overhead (string concatenation before tokenization)
- Bi-encoder reranking cost = 1 batch encode + N dot products (O(N*dim) after encode)

## Latency Benchmark (Intel Xeon Skylake, CPU, 4 threads)

Single-text and batch (10 texts) encoding latency via Python ctypes wrapper.

| Model | Dim | Single (ms) | Batch 10 (ms) | Texts/s |
|-------|-----|------------|---------------|---------|
| all-MiniLM-L6-v2 | 384 | 12.7 | 48.8 | 205 |
| bge-small-en-v1.5 | 384 | 34.5 | 537.3 | 19 |
| all-MiniLM-L12-v2 | 384 | 443.0 | 239.0 | 42 |
| bge-base-en-v1.5 | 768 | 124.4 | 543.4 | 18 |
| all-mpnet-base-v2 | 768 | 66.4 | 292.9 | 34 |
| nomic-embed-text-v1.5 | 768 | 88.9 | 310.2 | 32 |

MiniLM-L6 is fastest (6.4ms single). NomicBERT is efficient for its size
(768d in 41.4ms). Batch throughput varies due to model size and graph complexity.

## Head-to-Head: CrispEmbed vs FastEmbed (ONNX)

Same models, same texts, same hardware (Intel Xeon, 4 threads, CPU-only).

| Model | Engine | Single (ms) | Batch 10 (ms) | Texts/s |
|-------|--------|------------|---------------|---------|
| all-MiniLM-L6-v2 | **CrispEmbed** | **6.4** | **23.6** | **424** |
| all-MiniLM-L6-v2 | FastEmbed | 60.8 | 255.9 | 39 |
| bge-small-en-v1.5 | CrispEmbed | 14.7 | 55.4 | 181 |
| bge-small-en-v1.5 | **FastEmbed** | **8.4** | **41.2** | **243** |
| snowflake-arctic-embed-m | CrispEmbed | 40.1 | **126.5** | **79** |
| snowflake-arctic-embed-m | FastEmbed | **33.3** | 127.5 | 78 |
| all-mpnet-base-v2 | CrispEmbed | 31.2 | 138.7 | 72 |
| nomic-embed-text-v1.5 | CrispEmbed | 41.4 | 150.6 | 66 |

**CrispEmbed vs FastEmbed**: CrispEmbed is **9.5x faster** on MiniLM-L6 (our most
optimized model: QKV fusion + flash attention + graph caching). On 12-layer models
(BGE-small, Arctic-M), FastEmbed's ONNX Runtime graph optimization (Level3 JIT,
operator fusion) gives it a slight edge. On Arctic-M batch, they're tied.

**Cosine similarity**: CrispEmbed vs FastEmbed cos=0.999999-1.000000 on all models.

## Per-Step Benchmark Instrumentation

Every runtime in CrispEmbed has opt-in per-step timing controlled by environment
variables. Set `CRISPEMBED_<MODULE>_BENCH=1` to get `[module-bench]` lines on
stderr showing millisecond timing for each processing phase (preprocess, encoder,
decoder, postprocess, per-tile, per-decode-step, total).

Zero overhead when unset — the flag is read once at init and stored as a bool.

| Category | Env vars |
|---|---|
| Embedding | `CRISPEMBED_CRISPEMBED_BENCH`, `VIT_EMBED`, `CNN_EMBED`, `CLIP_TEXT`, `LFM2_EMBED`, `DECODER_EMBED` |
| OCR detect | `CRISPEMBED_OCR_DETECT_BENCH`, `LAYOUT_DETECT`, `SURYA_DET`, `CC_DETECT` |
| OCR recognize | `CRISPEMBED_PARSEQ_BENCH`, `BTTR`, `HMER`, `POSFORMER`, `TESSERACT`, `PIX2STRUCT`, `MIXTEX`, `MATH_OCR`, `PPFN`, `PPFN_L` |
| VLM/LLM OCR | `CRISPEMBED_QWEN2VL_BENCH`, `GOT_OCR`, `GLM_OCR`, `GRANITE_OCR`, `INTERNVL2`, `DEEPSEEK_OCR2`, `LIGHTONOCR`, `SMOLDOCLING` |
| Super-resolution | `CRISPEMBED_ESRGAN_BENCH`, `DAT_SR`, `HAT_SR`, `PAN_SR`, `SAFMN_SR`, `SWINIR_SR`, `TBSRN_SR`, `TEXT_SR` |
| Denoise/restore | `CRISPEMBED_NAFNET_BENCH`, `SCUNET`, `RESTORMER`, `INSTRUCTIR`, `ADAIR` |
| NER/KIE | `CRISPEMBED_GLINER_BENCH`, `BERT_NER`, `LILT_KIE` |
| Pipeline | `CRISPEMBED_OCR_PIPELINE_BENCH`, `OCR_ORCH`, `KIE_PIPELINE`, `SCAN_CLEANUP`, `TABLE_PARSE` |
| Misc | `CRISPEMBED_PCS_BENCH`, `FIREREDPUNC`, `BIDIRLM_AUDIO`, `BIDIRLM_VISION`, `FACE_ALIGN`, `DEWARP`, `TPS_LOCNET` |

Example:
```
CRISPEMBED_PARSEQ_BENCH=1 ./crispembed-cli ocr image.png
# [parseq-bench] preprocess: 0.3 ms
# [parseq-bench] encoder graph (12 layers): 4.2 ms
# [parseq-bench] decoder CA K/V precompute: 0.1 ms
# [parseq-bench] decoder total (5 steps): 1.8 ms
# [parseq-bench] total: 6.4 ms
```

---

## Runtime Optimization Audit (June 2026)

Full line-by-line code review of all ~57K lines of C++ across 60+ runtime files.
Covers every runtime in the codebase: what optimizations are already in place,
and where the biggest opportunities remain.

### Methodology

Every `.cpp` and `.h` file in `src/` was read in full. Findings are grouped by
runtime category. "Existing" means the optimization is already implemented;
"Missing" means there is a concrete opportunity for improvement.

---

### 1. Core Shared Infrastructure (`src/core/`)

**Files**: `cpu_ops.h` (292L), `vlm_attention.h` (222L), `bpe.h` (218L),
`gguf_loader.cpp/.h` (487L), `mel.cpp/.h` (416L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Memory-mapped model loading | `gguf_loader.cpp` | `mmap`/`MapViewOfFile`, zero-copy weight access |
| Double-precision accumulator | `cpu_ops.h` LayerNorm/RMSNorm | Prevents float cancellation on long vectors |
| GPU-safe dequantization | `cpu_ops.h` `to_f32()` | Uses `ggml_backend_tensor_get`, works for Metal/CUDA tensors |
| Quantized weight support | `cpu_ops.h` `to_f32()` | Handles F32/F16/Q4/Q8 via `ggml_get_type_traits()->to_float` |
| In-place activations | `cpu_ops.h` | `silu_inplace`, `hardswish_inplace`, `relu6_inplace` |
| Numerically-stable softmax | `cpu_ops.h` | Max-subtract before `expf` |
| GQA support | `vlm_attention.h` | `kv_repeat = n_heads / n_kv_heads` reduces KV memory |
| Lazy byte_encoder table | `bpe.h` | Built once, cached in static |
| Two-pass GGUF loading | `gguf_loader.cpp` | Metadata pass is no-alloc |
| Mel spectrogram parameterization | `mel.cpp` | Single code path for 9 audio models |

#### Opportunities

| Priority | Location | Issue | Impact |
|----------|----------|-------|--------|
| **P0** | `cpu_ops.h` `linear_cpu` | No SIMD — naive scalar matmul O(N*M) | 4-8x with AVX2/NEON |
| **P0** | `cpu_ops.h` `linear_cpu` (tensor overload) | Re-dequantizes full weight matrix every call — no caching | Eliminates thousands of redundant alloc+dequant per decode |
| **P1** | `vlm_attention.h` `apply_rope` | `powf`/`cosf`/`sinf` computed per-element; no frequency table precomputation | 3-5x on RoPE-heavy models |
| **P1** | `mel.cpp` mel projection | Naive triple-loop matmul (T*128*201 ≈ 38M scalar MACs) | 10-20x with SIMD/BLAS |
| **P1** | `cpu_ops.h` `conv2d_cpu` | 6-nested scalar loops, no im2col or tiling | 5-10x with im2col+GEMM |
| **P2** | `vlm_attention.h` `gqa_attn_step` | `std::vector<float> scores(n_kv)` allocated per-head inside loop | Remove per-head allocation churn |
| **P2** | `vlm_attention.h` `swiglu_ffn` | Allocates two intermediate_dim vectors every call | Pre-allocate once |
| **P2** | `mel.cpp` STFT loop | Each frame's FFT is independent — no OpenMP parallelism | Linear speedup with core count |
| **P2** | `gguf_loader.cpp` mmap | No `madvise(MADV_SEQUENTIAL)` hint | Better kernel readahead on cold loads |
| **P3** | `gguf_loader.h` tensor map | `std::map` instead of `std::unordered_map` | ~2-5x faster tensor lookups |
| **P3** | `bpe.h` BPE merge loop | O(N^2) in symbol count; `vector::erase` from middle | Priority queue → O(N log N) |
| **P3** | `cpu_ops.h` `layernorm2d_cpu` | Iterates `(y, x, c)` but accesses stride-H*W — cache-hostile | NHWC layout or transpose |

---

### 2. VLM OCR Runtimes (Vision-Language Models)

**Files**: `qwen2vl_ocr` (2432L), `deepseek_ocr2` (1719L), `internvl2_ocr` (1715L),
`granite_vision_ocr` (614L), `got_ocr` (1455L), `glm_ocr` (1216L),
`lightonocr` (1365L), `smoldocling_ocr` (1011L), `pix2struct` (690L)

#### Optimization maturity ranking

> **REFRESH 2026-07-20 (code-verified):** every VLM decoder below now DEFAULTS to a
> ggml F16-KV GPU decode path; the `core_vlm` CPU-scalar decode survives only as a
> gated fallback (`CRISPEMBED_*_SCALAR` / `use_ggml` guards). The pre-refresh columns
> claiming "F32 CPU vectors" / "CPU scalar (core_vlm)" / "no KV cache" for
> qwen2vl/smoldocling/granite/pix2struct were STALE. Corrected:

| Rank | Runtime | LLM decode (default) | KV cache | GPU |
|------|---------|----------------------|----------|-----|
| 1 | **internvl2_ocr** | ggml flash_attn | F16 ggml tensor (zero-copy) | Yes |
| 2 | **glm_ocr** | ggml flash_attn (monolithic) | F16 ggml tensor | Yes |
| 3 | **got_ocr** | ggml flash_attn | F16 ggml tensor | Yes |
| 4 | **qwen2vl_ocr** | ggml + `build_decode_step_graph` | **F16 ggml backend** (`alloc_kv_cache`) | Yes |
| 5 | **lightonocr** | ggml flash_attn | F16 ggml persistent (`ggml_cpy`) | Yes |
| 6 | **deepseek_ocr2** | ggml per-layer graphs + flash | **F32** ggml (`alloc_ds_kv_cache`); F16 KV + persistent single-graph both still OPEN here | Yes |
| 7 | **smoldocling_ocr** | `sd_run_llm_body` ggml (default; `use_ggml`) | **F16 ggml backend**; core_vlm = fallback | Yes |
| 8 | **granite_vision_ocr** | `gv_run_llm_body` ggml (default; diff cos 0.9999) | **F16 ggml backend**; core_vlm = opt-out | Yes |
| 9 | **pix2struct** | CPU scalar + DequantCache | KV cache (Phase 2) — CPU, GPU port low-priority | No |

#### Already optimized (best practices found in at least one runtime)

| Technique | Where | Notes |
|-----------|-------|-------|
| Flash attention (`ggml_flash_attn_ext`) | internvl2, glm, got, lightonocr, smoldocling (vision) | Fused Q@K+softmax+V in single op |
| F16 KV cache in ggml tensors | internvl2, glm, got, lightonocr | Zero-copy view+cpy writes, halves memory |
| Prefill/decode separation | qwen2vl, internvl2, deepseek, got, glm, lightonocr | Full-sequence prefill, single-token decode |
| Fused QKV projection | qwen2vl | Single matmul for Q/K/V |
| `ggml_backend_sched` GPU dispatch | qwen2vl, internvl2, deepseek, got, glm | Automatic CPU/GPU placement |
| Precomputed RoPE tables | qwen2vl (2D), got, lightonocr (2D) | Host-side cos/sin computed once |
| Monolithic vision graph | glm, lightonocr | All layers in single graph (vs per-layer rebuild) |
| Skip logits during prefill | smoldocling | Skips V-sized LM head matmul for non-last tokens |
| Lazy expert dequant (MoE) | deepseek | Only dequantizes selected experts |
| Multi-threaded MoE dispatch | deepseek | Token-parallel expert evaluation |
| Periodic wbufs.clear() | smoldocling | Prevents unbounded dequant buffer growth |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| ~~**P0**~~ DONE | ~~Adopt F16 ggml KV cache (internvl2 pattern)~~ **— landed; all VLM decoders default to ggml F16-KV GPU decode (verified 2026-07-20, see maturity table)** | qwen2vl, deepseek, smoldocling, granite | Eliminates O(seq_len) per-step re-upload; halves memory |
| **P0** | Use `ggml_flash_attn_ext` for LLM decode | qwen2vl, deepseek | qwen2vl uses manual Q@K+softmax+V; deepseek uses per-layer graphs |
| **P0** | Move granite to ggml graphs | granite_vision_ocr | Entire engine is CPU-scalar — 10-50x potential speedup |
| **P0** | Implement batched prefill for smoldocling/granite | smoldocling, granite | Token-at-a-time through 30-40 LLM layers is catastrophic |
| **P0** | Move pix2struct to ggml graphs + add KV cache | pix2struct | Fully scalar, no KV cache, O(T^2) recompute per step |
| **P1** | Patch embedding conv → ggml matmul | ALL 9 runtimes | Every runtime uses scalar 6-deep nested loops |
| **P1** | Move deepseek Qwen2 encoder to ggml | deepseek_ocr2 | 24-layer bidirectional transformer entirely CPU-scalar |
| **P1** | Single multi-layer LLM graph (vs per-layer) | deepseek | 12 graph builds per decode token |
| **P1** | Cache dequantized weights | qwen2vl, deepseek, lightonocr, got, smoldocling, granite | `to_f32()` re-dequantizes same weights every call |
| **P1** | Scalar CPU downsample/merger → ggml | glm, got | Conv+matmul neck/projector still scalar |
| **P2** | InternVl2: native GQA in flash_attn (skip ggml_repeat) | internvl2 | Avoids duplicating KV heads before attention |
| **P2** | Vision tiles: batch multiple tiles in one graph | internvl2 | Currently sequential per-tile graph allocation |
| **P2** | Token embed via direct read (not mini-graph) | qwen2vl | Building a full ggml graph for one `ggml_get_rows` |
| **P2** | Decode graph reuse (not rebuild per step) | deepseek | Graph structure is identical across steps |
| **P2** | Windowed attention in qwen2vl vision | qwen2vl | window_size=112 declared but unused in graph |
| **P3** | LM head on CPU → ggml for deepseek final norm+head | deepseek | (D=1280, V=129280) scalar matmul for lm_head |
| **P3** | F32 causal mask → F16 | qwen2vl | internvl2 already uses F16 mask |

---

### 3. Math/Formula OCR Runtimes

**Files**: `math_ocr` (1241L), `mixtex_ocr` (1198L), `bttr_ocr` (1134L),
`hmer_ocr` (1013L), `posformer_ocr` (946L), `ppformulanet_ocr` (944L),
`ppformulanet_l_ocr` (1474L)

#### Encoder optimization ranking

| Rank | Runtime | Encoder type | Approach |
|------|---------|-------------|----------|
| 1 | **ppformulanet_l_ocr** | SAM-ViT | ggml graph, batched windows, precomputed RPE |
| 2 | **math_ocr** | DeiT | ggml graph |
| 3 | **ppformulanet_ocr** | HGNetv2 (CNN) | Scalar CPU with shared `core/cpu_ops.h` helpers |
| 4 | **mixtex_ocr** | Swin-Tiny | Scalar CPU with shared helpers |
| 5 | **bttr_ocr** | DenseNet | Scalar CPU with duplicated local helpers |
| 5 | **posformer_ocr** | DenseNet | Scalar CPU with duplicated local helpers |
| 5 | **hmer_ocr** | DenseNet-121 | Scalar CPU with duplicated local helpers |

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| ggml graph encoder (SIMD matmuls) | ppformulanet_l, math_ocr | Vision layers computed via ggml graphs |
| Batched windows in ggml graph | ppformulanet_l | All 16 windows processed in parallel |
| Precomputed RPE lookup tables at init | ppformulanet_l | `get_rel_pos()` done once, stored per-layer |
| Cross-attention K/V pre-computation | ALL 7 runtimes | Projected once from encoder output before decode loop |
| Self-attention KV cache | ALL except hmer (GRU) | Per-layer growing cache for autoregressive decoding |
| Dequant cache | math_ocr, bttr, hmer, posformer | Avoids redundant F16→F32 conversion |
| Pre-cached embeddings before decode loop | math_ocr | Token + position tables dequantized once |
| Folded BatchNorm | hmer | BN params pre-folded into conv weights |
| Beam search | bttr_ocr | Full beam search with length normalization |
| Bilinear image resize | bttr, hmer, posformer | Higher quality than nearest-neighbor |
| GELU as tanh approximation | ppformulanet | Avoids expensive `erf()` |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | DenseNet encoder → ggml graphs or im2col+GEMM | bttr, posformer, hmer | All convolutions are 7-nested-loop scalar — dominates runtime |
| **P0** | Swin encoder → ggml graphs | mixtex | 12500-token window attention is scalar O(N^2*D) per window |
| **P0** | HGNetv2 CNN encoder → ggml | ppformulanet | 57M-param CNN at 384x384 via scalar `conv2d_cpu` |
| **P1** | Add beam search | mixtex, math_ocr, hmer, posformer, ppformulanet, ppformulanet_l | Only bttr has it; beam width=3 helps math OCR accuracy significantly |
| **P1** | Migrate duplicated helpers to `core/cpu_ops.h` | bttr, hmer, posformer | ~300 lines of duplicated conv2d/relu/layernorm/linear in each |
| **P1** | Cache dequantized weights at init | mixtex, ppformulanet, ppformulanet_l | `to_f32()` called per-block per-call, same weights every time |
| **P1** | ppformulanet_l: scalar decoder → ggml | ppformulanet_l | Encoder is ggml-optimized but 8-layer D=512 decoder is still scalar |
| **P2** | Pre-compute attention masks (shifted windows) | mixtex | Recomputed from scratch per block — deterministic for fixed dims |
| **P2** | Pre-compute 2D positional encoding | bttr, posformer | sinf/cosf/powf recomputed every inference call |
| **P2** | ggml context reuse across layers | ppformulanet_l | New 8MB context allocated and freed for each of 12 layers |
| **P2** | Global dequant cache → per-context | math_ocr | Global static `unordered_map` is thread-unsafe |
| **P2** | Nearest-neighbor → bilinear resize | math_ocr, mixtex, ppformulanet, ppformulanet_l | 4 of 7 runtimes use nearest-neighbor |
| **P3** | bttr beam search: top-K selection instead of full sort | bttr | O(V*beam_width) candidates created then sorted |
| **P3** | hmer coverage conv per step | hmer | conv2d(256,256,3x3) per decoder step — expensive attention mechanism |

---

### 4. Embedding & NER Runtimes

**Files**: `decoder_embed` (1638L), `vit_embed` (674L), `clip_text_embed` (433L),
`cnn_embed` (1323L), `lfm2_embed` (722L), `bert_ner` (321L), `gliner_ner` (1703L),
`lilt_kie` (676L), `fireredpunc` (802L), `bidirlm_vision` (692L), `bidirlm_audio` (129L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Flash attention | vit_embed, clip_text, lfm2_embed, gliner_ner (GQA), fireredpunc, decoder_embed (batch path) | `ggml_flash_attn_ext` |
| Fused QKV weights | vit_embed, bidirlm_vision | Q/K/V concatenated at load → single matmul |
| Batched encoding with prefix sharing | decoder_embed | Detects shared prefix, deduplicates (B-1)*P tokens |
| F16 attention mask | decoder_embed, clip_text | Halves mask memory |
| Fused soft_max_ext | decoder_embed (batch), bidirlm_vision | Scale + mask + softmax in one ggml op |
| BN folding at load | cnn_embed | BatchNorm params pre-folded into affine scale+shift |
| LoRA hot-swap | decoder_embed | CPU-side merge/unmerge with lazy base weight snapshot |
| Pre-cached BiLSTM weights | gliner_ner | Dequantized to F32 once at init |
| DeBERTa disentangled attention | gliner_ner | Full c2c + c2p + p2c implementation |
| Pre-computed bilinear position interpolation | bidirlm_vision | Corner indices + weights baked once per encode |
| Pre-computed 2D RoPE cos/sin | bidirlm_vision | Full tables on CPU, passed as graph inputs |
| Generic ONNX graph replayer | cnn_embed | Can replay arbitrary CNN topologies from metadata |
| `ggml_gallocr` reuse | lfm2_embed | Allocator stored on context, reused across calls |
| Gemma3 numerical stability | decoder_embed | RMSNorm output clamped to [-1000, 1000] for F16 safety |
| Delegates to CrispASR encoder | bidirlm_audio | Reuses existing optimized audio encoder |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | No flash attention in single-text path | decoder_embed | Uses manual Q@K+softmax+V; only batch path uses flash_attn |
| **P1** | BiLSTM is fully scalar | gliner_ner | 4*512*1024 + 4*512*512 ≈ 3M MACs per timestep, no SIMD/BLAS |
| **P1** | Layer fusion matmuls are scalar | gliner_ner | [1024, 1024] output projection per token via scalar loops |
| **P1** | Graph rebuilt every call | ALL 11 runtimes | Graph structure is identical for same seq_len; should cache |
| **P1** | No flash attention | bidirlm_vision, lilt_kie | Manual Q@K+softmax+V despite amenable structure |
| **P2** | Fuse QKV in clip_text | clip_text_embed | 3 separate matmuls where 1 would suffice |
| **P2** | Scalar L2 normalization | decoder_embed, vit_embed, lfm2_embed, bidirlm_audio | Could use SIMD or ggml ops |
| **P2** | Scalar dense projection matmul | decoder_embed | Triple-nested scalar loop for post-pooling projection |
| **P2** | DeBERTa relative position expansion O(T^2*H) | gliner_ner | Creates [H, T*T] F32 tensor on CPU every call; T=200 → 117MB |
| **P2** | `ggml_gallocr` rebuilt per call | vit_embed, clip_text, cnn_embed, fireredpunc, gliner_ner, decoder_embed | Only lfm2_embed reuses the allocator |
| **P3** | No batched encode API | vit_embed, clip_text, lfm2_embed, bert_ner, gliner_ner, lilt_kie, fireredpunc | Single-input only |
| **P3** | Conv1D kernel cast every call | lfm2_embed | `ggml_cast` adds a graph node per invocation; pre-cast at load |
| **P3** | F32 attention mask | bidirlm_vision | F16 would halve the 20MB mask for 2304 tokens |
| **P3** | WordPiece re-tokenization for word counting | fireredpunc | Re-tokenizes each word to count subtokens; track during initial pass |

---

### 5. Super-Resolution & Image Restoration Runtimes

**Files**: `dat_sr` (1396L), `hat_sr` (945L), `swinir_sr` (695L), `esrgan_sr` (252L),
`safmn_sr` (438L), `pan_sr` (383L), `tbsrn_sr` (533L), `text_sr` (670L),
`nafnet_denoise` (564L), `scunet_denoise` (792L), `restormer` (749L),
`instructir` (469L), `adair` (944L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Tiling with Hann-window overlap blending | dat_sr, hat_sr, swinir_sr, pan_sr, text_sr, restormer | Raised-cosine window prevents seam artifacts |
| Dequant cache | dat_sr | `dequant_cache` avoids re-dequantizing the same tensor |
| Ping-pong buffer reuse | esrgan_sr, nafnet, text_sr | Swap buf_a/buf_b to avoid allocation per layer |
| BatchNorm fusion at inference | dat_sr, tbsrn_sr | Pre-computed `scale = weight / sqrt(var+eps)` |
| GPU-safe tensor reads | 12 of 13 runtimes | `ggml_backend_tensor_get()` instead of `tensor->data` |
| Transposed attention (C×C not HW×HW) | restormer, adair | Efficient for high-resolution images |
| Scratch buffer reuse | safmn_sr, swinir_sr, tbsrn_sr, nafnet, text_sr | Pre-allocated tmp buffers passed to blocks |
| Bicubic upscale with Keys kernel | text_sr | Proper reconstruction filter |
| Single-tile fast path | dat_sr | Skips tiling overhead for small images |
| FORCE_CPU env var | Most runtimes | Debug override for backend selection |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | No SIMD anywhere — all conv/linear/attention is scalar | ALL 13 runtimes | conv2d accounts for ~80% of compute; 5-10x with SIMD |
| **P0** | No weight dequant caching | 12 of 13 (all except dat_sr) | Re-dequant same weights per-block per-image |
| **P0** | Per-pixel vector allocations in scunet | scunet_denoise | `std::vector<float>` allocated per spatial position in LN and MLP — 100K+ heap allocs per Swin block |
| **P1** | No tiling support | esrgan, safmn, nafnet, scunet, instructir, adair | OOM or poor cache behavior for images >512px |
| **P1** | Batch linear/GEMM instead of per-token calls | dat_sr, swinir_sr, hat_sr, scunet | QKV as N separate `linear_cpu` calls → one GEMM |
| **P1** | Redundant CHW↔HWC layout conversions | dat_sr, hat_sr | 30-50 full-image transposes per forward pass |
| **P2** | Pre-compute attention masks and position biases | hat_sr, swinir_sr, dat_sr | Rebuilt per tile despite being deterministic for fixed size |
| **P2** | `ctx->get()` unbounded wbufs growth | hat_sr, swinir_sr, pan_sr, text_sr, nafnet, restormer, instructir, adair | Appends new dequantized vector every call, never reuses |
| **P2** | Fuse BatchNorm into conv weights at model load | dat_sr, tbsrn_sr | Currently applied as separate pass after conv |
| **P2** | instructir SCA weight dequant inside per-channel loop | instructir | Re-dequantizes entire weight matrix C times |
| **P3** | scunet conv_transpose2d scatter-add | scunet | Writes to output with random access — cache-unfriendly |
| **P3** | PE2D recomputed every SRB iteration | tbsrn_sr | `tbsrn_pe2d(64, ...)` called 5 times with identical params |
| **P3** | restormer rst_layernorm_bf computes variance twice | restormer | First sum-of-squares pass is dead work |
| **P3** | adair FFT zero-pads to next power of 2 | adair | 129→256, wastes ~2x compute; mixed-radix would help |

---

### 6. Detection, Pipelines & Utilities

**Files**: `layout_detect` (1872L), `surya_det` (1341L), `ocr_detect` (947L),
`parseq_ocr` (810L), `tesseract_lstm` (663L), `ocr_pipeline` (169L),
`ocr_orchestrator` (940L), `ocr_render` (600L), `table_parse` (393L),
`kie_pipeline` (316L), `cc_detect` (280L), `image_preprocess` (520L),
`classical_preproc` (690L), `face_align` (193L), `dewarp` (309L),
`scan_cleanup` (572L), `morph_fast` (312L), `tps_warp` + `tps_locnet` (508L),
`pdf_info` (739L), `pcs` (817L), `tokenizer*` (764L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| ggml graph for full backbone+encoder | layout_detect, ocr_detect | ResNet + FPN + attention all in one graph |
| ggml graph for ViT encoder | parseq_ocr | 12-layer ViT with flash_attn |
| ggml graph for XLM-RoBERTa | pcs | 12-layer encoder with flash_attn |
| Hybrid graph + scalar forward | surya_det | Stages 0-2 ggml graph, LiteMLA scalar |
| Flash attention | layout_detect (AIFI), parseq_ocr, pcs | `ggml_flash_attn_ext` where applicable |
| Dequant cache | parseq_ocr | Maps tensor data pointers to F32 buffers |
| All weights dequanted at load | tesseract_lstm | Zero runtime dequant cost |
| BN pre-folded into conv | surya_det | Eliminates BN arithmetic at runtime |
| Union-find with path compression | cc_detect | O(α(N)) CC labeling |
| 32-pixel word-level morphology | morph_fast | 32x throughput vs per-pixel ops |
| Integral images for Sauvola binarization | scan_cleanup, classical_preproc | O(1) per-pixel mean/variance |
| Separable bicubic resize with anti-aliasing | image_preprocess | Matches torchvision quality |
| `__builtin_popcount` for row sums | classical_preproc | Hardware-accelerated bit counting |
| partial_sort for top-K queries | layout_detect | Avoids full sort |
| `std::nth_element` for thresholds | surya_det | O(N) partial sort |
| Viterbi DP for SentencePiece | tokenizer_spm | Optimal segmentation |
| Convex hull + rotating calipers | ocr_detect | Min-area rotated rectangles |
| Lazy engine loading | ocr_orchestrator | Unused engines have zero overhead |
| Early exit for flat pages | dewarp | Skip warp if max_disp < 2px |
| DPI estimation via PDF metadata | ocr_orchestrator | Auto-selects SR tier |
| Pre-computed resampling weights | image_preprocess | Index + weight arrays built once per dimension |
| Gaussian elimination with partial pivoting | tps_warp | Robust TPS solve |
| Cross-attention K/V pre-computation | parseq_ocr | Computed once from encoder output |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | Deformable cross-attention is CPU-scalar | layout_detect | 6-nested-loop bilinear sampling — dominates decoder runtime |
| **P1** | LSTM gates have no SIMD | tesseract_lstm | Hot inner dot-product loop is unvectorized |
| **P1** | LiteMLA attention is CPU-scalar | surya_det | O(N^2 * head_dim) scalar matmuls (stubbed graph path) |
| **P1** | Sequential region recognition | ocr_pipeline, table_parse | Each crop recognized individually — batch into single encoder pass |
| **P1** | Image loaded from disk multiple times | ocr_orchestrator | stbi_load called N times for N engine attempts on same image |
| **P1** | Cleaned image written to temp PNG then re-loaded | ocr_orchestrator | PNG encode/decode round-trip; pass pixel buffer directly |
| **P1** | min_pool/max_pool are O(K^2) per pixel | scan_cleanup | K=51 means ~2500 comparisons/pixel; deque-based sliding window → O(1) amortized |
| **P2** | Otsu threshold duplicated 6 times | table_parse, cc_detect, classical_preproc, scan_cleanup, dewarp | Extract to `core/` shared utility |
| **P2** | Per-step allocations in parseq AR decode | parseq_ocr | ~15 vectors allocated/freed per decode step × 26 steps |
| **P2** | TPS warp evaluates all N control points per pixel | tps_warp | Coarse grid + bilinear interpolation of displacement field |
| **P2** | No multithreading in pixel-level ops | image_preprocess, dewarp, scan_cleanup, face_align | All pixel loops single-threaded despite accepting n_threads |
| **P2** | BPE merge is O(N^2 * V) | tokenizer_bpe | Priority queue → O(N log V) |
| **P2** | Locnet weights re-dequantized every predict call | tps_locnet | Cache F32 weights at load time |
| **P2** | Hough voting O(edge_pixels * n_angles) | scan_cleanup | Quadratic for dense images |
| **P3** | WordPiece uses linear scan for longest match | tokenizer | Trie would be O(len) |
| **P3** | PDF parser loads entire file into memory | pdf_info | mmap would handle large PDFs |
| **P3** | Debug fprintf unconditionally in production | layout_detect, surya_det, ocr_detect | Should be gated behind verbosity level |
| **P3** | `std::vector<float>` return-by-value in hot paths | surya_det, parseq_ocr, face_align | Allocates and copies large buffers; use pre-allocated workspaces |

---

### Cross-Cutting Summary

#### What the codebase does well

1. **ggml graph acceleration** — The best runtimes (internvl2, glm_ocr, decoder_embed batch)
   use ggml compute graphs for all heavy math, getting SIMD-optimized matmuls and
   automatic GPU dispatch for free.

2. **Flash attention** — Used in 10+ runtimes for fused Q@K+softmax+V with proper scaling.

3. **Quantized model support** — Universal GGUF loading handles F32/F16/Q8_0/Q4_K
   transparently. Cosine similarity vs F32 is >0.995 for Q8_0 across all models.

4. **Memory-mapped weights** — `mmap`/`MapViewOfFile` in `gguf_loader.cpp` avoids
   copying multi-GB model files into userspace.

5. **KV cache** — Most autoregressive decoders implement proper KV caching with
   incremental append (best: internvl2's F16 ggml-resident zero-copy cache).

6. **Tiling with overlap blending** — SR/restoration runtimes handle arbitrary image
   sizes via overlapping tiles with Hann-window blending.

#### Top 10 highest-impact optimization opportunities

| # | Opportunity | Scope | Status |
|---|------------|-------|--------|
| 1 | **SIMD in `core/cpu_ops.h` helpers** | 30+ runtimes | **DONE** — `dot_product()` AVX2+FMA/NEON, 710 FMA instructions |
| 2 | **Dequantized weight caching** | ~40 runtimes | **DONE** — `DequantCache` in core; migrated smoldocling + granite |
| 3 | **Adopt F16 ggml KV cache** (internvl2 pattern) | 6 VLM decoders | Partial — pix2struct (F32 vector), lightonocr, granite, smoldocling, qwen2vl done |
| 4 | **Flash attention everywhere** | 5 runtimes | **DONE** (3/5) — decoder_embed, bidirlm_vision, pix2struct. lilt_kie incompatible (BiACM). deepseek pending. |
| 5 | **Move remaining scalar encoders to ggml graphs** | 7 encoders | **DONE** (pix2struct). DenseNet (bttr/posformer/hmer) and Swin (mixtex) remain. |
| 6 | **Batched prefill for VLM decoders** | smoldocling, granite | **DONE** — smoldocling F16 KV + batched prefill, granite projector+LLM graphs |
| 7 | **Graph caching** | All 60+ runtimes | Pending (architectural) |
| 8 | **Pre-compute RoPE frequency tables** | core_vlm users | **DONE** — `RoPEFreqTable`; migrated smoldocling + granite |
| 9 | **Batch linear → GEMM** in SR attention | 5 SR runtimes | **DONE** — dat_sr, swinir_sr, hat_sr, scunet, mixtex via `linear_batch_cpu` |
| 10 | **Eliminate per-step heap allocations** | 12 runtimes | **DONE** — pix2struct, bttr, posformer, hmer, math, parseq, mha_1q_cpu, vlm_attention, layernorm2d |
| 11 | **BPE tokenizer O(N²) → O(N log N)** | bpe.h + tokenizer_bpe | **DONE** — linked list + priority queue |
| 12 | **`std::unordered_map` for tensor lookup** | gguf_loader + 14 files | **DONE** — O(1) avg lookups |

#### Architectural recommendations

1. ~~Centralize dequant caching in `core/cpu_ops.h`~~ — **DONE**: `DequantCache` struct
   added. Migrated in smoldocling_ocr and granite_vision_ocr.

2. ~~Add SIMD to `linear_cpu` and `conv2d_cpu`~~ — **DONE** for `linear_cpu` (AVX2+FMA
   and NEON via `dot_product()`). `conv2d_cpu` still scalar (needs im2col restructure).

3. **Standardize KV cache on internvl2 pattern**: F16 ggml backend tensors with
   `ggml_view` + `ggml_cpy` writes. Port this to all VLM decoders.

4. **Migrate remaining duplicated helpers**: bttr, hmer, and posformer each have ~300
   lines of duplicated conv2d/relu/layernorm/linear. Migrate to `core/cpu_ops.h`.

5. **`ggml_gallocr` reuse** — DONE. Persistent gallocr on context for 7 engines
   (vit_embed, clip_text_embed, parseq_ocr, cnn_embed, ocr_detect, surya_det,
   layout_detect). LFM2 migrated to `ggml_backend_sched` + T-bucketing.

---

## Runtime Optimization Audit — Re-verification (2026-07-11)

Full re-sweep of the codebase against the June audit above. **Nearly every P0/P1
the June audit flagged has since been executed.** The tables above are retained
for history but are now stale: read this section for the current state. Findings
here were verified against current code (`git` HEAD), not carried from the doc.

### June-audit claims that are now WRONG (code has moved on)

| June claim | Current reality | Evidence |
|---|---|---|
| `conv2d_cpu` "still scalar / needs im2col restructure" (arch-rec #2) | Per-patch gather into a `thread_local` buffer + SIMD `dot_product` per output channel, with a hoisted interior-fast-path boundary check. Effectively single-patch im2col+SIMD. | `core/cpu_ops.h:345-400` |
| `mel.cpp` projection "naive triple-loop matmul" | `core_cpu::dot_product` fast path for the contiguous layout; scalar retained only for transposed/accumulator cases | `core/mel.cpp:116-117` |
| VLM decoders "F32 CPU KV re-uploaded each step / CPU-scalar" (qwen2vl, deepseek, smoldocling, granite, pix2struct) | All default to ggml graphs with **F16 device-resident KV** + `ggml_flash_attn_ext`. Scalar is an env-gated fallback. | qwen2vl_ocr.cpp:1091-1092,2412; deepseek_ocr2.cpp:154,1604-1620; granite_vision_ocr.cpp:626-627; smoldocling_ocr.cpp:685-686; pix2struct.cpp:347 |
| SR "No SIMD anywhere / no dequant caching 12-of-13 / no tiling" | 11/13 SR runtimes on ggml graphs; DequantCache fleet-wide (12 files); Hann-window tiling universal | esrgan_sr.cpp:362; instructir.cpp:164; scunet_denoise.cpp:327 |
| decoder_embed "no flash in single-text path" | Single-text path (B≤1) now calls `ggml_flash_attn_ext` | decoder_embed.cpp:1196,1421 |
| gliner "BiLSTM fully scalar" | Gate matmuls use `core_cpu::dot_product` (SIMD); only the per-timestep sequencing is inherent | gliner_ner.cpp:915-916 |
| tesseract "LSTM gates no SIMD" | SIMD via `core_cpu::dot_product` | tesseract_lstm.cpp:256-257 |
| Math-OCR scalar encoders (DenseNet bttr/hmer/posformer, HGNetv2 ppformulanet, Swin mixtex) | DenseNet/HGNetv2 → ggml graphs (default); mixtex projections on ggml (window attention still scalar — see gaps) | bttr/posformer/hmer/ppformulanet; mixtex_ocr.cpp:126 |

### Verified DONE since June (net-new work)

- **Device-resident F16 KV cache** across the VLM decoder set; **persistent
  single decode graph** in deepseek_ocr2 and math_ocr (TrOCR ~4×). Other VLM
  decoders (qwen2vl/granite/smoldocling) have device-resident KV but still
  rebuild the decode graph per step.
- **WebGPU/WASM tier** (OCR build): ~950 lines of authored WGSL kernels
  (LayerNorm, IM2COL/CONV_2D/POOL_2D/CONV_TRANSPOSE_2D/UPSCALE/ARANGE) landed in
  the pinned `ggml` submodule. Detection ~60×, det+rec pipeline ~1.8×, ~2.8×
  total vs SIMD-CPU. Multithreaded via `--proxy-to-pthread`.
- **Beam search** added to math_ocr, bttr, ppformulanet, ppformulanet_l.
- **imatrix** quant rollout (20 models); confirmed **zero inference cost** —
  the eval-callback early-returns unless `CRISPEMBED_IMATRIX_OUT` is set
  (`imatrix.cpp:131`). It is a calibration/quant-time artifact only.

### True remaining gaps (2026-07-11)

| P | Area | Gap | Impact |
|---|---|---|---|
| **P1** | Graph caching (all runtimes) | **0 runtimes reuse the built cgraph.** Decoders rebuild + `sched_reset`+`alloc_graph` per token; device KV landed but the graph around it is rebuilt each step. Blocked on WebGPU (traps `unreachable`) — needs per-backend gating (safe on Metal/CPU). | #1 unrealized lever |
| **P1** | layout_detect | Deformable cross-attention still CPU-scalar 6-nested bilinear grid-sample | Dominates DETR decoder; the one surviving June P0 |
| **P1** | text_sr | ~~Only SR engine still fully scalar (`tsr_conv2d`)~~ — conv now delegates to SIMD `core_cpu::conv2d_cpu` (this branch). Remaining: a full ggml graph for GPU offload | Convs SIMD-accelerated; GPU path still open |
| **P1** | WebGPU embedding tier | `build-embed-wasm.sh` has no `--webgpu` path — text embeddings are CPU-only in browser | Whole embedding browser tier misses the proven GPU path |
| **P2** | scunet_denoise | Swin blocks still scalar; only SR engine without DequantCache (`scunet_denoise.cpp:32`) | Transformer half unaccelerated + repeated dequant |
| **P2** | SR-on-GPU (whole family) | **Correction:** the ENTIRE SR family runs conv on a CPU-only `enc_sched` (`swinir_sr.cpp:447` prints `ggml_conv_2d (CPU sched)`); dat/hat/swinir use `init_best` only to LOAD weights, then copy dequantized weights into a CPU-resident context. esrgan/safmn/restormer/instructir just skip that copy. There is no GPU sibling to match — SR-on-GPU is unsolved research (Metal `ggml_conv_2d` + GPU-resident weight/graph path), not a residency toggle | Reprioritized down |
| DONE | safmn honor `n_threads` | Was hardcoded to a 1-thread conv sched (`safmn_sr.cpp:255`); now honors `-t N` like siblings | **~2.3×** (16.2s→7.1s, 8-core Mac), bit-identical output |
| **P2** | mixtex_ocr | Swin window attention still scalar (`mixtex_ocr.cpp:126`) | Encoder O(N²·D)-bound |
| **P2** | qwen2vl/granite/smoldocling | Decode graph rebuilt per step (KV device-resident, but not the persistent-graph pattern deepseek/math use) | Per-step build/launch overhead |
| **P2** | deepseek_ocr2 | LLM/enc flash are opt-in default-off (measured slower on CPU); re-benchmark on Metal/CUDA | Backend-dependent |
| **P3** | Build/infra | No LTO/IPO; `GGML_BLAS=OFF` (Accelerate not guaranteed for CPU-fallback matmul); `--gpu-backend` ignored (`crispembed.cpp:81` calls `init_best()` directly); app-level OpenMP possibly unlinked; Metal F16 mul_mm guard in only 5/~40 GPU files | Broad low-effort |
| **P3** | Misc | ocr_orchestrator PNG round-trip + N reloads; gliner DeBERTa rel-pos [H,T²] ~117MB/call; ppformulanet_l decoder scalar; `conv2d_cpu` not GEMM-batched/multithreaded | Localized |

### Highest-ceiling paths forward

1. **Decode-step graph cache** (per-backend gated) — cache the *decode-step*
   graph, not the encoder graph (encoder caching is a measured dud + a GPU
   use-after-free landmine). Templates: the `sched_reserve`+T-bucket pattern in
   the text encoder and lfm2.
2. **ggml-metal ICB (indirect command buffer) replay** — Metal decode is
   per-op-dispatch bound; CUDA-graph capture already solves the CUDA side.
3. **Finish residual scalar kernels** (layout_detect deformable, text_sr, scunet
   Swin, mixtex window attn) and upgrade `conv2d_cpu` per-patch → true
   im2col+GEMM (batch all output channels) + multithread.

---

## VLM OCR Benchmarks (Intel Xeon Skylake, 4 threads, CPU-only)

### Qwen3-VL-2B-Instruct (q4_k, 1.5 GB)

End-to-end OCR on 800×300 invoice image. `QWEN_DBG=1` for per-stage timing.

| Setting | Patches | Vision | Prefill | Decode/step | Quality |
|---------|---------|--------|---------|-------------|---------|
| Default (max_pixels=16M) | 900 (18×50) | 24.5s | 35.3s | 5.0s | 5/5 lines |
| `CRISPEMBED_MAX_PIXELS=65536` | 208 (8×26) | 15.0s | 21.7s | — | 4/5 lines |

**Speedup**: 1.6× faster vision+prefill (60s → 37s) at minor quality loss.

`CRISPEMBED_MAX_PIXELS` reduces input resolution before patch extraction.
Useful for CPU-only deployment where speed matters more than pixel-perfect OCR.
Applies to all VLM OCR engines that use `image_preprocess.cpp`.

## Local M1 Metal OCR engine sweep (2026-07-31)

Command: `python3 tests/ocr_engine_benchmark.py --repeats 1 --timeout 45
--output /tmp/crispembed-ocr-benchmark.json`.  These are cold end-to-end
process times on the local Apple M1; they include model loading and should not
be read as steady-state service throughput.  Quality is scored only against
the manifest's known fixture text.

| Engine | Status | Cold ms | Quality |
|---|---:|---:|---|
| GOT-OCR2 | ok | 15,662 | exact |
| GLM-OCR | ok | 32,884 | exact |
| InternVL2-1B | ok | 24,908 | CER 0.540 (prompt text included) |
| Qwen2-VL-3B | timeout/error | 70,757 | no transcript before 45 s |
| LightOnOCR | ok | 31,561 | unscored; plausible transcript |
| MixTeX | ok | 7,523 | exact specialist formula |
| Flova | ok | 16,153 | exact specialist LilyPond |
| Pix2TeX | ok | 5,520 | exact specialist formula |
| Texteller-3 | ok | 11,403 | CER 7.293; unusable on this fixture |
| Tesseract-LSTM line-crop pipeline | ok | 7,552 | CER 0.040; 10 DBNet regions, all recognized |
| PARSeq-tiny | ok | 921 | unscored full-page smoke (`Gooducalicanos.com`); scene-line recognizer |

The proper DBNet+TrOCR Q8 pipeline remains the ordinary document baseline:
10/10 regions, 10/10 recognized regions, 8.05 s cold on the same M1 run. The
Tesseract result is now measured through the actual DBNet→line-crop→LSTM
pipeline: 10 regions, all recognized, 7.55 s cold / 8.09 s warm, CER 0.040
(punctuation-only drift). PARSeq remains recognizer-only and still needs a
line-crop orchestrator benchmark.

The manifest contained 51 entries: 8 completed, 1 timed out, and 42 explicit
skips because a sample or local model was unavailable.  This is a coverage
report, not a claim that the skipped engines are unsupported.  The reusable
driver stores all output and stderr tails in JSON for follow-up runs.

### Tesseract reference parity and gated page-segmentation cost (2026-08-01)

This is a same-fixture quality/cost cross-check on `scan_strip.png`, not a
claim that all full-page Tesseract behavior is matched. Official timings are
stock Tesseract CLI TSV wall time; native timings are the instrumented
detector→group→crop→recognizer stage total. The native subprocess elapsed time
also includes test-binary/model setup and is therefore not used as the pure
pipeline comparison.

| Path | Output quality vs official | Official wall ms | Native stage ms | Native result |
|---|---|---:|---:|---|
| Legacy/fallback | Best current native path, but CER/WER `0.0179/0.0841`; confidence `0.895` vs `0.9108` | 315.9–349.9 | 310.7 | 12 regions, 567 chars |
| Component | Worse: CER/WER `0.0322/0.1121` | 315.9–349.9 | 266.8 | 12 regions, 569 chars |
| Baseline | Same CER/WER as legacy, no quality gain; IoU lower | 315.9–349.9 | 282.2 | 12 regions |
| Projection | Worse: CER/WER `0.0250/0.1121`; IoU best but text worse | 315.9–349.9 | 360.1 | 12 regions |

Native recognition dominates the stage (`260.3–353.8 ms`); detector and crop
were approximately `3–4 ms` each. A worker sweep retained identical CER/WER
and measured native stage totals of `690.3 ms` at one worker, `300.7 ms` at four,
and `292.1 ms` at eight. The immediate performance TODO is recognizer batching,
graph/weight reuse, and fair warm-run measurement; the detector is not the
current bottleneck. The immediate quality TODO is full-page crop/spacing/text
parity: native is not yet output-equivalent even where region count matches.

An activation-scratch reuse prototype is gated by
`CRISPEMBED_TESSERACT_REUSE_SCRATCH`; it preserves CER/WER but measured about
`279.1 ms` versus `282.3 ms` in one paired run, while earlier repeated runs
were `329–338 ms` versus the prior `~300 ms` result. The variance is too large
to claim an improvement, so it is disabled by default and remains an
optimization TODO.

Use `tools/benchmark_tesseract_page.py` for repeated, policy-specific runs;
its summary separates official CLI, native subprocess, native stage, and
recognizer timings and retains every per-run quality record.

The German official-print page remains materially worse: native default is 21
regions vs official 25, CER `0.307`, WER `0.404`, and confidence `0.836` vs
`0.866`. Paired warm/cold timing and per-stage reference timing for this page
remain TODOs. The Fraktur line diagnostic is also not a speed claim because its
available input is a full page under PSM7 rather than an identical transcribed
line crop.

Latest normalized-artifact rerun (current Fraktur Q8 artifact) is worse still:
official Tesseract took `9.34 s` for 25 lines/881 chars at confidence `0.8658`,
while native took `38.69 s` of stage time for 23 regions/1,235 chars at
confidence `0.768`, CER `0.5279`, and WER `0.5390`. Native recognition consumed
`38.34 s`; detection was `102 ms` and crop `250 ms`. The earlier Q8/F16
measurement is retained as historical until artifact and control conditions
are pinned identically. This is an explicit speed and quality blocker.

### Fraktur recognizer precision matrix (same German page/control)

| Recognizer artifact | Native stage | Regions/chars | Confidence | CER/WER | Assessment |
|---|---:|---:|---:|---:|---|
| `frk-q8_0` | 38.69 s | 23 / 1,235 | 0.768 | 0.5279 / 0.5390 | Faster, but worse text |
| `frk-f32` | 102.41 s | 23 / 1,164 | 0.767 | 0.4672 / 0.5461 | Better CER, far too slow |
| `frk-int8-source-q8-candidate` | 64.44 s | 23 / 1,164 | 0.767 | 0.4672 / 0.5461 | F32-like output, still too slow |
| `frk-mixed-lstm0hh-f32` | 23.42 s | 23 / 1,146 | 0.765 | 0.4603 / 0.5390 | Best measured CER/speed frontier, still worse than official |

Official Tesseract remained 25 lines/881 chars. Precision therefore changes
output quality as well as speed; optimizing standard Q8 alone cannot achieve
reference quality. Same-artifact warm/cold benchmarks and recognizer
optimization remain required before selecting the production Fraktur tier.

The mixed-precision candidate is generated with
`models/mix-tesseract-gguf.py`: Q8 remains the default base, while explicitly
selected critical tensors are copied from F32. The selected
`lstm.0.weight_hh` profile is not a production default; it remains gated
until repeat benchmarks, page-region parity, and decoded-text quality gates
improve.

Fresh Miniconda regeneration from `/opt/homebrew/share/tessdata/frk.traineddata`
now gives exact input parity (`cosine=1.0`, both norms `122.453`). Against
now gives exact input parity (`cosine=1.0`, both norms `122.453`). The old Q8
artifact lacked `sample_iteration`, causing the earlier `0.983119` logits
result and seeded-padding mismatch. A freshly converted F32 model reaches
9/9 stages with logits cosine `0.993819`; a mixed Q8/F32 candidate carrying
the recovered seed reaches 9/9 and `0.994876`. Both still decode differently
from Python, so the mixed candidate is not production-accepted. References are
stored at `/Volumes/backups/ai/crispembed-gguf/tesseract-frk-ref-fresh.gguf`
and `tesseract-frk-ref-int8fc.gguf`.

GGUF metadata audit of `/Volumes/backups/ai/crispembed-gguf/` found 46
Tesseract model artifacts: 45 lack `tesseract_lstm.sample_iteration`; only
`tesseract-eng-homebrew-intmeta-f32-sample6352704.gguf` carries it. The missing
seed can change every out-of-bounds Convolve padding value, so those artifacts
require regeneration or metadata-preserving reconstruction before parity
acceptance.

After regenerating the Python reference with exact int8 LSTM arithmetic, the
fresh F32 Fraktur GGUF passes all 9 captured stages exactly (final logits max
error `2.09e-7`) and decoded text matches Python. The seed-preserving mixed
Q8/F32 candidate remains below parity (`logits cosine 0.989655`) and decodes
differently; quantization quality is the remaining blocker.

Quantization policy improvement: `models/quantize.py` now supports repeatable
`--keep-pattern` rules, allowing callers to retain critical recurrent or
output tensors at source precision without changing the established default
quantization behavior. The policy is unit-tested and remains opt-in.

The public-domain fixture smoke path (`tests/ocr_fixture_smoke.py`) exercised
seven CC0/public-domain images through Tesseract plus skew/content detection:
all PNG/JPEG paths passed.  The original TIFF receipt correctly exposed a
format gap (`cannot load`); a PNG derivative is now included for the OCR
pipeline while the source TIFF remains available for a future native TIFF
decoder test.

### Tesseract runtime regression and recovery

A remote-main merge temporarily replaced the int-mode/scratch Tesseract
runtime with an older F32-only implementation. On
`tests/regression/images/scan_strip.png` with the same Fraktur Q8 artifact,
recognition measured `50.15 s` in that regression. Restoring the known-good
runtime and adding LUTs for the existing Tesseract nonlinear interpolation
contract measured `34.32 s`, with unchanged output: 12 regions, 566 chars,
CER `0.03375`, and WER `0.15044`. The required int-mode, LUT, and gated
scratch symbols are now protected by a runtime-contract test. The remaining
speed gap to official Tesseract is still an active TODO.

### Full local matrix comparison (M1 Metal, 2026-07-31)

The expanded manifest sweep completed 11 engines, recorded 2 errors, and
reported 41 explicit non-sample/non-model skips. Representative outputs:

| Engine/lane | Cold ms | Result |
|---|---:|---|
| GOT-OCR2 | 22,073 | exact fox transcript |
| GLM-OCR | 38,086 | exact fox transcript |
| InternVL2-1B | 28,523 | transcript plus prompt wrapper; CER 0.54 |
| Qwen2-VL-3B | 90,113 timeout | no output within limit |
| LightOnOCR | 69,289 | plausible transcript; currently unscored |
| Tesseract via DBNet line crops | 32,041 | 10 regions; CER 0.040 |
| PARSeq | 6,252 | `Gooducalicanos.com`; recognizer-only smoke |
| SmolDocling | 16,334 | text present but duplicated DocTags regions; payload CER 0.86 |
| MixTeX | 13,286 | exact specialist LaTeX |
| Flova | 36,293 | exact LilyPond |
| Pix2TeX | 8,980 | exact LaTeX |
| TexTeller | 18,491 | CER 7.293; unusable on fixture |

SmolDocling is therefore supported and live-tested; its next fix is structural
deduplication/DocTags parsing, not model discovery. Unlimited-OCR's Q4_K stacked
artifact is complete at 2,252,419,328 bytes and now has a successful M1 Metal
run when loaded from the system volume: 45,967 ms total (SAM 15,663 ms, CLIP
2,260 ms, projection/assembly 5,835 ms, decoder 22,205 ms), with two correctly
decoded text regions. The external backup-volume no-copy path (`UOCR_MMAP=1`)
also completes: 40,391 ms cold benchmark time and CER 0.010, with the one
character difference being a harmless title-box coordinate drift. Qwen2-VL is
runnable but did not complete this M1 budget.

### Tesseract seeded model rebuild (2026-08-01)

The 45 unseeded model artifacts were not all independently valid: missing
`tesseract_lstm.sample_iteration` changes seeded out-of-bounds convolution
padding. The 12 installed canonical sources were hash-matched to the old
GGUFs, then freshly converted with Miniconda. The backup store now contains 42
readable `*-seeded.gguf` companions: F32/F16 are freshly converted, while Q8_0
and Q4_K retain the old quantized tensor bytes and receive only the verified
source metadata. All 42 carry a nonzero seed. No speed or OCR-quality claim is
made yet for quantized companions; per-language `crispembed-diff` and decoded
output checks remain TODO. One old Fraktur mixed candidate is truncated and
was excluded.

Chinese seeded F32 is the first decoded-output exception: all 9 stages pass
with aligned magnitudes, but the old native decoder returned an empty string
while the Python reference returned `<141>`. This is a harness-blind recoder
mapping defect, not a graph discrepancy. The native fallback now exposes the
unmapped class; no Chinese OCR-quality or quantized-speed claim is accepted
until recode-beam composition is implemented and tested.

German's apparent quality gap was traced to the Python reference, not the
native graph. Upstream Tesseract's `generate_lut.py` computes the 4096 tanh and
logistic table entries with double-precision `math.tanh/exp`, then stores them
as `TFloat`; the reference had evaluated NumPy float32 nonlinearities directly.
Regenerated German references now pass all 9 stages exactly through the LSTM
and finish at max logit error `3.58e-7`; native and Python both decode ` s.`.

The corrected seeded F32 sweep has exact native/Python decoded parity for all
12 languages on the controlled line. The stale Spanish reference was also
regenerated after the LUT correction; its former one-blank decoded mismatch
was a reference artifact. Korean's prior 6/200 argmax differences disappear
when the production native LUT uses the same generated-table values as
upstream Tesseract; the final Korean run has 0/200 mismatches and exits 0.

All 51 corrected canonical F32/F16/Q8_0/Q4_K files are now uploaded to the
intended `cstr/tesseract-lstm-GGUF` and `cstr/tesseract-frk-GGUF` repositories.
Remote metadata spot-checks confirm nonzero `sample_iteration`; no
`mlx-community` upload was made.

### Tesseract cached-int8 recurrent kernel gate

On the same scan-strip input and `tesseract-frk-q8_0.gguf`, cached and
uncached int-mode decoding both returned `SEEEES`. The cached path measured
`35.4 ms` LSTM time versus `1,035.6 ms` with
`CRISPEMBED_TESSERACT_DISABLE_INT_CACHE=1`, a `29.3x` speedup with identical
decoded output. Cached mode is therefore the default; the environment gate is
retained for parity diagnostics and alternate architectures.

Full-page validation on `scan_strip.png` confirmed the same result: cached and
uncached paths both produced 12 regions/566 chars with CER `0.03375` and WER
`0.15044`. Cached native stage time was `22.11 s` versus `157.59 s` uncached,
for a `7.1x` speedup; detect plus crop was only `46.1 ms` cached. The remaining
Fraktur page-quality gap is therefore in recognition/output parity, not DBNet
or crop throughput.

The comparator now stores both normalized decoded strings. The scan-strip
official/native pair is 451/566 chars with CER `0.03375`; representative
differences are `50`→`80`, `ay`→`8ay`, capitalization (`Such`/`such`,
`Scheme`/`scheme`), and punctuation/hyphen spacing. This confirms the next
quality work should inspect crop geometry and decode semantics, not detector
throughput.

### Tesseract crop-border A/B (2026-08-01)

The Fraktur line crop now has an opt-in `CRISPEMBED_TESSERACT_CROP_PAD` gate;
the default remains 2 pixels. On `scan_strip.png`, all candidates produced 12
regions, so this is not a segmentation-count issue:

| Border | Chars | CER | WER | Recognize ms |
|---:|---:|---:|---:|---:|
| 0 px | 570 | 0.07460 | 0.30088 | 7,237.5 |
| 1 px | 567 | 0.04796 | 0.20354 | 6,686.3 |
| 2 px (default) | 566 | 0.03375 | 0.15044 | 9,217.4 |
| 4 px | 571 | 0.03552 | 0.15044 | 10,666.3 |

The 2-pixel crop remains the best measured quality point. The gate is retained
for other scan resolutions and architectures; the next quality TODO is
Tesseract-compatible decode/recoder semantics for the residual substitutions
and punctuation differences.

### Tesseract page-segmentation and beam A/B (2026-08-01)

The existing page-segmentation policies were compared on the same fixture and
official reference. Every policy emitted 12 regions:

| Policy | Chars | CER | WER | Recognize ms | Output result |
|---|---:|---:|---:|---:|---|
| Legacy fallback | 566 | 0.03375 | 0.15044 | 9,217 | baseline native text |
| Projection | 567 | 0.03197 | 0.12389 | 9,661 | best measured WER/CER |
| Baseline matcher | 566 | 0.03375 | 0.15044 | 14,720 | identical quality, slower |
| Projection + beam 8 | 567 | 0.03197 | 0.12389 | 29,748 | text-identical to greedy |

Projection remains opt-in because its CER improvement is small and it does not
reach official output parity; beam width 8 is retained only for diagnostics
because it adds roughly 3x recognition cost without changing text. The next
quality work is line-image/crop geometry and Tesseract decoder semantics.

The line-confidence comparator now accepts `--tessdata-dir` so official TSV
results do not depend on a potentially stale `TESSDATA_PREFIX`. On the valid
German tiny-line fixture with `/opt/homebrew/share/tessdata`, official output
is `1` at word confidence `0.588557`; native greedy is `G` with word confidence
`0.883064`, while beam-8 is `GEIEE` with sequence confidence `0.535476` and
zero fabricated character confidences. The beam contract passes, but text and
greedy confidence calibration are worse than the official reference and remain
TODOs.

With `--require-official-words --require-greedy-text-match`, that same fixture
exits `1`: the official-word gate passes, while the text gate fails. This keeps
the confidence contract from being mistaken for OCR-quality parity.

### Tesseract page-box geometry A/B (2026-08-01)

`CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD` now controls the symmetric expansion of
legacy component rows; the default remains 3 px. On the scan-strip fixture,
both tighter candidates preserved 12 regions and the same decoded text:

| Box pad | Chars | CER | WER | Recognize ms |
|---:|---:|---:|---:|---:|
| 1 px | 566 | 0.03375 | 0.15044 | 17,453 |
| 2 px | 566 | 0.03375 | 0.15044 | 12,155 |
| 3 px (default) | 566 | 0.03375 | 0.15044 | 9,217 |

The box geometry is therefore not the dominant error on this fixture. Keep
the gate for other scan resolutions, but do not change the default or use
tighter boxes as a quality claim.

### Tesseract composed-recorder gate (2026-08-01)

`CRISPEMBED_TESSERACT_RECODE_COMPOSE` now enables exact segmentation of
collapsed CTC classes into serialized multi-code unichar entries. It is
opt-in: the existing single-code fallback remains the production default.
Fraktur default versus opt-in output/confidence is byte-identical on the
controlled line, and a Chinese smoke input passes both modes without a crash;
that smoke did not exercise a multi-code emission, so no quality promotion is
claimed yet.

### Tesseract confidence harness and line calibration (2026-08-01)

The confidence comparator was hardened against non-UTF-8 Tesseract stderr and
stale inherited `TESSDATA_PREFIX` values. On a cropped Fraktur line, official
and native F32 text differed only by one missing space (`1 hey` vs `1hey`).
Official mean word confidence was `0.7060`; native greedy word confidence was
`0.9726`, while beam reported sequence confidence `0.9924` and no character
confidences. This is a calibration/aggregation gap, not evidence for changing
the recognizer weights or beam default.

The page comparator now uses the same explicit tessdata/environment isolation.
The scan-strip baseline is unchanged with that correction: official 12 lines,
113 words, 451 chars versus native 12 regions and 566 chars, CER `0.03375`,
WER `0.15044`.

The page comparator now has an opt-in `--require-text-match` gate and retains
the normalized official/native page strings in its comparison output. The
scan-strip baseline therefore remains explicitly non-green for exact output
parity even though its CER/WER metrics are measurable.

The confidence harness was rerun after rebuilding `test-confidence`, using the
explicit Homebrew tessdata directory and the seeded Fraktur Q8 GGUF. Official
PSM 7 TSV returned `iE` at mean word confidence `0.043433` in `5,881 ms`;
native greedy returned `BEEES` at word confidence `0.884625` in `305 ms`, and
beam-8 returned the same `BEEES` with sequence confidence `0.644788` and zero
per-character confidences in `984 ms`. The official-word check passed, but
decoded text and greedy calibration did not. This is evidence for a remaining
Tesseract decoder/recoder and confidence-aggregation quality TODO, not a
performance acceptance result; the beam path remains diagnostic.

Converter smoke (2026-08-01): Miniconda converted the installed Homebrew
`eng.traineddata` to `/tmp/crispembed-eng-dawg-smoke.gguf` successfully. The
6.6 MiB GGUF contains the three available LSTM DAWG payloads
(`lstm-punc-dawg`, `lstm-system-dawg`, and `lstm-number-dawg`), each with a
base64 payload and SHA-256 metadata. This verifies preservation only; the
artifact is not a promoted backup model and native dictionary scoring is still
unimplemented.

The regenerated DAWG-bearing smoke GGUF loads successfully in the native
runtime and reports `dawg=3`; the live confidence target passed `35/35` checks
on `scan_strip.png`. The decoded smoke text was `Se`; this validates metadata
acceptance only and is not a page-quality or DAWG-parity result.

The native load path now performs the same structural checks in a standalone
DAWG validator. `test-tesseract-dawg` passes the minimal valid edge fixture and
rejects malformed input; this adds negligible load-time validation and no
runtime OCR scoring cost because DAWG traversal remains disabled.

The opt-in system-DAWG prefix filter was A/B tested on the regenerated English
smoke GGUF with recoder beam width 8. Both unfiltered and filtered runs passed
`37/37`, decoded `Se`, and reported sequence confidence `0.562293`; the filter
did not alter this fixture. This is a safety/observability result, not dictionary
quality parity, and the default remains unchanged.

A seeded-artifact page-gate rerun correction (2026-08-01): the earlier 2-box
report was stale binary evidence. After rebuilding `test-ocr-orchestrator`
following the remote pageseg changes, the canonical Q8 DBNet IC15 detector plus
corrected Fraktur seeded F32 and Q8_0 recognizers both emitted 12 boxes/lines
and passed the pipeline gate. Exact text still fails: both runs measured
CER/WER `0.03922/0.13274`; F32 took 12,373 ms total with confidence delta
`0.01647`, and Q8 took 14,560 ms with confidence delta `0.01447`. The remaining
quality gap is punctuation, spacing, and glyph output from line
recognition/decoding, not detector box count or a precision-only failure. The
stale 2-box result is rejected and should not be used as a performance or
compatibility baseline.

The native crop diagnostic now dumps the exact recognizer inputs on demand via
`CRISPEMBED_TESSERACT_CROP_DUMP_DIR`. The rebuilt Q8 scan-strip run produced
12 grayscale crops, with heights 22–32 px and the final crop 76×25 px. This
confirms valid line geometry, but does not yet establish equivalence with
Tesseract CLI's internal line normalization. A direct single-crop CLI A/B was
not accepted because the installed Homebrew Tesseract/Leptonica could not
reopen a valid dumped PNG; repeat after fixing that environment before drawing
quality conclusions.

The diagnostic also emits `crops.tsv`. A verified Q8 run produced 12 records
plus the header; source boxes map to crop sizes 438×22 through 462×32, with a
final 76×25 crop. The first line begins at page `y=0`, so edge clipping is now
an explicit geometry item for the official-Tesseract comparison.

The opt-in vertical ink-trim A/B is a rejected quality optimization: native
recognition improved from 11,351.6 ms to 10,407.1 ms, but CER/WER degraded
from `0.03922/0.13274` to `0.04278/0.14159` and the character delta grew from
116 to 121. Keep `CRISPEMBED_TESSERACT_CROP_TRIM_INK` diagnostic-only.

The component page-box pad A/B is also neutral for quality: with
`CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD=0`, native output remained byte-identical
to the default and CER/WER stayed `0.03922/0.13274` with 12 regions. Do not
count this as a speed win; the isolated run's timing was not stable enough for
an optimization claim.

The existing component-row segmentation A/B is a quality regression: it kept
12 regions but produced CER/WER `0.10873/0.20354` versus the legacy baseline
`0.03922/0.13274`, including a corrupted first line. Keep the component policy
diagnostic-only and do not use the malformed-path run as benchmark evidence.

`tools/compare_tesseract_crop_geometry.py` now provides a reproducible
geometry-only benchmark. The current 12-line run reports mean native-minus-
official deltas `dx=-2.08`, `dy=+1.83`, `dw=+4.33`, `dh=+1.50`; worst rows are
width `+80`, vertical offset `+14`, and height `+12`. These are row-boundary
quality findings, not a measured runtime regression.

The gated row-blob-bounds A/B fixes the largest local geometry error: CER/WER
improved to `0.03209/0.11504`, mean width delta fell to `+2.42`, and worst
width delta fell from `+80` to `+13`, with 12 regions preserved. This is a
quality improvement on scan-strip, but remains diagnostic-only until validated
on more page fixtures; exact output parity still fails.

The per-line page comparator was corrected to group official TSV words by
page/block/paragraph/line rather than by `word_num`. On the corrected
row-blob-bounds run, both paths emit 12 lines and only 3/12 lines match
exactly. The first differing line is line 0 (`<< 4 ...` official versus
`“< A ...` native); lines 4, 7, and 9 match exactly. Overall CER/WER remains
`0.03209/0.11504`, so this is a recognition/crop or decoder-quality TODO,
not a segmentation-count or ordering failure. Native benchmark was
`detect=89.5 ms`, `crop=258.5 ms`, `recognize=17216.3 ms`, `total=17564.4 ms`;
official Tesseract CLI elapsed `47761.8 ms` in this run, but these timings are
not yet a controlled backend-speed comparison.

The first divergent line was checked at the tensor boundary. Native crop 0
was dumped and a Python reference was regenerated from the installed Fraktur
traineddata. `test-tesseract-lstm-diff` passed every captured stage (input,
convolution, conv-FC, maxpool, four LSTM stages, and logits); the lowest
cosine was `0.997755`, with recurrent mine/ref norms `35.8611/35.8704`, and
the native/Python decoded strings were identical. The official Homebrew CLI
cannot reopen local PNG/PGM/TIFF files in this environment, so direct CLI
single-crop confirmation is blocked; the page-level mismatch is nevertheless
localized to official page segmentation/line normalization rather than GGUF
recognition math. Use the comparator's new `--crop-dump-dir` option for fresh
crop manifests. `tools/compare_tesseract_crop_diff.py` now automates the
per-crop Python-reference regeneration and native `test-tesseract-lstm-diff`
run while refusing to overwrite an existing reference.

On the CC0 German printed-document fixture, official Tesseract emitted 28
lines/153 words/897 characters while native DBNet emitted 23 lines/862
characters. CER/WER was `0.32984/0.67974`; native stages measured
`detect=982.4 ms`, `crop=670.0 ms`, `recognize=19594.7 ms`, and
`total=21247.2 ms`. Since five lines are missing or merged before recognition,
index-paired per-line errors are not a valid recognizer benchmark. The page
comparator now reports `alignment_valid=false` when line counts differ. This
fixture is a detector/line-geometry TODO, separate from the crop-level tensor
parity proven on scan-strip.

The comparator now exposes the native Tesseract-like route explicitly with
`--native-pageseg`. On `scan_strip.png`, this route produced 12/12 lines,
CER/WER `0.03209/0.11504`, and 3/12 exact lines. Its native stage timing was
`detect=12.6 ms`, `crop=644.8 ms`, `recognize=11856.4 ms`,
`total=12513.8 ms`. The route is not using DBNet for box generation; its
quality is identical to the established classical row path, so the remaining
gap is page segmentation/line normalization and decoder semantics.

The repeated benchmark wrapper now accepts `--native-pageseg` and records
`detector_route`, preserving the DBNet-versus-native distinction across
multi-repeat timing runs. Its route flag and comparator selection are covered
by the 10-test focused harness.

On the CC0 German page, the explicit native route emitted the same 23 lines
and 862 characters as the DBNet route, versus 28 official lines and 897
characters. CER/WER stayed `0.32984/0.67974`; native timing was
`detect=1014.9 ms`, `crop=605.7 ms`, `recognize=14263.4 ms`,
`total=15885.6 ms`. This is a shared five-line geometry/coverage gap, not
evidence that either recognizer is worse on aligned crops.

The German native crop manifest has 23 rows versus 28 official TSV rows. The
geometry comparator now marks this as `alignment_valid=false` and reports the
number of index-paired rows; its former mean `dy=257.7` was an alignment
artifact, not a measured crop offset. A merge-aware line matcher remains a
detector/geometry TODO before using per-row geometry deltas on this fixture.

The crop comparator now has `--match-by-geometry`: the German native run
matched 23 rows monotonically and exposed five unmatched official rows
(`0,2,3,4,26`). It still exits 1 for the count mismatch, and the resulting
matched deltas remain diagnostic until one-to-many merged-row matching is
implemented.

Source inspection indicates several official rows are nested decorative marks
inside larger text boxes, so the merge report now labels a primary official
box and nested rows. This prevents a speculative production split based only
on TSV row count. On German, native row 0 has primary official index 1 and
nested indices 2 and 4; native row 9 has primary index 13 and nested index 12;
native row 22 has primary index 26 with no fully-contained nested row.

The geometry report now exposes `merged_official_groups` when one native row
covers at least half the vertical extent of multiple official rows. This
separates merge candidates from genuinely missing rows without changing the
production native pageseg policy.

On the German fixture, the report finds merge candidates native `0` →
official `1..4`, native `9` → official `12..13`, and native `22` → official
`26..27`; official row `0` remains unmatched. These are concrete geometry
targets for row-splitting work, not recognizer timing or tensor-parity data.

## AdaIR F16 runtime audit (2026-08-02)

The AdaIR F32 reference path remains valid on the 64×64 diff fixture: ggml
convolution measured cosine `0.999382`, max absolute error `0.027892`, and
about `2.65 s` inference. The scalar-gated F32 path also passed at cosine
`0.999379`, but measured about `16.8 s`, so it remains a correctness fallback,
not a performance path.

The original `adair-5d-f16.gguf` reproduced the backend assertion in the
per-kernel CPU convolution cache. Allocation guards now prevent the process
abort and disable the ggml convolution route for that context, preserving the
existing scalar fallback. The completed F16 fallback runs returned cosine
`0.729509` and max absolute error `0.707725`; therefore crash-freedom must not
be confused with output parity. Timings ranged from roughly `7.3 s` to
`180.5 s` while other large builds were contending for the host, so they are
not suitable as a stable benchmark.

The F32→F16 rebuild made with the repository quantizer was also tested and
produced the same F16 cosine failure. Tensor inspection found representative
metadata changes from `[3,3,3,48]` to `[27,48]` and `[1,1,48,144]` to
`[48,144]`, with a mixture of F32 and F16 tensor types. Raw values for the
sampled weights were close to the F32 source, but the runtime still reported
an allocation size of zero for the F16 kernel descriptor. Explicit CPU buffer
selection and manual buffer allocation did not change the outcome and were
reverted.

Current status: F32 is the only AdaIR precision cleared for release. F16
requires a loader/converter descriptor audit, tensor-level dequant parity
checks, and an end-to-end cosine gate before it can be uploaded or registered.

## AdaIR F16 — root-caused and fixed (2026-08-02, `feat/ocr-followups`)

The audit above closed on the right observation — a zero-size F16 kernel
descriptor — but attributed it to the buffer allocator. It was neither an
allocator nor a converter defect. Two facts the audit had already collected
point at the real cause once they are read together: the header shapes change
(`[1,1,IC,OC]` → `[IC,OC]`) while the *values* do not.

`tools/quantize.cpp` (~line 167) flattens every 4-D F32 conv weight to 2-D
`[IC*KH*KW, OC]` so the output header is valid for a quantized tensor. That is
deliberate and other engines depend on it. But `src/adair.cpp` inferred three
hidden widths from `->ne[3]`, which is `1` on any flattened tensor:
`gdfn_forward`'s hidden width, and the FreModule `rate_conv` and `ChannelGate`
MLP widths. `hidden = 1` makes `half = hidden / 2` zero, so the next 1×1 conv is
built with `ic == 0` — a kernel with zero elements. That is the descriptor the
allocator refused, and with the guards in place it degraded to cos `0.729509`
instead of aborting.

Independent check that the artifacts were never at fault: across 60 randomly
sampled tensors, `adair-5d-f16.gguf` versus `adair-5d-f32.gguf` gives worst
cosine `0.999998` and worst max_abs `1.22e-4` — pure F16 rounding.

Fix: `conv1x1_out_channels()` derives OC from `ggml_nelements(t) / ic`, correct
under both layouts, with fail-loud guards at all three sites.
`ADAIR_LEGACY_NE3_DIMS=1` restores the old `ne[3]` read so both arms live in one
binary.

Measured on the 64×64 `adair-ref.gguf` fixture, same binary, ggml conv path:

| artifact | arm | cos | max_abs |
|---|---|--:|--:|
| `adair-5d-f32.gguf` | default | `0.999382` | `0.027892` |
| `adair-5d-f16.gguf` | default (fixed) | **`0.999383`** | `0.027871` |
| `adair-5d-f16.gguf` | `ADAIR_LEGACY_NE3_DIMS=1` | `0.729509` | `0.707725` |

The f32 number reproduces the audit's `0.999382` exactly, so it is the
regression control, not a re-measurement. **No timings are quoted here**: the
box carried load average 55–127 from parallel agent builds throughout, and the
same 64×64 fixture took `312 s` at f32 against a quiet-box reference of
`2.65 s`. Re-time on a quiet machine before any performance claim.

Remaining before the f16 can ship: upload it to `cstr/adair-GGUF`, add its
SHA-256 to `examples/cli/model_hashes.h`, and repoint the `adair-5d` registry
entry. The runtime no longer blocks it.

End-to-end through the real CLI, not just the diff harness (HARD RULE #3):
`crispembed --adair-model <f16|f32> --adair <96x96 crop of tests/regression/images/fox.png>`
returns rc=0 for both at the same 27,661-byte PPM, and the two restored images
agree at cosine `1.0`, max_abs `1/255`, mean_abs `7e-5`. Output mean/std is
`242.821 / 35.92` for both — real image content, not a blank or saturated frame.

**Which f16 artifacts are exposed at all.** Only the ones `tools/quantize.cpp`
produced. A converter-emitted f16 keeps its 4-D shapes: `surya-det-f16.gguf` has
79 genuinely 4-D F16 tensors (`stem.in_conv.weight` is `[3,3,3,32]`), whereas
`adair-5d-f16.gguf` — quantizer output — has none. So "f16" alone does not
predict the layout; the *producer* does.

**Same bug class, latent, deliberately not fixed here.** `src/surya_det.cpp:700`
(`const int64_t OC = w->ne[3]`) and `src/tps_locnet.cpp:219`
(`net->channels[i] = net->conv[i].w->ne[3]`) read a conv output-channel count off
`ne[3]` the same way. Neither is broken today, because neither ships a
quantizer-produced artifact — but either would misread one the moment someone
runs `crispembed-quantize` over it. `src/cnn_embed.cpp:148` already handles both
layouts (`ggml_n_dims(dw_w) == 2 ? ne[1] : ne[3]`) and is the in-repo precedent.
Left alone on purpose: there is no artifact on this box that would make a fix
verifiable, and shipping an unverified edit is precisely the failure mode this
entry corrects.
