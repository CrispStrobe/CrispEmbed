# Cache-closed first-image layer0 capture

Kaggle execution144613166/version1 passes the bounded test-only native prefill
capture at controller `3b4b5d9cbe01d548a55d97304809f21989aa341d`, compiling the
unchanged historical `a61642056` runtime and ggml `3a27b6c6`. The [selected safe receipt](../tests/results/qwen35-ocr/lightonocr3-4b-book-image-layer0-review.json)
records exact full sources, generated driver/native binary hashes, numerical
comparisons and the33,438-byte raw report SHA256. Arrays, original sequences,
operator access details and large model transports remain private/remote.

CPU-session admission was rejected at the five-session limit. The explicitly
authorized GPU-session fallback supplied separate capacity; the native runtime
was built and executed on CPU with CUDA/Vulkan disabled. Initial cgroup headroom
was31.23GB with no swap. Worker time was351.02 seconds, bootstrap371.37 seconds,
under2400/2700-second limits; these include model transfer, build and observation
and are not production latency measurements. Four preparation controls pass.

All82 retained processor/vision/prefill stage hashes, including actual firstscore,
match historical plannerON bytes. No decoded generation or source/F32-text model
forward occurs. The preserved old full-decode timeout remains incomplete.

The observer replays layer0 across every actual planner tile. Every tile hidden
output matches actual main prefill rows bit-for-bit; previous caches are unchanged,
and final convolution/recurrent state plus policy flags match main layer0 cache.
First image16 rows consume the resulting text4 cache. A separate projected-core
replay matches the actual block's attention, convolution and recurrent bytes.
This closes the captured native cache by actual main continuation, rather than
assuming that equal text hidden outputs imply equal recurrent state.

On these first image16 rows, source's actual252-row image graph and native agree
bit-for-bit on input embeddings, input norm, QKV, Z, alpha/beta, gated GDN values,
attention projection, residual, postnorm and FFN gate/up. Source value-head order
is mapped through the existing authenticated permutation. Earliest recorded
difference is FFN SwiGLU:

| Stage | RMS error | Max absolute error | Source RMS magnitude |
|---|---:|---:|---:|
| SwiGLU |3.04794e-9|1.19209e-7|0.0595794|
| Down projection |3.86278e-9|4.47035e-8|0.0585933|
| Layer0 hidden |5.50806e-9|1.19209e-7|0.212531|

This independently extends the first-text/cached405 same-input boundary to the
first image chunk. It supports a separately gated source-operator FFN arithmetic
candidate; it does not prove a final decoded benefit. The earlier aggregate252-row
image layer0 error (RMS1.43308e-6/max8.72016e-5) exceeds this first16-row result.
Later image chunks/operators remain unobserved here, so their larger residual
cannot be uniquely assigned to SwiGLU or recurrent batching. Source persistent
cache layouts remain unverified and are not compared. Preserve all old gates,
negative evidence and frozen quality floors; no default or release change follows.
