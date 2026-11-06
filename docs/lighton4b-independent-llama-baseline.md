# Independent llama.cpp baseline: source review and frozen execution plan

This is a distinct upstream implementation and published-weight baseline, not a
same-weight CrispEmbed operator test. No benchmark result is claimed yet.

The actual upstream code is pinned to
[`6c01da6278ab6c69ed67d9124806b50d2b299ba7`](https://github.com/ggml-org/llama.cpp/tree/6c01da6278ab6c69ed67d9124806b50d2b299ba7).
Its [`Qwen3_5TextModel`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/conversion/qwen.py#L693)
registers the official model class; [`Qwen3VLVisionModel`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/conversion/qwen3vl.py#L16)
converts its vision projector. The actual C++ graph is
[`llama_model_qwen35::build_arch_graph`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/models/qwen35.cpp).
Generic older LightOnOCR support is insufficient evidence for this Qwen3.5-based4B;
these registrations and graph are the relevant path.

The primary publisher's [model card at immutable8406c78c](https://huggingface.co/richardyoung/LightOnOCR-3-4B-GGUF/blob/8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01/README.md)
provides Q4_K_M/Q6_K/Q8_0 and a separate F16 projector, says it uses an importance
matrix and omits missing MTP weights. It reports the following **publisher claims**:

| Decoder | Text-only Wikitext2 KLD | Top-token match | Perplexity |
|---|---:|---:|---:|
| Q4_K_M |0.0210|92.8%|8.67|
| Q6_K |0.0042|96.6%|8.57|
| Q8_0 |0.0007|98.5%|8.54|

The card identifies40×512-token chunks and a synthetic-invoice Ollama check of
Q4/Q8. It does not provide an exact engine/converter SHA, complete commands,
hardware, raw benchmark receipts, original-checkpoint revision or invoice input
hash. These are useful leads, not independently reproduced OCR measurements.
The accompanying3,626,496-byte importance matrix is published; its data label alone
does not establish OCR-specific calibration coverage. General quality labels on
other [static](https://huggingface.co/mradermacher/LightOnOCR-3-4B-GGUF/tree/d73ad16828c669dce794e96c8368197cd80ef77d)
and [quantized](https://huggingface.co/prithivMLmods/LightOnOCR-3-4B-GGUF/tree/8a5f9c3f8687860e052bf30efa043e33ea1ebcb7)
model listings likewise do not supply a reproducible paired OCR benchmark.

These models cannot be loaded into the current CrispEmbed owner unchanged.
Upstream splits text and vision GGUFs and applies coordinated V-head permutations
in `_LinearAttentionVReorderBase`, transformed A/RMS parameters and its own tensor
naming. Those transformations must remain paired with the upstream C++ graph.
Upstream Q4_K_M is a mixed precision policy in
[`llama-quant.cpp`](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/src/llama-quant.cpp),
including selected higher-bit V/down projections; it is neither our plain Q4_K
nor the exact24-GDN-output recipe. The published Q6_K is a whole decoder recipe,
not our24 selected Q6 matrices. No tensor/permutation or checkpoint-byte equality
is asserted without a separate conversion audit.

The [predeclared protocol](lighton4b-independent-llama-baseline-v1.json) pins every
whole file, immutable model revision and the three already-observed input cases:
form/table/Arabic. Existing original BF16 captures are rehashed and reused; no
original forward or untouched heldout input is requested. The hosted-only
[worker](../tools/ci-heavy/lighton4b_independent_llama_baseline.py) builds only the
pinned upstream server and runs nine sequential fresh contexts, one CPU process
per case/quant, four threads,256 greedy tokens, image-only transcription and
thinking disabled. Q8 decoder with F16 projector is the affordable control;
an8.4GB F16 decoder is excluded from the bounded standard CPU worker.

Before inference, the upstream rendered chat must exactly equal the frozen source
wrapper. Tokenizing its image-expanded form must reproduce each original `.npy`
input-ID SHA; the actual request must use the same expected prompt/image token
count. Canonical PNG transport preserves Pillow-decoded RGB while avoiding a new
JPEG decoder difference. Image-token min/max64/256 match the admitted processor
profile. Upstream interpolation and F16 vision arithmetic remain independent:
these controls do not claim byte-exact vision features. Any prompt/geometry guard
failure is retained instead of silently changing the comparison.

The native `/completion` API retains raw response JSON, token IDs, decoded text,
stop_type/EOS/cap, top probabilities, settings and inference timings. Raw IDs may
exclude the EOG token; its presence is reported separately from server EOS. CER,
WER and prefix agreement use original model text, not human OCR gold. Runtime
startup/wall time and process peak RSS are recorded separately; no cross-engine
speed claim follows. The exact native endpoint is chosen because the experimental
mtmd CLI treats an empty `-p` as interactive, which would not run this input as
intended.

Preflight requires9GiB available cgroup-aware RAM,25GiB scratch and1GiB output.
The four complete model/projector downloads total11,327,686,848 bytes; only one
decoder is retained/loaded at a time and only verified task-owned files are
removed. Build uses four jobs, CPU AVX2 with native/AVX512/GPU/BLAS off. Each request
has1200 seconds and the entire worker a180-minute alarm, with task process-group
cleanup on failure/cancel. Large model files remain hosted;90-day artifacts retain
small reports/inputs/text/IDs/logs and source/version receipts. No shared runtime,
new PR, release, alias or default changes. Dispatch requires review of the exact
worker/protocol source and resources; preparation is not qualification.

User-directed infrastructure change: initial GitHub42235476177 is cancelled after
the pinned upstream build and before any inference (phase `download-projector`,
no arms). Artifact13515095161 retains the partial report/build logs,59,677 bytes,
SHA256 `3f977c197613323efe132fc2c8693ac2ec1899a124fc305da6bc0a092137a797`.
The same model/input/prompt/generation protocol moves to authorized Kaggle CPU;
provider admission and ancestor-aware cgroup inspection are explicit. No duplicate
GitHub job or model comparison is dispatched.
