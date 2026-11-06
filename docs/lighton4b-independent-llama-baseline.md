# Independent llama.cpp LightOn4B precision baseline

Authorized Kaggle CPU version3 completes **all nine fixed arms** at source
`4d2184edecf8fbb883d7c96770d57da7c6221c11`. Full upstream Q6_K exactly reproduces
original BF16 table222 IDs/text/EOS and Arabic6 IDs/text/EOS, but still differs on
form. Upstream Q4_K_M restores form's figure prefix and EOS, while table hits the
unchanged256-token cap. Q8 differs on form and by10 characters on table. These
are measured independent-engine/published-weight outcomes, not proof that one
quantizer or runtime matches the original checkpoint on every input.

| Upstream decoder / frozen case | IDs / stop | Initial matching IDs | Character edits | Word edits | Wall with load s | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|
| Q4_K_M / simple_form | 170 / eos | 9 | 189/656 | 46/90 | 73.96 | 5,418,917,888 |
| Q4_K_M / simple_table | 256 / limit | 37 | 315/607 | 88/106 | 86.61 | 5,374,857,216 |
| Q4_K_M / arabic_printed_line | 6 / eos | 6 | 0/12 | 0/2 | 17.58 | 5,372,526,592 |
| Q6_K / simple_form | 182 / eos | 44 | 154/656 | 31/90 | 78.19 | 4,525,977,600 |
| Q6_K / simple_table | 222 / eos | 222 | 0/607 | 0/106 | 82.42 | 4,482,211,840 |
| Q6_K / arabic_printed_line | 6 / eos | 6 | 0/12 | 0/2 | 14.75 | 4,479,565,824 |
| Q8_0 / simple_form | 160 / eos | 9 | 140/656 | 35/90 | 87.67 | 5,541,597,184 |
| Q8_0 / simple_table | 223 / eos | 30 | 10/607 | 1/106 | 98.19 | 5,498,720,256 |
| Q8_0 / arabic_printed_line | 6 / eos | 6 | 0/12 | 0/2 | 15.67 | 5,495,996,416 |

Original BF16 references complete at form177/table222/Arabic6 tokens, all EOS.
Eight upstream arms reach primary EOS248046, which is included in the raw IDs;
Q4_K_M table is capped256 without EOS. All three Arabic outputs are exact.
Q6 table is exact; Q8 table differs by one word (10 characters), despite its
higher stored precision. No monotonic decoded-quality or overall model/default
qualification follows from the three observed inputs. No heldout input or
original-model forward is executed.

Compared with historical corrected native plain-Q4 form (256 capped/noEOS,
543/656 character edits), this upstream Q4_K_M run produces170 IDs/EOS with
189/656 edits and a `<figure>` prefix. This is an actionable independent baseline,
**not an isolated runtime improvement**: whole published weights, importance
matrix, mixed tensor precision, F16 vision, layout and arithmetic all differ.
On table, our earlier24-output Q6/Q8 candidate yields228 IDs/EOS with224/607
edits, versus upstream fullQ6's exact222 and upstream Q4_M's capped315/607.
These comparisons motivate controlled conversion/precision attribution rather
than importing upstream tensor payloads into the HF-order owner.

All requests use four CPU threads, no offload/flash-attention, greedy sampling,
cap256, empty image-only transcription and thinking disabled. Each starts a fresh
server process. Exact source chat wrappers, all nine expanded original input-ID
`.npy` hashes and actual154/86/82 prompt lengths pass before comparison. Canonical
Pillow11.3.0 RGB PNG transport preserves input decoding; upstream resize/filtering
and F16 vision features remain unproven against source bytes. Q8 is a decoder
control with the same F16 projector, not an unquantized BF16/F16 decoder control.

The worker independently authenticates the whole published files at fixed revision
`8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01` before loading:

| File | Bytes | SHA256 |
|---|---:|---|
| Q4_K_M |2,708,804,576|`b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26`|
| Q6_K |3,464,055,776|`290e2b46627c9d3f932c25ef6697fc71a37a9b226337d761a0190185eaa738b7`|
| Q8_0 |4,482,403,296|`51e2913cad715f2516a7ab1a6f7677ebc6122361b80ba21bffc490198621ec0a`|
| F16 projector |672,423,200|`21eae623f60580f19ae4f04716292cd76ad36b420ee01e761254c8b1ec966321`|

Only one decoder is loaded at a time. Cgroup-aware available RAM/disk bounds pass;
no GPU is requested. AVX2 is explicit, native/AVX512/GPU/BLAS builds are off. Exact
upstream source is reverified before compilation; executable SHA256 is
`ae5a461b42e3195792c1b4e9c4318077a864f308acfacaf0c7b1b37e9177f223`.
Per-request1200-second and whole-worker180-minute limits stay unchanged.

The native timing fields report approximately3.61–4.00 tokens/s Q4_M,
3.21–3.28 Q6 and2.55–2.73 Q8; wall times include fresh model loading. Source
versions, CPU, CMake options/cache, package versions, raw request settings,
responses and RSS are retained. These single ordered observations are not a
repeated or hardware-normalized cross-engine speed benchmark. The response's
`tokens_cached` is the final populated context, not reused prompt tokens;
`timings.cache_n=0` in every fresh process proves no prompt reuse. This distinction
is checked in the independent review.

The [result](../tests/results/qwen35-ocr/lightonocr3-4b-independent-llama/independent-llama.json)
SHA256 is `5a27200a32a8c18a918a7b5d0ecd01b8a6a80e3e95d5bd1c32f2b941730bcb5d`.
The [independent audit](../tests/results/qwen35-ocr/lightonocr3-4b-independent-llama/independent-review.json)
checks all27 raw ID/text/response hashes, all original-capture hashes, nine source
prompt-ID serialization hashes, raw primary EOS/limit fields and separately
recomputes all character/word edit distances with a full-matrix algorithm. Metrics
measure agreement with the original model's outputs, not human OCR accuracy.
All115 remote outputs total1,984,737 bytes and are retained on the versioned
service output and in this evidence tree. No model payload is downloaded locally.

The exact credential-free [submitted bootstrap](../tests/results/qwen35-ocr/lightonocr3-4b-independent-llama/kaggle-bootstrap-v3.py)
SHA256 `9bc2d0e788bd605f2bdfc022297c717c462e8f3a4ab019597cec794f6df2775d`
is independently verified through the service API. Private service binding and
access references stay outside Git. The [protocol](lighton4b-independent-llama-baseline-v1.json)
and [worker](../tools/ci-heavy/lighton4b_independent_llama_baseline.py) reproduce the
closed nine-arm CPU experiment without an original rerun.

Initial failures are preserved beside the result: GitHub42235476177 was cancelled
at the user's infrastructure change after build/before inference; its artifact
13515095161 is59,677 bytes/SHA256
`3f977c197613323efe132fc2c8693ac2ec1899a124fc305da6bc0a092137a797`.
Kaggle version1 fails source identity lookup from its launch directory, before
build/model. Version2 passes build/model/prompt identity but rejects media
transport before vision/prefill: the server requires its process-specific
`/props.media_marker`, not libmtmd's default. Version3 normalizes only this
transport marker while retaining every original wrapper/ID/geometry guard.
No original forward, threshold waiver, prompt/cap change, hidden fixture selection,
new PR, model publication, default or release change follows.

## Primary upstream and conversion evidence

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

