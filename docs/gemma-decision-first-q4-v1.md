# Typed Gemma first protected mixed Q4 source preparation

This separately versioned engine-only experiment reuses the immutable
[complete typed-decision F32 publication](https://huggingface.co/cstr/embeddinggemma-2-ft-typed-decision-GGUF/tree/d8427e48a0c20bc63216d332ec0b5e8ec43430c9)
and [original fixtures](https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/tree/508188c36d0550b12a6a467f5d394c10aa33e04c/gemma-decision/original-fp32).
It does not run the original model again. The
[precision audit](encoder-precision-evidence-20261029.md) preserves the published
Q8 approximation's embedding drift and eight matching answers. Those Q8 values
are not a Q4 acceptance budget.

The [closed protocol](gemma-decision-first-q4-v1.json) freezes all engine,
loader, tokenizer, renderer, GGML, quantizer, fixture and new caller source.
The [standalone hosted workflow](../.github/workflows/gemma-decision-first-q4-v1.yml)
builds only the actual unchanged Gemma engine and dependencies, the actual
`tools/quantize.cpp` executable, and test probes. It uses actual a06 CPU GGML
with LLAMAFILE ON, the source-derived root CPU ISA function, and two workers.
This is not a public C ABI, CLI/HTTP, Rust/Dart, GPU, release or app qualification.
The comparator stays in the test executable and must be absent from the engine
archive. No production source, CMake, arithmetic or gate changes are made.

Before any production-model quantization, the pinned1,102,182,464-byte complete
F32 must pass all five probes and145 original stage comparisons: exact18/20/25/
19/549 token IDs, original axis order, ordinary/observed/repeat returned-byte
identity, minimum row cosine0.99999 and relative L2 at most0.003. Actual
`crispembed_diff::Ref` metrics and independent NumPy metrics include absolute,
RMS/mean errors and magnitudes. There is no new universal absolute0.002 gate.
All eight original typed answers and the original head-on-original-embedding
controls must pass too. Head control embeddings are used only for that explicit
component test; actual composed decisions encode raw frozen requests through
the unchanged renderer, encoder and head.

Two storage recipes are declared before outputs. Their actual complete436-
tensor census must match the authenticated F32 header before native forwards:

| Recipe | Actual low-precision roles | Protected F32 roles | Expected inventory |
| --- | --- | --- | --- |
| `broad-mixed` | Eligible encoder2D matrices; generic quantizer retains the token table at Q8_0 | Head, final embedding projection, norms, layer scalars and temperatures |217 Q4_K +1 Q8_0 +218 F32 |
| `protected-input` | Only24 layers' attention q/k/v/o and MLP gate/up/down | Token embedding, all PLE, head, embedding projection, norms/scalars/temperatures |168 Q4_K +268 F32 |

The input protection is an architecture-based hypothesis, not an established
precision sensitivity result. Both recipes use the actual C quantizer and
standard file-type14/quantization-version2. No Q4_0 fallback is admitted. Every
critical F32 payload hash must remain exact; all original metadata stays exact
apart from the declared precision fields. Q4/Q8 scales must be finite. Semantic
axes remain exact; the only allowed physical rank canonicalization is the
single-row `head.residual.3.weight`, `[256,1]` to `[256]`, with exact F32 bytes.
Small hosted C-quantizer fixtures verify the actual storage/guard policy.

Each completed model panel performs exactly88 encoder forwards:15 for the five
ordinary/observed/repeat probes,30 explicit original decision-bank texts, and43
source-derived internal calls from three actual engine requests plus the first
request repeated. The latter counts follow the unchanged per-request text-cache
and the exact13/11/6 unique rendered banks; they are labeled derived, not an
instrumented production counter. Sixteen repeated head-on-original calls perform
no encoder forward. Three invalid-schema calls are rejected before encoding,
then valid requests demonstrate recovery. The maximum across F32 and both
candidates is264 encoder forwards, with zero original forwards. Failed attempts
remain visible and do not count as completed panels.

Both candidates retain all five/145 stage flags, three actual-Ref comparisons
covering all30 bank vectors, eight
answers, source/head controls and any failures. The old F32 probability tests
remain `atol=3e-4, rtol=3e-4`, score expected-level tolerance0.003; head logits
remain `3e-5/3e-5` and probabilities `3e-6/3e-5`. These remain separately labeled
strict F32 flags for Q4. No new relaxed Q4 budget, selected recipe, independent
task benchmark, alias or publication is introduced. Original token-ID gold is
available for the five probes; the frozen decision bank contains embedding gold
but no original token-ID arrays. Native bank IDs are retained without inventing
such targets.

Models, original fixtures, complete native dumps and lossless outputs stay
remote for90 days. Only small source/compiler/tensor/numerical reports need local
review. Every phase retains its own structured rejection, including the current-
source F32 barrier. The existing Q8 alias/publication and historical producers
remain unchanged; Q8-specific Rust/Dart work is separate.
