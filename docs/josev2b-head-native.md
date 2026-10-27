# Bounded original-reference Josev joint head

This component implements only the pinned `JointSchemaHead`, independently
decomposed into F32 GGML operations. The publisher's
[source](https://huggingface.co/mayflowergmbh/Josev-2B/blob/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/joint_schema_model.py)
and [Apache-2.0 license](https://huggingface.co/mayflowergmbh/Josev-2B/blob/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/LICENSE)
remain the blueprint. There is no backbone, tokenizer, embedding-table lookup,
public API, registry entry, release qualification or default change.

The reviewed original receipt is
`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`,
worker `b2edf8c83c9aca1f8a9210198d47798ac13828b8`. Its four ordinary/observed/repeat
cases pass exact neutrality; all122 loaded head tensors have exact promoted
checkpoint identities. These are original results, not native results. The new
[contract](josev2b-head-native-contract.json) freezes their head-stage order,
occurrences and logical shapes without copying activation arrays into Git.

The internal [header](../src/josev2b_head.h) accepts contiguous original F32 hidden
states `[1,T,2048]` without padding and ordered fields with types, nonempty question
and option token spans, and raw per-option lexical means `[N,2048]`. The lexical
means must come from the actual tied output embedding captured by the original;
contextual hidden means are not a substitute. The first supported subset is
`1<=T<=512`, at most16 fields and128 total options. Empty spans, shape mismatches,
nonfinite inputs, duplicate field IDs and invalid types reject before graph setup.
One context owns its backend/weights and supports one call at a time.

Both load and run require the value-parsed default-off
`CRISPEMBED_JOSEV_HEAD` gate; unset/empty/0 reject. The GGUF architecture must be
`josev2b_head_reference`, with all122 exact `head.` tensor names/shapes, F32 types,
30,472,708 parameters, original checkpoint/source/receipt/worker/configuration and
`official-pytorch-no-extra-temperature` metadata. No unknown tensor or implicit
quantized path is accepted. The parent must authenticate the actual artifact and
[conversion receipt](josev2b-head-conversion.md) before loading it. Loaded weights
are also checked for finite values.

Execution preserves hidden LayerNorm before span means, the last normalized
hidden token as global vector, all six learned projections, type embeddings,
two pre-norm evidence-routing blocks, option-weighted summaries and four
`TransformerDecoderLayer(norm_first=True)` blocks. Each MHA uses eight64-wide
heads and unmasked scaled dot-product attention; feedforward width is2048 with
GELU-erf. LayerNorm epsilon is`1e-5`, lexical/anchor normalization epsilon`1e-12`,
and cosine epsilon`1e-8`. The final expression is exactly
`prior + sigmoid(residual_gate) * (exp(clamp(joint_logit_scale,max=ln(100))) * cosine + residual)`;
the prior uses its own equivalently clamped exponential scale. There are no extra
temperatures. Matmuls explicitly request `GGML_PREC_F32`; no flash attention,
F16 storage or quantized conversion is introduced. Actual numerical agreement
still requires hosted measurement against every frozen stage and final output.

Backend selection is explicit: `cpu` or an exact registered CUDA device such as
`CUDA0`. Missing devices or unsupported operations reject; CUDA never silently
falls back to CPU. The actual backend name is recorded. CPU defaults to two
threads in the probe. The optional CUDA path is intended for the approved genuine
GPU reference task, not a CPU-only Kaggle task or hardware retry loop.

The standalone test probe compiles with `CRISPEMBED_JOSEV_HEAD_TESTING` and is
separate from ordinary library objects. Its observer retains connected
`ggml_view_4d` nodes and every storage owner before allocation. It preserves
logical original shapes, full call order, and repeated labels with an explicit
zero-based occurrence; a dictionary keyed only by label loses per-field outputs.
Ordinary compilation removes the observer event code, labels and symbols.

Hosted invocation after artifact review:

```sh
CRISPEMBED_JOSEV_HEAD=1 test-josev2b-head head-f32.gguf input.json output cpu \
  51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090 \
  b2edf8c83c9aca1f8a9210198d47798ac13828b8
```

The input JSON has exactly`schema,tokens,hidden_f32,fields`; each field has
`id,type,question_start,question_end,options,lexical_f32`. Each option has`start,end`.
`schema` is`crispembed-josev-head-input-v1`; all raw files are flat relative
filenames beside the manifest, little-endian IEEE F32 with exact extents.
The frozen contract contains complete four-case examples with the actual spans.
[Python contract helpers](../tools/josev2b_head_native.py) validate input and event
metadata without importing array/framework packages.

The output has`stages.jsonl` and numbered raw `.f32` files, plus`probe.json` with
ordered fields, logits/probabilities, backend, timings, and exact ordinary/observer
and ordinary/repeat bit comparisons. Additional `field-N-ordinary-*` and
`field-N-repeat-*` raw files support independent neutrality checks. A partial
failure retains already written stages; the parent runner must preserve the
failure log and reject qualification. The parent adapter must hash all raw bytes,
map `(stage,occurrence)`, compare the unchanged`0.002` maximum absolute/`0.99999`
cosine gates with crispembed-diff plus independent metrics, and inspect final
logits/probabilities and decoded official response. No numeric gate is waived by
this source preparation.

Hosted `--contracts` exercises eight metadata fixtures and eight invalid input
contracts without weights; `--gates` proves early unset/empty/0 rejection.
`--reject FILE RECEIPT_SHA WORKER_SHA` reports actual loader rejection diagnostics.
Local validation is limited to stdlib source/contract controls and formatting.
No CMake/shared workflow/API/binding files are changed in this lane; the parent
owns hosted build/conversion/replay integration. No native outcome is claimed.
