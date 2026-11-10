# LightOn4B full-attention decoder block component

Draft PR149 stacks linear-block PR148 (`fd970f348`). The separate
`lighton4b_attention_block_cpu` path composes input zero-centered RMS, gated
causal attention, residual, post-RMS,9216-wide SwiGLU FFN and final residual at
hidden2560. Geometry is exactly16 query heads,4 KV heads,width256,partial rotary64,
interleaved sections11/11/10 and theta10,000,000. All historical attention/decoder
sources and the previously tested linear-block header stay unchanged.

The new `CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK` gate and `CRISPEMBED_QWEN35_OCR`
are required; defaults stay off. Weights are borrowed with exact shapes/types,
allocated data, contiguous layout and unique tensor identities. Protected
parameters remain F32; matrices admit F32/F16 headers, with actual numerical
evidence restricted to F32. The component bounds batch1 and1–16 new tokens,
4096 total tokens, validates4-head/256-wide immutable K/V caches and rejects
malformed cache lengths/geometry or nonfinite positions. Initial cache is null.
There is no vocabulary projection, complete decoder or model admission.

Hosted [41977890812](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41977890812)
at `9f64a417a663a63af1381e6b20d55400fa766e81` passes51 original/native stage pairs,
72 native controls and9 bit-exact original observer-neutrality checks. The
original pinned Transformers5.5.4 `Qwen3_5DecoderLayer` at index3 runs first on
eleven full-width synthetic F32 parameter arrays, seed195033. Eager causal masks
and distinct nonzero multimodal position axes exercise two-token prefill, one
cached token and the three-token full prefix. Native cached K/V come only from
native prefill. No checkpoint is acquired.

Stages include input norm, Q/K/V projections, Q/K norms, gated pooled attention,
attention projection/residual, post norm, FFN gate/up/activation/down, final
hidden and K/V caches. Controls cover gates, token bounds, a malformed norm,
wrong layer type, nonfinite positions, cache batch/head/width/length/token count,
immutable previous cache and stage output writes. They do not exhaust all
malformed weights, cache values or supported token counts.

Independent retained-artifact review verifies all six directly recorded source
hashes, every original raw-array hash, all51 shapes/element counts and recomputes
max absolute errors and minimum row cosines from original/native vectors.
Earliest nonzero error is `prefill.input_norm`,3.5762786865e-7 against magnitude
1.429962039. Worst error is `cached0.activation`,5.7220458984e-6 against4.790956020.
Final hidden error is3.7550926208e-6 for prefill/full-prefix (magnitude2.399322987)
and3.0994415283e-6 for the cached token (magnitude2.096099854). The
[review receipt](../tests/results/qwen35-ocr/lightonocr3-4b-attention-block.json)
retains all measured stage errors/magnitudes, source/runtime/executable hashes,
parameter records and observer controls. Raw vectors remain in the hosted artifact.

The shared original loader verifies GDN function source hashes as additional
checks, but GDN functions are not executed by this attention case. The public
review and subsequent reporting-only driver update label these as auxiliary
source checks. Executed decoder/attention/rotary/MLP/RMS code is pinned by the
verified complete original modeling-module hash. No array or metric changes.

Evidence is one synthetic CPU/F32 full-attention block, batch1,2+1/3 tokens.
F16, checkpoint-fed stages, mixed linear/full-attention32-layer execution,
final norm/streamed full-vocabulary logits, tokenizer/preprocessing, actual
text/EOS and quality remain open. No format5 recognition, public alias, release,
default flip or full-model qualification follows from these component comparisons.
